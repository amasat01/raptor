//! The target card of a wing: the numbers a target's runtime reads about the hardware class it runs on (the linear-algebra limits the
//! unit texts are compiled against and the fitted weights of the dispatch cost model), read and checked here, and written as the
//! wing's generated facts header. One card per wing in `cards/`; the header is produced from it and from nothing else.
use rcore::json::{parse, J};

/// The fields of a weight row, in the order of the header's `EG_WEIGHT_ROWS` macro after the three key fields.
const VALUES: [&str; 11] = [
    "arith32",
    "special32",
    "special64",
    "launch_seconds",
    "split_seconds",
    "sort_seconds",
    "lane_rate_per_core",
    "memory_rate",
    "cache_ratio",
    "chain_seconds_per_core",
    "vector_width",
];
/// The key fields of a weight row: the hardware class it holds for.
const KEYS: [&str; 3] = ["vendor", "family", "model"];
/// The values a device row may leave at zero, which means "read it from the live device".
const LIVE_ON_DEVICE: [&str; 3] = [
    "lane_rate_per_core",
    "memory_rate",
    "chain_seconds_per_core",
];
/// Facts that vary with the machine or the element type: they are read live and never stated in a row.
const LIVE_FACTS: [&str; 2] = ["cache_bytes", "elem_bytes"];
/// The vendor numbers a host class row may hold: 1 Intel, 2 AMD.
const HOST_VENDORS: [i64; 2] = [1, 2];

/// The side of the machine a card describes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Target {
    /// The host processor.
    Host,
    /// The device.
    Device,
    /// No target: a wing that only names itself and states its limits, with no rows.
    None,
}

/// One row of fitted weights for a hardware class. A row whose three key fields are zero is the target's built-in row, which an
/// unknown class takes.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightRow {
    /// Host: 1 Intel, 2 AMD; device: 0.
    pub vendor: i64,
    /// Host: the processor's display family; device: the compute capability's major number.
    pub family: i64,
    /// Host: the processor's display model; device: the compute capability's minor number.
    pub model: i64,
    /// The values, in the order of `VALUES`.
    pub values: [f64; 11],
}

/// One row saying how a unit's dense contraction runs on the device for a hardware class and an element width. A row whose family
/// and model are both zero is the built-in row of its width, which an unknown class takes.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteRow {
    /// The compute capability's major number; 0 with a zero model is the built-in row.
    pub family: i64,
    /// The compute capability's minor number.
    pub model: i64,
    /// The element width in bits: 32 or 64.
    pub dtype_bits: i64,
    /// Whether the contraction goes to the matrix-multiply routine (true) or runs the unit's own kernel (false).
    pub gemm: bool,
}

/// A wing's card.
#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    /// The wing's name.
    pub wing: String,
    /// The side it describes.
    pub target: Target,
    /// Up to this dimension one thread holds a matrix.
    pub max_dim: i64,
    /// Up to this dimension a team of threads works on a matrix, when the wing has teams.
    pub team_max_dim: Option<i64>,
    /// Where the weights come from, as one line of text.
    pub note: String,
    /// The weight rows: the built-in row first, then the class rows ordered by key.
    pub rows: Vec<WeightRow>,
    /// The gemm route rows of a device card, the built-in rows first, then the class rows ordered by key; none on a card without them.
    pub routes: Vec<RouteRow>,
}

/// The members of an object, refusing one that is not in `known` and one a card repeats.
fn members<'a>(
    card: &str,
    what: &str,
    json: &'a J,
    known: &[&str],
) -> Result<&'a [(String, J)], String> {
    let J::Obj(pairs) = json else {
        return Err(format!("card {card}: {what} is not an object"));
    };
    for (at, (name, _)) in pairs.iter().enumerate() {
        if LIVE_FACTS.contains(&name.as_str()) {
            return Err(format!(
                "card {card}: {what} states {name}, which is read live and never stated in a row"
            ));
        }
        if !known.contains(&name.as_str()) {
            return Err(format!("card {card}: {what} has an unknown field {name}"));
        }
        if pairs[..at].iter().any(|(other, _)| other == name) {
            return Err(format!("card {card}: {what} states {name} twice"));
        }
    }
    Ok(pairs)
}

/// The number a member holds, refusing a missing one and a value that is not a number.
fn number(card: &str, what: &str, json: &J, name: &str) -> Result<f64, String> {
    match json.get(name) {
        Some(J::Num(n)) => Ok(*n),
        Some(_) => Err(format!("card {card}: {what} field {name} is not a number")),
        None => Err(format!("card {card}: {what} is missing the field {name}")),
    }
}

/// The whole number a member holds.
fn whole(card: &str, what: &str, json: &J, name: &str) -> Result<i64, String> {
    let n = number(card, what, json, name)?;
    if n.fract() != 0.0 || n < 0.0 {
        return Err(format!(
            "card {card}: {what} field {name} is not a whole number from zero"
        ));
    }
    Ok(n as i64)
}

/// One weight row, checked against the target's rules for its fields.
fn read_row(card: &str, target: Target, json: &J, at: usize) -> Result<WeightRow, String> {
    let what = format!("weight row {at}");
    let known: Vec<&str> = KEYS.iter().chain(VALUES.iter()).copied().collect();
    members(card, &what, json, &known)?;
    let [vendor, family, model] = KEYS.map(|k| whole(card, &what, json, k));
    let mut values = [0.0; 11];
    for (slot, name) in values.iter_mut().zip(VALUES) {
        *slot = number(card, &what, json, name)?;
        let live = target == Target::Device && LIVE_ON_DEVICE.contains(&name);
        if *slot < 0.0 || (*slot == 0.0 && !live) {
            return Err(format!(
                "card {card}: {what} field {name} is {slot}; it must be positive{}",
                if target == Target::Device && live {
                    ""
                } else {
                    " (zero means live only for a device's lane, memory and chain rates)"
                }
            ));
        }
    }
    Ok(WeightRow {
        vendor: vendor?,
        family: family?,
        model: model?,
        values,
    })
}

/// Checks the key fields of a class row: a host class names its vendor (1 Intel, 2 AMD) and family, a device class has vendor 0 and a
/// family, so a zero in the key is never a class.
fn check_class(card: &str, target: Target, row: &WeightRow, at: usize) -> Result<(), String> {
    let bad_vendor = match target {
        Target::Host => !HOST_VENDORS.contains(&row.vendor),
        Target::Device | Target::None => row.vendor != 0,
    };
    if bad_vendor || row.family == 0 {
        return Err(format!(
            "card {card}: weight row {at} is a class row with a zero or unknown key ({}, {}, {}): a class names its vendor and family",
            row.vendor, row.family, row.model
        ));
    }
    Ok(())
}

/// One gemm route row: the key (family, model, dtype) and the route as words.
fn read_route(card: &str, json: &J, at: usize) -> Result<RouteRow, String> {
    let what = format!("route row {at}");
    members(card, &what, json, &["family", "model", "dtype", "route"])?;
    let dtype_bits = match json.get("dtype").and_then(J::str) {
        Some("f32") => 32,
        Some("f64") => 64,
        Some(other) => {
            return Err(format!(
                "card {card}: {what} has the dtype {other}, which has no route: f32 or f64"
            ))
        }
        None => return Err(format!("card {card}: {what} is missing the field dtype")),
    };
    let gemm = match json.get("route").and_then(J::str) {
        Some("gemm") => true,
        Some("direct") => false,
        Some(other) => {
            return Err(format!(
                "card {card}: {what} has the route {other}: gemm or direct"
            ))
        }
        None => return Err(format!("card {card}: {what} is missing the field route")),
    };
    Ok(RouteRow {
        family: whole(card, &what, json, "family")?,
        model: whole(card, &what, json, "model")?,
        dtype_bits,
        gemm,
    })
}

/// Reads and checks the route rows of a device card: no key twice, a class row (a key that is not 0, 0) names its family, and the
/// built-in row exists for both widths.
fn read_routes(card: &str, rows: &[J]) -> Result<Vec<RouteRow>, String> {
    let mut routes = vec![];
    for (at, row) in rows.iter().enumerate() {
        let route = read_route(card, row, at)?;
        let key = |r: &RouteRow| (r.family, r.model, r.dtype_bits);
        if routes.iter().any(|other| key(other) == key(&route)) {
            return Err(format!(
                "card {card}: two route rows hold the key (device, {}, {}, {} bits)",
                route.family, route.model, route.dtype_bits
            ));
        }
        if route.family == 0 && route.model != 0 {
            return Err(format!(
                "card {card}: route row {at} is a class row with a zero key ({}, {}): a class names its family",
                route.family, route.model
            ));
        }
        routes.push(route);
    }
    for bits in [32, 64] {
        if !routes
            .iter()
            .any(|r| (r.family, r.model, r.dtype_bits) == (0, 0, bits))
        {
            return Err(format!(
                "card {card}: no built-in route row (family and model 0) for {bits} bits"
            ));
        }
    }
    routes.sort_by_key(|r| (r.family, r.model, r.dtype_bits));
    Ok(routes)
}

/// Reads a card from its JSON text. Refuses, by name: a missing or unknown field, a field stated twice, `cache_bytes` or
/// `elem_bytes`, two rows with one key, no built-in row or two for the target, a class row with a zero key, and a value that is not
/// positive where the target needs one (a device's lane, memory and chain rates may be zero: they are read live). A device card may
/// state `gemm_routes`; on a card of any other target they are refused, and a device card that states them has the built-in row for 32
/// and for 64 bits, with no key twice, no unknown route or dtype word and no class row with a zero family.
pub fn read_card(text: &str) -> Result<Card, String> {
    let json = parse(text).map_err(|e| format!("card: {e}"))?;
    let wing = json
        .get("wing")
        .and_then(J::str)
        .ok_or("card: the field wing is missing or not text")?
        .to_string();
    let card = wing.as_str();
    members(
        card,
        "the card",
        &json,
        &["wing", "target", "linalg", "note", "weights", "gemm_routes"],
    )?;
    let target = match json.get("target").and_then(J::str) {
        Some("host") => Target::Host,
        Some("device") => Target::Device,
        Some("none") => Target::None,
        _ => {
            return Err(format!(
                "card {card}: the field target is missing or is not host or device"
            ))
        }
    };
    let linalg = json
        .get("linalg")
        .ok_or_else(|| format!("card {card}: the card is missing the field linalg"))?;
    members(card, "linalg", linalg, &["max_dim", "team_max_dim"])?;
    let max_dim = whole(card, "linalg", linalg, "max_dim")?;
    let team_max_dim = linalg
        .get("team_max_dim")
        .map(|_| whole(card, "linalg", linalg, "team_max_dim"))
        .transpose()?;
    let note = json
        .get("note")
        .and_then(J::str)
        .ok_or_else(|| format!("card {card}: the field note is missing or not text"))?;
    if note.contains("*/") || note.contains('\n') {
        return Err(format!(
            "card {card}: the note is one line and holds no comment end"
        ));
    }
    if target == Target::None {
        if json.get("weights").is_some() || json.get("gemm_routes").is_some() {
            return Err(format!(
                "card {card}: rows are stated on a card that is not a target card"
            ));
        }
        return Ok(Card {
            wing,
            target,
            max_dim,
            team_max_dim,
            note: note.to_string(),
            rows: vec![],
            routes: vec![],
        });
    }
    let weights = json
        .get("weights")
        .map(J::arr)
        .ok_or_else(|| format!("card {card}: the card is missing the field weights"))?;
    let mut rows = vec![];
    for (at, row) in weights.iter().enumerate() {
        rows.push(read_row(card, target, row, at)?);
    }
    let key = |r: &WeightRow| (r.vendor, r.family, r.model);
    for (at, row) in rows.iter().enumerate() {
        if rows[..at].iter().any(|other| key(other) == key(row)) {
            return Err(format!(
                "card {card}: two weight rows hold the key ({}, {}, {})",
                row.vendor, row.family, row.model
            ));
        }
        if key(row) != (0, 0, 0) {
            check_class(card, target, row, at)?;
        }
    }
    if !rows.iter().any(|row| key(row) == (0, 0, 0)) {
        return Err(format!(
            "card {card}: no built-in row (vendor, family and model all 0) for the target"
        ));
    }
    rows.sort_by_key(key);
    let routes = match (json.get("gemm_routes"), target) {
        (None, _) => vec![],
        (Some(listed), Target::Device) => read_routes(card, listed.arr())?,
        (Some(_), _) => {
            return Err(format!(
                "card {card}: route rows are stated on a card that is not the device target card"
            ))
        }
    };
    Ok(Card {
        wing,
        target,
        max_dim,
        team_max_dim,
        note: note.to_string(),
        rows,
        routes,
    })
}

/// The wing's generated facts header: its name and the linear-algebra limits, and nothing else. The weight and route rows of the
/// targets go to the door's rows header (`door_rows_header`): a unit source includes one wing's facts header, never two.
pub fn facts_header(card: &Card) -> String {
    let up = card.wing.to_uppercase();
    let mut lines = vec![
        format!(
            "/* GENERATED by gen_facts_header from the card of the raptor wing {}. Do not edit. */",
            card.wing
        ),
        format!("#ifndef RAPTOR_FACTS_{up}_H"),
        format!("#define RAPTOR_FACTS_{up}_H"),
        String::new(),
        format!("#define RAPTOR_WING_NAME \"{}\"", card.wing),
        "/* up to this dimension one thread holds a matrix (the linear-algebra max_dim) */"
            .to_string(),
        format!("#define RAPTOR_LINALG_MAX_DIM {}", card.max_dim),
    ];
    if let Some(team) = card.team_max_dim {
        lines.push("/* up to this dimension a team of threads works on a matrix (the linear-algebra team_max_dim) */".to_string());
        lines.push(format!("#define RAPTOR_LINALG_TEAM_MAX_DIM {team}"));
    }
    lines.push(String::new());
    lines.push(format!("#endif /* RAPTOR_FACTS_{up}_H */"));
    lines.push(String::new());
    lines.join("\n")
}

/// The rows header of a door build, from the cards of the targets the door is built with: the weight rows of the host card and
/// then of the device card, if there is one (the built-in row of a target first, then its class rows by key, each row keeping its
/// target column), and the device's gemm route rows. `EG_GEMM_ROUTE_ROWS` is always defined, with an empty expansion when there is
/// no device card or it states no routes. Refuses a first card that is not the host target card, a second that is not the device
/// target card, and, by name, a card that is no target card at all.
pub fn door_rows_header(host: &Card, device: Option<&Card>) -> Result<String, String> {
    let cards: Vec<&Card> = std::iter::once(host).chain(device).collect();
    for card in &cards {
        if card.target == Target::None {
            return Err(format!(
                "card {}: not a target card, it has no rows",
                card.wing
            ));
        }
    }
    if host.target != Target::Host {
        return Err(format!(
            "card {}: the door's rows need a host card first, not a device card",
            host.wing
        ));
    }
    if let Some(card) = device.filter(|card| card.target != Target::Device) {
        return Err(format!(
            "card {}: two cards of one target: the door's second card is the device card",
            card.wing
        ));
    }
    let wings: Vec<&str> = cards.iter().map(|c| c.wing.as_str()).collect();
    let mut lines = vec![
        format!(
            "/* GENERATED by gen_facts_header from the cards of the raptor wings {}. Do not edit. */",
            wings.join(" and ")
        ),
        "#ifndef EAGLE_DOOR_ROWS_H".to_string(),
        "#define EAGLE_DOOR_ROWS_H".to_string(),
        String::new(),
    ];
    lines.extend(cards.iter().map(|c| format!("/* {} */", c.note)));
    lines.push(format!(
        "/* One X(target, {}, {}) per row of dispatch cost weights: target 0 host or 1 device; the row of key 0, 0, 0 is the built-in one. */",
        KEYS.join(", "),
        VALUES.join(", ")
    ));
    lines.push("#define EG_WEIGHT_ROWS(X) \\".to_string());
    let rows: Vec<String> = cards
        .iter()
        .flat_map(|card| {
            let target = i64::from(card.target == Target::Device);
            card.rows.iter().map(move |row| {
                let values: Vec<String> = row.values.iter().map(|v| format!("{v:?}")).collect();
                format!(
                    "  X({target}, {}, {}, {}, {})",
                    row.vendor,
                    row.family,
                    row.model,
                    values.join(", ")
                )
            })
        })
        .collect();
    lines.push(rows.join(" \\\n"));
    lines.push(String::new());
    lines.push("/* One X(target, family, model, dtype_bits, route) per row of gemm routes: target 1 device; family, model the compute capability's major, minor, both 0 the built-in row of the width; dtype_bits 32 or 64; route 0 direct (the unit's own kernel), 1 gemm. */".to_string());
    let routes: Vec<String> = device
        .iter()
        .flat_map(|card| card.routes.iter())
        .map(|r| {
            format!(
                "  X(1, {}, {}, {}, {})",
                r.family,
                r.model,
                r.dtype_bits,
                i64::from(r.gemm)
            )
        })
        .collect();
    if routes.is_empty() {
        lines.push("#define EG_GEMM_ROUTE_ROWS(X)".to_string());
    } else {
        lines.push("#define EG_GEMM_ROUTE_ROWS(X) \\".to_string());
        lines.push(routes.join(" \\\n"));
    }
    lines.push(String::new());
    lines.push("#endif /* EAGLE_DOOR_ROWS_H */".to_string());
    lines.push(String::new());
    Ok(lines.join("\n"))
}
