// Generates the class specs (class names, slots, dedup recipes, watch flags, constants) from raptor/ir/core.json, the manifest of
// the core kinds. No dependency: the reader is src/json.rs, shared with the binding.
#[path = "src/json.rs"]
mod json;
use std::{env, fs, path::Path};
/// A string as a quoted Rust literal.
fn q(s: &str) -> String {
    format!("{:?}", s)
}
/// One class of the manifest: (class, base class, slots, binds, dedup code, watch).
type ClassRow = (
    String,
    String,
    Vec<String>,
    Option<bool>,
    Option<String>,
    Option<u8>,
    Vec<String>,
    Vec<String>,
);

/// One row of the rule table: (min, max or -1, result code, operand index, dtype, shape).
type RuleRow = (i64, i64, u8, i64, String, Vec<i64>);

/// Writes the constants of the manifest that are plain lists and strings (the ones without a group prefix).
fn emit_constants(m: &json::J, out: &mut String) {
    for (k, v) in m.get("consts").unwrap().obj() {
        if k.contains('.') {
            continue;
        } // "group.NAME": a fact held once for both languages, emitted below as a typed constant
        if let json::J::Arr(a) = v {
            let items: Vec<String> = a.iter().map(|x| q(x.str().unwrap())).collect();
            out.push_str(&format!("    ({}, &[{}]),\n", q(k), items.join(", ")));
        }
    }
    out.push_str("];\n/// The string constants of the manifest, by name.\npub static STR_CONSTS: &[(&str, &str)] = &[\n");
    for (k, v) in m.get("consts").unwrap().obj() {
        if let json::J::Str(x) = v {
            if !k.contains('.') {
                out.push_str(&format!("    ({}, {}),\n", q(k), q(x)))
            }
        }
    }
    out.push_str("];\n");
}

/// Writes the facts of the manifest (names with a group prefix) as typed constants and the table that lists them.
fn emit_facts(m: &json::J, out: &mut String) {
    // the facts: an integer, a string or a list of strings, as a typed constant of `facts` (the name after the group)
    out.push_str(
        "/// A fact of the manifest: an integer, a string or a list of strings.\npub enum Fact {\n    /// An integer fact.\n    I(i64),\n    /// A string fact.\n    S(&'static str),\n    /// A list fact.\n    L(&'static [&'static str]),\n}\n/// The facts of the manifest as typed constants, named after the part past the group.\npub mod facts {\n",
    );
    let mut rows = String::new();
    for (k, v) in m.get("consts").unwrap().obj() {
        let Some((_, name)) = k.split_once('.') else {
            continue;
        };
        let kind = match v {
            json::J::Num(_) => "I",
            json::J::Str(_) => "S",
            _ => "L",
        };
        rows.push_str(&format!(
            "    ({}, {}),\n",
            q(k),
            if kind == "L" {
                format!("Fact::L(&facts::{name})")
            } else {
                format!("Fact::{kind}(facts::{name})")
            }
        ));
        match v {
            json::J::Num(n) => out.push_str(&format!(
                "    /// The fact {k}.\n    pub const {name}: i64 = {};\n",
                *n as i64
            )),
            json::J::Str(s) => out.push_str(&format!(
                "    /// The fact {k}.\n    pub const {name}: &str = {};\n",
                q(s)
            )),
            json::J::Arr(a) => out.push_str(&format!(
                "    /// The fact {k}.\n    pub const {name}: [&str; {}] = [{}];\n",
                a.len(),
                a.iter()
                    .map(|x| q(x.str().unwrap()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            _ => panic!("fact {k}"),
        }
    }
    out.push_str("}\n");
    out.push_str(&format!("/// the facts as compiled, by \"group.NAME\" (what the binding reports to the drift test)\npub static FACT_ROWS: &[(&str, Fact)] = &[\n{rows}];\n"));
}

/// Reads the text of one dedup item of the manifest (`f:name`, `r:name`, `k`, ...) as the source of an `Item`.
fn item(x: &str) -> String {
    if x == "k" {
        return "Item::K".into();
    }
    let p: Vec<&str> = x.splitn(3, ':').collect();
    match p[0] {
        "f" => format!("Item::Field({})", q(p[1])),
        "r" => format!("Item::ReprOf({})", q(p[1])),
        "s" => format!("Item::Lit({})", q(p[1])),
        "names" => format!("Item::Names({})", q(p[1])),
        "skeys" => format!("Item::SinkKeys({})", q(p[1])),
        "body" => "Item::Body".to_string(),
        "litif" => format!("Item::LitIfSet({}, {})", q(p[1]), q(p[2])),
        "fieldif" => format!("Item::FieldIfSet({}, {})", q(p[1]), q(p[2])),
        "fx" => format!(
            "Item::FieldUnless({}, &[{}])",
            q(p[1]),
            p[2].split(',').map(q).collect::<Vec<_>>().join(", ")
        ),
        _ => panic!("dedup item {x}"),
    }
}

/// Reads every class of the manifest, with the binding, dedup recipe and watch flag a subclass takes from its base class.
fn read_classes(kinds: &[json::J]) -> Vec<ClassRow> {
    // (class, base class, slots, binds, dedup code, watch)
    let mut cl: Vec<ClassRow> = vec![];
    for k in kinds {
        let name = k.get("class").unwrap().str().unwrap().to_string();
        let base = match k.get("refines").and_then(|r| r.str()) {
            Some(r) => kinds
                .iter()
                .find(|x| x.get("name").unwrap().str() == Some(r))
                .unwrap()
                .get("class")
                .unwrap()
                .str()
                .unwrap()
                .to_string(),
            None => String::new(),
        };
        let slots = k.get("slots").map(|s| s.strs()).unwrap_or_default();
        let binds = k.get("binding").and_then(|b| b.str()).map(|b| b == "slot");
        let dedup = k.get("dedup").map(|d| {
            let w = d.strs();
            match w[0].as_str() {
                "none" => "Dedup::None".to_string(),
                "py" => "Dedup::Py".to_string(),
                "one" => format!("Dedup::One({})", item(&w[1])),
                "tup" => format!(
                    "Dedup::Tup(&[{}])",
                    w[1..]
                        .iter()
                        .map(|x| item(x))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                _ => panic!("dedup mode"),
            }
        });
        let watch = k
            .get("watch")
            .and_then(|w| w.str())
            .map(|w| if w == "all" { 1 } else { 2 });
        let feathers = k.get("feathers").map(|f| f.strs()).unwrap_or_default();
        let state = k.get("state").map(|f| f.strs()).unwrap_or_default();
        cl.push((name, base, slots, binds, dedup, watch, feathers, state));
    }
    for i in 0..cl.len() {
        let base = cl[i].1.clone();
        if base.is_empty() {
            continue;
        }
        let b = cl.iter().find(|c| c.0 == base).unwrap().clone();
        if cl[i].3.is_none() {
            cl[i].3 = b.3
        }
        if cl[i].4.is_none() {
            cl[i].4 = b.4.clone()
        }
        if cl[i].5.is_none() {
            cl[i].5 = b.5
        }
    }
    cl
}

/// Reads the operand-count and result-type rule of one kind of the manifest.
fn rule_of(k: &json::J) -> RuleRow {
    let (mut lo, mut hi) = (0i64, -1i64);
    if let Some(o) = k.get("operands") {
        lo = o.get("min").map(|x| x.int()).unwrap_or(0);
        hi = match o.get("max") {
            Some(m) => {
                if m.str() == Some("many") {
                    -1
                } else {
                    m.int()
                }
            }
            None => -1,
        };
    }
    let (mut code, mut idx, mut dt, mut sh) = (0u8, 0i64, String::new(), vec![]);
    if let Some(r) = k.get("result") {
        let a = r.arr();
        match a[0].str().unwrap() {
            "carried" => {}
            "same_as" => {
                code = 1;
                idx = a[1].int()
            }
            "scalar_of" => {
                code = 2;
                idx = a[1].int()
            }
            "fixed" => {
                code = 3;
                dt = a[1].str().unwrap().to_string();
                sh = a[2].arr().iter().map(|x| x.int()).collect()
            }
            w => panic!("result rule {w}"),
        }
    }
    (lo, hi, code, idx, dt, sh)
}

/// Writes the table of rules, one row per kind of the manifest.
fn emit_rules(kinds: &[json::J], out: &mut String) {
    out.push_str(
        "/// The operand-count and result-type rule of each kind: (min, max or -1, result code, operand index, dtype, shape).\n#[allow(clippy::type_complexity)] // a generated table of rows, one tuple per class\n",
    );
    out.push_str("pub static RULES: &[(u32, i32, u8, u32, &str, &[i64])] = &[\n");
    for k in kinds {
        let (lo, hi, code, idx, dt, sh) = rule_of(k);
        out.push_str(&format!(
            "    ({lo}, {hi}, {code}, {idx}, {}, &[{}]),\n",
            q(&dt),
            sh.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out.push_str("];\n");
}

/// Writes the table of class specs.
fn emit_classes(cl: &[ClassRow], out: &mut String) {
    out.push_str("/// The node classes of the manifest, in its order.\npub static CLASSES: &[ClassSpec] = &[\n");
    for (n, b, s, bind, d, wt, f, st) in cl {
        out.push_str(&format!("    ClassSpec {{ name: {}, base: {}, slots: &[{}], binds: {}, dedup: {}, watch: {}, feathers: &[{}], state: &[{}] }},\n", q(n), q(b),
            s.iter().map(|x| q(x)).collect::<Vec<_>>().join(", "), bind.unwrap_or(false), d.clone().unwrap_or("Dedup::Py".into()), wt.unwrap_or(0), f.iter().map(|x| q(x)).collect::<Vec<_>>().join(", "), st.iter().map(|x| q(x)).collect::<Vec<_>>().join(", ")));
    }
    out.push_str("];\n");
}

/// Writes the generated specs file into the build output directory.
fn main() {
    println!("cargo:rerun-if-changed=../../raptor/ir/core.json");
    let m = json::parse(&fs::read_to_string("../../raptor/ir/core.json").unwrap()).unwrap();
    let mut out = String::from("/// The list constants of the manifest, by name.\npub static CONSTS: &[(&str, &[&str])] = &[\n");
    emit_constants(&m, &mut out);
    emit_facts(&m, &mut out);
    let kinds = m.get("kinds").unwrap().arr();
    let cl = read_classes(kinds);
    emit_rules(kinds, &mut out);
    emit_classes(&cl, &mut out);
    fs::write(
        Path::new(&env::var("OUT_DIR").unwrap()).join("specs.rs"),
        out,
    )
    .unwrap();
}
