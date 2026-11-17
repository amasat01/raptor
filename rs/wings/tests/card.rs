//! The wing target cards: the reader's refusals by name, the header made from each committed card byte for byte as committed, and the
//! weight-row macro expanded by a C compiler into a table.
use raptor_wings::card::{door_rows_header, facts_header, read_card};
use std::process::Command;

/// The committed cards and the headers made from them.
const CPU: (&str, &str) = (
    include_str!("../cards/cpu.json"),
    include_str!("../cards/cpu/raptor_facts.h"),
);
/// The device wing's card and header.
const CUDA: (&str, &str) = (
    include_str!("../cards/cuda.json"),
    include_str!("../cards/cuda/raptor_facts.h"),
);

/// The rows headers of a door build made with the host card alone and with both cards, as committed.
const DOOR_HOST: &str = include_str!("../cards/door/host/eagle_door_rows.h");
/// The rows header of a door build made with both cards.
const DOOR_BOTH: &str = include_str!("../cards/door/host_device/eagle_door_rows.h");

/// A card of a wing that is no target: its name and limits, no rows.
const NONE: &str =
    r#"{"wing": "x86", "target": "none", "linalg": {"max_dim": 32}, "note": "a card"}"#;

/// A host weight row of the given key with every value positive, as JSON.
fn row(key: (i64, i64, i64)) -> String {
    format!(
        r#"{{"vendor": {}, "family": {}, "model": {}, "arith32": 1.5, "special32": 25.0, "special64": 120.0, "launch_seconds": 1e-6,
 "split_seconds": 1e-10, "sort_seconds": 1e-8, "lane_rate_per_core": 6e9, "memory_rate": 5e10, "cache_ratio": 5.0,
 "chain_seconds_per_core": 1e-9, "vector_width": 8}}"#,
        key.0, key.1, key.2
    )
}

/// A host card with the given rows (JSON texts) and extra members at the top.
fn host(rows: &[String], extra: &str) -> String {
    format!(
        r#"{{"wing": "t", "target": "host", "linalg": {{"max_dim": 32}}, "note": "a test card", {extra} "weights": [{}]}}"#,
        rows.join(", ")
    )
}

/// The refusal text of a card that must not read.
fn refused(text: &str) -> String {
    read_card(text).expect_err("the card is refused")
}

#[test]
fn the_header_of_each_committed_card_is_the_committed_one() {
    for (card, header) in [CPU, CUDA] {
        assert_eq!(facts_header(&read_card(card).unwrap()), header);
        // the wing's header names the wing and its limits and holds no row block: a source includes one wing's header only
        assert!(!header.contains("EG_WEIGHT_ROWS") && !header.contains("EG_GEMM_ROUTE_ROWS"));
    }
    let (cpu, cuda) = (read_card(CPU.0).unwrap(), read_card(CUDA.0).unwrap());
    assert_eq!(door_rows_header(&cpu, None).unwrap(), DOOR_HOST);
    assert_eq!(door_rows_header(&cpu, Some(&cuda)).unwrap(), DOOR_BOTH);
}

#[test]
fn the_door_rows_are_refused_for_the_wrong_cards_by_name() {
    let (cpu, cuda) = (read_card(CPU.0).unwrap(), read_card(CUDA.0).unwrap());
    assert!(door_rows_header(&cuda, None)
        .unwrap_err()
        .contains("host card first"));
    assert!(door_rows_header(&cpu, Some(&cpu))
        .unwrap_err()
        .contains("two cards of one target"));
    assert!(door_rows_header(&cuda, Some(&cuda))
        .unwrap_err()
        .contains("host card first"));
    let none = read_card(NONE).unwrap();
    assert!(door_rows_header(&none, None)
        .unwrap_err()
        .contains("not a target card"));
    assert!(door_rows_header(&cpu, Some(&none))
        .unwrap_err()
        .contains("not a target card"));
}

#[test]
fn a_card_that_is_no_target_card_states_no_rows() {
    let base = NONE;
    let card = read_card(base).unwrap();
    assert!(card.rows.is_empty() && card.routes.is_empty());
    assert!(facts_header(&card).contains("RAPTOR_LINALG_MAX_DIM 32"));
    let with = base.replace(
        r#""note": "a card""#,
        &format!(r#""note": "a card", "weights": [{}]"#, row((0, 0, 0))),
    );
    assert!(refused(&with).contains("not a target card"));
    let routes = base.replace(
        r#""note": "a card""#,
        r#""note": "a card", "gemm_routes": []"#,
    );
    assert!(refused(&routes).contains("not a target card"));
}

#[test]
fn the_built_in_rows_hold_the_fitted_values_and_the_conversions() {
    let host = read_card(CPU.0).unwrap().rows;
    assert_eq!(host.len(), 1);
    // 24.5e9 plain 64-bit operations per second over 4 cores, and 0.23e-9 seconds across 4 cores times 4
    assert_eq!(host[0].values[6], 24.5e9 / 4.0);
    assert!((host[0].values[9] - 0.23e-9 * 4.0).abs() < 1e-21);
    let device = read_card(CUDA.0).unwrap().rows;
    assert_eq!(
        [
            device[0].values[6],
            device[0].values[7],
            device[0].values[9]
        ],
        [0.0; 3]
    );
}

#[test]
fn a_class_row_follows_the_built_in_row_in_the_order_of_its_key() {
    let card = read_card(&host(
        &[row((1, 6, 85)), row((0, 0, 0)), row((2, 23, 1))],
        "",
    ))
    .unwrap();
    let keys: Vec<_> = card
        .rows
        .iter()
        .map(|r| (r.vendor, r.family, r.model))
        .collect();
    assert_eq!(keys, [(0, 0, 0), (1, 6, 85), (2, 23, 1)]);
}

#[test]
fn a_missing_or_unknown_field_is_refused_by_name() {
    let short = row((0, 0, 0)).replace(r#""vector_width": 8"#, r#""vector_width": 8, "turbo": 2"#);
    assert!(refused(&host(&[short], "")).contains("unknown field turbo"));
    let missing = row((0, 0, 0)).replace(r#""sort_seconds": 1e-8, "#, "");
    assert!(refused(&host(&[missing], "")).contains("missing the field sort_seconds"));
    assert!(refused(&host(&[row((0, 0, 0))], r#""colour": 1,"#)).contains("unknown field colour"));
    let twice = row((0, 0, 0)).replace(r#""vendor": 0,"#, r#""vendor": 0, "vendor": 0,"#);
    assert!(refused(&host(&[twice], "")).contains("states vendor twice"));
}

#[test]
fn the_live_facts_are_refused_in_a_row() {
    for fact in ["cache_bytes", "elem_bytes"] {
        let stated = row((0, 0, 0)).replace(
            r#""vector_width": 8"#,
            &format!(r#""vector_width": 8, "{fact}": 8"#),
        );
        let why = refused(&host(&[stated], ""));
        assert!(why.contains(fact) && why.contains("read live"), "{why}");
    }
}

#[test]
fn two_rows_with_one_key_and_a_wrong_number_of_built_in_rows_are_refused() {
    assert!(refused(&host(
        &[row((1, 6, 85)), row((1, 6, 85)), row((0, 0, 0))],
        ""
    ))
    .contains("two weight rows hold the key (1, 6, 85)"));
    assert!(refused(&host(&[row((0, 0, 0)), row((0, 0, 0))], ""))
        .contains("two weight rows hold the key (0, 0, 0)"));
    assert!(refused(&host(&[row((1, 6, 85))], "")).contains("no built-in row"));
    assert!(refused(&host(&[], "")).contains("no built-in row"));
}

#[test]
fn a_class_row_with_a_zero_key_is_refused() {
    for key in [(0, 6, 85), (1, 0, 85), (3, 6, 85)] {
        let why = refused(&host(&[row((0, 0, 0)), row(key)], ""));
        assert!(
            why.contains("class row with a zero or unknown key"),
            "{key:?}: {why}"
        );
    }
}

#[test]
fn a_value_that_is_not_positive_is_refused_on_a_host() {
    for (field, bad) in [
        ("arith32", "0"),
        ("lane_rate_per_core", "0"),
        ("chain_seconds_per_core", "-1e-9"),
    ] {
        let text = host(&[row((0, 0, 0))], "");
        let at = text.find(&format!(r#""{field}": "#)).unwrap();
        let end = at + text[at..].find([',', '}']).unwrap();
        let edited = format!(r#"{}"{field}": {bad}{}"#, &text[..at], &text[end..]);
        let why = refused(&edited);
        assert!(
            why.contains(field) && why.contains("must be positive"),
            "{field}: {why}"
        );
    }
}

#[test]
fn a_device_row_may_leave_only_its_live_rates_at_zero() {
    let device = CUDA.0.replace(r#""arith32": 1.0"#, r#""arith32": 0.0"#);
    assert!(refused(&device).contains("arith32"));
    let negative = CUDA
        .0
        .replace(r#""memory_rate": 0.0"#, r#""memory_rate": -1.0"#);
    assert!(refused(&negative).contains("memory_rate"));
    assert!(read_card(CUDA.0).is_ok());
}

#[test]
fn the_weight_rows_macro_expands_into_a_table_a_c_compiler_reads() {
    let card = read_card(&host(&[row((1, 6, 85)), row((0, 0, 0))], "")).unwrap();
    let dir = std::env::temp_dir().join("raptor_card_c");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("raptor_facts.h"),
        door_rows_header(&card, None).unwrap(),
    )
    .unwrap();
    let source = r#"
#include "raptor_facts.h"
#include <stdio.h>
struct weights { int target, vendor, family, model; double arith32, special32, special64, launch, split, sort, lane, memory, cache, chain, vector; };
#define ROW(t, v, f, m, a, s32, s64, l, sp, so, lr, mr, cr, ch, vw) { t, v, f, m, a, s32, s64, l, sp, so, lr, mr, cr, ch, vw },
static const struct weights table[] = { EG_WEIGHT_ROWS(ROW) };
int main(void) {
  int n = (int)(sizeof table / sizeof table[0]);
  printf("%d %d %d %g\n", n, table[0].family, table[1].model, table[1].special64);
  return 0;
}
"#;
    std::fs::write(dir.join("table.c"), source).unwrap();
    let binary = dir.join("table");
    let built = Command::new("clang")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-I"])
        .arg(&dir)
        .arg(dir.join("table.c"))
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("clang runs");
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    // two rows, the built-in row first (family 0) and the class row second (model 85, special64 120)
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "2 0 85 120");
}

/// The device card with the given route rows (JSON texts) in place of its own.
fn device(routes: &[&str]) -> String {
    let head = CUDA.0.split(r#""gemm_routes""#).next().unwrap();
    format!(r#"{head}"gemm_routes": [{}]}}"#, routes.join(", "))
}

/// A route row of a device card.
fn route(family: i64, model: i64, dtype: &str, route: &str) -> String {
    format!(r#"{{"family": {family}, "model": {model}, "dtype": "{dtype}", "route": "{route}"}}"#)
}

/// The two built-in route rows every device card that states routes holds.
fn built_in() -> [String; 2] {
    [route(0, 0, "f32", "gemm"), route(0, 0, "f64", "gemm")]
}

#[test]
fn the_committed_device_card_states_the_built_in_routes_and_the_host_card_none() {
    let routes = read_card(CUDA.0).unwrap().routes;
    let keys: Vec<_> = routes
        .iter()
        .map(|r| (r.family, r.model, r.dtype_bits, r.gemm))
        .collect();
    assert_eq!(keys, [(0, 0, 32, true), (0, 0, 64, true)]);
    assert!(read_card(CPU.0).unwrap().routes.is_empty());
}

#[test]
fn a_class_route_row_follows_the_built_in_rows_in_the_order_of_its_key() {
    let [a, b] = built_in();
    let (c, d) = (route(7, 5, "f64", "direct"), route(7, 0, "f32", "direct"));
    let card = read_card(&device(&[&c, &a, &d, &b])).unwrap();
    let keys: Vec<_> = card
        .routes
        .iter()
        .map(|r| (r.family, r.model, r.dtype_bits, r.gemm))
        .collect();
    assert_eq!(
        keys,
        [
            (0, 0, 32, true),
            (0, 0, 64, true),
            (7, 0, 32, false),
            (7, 5, 64, false)
        ]
    );
}

#[test]
fn a_route_row_with_a_missing_or_unknown_field_or_word_is_refused_by_name() {
    let [a, b] = built_in();
    let with = |extra: String| refused(&device(&[&a, &b, &extra]));
    assert!(
        with(route(7, 0, "f32", "direct").replace(r#", "route": "direct""#, ""))
            .contains("missing the field route")
    );
    assert!(
        with(route(7, 0, "f32", "direct").replace(r#""dtype": "f32", "#, ""))
            .contains("missing the field dtype")
    );
    assert!(
        with(route(7, 0, "f32", "direct").replace(r#""model": 0, "#, ""))
            .contains("missing the field model")
    );
    assert!(
        with(route(7, 0, "f32", "direct").replace("}", r#", "fast": 1}"#))
            .contains("unknown field fast")
    );
    assert!(with(route(7, 0, "f32", "blas")).contains("route blas: gemm or direct"));
    assert!(with(route(7, 0, "i32", "gemm")).contains("dtype i32"));
}

#[test]
fn route_rows_with_one_key_or_without_a_built_in_row_for_each_width_are_refused() {
    let [a, b] = built_in();
    let again = route(0, 0, "f32", "direct");
    assert!(refused(&device(&[&a, &b, &again]))
        .contains("two route rows hold the key (device, 0, 0, 32 bits)"));
    assert!(
        refused(&device(&[&a])).contains("no built-in route row (family and model 0) for 64 bits")
    );
    assert!(
        refused(&device(&[&b])).contains("no built-in route row (family and model 0) for 32 bits")
    );
    let class = route(7, 0, "f64", "gemm");
    assert!(refused(&device(&[&a, &class])).contains("for 64 bits"));
}

#[test]
fn a_route_class_row_with_a_zero_key_is_refused() {
    let [a, b] = built_in();
    let why = refused(&device(&[&a, &b, &route(0, 5, "f32", "gemm")]));
    assert!(why.contains("class row with a zero key"), "{why}");
}

#[test]
fn route_rows_are_refused_on_a_card_that_is_not_the_device_card() {
    let [a, b] = built_in();
    let text = host(&[row((0, 0, 0))], &format!(r#""gemm_routes": [{a}, {b}],"#));
    assert!(refused(&text).contains("not the device target card"));
}

#[test]
fn the_route_rows_macro_expands_into_a_table_a_c_compiler_reads() {
    let [a, b] = built_in();
    let class = route(7, 5, "f64", "direct");
    let card = read_card(&device(&[&class, &a, &b])).unwrap();
    let dir = std::env::temp_dir().join("raptor_card_routes_c");
    std::fs::create_dir_all(&dir).unwrap();
    let rows = door_rows_header(&read_card(CPU.0).unwrap(), Some(&card)).unwrap();
    std::fs::write(dir.join("raptor_facts.h"), rows).unwrap();
    let source = r#"
#include "raptor_facts.h"
#include <stdio.h>
struct route { int target, family, model, bits, route; };
#define ROW(t, f, m, b, r) { t, f, m, b, r },
static const struct route table[] = { EG_GEMM_ROUTE_ROWS(ROW) };
int main(void) {
  int n = (int)(sizeof table / sizeof table[0]);
  printf("%d %d %d %d\n", n, table[0].bits, table[2].family, table[2].route);
  return 0;
}
"#;
    std::fs::write(dir.join("routes.c"), source).unwrap();
    let binary = dir.join("routes");
    let built = Command::new("clang")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-I"])
        .arg(&dir)
        .arg(dir.join("routes.c"))
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("clang runs");
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    // three rows: the two built-in rows first (32 bits first), then the class row, family 7, direct (0)
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "3 32 7 0");
}

/// Counts the rows of each macro in a door rows header by a C compiler: prints weights, routes and the first device weight row's
/// built-in lane rate (0, live) so that a value is read as well.
fn rows_of(header: &str, name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("raptor_card_door_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("eagle_door_rows.h"), header).unwrap();
    let source = r#"
#include "eagle_door_rows.h"
#include <stdio.h>
#define COUNT_WEIGHT(t, v, f, m, a, s32, s64, l, sp, so, lr, mr, cr, ch, vw) +1
#define COUNT_ROUTE(t, f, m, b, r) +1
#define LAST_TARGET(t, v, f, m, a, s32, s64, l, sp, so, lr, mr, cr, ch, vw) last = t;
int main(void) {
  int last = -1;
  EG_WEIGHT_ROWS(LAST_TARGET)
  printf("%d %d %d\n", 0 EG_WEIGHT_ROWS(COUNT_WEIGHT), 0 EG_GEMM_ROUTE_ROWS(COUNT_ROUTE), last);
  return 0;
}
"#;
    std::fs::write(dir.join("rows.c"), source).unwrap();
    let binary = dir.join("rows");
    let built = Command::new("clang")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-I"])
        .arg(&dir)
        .arg(dir.join("rows.c"))
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("clang runs");
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    String::from_utf8_lossy(&Command::new(&binary).output().unwrap().stdout)
        .trim()
        .to_string()
}

#[test]
fn both_committed_door_headers_expand_both_macros_in_a_c_compiler() {
    // weight rows, route rows, and the target of the last weight row: the host alone has one weight row and an empty route macro
    assert_eq!(rows_of(DOOR_HOST, "host"), "1 0 0");
    assert_eq!(rows_of(DOOR_BOTH, "both"), "2 2 1");
}
