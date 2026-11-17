//! The contraction recogniser against the old recogniser's own answers.
//!
//! Oracle: `tests/data/contraction_oracle.json`, written by `tests/tools/gen_contraction_oracle.py` by running
//! `hawk.passes.contraction.recognize` on each kernel. Each case holds the kernel's IR dump, its sinks as positions in the dump,
//! the names of its staged planes and the record the old recogniser answered (null: not a contraction).
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_ctx_free, rx_ctx_new, rx_last_error, rx_load, RX_OK};
use raptor_wings::contraction::{recognize, Contraction, OperandRole, Stride, TermForm};
use rcore::json::{parse, J};
use std::ffi::CStr;

/// The recorded cases of the oracle file.
const ORACLE: &str = include_str!("data/contraction_oracle.json");

/// The recorded text of a field.
fn text(row: &J, field: &str) -> String {
    row.get(field).and_then(J::str).expect("a text").to_string()
}

/// The recorded whole number of a field.
fn number(row: &J, field: &str) -> i64 {
    match row.get(field) {
        Some(J::Num(n)) => *n as i64,
        _ => panic!("{field} is a number"),
    }
}

/// The recorded flag of a field.
fn flag(row: &J, field: &str) -> bool {
    matches!(row.get(field), Some(J::Bool(true)))
}

/// The recorded reasons of a field.
fn reasons(row: &J) -> Vec<String> {
    row.get("refusals")
        .expect("refusals")
        .arr()
        .iter()
        .map(|r| r.str().expect("a text").to_string())
        .collect()
}

/// Checks the answer for one case against its record.
fn check(name: &str, got: Option<Contraction>, want: &J) {
    let Some(got) = got else {
        assert!(
            matches!(want, J::Null),
            "{name}: not recognised, the oracle recognised it"
        );
        return;
    };
    assert!(
        !matches!(want, J::Null),
        "{name}: recognised, the oracle did not"
    );
    assert_eq!(got.carry, text(want, "carry"), "{name}");
    assert_eq!(got.reduce_var, text(want, "reduce_var"), "{name}");
    assert_eq!(
        (got.reduce_extent, got.reduce_start, got.reduce_step),
        (
            number(want, "reduce_extent"),
            number(want, "reduce_start"),
            number(want, "reduce_step")
        ),
        "{name}"
    );
    let class = if got.operands.len() == 1 {
        "gemv"
    } else {
        "gemm"
    };
    assert_eq!(class, text(want, "operand_class"), "{name}");
    let form = match got.term_form {
        TermForm::BareRead => "bare_read",
        TermForm::ProductOfReads => "product_of_reads",
        TermForm::Transformed => "transformed",
    };
    assert_eq!(form, text(want, "term_form"), "{name}");
    let kinds = ["scalar", "vector", "matrix"];
    let outputs: Vec<(String, String)> = got
        .outputs
        .iter()
        .map(|(n, r)| (n.clone(), kinds[*r].to_string()))
        .collect();
    let recorded: Vec<(String, String)> = want
        .get("outputs")
        .expect("outputs")
        .arr()
        .iter()
        .map(|o| {
            (
                o.arr()[0].str().unwrap().to_string(),
                o.arr()[1].str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(outputs, recorded, "{name}");
    assert_eq!(
        (got.dense, got.refusals.clone()),
        (flag(want, "dense"), reasons(want)),
        "{name}"
    );
    let fanin: Vec<(String, Option<i64>)> = want
        .get("staged_fanin")
        .expect("fanin")
        .arr()
        .iter()
        .map(|f| {
            let n = match &f.arr()[1] {
                J::Num(n) => Some(*n as i64),
                _ => None,
            };
            (f.arr()[0].str().unwrap().to_string(), n)
        })
        .collect();
    assert_eq!(got.staged_fanin, fanin, "{name}");
    let operands = want.get("operands").expect("operands").arr();
    assert_eq!(got.operands.len(), operands.len(), "{name}");
    for (op, row) in got.operands.iter().zip(operands) {
        let who = format!("{name}/{}", op.name);
        assert_eq!(op.name, text(row, "name"), "{who}");
        let stride = match row.get("reduce_stride") {
            Some(J::Num(n)) => Some(Stride::Int(*n as i64)),
            Some(J::Str(s)) => Some(Stride::Symbol(s.clone())),
            _ => None,
        };
        assert_eq!(op.reduce_stride, stride, "{who}");
        assert_eq!(
            (op.dense, op.refusals.clone()),
            (flag(row, "dense"), reasons(row)),
            "{who}"
        );
        let role = text(row, "role");
        let expected = match role.as_str() {
            "staged" => OperandRole::Staged,
            "wide" => OperandRole::Wide,
            "computed" => OperandRole::Computed,
            _ => OperandRole::Plane,
        };
        assert_eq!(op.role, expected, "{who}: {role}");
    }
}

/// Loads a case's dump into a fresh context: the context and the handle of each dump position.
fn load_case(case: &J) -> (*mut Ctx, Vec<u32>) {
    let c = rx_ctx_new();
    let dump = case.get("dump").and_then(J::str).expect("dump text");
    let mut roots = vec![0u32; 4096];
    let mut n = 0;
    let status = rx_load(
        c,
        dump.as_ptr().cast(),
        dump.len(),
        roots.as_mut_ptr(),
        roots.len(),
        &mut n,
    );
    let why = unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy();
    assert_eq!(status, RX_OK, "{why}");
    roots.truncate(n);
    (c, roots)
}

#[test]
fn every_recorded_kernel_is_recognised_as_the_old_recogniser_did() {
    let oracle = parse(ORACLE).expect("the oracle parses");
    let cases = oracle.get("cases").expect("cases");
    let J::Obj(cases) = cases else {
        panic!("cases is an object")
    };
    assert!(cases.len() >= 20, "{} cases", cases.len());
    for (name, case) in cases {
        let (c, roots) = load_case(case);
        let sinks: Vec<u32> = case
            .get("sinks")
            .expect("sinks")
            .arr()
            .iter()
            .map(|p| match p {
                J::Num(n) => roots[*n as usize],
                _ => panic!("{name}: a sink with no position"),
            })
            .collect();
        let staged: Vec<String> = case
            .get("staged")
            .expect("staged")
            .arr()
            .iter()
            .map(|s| s.str().unwrap().to_string())
            .collect();
        // SAFETY: `c` is the live context made above; nothing else holds it.
        let got = recognize(unsafe { &mut *c }, &sinks, &staged).expect("an answer");
        check(name, got, case.get("record").expect("record"));
        rx_ctx_free(c);
    }
}
