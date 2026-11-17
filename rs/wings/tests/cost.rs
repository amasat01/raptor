//! The per-branch cost of dispatch groups against the Python model's own answers.
//!
//! Oracle: `tests/data/cost_oracle.json`, written by `tests/tools/gen_cost_oracle.py` by running the Python cost model
//! (`dispatch_costs`) on each kernel. Each case holds the kernel's IR dump, its walk order as positions in the dump, and the
//! recorded cost of every branch of every dispatch group.
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_ctx_free, rx_ctx_new, rx_last_error, rx_load, RX_OK};
use raptor_wings::cost::{dispatch_costs, fp64_ops, BranchCost, GroupCost};
use rcore::json::{parse, J};
use std::ffi::CStr;

/// The recorded cases of the oracle file.
const ORACLE: &str = include_str!("data/cost_oracle.json");

/// A recorded number of a branch row, by field name (a missing or null `stores` is None).
fn recorded(row: &J, field: &str) -> Option<f64> {
    match row.get(field) {
        Some(J::Num(n)) => Some(*n),
        _ => None,
    }
}

/// The branch cost the oracle recorded in `row`.
fn expected_branch(row: &J) -> BranchCost {
    let count = |field: &str| recorded(row, field).expect("every counted field is recorded");
    BranchCost {
        arith32: count("arith32"),
        arith64: count("arith64"),
        special32: count("special32"),
        special64: count("special64"),
        loads: count("loads"),
        gathers: count("gathers"),
        chain32: count("chain32"),
        chain64: count("chain64"),
        chain_special32: count("chainSpecial32"),
        chain_special64: count("chainSpecial64"),
        stores: recorded(row, "stores"),
    }
}

/// The group cost the oracle recorded.
fn expected_group(group: &J) -> GroupCost {
    let selector_reads = group.get("selector_reads").expect("reads").arr();
    GroupCost {
        branches: group
            .get("branches")
            .expect("branches")
            .arr()
            .iter()
            .map(expected_branch)
            .collect(),
        selector_reads: selector_reads
            .iter()
            .map(|pair| {
                (
                    pair.arr()[0].str().unwrap().into(),
                    pair.arr()[1].str().unwrap().into(),
                )
            })
            .collect(),
    }
}

/// Loads a case's dump into a fresh context: the context and the handle of each dump position.
fn load_case(case: &J) -> (*mut Ctx, Vec<u32>) {
    let c = rx_ctx_new();
    let text = case.get("dump").and_then(J::str).expect("dump text");
    let mut roots = vec![0u32; case.get("order").expect("order").arr().len() + 4096];
    let mut n = 0;
    let status = rx_load(
        c,
        text.as_ptr().cast(),
        text.len(),
        roots.as_mut_ptr(),
        roots.len(),
        &mut n,
    );
    let why = unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy();
    assert_eq!(status, RX_OK, "{why}");
    roots.truncate(n);
    (c, roots)
}

/// The handles of a case's walk order.
fn walk_order(case: &J, handles: &[u32]) -> Vec<u32> {
    case.get("order")
        .expect("order")
        .arr()
        .iter()
        .map(|p| handles[usize::try_from(p.int()).expect("a position")])
        .collect()
}

/// The cases of the oracle.
fn cases() -> Vec<J> {
    parse(ORACLE)
        .expect("the oracle file is JSON")
        .get("cases")
        .expect("cases")
        .arr()
        .to_vec()
}

#[test]
fn every_branch_of_every_recorded_group_costs_what_the_python_model_says() {
    let (mut kernels, mut branches) = (0, 0);
    for case in cases() {
        let name = case.get("name").and_then(J::str).unwrap();
        let (c, handles) = load_case(&case);
        let order = walk_order(&case, &handles);
        // SAFETY: `c` is the live context `load_case` made and nothing else holds it.
        let ctx = unsafe { &*c };
        let got = dispatch_costs(ctx, &order).unwrap_or_else(|why| panic!("{name}: {why:?}"));
        let want: Vec<GroupCost> = case
            .get("groups")
            .unwrap()
            .arr()
            .iter()
            .map(expected_group)
            .collect();
        assert_eq!(got, want, "{name}");
        kernels += 1;
        branches += want.iter().map(|g| g.branches.len()).sum::<usize>();
        rx_ctx_free(c);
    }
    assert_eq!(
        (kernels, branches),
        (29, 71),
        "the oracle holds the recorded kernels and branches"
    );
}

#[test]
fn the_64_bit_total_of_a_walk_is_what_the_python_model_counts() {
    for case in cases() {
        let name = case.get("name").and_then(J::str).unwrap();
        let (c, handles) = load_case(&case);
        let order: Vec<u32> = walk_order(&case, &handles);
        // SAFETY: `c` is the live context `load_case` made and nothing else holds it.
        let ctx = unsafe { &*c };
        let want = case.get("fp64_ops").unwrap().int();
        assert_eq!(fp64_ops(ctx, &order).unwrap(), want, "{name}");
        rx_ctx_free(c);
    }
}
