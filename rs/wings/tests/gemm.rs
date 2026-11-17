//! The matrix product of a contraction against what the old route planned.
//!
//! Oracle: `tests/data/gemm_oracle.json`, written by `tests/tools/gen_gemm_oracle.py` by running `eagle.gemm._route._Route` (the
//! route's own planning) on each kernel for two lane counts. Each case holds the kernel's IR dump, its sinks, its staged planes and
//! the plans (null: the old route did not take the kernel).
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_ctx_free, rx_ctx_new, rx_last_error, rx_load, RX_OK};
use raptor_wings::contraction::recognize;
use raptor_wings::gemm::{gemm_plan, Extent};
use rcore::json::{parse, J};
use std::ffi::CStr;

/// The recorded cases of the oracle file.
const ORACLE: &str = include_str!("data/gemm_oracle.json");

/// The kernels the old route took that the strides alone do not let the library state: the free extent of the operand read
/// by the major half of the lane split is declared on a plane whose contracted axis is the minor one.
const FREE_EXTENT_NOT_STATED: [&str; 2] = ["both_minor", "f_minor"];

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

/// The stored shape the oracle recorded for a plane.
fn recorded_shape(plan: &J, name: &str) -> (i64, i64) {
    let shape = plan
        .get("shapes")
        .and_then(|s| s.get(name))
        .expect("a shape")
        .arr();
    let n = |i: usize| {
        if let J::Num(n) = &shape[i] {
            *n as i64
        } else {
            panic!("a number")
        }
    };
    (n(0), n(1))
}

/// A stored extent made a number for `lanes` lanes sharing slots in groups of `fanin`.
fn resolved(extent: Extent, lanes: i64, fanin: i64) -> i64 {
    match extent {
        Extent::Fixed(n) => n,
        Extent::PerLane => lanes / fanin,
    }
}

#[test]
fn the_forward_product_of_every_recorded_kernel_is_the_one_the_old_route_planned() {
    let oracle = parse(ORACLE).expect("the oracle parses");
    let J::Obj(cases) = oracle.get("cases").expect("cases") else {
        panic!("cases is an object")
    };
    let mut stated = 0;
    for (name, case) in cases {
        let (c, roots) = load_case(case);
        let sinks: Vec<u32> = case
            .get("sinks")
            .expect("sinks")
            .arr()
            .iter()
            .map(|p| {
                if let J::Num(n) = p {
                    roots[*n as usize]
                } else {
                    panic!("a sink")
                }
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
        let found = recognize(unsafe { &mut *c }, &sinks, &staged).expect("an answer");
        let plan = found.as_ref().and_then(gemm_plan);
        rx_ctx_free(c);
        let J::Null = case.get("plans").expect("plans") else {
            if FREE_EXTENT_NOT_STATED.contains(&name.as_str()) {
                assert!(
                    plan.is_none(),
                    "{name}: stated a shape the strides do not state"
                );
                continue;
            }
            let plan = plan.unwrap_or_else(|| panic!("{name}: the old route took it"));
            stated += 1;
            for recorded in case.get("plans").unwrap().arr() {
                let lanes = if let Some(J::Num(n)) = recorded.get("n") {
                    *n as i64
                } else {
                    panic!("n")
                };
                let step = &recorded.get("forward").expect("forward").arr()[0];
                let text = |f: &str| step.get(f).and_then(J::str).expect("a name").to_string();
                let flag = |f: &str| matches!(step.get(f), Some(J::Bool(true)));
                let call = &plan.forward;
                assert_eq!(
                    (call.a.clone(), call.b.clone(), call.c.clone()),
                    (text("a"), text("b"), text("out")),
                    "{name}"
                );
                assert_eq!(
                    (call.trans_a, call.trans_b, flag("transpose_out")),
                    (flag("transpose_a"), flag("transpose_b"), false),
                    "{name}"
                );
                let fanin = plan.lane_fanin.max(1);
                let adjoint = recorded.get("adjoint").expect("adjoint").arr();
                for (call, step) in plan.adjoint.iter().zip(adjoint) {
                    let text = |f: &str| step.get(f).and_then(J::str).expect("a name").to_string();
                    let flag = |f: &str| matches!(step.get(f), Some(J::Bool(true)));
                    assert_eq!(
                        (call.a.clone(), call.b.clone(), call.c.clone()),
                        (text("a"), text("b"), text("out")),
                        "{name}"
                    );
                    assert_eq!(
                        (call.trans_a, call.trans_b, call.trans_c, call.accumulate),
                        (
                            flag("transpose_a"),
                            flag("transpose_b"),
                            flag("transpose_out"),
                            true
                        ),
                        "{name}"
                    );
                    for (shape, who) in call.shapes.iter().zip([&call.a, &call.b, &call.c]) {
                        let got = (
                            resolved(shape.0, lanes, fanin),
                            resolved(shape.1, lanes, fanin),
                        );
                        assert_eq!(got, recorded_shape(recorded, who), "{name}/{who}");
                    }
                }
                for (shape, who) in call.shapes.iter().zip([&call.a, &call.b, &call.c]) {
                    let got = (
                        resolved(shape.0, lanes, fanin),
                        resolved(shape.1, lanes, fanin),
                    );
                    assert_eq!(got, recorded_shape(recorded, who), "{name}/{who}");
                }
            }
            continue;
        };
        assert!(
            plan.is_none(),
            "{name}: stated a product the old route did not take"
        );
    }
    assert!(stated >= 5, "{stated} routed kernels stated");
}
