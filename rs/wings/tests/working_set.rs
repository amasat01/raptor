//! The working-set figure of a kernel against the values the Python model recorded for the same graphs.
//!
//! Oracle: `tests/data/working_set_oracle.json`, the figure of the Python model (`op_bytes` and the rolled-site test of the old IR
//! package) for each kernel under several fold thresholds and forms. Each case holds the kernel's IR dump, its sinks and its walk order
//! as positions in the dump (the library's own test reads the sinks).
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_ctx_free, rx_ctx_new, rx_last_error, rx_load, RX_OK};
use raptor_wings::working_set::working_set_bytes;
use rcore::json::{parse, J};
use std::ffi::CStr;

/// The recorded cases of the oracle file.
const ORACLE: &str = include_str!("data/working_set_oracle.json");

/// The recorded cases.
fn cases() -> Vec<J> {
    parse(ORACLE).unwrap().get("cases").unwrap().arr().to_vec()
}

/// A case's dump loaded into a fresh context, with the handles of its walk order.
fn load(case: &J) -> (*mut Ctx, Vec<u32>) {
    let c = rx_ctx_new();
    let text = case.get("dump").and_then(J::str).unwrap();
    let (mut handles, mut n) = (vec![0u32; 4096], 0);
    let status = rx_load(
        c,
        text.as_ptr().cast(),
        text.len(),
        handles.as_mut_ptr(),
        handles.len(),
        &mut n,
    );
    let why = unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy();
    assert_eq!(status, RX_OK, "{why}");
    let order = case.get("order").unwrap().arr();
    (
        c,
        order
            .iter()
            .map(|p| handles[usize::try_from(p.int()).unwrap()])
            .collect(),
    )
}

#[test]
fn the_figure_of_each_recorded_kernel_is_what_the_python_model_gives() {
    let (mut kernels, mut figures) = (0, 0);
    for case in cases() {
        let name = case.get("name").and_then(J::str).unwrap();
        let (c, order) = load(&case);
        // SAFETY: `c` is the live context `load` made and nothing else holds it.
        let ctx = unsafe { &*c };
        for row in case.get("figures").unwrap().arr() {
            let fold_min = row.get("rolled_fold_min").unwrap().int();
            let form = row.get("form").and_then(J::str).unwrap();
            let got = working_set_bytes(ctx, &order, Some(fold_min), form)
                .unwrap_or_else(|why| panic!("{name} {form} {fold_min}: {why:?}"));
            assert_eq!(
                got,
                row.get("bytes").unwrap().int(),
                "{name} under {form} with fold threshold {fold_min}"
            );
            figures += 1;
        }
        kernels += 1;
        rx_ctx_free(c);
    }
    assert_eq!(
        (kernels, figures),
        (15, 60),
        "the oracle holds the recorded kernels and figures"
    );
}

#[test]
fn a_build_without_a_fold_threshold_is_refused_where_one_is_read() {
    let case = cases()
        .into_iter()
        .find(|c| c.get("name").and_then(J::str) == Some("mv_rolled"))
        .unwrap();
    let (c, order) = load(&case);
    // SAFETY: `c` is the live context `load` made and nothing else holds it.
    let ctx = unsafe { &*c };
    assert!(working_set_bytes(ctx, &order, None, "auto").is_err());
    assert!(working_set_bytes(ctx, &order, None, "static").is_ok());
    rx_ctx_free(c);
}
