//! Tests of the chain composition, the cut rule and the pieces, on a chain of two small kernels: the first writes `y` from `x`,
//! the second writes `z` from `y`.
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_ctx_free, rx_ctx_new, rx_last_error, rx_make};
use raptor_wings::chain::{bind, cuts, pieces, FusionRow, Member};
use std::collections::HashSet;
use std::ffi::{CStr, CString};

/// A node of the class with the given type, operands and fields text.
fn make(c: *mut Ctx, cls: &str, ops: &[u32], fields: &str) -> u32 {
    let (cls, dtype, f) = (
        CString::new(cls).unwrap(),
        CString::new("f64").unwrap(),
        CString::new(fields).unwrap(),
    );
    let mut out = 0;
    let status = rx_make(
        c,
        cls.as_ptr(),
        dtype.as_ptr(),
        std::ptr::null(),
        0,
        std::ptr::null(),
        ops.as_ptr(),
        ops.len(),
        f.as_ptr(),
        f.as_bytes().len(),
        &mut out,
    );
    let why = unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy();
    assert_eq!(status, 0, "{why}");
    out
}

/// A read of the plane `name` as a per-sample value.
fn leaf(c: *mut Ctx, name: &str) -> u32 {
    let fields = format!(
        "((\"kind\", \"vocab_read\"), (\"role\", \"per_sample\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"
    );
    make(c, "Leaf", &[], &fields)
}

/// A write of `value` to the mutable plane `name`.
fn assign(c: *mut Ctx, name: &str, value: u32) -> u32 {
    let fields =
        format!("((\"kind\", \"assign\"), (\"name\", \"{name}\"), (\"role\", \"mutable\"))");
    make(c, "Assign", &[value], &fields)
}

/// The exponential of a value.
fn exp(c: *mut Ctx, value: u32) -> u32 {
    make(
        c,
        "Op",
        &[value],
        "((\"kind\", \"exp\"), (\"literal\", None))",
    )
}

/// The two kernels' sinks.
fn kernels(c: *mut Ctx) -> ([u32; 1], [u32; 1]) {
    let first = assign(c, "y", exp(c, leaf(c, "x")));
    let second = assign(c, "z", exp(c, leaf(c, "y")));
    ([first], [second])
}

/// A row that fuses up to `file` registers and has no measured costs.
fn row(file: i64) -> FusionRow {
    FusionRow {
        regs_base: 0,
        regs_file: file,
        launch_us: None,
        plane_us_per_byte: None,
        local_over_plane: None,
    }
}

#[test]
fn binding_composes_the_members_into_one_set_of_sinks() {
    let c = rx_ctx_new();
    let (first, second) = kernels(c);
    let members = [
        Member {
            name: "a",
            sinks: &first,
        },
        Member {
            name: "b",
            sinks: &second,
        },
    ];
    let context = unsafe { &mut *c };
    let composed = bind(context, &members, &HashSet::new()).unwrap();
    assert_eq!(composed.sinks.len(), 1, "only the output z stays a sink");
    assert_eq!(
        composed.crossing,
        vec![1],
        "one element crosses the boundary"
    );
    let value = context.nodes[composed.sinks[0] as usize].ops[0];
    let inner = context.nodes[value as usize].ops[0];
    assert_eq!(
        context.nodes[inner as usize].ops.len(),
        1,
        "y is bound to exp(x), not read"
    );
    rx_ctx_free(c);
}

#[test]
fn a_wide_register_file_fuses_every_boundary() {
    let c = rx_ctx_new();
    let (first, second) = kernels(c);
    let members = [
        Member {
            name: "a",
            sinks: &first,
        },
        Member {
            name: "b",
            sinks: &second,
        },
    ];
    let found = cuts(unsafe { &mut *c }, &members, &row(1000), (false, 1 << 16)).unwrap();
    assert!(found.is_empty());
    rx_ctx_free(c);
}

#[test]
fn a_cut_chain_hands_its_value_on_in_a_plane() {
    let c = rx_ctx_new();
    let (first, second) = kernels(c);
    let members = [
        Member {
            name: "a",
            sinks: &first,
        },
        Member {
            name: "b",
            sinks: &second,
        },
    ];
    let parts = pieces(unsafe { &mut *c }, &members, &[1]).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].planes.len(), 1);
    assert_eq!(parts[0].planes[0].name, "hwc_y");
    assert_eq!(parts[0].planes[0].dtype, "real");
    assert!(parts[1].planes.is_empty());
    rx_ctx_free(c);
}
