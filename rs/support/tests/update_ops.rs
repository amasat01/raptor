//! The operations of the one `Update` element that the typing admits, and the value type each one has.
mod common;
use common::*;
use raptor_support::safe::*;

/// An `Update` of `op` over the operands `ops` with the extra `fields`, made in a context that holds one real leaf and one
/// integer leaf; the operands are named by the leaf they are.
fn update(c: *mut raptor_ir::record::Ctx, dtype: &str, ops: &[u32], op: &str, extra: &str) -> u32 {
    let fields = format!("(('name', 'o'), ('op', '{op}'){extra}, ('reduced', None))");
    mk_of(c, "Update", dtype, None, ops, &fields)
}

fn leaf(c: *mut raptor_ir::record::Ctx, dtype: &str, name: &str, wire: u32) -> u32 {
    let fields = format!(
        "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '{name}'), ('quantity', None), ('wire_index', {wire}), ('sorted', False))"
    );
    mk_of(c, "Leaf", dtype, None, &[], &fields)
}

/// Oracle: the forms section 10 of the design lists, written by hand: every operation on a type it admits builds, and a
/// compare-exchange takes its expected value right after the new one.
#[test]
fn every_operation_builds_on_the_types_it_admits() {
    let c = rx_ctx_new();
    let (real, single, wide, word) = (
        leaf(c, "f64", "x", 0),
        leaf(c, "f32", "s", 1),
        leaf(c, "i64", "w", 2),
        leaf(c, "uint32", "u", 3),
    );
    for op in ["store", "add", "min", "max"] {
        update(c, "f64", &[real], op, "");
        update(c, "f32", &[single], op, "");
        update(c, "i64", &[wide], op, "");
    }
    for op in ["and", "or", "xor"] {
        update(c, "i64", &[wide], op, "");
        update(c, "uint32", &[word], op, "");
    }
    update(c, "f64", &[real], "store", ", ('conflict', 'any')");
    update(c, "f64", &[real], "exchange", ", ('returns_old', True)");
    update(
        c,
        "f64",
        &[real, real],
        "compare_exchange",
        ", ('returns_old', True)",
    );
    update(
        c,
        "i64",
        &[wide],
        "add",
        ", ('summation', 'plain'), ('returns_old', True)",
    );
}

/// Oracle: the forms section 10 lists: a min and a max take the lane order, which fixes their winner; the other operations
/// that are not an add refuse it by name.
#[test]
fn the_lane_order_is_admitted_for_an_add_a_min_and_a_max_only() {
    let c = rx_ctx_new();
    let (real, word) = (leaf(c, "f64", "x", 0), leaf(c, "uint32", "u", 3));
    for op in ["add", "min", "max"] {
        update(c, "f64", &[real], op, ", ('order', 'lane')");
    }
    for (op, dtype, operand) in [("store", "f64", real), ("and", "uint32", word)] {
        let fields =
            format!("(('name', 'o'), ('op', '{op}'), ('order', 'lane'), ('reduced', None))");
        let mut out = u32::MAX;
        let (class, dtype) = (c"Update", std::ffi::CString::new(dtype).unwrap());
        let status = rx_make(
            c,
            class.as_ptr(),
            dtype.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            &operand,
            1,
            fields.as_ptr() as *const _,
            fields.len(),
            &mut out,
        );
        assert_ne!(status, RX_OK, "{op} took the lane order");
        assert!(
            err(c).contains(&format!(
                "is written for an add, a min or a max, not for '{op}'"
            )),
            "{}",
            err(c)
        );
    }
}
