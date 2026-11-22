//! The zeroing fact of a walk: which written planes the runtime zeroes before the run, and the argument order it keeps.
mod common;
use common::*;
use raptor_support::safe::*;
use std::ffi::CString;

/// The walk over sinks that write `plane_out` planes through the updates `(name, op)`, a leaf `x` read per sample as the value.
fn walk_of(updates: &[(&str, &str)]) -> (*mut raptor_ir::record::Ctx, u32) {
    let c = rx_ctx_new();
    let x = mk(
        c,
        "Leaf",
        None,
        &[],
        "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))",
    );
    let sinks: Vec<u32> = updates
        .iter()
        .map(|(name, op)| {
            let fields = format!("(('name', '{name}'), ('op', '{op}'), ('reduced', None))");
            mk(c, "Update", None, &[x], &fields)
        })
        .collect();
    let mut walk = 0;
    let none = std::ptr::null();
    let status = rx_canonical(c, sinks.as_ptr(), sinks.len(), none, none, none, &mut walk);
    assert_eq!(status, RX_OK);
    (c, walk)
}

fn zeroed(c: *mut raptor_ir::record::Ctx, walk: u32, name: &str) -> i32 {
    let (role, name) = (
        CString::new("plane_out").unwrap(),
        CString::new(name).unwrap(),
    );
    let mut out = -1;
    assert_eq!(
        rx_walk_zeroed(c, walk, role.as_ptr(), name.as_ptr(), &mut out),
        RX_OK
    );
    out
}

/// Oracle: the rule written by hand: a store leaves its plane as the caller gave it, an add or a bit-or makes it an
/// accumulate target the runtime zeroes; a name the walk does not bind is not zeroed.
#[test]
fn only_a_plane_an_accumulating_update_writes_is_zeroed() {
    let (c, walk) = walk_of(&[("kept", "store"), ("summed", "add"), ("bits", "or")]);
    assert_eq!(zeroed(c, walk, "kept"), 0);
    assert_eq!(zeroed(c, walk, "summed"), 1);
    assert_eq!(zeroed(c, walk, "bits"), 1);
    assert_eq!(zeroed(c, walk, "absent"), 0);
}

/// Oracle: the order the unit lists its slots in, written by hand: the planes the runtime leaves as given come before the planes
/// it zeroes, each group by name, whatever the names sort to.
#[test]
fn the_planes_a_store_writes_are_listed_before_the_planes_it_zeroes() {
    let (c, walk) = walk_of(&[
        ("a_sum", "add"),
        ("z_kept", "store"),
        ("b_sum", "add"),
        ("y_kept", "store"),
    ]);
    let (mut len, mut count) = (0usize, 0usize);
    let mut buf = vec![0 as std::ffi::c_char; 512];
    assert_eq!(
        rx_walk_args(c, walk, buf.as_mut_ptr(), 512, &mut len, &mut count),
        RX_OK
    );
    let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
    let texts: Vec<&str> = bytes
        .split(|&b| b == 0)
        .take(count * 2)
        .map(|t| std::str::from_utf8(t).unwrap())
        .collect();
    let written: Vec<&str> = texts
        .chunks(2)
        .filter(|p| p[0] == "plane_out")
        .map(|p| p[1])
        .collect();
    assert_eq!(written, ["y_kept", "z_kept", "a_sum", "b_sum"]);
}
