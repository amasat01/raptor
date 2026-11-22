//! The counting plane a slot states: which plane a one-element count belongs to is a fact of the declaration, never read off a name.
mod common;
use common::*;
use raptor_ir::typed::RxPair;
use raptor_ir::walkreads::rx_walk_counts;
use raptor_support::safe::*;
use std::ffi::{c_char, CString};

/// The text of a pair part.
fn text(part: &raptor_ir::typed::RxPairPart) -> String {
    let bytes: Vec<u8> = part
        .text
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8)
        .collect();
    String::from_utf8(bytes).unwrap()
}

/// The walk over an integer plane `n` that an add updates and a plane `out` that a store updates, the slots declared as `declared`;
/// the status of the canonical call and the walk.
fn walk_of(declared: &str) -> (i32, *mut raptor_ir::record::Ctx, u32) {
    let c = rx_ctx_new();
    let x = mk_of(
        c,
        "Leaf",
        "i32",
        None,
        &[],
        "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))",
    );
    let sinks: Vec<u32> = [("n", "add"), ("out", "store")]
        .iter()
        .map(|(name, op)| {
            let fields = format!("(('name', '{name}'), ('op', '{op}'), ('reduced', None))");
            mk_of(c, "Update", "i32", None, &[x], &fields)
        })
        .collect();
    let declared = CString::new(declared).unwrap();
    let none: *const c_char = std::ptr::null();
    let mut walk = 0;
    let status = rx_canonical(
        c,
        sinks.as_ptr(),
        sinks.len(),
        declared.as_ptr(),
        none,
        none,
        &mut walk,
    );
    (status, c, walk)
}

fn slot(name: &str, counts: &str) -> String {
    format!(
        "(Slot(role='plane_out', name='{name}', ttype=TensorType(shape=(), dtype='i32', tag=None){counts}), )"
    )
}

/// Oracle: the declaration written by hand: the walk lists the one pair it states, the count first.
#[test]
fn a_slot_that_states_what_it_counts_is_listed_with_that_plane() {
    let (status, c, walk) = walk_of(&slot("n", ", counts='out'"));
    assert_eq!(status, RX_OK);
    let mut buf: Vec<RxPair> = Vec::new();
    buf.resize_with(4, || unsafe { std::mem::zeroed::<RxPair>() });
    let mut n = 0;
    assert_eq!(
        unsafe { rx_walk_counts(c, walk, buf.as_mut_ptr(), buf.len(), &mut n) },
        0
    );
    assert_eq!(n, 1);
    assert_eq!(
        (text(&buf[0].first), text(&buf[0].second)),
        ("n".into(), "out".into())
    );
}

/// Oracle: the declaration written by hand: a slot that states nothing counts nothing, whatever its name says.
#[test]
fn a_slot_that_states_no_count_is_not_one() {
    let (status, c, walk) = walk_of(&slot("out_count", ""));
    assert_eq!(status, RX_OK);
    let mut buf: Vec<RxPair> = Vec::new();
    buf.resize_with(4, || unsafe { std::mem::zeroed::<RxPair>() });
    let mut n = 9;
    assert_eq!(
        unsafe { rx_walk_counts(c, walk, buf.as_mut_ptr(), buf.len(), &mut n) },
        0
    );
    assert_eq!(n, 0);
}

/// Oracle: the declaration written by hand: a count of a plane the walk does not bind is refused by name.
#[test]
fn a_count_of_a_plane_the_walk_does_not_bind_is_refused() {
    let (status, _, _) = walk_of(&slot("n", ", counts='absent'"));
    assert_ne!(status, RX_OK);
}
