//! The winner plane a slot states: which plane a plane of lanes ranks is a fact of the declaration, never read off a name.
mod common;
use common::*;
use raptor_ir::typed::RxPair;
use raptor_ir::walkreads::rx_walk_winners;
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

/// The walk over a real plane `best` that a minimum in lane order updates, the slots declared as `declared`; the status of the
/// canonical call and the walk.
fn walk_of(declared: &str) -> (i32, *mut raptor_ir::record::Ctx, u32) {
    let c = rx_ctx_new();
    let x = mk_of(
        c,
        "Leaf",
        "f64",
        None,
        &[],
        "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))",
    );
    let fields = "(('name', 'best'), ('op', 'min'), ('order', 'lane'), ('reduced', None))";
    let sink = mk_of(c, "Update", "f64", None, &[x], fields);
    let declared = CString::new(declared).unwrap();
    let none: *const c_char = std::ptr::null();
    let mut walk = 0;
    let status = rx_canonical(c, &sink, 1, declared.as_ptr(), none, none, &mut walk);
    (status, c, walk)
}

/// The declaration of the ranked plane and of a plane named `winner` of the type `dtype`, which states `winner_of`.
fn slots(dtype: &str, winner_of: &str) -> String {
    let real = "TensorType(shape=(), dtype='f64', tag=None)";
    format!(
        "(Slot(role='plane_out', name='best', ttype={real}), \
          Slot(role='plane_out', name='winner', ttype=TensorType(shape=(), dtype='{dtype}', tag=None){winner_of}), )"
    )
}

/// Oracle: the declaration written by hand: the walk lists the one pair it states, the winner first.
#[test]
fn a_slot_that_states_what_it_ranks_is_listed_with_that_plane() {
    let (status, c, walk) = walk_of(&slots("uint32", ", winner_of='best'"));
    assert_eq!(status, RX_OK, "{}", err(c));
    let mut buf: Vec<RxPair> = Vec::new();
    buf.resize_with(4, || unsafe { std::mem::zeroed::<RxPair>() });
    let mut n = 0;
    assert_eq!(
        unsafe { rx_walk_winners(c, walk, buf.as_mut_ptr(), buf.len(), &mut n) },
        0
    );
    assert_eq!(n, 1);
    assert_eq!(
        (text(&buf[0].first), text(&buf[0].second)),
        ("winner".into(), "best".into())
    );
}

/// Oracle: the declaration written by hand: a slot that states no ranking ranks nothing, whatever its name says.
#[test]
fn a_slot_that_states_no_ranking_is_not_a_winner() {
    let (status, c, walk) = walk_of(&slots("uint32", ""));
    assert_eq!(status, RX_OK);
    let mut buf: Vec<RxPair> = Vec::new();
    buf.resize_with(4, || unsafe { std::mem::zeroed::<RxPair>() });
    let mut n = 9;
    assert_eq!(
        unsafe { rx_walk_winners(c, walk, buf.as_mut_ptr(), buf.len(), &mut n) },
        0
    );
    assert_eq!(n, 0);
}

/// Oracle: the declaration written by hand: a winner of a plane the walk does not bind, and a winner that is not an unsigned
/// plane, are refused by name.
#[test]
fn a_winner_of_an_unbound_plane_or_of_another_type_is_refused() {
    let (status, c, _) = walk_of(&slots("uint32", ", winner_of='absent'"));
    assert_ne!(status, RX_OK);
    assert!(err(c).contains("winner_of"), "{}", err(c));
    let (status, c, _) = walk_of(&slots("f64", ", winner_of='best'"));
    assert_ne!(status, RX_OK);
    assert!(err(c).contains("winner_of"), "{}", err(c));
}
