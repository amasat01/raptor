//! Tuple fields of pairs and of bounds. Oracle: the values of `ReduceSamples.kept` and `LoopIndex.bounds` the previous package held
//! for the same nodes (recorded below): `(('net', None),)`, `((None, 4), ('a', None))`, `(('a', 4), ('b', 8))`, `()`, `(0, 3)`, `None`.
use raptor_ir::door::{RX_ABSENT, RX_OK};
use raptor_ir::record::Ctx;
use raptor_ir::typed::{rx_field_ints, rx_field_pairs, RxPair};
use std::ffi::{c_char, CStr, CString};

/// A part as (kind, number, text) with the unused ones empty.
type Shown = (i32, i64, String);

/// A part of a pair, shown.
fn part(p: &raptor_ir::typed::RxPairPart) -> Shown {
    let text = unsafe { CStr::from_ptr(p.text.as_ptr()) }
        .to_str()
        .unwrap()
        .to_string();
    (p.kind, p.number, text)
}

/// The pairs the door answers for the field `kept` of a reduction holding `kept`.
fn kept_pairs(kept: &str) -> Vec<(Shown, Shown)> {
    let mut c = Ctx::new().unwrap();
    let fields = "(('kind', 'vocab_read'), ('role', 'per_sample'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
    let x = c.make("Leaf", "f64", &[], None, &[], fields).unwrap();
    let k = c
        .make("Leaf", "i32", &[], None, &[], &fields.replace("'x'", "'k'"))
        .unwrap();
    let params =
        format!("(('op', 'sum'), ('value', #0), ('key', #1), ('count', #1), ('kept', {kept}))");
    let node = c.new_node("ReduceSamples", &params, &[x, k]).unwrap();
    let ctx: *mut Ctx = &mut c;
    let name = CString::new("kept").unwrap();
    let blank = || RxPair {
        first: raptor_ir::typed::RxPairPart {
            kind: 0,
            number: 0,
            text: [0 as c_char; 64],
        },
        second: raptor_ir::typed::RxPairPart {
            kind: 0,
            number: 0,
            text: [0 as c_char; 64],
        },
    };
    let mut buf: Vec<RxPair> = (0..4).map(|_| blank()).collect();
    let mut n = 0;
    assert_eq!(
        unsafe { rx_field_pairs(ctx, node, name.as_ptr(), buf.as_mut_ptr(), 4, &mut n) },
        RX_OK
    );
    buf[..n]
        .iter()
        .map(|p| (part(&p.first), part(&p.second)))
        .collect()
}

/// A part that is absent is its kind, and a present one carries its value; the empty tuple has no pairs.
#[test]
fn the_pairs_of_a_field_keep_absent_parts_absent() {
    let none = || (0, 0, String::new());
    let text = |t: &str| (2, 0, t.to_string());
    let number = |v: i64| (1, v, String::new());
    assert_eq!(kept_pairs("(('net', None),)"), vec![(text("net"), none())]);
    assert_eq!(
        kept_pairs("((None, 4), ('a', None))"),
        vec![(none(), number(4)), (text("a"), none())]
    );
    assert_eq!(
        kept_pairs("(('a', 4), ('b', 8))"),
        vec![(text("a"), number(4)), (text("b"), number(8))]
    );
    assert!(kept_pairs("()").is_empty());
}

/// The bounds of a loop index are a pair of integers or None, which the integer reader already answers.
#[test]
fn the_bounds_of_a_loop_index_read_as_integers_or_absent() {
    let mut c = Ctx::new().unwrap();
    let mut bounds = |text: &str| {
        let params = format!("(('depth', 0), ('name', '_'), ('bounds', {text}))");
        let index = c
            .make("LoopIndex", "int32", &[], None, &[], &params)
            .unwrap();
        let (mut buf, mut n) = ([0_i64; 4], 0);
        let name = CString::new("bounds").unwrap();
        let status =
            unsafe { rx_field_ints(&mut c, index, name.as_ptr(), buf.as_mut_ptr(), 4, &mut n) };
        (status, buf[..n].to_vec())
    };
    assert_eq!(bounds("(0, 3)"), (RX_OK, vec![0, 3]));
    assert_eq!(bounds("None").0, RX_ABSENT);
}
