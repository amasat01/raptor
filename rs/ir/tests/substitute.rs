//! The guarantee that substituting into a loop gives it an index and carries of its own, and the refusal of rewriting a derived
//! loop by substitution.
use raptor_ir::build::{rx_rebuild, LoopParts};
use raptor_ir::door::{rx_last_code, rx_last_error, RX_REFUSED};
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::HashMap;

/// A free leaf named `name`, of the role the templates' placeholders have.
fn leaf(c: &mut Ctx, name: &str) -> u32 {
    let fields = format!("(('kind', 'uniform'), ('role', 'uniform'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))");
    c.make("Leaf", "f64", &[], None, &[], &fields).unwrap()
}

/// A loop `c = c + in0` over three trips, its index and carry, and the free leaf it reads; `sink` gives it a body sink.
fn counting_loop(c: &mut Ctx, free: u32, sink: bool) -> (u32, u32, u32) {
    let index = c
        .make(
            "LoopIndex",
            "int32",
            &[],
            None,
            &[],
            "(('depth', 0), ('name', '_'), ('bounds', (0, 3)))",
        )
        .unwrap();
    let carry = c
        .make(
            "LoopCarry",
            "f64",
            &[],
            None,
            &[],
            "(('depth', 0), ('slot', 0), ('name', 'c'))",
        )
        .unwrap();
    let next = c.make_op("add", &[carry, free], &Val::None).unwrap();
    let sinks = if sink {
        vec![c
            .new_node("Assign", "(('name', 'y'), ('value', #0))", &[carry])
            .unwrap()]
    } else {
        vec![]
    };
    let parts = LoopParts {
        index,
        carries: &[carry],
        inits: &[free],
        nexts: &[next],
        range: (0, 3, 1),
        body_sinks: &sinks,
        exit_cond: None,
        exit_values: &[],
        count: None,
        count_tail: false,
        dynamic: None,
    };
    (index, carry, c.build_loop(&parts).unwrap())
}

/// The nodes a field of node `h` holds (one reference, or a tuple of them).
fn refs_of(c: &Ctx, h: u32, field: &str) -> Result<Vec<u32>, Fault> {
    Ok(match c.field(h, field)? {
        Val::Node(n) => vec![n],
        Val::Tuple(t) | Val::List(t) => t
            .iter()
            .filter_map(|v| if let Val::Node(n) = v { Some(*n) } else { None })
            .collect(),
        _ => vec![],
    })
}

/// The value of slot 0 of a loop.
fn loop_value(c: &mut Ctx, lp: u32) -> u32 {
    c.make("LoopValue", "f64", &[], None, &[lp], "(('literal', 0),)")
        .unwrap()
}

/// Substituting into a loop gives the rebuilt loop an index and carries of its own: two instances of one template share no
/// placeholder, and the new body reads only its own carry.
#[test]
fn substituting_into_a_loop_gives_fresh_index_and_carries() {
    let mut c = Ctx::new().unwrap();
    let (in0, in1) = (leaf(&mut c, "$in0"), leaf(&mut c, "$dot0"));
    let (index, carry, lp) = counting_loop(&mut c, in0, false);
    let template = loop_value(&mut c, lp);
    let instance = |c: &mut Ctx, to: u32| {
        let made = c
            .substitute(&[template], &HashMap::from([(in0, to)]))
            .unwrap()[0];
        let new_loop = c.node(made).unwrap().ops[0];
        (made, new_loop)
    };
    let (_, first_loop) = instance(&mut c, in1);
    let (_, second_loop) = instance(&mut c, in1);
    assert_ne!(first_loop, lp);
    let own = |c: &Ctx, l: u32| {
        let mut v = refs_of(c, l, "index").unwrap();
        v.extend(refs_of(c, l, "carries").unwrap());
        v
    };
    let (old_own, a_own, b_own) = (own(&c, lp), own(&c, first_loop), own(&c, second_loop));
    for (x, y) in [(&old_own, &a_own), (&old_own, &b_own), (&a_own, &b_own)] {
        assert!(x.iter().all(|n| !y.contains(n)), "a placeholder is shared");
    }
    assert_eq!(
        (old_own, refs_of(&c, lp, "carries").unwrap()),
        (vec![index, carry], vec![carry])
    );
    // the new body advances the new carry, never the old one
    let next = refs_of(&c, first_loop, "nexts").unwrap()[0];
    assert_eq!(c.node(next).unwrap().ops, vec![a_own[1], in1]);
}

/// The refusal text and code the door leaves for the last call.
fn last_refusal(c: &Ctx) -> (String, String) {
    let text = |p: *const std::ffi::c_char| {
        // SAFETY: both reads answer the context's own NUL-terminated strings, valid until the next call.
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    };
    // SAFETY: `c` is a live context.
    unsafe { (text(rx_last_error(c)), text(rx_last_code(c))) }
}

#[test]
fn rewriting_a_derived_loop_is_refused_with_a_code_and_a_neighbouring_refusal_has_none() {
    let mut c = Ctx::new().unwrap();
    let free = leaf(&mut c, "in0");
    let (_, _, derived) = counting_loop(&mut c, free, true);
    let operands = c.node(derived).unwrap().ops.clone();
    let mut out = 0;
    // SAFETY: the context is live and the operand list and `out` outlive the calls.
    let status =
        unsafe { rx_rebuild(&mut c, derived, operands.as_ptr(), operands.len(), &mut out) };
    assert_eq!(status, RX_REFUSED);
    assert_eq!(
        last_refusal(&c),
        (
            "a loop that commits once per iteration (a derived loop) cannot be rewritten by substitution".into(),
            "derived_loop_rewrite".into()
        )
    );
    // a refusal of the same call that is not this one carries no code, and the earlier code does not outlive its call
    // SAFETY: as above.
    let status = unsafe {
        rx_rebuild(
            &mut c,
            u32::MAX,
            operands.as_ptr(),
            operands.len(),
            &mut out,
        )
    };
    assert_eq!(status, RX_REFUSED);
    let (text, code) = last_refusal(&c);
    assert!(text.contains("is not in this context"), "{text}");
    assert_eq!(code, "");
}
