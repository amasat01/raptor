//! The checks a derivative template passes (closure of the graph, reserved kind names) and the guarantee that substituting into a
//! loop gives it an index and carries of its own. Oracle: tests/data/template_cases.json, recorded by running the previous
//! Python check of the same templates on the same hand-built graphs.
use raptor_ir::build::{rx_rebuild, LoopParts};
use raptor_ir::door::{rx_last_code, rx_last_error, RX_REFUSED};
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::Val;
use rcore::json::{self, J};
use std::collections::HashMap;

/// The recorded answers.
const CASES: &str = include_str!("data/template_cases.json");

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

/// The root of a recorded case, built as the recording built it.
fn root_of(c: &mut Ctx, case: &str) -> u32 {
    let (i0, d0) = (leaf(c, "$in0"), leaf(c, "$dot0"));
    let looped = |c: &mut Ctx, free: u32, sink: bool| {
        let (_, _, lp) = counting_loop(c, free, sink);
        loop_value(c, lp)
    };
    match case {
        "ok_plain" => c.make_op("add", &[i0, d0], &Val::None).unwrap(),
        "ok_loop" => looped(c, i0, false),
        "free_leaf" => {
            let zz = leaf(c, "zz");
            c.make_op("add", &[i0, zz], &Val::None).unwrap()
        }
        "free_leaf_in_loop" => {
            let zz = leaf(c, "zz");
            looped(c, zz, false)
        }
        "index_outside" => {
            let (index, _, _) = counting_loop(c, i0, false);
            let as_float = c
                .make_op("cast", &[index], &Val::Str("f64".into()))
                .unwrap();
            c.make_op("mul", &[i0, as_float], &Val::None).unwrap()
        }
        "carry_outside" => {
            let (_, carry, _) = counting_loop(c, i0, false);
            c.make_op("add", &[i0, carry], &Val::None).unwrap()
        }
        "tape_read" => {
            let (index, _, lp) = counting_loop(c, i0, false);
            let fields = "(('literal', 0), ('slot', 0))";
            c.make("TapeRead", "f64", &[], None, &[lp, index], fields)
                .unwrap()
        }
        "body_sinks" => looped(c, i0, true),
        "edges_outside" => {
            let key = c.make_op("cast", &[i0], &Val::Str("int32".into())).unwrap();
            let params = "(('op', 'sum'), ('value', #0), ('key', #1))";
            c.new_node("ReduceSamples", params, &[i0, key]).unwrap()
        }
        other => panic!("no builder for {other}"),
    }
}

/// The placeholders a template of the recorded cases may read freely.
fn allowed() -> Vec<String> {
    ["$in0", "$dot0", "$out"].map(String::from).to_vec()
}

/// Every recorded case is accepted or refused with the text the previous check gave.
#[test]
fn each_template_is_accepted_or_refused_with_the_recorded_text() {
    let all = json::parse(CASES).unwrap();
    let cases = all.get("template").unwrap().obj();
    assert_eq!(cases.len(), 9);
    for (case, want) in cases {
        let mut c = Ctx::new().unwrap();
        let root = root_of(&mut c, case);
        let got = c.rule_template_check("p", &[root], &allowed());
        match (want.str().unwrap(), got) {
            ("ok", Ok(())) => {}
            (text, Err(Fault::Refused(m))) if text != "ok" => assert_eq!(m, text, "{case}"),
            (text, other) => panic!("{case}: wanted {text:?}, got ok={}", other.is_ok()),
        }
    }
}

/// A name is reserved exactly where the previous check reserved it: the primitive kind, the refinable core kinds, declared kinds.
#[test]
fn reserved_kind_names_are_the_recorded_ones() {
    let all = json::parse(CASES).unwrap();
    let mut c = Ctx::new().unwrap();
    for (name, want) in all.get("reserved").unwrap().obj() {
        assert_eq!(
            c.kind_reserved(name),
            matches!(want, J::Bool(true)),
            "{name}"
        );
    }
    let demo = r#"{"name":"demo","dialect":"demo","kinds":[{"name":"demo.twice","family":"expr","operands":{"min":1,"max":1},"result":["same_as",0]}]}"#;
    assert!(!c.kind_reserved("demo.twice"));
    c.declare_dialect(demo, "()").unwrap();
    assert!(c.kind_reserved("demo.twice"));
}

/// Substituting into a loop gives the rebuilt loop an index and carries of its own: two instances of one template share no
/// placeholder, and each is a closed template whose body reads only its own carry.
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
    let (first, first_loop) = instance(&mut c, in1);
    let (second, second_loop) = instance(&mut c, in1);
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
    for made in [first, second] {
        c.rule_template_check("p", &[made], &allowed()).unwrap();
    }
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
