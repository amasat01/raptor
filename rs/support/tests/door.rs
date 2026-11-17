//! The door's C functions only.
use raptor_ir::record::Ctx;
use raptor_support::safe::*;
use rcore::{walk, Rec};
use std::ffi::{CStr, CString};

fn mk(
    c: *mut Ctx,
    cls: &str,
    dtype: &str,
    shape: &[i64],
    tag: Option<&str>,
    ops: &[u32],
    fields: &str,
) -> (i32, u32) {
    let (cls, dtype, tag) = (
        CString::new(cls).unwrap(),
        CString::new(dtype).unwrap(),
        tag.map(|t| CString::new(t).unwrap()),
    );
    let mut out = u32::MAX;
    let st = rx_make(
        c,
        cls.as_ptr(),
        dtype.as_ptr(),
        shape.as_ptr(),
        shape.len(),
        tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
        ops.as_ptr(),
        ops.len(),
        fields.as_ptr() as *const _,
        fields.len(),
        &mut out,
    );
    (st, out)
}
fn ok(c: *mut Ctx, cls: &str, ops: &[u32], fields: &str) -> u32 {
    let (st, h) = mk(c, cls, "f64", &[], None, ops, fields);
    assert_eq!(st, RX_OK, "{cls}: {}", err(c));
    h
}
fn err(c: *mut Ctx) -> String {
    unsafe { CStr::from_ptr(rx_last_error(c)) }
        .to_str()
        .unwrap()
        .to_string()
}
fn text(
    c: *mut Ctx,
    f: impl Fn(*mut Ctx, *mut std::ffi::c_char, usize, *mut usize) -> i32,
) -> String {
    let mut len = 0usize;
    assert_eq!(f(c, std::ptr::null_mut(), 0, &mut len), RX_TRUNC);
    let mut buf = vec![1 as std::ffi::c_char; len + 1];
    assert_eq!(f(c, buf.as_mut_ptr(), len + 1, &mut len), RX_OK);
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_str()
        .unwrap()
        .to_string()
}
const LEAF: &str = "(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '{}'), ('quantity', None), ('wire_index', 0), ('sorted', False))";

fn expression(c: *mut Ctx) -> [u32; 6] {
    let a = ok(c, "Leaf", &[], &LEAF.replace("{}", "a"));
    let b = ok(c, "Leaf", &[], &LEAF.replace("{}", "b"));
    let s = ok(c, "Op", &[a, b], "(('kind', 'add'), ('literal', None))");
    let k = ok(c, "Const", &[], "(('literal', 2.0),)");
    let e = ok(c, "Op", &[s, k], "(('kind', 'add'), ('literal', None))");
    let y = ok(c, "Assign", &[e], "(('name', 'y'),)");
    [a, b, s, k, e, y]
}

#[test]
fn probe_expression_reads_back() {
    let c = rx_ctx_new();
    let [a, b, s, k, e, y] = expression(c);
    let cls = ["Leaf", "Leaf", "Op", "Const", "Op", "Assign"];
    for (h, want) in [a, b, s, k, e, y].iter().zip(cls) {
        assert_eq!(
            text(c, |c, buf, cap, len| rx_class(c, *h, buf, cap, len)),
            want
        );
        assert_eq!(
            text(c, |c, buf, cap, len| rx_ttype(c, *h, buf, cap, len)),
            "TensorType(shape=(), dtype='f64', tag=None)"
        );
    }
    let ops = |h: u32| {
        let mut v = [0u32; 4];
        let mut n = 0;
        assert_eq!(rx_operands(c, h, v.as_mut_ptr(), 4, &mut n), RX_OK);
        v[..n].to_vec()
    };
    assert_eq!(
        (ops(a), ops(s), ops(k), ops(e), ops(y)),
        (vec![], vec![a, b], vec![], vec![s, k], vec![e])
    );
    let mut n = 0;
    let mut small = [0u32; 1];
    assert_eq!(rx_operands(c, s, small.as_mut_ptr(), 1, &mut n), RX_TRUNC);
    assert_eq!(n, 2);
    let field = |h: u32, f: &str| {
        let f = CString::new(f).unwrap();
        text(c, |c, buf, cap, len| {
            rx_field(c, h, f.as_ptr(), buf, cap, len)
        })
    };
    assert_eq!(
        (
            field(a, "name").as_str(),
            field(a, "role").as_str(),
            field(a, "wire_index").as_str(),
            field(a, "sorted").as_str()
        ),
        ("'a'", "'vec_in'", "0", "False")
    );
    assert_eq!(
        (
            field(k, "literal").as_str(),
            field(s, "kind").as_str(),
            field(s, "literal").as_str()
        ),
        ("2.0", "'add'", "None")
    );
    assert_eq!(
        (
            field(y, "name").as_str(),
            field(y, "role").as_str(),
            field(y, "kind").as_str()
        ),
        ("'y'", "'mutable'", "'assign'")
    );
    assert_eq!(field(k, "kind"), "'constant'");
    rx_ctx_free(c);
}

#[test]
fn one_record_per_call() {
    let c = rx_ctx_new();
    let f = "(('literal', 2.0),)";
    assert_ne!(ok(c, "Const", &[], f), ok(c, "Const", &[], f));
    rx_ctx_free(c);
}

#[test]
fn key_lines_equal_today() {
    let c = rx_ctx_new();
    let hs = expression(c);
    let recs: Vec<Rec> = hs
        .iter()
        .map(|&h| unsafe { &*c }.rec_of(h).unwrap())
        .collect();
    let (order, pre) = walk(&recs, &[hs[5]]).unwrap();
    assert_eq!(order.len(), 6);
    assert_eq!(pre, "0|('leaf', 'vec_in', 'a')\n1|('leaf', 'vec_in', 'b')\n2|('add', 'f64', (), None, (0, 1), 'None')\n\
                     3|('constant', 'f64', (), None, (), '2.0')\n4|('add', 'f64', (), None, (2, 3), 'None')\n5|('assign', 'f64', (), None, (4,), ('mutable', 'y'))\n");
    rx_ctx_free(c);
}

#[test]
fn refusals_and_stubs() {
    let c = rx_ctx_new();
    let x = ok(c, "Leaf", &[], &LEAF.replace("{}", "x"));
    let cases: [(&str, &str, &[u32], &str, i32); 8] = [
        ("Nope", "f64", &[], "()", RX_REFUSED),
        ("Const", "f64", &[9], "(('literal', 1),)", RX_REFUSED),
        ("Const", "f64", &[], "(('literl', 1),)", RX_REFUSED),
        ("Const", "f64", &[], "()", RX_REFUSED),
        ("Const", "f64", &[], "(('literal', 1", RX_REFUSED),
        ("Const", "f77", &[], "(('literal', 1),)", RX_BADARG),
        (
            "Assign",
            "f64",
            &[x],
            "(('name', 'y'), ('role', 'out'))",
            RX_REFUSED,
        ),
        ("Const", "f64", &[], "(('literal', #5),)", RX_REFUSED),
    ];
    for (cls, dt, ops, f, want) in cases {
        assert_eq!(mk(c, cls, dt, &[], None, ops, f).0, want, "{cls} {f}");
    }
    assert_eq!(
        mk(c, "Const", "f64", &[-1], None, &[], "(('literal', 1),)").0,
        RX_BADARG
    );
    assert_eq!(rx_class(c, 99, std::ptr::null_mut(), 0, &mut 0), RX_REFUSED);
    assert_eq!(
        rx_declare_dialect(c, std::ptr::null(), 0, std::ptr::null()),
        RX_BADARG
    );
    assert_eq!(rx_ctx_free_null(), ());
    assert_eq!(
        rx_make(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            std::ptr::null_mut()
        ),
        RX_BADARG
    );
    rx_ctx_free(c);
}
fn rx_ctx_free_null() {
    rx_ctx_free(std::ptr::null_mut())
}

/// (class, operand count, fields) of a minimal valid record of each core class.
#[test]
fn every_core_class() {
    let c = rx_ctx_new();
    let (l, i) = (
        ok(c, "Leaf", &[], &LEAF.replace("{}", "p")),
        ok(c, "Const", &[], "(('literal', 0),)"),
    );
    let ix = ok(
        c,
        "LoopIndex",
        &[],
        "(('depth', 0), ('name', 'i'), ('bounds', None))",
    );
    let cr = ok(
        c,
        "LoopCarry",
        &[],
        "(('depth', 0), ('slot', 0), ('name', 'c'))",
    );
    let sink = ok(c, "Assign", &[i], "(('name', 'o'),)");
    assert_eq!((l, i, ix, cr, sink), (0, 1, 2, 3, 4));
    let si = mk(c, "Const", "i32", &[], None, &[], "(('literal', 0),)").1;
    let cb = mk(c, "Const", "bool", &[], None, &[], "(('literal', True),)").1;
    let loop_f = "(('index', #2), ('carries', (#3,)), ('nexts', (#0,)), ('start', 0), ('stop', 2), ('step', 1), ('trip', 2), ('carry_count', 1), ('body_sinks', (#4,)), \
                  ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), ('count_tail', False), ('dynamic', None), ('_free_end', 1))";
    let rows: Vec<(&str, Vec<u32>, String)> = vec![
        ("Node", vec![], "()".into()),
        ("Leaf", vec![], LEAF.replace("{}", "q")),
        ("Const", vec![], "(('literal', 1),)".into()),
        (
            "RoleConst",
            vec![],
            "(('literal', 1), ('label', 'n'))".into(),
        ),
        ("SampleIndex", vec![], "()".into()),
        ("PlaneExtent", vec![l], "()".into()),
        ("SamplePosition", vec![], "()".into()),
        ("Op", vec![i], "(('kind', 'neg'), ('literal', None))".into()),
        ("At", vec![l, i], "()".into()),
        (
            "Primitive",
            vec![i],
            "(('name', 'f'), ('input_count', 0))".into(),
        ),
        ("Select", vec![i, i, i], "()".into()),
        ("Dispatch", vec![si, i, i], "()".into()),
        (
            "Sink",
            vec![i],
            "(('kind', 'assign'), ('role', 'mutable'), ('name', 'z'))".into(),
        ),
        ("Assign", vec![i], "(('name', 'a'),)".into()),
        ("WideWrite", vec![i], "(('name', 'w'),)".into()),
        (
            "AccumWrite",
            vec![i],
            "(('name', 'acc'), ('contract', 'auto'), ('reduced', None))".into(),
        ),
        (
            "ReduceSamples",
            vec![i],
            "(('op', 'sum'), ('key', None), ('count', None), ('contract', 'auto'), ('kept', ()))"
                .into(),
        ),
        (
            "SampleSum",
            vec![i],
            "(('name', 's'), ('op', 'sum'), ('cell', 's'), ('columns', 1))".into(),
        ),
        (
            "LoopIndex",
            vec![],
            "(('depth', 0), ('name', 'j'), ('bounds', None))".into(),
        ),
        (
            "LoopCarry",
            vec![],
            "(('depth', 0), ('slot', 0), ('name', 'd'))".into(),
        ),
        ("Loop", vec![i], loop_f.into()),
    ];
    assert_eq!(rows.len(), 21);
    let mut made = vec![];
    for (cls, ops, f) in &rows {
        made.push((cls.to_string(), ok(c, cls, ops, f)))
    }
    let lp = made.last().unwrap().1;
    made.push((
        "TapeWrite".into(),
        mk(
            c,
            "TapeWrite",
            "f64",
            &[2],
            None,
            &[si, i, i],
            "(('name', 't'), ('width', 1))",
        )
        .1,
    ));
    made.push((
        "Finish".into(),
        mk(c, "Finish", "bool", &[], None, &[cb], "(('name', 'f'),)").1,
    ));
    let lp2 = mk(
        c,
        "Loop",
        "f64",
        &[],
        None,
        &[i],
        &loop_f.replace(
            "('exit_cond', None), ('exit_values', ())",
            &format!("('exit_cond', #{cb}), ('exit_values', (#{i},))"),
        ),
    )
    .1;
    made.push((
        "LoopCount".into(),
        mk(c, "LoopCount", "i32", &[], None, &[lp2], "()").1,
    ));
    for (cls, ops, f) in [
        ("LoopValue", vec![lp], "(('literal', 0),)"),
        ("TapeRead", vec![lp, i], "(('literal', 0), ('slot', 0))"),
    ] {
        made.push((cls.to_string(), ok(c, cls, &ops, f)));
    }
    assert_eq!(made.len(), 26);
    for (cls, h) in &made {
        assert_eq!(
            text(c, |c, buf, cap, len| rx_class(c, *h, buf, cap, len)),
            *cls
        );
        assert!(unsafe { &*c }.rec_of(*h).is_ok(), "{cls}");
    }
    rx_ctx_free(c);
}

const DEMO: &str = r#"{"name":"demo","dialect":"demo","kinds":[
  {"name":"demo.twice","family":"expr","operands":{"min":1,"max":1},"result":["same_as",0],"fields":[{"name":"k","type":"int","default":2}]},
  {"name":"demo.loud","family":"sink","refines":"assign","operands":{"min":1,"max":1}}]}"#;

#[test]
fn dialect_declare_and_make() {
    let c = rx_ctx_new();
    let decl = |c: *mut Ctx, m: &str| {
        rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), c"()".as_ptr())
    };
    assert_eq!(decl(c, DEMO), RX_OK, "{}", err(c));
    assert_eq!(
        decl(c, DEMO),
        RX_OK,
        "an equal declaration is accepted again"
    );
    assert_eq!(
        decl(c, &DEMO.replace("\"default\":2", "\"default\":3")),
        RX_REFUSED
    );
    let x = ok(c, "Const", &[], "(('literal', 1.0),)");
    let t = ok(c, "DemoTwice", &[x], "()");
    assert_eq!(
        text(c, |c, b, cap, l| rx_class(c, t, b, cap, l)),
        "DemoTwice"
    );
    let f = CString::new("k").unwrap();
    assert_eq!(
        text(c, |c, b, cap, l| rx_field(c, t, f.as_ptr(), b, cap, l)),
        "2"
    );
    assert_eq!(
        text(c, |c, b, cap, l| rx_field(
            c,
            t,
            c"kind".as_ptr(),
            b,
            cap,
            l
        )),
        "'demo.twice'"
    );
    assert_eq!(
        mk(c, "DemoTwice", "f64", &[], None, &[x, x], "()").0,
        RX_REFUSED
    );
    assert_eq!(
        mk(c, "DemoTwice", "f64", &[], None, &[x], "(('k', 'a'),)").0,
        RX_REFUSED
    );
    assert_eq!(
        mk(c, "DemoTwice", "i32", &[], None, &[x], "()").0,
        RX_REFUSED
    );
    ok(c, "DemoLoud", &[x], "(('name', 'o'),)");
    let c2 = rx_ctx_new();
    assert_eq!(
        mk(c2, "DemoTwice", "f64", &[], None, &[], "()").0,
        RX_REFUSED,
        "dialect classes are per context"
    );
    rx_ctx_free(c2);
    rx_ctx_free(c);
}

#[test]
fn dialect_feathers_and_spelling() {
    let c = rx_ctx_new();
    let decl = |c: *mut Ctx, m: &str| {
        rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), c"()".as_ptr())
    };
    let row = |extra: &str| {
        format!(
            r#"{{"name":"demo","kinds":[{{"name":"demo.k","family":"expr","operands":{{"min":1,"max":1}},"result":["same_as",0]{extra}}}]}}"#
        )
    };
    assert_eq!(decl(c, &row(r#","feathers":["nope"]"#)), RX_REFUSED);
    assert_eq!(err(c), "kind 'demo.k': unknown feather 'nope'");
    assert_eq!(
        decl(
            c,
            &row(r#","spelling":{"dialect":"demo","name":"k","args":["ref3"]}"#)
        ),
        RX_REFUSED
    );
    assert_eq!(
        err(c),
        "kind 'demo.k': spelling args ('ref3',) are not node|target|ref<i> within the operands"
    );
    assert_eq!(
        decl(
            c,
            &row(r#","spelling":{"dialect":"demo","name":"k","args":["ref0","node"]}"#)
        ),
        RX_OK,
        "{}",
        err(c)
    );
    let c2 = rx_ctx_new();
    assert_eq!(
        decl(
            c2,
            &row(r#","feathers":["reads_by_element","reads_by_element"]"#)
        ),
        RX_OK,
        "{}",
        err(c2)
    );
    let x = ok(c2, "Const", &[], "(('literal', 1.0),)");
    let k = ok(c2, "DemoK", &[x], "()");
    assert_eq!(
        text(c2, |c, b, cap, l| rx_field(
            c,
            k,
            c"feathers".as_ptr(),
            b,
            cap,
            l
        )),
        "('reads_by_element',)"
    );
    rx_ctx_free(c2);
    rx_ctx_free(c);
}

#[test]
fn rule_before_constructor_and_evaluator_failures() {
    let c = rx_ctx_new();
    let x = ok(c, "Const", &[], "(('literal', 1.0),)");
    // the operand-count and result-type rule of the kind come first, whatever the constructor would say
    assert_eq!(
        mk(c, "Select", "f64", &[], None, &[x, x], "()").0,
        RX_REFUSED
    );
    assert_eq!(err(c), "select: takes 3..3 operand(s), got 2");
    let (_, b) = mk(c, "Const", "bool", &[], None, &[], "(('literal', True),)");
    assert_eq!(
        mk(c, "Finish", "f64", &[], None, &[b], "(('name', 'f'),)").0,
        RX_REFUSED
    );
    assert_eq!(
        err(c),
        "finish: type f64 [] is not what its result rule derives, bool []"
    );
    // a refused make leaves no node behind
    assert_eq!(
        mk(c, "Const", "f64", &[], None, &[], "(('literal', 2.0),)").1,
        2
    );
    // a constructor that cannot read the state is not a refusal text of its own
    let c3 = rx_ctx_new();
    let l = ok(c3, "Leaf", &[], &LEAF.replace("{}", "p"));
    let i = ok(c3, "Const", &[], "(('literal', 0),)");
    ok(
        c3,
        "LoopIndex",
        &[],
        "(('depth', 0), ('name', 'i'), ('bounds', None))",
    );
    ok(
        c3,
        "LoopCarry",
        &[],
        "(('depth', 0), ('slot', 0), ('name', 'c'))",
    );
    ok(c3, "Assign", &[i], "(('name', 'o'),)");
    let lf = "(('index', #2), ('carries', (#3,)), ('nexts', (#0,)), ('start', 0), ('stop', 2), ('step', 1), ('trip', 2), ('carry_count', 1), ('body_sinks', (#4,)), \
              ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), ('count_tail', False), ('dynamic', None), ('_free_end', 1))";
    ok(c3, "Loop", &[i], lf);
    let _ = l;
    assert_eq!(
        mk(
            c3,
            "Loop",
            "f64",
            &[],
            None,
            &[i],
            &lf.replace("('carries', (#3,))", "('carries', 3)")
        )
        .0,
        RX_REFUSED
    );
    assert!(
        err(c3).starts_with("Loop: its state cannot be read by its constructor ("),
        "{}",
        err(c3)
    );
    rx_ctx_free(c3);
    // a dialect node meets the same base refusals: the typed door refuses a bad dtype or extent before any program runs
    let m = DEMO;
    assert_eq!(
        rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), c"()".as_ptr()),
        RX_OK
    );
    for cls in ["Const", "DemoTwice"] {
        let (ops, f): (&[u32], &str) = if cls == "Const" {
            (&[], "(('literal', 1.0),)")
        } else {
            (&[x], "()")
        };
        assert_eq!(mk(c, cls, "f77", &[], None, ops, f).0, RX_BADARG, "{cls}");
        assert_eq!(mk(c, cls, "f64", &[-2], None, ops, f).0, RX_BADARG, "{cls}");
    }
    rx_ctx_free(c);
}

#[test]
fn parameter_form_and_typed_op() {
    let c = rx_ctx_new();
    let tt = "TensorType(shape=(), dtype='f64', tag=None)";
    let new = |c: *mut Ctx, cls: &str, p: &str, refs: &[u32]| {
        let cls = CString::new(cls).unwrap();
        let mut out = u32::MAX;
        (
            rx_new(
                c,
                cls.as_ptr(),
                p.as_ptr() as *const _,
                p.len(),
                refs.as_ptr(),
                refs.len(),
                &mut out,
            ),
            out,
        )
    };
    let (st, a) = new(
        c,
        "Leaf",
        &format!("(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', 'a'), ('ttype', {tt}))"),
        &[],
    );
    assert_eq!(st, RX_OK, "{}", err(c));
    let (_, k) = new(
        c,
        "Const",
        &format!("(('literal', 2.0), ('ttype', {tt}))"),
        &[],
    );
    // a reference is an index into refs; the type comes from the declared program
    let (st, s) = new(
        c,
        "Op",
        &format!("(('kind', 'add'), ('operands', (#0, #1)), ('ttype', {tt}))"),
        &[a, k],
    );
    assert_eq!(st, RX_OK, "{}", err(c));
    let (st, y) = new(c, "Assign", "(('name', 'y'), ('value', #0))", &[s]);
    assert_eq!(st, RX_OK, "{}", err(c));
    assert_eq!(
        text(c, |c, b, cap, l| rx_field(
            c,
            y,
            c"role".as_ptr(),
            b,
            cap,
            l
        )),
        "'mutable'"
    );
    assert_eq!(text(c, |c, b, cap, l| rx_ttype(c, y, b, cap, l)), tt);
    assert_eq!(
        new(c, "Assign", "(('name', 'y'), ('value', #3))", &[s]).0,
        RX_REFUSED
    );
    assert_eq!(
        err(c),
        "node reference #3 is outside the 1 reference(s) given"
    );
    assert_eq!(new(c, "Assign", "(('name', 'y'))", &[]).0, RX_REFUSED);
    assert_eq!(
        new(
            c,
            "Assign",
            "(('name', 'y'), ('value', #0), ('bogus', 1))",
            &[s]
        )
        .0,
        RX_REFUSED
    );
    assert_eq!(new(c, "Nope", "()", &[]).0, RX_REFUSED);
    // the typed op infers the type; a refusal appends nothing
    let before = unsafe { &*c }.nodes.len();
    let mut out = 0u32;
    let lit = "";
    assert_eq!(
        rx_make_op(
            c,
            c"mul".as_ptr(),
            [a, k].as_ptr(),
            2,
            lit.as_ptr() as *const _,
            0,
            &mut out
        ),
        RX_OK,
        "{}",
        err(c)
    );
    assert_eq!(text(c, |c, b, cap, l| rx_ttype(c, out, b, cap, l)), tt);
    assert_eq!(
        rx_make_op(
            c,
            c"bogus".as_ptr(),
            [a].as_ptr(),
            1,
            lit.as_ptr() as *const _,
            0,
            &mut out
        ),
        RX_REFUSED
    );
    assert!(err(c).starts_with("unknown op kind 'bogus'"));
    assert_eq!(unsafe { &*c }.nodes.len(), before + 1);
    // the same op built without a literal, and one that carries the literal of another node
    let mut plain = 0u32;
    assert_eq!(
        rx_make_op_like(
            c,
            c"mul".as_ptr(),
            [a, k].as_ptr(),
            2,
            std::ptr::null(),
            &mut plain
        ),
        RX_OK,
        "{}",
        err(c)
    );
    assert_eq!(text(c, |c, b, cap, l| rx_ttype(c, plain, b, cap, l)), tt);
    let like = ok(c, "Op", &[a, k], "(('kind', 'mul'), ('literal', (1, 2)))");
    let mut carried = 0u32;
    assert_eq!(
        rx_make_op_like(c, c"mul".as_ptr(), [a, k].as_ptr(), 2, &like, &mut carried),
        RX_OK,
        "{}",
        err(c)
    );
    assert_eq!(
        text(c, |c, b, cap, l| rx_field(
            c,
            carried,
            c"literal".as_ptr(),
            b,
            cap,
            l
        )),
        "(1, 2)"
    );
    assert_eq!(
        rx_make_op_like(
            c,
            c"bogus".as_ptr(),
            [a].as_ptr(),
            1,
            std::ptr::null(),
            &mut plain
        ),
        RX_REFUSED
    );
    rx_ctx_free(c);
}

#[test]
fn positions_bodies_field2_and_set_error() {
    let c = rx_ctx_new();
    let leaf = |n: &str| LEAF.replace("{}", n);
    let a = ok(c, "Leaf", &[], &leaf("a"));
    // two equal but separate constants: one representative in the canonical order, both answer its position
    let k1 = ok(c, "Const", &[], "(('literal', 2.0),)");
    let k2 = ok(c, "Const", &[], "(('literal', 2.0),)");
    let s = ok(c, "Op", &[a, k1], "(('kind', 'add'), ('literal', None))");
    let e = ok(c, "Op", &[s, k2], "(('kind', 'add'), ('literal', None))");
    let y = ok(c, "Assign", &[e], "(('name', 'y'),)");
    let stray = ok(c, "Const", &[], "(('literal', 9.0),)");
    let mut w = 0;
    assert_eq!(
        rx_canonical(
            c,
            &y,
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &mut w
        ),
        RX_OK,
        "{}",
        err(c)
    );
    let pos = |h: u32| {
        let mut p = 99u32;
        (rx_walk_position(c, w, h, &mut p), p)
    };
    let (p1, p2) = (pos(k1), pos(k2));
    assert_eq!((p1.0, p2.0, p1.1), (RX_OK, RX_OK, p2.1));
    assert_eq!(pos(a), (RX_OK, 0));
    assert_eq!(pos(stray).0, RX_REFUSED);
    assert!(
        err(c).contains("not reached by the walk's traversal") && err(c).contains("loop body"),
        "{}",
        err(c)
    );
    assert_eq!(pos(999).0, RX_REFUSED);
    assert_eq!(rx_walk_position(c, 7, a, &mut 0), RX_REFUSED);
    // loop bodies
    let (i, ix) = (
        ok(c, "Const", &[], "(('literal', 0),)"),
        ok(
            c,
            "LoopIndex",
            &[],
            "(('depth', 0), ('name', 'i'), ('bounds', None))",
        ),
    );
    let cr = ok(
        c,
        "LoopCarry",
        &[],
        "(('depth', 0), ('slot', 0), ('name', 'c'))",
    );
    let inner = ok(c, "Op", &[cr, cr], "(('kind', 'add'), ('literal', None))");
    let bs = ok(c, "Assign", &[inner], "(('name', 'o'),)");
    let lf = format!("(('index', #{ix}), ('carries', (#{cr},)), ('nexts', (#{inner},)), ('start', 0), ('stop', 2), ('step', 1), ('trip', 2), ('carry_count', 1), ('body_sinks', (#{bs},)), \
                      ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), ('count_tail', False), ('dynamic', None), ('_free_end', 1))");
    let lp = ok(c, "Loop", &[i], &lf);
    let list = |f: fn(*mut Ctx, u32, *mut u32, usize, *mut usize) -> i32, h: u32| {
        let (mut v, mut n) = ([0u32; 8], 0usize);
        (f(c, h, v.as_mut_ptr(), 8, &mut n), v[..n].to_vec())
    };
    assert_eq!(list(rx_body_nodes, lp), (RX_OK, vec![inner]));
    assert_eq!(list(rx_body_roots, lp), (RX_OK, vec![inner, inner]));
    let mut small = [0u32; 1];
    let mut n = 0usize;
    assert_eq!(
        rx_body_roots(c, lp, small.as_mut_ptr(), 1, &mut n),
        RX_TRUNC
    );
    assert_eq!(n, 2);
    assert_eq!(list(rx_body_nodes, a).0, RX_REFUSED);
    assert!(err(c).contains("not a Loop"));
    // a field with references: indices in the text, handles beside it
    let (mut kt, mut len, mut refs, mut nr) =
        ([0 as std::ffi::c_char; 128], 0usize, [0u32; 4], 0usize);
    assert_eq!(
        rx_field2(
            c,
            lp,
            c"carries".as_ptr(),
            kt.as_mut_ptr(),
            128,
            &mut len,
            refs.as_mut_ptr(),
            4,
            &mut nr
        ),
        RX_OK,
        "{}",
        err(c)
    );
    assert_eq!(
        (
            unsafe { CStr::from_ptr(kt.as_ptr()) }.to_str().unwrap(),
            &refs[..nr]
        ),
        ("(#0,)", &[cr][..])
    );
    assert_eq!(
        rx_field2(
            c,
            lp,
            c"start".as_ptr(),
            kt.as_mut_ptr(),
            128,
            &mut len,
            refs.as_mut_ptr(),
            4,
            &mut nr
        ),
        RX_OK
    );
    assert_eq!(
        (unsafe { CStr::from_ptr(kt.as_ptr()) }.to_str().unwrap(), nr),
        ("0", 0)
    );
    assert_eq!(
        rx_field2(
            c,
            lp,
            c"carries".as_ptr(),
            kt.as_mut_ptr(),
            128,
            &mut len,
            refs.as_mut_ptr(),
            0,
            &mut nr
        ),
        RX_TRUNC
    );
    assert_eq!(nr, 1);
    assert_eq!(
        rx_field2(
            c,
            lp,
            c"nope".as_ptr(),
            kt.as_mut_ptr(),
            128,
            &mut len,
            refs.as_mut_ptr(),
            4,
            &mut nr
        ),
        RX_REFUSED
    );
    // a backend reports its own refusal through the context
    assert_eq!(
        rx_set_error(c, c"writer: no such target".as_ptr(), 22),
        RX_OK
    );
    assert_eq!(err(c), "writer: no such target");
    assert_eq!(
        rx_set_error(std::ptr::null_mut(), c"x".as_ptr(), 1),
        RX_BADARG
    );
    rx_ctx_free(c);
}

#[test]
fn constants_by_name() {
    let c = rx_ctx_new();
    let get = |n: &str| {
        let n = CString::new(n).unwrap();
        let (mut v, mut len) = ([0 as std::ffi::c_char; 4096], 0usize);
        let st = rx_const(c, n.as_ptr(), v.as_mut_ptr(), 4096, &mut len);
        (
            st,
            unsafe { CStr::from_ptr(v.as_ptr()) }
                .to_str()
                .unwrap()
                .to_string(),
        )
    };
    // every table the writers carry a copy of today
    assert_eq!(
        get("INT_PLANE_DTYPES"),
        (RX_OK, r#"["int32","uint32"]"#.to_string())
    );
    assert_eq!(get("PARAM_CELL"), (RX_OK, r#""param_cell""#.to_string()));
    assert_eq!(get("KEYED_AUTO"), (RX_OK, r#""keyed_auto""#.to_string()));
    assert_eq!(get("ir.BODY_KEY_HEX").0, RX_OK);
    for n in [
        "AT_ROLES",
        "CONTRACTS",
        "SCATTER_CONTRACTS",
        "ROLE_ORDER",
        "ELEMENTWISE_BINARY",
        "UNARY_MATH",
        "ELEMENTWISE_TERNARY",
        "COMPARE",
        "LOGICAL",
        "INT_BINARY",
        "INT_UNARY",
        "CAST_TARGETS",
        "LINALG_FACTORISATIONS",
        "LINALG_CONSUMERS",
        "LINALG_PROJECTIONS",
        "LSTSQ_OPS",
        "QR_OPS",
        "FACTOR_TAG_PAIRS",
        "RAY_OPS",
        "KIND_FEATHERS",
        "NAMESPACE_SEP",
    ] {
        let (st, t) = get(n);
        assert!(st == RX_OK && rcore::json::parse(&t).is_ok(), "{n}");
    }
    assert!(get("AT_ROLES").1.contains("lookup"));
    assert_eq!(get("NOPE").0, RX_REFUSED);
    assert_eq!(err(c), "raptor owns no constant named \"NOPE\"");
    let mut len = 0;
    assert_eq!(
        rx_const(c, c"CONTRACTS".as_ptr(), std::ptr::null_mut(), 0, &mut len),
        RX_TRUNC
    );
    assert_eq!(len, get("CONTRACTS").1.len());
    println!("names answered: {}", raptor_ir::consts::count());
    constant_names(c);
    rx_ctx_free(c);
}

/// A list constant answers its texts one by one, a string constant its one text; an unknown name is
/// refused.
fn constant_names(c: *mut Ctx) {
    let names = |n: &str| {
        let n = CString::new(n).unwrap();
        let (mut len, mut count) = (0usize, 0usize);
        let st = rx_const_names(c, n.as_ptr(), std::ptr::null_mut(), 0, &mut len, &mut count);
        if st != RX_TRUNC && st != RX_OK {
            return (st, vec![]);
        }
        let mut buf = vec![0 as std::ffi::c_char; len + 1];
        let st = rx_const_names(
            c,
            n.as_ptr(),
            buf.as_mut_ptr(),
            len + 1,
            &mut len,
            &mut count,
        );
        let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
        let joined = String::from_utf8(bytes).unwrap();
        let items: Vec<String> = joined.split_terminator('\0').map(String::from).collect();
        assert_eq!(items.len(), count);
        (st, items)
    };
    assert_eq!(
        names("INT_PLANE_DTYPES"),
        (RX_OK, vec!["int32".to_string(), "uint32".to_string()])
    );
    assert_eq!(names("PARAM_CELL"), (RX_OK, vec!["param_cell".to_string()]));
    assert_eq!(names("NOPE").0, RX_REFUSED);
    assert_eq!(err(c), "raptor owns no list of names under \"NOPE\"");
}

/// A raw block's text and language are fields of its node, beside the repr its `literal` answers.
#[test]
fn a_raw_block_answers_its_text_and_language() {
    let c = rx_ctx_new();
    let leaf = ok(c, "Leaf", &[], &LEAF.replace("{}", "psc_x"));
    let block = "RawBlock(name='raw', access='sample_local', text='mut_y = psc_x;', \
                 language='aether', reads=(), writes=(), \
                 returns=TensorType(shape=(), dtype='f64', tag=None))";
    let raw = ok(
        c,
        "Op",
        &[leaf],
        &format!("(('kind', 'raw_device'), ('literal', {block}))"),
    );
    let field = |name: &str| {
        let name = CString::new(name).unwrap();
        text(c, |c, b, cap, l| rx_field(c, raw, name.as_ptr(), b, cap, l))
    };
    assert_eq!(field("text"), "'mut_y = psc_x;'");
    assert_eq!(field("language"), "'aether'");
    assert!(field("literal").starts_with("RawBlock("));
    let name = CString::new("text").unwrap();
    let mut len = 0usize;
    assert_eq!(
        rx_field(c, leaf, name.as_ptr(), std::ptr::null_mut(), 0, &mut len),
        RX_REFUSED
    );
    rx_ctx_free(c);
}

/// The loop constructor computes the boundary: an invariant read of the body becomes an operand
/// after the initial values.
#[test]
fn the_loop_constructor_computes_its_boundary() {
    let c = rx_ctx_new();
    let index_fields = "(('depth', 0), ('name', 'k'), ('bounds', None))";
    let index = mk(c, "LoopIndex", "i32", &[], None, &[], index_fields).1;
    let carry = ok(
        c,
        "LoopCarry",
        &[],
        "(('depth', 0), ('slot', 0), ('name', 's'))",
    );
    let init = ok(c, "Const", &[], "(('literal', 0.0),)");
    let read = ok(c, "Leaf", &[], &LEAF.replace("{}", "x"));
    let next = ok(
        c,
        "Op",
        &[carry, read],
        "(('kind', 'add'), ('literal', None))",
    );
    let mut lp = 0u32;
    let none = std::ptr::null();
    let st = rx_new_loop(
        c,
        index,
        &carry,
        1,
        &init,
        &next,
        0,
        3,
        1,
        none,
        0,
        none,
        none,
        0,
        none,
        0,
        none as *const _,
        &mut lp,
    );
    assert_eq!(st, RX_OK, "{}", err(c));
    let mut ops = [0u32; 4];
    let mut n = 0usize;
    assert_eq!(
        rx_operands(c, lp, ops.as_mut_ptr(), ops.len(), &mut n),
        RX_OK
    );
    assert_eq!(&ops[..n], &[init, read]);
    rx_ctx_free(c);
}

/// A declared sink states its own role: each role the manifest lists is accepted, the role of the
/// core kind it refines is only the default, and any other role is refused.
#[test]
fn a_declared_sink_takes_any_listed_role() {
    let c = rx_ctx_new();
    let manifest = r#"{"name":"demo","kinds":[{"name":"demo.ordered","family":"sink",
        "refines":"wide_write","operands":{"min":3,"max":3},"result":["same_as",0],
        "fields":[{"name":"kept","type":"tuple"}],"identity":["role","name","kept"]}]}"#;
    let declared = rx_declare_dialect(
        c,
        manifest.as_ptr() as *const _,
        manifest.len(),
        c"()".as_ptr(),
    );
    assert_eq!(declared, RX_OK);
    let x = ok(c, "Const", &[], "(('literal', 1.0),)");
    let roles = |c: *mut Ctx| -> Vec<String> {
        let mut len = 0usize;
        let null = std::ptr::null_mut();
        assert_eq!(
            rx_const(c, c"SINK_ROLES".as_ptr(), null, 0, &mut len),
            RX_TRUNC
        );
        let listed = text(c, |c, b, cap, l| {
            rx_const(c, c"SINK_ROLES".as_ptr(), b, cap, l)
        });
        let items = listed.trim_matches(['[', ']']).split(',');
        items.map(|r| r.trim_matches('"').to_string()).collect()
    };
    let listed = roles(c);
    assert_eq!(listed, ["mutable", "out", "wide_out", "accum_out"]);
    for role in &listed {
        let fields = format!("(('name', 'bar_W'), ('role', '{role}'), ('kept', ()))");
        assert_eq!(
            mk(c, "DemoOrdered", "f64", &[], None, &[x, x, x], &fields).0,
            RX_OK,
            "{role}"
        );
    }
    let fixed = mk(
        c,
        "DemoOrdered",
        "f64",
        &[],
        None,
        &[x, x, x],
        "(('name', 'w'), ('kept', ()))",
    );
    assert_eq!(fixed.0, RX_OK, "{}", err(c));
    let role = text(c, |c, b, cap, l| {
        rx_field(c, fixed.1, c"role".as_ptr(), b, cap, l)
    });
    assert_eq!(role, "'wide_out'");
    let odd = "(('name', 'w'), ('role', 'nope'), ('kept', ()))";
    let bad = mk(c, "DemoOrdered", "f64", &[], None, &[x, x, x], odd);
    assert_eq!(bad.0, RX_REFUSED);
    assert_eq!(
        err(c),
        "sink 'w': unknown sink role 'nope', expected one of \
         ('mutable', 'out', 'wide_out', 'accum_out')"
    );
    // the node with role accum_out dumps and loads again in a context that declares the kind
    let kept = mk(
        c,
        "DemoOrdered",
        "f64",
        &[],
        None,
        &[x, x, x],
        "(('name', 'bar_W'), ('role', 'accum_out'), ('kept', ()))",
    )
    .1;
    let dump = |c: *mut Ctx, roots: &[u32]| {
        text(c, |c, b, cap, l| {
            raptor_support::safe::rx_dump(c, roots.as_ptr(), roots.len(), b, cap, l)
        })
    };
    let written = dump(c, &[kept]);
    let d = rx_ctx_new();
    let declared = rx_declare_dialect(
        d,
        manifest.as_ptr() as *const _,
        manifest.len(),
        c"()".as_ptr(),
    );
    assert_eq!(declared, RX_OK);
    let (mut roots, mut n) = ([0u32; 4], 0usize);
    let (text_ptr, len) = (written.as_ptr() as *const _, written.len());
    assert_eq!(
        raptor_support::safe::rx_load(d, text_ptr, len, roots.as_mut_ptr(), 4, &mut n),
        RX_OK
    );
    assert_eq!(dump(d, &roots[..n]), written);
    rx_ctx_free(d);
    rx_ctx_free(c);
}
