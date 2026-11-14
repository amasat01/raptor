//! The door's C functions only.
use raptor_ir::canon::*;
use raptor_ir::dialect::*;
use raptor_ir::door::*;
use raptor_ir::pseudo::*;
use raptor_ir::record::Fault;
use raptor_ir::record::Ctx;
use rcore::{walk, Rec};
use std::ffi::{CStr, CString};

fn mk(c: *mut Ctx, cls: &str, dtype: &str, shape: &[i64], tag: Option<&str>, ops: &[u32], fields: &str) -> (i32, u32) {
    let (cls, dtype, tag) = (CString::new(cls).unwrap(), CString::new(dtype).unwrap(), tag.map(|t| CString::new(t).unwrap()));
    let mut out = u32::MAX;
    let st = rx_make(c, cls.as_ptr(), dtype.as_ptr(), shape.as_ptr(), shape.len(), tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
                     ops.as_ptr(), ops.len(), fields.as_ptr() as *const _, fields.len(), &mut out);
    (st, out)
}
fn ok(c: *mut Ctx, cls: &str, ops: &[u32], fields: &str) -> u32 {
    let (st, h) = mk(c, cls, "f64", &[], None, ops, fields);
    assert_eq!(st, RX_OK, "{cls}: {}", err(c));
    h
}
fn err(c: *mut Ctx) -> String { unsafe { CStr::from_ptr(rx_last_error(c)) }.to_str().unwrap().to_string() }
fn text(c: *mut Ctx, f: impl Fn(*mut Ctx, *mut std::ffi::c_char, usize, *mut usize) -> i32) -> String {
    let mut len = 0usize;
    assert_eq!(f(c, std::ptr::null_mut(), 0, &mut len), RX_TRUNC);
    let mut buf = vec![1 as std::ffi::c_char; len + 1];
    assert_eq!(f(c, buf.as_mut_ptr(), len + 1, &mut len), RX_OK);
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap().to_string()
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
        assert_eq!(text(c, |c, buf, cap, len| rx_class(c, *h, buf, cap, len)), want);
        assert_eq!(text(c, |c, buf, cap, len| rx_ttype(c, *h, buf, cap, len)), "TensorType(shape=(), dtype='f64', tag=None)");
    }
    let ops = |h: u32| { let mut v = [0u32; 4]; let mut n = 0; assert_eq!(rx_operands(c, h, v.as_mut_ptr(), 4, &mut n), RX_OK); v[..n].to_vec() };
    assert_eq!((ops(a), ops(s), ops(k), ops(e), ops(y)), (vec![], vec![a, b], vec![], vec![s, k], vec![e]));
    let mut n = 0; let mut small = [0u32; 1];
    assert_eq!(rx_operands(c, s, small.as_mut_ptr(), 1, &mut n), RX_TRUNC);
    assert_eq!(n, 2);
    let field = |h: u32, f: &str| { let f = CString::new(f).unwrap(); text(c, |c, buf, cap, len| rx_field(c, h, f.as_ptr(), buf, cap, len)) };
    assert_eq!((field(a, "name").as_str(), field(a, "role").as_str(), field(a, "wire_index").as_str(), field(a, "sorted").as_str()), ("'a'", "'vec_in'", "0", "False"));
    assert_eq!((field(k, "literal").as_str(), field(s, "kind").as_str(), field(s, "literal").as_str()), ("2.0", "'add'", "None"));
    assert_eq!((field(y, "name").as_str(), field(y, "role").as_str(), field(y, "kind").as_str()), ("'y'", "'mutable'", "'assign'"));
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
    let recs: Vec<Rec> = hs.iter().map(|&h| unsafe { &*c }.rec_of(h).unwrap()).collect();
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
        ("Nope", "f64", &[], "()", RX_REFUSED), ("Const", "f64", &[9], "(('literal', 1),)", RX_REFUSED),
        ("Const", "f64", &[], "(('literl', 1),)", RX_REFUSED), ("Const", "f64", &[], "()", RX_REFUSED),
        ("Const", "f64", &[], "(('literal', 1", RX_REFUSED), ("Const", "f77", &[], "(('literal', 1),)", RX_BADARG),
        ("Assign", "f64", &[x], "(('name', 'y'), ('role', 'out'))", RX_REFUSED), ("Const", "f64", &[], "(('literal', #5),)", RX_REFUSED),
    ];
    for (cls, dt, ops, f, want) in cases { assert_eq!(mk(c, cls, dt, &[], None, ops, f).0, want, "{cls} {f}"); }
    assert_eq!(mk(c, "Const", "f64", &[-1], None, &[], "(('literal', 1),)").0, RX_BADARG);
    assert_eq!(rx_class(c, 99, std::ptr::null_mut(), 0, &mut 0), RX_REFUSED);
    assert_eq!(rx_declare_dialect(c, std::ptr::null(), 0, std::ptr::null()), RX_BADARG);
    assert_eq!(rx_ctx_free_null(), ());
    assert_eq!(rx_make(std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null_mut()), RX_BADARG);
    rx_ctx_free(c);
}
fn rx_ctx_free_null() { rx_ctx_free(std::ptr::null_mut()) }

/// (class, operand count, fields) of a minimal valid record of each core class.
#[test]
fn every_core_class() {
    let c = rx_ctx_new();
    let (l, i) = (ok(c, "Leaf", &[], &LEAF.replace("{}", "p")), ok(c, "Const", &[], "(('literal', 0),)"));
    let ix = ok(c, "LoopIndex", &[], "(('depth', 0), ('name', 'i'), ('bounds', None))");
    let cr = ok(c, "LoopCarry", &[], "(('depth', 0), ('slot', 0), ('name', 'c'))");
    let sink = ok(c, "Assign", &[i], "(('name', 'o'),)");
    assert_eq!((l, i, ix, cr, sink), (0, 1, 2, 3, 4));
    let si = mk(c, "Const", "i32", &[], None, &[], "(('literal', 0),)").1;
    let cb = mk(c, "Const", "bool", &[], None, &[], "(('literal', True),)").1;
    let loop_f = "(('index', #2), ('carries', (#3,)), ('nexts', (#0,)), ('start', 0), ('stop', 2), ('step', 1), ('trip', 2), ('carry_count', 1), ('body_sinks', (#4,)), \
                  ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), ('count_tail', False), ('dynamic', None), ('_free_end', 1))";
    let rows: Vec<(&str, Vec<u32>, String)> = vec![
        ("Node", vec![], "()".into()), ("Leaf", vec![], LEAF.replace("{}", "q")), ("Const", vec![], "(('literal', 1),)".into()),
        ("RoleConst", vec![], "(('literal', 1), ('label', 'n'))".into()), ("SampleIndex", vec![], "()".into()), ("PlaneExtent", vec![l], "()".into()),
        ("SamplePosition", vec![], "()".into()), ("Op", vec![i], "(('kind', 'neg'), ('literal', None))".into()), ("At", vec![l, i], "()".into()),
        ("Primitive", vec![i], "(('name', 'f'), ('input_count', 0))".into()), ("Select", vec![i, i, i], "()".into()), ("Dispatch", vec![si, i, i], "()".into()),
        ("Sink", vec![i], "(('kind', 'assign'), ('role', 'mutable'), ('name', 'z'))".into()), ("Assign", vec![i], "(('name', 'a'),)".into()),
        ("WideWrite", vec![i], "(('name', 'w'),)".into()), ("AccumWrite", vec![i], "(('name', 'acc'), ('contract', 'auto'), ('reduced', None))".into()),
        ("ReduceSamples", vec![i], "(('op', 'sum'), ('key', None), ('count', None), ('contract', 'auto'), ('kept', ()))".into()),
        ("SampleSum", vec![i], "(('name', 's'), ('op', 'sum'), ('cell', 's'), ('columns', 1))".into()), ("LoopIndex", vec![], "(('depth', 0), ('name', 'j'), ('bounds', None))".into()),
        ("LoopCarry", vec![], "(('depth', 0), ('slot', 0), ('name', 'd'))".into()), ("Loop", vec![i], loop_f.into()),
    ];
    assert_eq!(rows.len(), 21);
    let mut made = vec![];
    for (cls, ops, f) in &rows { made.push((cls.to_string(), ok(c, cls, ops, f))) }
    let lp = made.last().unwrap().1;
    made.push(("TapeWrite".into(), mk(c, "TapeWrite", "f64", &[2], None, &[si, i, i], "(('name', 't'), ('width', 1))").1));
    made.push(("Finish".into(), mk(c, "Finish", "bool", &[], None, &[cb], "(('name', 'f'),)").1));
    let lp2 = mk(c, "Loop", "f64", &[], None, &[i], &loop_f.replace("('exit_cond', None), ('exit_values', ())", &format!("('exit_cond', #{cb}), ('exit_values', (#{i},))"))).1;
    made.push(("LoopCount".into(), mk(c, "LoopCount", "i32", &[], None, &[lp2], "()").1));
    for (cls, ops, f) in [("LoopValue", vec![lp], "(('literal', 0),)"),  ("TapeRead", vec![lp, i], "(('literal', 0), ('slot', 0))")] {
        made.push((cls.to_string(), ok(c, cls, &ops, f)));
    }
    assert_eq!(made.len(), 26);
    for (cls, h) in &made {
        assert_eq!(text(c, |c, buf, cap, len| rx_class(c, *h, buf, cap, len)), *cls);
        let r = unsafe { &*c }.rec_of(*h);
        if cls == "Loop" { assert!(matches!(r, Err(Fault::Refused(m)) if m.contains("not in this build")), "{cls}") } else { assert!(r.is_ok(), "{cls}") }
    }
    rx_ctx_free(c);
}

const DEMO: &str = r#"{"name":"demo","dialect":"demo","kinds":[
  {"name":"demo.twice","family":"expr","operands":{"min":1,"max":1},"result":["same_as",0],"fields":[{"name":"k","type":"int","default":2}]},
  {"name":"demo.loud","family":"sink","refines":"assign","operands":{"min":1,"max":1}}]}"#;

#[test]
fn dialect_declare_and_make() {
    let c = rx_ctx_new();
    let decl = |c: *mut Ctx, m: &str| rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), c"()".as_ptr());
    assert_eq!(decl(c, DEMO), RX_OK, "{}", err(c));
    assert_eq!(decl(c, DEMO), RX_OK, "an equal declaration is accepted again");
    assert_eq!(decl(c, &DEMO.replace("\"default\":2", "\"default\":3")), RX_REFUSED);
    let x = ok(c, "Const", &[], "(('literal', 1.0),)");
    let t = ok(c, "DemoTwice", &[x], "()");
    assert_eq!(text(c, |c, b, cap, l| rx_class(c, t, b, cap, l)), "DemoTwice");
    let f = CString::new("k").unwrap();
    assert_eq!(text(c, |c, b, cap, l| rx_field(c, t, f.as_ptr(), b, cap, l)), "2");
    assert_eq!(text(c, |c, b, cap, l| rx_field(c, t, c"kind".as_ptr(), b, cap, l)), "'demo.twice'");
    assert_eq!(mk(c, "DemoTwice", "f64", &[], None, &[x, x], "()").0, RX_REFUSED);
    assert_eq!(mk(c, "DemoTwice", "f64", &[], None, &[x], "(('k', 'a'),)").0, RX_REFUSED);
    assert_eq!(mk(c, "DemoTwice", "i32", &[], None, &[x], "()").0, RX_REFUSED);
    ok(c, "DemoLoud", &[x], "(('name', 'o'),)");
    let c2 = rx_ctx_new();
    assert_eq!(mk(c2, "DemoTwice", "f64", &[], None, &[], "()").0, RX_REFUSED, "dialect classes are per context");
    rx_ctx_free(c2);
    rx_ctx_free(c);
}

#[test]
fn dialect_feathers_and_spelling() {
    let c = rx_ctx_new();
    let decl = |c: *mut Ctx, m: &str| rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), c"()".as_ptr());
    let row = |extra: &str| format!(r#"{{"name":"demo","kinds":[{{"name":"demo.k","family":"expr","operands":{{"min":1,"max":1}},"result":["same_as",0]{extra}}}]}}"#);
    assert_eq!(decl(c, &row(r#","feathers":["nope"]"#)), RX_REFUSED);
    assert_eq!(err(c), "kind 'demo.k': unknown feather 'nope'");
    assert_eq!(decl(c, &row(r#","spelling":{"dialect":"demo","name":"k","args":["ref3"]}"#)), RX_REFUSED);
    assert_eq!(err(c), "kind 'demo.k': spelling args ('ref3',) are not node|target|ref<i> within the operands");
    assert_eq!(decl(c, &row(r#","spelling":{"dialect":"demo","name":"k","args":["ref0","node"]}"#)), RX_OK, "{}", err(c));
    let c2 = rx_ctx_new();
    assert_eq!(decl(c2, &row(r#","feathers":["reads_by_element","reads_by_element"]"#)), RX_OK, "{}", err(c2));
    let x = ok(c2, "Const", &[], "(('literal', 1.0),)");
    let k = ok(c2, "DemoK", &[x], "()");
    assert_eq!(text(c2, |c, b, cap, l| rx_field(c, k, c"feathers".as_ptr(), b, cap, l)), "('reads_by_element',)");
    rx_ctx_free(c2);
    rx_ctx_free(c);
}

#[test]
fn rule_before_constructor_and_evaluator_failures() {
    let c = rx_ctx_new();
    let x = ok(c, "Const", &[], "(('literal', 1.0),)");
    // the operand-count and result-type rule of the kind come first, whatever the constructor would say
    assert_eq!(mk(c, "Select", "f64", &[], None, &[x, x], "()").0, RX_REFUSED);
    assert_eq!(err(c), "select: takes 3..3 operand(s), got 2");
    let (_, b) = mk(c, "Const", "bool", &[], None, &[], "(('literal', True),)");
    assert_eq!(mk(c, "Finish", "f64", &[], None, &[b], "(('name', 'f'),)").0, RX_REFUSED);
    assert_eq!(err(c), "finish: type f64 [] is not what its result rule derives, bool []");
    // a refused make leaves no node behind
    assert_eq!(mk(c, "Const", "f64", &[], None, &[], "(('literal', 2.0),)").1, 2);
    // a constructor that cannot read the state is not a refusal text of its own
    let c3 = rx_ctx_new();
    let l = ok(c3, "Leaf", &[], &LEAF.replace("{}", "p"));
    let i = ok(c3, "Const", &[], "(('literal', 0),)");
    ok(c3, "LoopIndex", &[], "(('depth', 0), ('name', 'i'), ('bounds', None))");
    ok(c3, "LoopCarry", &[], "(('depth', 0), ('slot', 0), ('name', 'c'))");
    ok(c3, "Assign", &[i], "(('name', 'o'),)");
    let lf = "(('index', #2), ('carries', (#3,)), ('nexts', (#0,)), ('start', 0), ('stop', 2), ('step', 1), ('trip', 2), ('carry_count', 1), ('body_sinks', (#4,)), \
              ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), ('count_tail', False), ('dynamic', None), ('_free_end', 1))";
    let lp = ok(c3, "Loop", &[i], lf);
    let _ = l;
    assert_eq!(mk(c3, "Loop", "f64", &[], None, &[i], &lf.replace("('carries', (#3,))", "('carries', 3)")).0, RX_REFUSED);
    assert!(err(c3).starts_with("Loop: its state cannot be read by its constructor ("), "{}", err(c3));
    rx_ctx_free(c3);
    // a dialect node meets the same base refusals: the typed door refuses a bad dtype or extent before any program runs
    let m = DEMO;
    assert_eq!(rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), c"()".as_ptr()), RX_OK);
    for cls in ["Const", "DemoTwice"] {
        let (ops, f): (&[u32], &str) = if cls == "Const" { (&[], "(('literal', 1.0),)") } else { (&[x], "()") };
        assert_eq!(mk(c, cls, "f77", &[], None, ops, f).0, RX_BADARG, "{cls}");
        assert_eq!(mk(c, cls, "f64", &[-2], None, ops, f).0, RX_BADARG, "{cls}");
    }
    rx_ctx_free(c);
}
