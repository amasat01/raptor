//! Dump and load through the door's C functions.
mod common;
use common::*;
use raptor_ir::door::*;
use raptor_ir::dump::*;
use rcore::keytext::py_str;

fn leaf_line(n: &str) -> String { format!("n|Leaf|f64|()|None|()|(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '{n}'), ('quantity', None), ('wire_index', 0), ('sorted', False))\n") }

#[test]
fn probe_expression_text() {
    let c = rx_ctx_new();
    let h = expression(c);
    let want = format!("raptor.ir.dump/1\n{}{}n|Op|f64|()|None|(#0, #1)|(('kind', 'add'), ('literal', None))\nn|Const|f64|()|None|()|(('literal', 2.0),)\n\
                        n|Op|f64|()|None|(#2, #3)|(('kind', 'add'), ('literal', None))\nn|Assign|f64|()|None|(#4,)|(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'))\nroot|#5\n",
                       leaf_line("a"), leaf_line("b"));
    assert_eq!(dump(c, &[h[5]]), want);
    rx_ctx_free(c);
}

/// dump, load into a second context, dump again: equal; returns the second context's node count.
fn round(c: *mut raptor_ir::record::Ctx, roots: &[u32]) -> (String, usize) {
    let t = dump(c, roots);
    let d = rx_ctx_new();
    let (st, r) = load(d, &t);
    assert_eq!(st, RX_OK, "{}", err(d));
    assert_eq!(r.len(), roots.len());
    assert_eq!(dump(d, &r), t);
    let n = count(d);
    rx_ctx_free(d);
    (t, n)
}

#[test]
fn round_trips() {
    let c = rx_ctx_new();
    let h = expression(c);
    assert_eq!(round(c, &[h[5]]).1, 6);
    // one object shared twice, two equal but separate objects
    let x = leaf(c, "x");
    let s = mk(c, "Op", None, &[x], "(('kind', 'neg'), ('literal', None))");
    let (k1, k2) = (mk(c, "Const", None, &[], "(('literal', 2.0),)"), mk(c, "Const", None, &[], "(('literal', 2.0),)"));
    let e = mk(c, "Op", None, &[s, s, k1, k2], ADD);
    assert_eq!(round(c, &[e]).1, 5);
    // a node reference in a field
    let r = mk(c, "AccumWrite", None, &[x], &format!("(('name', 'a'), ('contract', 'auto'), ('reduced', #{k1}))"));
    assert_eq!(round(c, &[r]).1, 3);
    // awkward text in a tag and a string field
    let odd = "a|b\"c'd\\e\nf";
    let l = mk(c, "Leaf", Some(odd), &[], &LEAF.replace("'{}'", &py_str(odd)));
    let (t, _) = round(c, &[l]);
    assert_eq!(t.lines().count(), 3);
    // shapes
    for sh in [&[][..], &[3], &[2, 3]] { let m = mk_shaped(c, sh); round(c, &[m]); }
    // value edges
    let lit = "(None, True, False, (), (1,), (-0.0, inf, -inf, nan, 1e+16, 5e-324), -7, 12345678901234567890)";
    let v = mk(c, "Const", None, &[], &format!("(('literal', {lit}),)"));
    let (t, _) = round(c, &[v]);
    assert!(t.contains("5e-324") && t.contains("12345678901234567890"));
    // a nested record
    let q = "Quantity(slug='q', name=None, wires=(Wire(name='a', role='vec_in', shape=(), dtype='f64', tag=None),), pack='auto')";
    let l = mk(c, "Leaf", None, &[], &LEAF.replace("'{}'", "'w'").replace("('quantity', None)", &format!("('quantity', {q})")));
    assert!(round(c, &[l]).0.contains("Wire(name='a'"));
    rx_ctx_free(c);
}

#[test]
fn loop_round_trip() {
    let c = rx_ctx_new();
    let (lp, sink) = a_loop(c);
    let (t, n) = round(c, &[lp]);
    assert_eq!(n, 6);
    // the loop's fields point at the records of its index, carry, next and body sink
    let d = rx_ctx_new();
    let (_, r) = load(d, &t);
    let f = |n: &str| { let n = std::ffi::CString::new(n).unwrap(); text(d, |c, b, cap, l| rx_field(c, r[0], n.as_ptr(), b, cap, l)) };
    assert_eq!((f("index"), f("carries"), f("nexts"), f("body_sinks")), ("#2".into(), "(#3,)".into(), "(#0,)".into(), "(#4,)".into()));
    let _ = sink;
    rx_ctx_free(d);
    rx_ctx_free(c);
}

#[test]
fn load_into_same_context_appends() {
    let c = rx_ctx_new();
    let h = expression(c);
    let t = dump(c, &[h[5]]);
    let (st, r) = load(c, &t);
    assert_eq!((st, count(c)), (RX_OK, 12));
    assert_eq!(r, vec![11]);
    assert_eq!(dump(c, &r), t);
    assert_eq!(dump(c, &[h[5]]), t);
    rx_ctx_free(c);
}

#[test]
fn reachability() {
    let c = rx_ctx_new();
    let h = expression(c);
    let stray = leaf(c, "stray");
    let t = dump(c, &[h[5]]);
    assert!(!t.contains("stray") && t.lines().count() == 8);
    // a root not reachable from another root still dumps, in arena order
    let t2 = dump(c, &[h[5], stray]);
    assert!(t2.contains("'stray'") && t2.ends_with("root|#5\nroot|#6\n"), "{t2}");
    // a root listed first or twice is written in the order given
    assert!(dump(c, &[stray, h[0]]).ends_with("root|#1\nroot|#0\n"));
    rx_ctx_free(c);
}

#[test]
fn refusals_name_the_line() {
    let c = rx_ctx_new();
    let h = expression(c);
    let t = dump(c, &[h[5]]);
    let before = count(c);
    let n1 = format!("{}\n", t.lines().nth(1).unwrap());
    let n1 = n1.as_str();
    let cases: Vec<(String, &str)> = vec![
        ("".into(), "line 1"), ("raptor.ir.dump/2\n".into(), "line 1: unknown version"), ("something\n".into(), "line 1: wrong or missing header"),
        (format!("raptor.ir.dump/1\n{n1}bogus|x\n"), "line 3: a line is neither"),
        ("raptor.ir.dump/1\nn|Op|f64|()|None|(#0,)|(('kind', 'neg'), ('literal', None))\n".into(), "line 2: reference #0 is not an earlier position"),
        (format!("raptor.ir.dump/1\n{n1}root|#1\n"), "line 3: root #1 is out of range"),
        (format!("raptor.ir.dump/1\n{n1}root|#0\n{n1}"), "line 4: node line after a root line"),
        (format!("raptor.ir.dump/1\n{n1}root|#0"), "line 3: text after the last newline"),
        ("raptor.ir.dump/1\nn|Const|f64|()|None|()|(('literal', (1,\n".into(), "line 2: unparsable field"),
        ("raptor.ir.dump/1\nn|Const|f64|()|None|()|(('litral', 1),)\n".into(), "line 2: "),
    ];
    for (text, want) in cases {
        let (st, _) = load(c, &text);
        assert_eq!(st, RX_REFUSED, "{text:?}");
        assert!(err(c).starts_with(want), "{text:?}: {}", err(c));
        assert_eq!(count(c), before, "a failed load leaves the arena");
    }
    // a buffer too small for the roots: TRUNC with the count, arena unchanged
    let mut n = 0;
    assert_eq!(rx_load(c, t.as_ptr() as *const _, t.len(), std::ptr::null_mut(), 0, &mut n), RX_TRUNC);
    assert_eq!((n, count(c)), (1, before));
    rx_ctx_free(c);
}
