//! Refusal parity with the declared checks, through the door's C functions. The cases come from tests/data/refusals.json: for each, the
//! pool of valid nodes, the bad node, and the message today's constructor gives.
mod common;
use common::*;
use raptor_ir::record::Ctx;
use raptor_support::safe::*;
use rcore::json::{self, J};
use rcore::keytext::py_str;
use std::ffi::CString;

/// Replaces each `#name` by the handle of that pool node.
fn subst(t: &str, names: &[String]) -> String {
    let (mut o, mut it) = (String::new(), t.char_indices().peekable());
    while let Some((i, ch)) = it.next() {
        if ch != '#' || !t[i + 1..].starts_with(|c: char| c.is_ascii_alphabetic()) {
            o.push(ch);
            continue;
        }
        let end = t[i + 1..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map_or(t.len(), |e| i + 1 + e);
        o += &format!(
            "#{}",
            names
                .iter()
                .position(|n| *n == t[i + 1..end])
                .expect("pool name")
        );
        while it.peek().is_some_and(|(j, _)| *j < end) {
            it.next();
        }
    }
    o
}

/// One make call from a case's node description; (status, handle).
fn make(c: *mut Ctx, s: &J, names: &[String]) -> (i32, u32) {
    let g = |k: &str| s.get(k).unwrap();
    let cls = CString::new(g("class").str().unwrap()).unwrap();
    let dtype = g("dtype").str().map(|d| CString::new(d).unwrap());
    let shape: Vec<i64> = g("shape").arr().iter().map(|x| x.int()).collect();
    let tag = g("tag").str().map(|d| CString::new(d).unwrap());
    let ops: Vec<u32> = g("ops")
        .arr()
        .iter()
        .map(|o| {
            o.str().map_or(o.int() as u32, |n| {
                names.iter().position(|x| x == n).unwrap() as u32
            })
        })
        .collect();
    let fields = subst(g("fields").str().unwrap(), names);
    let mut out = u32::MAX;
    let st = rx_make(
        c,
        cls.as_ptr(),
        dtype.as_ref().map_or(std::ptr::null(), |d| d.as_ptr()),
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

/// A repr of a node is `<Class ...>` today and `<Class #h>` here: both are masked to `<Class>`.
fn mask(m: &str) -> String {
    let (mut o, mut r) = (String::new(), m);
    while let Some(i) = r.find('<') {
        let w: String = r[i + 1..]
            .chars()
            .take_while(|c| c.is_alphanumeric())
            .collect();
        o += &format!("{}<{w}>", &r[..i]);
        r = &r[i + r[i..].find('>').unwrap() + 1..];
    }
    o + r
}

fn status(s: &str) -> i32 {
    match s {
        "RX_REFUSED" => RX_REFUSED,
        "RX_BADARG" => RX_BADARG,
        _ => panic!("{s}"),
    }
}

#[test]
fn declared_checks_have_parity() {
    let d = json::parse(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/refusals.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let pool = d.get("pool").unwrap().arr();
    let names: Vec<String> = pool
        .iter()
        .map(|p| p.get("name").unwrap().str().unwrap().to_string())
        .collect();
    let (mut a, mut b, mut cc, mut q, mut t) = (0, 0, 0, 0, 0);
    for case in d.get("cases").unwrap().arr() {
        let (group, id) = (
            case.get("group").unwrap().str().unwrap(),
            case.get("id").unwrap().int(),
        );
        let (expect, today) = (
            case.get("expect").unwrap().str().unwrap(),
            case.get("raw").unwrap().str().unwrap(),
        );
        let c = rx_ctx_new();
        for (i, p) in pool.iter().enumerate() {
            assert_eq!(
                make(c, p, &names),
                (RX_OK, i as u32),
                "pool {}: {}",
                names[i],
                err(c)
            )
        }
        let tag = format!("{group} {id}");
        let standing = case.get("standing").filter(|s| !matches!(s, J::Null));
        let before = count(c);
        match (
            expect,
            standing.and_then(|s| s.get("door")),
            case.get("bad").filter(|s| !matches!(s, J::Null)),
        ) {
            ("a", _, Some(bad)) | ("b", _, Some(bad)) => {
                assert_eq!(make(c, bad, &names).0, RX_REFUSED, "{tag}");
                let m = err(c);
                if expect == "a" {
                    assert_eq!(m, today, "{tag}")
                } else {
                    assert_eq!(mask(&m), mask(today), "{tag}");
                    assert!(m.contains(" #"), "{tag}: {m}")
                }
            }
            ("c", Some(door), _) => {
                let st = standing.unwrap();
                assert_eq!(
                    make(c, door, &names).0,
                    status(st.get("status").unwrap().str().unwrap()),
                    "{tag}"
                );
                assert!(
                    err(c).contains(st.get("msg").unwrap().str().unwrap()),
                    "{tag}: {}",
                    err(c)
                );
            }
            ("c", None, _) => assert!(standing.unwrap().get("none").is_some(), "{tag}"),
            _ => panic!("{tag}: no case description"),
        }
        assert_eq!(count(c), before, "{tag}: a refusal appends nothing");
        match (group, expect) {
            ("check", "a") => a += 1,
            ("check", "b") => b += 1,
            ("check", "c") => cc += 1,
            ("quantity", _) => q += 1,
            ("ttype", _) => t += 1,
            _ => panic!("{tag}"),
        }
        rx_ctx_free(c);
    }
    println!("text-equal {a} / 51\ndifferent by design {b}\nunreachable behind the door {cc}\nquantity {q}, type {t}");
    assert_eq!(a + b + cc, 51);
    assert!(a > 0 && b > 0 && cc > 0 && q == 6 && t == 4);
}

/// A tagged value of the corpus as value text; `h` maps node numbers to handles.
fn tagged(v: &J, h: &[u32]) -> String {
    match v {
        J::Null => "None".into(),
        J::Bool(b) => (if *b { "True" } else { "False" }).into(),
        J::Obj(_) => {
            let g = |k: &str| v.get(k);
            if let Some(n) = g("n") {
                format!("#{}", h[n.int() as usize])
            } else if let Some(t) = g("t") {
                format!(
                    "({}{})",
                    t.arr()
                        .iter()
                        .map(|x| tagged(x, h))
                        .collect::<Vec<_>>()
                        .join(", "),
                    if t.arr().len() == 1 { "," } else { "" }
                )
            } else if let Some(l) = g("l") {
                format!(
                    "[{}]",
                    l.arr()
                        .iter()
                        .map(|x| tagged(x, h))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else if let Some(i) = g("i") {
                i.str().unwrap().into()
            } else if let Some(f) = g("f") {
                f.str().unwrap().into()
            } else if let Some(s) = g("s") {
                py_str(s.str().unwrap())
            } else if let Some(r) = g("r") {
                format!(
                    "{}({})",
                    r.str().unwrap(),
                    g("v")
                        .unwrap()
                        .obj()
                        .iter()
                        .map(|(k, x)| format!("{k}={}", tagged(x, h)))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else {
                panic!("untagged value")
            }
        }
        _ => panic!("untagged value"),
    }
}

/// Every node of every walk of a corpus file goes through rx_make with the checks on (WALKS names the file).
#[test]
#[ignore]
fn corpus_positive() {
    let Ok(path) = std::env::var("WALKS") else {
        return;
    };
    let (mut checked, mut skipped, mut refused, mut msgs) =
        (0, 0, 0, std::collections::BTreeSet::new());
    let mut classes = std::collections::BTreeMap::<String, usize>::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let w = json::parse(line).unwrap();
        let Some(nodes) = w.get("inputs").and_then(|i| i.get("nodes")) else {
            continue;
        };
        let c = rx_ctx_new();
        let mut h: Vec<u32> = vec![];
        for n in nodes.arr() {
            let (cls, v) = (n.get("c").unwrap().str().unwrap(), n.get("v").unwrap());
            if !unsafe { &*c }.table.find(cls).is_some() {
                skipped += 1;
                h.push(u32::MAX);
                continue;
            }
            let tt = v.get("ttype").unwrap().get("v").unwrap();
            let shape: Vec<i64> = tt
                .get("shape")
                .unwrap()
                .get("t")
                .unwrap()
                .arr()
                .iter()
                .map(|x| x.get("i").unwrap().str().unwrap().parse().unwrap())
                .collect();
            let ops: Vec<u32> = v
                .get("operands")
                .unwrap()
                .get("t")
                .unwrap()
                .arr()
                .iter()
                .map(|x| h[x.get("n").unwrap().int() as usize])
                .collect();
            let fields = format!(
                "({}{})",
                v.obj()
                    .iter()
                    .filter(|(k, _)| k != "ttype" && k != "operands")
                    .map(|(k, x)| format!("({}, {})", py_str(k), tagged(x, &h)))
                    .collect::<Vec<_>>()
                    .join(", "),
                if v.obj().len() == 3 { "," } else { "" }
            );
            if ops.contains(&u32::MAX) || fields.contains("#4294967295") {
                skipped += 1;
                h.push(u32::MAX);
                continue;
            }
            let (cl, dt) = (
                CString::new(cls).unwrap(),
                CString::new(tt.get("dtype").unwrap().get("s").unwrap().str().unwrap()).unwrap(),
            );
            let tag = tt
                .get("tag")
                .and_then(|t| t.get("s"))
                .and_then(J::str)
                .map(|t| CString::new(t).unwrap());
            let mut out = 0;
            let st = rx_make(
                c,
                cl.as_ptr(),
                dt.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
                ops.as_ptr(),
                ops.len(),
                fields.as_ptr() as *const _,
                fields.len(),
                &mut out,
            );
            checked += 1;
            *classes.entry(cls.to_string()).or_default() += 1;
            if st != RX_OK {
                refused += 1;
                msgs.insert(format!("{cls}: {}", err(c)));
            }
            h.push(out);
        }
        rx_ctx_free(c);
    }
    println!("nodes checked {checked}, refused {refused}, skipped (dialect) {skipped}");
    println!("by class: {classes:?}");
    msgs.iter().for_each(|m| println!("  {m}"));
    assert!(checked > 0 && refused == 0);
}

/// The state a program produces must equal the record's: a derived slot that is wrong is refused.
#[test]
fn derived_slot_must_match() {
    let c = rx_ctx_new();
    let x = leaf(c, "x");
    let f = |n: i32| {
        format!("(('kind', 'primitive'), ('literal', None), ('name', 'f'), ('input_count', {n}))")
    };
    mk(c, "Primitive", None, &[x], &f(0));
    let (cls, dt, op, fl) = (
        CString::new("Primitive").unwrap(),
        CString::new("f64").unwrap(),
        [x],
        f(3),
    );
    let mut out = 0;
    assert_eq!(
        rx_make(
            c,
            cls.as_ptr(),
            dt.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            op.as_ptr(),
            1,
            fl.as_ptr() as *const _,
            fl.len(),
            &mut out
        ),
        RX_REFUSED
    );
    assert_eq!(
        err(c),
        "Primitive: input_count does not match its constructor (got 3, constructor gives 0)"
    );
    rx_ctx_free(c);
}

/// Every refusal of the op typing rules: the message of the typed constructor equals what the frozen constructor gave (tests/data/typing_cases.json).
#[test]
fn typing_refusals_have_parity() {
    let d = json::parse(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/typing_cases.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let (mut n, mut equal) = (0, 0);
    for case in d.arr() {
        let Some(want) = case.get("message").and_then(|m| m.str()) else {
            continue;
        };
        n += 1;
        let c = rx_ctx_new();
        let mut ops: Vec<u32> = vec![];
        for o in case.get("operands").unwrap().arr() {
            let dt = CString::new(o.get("dtype").unwrap().str().unwrap()).unwrap();
            let shape: Vec<i64> = o
                .get("shape")
                .unwrap()
                .arr()
                .iter()
                .map(|x| x.int())
                .collect();
            let tag = o
                .get("tag")
                .and_then(|t| t.str())
                .map(|t| CString::new(t).unwrap());
            let lit = match o.get("dtype").unwrap().str().unwrap() {
                "bool" => "(('literal', True),)",
                "f64" | "f32" => "(('literal', 0.0),)",
                _ => "(('literal', 0),)",
            };
            let (cls, mut out) = (CString::new("Const").unwrap(), 0u32);
            let st = rx_make(
                c,
                cls.as_ptr(),
                dt.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
                std::ptr::null(),
                0,
                lit.as_ptr() as *const _,
                lit.len(),
                &mut out,
            );
            assert_eq!(st, RX_OK, "{}", err(c));
            ops.push(out);
        }
        let (kind, lit) = (
            CString::new(case.get("kind").unwrap().str().unwrap()).unwrap(),
            case.get("literal").unwrap().str().unwrap(),
        );
        let lit = if lit == "None" { "" } else { lit };
        let mut out = 0u32;
        let st = rx_make_op(
            c,
            kind.as_ptr(),
            ops.as_ptr(),
            ops.len(),
            lit.as_ptr() as *const _,
            lit.len(),
            &mut out,
        );
        assert_eq!(
            st,
            RX_REFUSED,
            "{} over {:?}",
            kind.to_str().unwrap(),
            case.get("operands").unwrap()
        );
        assert_eq!(err(c), want, "{}", kind.to_str().unwrap());
        assert_eq!(count(c), ops.len(), "a refusal appends nothing");
        equal += 1;
        rx_ctx_free(c);
    }
    println!("typing refusals text-equal {equal} / {n}");
    assert!(n >= 80 && equal == n);
}
