//! The derivative rules over the nodes of a context. Oracle: tests/data/rule_cases.json, recorded by running the previous Python
//! interpreter of the same manifest tables on the same operand graphs. Each case names a rule, the operand graph, and the
//! structural text of the nodes the old run yielded (one line per distinct node in post-order, so creation order does not
//! matter) or the refusal it gave. Every kind of every rule table has a case.
use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::rules::{rx_rule_factor, rx_rule_kinds};
use raptor_ir::value::{parse, text_val, Val};
use rcore::json::{self, J};
use std::collections::HashMap;
use std::ffi::{c_char, CString};

/// The recorded cases.
const CASES: &str = include_str!("data/rule_cases.json");

/// The fields of a leaf node of the given name.
fn leaf_fields(name: &str) -> String {
    format!("(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))")
}

/// A leaf named `name` of the type of node `like`.
fn leaf_like(ctx: &mut Ctx, name: &str, like: u32) -> u32 {
    let n = ctx.node(like).unwrap();
    let (dtype, shape, tag) = (n.dtype.clone(), n.shape.clone(), n.tag.clone());
    ctx.make(
        "Leaf",
        &dtype,
        &shape,
        tag.as_deref(),
        &[],
        &leaf_fields(name),
    )
    .unwrap()
}

/// The graph a spec describes.
fn build(ctx: &mut Ctx, spec: &J) -> Result<u32, Fault> {
    if let Some(name) = spec.get("leaf").and_then(J::str) {
        let shape: Vec<i64> = spec
            .get("shape")
            .unwrap()
            .arr()
            .iter()
            .map(J::int)
            .collect();
        let dtype = spec.get("dtype").and_then(J::str).unwrap();
        return ctx.make("Leaf", dtype, &shape, None, &[], &leaf_fields(name));
    }
    let args = spec.get("args").unwrap().arr();
    let mut ops = vec![];
    for a in args {
        ops.push(build(ctx, a)?);
    }
    if let Some(op) = spec.get("reduce").and_then(J::str) {
        let params = format!("(('op', '{op}'), ('value', #0))");
        return ctx.new_node("ReduceSamples", &params, &ops);
    }
    let lit = parse(spec.get("lit").and_then(J::str).unwrap()).unwrap();
    ctx.make_op(spec.get("kind").and_then(J::str).unwrap(), &ops, &lit)
}

/// The structural post-order text of the graphs under `roots`: one line per distinct node, then one per root.
fn canon(ctx: &Ctx, roots: &[u32]) -> Vec<String> {
    let (mut seen, mut lines, mut memo) = (HashMap::new(), vec![], HashMap::new());
    let mut out = vec![];
    for &r in roots {
        out.push(visit(ctx, r, &mut seen, &mut lines, &mut memo));
    }
    lines.extend(out.iter().map(|i| format!("root|{i}")));
    lines
}

/// The index of the line of node `h`, adding the lines of what it reads first.
fn visit(
    ctx: &Ctx,
    h: u32,
    seen: &mut HashMap<String, usize>,
    lines: &mut Vec<String>,
    memo: &mut HashMap<u32, usize>,
) -> usize {
    if let Some(&i) = memo.get(&h) {
        return i;
    }
    let node = ctx.node(h).unwrap();
    let ops: Vec<Val> = node
        .ops
        .clone()
        .into_iter()
        .map(|o| Val::Int(visit(ctx, o, seen, lines, memo) as i128))
        .collect();
    let class = ctx.table.classes[node.class].name.clone();
    let text = |f: &str| text_val(&ctx.field(h, f).unwrap());
    let key = match class.as_str() {
        "Op" => format!("{}|{}", ctx.str_field(h, "kind").unwrap(), text("literal")),
        "Const" => text("literal"),
        "Leaf" => ctx.str_field(h, "name").unwrap(),
        "Select" => String::new(),
        "ReduceSamples" => ctx.str_field(h, "op").unwrap(),
        other => panic!("class {other} has no text form"),
    };
    let shape = Val::Tuple(node.shape.iter().map(|&d| Val::Int(d as i128)).collect());
    let tag = node.tag.clone().map_or(Val::None, Val::Str);
    let line = format!(
        "{class}|{}|{}|{}|{}|{key}",
        node.dtype,
        text_val(&shape),
        text_val(&tag),
        text_val(&Val::Tuple(ops))
    );
    let next = seen.len();
    let i = *seen.entry(line.clone()).or_insert(next);
    if i == lines.len() {
        lines.push(line);
    }
    memo.insert(h, i);
    i
}

/// The nodes a case's rule yields over its graph, or its refusal.
fn run(ctx: &mut Ctx, case: &J) -> Result<Vec<u32>, Fault> {
    let (table, kind) = (
        case.get("table").and_then(J::str).unwrap(),
        case.get("kind").and_then(J::str).unwrap(),
    );
    let vjp = case.get("mode").and_then(J::str) == Some("vjp");
    let n = build(ctx, case.get("spec").unwrap())?;
    let ops = ctx.node(n)?.ops.clone();
    let g = leaf_like(ctx, "g", n);
    let mask: Vec<bool> = case
        .get("mask")
        .map_or(&[][..], J::arr)
        .iter()
        .map(|m| matches!(m, J::Bool(true)))
        .collect();
    let mut tangents = vec![];
    for (i, (&o, live)) in ops.iter().zip(&mask).enumerate() {
        if *live {
            tangents.push((o, leaf_like(ctx, &format!("t{i}"), o)));
        }
    }
    let name = |s: &str| s.to_string();
    match table {
        "unary_d" => ctx.rule_factor(table, kind, &[(name("n"), n), (name("x"), ops[0])], &[]),
        "bin_vjp" => {
            let env = [("n", n), ("g", g), ("a", ops[0]), ("b", ops[1])];
            let env: Vec<(String, u32)> = env.iter().map(|(k, v)| (name(k), *v)).collect();
            ctx.rule_factor(table, kind, &env, &[])
        }
        "bin_jvp" => {
            let mut env = vec![(name("n"), n), (name("a"), ops[0]), (name("b"), ops[1])];
            let mut absent = vec![];
            for (i, label) in ["ta", "tb"].iter().enumerate() {
                match tangents.iter().find(|p| p.0 == ops[i]) {
                    Some(p) => env.push((name(label), p.1)),
                    None => absent.push(name(label)),
                }
            }
            ctx.rule_factor(table, kind, &env, &absent)
        }
        "reduce" if vjp => ctx.reduce_vjp(n, g),
        "reduce" => ctx.reduce_jvp(n, &tangents),
        _ if vjp => ctx.rule_vjp(table, kind, n, g),
        _ => ctx.rule_jvp(table, kind, n, &tangents),
    }
}

/// Every recorded case yields what the old interpreter yielded: the same nodes as text, or the same refusal.
#[test]
fn every_rule_gives_what_the_recorded_run_gave() {
    let all = json::parse(CASES).unwrap();
    let cases = all.get("cases").unwrap().arr();
    assert!(cases.len() > 400, "only {} cases", cases.len());
    for case in cases {
        let id = case.get("id").and_then(J::str).unwrap();
        let mut ctx = Ctx::new().unwrap();
        let got = run(&mut ctx, case);
        match (case.get("error").and_then(J::str), got) {
            (Some(want), Err(Fault::Refused(m))) => assert_eq!(m, want, "{id}"),
            (Some(want), other) => {
                panic!("{id}: wanted the refusal {want:?}, got {:?}", other.is_ok())
            }
            (None, Ok(roots)) => {
                let want: Vec<String> = case
                    .get("expect")
                    .unwrap()
                    .arr()
                    .iter()
                    .map(|l| l.str().unwrap().to_string())
                    .collect();
                assert_eq!(canon(&ctx, &roots), want, "{id}");
            }
            (None, Err(f)) => panic!("{id}: refused: {}", fault_text(&f)),
        }
    }
}

/// The text of a fault.
fn fault_text(f: &Fault) -> String {
    match f {
        Fault::Refused(m) | Fault::BadArg(m) => m.clone(),
        _ => "absent or truncated".into(),
    }
}

/// Every kind of every rule table has a recorded case that yields nodes, and the tables list the kinds the record lists.
#[test]
fn every_kind_of_every_table_is_covered_and_listed() {
    let all = json::parse(CASES).unwrap();
    let ctx = Ctx::new().unwrap();
    for (table, kinds) in all.get("kinds").unwrap().obj() {
        let listed = ctx.rule_kinds(table);
        assert_eq!(listed, kinds.strs(), "kinds of {table}");
        for kind in &listed {
            let covered = all.get("cases").unwrap().arr().iter().any(|c| {
                c.get("table").and_then(J::str) == Some(table)
                    && c.get("kind").and_then(J::str) == Some(kind)
                    && c.get("error").is_none()
            });
            assert!(covered, "{table}/{kind} has no case that yields nodes");
        }
    }
    for (mode, kinds) in all.get("modes").unwrap().obj() {
        assert_eq!(ctx.rule_modes(mode), kinds.strs(), "kinds of mode {mode}");
    }
}

/// A rule table nobody wrote is refused by name, not answered with nothing.
#[test]
fn a_kind_without_a_template_is_refused_by_name() {
    let mut ctx = Ctx::new().unwrap();
    let x = build(
        &mut ctx,
        &json::parse(r#"{"leaf":"x","shape":[],"dtype":"f64"}"#).unwrap(),
    )
    .unwrap();
    let Err(Fault::Refused(m)) = ctx.rule_factor("unary_d", "nosuchkind", &[("x".into(), x)], &[])
    else {
        panic!("expected a refusal")
    };
    assert_eq!(m, "rules: no unary_d template for nosuchkind");
}

/// The C doors carry the same answers: the kinds of a table as names, and a factor into a caller's buffer.
#[test]
fn the_c_doors_answer_as_the_rust_calls_do() {
    let ctx = rx_ctx_new();
    let ok = |status: i32| assert_eq!(status, RX_OK);
    let (table, mut buf, mut len, mut count) =
        (CString::new("unary_d").unwrap(), [0 as c_char; 512], 0, 0);
    ok(unsafe {
        rx_rule_kinds(
            ctx,
            table.as_ptr(),
            buf.as_mut_ptr(),
            buf.len(),
            &mut len,
            &mut count,
        )
    });
    assert_eq!(count, 26);
    let names = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
    assert_eq!(names.to_str().unwrap(), "neg");
    // d/dx neg(x) over one bound leaf: a single node comes back
    let c = unsafe { &mut *ctx };
    let x = build(
        c,
        &json::parse(r#"{"leaf":"x","shape":[],"dtype":"f64"}"#).unwrap(),
    )
    .unwrap();
    let n = c.make_op("neg", &[x], &Val::None).unwrap();
    let (nodes, bound, kind) = ([n, x], "n x", CString::new("neg").unwrap());
    let (mut out, mut got) = ([0u32; 4], 0);
    ok(unsafe {
        rx_rule_factor(
            ctx,
            table.as_ptr(),
            kind.as_ptr(),
            bound.as_ptr().cast(),
            bound.len(),
            nodes.as_ptr(),
            2,
            std::ptr::null(),
            0,
            out.as_mut_ptr(),
            out.len(),
            &mut got,
        )
    });
    assert_eq!(got, 1);
    unsafe { rx_ctx_free(ctx) };
}

/// The literal of the zero of each dtype, as the field text states it.
fn zero_literal(ctx: &mut Ctx, dtype: &str) -> Result<Val, Fault> {
    let zero = ctx.zero_of(dtype, &[])?;
    assert_eq!(ctx.node(zero).unwrap().dtype, dtype);
    ctx.field(zero, "literal")
}

#[test]
fn a_zero_states_the_literal_of_its_own_dtype_and_a_dtype_without_one_is_refused() {
    let mut ctx = Ctx::new().unwrap();
    assert_eq!(zero_literal(&mut ctx, "f64").unwrap(), Val::Float(0.0));
    assert_eq!(zero_literal(&mut ctx, "f32").unwrap(), Val::Float(0.0));
    // the integer names the IR states: an index's (`i32`, `i64`) and an integer plane's (`int32`, `uint32`)
    for integer in ["i32", "i64", "int32", "uint32"] {
        assert_eq!(
            zero_literal(&mut ctx, integer).unwrap(),
            Val::Int(0),
            "{integer}"
        );
    }
    assert_eq!(zero_literal(&mut ctx, "bool").unwrap(), Val::Bool(false));
    let refused = ctx.zero_of("c128", &[]).unwrap_err();
    assert!(fault_text(&refused).contains("dtype \"c128\" has no zero"));
    // a splat of the zero keeps the dtype
    let splat = ctx.zero_of("f32", &[3]).unwrap();
    assert_eq!(ctx.node(splat).unwrap().dtype, "f32");
}
