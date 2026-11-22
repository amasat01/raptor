//! Element access of a value with one index operand per value axis: its typing and its derivative rules.
//! Oracle for the rules: the reverse of a read at indices is an accumulate at the same indices and the reverse of a
//! store is a read and a zeroing at the same indices (the single-index rules, carried over a list).
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::parse;

/// A leaf of the given dtype and shape.
fn leaf(ctx: &mut Ctx, name: &str, dtype: &str, shape: &[i64]) -> u32 {
    let fields = format!("(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))");
    ctx.make("Leaf", dtype, shape, None, &[], &fields).unwrap()
}

/// The node of `kind` over `ops`, or the refusal text.
fn op(ctx: &mut Ctx, kind: &str, ops: &[u32]) -> Result<u32, String> {
    let lit = parse("None").unwrap();
    ctx.make_op(kind, ops, &lit).map_err(|f| match f {
        Fault::Refused(m) | Fault::BadArg(m) => m,
        _ => "other fault".into(),
    })
}

/// The kind of node `n` and its operand list.
fn shape_of(ctx: &Ctx, n: u32) -> (Vec<i64>, Vec<u32>) {
    let node = ctx.node(n).unwrap();
    (node.shape.clone(), node.ops.clone())
}

#[test]
fn a_read_takes_one_integer_index_per_axis_of_a_value_of_rank_three() {
    let mut ctx = Ctx::new().unwrap();
    let v = leaf(&mut ctx, "v", "f64", &[2, 3, 4]);
    let idx: Vec<u32> = ["i", "j", "k"]
        .iter()
        .map(|n| leaf(&mut ctx, n, "i32", &[]))
        .collect();
    let read = op(&mut ctx, "component_at", &[v, idx[0], idx[1], idx[2]]).unwrap();
    assert_eq!(shape_of(&ctx, read).0, Vec::<i64>::new());
    let err = op(&mut ctx, "component_at", &[v, idx[0], idx[1]]).unwrap_err();
    assert!(
        err.contains("(2, 3, 4)") && err.contains("3 axes") && err.contains("2 index"),
        "{err}"
    );
}

#[test]
fn an_index_that_is_not_an_integer_or_a_value_of_rank_zero_is_refused() {
    let mut ctx = Ctx::new().unwrap();
    let v = leaf(&mut ctx, "v", "f64", &[2, 3]);
    let i = leaf(&mut ctx, "i", "i32", &[]);
    let x = leaf(&mut ctx, "x", "f64", &[]);
    let err = op(&mut ctx, "component_at", &[v, i, x]).unwrap_err();
    assert!(err.contains("index must be a rank-0 integer"), "{err}");
    let s = leaf(&mut ctx, "s", "f64", &[]);
    let err = op(&mut ctx, "component_at", &[s, i]).unwrap_err();
    assert!(err.contains("rank one or more"), "{err}");
}

#[test]
fn a_store_and_an_accumulate_take_the_indices_then_the_entry_and_keep_the_value_type() {
    let mut ctx = Ctx::new().unwrap();
    let v = leaf(&mut ctx, "v", "f64", &[2, 3]);
    let (i, j) = (
        leaf(&mut ctx, "i", "i32", &[]),
        leaf(&mut ctx, "j", "i32", &[]),
    );
    let e = leaf(&mut ctx, "e", "f64", &[]);
    for kind in ["set_component_at", "add_at"] {
        let n = op(&mut ctx, kind, &[v, i, j, e]).unwrap();
        assert_eq!(shape_of(&ctx, n).0, vec![2, 3], "{kind}");
        let err = op(&mut ctx, kind, &[v, i, e]).unwrap_err();
        assert!(
            err.contains("2 axes") && err.contains("1 index"),
            "{kind}: {err}"
        );
    }
}

#[test]
fn the_reverse_rules_carry_the_index_list() {
    let mut ctx = Ctx::new().unwrap();
    let v = leaf(&mut ctx, "v", "f64", &[2, 3]);
    let (i, j) = (
        leaf(&mut ctx, "i", "i32", &[]),
        leaf(&mut ctx, "j", "i32", &[]),
    );
    let e = leaf(&mut ctx, "e", "f64", &[]);
    let read = op(&mut ctx, "component_at", &[v, i, j]).unwrap();
    let g = leaf(&mut ctx, "g", "f64", &[]);
    let out = ctx.rule_vjp("core", "component_at", read, g).unwrap();
    // the pair (child v, add_at(zeros, i, j, g))
    assert_eq!(out[0], v);
    let (_, add_ops) = shape_of(&ctx, out[1]);
    assert_eq!(&add_ops[1..], &[i, j, g]);

    let store = op(&mut ctx, "set_component_at", &[v, i, j, e]).unwrap();
    let g2 = leaf(&mut ctx, "g2", "f64", &[2, 3]);
    let out = ctx.rule_vjp("core", "set_component_at", store, g2).unwrap();
    let (_, zeroing) = shape_of(&ctx, out[1]);
    assert_eq!(&zeroing[1..3], &[i, j]);
    let (_, entry) = shape_of(&ctx, out[3]);
    assert_eq!(entry, vec![g2, i, j]);

    let acc = op(&mut ctx, "add_at", &[v, i, j, e]).unwrap();
    let out = ctx.rule_vjp("core", "add_at", acc, g2).unwrap();
    let (_, entry) = shape_of(&ctx, out[3]);
    assert_eq!(entry, vec![g2, i, j]);
}

#[test]
fn the_forward_rules_carry_the_index_list() {
    let mut ctx = Ctx::new().unwrap();
    let v = leaf(&mut ctx, "v", "f64", &[2, 3]);
    let (i, j) = (
        leaf(&mut ctx, "i", "i32", &[]),
        leaf(&mut ctx, "j", "i32", &[]),
    );
    let e = leaf(&mut ctx, "e", "f64", &[]);
    let tv = leaf(&mut ctx, "tv", "f64", &[2, 3]);
    let read = op(&mut ctx, "component_at", &[v, i, j]).unwrap();
    let out = ctx
        .rule_jvp("core", "component_at", read, &[(v, tv)])
        .unwrap();
    assert_eq!(shape_of(&ctx, out[0]).1, vec![tv, i, j]);

    let te = leaf(&mut ctx, "te", "f64", &[]);
    for kind in ["set_component_at", "add_at"] {
        let n = op(&mut ctx, kind, &[v, i, j, e]).unwrap();
        let out = ctx.rule_jvp("core", kind, n, &[(v, tv), (e, te)]).unwrap();
        assert_eq!(shape_of(&ctx, out[0]).1, vec![tv, i, j, te], "{kind}");
    }
}
