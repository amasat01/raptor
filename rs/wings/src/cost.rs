//! The 64-bit floating-point operations a set of nodes runs once, counted in the IR: the per-class operation count of a
//! node, kept for the 64-bit results only. A step of a finishing kernel is priced by it.
use super::{is_a, konst, library_kind, op_kind, R};
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::typing::all_kinds;

/// Ops that only move or re-label values.
const FREE: [&str; 9] = [
    "component",
    "component_at",
    "set_component_at",
    "add_at",
    "vec",
    "splat",
    "transpose",
    "as_pure",
    "as_vec3",
];
/// The operations of one lane of the ray ops, by kind.
const RAYS: [(&str, f64); 3] = [("ray_tri", 70.0), ("ray_aabb", 60.0), ("ray_sphere", 45.0)];
/// The operations of the quaternion ops, by kind.
const QUATS: [(&str, f64); 3] = [
    ("quat_mul", 28.0),
    ("quat_rotate", 56.0),
    ("quat_recip", 8.0),
];

/// The number of elements of the node's value.
fn elements(c: &Ctx, h: u32) -> f64 {
    c.nodes[h as usize].shape.iter().product::<i64>() as f64
}

/// Whether the node computes in 64 bits: a float result, or an operand that is one, and not an integer op.
fn wide(c: &Ctx, h: u32) -> R<bool> {
    let node = &c.nodes[h as usize];
    let int_dtype = konst("INT_PLANE_DTYPES").contains(&node.dtype.as_str());
    let int_kind = super::in_kinds(c, h, |k| {
        konst("INT_BINARY").contains(&k) || konst("INT_UNARY").contains(&k)
    })?;
    if int_dtype || int_kind {
        return Ok(false);
    }
    Ok(node.dtype == "f64" || node.ops.iter().any(|&o| c.nodes[o as usize].dtype == "f64"))
}

/// The operands' shapes the linear-algebra count reads: the matrix side and the right-hand side's columns.
fn geometry(c: &Ctx, h: u32, kind: &str) -> (f64, f64) {
    let ops = &c.nodes[h as usize].ops;
    let n = *c.nodes[ops[0] as usize].shape.last().unwrap_or(&1) as f64;
    let rhs = if konst("LINALG_CONSUMERS").contains(&kind) {
        &c.nodes[ops[1] as usize].shape
    } else {
        &vec![]
    };
    (n, if rhs.len() == 2 { rhs[1] as f64 } else { 1.0 })
}

/// The operations of a linear-algebra op.
fn linalg_ops(c: &Ctx, h: u32, kind: &str) -> f64 {
    let (n, k) = geometry(c, h, kind);
    let m = c.nodes[c.nodes[h as usize].ops[0] as usize].shape[0] as f64;
    let sweeps = match n as i64 {
        ..=2 => 2.0,
        3..=4 => 6.0,
        5..=8 => 8.0,
        9..=16 => 12.0,
        _ => 15.0,
    };
    match kind {
        "qr" => m * n * n,
        "qr_solve" => m * n * k + n * n * k,
        "lu" => n.powi(3) / 3.0,
        "chol" => n.powi(3) / 6.0,
        "eigh" => sweeps * n * (n - 1.0) / 2.0 * (6.0 * n + 10.0),
        "lu_det" => n,
        _ if konst("LINALG_CONSUMERS").contains(&kind) => {
            n * n * k
                / if matches!(kind, "tri_lower" | "tri_upper_t") {
                    2.0
                } else {
                    1.0
                }
        }
        _ => 0.0,
    }
}

/// The operations an op of this kind runs, in whichever class.
fn op_ops(c: &Ctx, h: u32, kind: &str) -> f64 {
    let ops = &c.nodes[h as usize].ops;
    let table = |t: &[(&str, f64)]| t.iter().find(|x| x.0 == kind).map(|x| x.1);
    if [
        "LINALG_FACTORISATIONS",
        "LINALG_CONSUMERS",
        "LINALG_PROJECTIONS",
    ]
    .iter()
    .any(|g| konst(g).contains(&kind))
    {
        return linalg_ops(c, h, kind);
    }
    if let Some(n) = table(&RAYS).or_else(|| table(&QUATS)) {
        return n;
    }
    if library_kind(kind) {
        return elements(c, h);
    }
    let shape = |i: usize| &c.nodes[ops[i] as usize].shape;
    match kind {
        "dot" | "norm" => 2.0 * elements(c, ops[0]) + f64::from(u8::from(kind == "norm")),
        "sum" => elements(c, ops[0]),
        "cross" => 9.0,
        "outer" => elements(c, h),
        "mv" | "mm" => {
            let cols = if kind == "mm" {
                *shape(1).last().unwrap_or(&1)
            } else {
                1
            };
            2.0 * shape(0).iter().product::<i64>() as f64 * cols as f64
        }
        _ => elements(c, h).max(1.0),
    }
}

/// The 64-bit floating-point operations the node adds itself, not counting its operands.
fn own_ops(c: &Ctx, h: u32) -> R<f64> {
    if is_a(c, h, "Leaf") || is_a(c, h, "At") {
        return Ok(0.0);
    }
    if is_a(c, h, "Loop") {
        return refuse("a loop inside the counted nodes has no trip count here");
    }
    if is_a(c, h, "Select") || is_a(c, h, "Dispatch") {
        return Ok(if wide(c, h)? {
            elements(c, h).max(1.0)
        } else {
            0.0
        });
    }
    let Some(kind) = op_kind(c, h)? else {
        return Ok(0.0);
    };
    let node = &c.nodes[h as usize];
    let widens =
        kind == "cast" && node.dtype == "i32" && c.nodes[node.ops[0] as usize].dtype == "int32";
    let free =
        FREE.contains(&kind.as_str()) || konst("LINALG_PROJECTIONS").contains(&kind.as_str());
    if !all_kinds().contains(&kind.as_str()) || free || widens || !wide(c, h)? {
        return Ok(0.0);
    }
    Ok(op_ops(c, h, &kind))
}

/// The 64-bit floating-point operations `nodes` run once: the arithmetic and library calls of the nodes that compute in 64 bits
/// (a library call counts once per element). Refuses a loop among the nodes.
pub fn fp64_ops(c: &Ctx, nodes: &[u32]) -> R<i64> {
    let mut total = 0.0;
    for &h in nodes {
        c.node(h)?;
        if !is_a(c, h, "Const") {
            total += own_ops(c, h)?;
        }
    }
    Ok(total as i64)
}

/// The 64-bit floating-point operations of one trip of the walk's one top-level loop (a finishing kernel's step); none when the
/// walk has no such loop or more than one.
pub fn step_fp64_ops(c: &Ctx, walk: u32) -> R<Option<i64>> {
    let order = c.walk(walk)?.order.clone();
    let mut loops = order.iter().copied().filter(|&n| is_a(c, n, "Loop"));
    match (loops.next(), loops.next()) {
        (Some(lp), None) => fp64_ops(c, &c.body_nodes(lp)?).map(Some),
        _ => Ok(None),
    }
}
