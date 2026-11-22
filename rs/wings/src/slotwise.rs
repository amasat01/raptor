//! Slotwise form: a wide value built from `vec`/`splat` becomes a `vec` of scalar parts with the structural zeros folded, so a
//! component only one case writes is one scalar and not a column of an aggregate. Also the analyses a derivative pass asks of the
//! slotwise ops it builds: the chain of `add_at` links under a node and whether a node is a vector of zeros.
//!
//! Rewritten: `component` of a `vec`; `sum` of a rank-1 `vec` (left-nested `add`); `add`, `sub`, `neg`, `mul`, `div`; `select`;
//! `dispatch` (one dispatch per slot, same selector and strategy, so the renderer still fuses them into one `switch`). Everything
//! else over a `vec` keeps its node. An `add_at` chain over a slot-form base (the adjoint of a runtime-indexed component) is
//! kept as a chain and pushed through the same ops, and lowered to selects where a static slot is forced. A loop is a scope and
//! is never entered. A node with nothing to rewrite comes back as the same node.
use crate::cells::{build, int_like};
use crate::place::one_index;
use crate::{flag_out, in_kinds, is_a, list_out, op_kind, R};
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::{ttype_val, Val};
use std::collections::HashMap;
use std::ffi::c_int;

/// The floating dtypes a slot form is built for.
const FLOAT: [&str; 2] = ["f64", "f32"];

/// The base of the `add_at` chain that ends at `node` and its links, as the flat list `index`,
/// `value` for each link in the order they were added.
struct Chain {
    /// The node the chain starts from: the first that is not an `add_at`.
    base: u32,
    /// The index and the value of each link, flattened.
    links: Vec<u32>,
}

/// Follows the `add_at` ops from `node` down to the node they add into.
fn chain(c: &Ctx, node: u32) -> R<Chain> {
    let (mut at, mut links) = (node, vec![]);
    while is_chain(c, at)? {
        let operands = &c.node(at)?.ops;
        if operands.len() < 3 {
            return refuse("an add_at has fewer than three operands");
        }
        links.push([operands[1], operands[2]]);
        at = operands[0];
    }
    Ok(Chain {
        base: at,
        links: links.iter().rev().flatten().copied().collect(),
    })
}

/// Whether the node is a literal zero scalar of a floating dtype.
fn is_zero_literal(c: &Ctx, node: u32) -> R<bool> {
    let record = c.node(node)?;
    if !is_a(c, node, "Const")
        || !record.shape.is_empty()
        || !matches!(record.dtype.as_str(), "f64" | "f32")
    {
        return Ok(false);
    }
    Ok(match c.field(node, "literal")? {
        Val::Int(i) => i == 0,
        Val::Float(f) => f == 0.0,
        _ => false,
    })
}

/// A `vec` or `splat` of zero literals (a vec with no parts counts as zero).
fn is_structural_zero(c: &Ctx, node: u32) -> R<bool> {
    let record = c.node(node)?;
    if !is_a(c, node, "Op") || !matches!(record.dtype.as_str(), "f64" | "f32" | "bool") {
        return Ok(false);
    }
    let parts: Vec<u32> = if in_kinds(c, node, |kind| kind == "vec")? {
        record.ops.clone()
    } else if in_kinds(c, node, |kind| kind == "splat")? {
        record.ops.first().copied().into_iter().collect()
    } else {
        return Ok(false);
    };
    for part in parts {
        if !is_zero_literal(c, part)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The base of an `add_at` chain in `*base` and its links as (index, value) pairs flattened into
/// `buf`, in the order they were added; `*n` counts the nodes written (twice the links). A node
/// that is no `add_at` is its own base with no links.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `base` is valid for
/// one write; `buf` is valid for `cap` nodes and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_slotwise_chain(
    ctx: *mut Ctx,
    node: u32,
    base: *mut u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let found = chain(c, node)?;
        if base.is_null() {
            return Err(Fault::BadArg("base is NULL".into()));
        }
        unsafe { *base = found.base }
        list_out(buf, cap, n, &found.links)
    })
}

/// Whether the node is a `vec` or `splat` of zero literals (`*out` is 0 or 1).
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `out` is valid for
/// one write.
#[no_mangle]
pub unsafe extern "C" fn rw_slotwise_is_structural_zero(
    ctx: *mut Ctx,
    node: u32,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| flag_out(out, is_structural_zero(c, node)?))
}

// ---- the rewrite

/// The refusal that leaves a node in its own form: a type the slot form cannot reproduce.
fn bail<T>() -> R<T> {
    refuse("slotwise: this node keeps its form")
}

/// The extents of node `n`.
fn shape_of(c: &Ctx, n: u32) -> R<Vec<i64>> {
    Ok(c.node(n)?.shape.clone())
}

/// The type of node `n` as a type value.
fn type_of(c: &Ctx, n: u32) -> R<Val> {
    let node = c.node(n)?;
    Ok(ttype_val(&node.dtype, &node.shape, node.tag.as_deref()))
}

/// Whether two nodes have one type.
fn same_type(c: &Ctx, a: u32, b: u32) -> R<bool> {
    let (x, y) = (c.node(a)?, c.node(b)?);
    Ok((&x.shape, &x.dtype, &x.tag) == (&y.shape, &y.dtype, &y.tag))
}

/// A rank-0 zero constant of `dtype`.
fn zero_const(c: &mut Ctx, dtype: &str) -> R<u32> {
    let given = vec![
        ("literal", Val::Float(0.0)),
        ("ttype", ttype_val(dtype, &[], None)),
    ];
    build(c, "Const", given)
}

/// The scalar parts of a `vec` or `splat` of floats (or bools), row-major.
fn slots(c: &Ctx, node: u32) -> R<Option<Vec<u32>>> {
    let record = c.node(node)?;
    let typed = FLOAT.contains(&record.dtype.as_str()) || record.dtype == "bool";
    if !is_a(c, node, "Op") || !typed {
        return Ok(None);
    }
    let width = usize::try_from(record.shape.iter().product::<i64>()).unwrap_or(0);
    Ok(match c.str_field(node, "kind")?.as_str() {
        "vec" => Some(record.ops.clone()),
        "splat" => record.ops.first().map(|&part| vec![part; width]),
        _ => None,
    })
}

/// Whether the node is a structural zero scalar.
fn is_zero(c: &Ctx, node: u32) -> R<bool> {
    is_zero_literal(c, node)
}

/// `parts` as a `vec` of `shape`; refuses a type the node `like` never had.
fn assemble(c: &mut Ctx, shape: &[i64], parts: &[u32], like: u32) -> R<u32> {
    let literal = if shape.len() >= 2 {
        Val::Tuple(shape.iter().map(|&d| Val::Int(i128::from(d))).collect())
    } else {
        Val::None
    };
    let out = c.make_op("vec", parts, &literal)?;
    if same_type(c, out, like)? {
        Ok(out)
    } else {
        bail()
    }
}

/// How one slot of an op is built from the slots of its operands.
#[derive(Clone, Copy)]
enum Builder {
    /// `a + b`, a zero folded away.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`, zero when either is.
    Mul,
    /// `a / b`, zero when `a` is.
    Div,
    /// `-a`.
    Neg,
    /// `cond ? a : b`.
    Where,
    /// A dispatch over the selector, pinned to the strategy.
    Dispatch(u32, &'static str),
}

/// The slot `args` build under `how`.
fn apply(c: &mut Ctx, how: Builder, args: &[u32]) -> R<u32> {
    match how {
        Builder::Add => add_(c, args[0], args[1]),
        Builder::Sub => sub_(c, args[0], args[1]),
        Builder::Mul => mul_(c, args[0], args[1]),
        Builder::Div => div_(c, args[0], args[1]),
        Builder::Neg => neg_(c, args[0]),
        Builder::Where => where_(c, args[0], args[1], args[2]),
        Builder::Dispatch(selector, strategy) => {
            let same = args.iter().all(|&p| p == args[0]);
            let zeros = args.iter().map(|&p| is_zero(c, p)).collect::<R<Vec<_>>>()?;
            if same || zeros.iter().all(|&z| z) {
                return Ok(args[0]);
            }
            dispatch_of(c, selector, args, strategy, args[0])
        }
    }
}

/// `vec(build(slot k of each operand) ...)` when every rank>=1 operand is a `vec`/`splat` of the node's shape (a rank-0 one
/// broadcasts).
fn per_slot(c: &mut Ctx, node: u32, ops: &[u32], how: Builder) -> R<Option<u32>> {
    let record = c.node(node)?;
    let (shape, float, tagged) = (
        record.shape.clone(),
        FLOAT.contains(&record.dtype.as_str()),
        record.tag.is_some(),
    );
    if shape.is_empty() || !float || tagged {
        return Ok(None);
    }
    let mut columns: Vec<Option<Vec<u32>>> = vec![];
    for &op in ops {
        if shape_of(c, op)?.is_empty() {
            columns.push(None);
            continue;
        }
        let found = if shape_of(c, op)? == shape {
            slots(c, op)?
        } else {
            None
        };
        match found {
            Some(column) => columns.push(Some(column)),
            None => return Ok(None),
        }
    }
    if columns.iter().all(Option::is_none) {
        return Ok(None);
    }
    let width = usize::try_from(shape.iter().product::<i64>()).unwrap_or(0);
    let mut parts = vec![];
    for k in 0..width {
        let args: Vec<u32> = ops
            .iter()
            .zip(&columns)
            .map(|(&op, column)| column.as_ref().map_or(op, |col| col[k]))
            .collect();
        parts.push(apply(c, how, &args)?);
    }
    assemble(c, &shape, &parts, node).map(Some)
}

/// `a + b` with a structural zero folded away.
fn add_(c: &mut Ctx, a: u32, b: u32) -> R<u32> {
    if is_zero(c, a)? {
        return Ok(b);
    }
    if is_zero(c, b)? {
        return Ok(a);
    }
    c.make_op("add", &[a, b], &Val::None)
}

/// `a - b` with a structural zero folded away.
fn sub_(c: &mut Ctx, a: u32, b: u32) -> R<u32> {
    if is_zero(c, b)? {
        return Ok(a);
    }
    if is_zero(c, a)? {
        return neg_(c, b);
    }
    c.make_op("sub", &[a, b], &Val::None)
}

/// `-a`, a zero staying itself.
fn neg_(c: &mut Ctx, a: u32) -> R<u32> {
    if is_zero(c, a)? {
        return Ok(a);
    }
    c.make_op("neg", &[a], &Val::None)
}

/// `a * b`, zero when either factor is.
fn mul_(c: &mut Ctx, a: u32, b: u32) -> R<u32> {
    if is_zero(c, a)? {
        return Ok(a);
    }
    if is_zero(c, b)? {
        return Ok(b);
    }
    c.make_op("mul", &[a, b], &Val::None)
}

/// `a / b`, zero when `a` is.
fn div_(c: &mut Ctx, a: u32, b: u32) -> R<u32> {
    if is_zero(c, a)? {
        return Ok(a);
    }
    c.make_op("div", &[a, b], &Val::None)
}

/// `cond ? a : b` of the type of `a`; one arm when both are the same node or both zero.
fn where_(c: &mut Ctx, cond: u32, a: u32, b: u32) -> R<u32> {
    if a == b || (is_zero(c, a)? && is_zero(c, b)?) {
        return Ok(a);
    }
    if !same_type(c, a, b)? {
        return bail();
    }
    select_of(c, cond, a, b, a)
}

/// A select of the arms, typed as the node `like`.
fn select_of(c: &mut Ctx, cond: u32, on_true: u32, on_false: u32, like: u32) -> R<u32> {
    let given = vec![
        ("cond", Val::Node(cond)),
        ("on_true", Val::Node(on_true)),
        ("on_false", Val::Node(on_false)),
        ("ttype", type_of(c, like)?),
    ];
    build(c, "Select", given)
}

/// The strategy word a dispatch is pinned to, from its feathers: a plain dispatch is the `switch` form.
fn strategy_of(c: &Ctx, node: u32) -> R<&'static str> {
    for (feather, word) in [
        ("splits_by_branch", "segmented"),
        ("counts_branch_mix", "auto"),
        ("evaluates_all_branches", "predicated"),
    ] {
        if c.has_feather(node, feather)? {
            return Ok(word);
        }
    }
    Ok("switch")
}

/// A dispatch of `selector` over `branches` pinned to `strategy` (the core kind for `switch`, else the wing's kind), typed as the
/// node `like`.
fn dispatch_of(c: &mut Ctx, selector: u32, branches: &[u32], strategy: &str, like: u32) -> R<u32> {
    let given = vec![
        ("selector", Val::Node(selector)),
        (
            "branches",
            Val::Tuple(branches.iter().map(|&b| Val::Node(b)).collect()),
        ),
        ("ttype", type_of(c, like)?),
    ];
    let core = build(c, "Dispatch", given)?;
    if strategy == "switch" {
        return Ok(core);
    }
    let record = c.node(core)?;
    let (ops, dtype) = (record.ops.clone(), record.dtype.clone());
    let (shape, tag) = (record.shape.clone(), record.tag.clone());
    let class = format!("{}Dispatch", strategy[..1].to_uppercase() + &strategy[1..]);
    c.make(&class, &dtype, &shape, tag.as_deref(), &ops, "()")
}

/// The slot form of one node over its already rewritten operands, or none; a refusal leaves the node in its own form.
fn rewrite(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    match rewrite_node(c, node, ops) {
        Err(Fault::Refused(_)) => Ok(None),
        other => other,
    }
}

/// The rules of `rewrite`, which lets a refusal through.
fn rewrite_node(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    for &op in ops {
        if is_chain(c, op)? {
            return runtime(c, node, ops);
        }
    }
    if is_a(c, node, "Dispatch") {
        return dispatch_rule(c, node, ops);
    }
    if is_a(c, node, "Select") {
        return per_slot(c, node, ops, Builder::Where);
    }
    match op_kind(c, node)?.as_deref() {
        Some("component") => component(c, node, ops),
        Some("sum") => sum(c, node, ops),
        Some(kind @ ("add" | "sub" | "mul" | "div" | "neg")) => elementwise(c, node, ops, kind),
        _ => Ok(None),
    }
}

/// An elementwise op per slot.
fn elementwise(c: &mut Ctx, node: u32, ops: &[u32], kind: &str) -> R<Option<u32>> {
    let how = match kind {
        "add" => Builder::Add,
        "sub" => Builder::Sub,
        "mul" => Builder::Mul,
        "div" => Builder::Div,
        _ => Builder::Neg,
    };
    per_slot(c, node, ops, how)
}

/// A dispatch per slot: one dispatch per slot with the same selector and strategy.
fn dispatch_rule(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    let (selector, branches) = (ops[0], &ops[1..]);
    let shape = shape_of(c, node)?;
    for &b in branches {
        if shape_of(c, b)? != shape {
            return Ok(None);
        }
    }
    let how = Builder::Dispatch(selector, strategy_of(c, node)?);
    per_slot(c, node, branches, how)
}

/// The slot a `component` reads, when its operand is a `vec` or `splat`.
fn component(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    let Some(parts) = slots(c, ops[0])? else {
        return Ok(None);
    };
    let shape = shape_of(c, ops[0])?;
    let at = match c.field(node, "literal")? {
        Val::Int(i) => i,
        Val::Tuple(at) if at.len() == shape.len() => {
            let mut flat = 0;
            for (k, &extent) in at.iter().zip(&shape) {
                let Val::Int(k) = k else { return bail() };
                flat = flat * i128::from(extent) + k;
            }
            flat
        }
        _ => return bail(),
    };
    let part = usize::try_from(at).ok().and_then(|i| parts.get(i).copied());
    match part {
        Some(p) if same_type(c, p, node)? => Ok(Some(p)),
        _ => bail(),
    }
}

/// The sum of a rank-1 `vec`, left-nested `add` in the language's own fold order.
fn sum(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    let parts = slots(c, ops[0])?;
    let Some(parts) = parts.filter(|_| shape_of(c, ops[0]).is_ok_and(|s| s.len() == 1)) else {
        return Ok(None);
    };
    let Some((&first, rest)) = parts.split_first() else {
        return bail();
    };
    let mut acc = first;
    for &part in rest {
        acc = add_(c, acc, part)?;
    }
    if same_type(c, acc, node)? {
        Ok(Some(acc))
    } else {
        bail()
    }
}

// ---- the runtime form: an add_at chain over a slot-form base

/// Whether the node is an `add_at`.
fn is_chain(c: &Ctx, node: u32) -> R<bool> {
    Ok(in_kinds(c, node, |kind| kind == "add_at")? && one_index(c, node))
}

/// The base of the chain under `node` and its links in order.
fn chain_of(c: &Ctx, node: u32) -> R<(u32, Vec<(u32, u32)>)> {
    let found = chain(c, node)?;
    Ok((
        found.base,
        found.links.chunks(2).map(|l| (l[0], l[1])).collect(),
    ))
}

/// `base` with every link added in order.
fn rechain(c: &mut Ctx, base: u32, links: &[(u32, u32)]) -> R<u32> {
    let mut at = base;
    for &(k, e) in links {
        at = c.make_op("add_at", &[at, k, e], &Val::None)?;
    }
    Ok(at)
}

/// The slots of `node` with every link of its chain as a select on the index.
fn lower(c: &mut Ctx, node: u32) -> R<Option<Vec<u32>>> {
    let (base, links) = chain_of(c, node)?;
    let Some(mut out) = slots(c, base)? else {
        return Ok(None);
    };
    let dtype = c.node(node)?.dtype.clone();
    let zero = zero_const(c, &dtype)?;
    for (k, e) in links {
        for (slot, part) in out.iter_mut().enumerate() {
            let number = int_like(c, slot as i64, k)?;
            let hit = c.make_op("eq", &[k, number], &Val::None)?;
            let chosen = select_of(c, hit, e, zero, e)?;
            *part = add_(c, *part, chosen)?;
        }
    }
    Ok(Some(out))
}

/// A chain lowered and assembled back into a `vec`.
fn lowered_chain(c: &mut Ctx, op: u32) -> R<Option<u32>> {
    let Some(parts) = lower(c, op)? else {
        return Ok(None);
    };
    let shape = shape_of(c, op)?;
    assemble(c, &shape, &parts, op).map(Some)
}

/// `node` over operands of which at least one is a chain.
fn runtime(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    if is_chain(c, node)? {
        return Ok(None); // a longer chain: already the form
    }
    if is_a(c, node, "Dispatch") {
        return runtime_dispatch(c, node, ops);
    }
    if is_a(c, node, "Select") {
        return runtime_select(c, node, ops);
    }
    let Some(kind) = op_kind(c, node)? else {
        return Ok(None);
    };
    match kind.as_str() {
        "component" => {
            let Some(parts) = lower(c, ops[0])? else {
                return Ok(None);
            };
            let shape = shape_of(c, ops[0])?;
            let whole = assemble(c, &shape, &parts, ops[0])?;
            component(c, node, &[whole])
        }
        "sum" => runtime_sum(c, node, ops[0]),
        "add" | "sub" | "neg" => runtime_linear(c, node, ops, &kind),
        "mul" | "div" => runtime_scale(c, node, ops, &kind),
        _ => lowered(c, node, ops),
    }
}

/// The sum of a chain: the sum of its base, then each link's value added.
fn runtime_sum(c: &mut Ctx, node: u32, operand: u32) -> R<Option<u32>> {
    let (base, links) = chain_of(c, operand)?;
    let mut acc = if slots(c, base)?.is_some() {
        match sum(c, node, &[base])? {
            Some(s) => s,
            None => return Ok(None),
        }
    } else {
        c.make_op("sum", &[base], &Val::None)?
    };
    for (_, e) in links {
        acc = add_(c, acc, e)?;
    }
    Ok(Some(acc))
}

/// `add`, `sub` or `neg` over chains: the op over the bases, then the links (negated on the subtracted side).
fn runtime_linear(c: &mut Ctx, node: u32, ops: &[u32], kind: &str) -> R<Option<u32>> {
    let mut bases = vec![];
    let mut links: Vec<(u32, u32)> = vec![];
    for (i, &op) in ops.iter().enumerate() {
        let (base, own) = chain_of(c, op)?;
        bases.push(base);
        for (k, e) in own {
            let value = if kind != "add" && (kind == "neg" || i == 1) {
                neg_(c, e)?
            } else {
                e
            };
            links.push((k, value));
        }
    }
    let inner = match elementwise(c, node, &bases, kind)? {
        Some(n) => n,
        None => c.make_op(kind, &bases, &Val::None)?,
    };
    rechain(c, inner, &links).map(Some)
}

/// `mul` or `div` of a chain by a rank-0 value (or a rank-0 value times a chain): the scale of each link.
fn runtime_scale(c: &mut Ctx, node: u32, ops: &[u32], kind: &str) -> R<Option<u32>> {
    let (a, b) = (ops[0], ops[1]);
    let a_chain = is_chain(c, a)? && shape_of(c, b)?.is_empty();
    let b_chain = kind == "mul" && is_chain(c, b)? && shape_of(c, a)?.is_empty();
    if !a_chain && !b_chain {
        return lowered(c, node, ops);
    }
    let (base, links) = chain_of(c, if a_chain { a } else { b })?;
    let pair = if a_chain { [base, b] } else { [a, base] };
    let inner = match elementwise(c, node, &pair, kind)? {
        Some(n) => n,
        None => c.make_op(kind, &pair, &Val::None)?,
    };
    let mut scaled = vec![];
    for (k, e) in links {
        let value = match (a_chain, kind) {
            (true, "mul") => mul_(c, e, b)?,
            (true, _) => div_(c, e, b)?,
            (false, _) => mul_(c, a, e)?,
        };
        scaled.push((k, value));
    }
    rechain(c, inner, &scaled).map(Some)
}

/// A rank-0 select of chains: the select of the bases, each link gated by the condition.
fn runtime_select(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    let (cond, a, b) = (ops[0], ops[1], ops[2]);
    let shape = shape_of(c, node)?;
    if !shape_of(c, cond)?.is_empty() || shape_of(c, a)? != shape {
        return lowered(c, node, ops);
    }
    let (a_base, a_links) = chain_of(c, a)?;
    let (b_base, b_links) = chain_of(c, b)?;
    let dtype = c.node(node)?.dtype.clone();
    let zero = zero_const(c, &dtype)?;
    let mut links = vec![];
    for (k, e) in a_links {
        links.push((k, gate(c, cond, e, zero, e)?));
    }
    for (k, e) in b_links {
        links.push((k, gate(c, cond, zero, e, e)?));
    }
    let inner = match per_slot(c, node, &[cond, a_base, b_base], Builder::Where)? {
        Some(n) => n,
        None => select_of(c, cond, a_base, b_base, node)?,
    };
    rechain(c, inner, &links).map(Some)
}

/// `cond ? on_true : on_false` where `value` is the link value in it. A link value that is a dispatch takes the gate into its
/// cases, so it stays a member of its switch.
fn gate(c: &mut Ctx, cond: u32, on_true: u32, on_false: u32, value: u32) -> R<u32> {
    if !is_a(c, value, "Dispatch") {
        return select_of(c, cond, on_true, on_false, value);
    }
    let zero = if value == on_true { on_false } else { on_true };
    let (selector, branches) = {
        let ops = &c.node(value)?.ops;
        (ops[0], ops[1..].to_vec())
    };
    let mut gated = vec![];
    for b in branches {
        gated.push(if value == on_true {
            where_(c, cond, b, zero)?
        } else {
            where_(c, cond, zero, b)?
        });
    }
    let strategy = strategy_of(c, value)?;
    dispatch_of(c, selector, &gated, strategy, value)
}

/// `node` over its operands with every chain lowered to slots.
fn lowered(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    let mut fresh = vec![];
    for &op in ops {
        if is_chain(c, op)? {
            match lowered_chain(c, op)? {
                Some(whole) => fresh.push(whole),
                None => return Ok(None),
            }
        } else {
            fresh.push(op);
        }
    }
    match rewrite(c, node, &fresh)? {
        Some(done) => Ok(Some(done)),
        None => c.rebuild(node, &fresh).map(Some),
    }
}

/// A dispatch of chains: each case's links become members of the same switch (a case without the link carries index 0 and
/// value 0, a no-op add).
fn runtime_dispatch(c: &mut Ctx, node: u32, ops: &[u32]) -> R<Option<u32>> {
    let (selector, branches) = (ops[0], &ops[1..]);
    let shape = shape_of(c, node)?;
    for &b in branches {
        if shape_of(c, b)? != shape {
            return Ok(None);
        }
    }
    let strategy = strategy_of(c, node)?;
    let (mut bases, mut links) = (vec![], vec![]);
    for (i, &b) in branches.iter().enumerate() {
        let (base, own) = chain_of(c, b)?;
        bases.push(base);
        for (k, e) in own {
            let zero_k = int_like(c, 0, k)?;
            let dtype = c.node(e)?.dtype.clone();
            let zero_e = zero_const(c, &dtype)?;
            let ks: Vec<u32> = (0..branches.len())
                .map(|j| if j == i { k } else { zero_k })
                .collect();
            let es: Vec<u32> = (0..branches.len())
                .map(|j| if j == i { e } else { zero_e })
                .collect();
            links.push((
                dispatch_of(c, selector, &ks, strategy, k)?,
                dispatch_of(c, selector, &es, strategy, e)?,
            ));
        }
    }
    let mut all = vec![selector];
    all.extend(&bases);
    let inner = match dispatch_rule(c, node, &all)? {
        Some(n) => n,
        None => dispatch_of(c, selector, &bases, strategy, node)?,
    };
    rechain(c, inner, &links).map(Some)
}

// ---- the pass

/// `roots` in slotwise form, one for one. A root may be a value, a sink (rebuilt over its rewritten value) or a loop.
pub fn slotwise(c: &mut Ctx, roots: &[u32]) -> R<Vec<u32>> {
    let mut memo: HashMap<u32, u32> = HashMap::new();
    for &root in roots {
        walk(c, root, &mut memo)?;
    }
    let mut out = vec![];
    for &root in roots {
        out.push(if is_a(c, root, "Sink") {
            resink(c, root, &memo)?
        } else {
            memo[&root]
        });
    }
    Ok(out)
}

/// Fills `memo` (node to rewritten node) for everything `root` reaches.
fn walk(c: &mut Ctx, root: u32, memo: &mut HashMap<u32, u32>) -> R<()> {
    let mut stack = vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if memo.contains_key(&node) {
            continue;
        }
        let operands = c.node(node)?.ops.clone();
        if is_a(c, node, "Loop") || operands.is_empty() {
            memo.insert(node, node);
            continue;
        }
        if !expanded {
            stack.push((node, true));
            stack.extend(
                operands
                    .iter()
                    .rev()
                    .filter(|o| !memo.contains_key(o))
                    .map(|&o| (o, false)),
            );
            continue;
        }
        if is_a(c, node, "Sink") {
            memo.insert(node, node); // a sink is rebuilt by `resink`
            continue;
        }
        let new: Vec<u32> = operands.iter().map(|o| memo[o]).collect();
        let got = match rewrite(c, node, &new)? {
            Some(n) => n,
            None if new != operands => c.rebuild(node, &new)?,
            None => node,
        };
        memo.insert(node, got);
    }
    Ok(())
}

/// A sink over the rewritten values of its operands; a sink of a kind that holds no value is left as it is.
fn resink(c: &mut Ctx, sink: u32, memo: &HashMap<u32, u32>) -> R<u32> {
    let operands = c.node(sink)?.ops.clone();
    let new: Vec<u32> = operands.iter().map(|o| memo[o]).collect();
    let rebuilt = ["SampleSum", "Assign", "WideWrite", "AccumWrite"]
        .iter()
        .any(|class| is_a(c, sink, class));
    if new == operands || !rebuilt {
        return Ok(sink);
    }
    c.rebuild(sink, &new)
}

/// The roots in slotwise form, one for one; see the module documentation.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `roots` is valid for `nroots` nodes, `buf` for `cap`
/// nodes and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_slotwise(
    ctx: *mut Ctx,
    roots: *const u32,
    nroots: usize,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let given = if nroots == 0 {
            &[][..]
        } else if roots.is_null() {
            return Err(Fault::BadArg("roots is NULL".into()));
        } else {
            unsafe { std::slice::from_raw_parts(roots, nroots) }
        };
        for &r in given {
            c.node(r)?;
        }
        let done = slotwise(c, given)?;
        list_out(buf, cap, n, &done)
    })
}
