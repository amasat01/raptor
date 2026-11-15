//! The lowering of the least-squares kinds (`lstsq`, `lstsq_gram_solve`, `lstsq_info`): a node
//! becomes the normal equations (a Cholesky factor of `J^T J`) or Householder QR, by the largest
//! dimension one lane factors alone. The nodes are made in the caller's context.
use crate::construct::refs;
use crate::{list_out, op_kind};
use raptor_ir::build::LoopParts;
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::HashMap;
use std::ffi::c_int;

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// The kinds of the least-squares nodes.
const LSTSQ_KINDS: [&str; 3] = ["lstsq", "lstsq_gram_solve", "lstsq_info"];

/// A new op of a kind over the operands, its type inferred from them.
fn make(ctx: &mut Ctx, kind: &str, operands: &[u32]) -> Outcome<u32> {
    ctx.make_op(kind, operands, &Val::None)
}

/// `J^T J`.
fn gram(ctx: &mut Ctx, j: u32) -> Outcome<u32> {
    let transposed = make(ctx, "transpose", &[j])?;
    make(ctx, "mm", &[transposed, j])
}

/// The nodes a least-squares node over operands `operands` is lowered to, for QR or the normal equations.
fn lower_node(ctx: &mut Ctx, kind: &str, operands: &[u32], use_qr: bool) -> Outcome<u32> {
    let j = operands[0];
    if kind == "lstsq_info" {
        return if use_qr {
            let factor = make(ctx, "qr", &[j])?;
            make(ctx, "qr_info", &[factor])
        } else {
            let factor = gram(ctx, j).and_then(|g| make(ctx, "chol", &[g]))?;
            make(ctx, "chol_info", &[factor])
        };
    }
    let b = *operands.get(1).ok_or(Fault::Refused(
        "a least-squares solve has no right-hand side".into(),
    ))?;
    if use_qr {
        let factor = make(ctx, "qr", &[j])?;
        return make(
            ctx,
            if kind == "lstsq" {
                "qr_solve"
            } else {
                "qr_gram_solve"
            },
            &[factor, b],
        );
    }
    let rhs = if kind == "lstsq" {
        let transposed = make(ctx, "transpose", &[j])?;
        let product = if ctx.node(b)?.shape.len() == 1 {
            "mv"
        } else {
            "mm"
        };
        make(ctx, product, &[transposed, b])?
    } else {
        b
    };
    let factor = gram(ctx, j).and_then(|g| make(ctx, "chol", &[g]))?;
    make(ctx, "cho_solve", &[factor, rhs])
}

/// The children of a node: its operands, and for a loop also the roots of its body.
fn children(ctx: &Ctx, node: u32) -> Outcome<Vec<u32>> {
    let mut found = ctx.nodes[node as usize].ops.clone();
    if ctx.table.is_a(ctx.nodes[node as usize].class, "Loop") {
        for field in ["nexts", "exit_values", "body_sinks"] {
            if let Val::Tuple(items) | Val::List(items) = ctx.field(node, field)? {
                found.extend(items.iter().filter_map(|v| {
                    if let Val::Node(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                }));
            }
        }
        if let Val::Node(exit) = ctx.field(node, "exit_cond")? {
            found.push(exit);
        }
    }
    Ok(found)
}

/// The loop over the lowered values of its inits, trip count, carried results, exit and commits;
/// the loop constructor finds the boundary again.
fn loop_over(ctx: &mut Ctx, node: u32, memo: &HashMap<u32, u32>) -> Outcome<u32> {
    let done = |nodes: Vec<u32>| -> Vec<u32> { nodes.iter().map(|n| memo[n]).collect() };
    let carries = refs(ctx, node, "carries")?;
    let index = refs(ctx, node, "index")?[0];
    let operands = ctx.nodes[node as usize].ops.clone();
    let inits = done(operands[..carries.len()].to_vec());
    let nexts = done(refs(ctx, node, "nexts")?);
    let sinks = done(refs(ctx, node, "body_sinks")?);
    let exit_values = done(refs(ctx, node, "exit_values")?);
    let exit_cond = refs(ctx, node, "exit_cond")?.first().map(|n| memo[n]);
    let count = if ctx.field(node, "count")? == Val::None {
        None
    } else {
        operands.last().map(|n| memo[n])
    };
    let int = |name: &str| match ctx.field(node, name) {
        Ok(Val::Int(i)) => Ok(i as i64),
        _ => refuse("a loop bound is an int"),
    };
    let range = (int("start")?, int("stop")?, int("step")?);
    let dynamic = match ctx.field(node, "dynamic")? {
        Val::Str(text) => Some(text),
        _ => None,
    };
    let count_tail = ctx.field(node, "count_tail")? == Val::Bool(true);
    let parts = LoopParts {
        index,
        carries: &carries,
        inits: &inits,
        nexts: &nexts,
        range,
        body_sinks: &sinks,
        exit_cond,
        exit_values: &exit_values,
        count,
        count_tail,
        dynamic: dynamic.as_deref(),
    };
    ctx.build_loop(&parts)
}

/// The sinks with every least-squares node lowered; the same sinks when there is none. `max_dim`
/// is the largest dimension one lane factors alone (refused when the target states none).
fn lowered_sinks(ctx: &mut Ctx, sinks: &[u32], max_dim: Option<i64>) -> Outcome<Vec<u32>> {
    let mut memo: HashMap<u32, u32> = HashMap::new();
    let mut stack: Vec<u32> = sinks.to_vec();
    while let Some(&node) = stack.last() {
        if memo.contains_key(&node) {
            stack.pop();
            continue;
        }
        let kids = children(ctx, node)?;
        let todo: Vec<u32> = kids
            .iter()
            .copied()
            .filter(|k| !memo.contains_key(k))
            .collect();
        if !todo.is_empty() {
            stack.extend(todo);
            continue;
        }
        stack.pop();
        if ctx.table.is_a(ctx.nodes[node as usize].class, "Loop") {
            let made = if kids.iter().all(|k| memo[k] == *k) {
                node
            } else {
                loop_over(ctx, node, &memo)?
            };
            memo.insert(node, made);
            continue;
        }
        let new: Vec<u32> = ctx.nodes[node as usize]
            .ops
            .iter()
            .map(|o| memo[o])
            .collect();
        let kind = op_kind(ctx, node)?;
        let lowered = match kind.as_deref().filter(|k| LSTSQ_KINDS.contains(k)) {
            Some(kind) => {
                let Some(limit) = max_dim else {
                    return refuse("a least-squares solve needs the target wing's linalg.max_dim, and this target's wing has none");
                };
                let columns = ctx.node(new[0])?.shape.get(1).copied().unwrap_or(0);
                lower_node(ctx, kind, &new, columns > limit)?
            }
            None if new == ctx.nodes[node as usize].ops => node,
            None => ctx.rebuild(node, &new)?,
        };
        memo.insert(node, lowered);
    }
    Ok(sinks.iter().map(|s| memo[s]).collect())
}

/// The sinks with the least-squares nodes lowered; see `raptor_wings.h`. `*changed` is 0 when the
/// sinks are returned as given.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_lstsq_lowered(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    max_dim: *const i64,
    buf: *mut u32,
    cap: usize,
    count: *mut usize,
    changed: *mut c_int,
) -> c_int {
    guard(ctx, |ctx| {
        if (nsinks > 0 && sinks.is_null()) || changed.is_null() {
            return Err(Fault::BadArg("sinks or changed is NULL".into()));
        }
        if cap < nsinks || (buf.is_null() && nsinks > 0) {
            // nothing is made until the caller's buffer can hold the answer
            return list_out(std::ptr::null_mut(), 0, count, &vec![0; nsinks]);
        }
        let given = if nsinks == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        let limit = if max_dim.is_null() {
            None
        } else {
            Some(unsafe { *max_dim })
        };
        let lowered = lowered_sinks(ctx, given, limit)?;
        unsafe { *changed = (lowered != given) as c_int }
        list_out(buf, cap, count, &lowered)
    })
}
