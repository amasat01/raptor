//! Questions about loops of the lowered graph that the writers ask when they cut a loop into steps.
use crate::{flag_out, flat, in_kinds, is_a, switch_form};
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::ffi::c_int;

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// Integer division and remainders: not safe to run for a lane that has stopped.
const INTEGER_DIVISIONS: [&str; 5] = ["div", "fmod", "remainder", "floordiv", "mod"];

/// The nodes held by a loop field that is a list of nodes (empty when the field is empty).
fn node_list(ctx: &Ctx, loop_node: u32, field: &str) -> Outcome<Vec<u32>> {
    match ctx.field(loop_node, field)? {
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .map(|item| match item {
                Val::Node(node) => Ok(*node),
                _ => refuse(format!("loop field {field:?} holds a non-node")),
            })
            .collect(),
        _ => refuse(format!("loop field {field:?} is not a list")),
    }
}

/// Whether the node makes a trip unsafe to run and discard for a stopped lane: an indexed read,
/// a rendered switch or an integer division.
fn blocks_speculation(ctx: &Ctx, node: u32) -> Outcome<bool> {
    if is_a(ctx, node, "At") || is_a(ctx, node, "TapeRead") {
        return Ok(true);
    }
    if is_a(ctx, node, "Dispatch") && switch_form(ctx, node)? {
        return Ok(true);
    }
    let integer_division = kind_is(ctx, node, |kind| INTEGER_DIVISIONS.contains(&kind))?
        && matches!(ctx.node(node)?.dtype.as_str(), "i32" | "i64");
    Ok(integer_division)
}

/// Whether the node is an op whose kind passes the test.
fn kind_is(ctx: &Ctx, node: u32, test: impl Fn(&str) -> bool) -> Outcome<bool> {
    in_kinds(ctx, node, test)
}

/// Whether one trip of the loop is safe to run and discard for a stopped lane: nothing in its body,
/// its carried values or its break condition reads at a computed index, switches, holds a raw
/// block or divides integers.
fn speculative_step(ctx: &Ctx, loop_node: u32) -> Outcome<bool> {
    ctx.node(loop_node)?;
    if !is_a(ctx, loop_node, "Loop") {
        return refuse(format!("node {loop_node} is not a Loop"));
    }
    let mut nodes = vec![];
    flat(ctx, &ctx.body_nodes(loop_node)?, &mut nodes)?;
    nodes.extend(node_list(ctx, loop_node, "nexts")?);
    if let Val::Node(exit) = ctx.field(loop_node, "exit_cond")? {
        nodes.push(exit);
    }
    for node in nodes {
        if blocks_speculation(ctx, node)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether one trip of the loop is safe to run speculatively; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_loops_speculative_step(
    ctx: *mut Ctx,
    loop_node: u32,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |ctx| flag_out(out, speculative_step(ctx, loop_node)?))
}

/// The given nodes and everything inside their loop bodies, in order; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_loops_flat(
    ctx: *mut Ctx,
    nodes: *const u32,
    n: usize,
    buf: *mut u32,
    cap: usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |ctx| {
        if n > 0 && nodes.is_null() {
            return Err(Fault::BadArg("nodes is NULL".into()));
        }
        let given = if n == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(nodes, n) }
        };
        let mut all = vec![];
        flat(ctx, given, &mut all)?;
        crate::list_out(buf, cap, count, &all)
    })
}
