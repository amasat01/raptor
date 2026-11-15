//! The analyses a derivative pass asks of the slotwise ops it builds: the chain of `add_at` links
//! under a node and whether a node is a vector of zeros.
use crate::{flag_out, in_kinds, is_a, list_out, R};
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::ffi::c_int;

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
    while in_kinds(c, at, |kind| kind == "add_at")? {
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
