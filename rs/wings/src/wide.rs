//! Questions about wide (vector) values: which nodes name planes, how element `k` of a vector is
//! read without building the whole, and vectors that are runs of reads of one plane.
use crate::{flag_out, in_kinds, is_a, unclamped, PLANE_PREFIX, RESIDENT_KIND};
use raptor_ir::door::guard;
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::HashMap;
use std::ffi::c_int;

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// Operations that act on each element alone.
const ELEMENTWISE: [&str; 9] = [
    "add", "sub", "mul", "div", "neg", "min", "max", "abs", "select",
];

/// A leaf that reads a vocabulary plane: its role, or None for any other node.
fn plane_role(ctx: &Ctx, node: u32) -> Outcome<Option<String>> {
    if is_a(ctx, node, "Leaf") && ctx.str_field(node, "kind")? == "vocab_read" {
        return ctx.str_field(node, "role").map(Some);
    }
    Ok(None)
}

/// Whether the node names a scratch plane the runtime made (flat or warp-tiled).
pub(crate) fn is_tiled(ctx: &Ctx, node: u32) -> Outcome<bool> {
    ctx.node(node)?;
    if plane_role(ctx, node)?.is_none_or(|role| role != "wide_in") {
        return Ok(false);
    }
    Ok(ctx.str_field(node, "name")?.starts_with(PLANE_PREFIX))
}

/// Whether the node names a plane a placement writes: a mutable's own, or a scratch plane.
pub(crate) fn is_scratch(ctx: &Ctx, node: u32) -> Outcome<bool> {
    ctx.node(node)?;
    match plane_role(ctx, node)? {
        Some(role) => Ok(role == "mutable" || is_tiled(ctx, node)?),
        None => Ok(false),
    }
}

/// Whether the node is the launch-start value of a vector mutable.
pub(crate) fn is_prior(ctx: &Ctx, node: u32) -> Outcome<bool> {
    let leaf = ctx.node(node)?;
    let rank_one = leaf.shape.len() == 1;
    Ok(is_a(ctx, node, "Leaf")
        && ctx.str_field(node, "kind")? == "prior_read"
        && ctx.str_field(node, "role")? == "mutable"
        && rank_one)
}

/// The rank of the value a node computes.
pub(crate) fn rank(ctx: &Ctx, node: u32) -> usize {
    ctx.nodes[node as usize].shape.len()
}

/// How element `k` of a rank-1 node is read without building the whole: `proxy`, `input`,
/// `parts`, `splat`, `elementwise`, else `none`.
pub(crate) fn element_kind(ctx: &Ctx, node: u32) -> Outcome<&'static str> {
    ctx.node(node)?;
    if in_kinds(ctx, node, |kind| kind == RESIDENT_KIND)? || is_scratch(ctx, node)? {
        return Ok("proxy");
    }
    if is_a(ctx, node, "Leaf") {
        let vector_input = ctx.str_field(node, "role")? == "plane_in" && rank(ctx, node) == 1;
        return Ok(if vector_input { "input" } else { "none" });
    }
    let is_select = is_a(ctx, node, "Select");
    if !(is_select || is_a(ctx, node, "Op")) || rank(ctx, node) != 1 {
        return Ok("none");
    }
    let operands = &ctx.nodes[node as usize].ops;
    if is_select
        && operands
            .first()
            .is_some_and(|&condition| rank(ctx, condition) > 0)
    {
        return Ok("none");
    }
    let kind = if is_select {
        "select".to_string()
    } else {
        ctx.str_field(node, "kind")?
    };
    Ok(match kind.as_str() {
        "vec" => "parts",
        "splat" => "splat",
        k if ELEMENTWISE.contains(&k) && operands.iter().all(|&o| rank(ctx, o) <= 1) => {
            "elementwise"
        }
        _ => "none",
    })
}

/// The integer a node is made of constants alone (`add`, `sub`, `mul` of them), else None.
fn const_int(ctx: &Ctx, node: u32) -> Outcome<Option<i128>> {
    if is_a(ctx, node, "Const") {
        let integer = matches!(ctx.node(node)?.dtype.as_str(), "i32" | "i64");
        return Ok(match ctx.field(node, "literal")? {
            Val::Int(i) if integer => Some(i),
            Val::Float(f) if integer => Some(f as i128),
            _ => None,
        });
    }
    let kind = if is_a(ctx, node, "Op") {
        ctx.str_field(node, "kind")?
    } else {
        String::new()
    };
    let operands = &ctx.nodes[node as usize].ops;
    if !matches!(kind.as_str(), "add" | "sub" | "mul") || operands.len() != 2 {
        return Ok(None);
    }
    let (Some(a), Some(b)) = (const_int(ctx, operands[0])?, const_int(ctx, operands[1])?) else {
        return Ok(None);
    };
    Ok(Some(match kind.as_str() {
        "add" => a + b,
        "sub" => a - b,
        _ => a * b,
    }))
}

/// `(base, offset)` of an index `base + constants`; the base is None when the index is all constant.
fn split_index(ctx: &Ctx, node: u32) -> Outcome<(Option<u32>, i128)> {
    if let Some(whole) = const_int(ctx, node)? {
        return Ok((None, whole));
    }
    if is_a(ctx, node, "Op") && ctx.str_field(node, "kind")? == "add" {
        let operands = ctx.nodes[node as usize].ops.clone();
        for (a, b) in [(operands[0], operands[1]), (operands[1], operands[0])] {
            if let Some(offset) = const_int(ctx, b)? {
                let (base, more) = split_index(ctx, a)?;
                return Ok((base, more + offset));
            }
        }
    }
    Ok((Some(node), 0))
}

/// A vector whose element `k` is the read `plane[base + start + k * step]`.
pub struct Gather {
    /// The plane read.
    pub plane: u32,
    /// The run-time part of the index, when there is one.
    pub base: Option<u32>,
    /// The index of the first element, without the run-time part.
    pub start: i128,
    /// The distance between the indices of neighbouring elements.
    pub step: i128,
    /// Whether the reads keep the clamp of the original index.
    pub clamped: bool,
}

/// Whether the node is a scalar read (`At`) of a plane, and which plane (role, name).
fn scalar_read(ctx: &Ctx, node: u32) -> Outcome<Option<(String, String)>> {
    if !is_a(ctx, node, "At") || rank(ctx, node) != 0 {
        return Ok(None);
    }
    let plane = ctx.nodes[node as usize].ops[0];
    Ok(Some((
        ctx.str_field(plane, "role")?,
        ctx.str_field(plane, "name")?,
    )))
}

/// The gather a `vec` of reads of one plane at an arithmetic run of indices is, else None.
pub(crate) fn gather(ctx: &Ctx, node: u32) -> Outcome<Option<Gather>> {
    ctx.node(node)?;
    let parts = ctx.nodes[node as usize].ops.clone();
    if !(in_kinds(ctx, node, |kind| kind == "vec")? && rank(ctx, node) == 1 && parts.len() > 1) {
        return Ok(None);
    }
    let first = parts[0];
    let Some(plane_id) = scalar_read(ctx, first)? else {
        return Ok(None);
    };
    // a gather is an arithmetic run of one flat index: a read with an index per axis of several axes is not one
    let sole_index = |read: u32| match ctx.nodes[read as usize].ops[..] {
        [_, index] => Some(index),
        _ => None,
    };
    // the index of an old-form read carries its clamp as an operation: the run is of the index before it
    let raw_index = |read: u32| sole_index(read).map(|index| unclamped(ctx, index));
    let Some(first_index) = raw_index(first) else {
        return Ok(None);
    };
    let (base0, start) = split_index(ctx, first_index)?;
    let mut step = 0;
    let mut seen = HashMap::new();
    for (k, &part) in parts.iter().enumerate() {
        if scalar_read(ctx, part)? != Some(plane_id.clone()) {
            return Ok(None);
        }
        let Some(index) = raw_index(part) else {
            return Ok(None);
        };
        let (base, offset) = split_index(ctx, index)?;
        let same_base = match (base, base0) {
            (None, None) => true,
            (Some(a), Some(b)) => ctx.same_index(a, b, &mut seen)?,
            _ => false,
        };
        if !same_base {
            return Ok(None);
        }
        if k == 1 {
            step = offset - start;
        }
        if k > 0 && offset != start + k as i128 * step {
            return Ok(None);
        }
    }
    // a run is spelled as clamped or unchecked reads: another edge rule keeps its reads one by one
    let edge = ctx.str_field(first, "edge")?;
    if !matches!(edge.as_str(), "clamp" | "unchecked") {
        return Ok(None);
    }
    let clamped = edge == "clamp" || sole_index(first) != raw_index(first);
    Ok(Some(Gather {
        plane: ctx.nodes[first as usize].ops[0],
        base: base0,
        start,
        step,
        clamped,
    }))
}

/// Whether the node names a plane a placement writes; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_is_scratch(ctx: *mut Ctx, node: u32, out: *mut c_int) -> c_int {
    guard(ctx, |ctx| flag_out(out, is_scratch(ctx, node)?))
}

/// Whether the node is the launch-start value of a vector mutable; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_is_prior(ctx: *mut Ctx, node: u32, out: *mut c_int) -> c_int {
    guard(ctx, |ctx| flag_out(out, is_prior(ctx, node)?))
}

/// How element `k` of a rank-1 node is read, as the `rw_element_kind` of `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_element_kind(ctx: *mut Ctx, node: u32, kind: *mut c_int) -> c_int {
    guard(ctx, |ctx| {
        if kind.is_null() {
            return Err(Fault::BadArg("kind is NULL".into()));
        }
        let code = match element_kind(ctx, node)? {
            "proxy" => 1,
            "input" => 2,
            "parts" => 3,
            "splat" => 4,
            "elementwise" => 5,
            _ => 0,
        };
        unsafe { *kind = code }
        Ok(())
    })
}

/// The gather a vector of reads is; `*found` is 0 when it is none. `*base` is `UINT32_MAX` when
/// the index has no runtime part. See `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_gather(
    ctx: *mut Ctx,
    node: u32,
    found: *mut c_int,
    plane: *mut u32,
    base: *mut u32,
    start: *mut i64,
    step: *mut i64,
    clamped: *mut c_int,
) -> c_int {
    guard(ctx, |ctx| {
        if [
            found as usize,
            plane as usize,
            base as usize,
            start as usize,
            step as usize,
            clamped as usize,
        ]
        .contains(&0)
        {
            return Err(Fault::BadArg("an output pointer is NULL".into()));
        }
        let answer = gather(ctx, node)?;
        unsafe {
            *found = answer.is_some() as c_int;
            if let Some(g) = answer {
                (*plane, *base, *start, *step, *clamped) = (
                    g.plane,
                    g.base.unwrap_or(u32::MAX),
                    g.start as i64,
                    g.step as i64,
                    g.clamped as c_int,
                );
            }
        }
        Ok(())
    })
}
