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

/// A vector whose element `k` is the read `plane[lead.., base + start + k * step]`: the running index is the last one.
pub struct Gather {
    /// The plane read.
    pub plane: u32,
    /// The index nodes ahead of the running one, shared by every read of the run (none for a plane read at one index).
    pub lead: Vec<u32>,
    /// The run-time part of the running index, when there is one.
    pub base: Option<u32>,
    /// The running index of the first element, without the run-time part.
    pub start: i128,
    /// The distance between the running indices of neighbouring elements.
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

/// The lead indices and the running index of a read: every index operand but the last, and the last. A read at one index carries
/// the clamp of an old-form read as an operation, so its run is of the index before it.
fn split_read(ctx: &Ctx, read: u32) -> Option<(Vec<u32>, u32)> {
    let (&last, lead) = ctx.nodes[read as usize].ops[1..].split_last()?;
    Some((
        lead.to_vec(),
        if lead.is_empty() {
            unclamped(ctx, last)
        } else {
            last
        },
    ))
}

/// The gather a `vec` of reads of one plane at an arithmetic run of the last index is, else None. Reads at several indices are a
/// run only when every read has the same lead indices and the same edge rule.
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
    let Some((lead0, first_index)) = split_read(ctx, first) else {
        return Ok(None);
    };
    let edge = ctx.str_field(first, "edge")?;
    let (base0, start) = split_index(ctx, first_index)?;
    let mut step = 0;
    let mut seen = HashMap::new();
    for (k, &part) in parts.iter().enumerate() {
        if scalar_read(ctx, part)? != Some(plane_id.clone()) {
            return Ok(None);
        }
        let Some((lead, index)) = split_read(ctx, part) else {
            return Ok(None);
        };
        if lead.len() != lead0.len() {
            return Ok(None);
        }
        for (&a, &b) in lead.iter().zip(&lead0) {
            if !ctx.same_index(a, b, &mut seen)? {
                return Ok(None);
            }
        }
        if !lead0.is_empty() && ctx.str_field(part, "edge")? != edge {
            return Ok(None);
        }
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
    if !matches!(edge.as_str(), "clamp" | "unchecked") {
        return Ok(None);
    }
    let last = *ctx.nodes[first as usize]
        .ops
        .last()
        .expect("a read has an index");
    Ok(Some(Gather {
        plane: ctx.nodes[first as usize].ops[0],
        lead: lead0,
        base: base0,
        start,
        step,
        clamped: edge == "clamp" || last != first_index,
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

/// The most index operands ahead of the running one that the gather answer holds: a plane has at most eight axes.
pub const GATHER_LEAD_MAX: usize = 8;

/// The answer of `rw_wide_gather`, as `raptor_wings_front.h` states it (`found` 0: no run).
#[repr(C)]
#[derive(Default)]
pub struct GatherAnswer {
    /// Whether the vector is a run of reads (0 or 1).
    pub found: c_int,
    /// The plane read.
    pub plane: u32,
    /// The run-time part of the running index, `u32::MAX` when it has none.
    pub base: u32,
    /// The running index of the first element, without its run-time part.
    pub start: i64,
    /// The distance between the running indices of neighbouring elements.
    pub step: i64,
    /// Whether the reads keep the clamp of the original index.
    pub clamped: c_int,
    /// How many lead index nodes `lead` holds.
    pub nlead: usize,
    /// The index nodes ahead of the running index, shared by every read of the run.
    pub lead: [u32; GATHER_LEAD_MAX],
}

/// The gather a vector of reads is; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_gather(ctx: *mut Ctx, node: u32, out: *mut GatherAnswer) -> c_int {
    guard(ctx, |ctx| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        let mut answer = GatherAnswer::default();
        if let Some(g) = gather(ctx, node)? {
            if g.lead.len() > GATHER_LEAD_MAX {
                return Err(Fault::BadArg(
                    "a run over more axes than a plane has".into(),
                ));
            }
            answer.found = 1;
            answer.plane = g.plane;
            answer.base = g.base.unwrap_or(u32::MAX);
            (answer.start, answer.step) = (g.start as i64, g.step as i64);
            answer.clamped = g.clamped as c_int;
            answer.nlead = g.lead.len();
            answer.lead[..g.lead.len()].copy_from_slice(&g.lead);
        }
        unsafe { *out = answer };
        Ok(())
    })
}
