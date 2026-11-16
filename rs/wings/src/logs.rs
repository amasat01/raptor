//! The ordered gradient: the reverse of an ordered keyed sum states each accumulate of a table's gradient as an ordered accumulate
//! of the lane's contribution into a cell. Each such accumulate becomes the stores of its contribution and of its cell into the
//! lane's next slots of two log planes, and one team stage then folds the log of a target: for each slot it adds the contributions
//! of a key in lane order and stores the sum at the cell the slot names. No atomics: the same bits on every launch.
//!
//! The values plane holds `width` slots per lane. The cells plane holds `width` slots per key, written by every lane of the key
//! with the same cell (the slot's cell depends on the key and the slot only), which is why the fold can read it at the key.
use crate::cells::build;
use crate::construct::refs;
use crate::scatter::constant;
use crate::stages::Keyed;
use crate::team::{
    sample_index, threads_for, Grid, Plane, Step, TeamLimits, TeamSpec, Unit, PLANE_PREFIX,
};
use crate::{is_a, R};
use raptor_ir::build::LoopParts;
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::value::Val;
use std::collections::HashSet;

/// The log of one target: the slots each lane holds and the lane axes the keyed sum kept.
#[derive(Debug, Clone, PartialEq)]
pub struct Log {
    /// The plane the gradient is stored into.
    pub target: String,
    /// The accumulates of the target, one slot each.
    pub width: i64,
    /// The kept lane axes of the keyed sum, each a label and an extent when it has them.
    pub kept: Vec<(Option<String>, Option<i64>)>,
}

/// What the rewrite made of a kernel's sinks.
#[derive(Debug, Clone, PartialEq)]
pub enum Logged {
    /// The sinks, each ordered accumulate replaced by its two log stores, and one log per target in first-added order.
    Done(Vec<u32>, Vec<Log>),
    /// An accumulate the log cannot hold, and why (the lowering names the kernel).
    Refused(String),
}

/// Where a site's slot is: a number, or a function of the index of the loop the site sits in.
enum Slot {
    /// The site is at the top level of the kernel: slot `n`.
    At(i64),
    /// The site is in a loop of the index `index` from `start`: the slots `base`, `base + 1`, ... of its iterations.
    PerTrip {
        /// The slot of the first iteration.
        base: i64,
        /// The loop's index node.
        index: u32,
        /// The loop's first index value.
        start: i64,
    },
}

/// The names of the log planes of the ordered accumulate into `target`: the values and the cells.
pub fn names(target: &str) -> [String; 2] {
    ["v", "i"].map(|role| format!("{PLANE_PREFIX}o_{target}_{role}"))
}

/// The store of `value` into slot `at` of the log plane `plane` of the accumulate into `target`.
fn log_store(
    ctx: &mut Ctx,
    (plane, target, kept): (&str, &str, &Val),
    (value, at, width): (u32, u32, i64),
) -> R<u32> {
    let given = vec![
        ("name", Val::Str(plane.into())),
        (
            "operands",
            Val::Tuple(vec![Val::Node(value), Val::Node(at)]),
        ),
        ("target", Val::Str(target.into())),
        ("width", Val::Int(i128::from(width))),
        ("kept", kept.clone()),
    ];
    build(ctx, "LogStore", given)
}

/// Whether the sink is an ordered accumulate.
fn is_ordered_accumulate(ctx: &Ctx, sink: u32) -> bool {
    is_a(ctx, sink, "OrderedAccumWrite")
}

/// A whole-number field of a loop.
fn loop_int(ctx: &Ctx, lp: u32, field: &str) -> R<i64> {
    match ctx.field(lp, field)? {
        Val::Int(n) => Ok(n as i64),
        _ => refuse(format!("a loop's {field} is not a whole number")),
    }
}

/// The ordered accumulates among the body sinks of the loop `lp`, or the reason the log cannot hold them: a step other than 1 (a slot is the iteration number), or a value that is not a scalar.
fn loop_sites(ctx: &Ctx, lp: u32) -> R<Result<Vec<u32>, String>> {
    let body = refs(ctx, lp, "body_sinks")?;
    let sites: Vec<u32> = body
        .iter()
        .copied()
        .filter(|&s| is_ordered_accumulate(ctx, s))
        .collect();
    let step = loop_int(ctx, lp, "step")?;
    if !sites.is_empty() && step != 1 {
        return Ok(Err(format!(
            "an ordered accumulate inside a loop of step {step} is not written: use a loop of step 1"
        )));
    }
    Ok(Ok(sites))
}

/// Whether a loop that is not a sink of the kernel holds an ordered accumulate: a loop inside the body of another, or one a sink or a
/// body reads as a value. Its slots would depend on two indices, which the log does not state.
fn nested_site(ctx: &Ctx, sinks: &[u32]) -> R<bool> {
    let (mut seen, mut stack) = (HashSet::new(), sinks.to_vec());
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        stack.extend(ctx.nodes[node as usize].ops.iter().copied());
        if is_a(ctx, node, "Loop") {
            stack.extend(ctx.body_nodes(node)?);
            let held = !sinks.contains(&node);
            let inner = refs(ctx, node, "body_sinks")?;
            if held && inner.iter().any(|&s| is_ordered_accumulate(ctx, s)) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The refusal of an accumulate of a value that is not a scalar, or none.
fn not_scalar(ctx: &Ctx, sink: u32) -> R<Option<String>> {
    if ctx.nodes[sink as usize].shape.is_empty() {
        return Ok(None);
    }
    Ok(Some(format!(
        "an ordered accumulate takes scalars, and the one into {} is not a scalar: accumulate each component on its own",
        ctx.str_field(sink, "name")?
    )))
}

/// The accumulates of each target in `sinks`, in the order the targets are first added to, with the slots each holds per lane: one per
/// site at the top level, one per iteration for a site in a loop; or the reason one cannot be held.
fn widths(ctx: &Ctx, sinks: &[u32]) -> R<Result<Vec<Log>, String>> {
    if nested_site(ctx, sinks)? {
        return Ok(Err(
            "an ordered accumulate inside nested loops is not written: put the accumulates in one loop, or outside any loop".into(),
        ));
    }
    let mut logs: Vec<Log> = vec![];
    let mut count = |ctx: &Ctx, sink: u32, slots: i64| -> R<Option<String>> {
        if let Some(why) = not_scalar(ctx, sink)? {
            return Ok(Some(why));
        }
        let target = ctx.str_field(sink, "name")?;
        match logs.iter_mut().find(|log| log.target == target) {
            Some(found) => found.width += slots,
            None => logs.push(Log {
                target,
                width: slots,
                kept: crate::stages::kept_of(ctx, sink)?,
            }),
        }
        Ok(None)
    };
    for &sink in sinks {
        if is_ordered_accumulate(ctx, sink) {
            if let Some(why) = count(ctx, sink, 1)? {
                return Ok(Err(why));
            }
        } else if is_a(ctx, sink, "Loop") {
            let sites = match loop_sites(ctx, sink)? {
                Ok(sites) => sites,
                Err(why) => return Ok(Err(why)),
            };
            let trip = loop_int(ctx, sink, "trip")?;
            for site in sites {
                if let Some(why) = count(ctx, site, trip)? {
                    return Ok(Err(why));
                }
            }
        }
    }
    Ok(Ok(logs))
}

/// The offset of a site's slot in the values plane (added to the lane's `lane * width`) and in the cells plane (added to the key's
/// `key * width`). A slot inside a loop is a 32-bit expression of the loop's index, shared by both planes.
fn offsets(ctx: &mut Ctx, slot: &Slot) -> R<(u32, u32)> {
    match *slot {
        Slot::At(n) => Ok((constant(ctx, n, "i64")?, constant(ctx, n, "i32")?)),
        Slot::PerTrip { base, index, start } => {
            let from = constant(ctx, start, "i32")?;
            let turn = ctx.make_op("sub", &[index, from], &Val::None)?;
            let first = constant(ctx, base, "i32")?;
            let at = ctx.make_op("add", &[first, turn], &Val::None)?;
            Ok((at, at))
        }
    }
}

/// The two log stores of the ordered accumulate `sink` at `slot` of a log of `width` slots: its value into the lane's slice of the
/// values plane, and its cell into the key's slice of the cells plane, so every lane of a key states the same cell.
fn stores(ctx: &mut Ctx, sink: u32, (width, slot): (i64, &Slot)) -> R<[u32; 2]> {
    let target = ctx.str_field(sink, "name")?;
    let kept = ctx.field(sink, "kept")?;
    let [values, cells] = names(&target);
    let (value, index, key) = {
        let ops = &ctx.nodes[sink as usize].ops;
        (ops[0], ops[1], ops[2])
    };
    let (value_offset, cell_offset) = offsets(ctx, slot)?;
    let (lane, stride) = (sample_index(ctx)?, constant(ctx, width, "i64")?);
    let scaled = ctx.make_op("mul", &[lane, stride], &Val::None)?;
    let value_at = ctx.make_op("add", &[scaled, value_offset], &Val::None)?;
    let key = ctx.make_op("cast", &[key], &Val::Str("i32".into()))?;
    let stride = constant(ctx, width, "i32")?;
    let scaled = ctx.make_op("mul", &[key, stride], &Val::None)?;
    let cell_at = ctx.make_op("add", &[scaled, cell_offset], &Val::None)?;
    let cell = ctx.make_op("cast", &[index], &Val::Str("f64".into()))?;
    Ok([
        log_store(ctx, (&values, &target, &kept), (value, value_at, width))?,
        log_store(ctx, (&cells, &target, &kept), (cell, cell_at, width))?,
    ])
}

/// The loop `lp` made again with the sinks `body` in place of its own, everything else kept.
fn with_body(ctx: &mut Ctx, lp: u32, body: &[u32]) -> R<u32> {
    let carries = refs(ctx, lp, "carries")?;
    let inits = ctx.nodes[lp as usize].ops[..carries.len()].to_vec();
    let nexts = refs(ctx, lp, "nexts")?;
    let exit_cond = refs(ctx, lp, "exit_cond")?.first().copied();
    let exit_values = refs(ctx, lp, "exit_values")?;
    let count = refs(ctx, lp, "count")?.first().copied();
    let dynamic = match ctx.field(lp, "dynamic")? {
        Val::Str(word) => Some(word),
        _ => None,
    };
    let parts = LoopParts {
        index: refs(ctx, lp, "index")?[0],
        carries: &carries,
        inits: &inits,
        nexts: &nexts,
        range: (
            loop_int(ctx, lp, "start")?,
            loop_int(ctx, lp, "stop")?,
            loop_int(ctx, lp, "step")?,
        ),
        body_sinks: body,
        exit_cond,
        exit_values: &exit_values,
        count,
        count_tail: ctx.field(lp, "count_tail")? == Val::Bool(true),
        dynamic: dynamic.as_deref(),
    };
    ctx.build_loop(&parts)
}

/// The sinks with each ordered accumulate replaced by the stores of its log slot, and the log of each target. A lane's accumulates of
/// a target take its slots in program order; a site in a loop takes one slot per iteration, `base + (index - start)`.
pub fn ordered(ctx: &mut Ctx, sinks: &[u32]) -> R<Logged> {
    let logs = match widths(ctx, sinks)? {
        Ok(logs) => logs,
        Err(why) => return Ok(Logged::Refused(why)),
    };
    let width_of = |ctx: &Ctx, sink: u32| -> R<(usize, i64)> {
        let target = ctx.str_field(sink, "name")?;
        match logs.iter().position(|log| log.target == target) {
            Some(at) => Ok((at, logs[at].width)),
            None => refuse(format!("{target}: an ordered accumulate without a log")),
        }
    };
    let mut used = vec![0_i64; logs.len()];
    let mut out = vec![];
    for &sink in sinks {
        if is_ordered_accumulate(ctx, sink) {
            let (at, width) = width_of(ctx, sink)?;
            out.extend(stores(ctx, sink, (width, &Slot::At(used[at])))?);
            used[at] += 1;
        } else if is_a(ctx, sink, "Loop") && !refs(ctx, sink, "body_sinks")?.is_empty() {
            let index = refs(ctx, sink, "index")?[0];
            let (start, trip) = (loop_int(ctx, sink, "start")?, loop_int(ctx, sink, "trip")?);
            let mut body = vec![];
            for inner in refs(ctx, sink, "body_sinks")? {
                if !is_ordered_accumulate(ctx, inner) {
                    body.push(inner);
                    continue;
                }
                let (at, width) = width_of(ctx, inner)?;
                let slot = Slot::PerTrip {
                    base: used[at],
                    index,
                    start,
                };
                body.extend(stores(ctx, inner, (width, &slot))?);
                used[at] += trip;
            }
            out.push(with_body(ctx, sink, &body)?);
        } else {
            out.push(sink);
        }
    }
    Ok(Logged::Done(out, logs))
}

/// The team stage that folds the log of `log.target`: each key's lanes are contiguous, and for each of the `log.width` slots the
/// key's contributions are added in lane order into the cell the slot names. The gradient plane is stored, not accumulated into.
pub fn fold_spec(log: &Log, limits: &TeamLimits) -> TeamSpec {
    let [values, cells] = names(&log.target);
    let planes = vec![
        Plane::per_lane(values.clone(), "real", log.width),
        Plane {
            name: cells.clone(),
            dtype: "real",
            per_lane: log.width,
            unit: Unit::Key,
            fill: None,
        },
    ];
    let roles = [("v", values), ("i", cells), ("out", log.target.clone())];
    TeamSpec {
        threads: threads_for(log.width, limits),
        steps: vec![Step {
            op: "ordered_fold",
            n: log.width,
            k: 0,
            m: 0,
            max_chunks: 0,
            planes: roles
                .into_iter()
                .map(|(role, name)| (role.to_string(), name))
                .collect(),
        }],
        planes,
        grid: Grid::Keys {
            keyed: Keyed {
                name: log.target.clone(),
                width: 1,
                kept: log.kept.clone(),
                payload: vec![],
            },
            groups: 1,
        },
        out: Some(log.target.clone()),
        reads: vec![],
    }
}
