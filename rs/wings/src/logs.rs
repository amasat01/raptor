//! The ordered gradient: the reverse of an ordered keyed sum states each accumulate of a table's gradient as an ordered accumulate
//! of the lane's contribution into a cell. Each such accumulate becomes the stores of its contribution and of its cell into the
//! lane's next slots of two log planes, and one team stage then folds the log of a target: for each slot it adds the contributions
//! of a key in lane order and stores the sum at the cell the slot names. No atomics: the same bits on every launch.
//!
//! The values plane holds `width` slots per lane. The cells plane holds `width` slots per key, written by every lane of the key
//! with the same cell (the slot's cell depends on the key and the slot only), which is why the fold can read it at the key.
use crate::cells::build;
use crate::scatter::constant;
use crate::stages::Keyed;
use crate::team::{
    sample_index, threads_for, Grid, Plane, Step, TeamLimits, TeamSpec, Unit, PLANE_PREFIX,
};
use crate::{is_a, R};
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::value::Val;

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
    /// An ordered accumulate sits in the body of a loop, whose slots by iteration are not planned.
    InLoop,
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

/// The accumulates of each target in `sinks`, in the order the targets are first added to, or none when one sits in a loop body.
fn widths(ctx: &Ctx, sinks: &[u32]) -> R<Option<Vec<Log>>> {
    let mut logs: Vec<Log> = vec![];
    for &sink in sinks {
        if is_a(ctx, sink, "Loop") {
            let mut body = vec![];
            raptor_ir::dump::refs(&ctx.field(sink, "body_sinks")?, &mut body);
            if body.iter().any(|&inner| is_ordered_accumulate(ctx, inner)) {
                return Ok(None);
            }
        }
        if !is_ordered_accumulate(ctx, sink) {
            continue;
        }
        if !ctx.nodes[sink as usize].shape.is_empty() {
            return refuse(format!(
                "an ordered gradient accumulates scalars; {} is not one",
                ctx.str_field(sink, "name")?
            ));
        }
        let target = ctx.str_field(sink, "name")?;
        match logs.iter_mut().find(|log| log.target == target) {
            Some(found) => found.width += 1,
            None => logs.push(Log {
                target,
                width: 1,
                kept: crate::stages::kept_of(ctx, sink)?,
            }),
        }
    }
    Ok(Some(logs))
}

/// The sinks with each ordered accumulate replaced by the stores of its log slot, and the log of each target. A lane's accumulates of
/// a target take its slots in program order, so the slot of a site is the number of accumulates of the target before it.
pub fn ordered(ctx: &mut Ctx, sinks: &[u32]) -> R<Logged> {
    let Some(logs) = widths(ctx, sinks)? else {
        return Ok(Logged::InLoop);
    };
    let mut used = vec![0_i64; logs.len()];
    let mut out = vec![];
    for &sink in sinks {
        if !is_ordered_accumulate(ctx, sink) {
            out.push(sink);
            continue;
        }
        let target = ctx.str_field(sink, "name")?;
        let Some(at) = logs.iter().position(|log| log.target == target) else {
            return refuse(format!("{target}: an ordered accumulate without a log"));
        };
        let (slot, width) = (used[at], logs[at].width);
        used[at] += 1;
        let kept = ctx.field(sink, "kept")?;
        let [values, cells] = names(&target);
        let (value, index, key) = {
            let ops = &ctx.nodes[sink as usize].ops;
            (ops[0], ops[1], ops[2])
        };
        // the value goes to the lane's slice of the values plane
        let (lane, stride, offset) = (
            sample_index(ctx)?,
            constant(ctx, width, "i64")?,
            constant(ctx, slot, "i64")?,
        );
        let scaled = ctx.make_op("mul", &[lane, stride], &Val::None)?;
        let value_at = ctx.make_op("add", &[scaled, offset], &Val::None)?;
        // the cell goes to the key's slice of the cells plane, so every lane of a key states the same cell
        let key = ctx.make_op("cast", &[key], &Val::Str("i32".into()))?;
        let (stride, offset) = (constant(ctx, width, "i32")?, constant(ctx, slot, "i32")?);
        let scaled = ctx.make_op("mul", &[key, stride], &Val::None)?;
        let cell_at = ctx.make_op("add", &[scaled, offset], &Val::None)?;
        let cell = ctx.make_op("cast", &[index], &Val::Str("f64".into()))?;
        out.push(log_store(
            ctx,
            (&values, &target, &kept),
            (value, value_at, width),
        )?);
        out.push(log_store(
            ctx,
            (&cells, &target, &kept),
            (cell, cell_at, width),
        )?);
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
