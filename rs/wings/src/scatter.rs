//! The ordered scatter: each `scatter(target, contract="ordered")[key] += value` becomes the stores of its key and value into the
//! lane's next slots of two log planes (one slot per add, in program order), and three team stages then add the logs into the
//! target: one block per cell zeroes the cell's partial sums, one per chunk of consecutive lanes adds its lanes' slots in order into
//! its own partial sum of each key, and one block per cell adds the cell's partial sums in chunk order. No atomics and no sort: the
//! same bits on every launch.
use crate::blocks::{block_shaped, ordered_case};
use crate::cells::{build, components, int_like};
use crate::logs::cell_of;
use crate::planes::type_parts;
use crate::stages::is_ordered_sink;
use crate::team::{sample_index, Grid, Plane, Step, TeamSpec, Unit, PLANE_PREFIX};
use crate::{is_a, is_routable_add, is_update_of, R};
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::value::{ttype_val, Val};

/// The log slots each scatter target holds per lane, in the order the targets are first added to.
pub type Slots = Vec<(String, i64)>;

/// The key a lane's log slot holds when the lane added nothing there; a chunk pass skips a negative key.
const NO_KEY: i64 = -1;

/// The scratch planes of the ordered scatter into `target`: the keys, the values, the partial sums and the status.
pub fn names(target: &str) -> [String; 4] {
    ["k", "v", "p", "m"].map(|role| format!("{PLANE_PREFIX}s_{target}_{role}"))
}

/// A constant of the type.
pub(crate) fn constant(ctx: &mut Ctx, value: i64, dtype: &str) -> R<u32> {
    build(
        ctx,
        "Const",
        vec![
            ("literal", Val::Int(i128::from(value))),
            ("ttype", ttype_val(dtype, &[], None)),
        ],
    )
}

/// The store of a value into slot `at` of a log plane of the scatter into `target`.
fn log_store(ctx: &mut Ctx, (plane, target): (&str, &str), (value, at): (u32, u32)) -> R<u32> {
    let operands = Val::Tuple(vec![Val::Node(value), Val::Node(at)]);
    let given = vec![
        ("name", Val::Str(plane.into())),
        ("operands", operands),
        ("target", Val::Str(target.into())),
    ];
    build(ctx, "ScatterLog", given)
}

/// What a routed keyed accumulate becomes.
enum Routed {
    /// It stays as it is (it is no sum over the lanes of a key, or its payload is too large for a team).
    Keep,
    /// The write of the block sum of its payload.
    Block(u32),
    /// Its payload has a shape whose block stages are not planned.
    Unwritten,
}

/// What the keyed accumulate `sink` becomes when the build routes accumulates to the ordered stages: a sum over the lanes of a key
/// is then one block sum of its payload, which the team stages of a block add in lane order.
fn route(ctx: &mut Ctx, sink: u32, max_dim: i64) -> R<Routed> {
    if !is_update_of(ctx, sink, "add")? || ctx.nodes[sink as usize].ops.len() < 2 {
        return Ok(Routed::Keep);
    }
    let Val::Node(reduced) = ctx.field(sink, "reduced")? else {
        return Ok(Routed::Keep);
    };
    if !is_routable_add(ctx, sink)? || ctx.str_field(reduced, "op")? != "sum" {
        return Ok(Routed::Keep);
    }
    let payload = ctx.nodes[reduced as usize].ops[0];
    if !ordered_case(ctx, payload, "ordered", Some(max_dim)) {
        let shaped = block_shaped(ctx, payload, "ordered", Some(max_dim));
        return Ok(if shaped {
            Routed::Unwritten
        } else {
            Routed::Keep
        });
    }
    let (key, count) = (ctx.field(reduced, "key")?, ctx.field(reduced, "count")?);
    let (Val::Node(key), Val::Node(count)) = (key, count) else {
        return Ok(Routed::Keep);
    };
    let operands = Val::Tuple([payload, key, count].map(Val::Node).to_vec());
    let kept = ctx.field(reduced, "kept")?;
    let given = vec![
        ("operands", operands),
        ("contract", Val::Str("ordered".into())),
        ("kept", kept),
    ];
    let block = build(ctx, "BlockSum", given)?;
    let given = vec![
        ("name", Val::Str(ctx.str_field(sink, "name")?)),
        ("operands", Val::Tuple(vec![Val::Node(block)])),
    ];
    build(ctx, "BlockSumWrite", given).map(Routed::Block)
}

/// The labels of the leading axes a value-shaped keyed add gives its plane, one per dimension of the value (`component` for a
/// vector, `row` and `col` for a matrix), each made distinct from the plane's declared labels by trailing underscores.
fn payload_axes(declared: &[Val], shape: &[i64]) -> Vec<Val> {
    let mut taken: Vec<String> = declared
        .iter()
        .filter_map(|axis| match axis {
            Val::Tuple(pair) => match pair.first() {
                Some(Val::Str(label)) => Some(label.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let stems: &[&str] = if shape.len() == 1 {
        &["component"]
    } else {
        &["row", "col"]
    };
    let mut axes = vec![];
    for (stem, extent) in stems.iter().zip(shape) {
        let mut label = (*stem).to_string();
        while taken.contains(&label) {
            label.push('_');
        }
        taken.push(label.clone());
        axes.push(Val::Tuple(vec![
            Val::Str(label),
            Val::Int(i128::from(*extent)),
        ]));
    }
    axes
}

/// One scalar add per component of the value-shaped keyed add `sink`, at the component's coordinates followed by the key's, into
/// the plane whose axes are the value's then the sink's. The log and the block stages of the ordered lowering hold scalar terms, so
/// a value in lane order is summed component by component, each component in lane order.
fn split_value(ctx: &mut Ctx, sink: u32) -> R<Vec<u32>> {
    if ctx.field(sink, "reduced")? != Val::None {
        return refuse("a value-shaped keyed add in lane order that sums over lanes is not built");
    }
    let parts = type_parts(ctx, sink);
    let scalar = ttype_val(&parts.dtype, &[], None);
    let (value, indices) = {
        let ops = &ctx.nodes[sink as usize].ops;
        (ops[0], ops[1..].to_vec())
    };
    let Val::Tuple(declared) = ctx.field(sink, "axes")? else {
        return refuse("a keyed add has no axes");
    };
    let mut axes = payload_axes(&declared, &parts.shape);
    axes.extend(declared);
    let mut out = vec![];
    for entry in components(&parts.shape) {
        let at = match &entry {
            Val::Tuple(each) => each.clone(),
            one => vec![one.clone()],
        };
        let mut coordinates = vec![];
        for k in &at {
            let Val::Int(k) = k else {
                return refuse("a component coordinate is a whole number");
            };
            coordinates.push(Val::Node(int_like(ctx, *k as i64, indices[0])?));
        }
        coordinates.extend(indices.iter().map(|&at| Val::Node(at)));
        let part = ctx.make_op("component", &[value], &entry)?;
        let mut given = vec![
            ("name", ctx.field(sink, "name")?),
            ("value", Val::Node(part)),
            ("indices", Val::Tuple(coordinates)),
            ("ttype", scalar.clone()),
            ("axes", Val::Tuple(axes.clone())),
        ];
        for field in ["op", "summation", "order", "edge"] {
            given.push((field, ctx.field(sink, field)?));
        }
        out.push(build(ctx, "Update", given)?);
    }
    Ok(out)
}

/// The sinks with each ordered accumulate replaced by the stores of its key and value into its log slots, and the slots each target
/// holds per lane, in the order the targets are first added to, or none when a payload's block stages are not planned. When the
/// build routes accumulates to the ordered stages the accumulator carries its own compensation, so no residual plane is kept, and
/// a sum over the lanes of a key is a block sum.
pub fn logged(
    ctx: &mut Ctx,
    sinks: &[u32],
    (routed, max_dim): (bool, i64),
) -> R<Option<(Vec<u32>, Slots)>> {
    let mut kept = vec![];
    let mut residual = vec![];
    let mut scalar_sinks = vec![];
    for &sink in sinks {
        if is_ordered_sink(ctx, sink, routed)? && !ctx.nodes[sink as usize].shape.is_empty() {
            scalar_sinks.extend(split_value(ctx, sink)?);
        } else {
            scalar_sinks.push(sink);
        }
    }
    let sinks = &scalar_sinks[..];
    for &sink in sinks {
        let own = ctx.compensates(sink)?;
        if routed && own {
            residual.push(format!("{}_c", ctx.str_field(sink, "name")?));
        }
    }
    for &sink in sinks {
        let dropped =
            routed && is_a(ctx, sink, "Update") && residual.contains(&ctx.str_field(sink, "name")?);
        if dropped {
            continue;
        }
        let routed_to = if routed {
            route(ctx, sink, max_dim)?
        } else {
            Routed::Keep
        };
        match routed_to {
            Routed::Keep => kept.push(sink),
            Routed::Block(write) => kept.push(write),
            Routed::Unwritten => return Ok(None),
        }
    }
    let mut width: Vec<(String, i64)> = vec![];
    for &sink in &kept {
        if is_ordered_sink(ctx, sink, routed)? {
            let name = ctx.str_field(sink, "name")?;
            match width.iter_mut().find(|w| w.0 == name) {
                Some(found) => found.1 += 1,
                None => width.push((name, 1)),
            }
        }
    }
    if width.is_empty() {
        return Ok(Some((kept, width)));
    }
    let mut used: Vec<(String, i64)> = vec![];
    let mut out = vec![];
    for sink in kept {
        if !is_ordered_sink(ctx, sink, routed)? {
            out.push(sink);
            continue;
        }
        let target = ctx.str_field(sink, "name")?;
        let slot = used.iter().find(|u| u.0 == target).map_or(0, |u| u.1);
        match used.iter_mut().find(|u| u.0 == target) {
            Some(found) => found.1 += 1,
            None => used.push((target.clone(), 1)),
        }
        let per_lane = width.iter().find(|w| w.0 == target).map_or(1, |w| w.1);
        let (keys, values) = (names(&target)[0].clone(), names(&target)[1].clone());
        let (lane, stride, offset) = (
            sample_index(ctx)?,
            constant(ctx, per_lane, "i64")?,
            constant(ctx, slot, "i64")?,
        );
        let scaled = ctx.make_op("mul", &[lane, stride], &Val::None)?;
        let at = ctx.make_op("add", &[scaled, offset], &Val::None)?;
        let (value, indices) = {
            let ops = &ctx.nodes[sink as usize].ops;
            (ops[0], ops[1..].to_vec())
        };
        let index = cell_of(ctx, sink, &indices)?;
        let key = match ctx.nodes[index as usize].dtype.as_str() {
            "int32" => index,
            _ => ctx.make_op("cast", &[index], &Val::Str("int32".into()))?,
        };
        out.push(log_store(ctx, (&keys, &target), (key, at))?);
        out.push(log_store(ctx, (&values, &target), (value, at))?);
    }
    Ok(Some((out, width)))
}

/// The three team stages that add the logs of the scatter into `target`, a lane holding `width` log slots.
pub fn stage_specs(target: &str, width: i64, (max_chunks, threads): (i64, i64)) -> Vec<TeamSpec> {
    let [keys, values, parts, status] = names(target);
    let logs = [
        Plane {
            name: keys.clone(),
            dtype: "int32",
            per_lane: width,
            unit: Unit::Lane,
            fill: Some(NO_KEY),
        },
        Plane::per_lane(values.clone(), "real", width),
    ];
    let part = Plane {
        name: parts.clone(),
        dtype: "partial",
        per_lane: 1,
        unit: Unit::Target,
        fill: None,
    };
    let flags = Plane {
        name: status.clone(),
        dtype: "int32",
        per_lane: 1 + max_chunks,
        unit: Unit::Once,
        fill: Some(0),
    };
    let stage = |op: &'static str,
                 roles: Vec<(&str, &String)>,
                 grid: Grid,
                 planes: Vec<Plane>,
                 out: Option<&str>| TeamSpec {
        threads,
        steps: vec![Step {
            op,
            n: width,
            k: 0,
            m: 0,
            max_chunks,
            planes: roles
                .into_iter()
                .map(|(role, name)| (role.to_string(), name.clone()))
                .collect(),
        }],
        planes,
        grid,
        out: out.map(String::from),
        reads: vec![],
    };
    let target_name = target.to_string();
    let out_plane = target.to_string();
    vec![
        stage(
            "scatter_zero",
            vec![("part", &parts), ("status", &status)],
            Grid::Targets {
                target: target_name.clone(),
            },
            vec![part.clone(), flags.clone()],
            None,
        ),
        stage(
            "scatter_chunks",
            vec![
                ("k", &keys),
                ("v", &values),
                ("part", &parts),
                ("status", &status),
            ],
            Grid::Chunks {
                target: target_name.clone(),
            },
            vec![logs[0].clone(), logs[1].clone(), part.clone(), flags],
            None,
        ),
        stage(
            "scatter_cells",
            vec![("part", &parts), ("out", &out_plane)],
            Grid::Targets {
                target: target_name,
            },
            vec![part],
            Some(target),
        ),
    ]
}

#[cfg(test)]
mod tests {
    //! The status plane has a flag for each chunk the card allows.
    use super::*;

    #[test]
    fn the_status_plane_is_as_wide_as_the_cards_chunk_count_plus_one() {
        for max_chunks in [64, 256] {
            let stages = stage_specs("t", 2, (max_chunks, 32));
            let status = &names("t")[3];
            for stage in &stages[..2] {
                let flags = stage.planes.iter().find(|p| &p.name == status).unwrap();
                assert_eq!(flags.per_lane, 1 + max_chunks);
            }
        }
    }
}
