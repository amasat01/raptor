//! The cells of a reduction over samples: how a reduction is stored to planes (one scalar plane per component of a vector or
//! matrix, keyed accumulates for a reduction over some lane axes) and how a later launch reads the finished value back.
use crate::planes::type_parts;
use crate::R;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::{ttype_val, Val};

/// The role constant label of the read of a one-cell parameter plane.
const PARAM_CELL: &str = "param_cell";

/// A node of the class from its parameters.
pub(crate) fn build(ctx: &mut Ctx, class: &str, given: Vec<(&str, Val)>) -> R<u32> {
    let index = ctx.table.find(class);
    let index = index.ok_or_else(|| Fault::Refused(format!("no {class} class")))?;
    let given: Vec<(String, Val)> = given.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    ctx.new_from(index, &given)
}

/// An integer constant of the type of `like`.
pub(crate) fn int_like(ctx: &mut Ctx, value: i64, like: u32) -> R<u32> {
    let kind = type_parts(ctx, like);
    let tt = ttype_val(&kind.dtype, &[], None);
    build(
        ctx,
        "Const",
        vec![("literal", Val::Int(i128::from(value))), ("ttype", tt)],
    )
}

/// The row-major component indices of a value of the shape: a number for a vector, a pair for a matrix.
fn components(shape: &[i64]) -> Vec<Val> {
    match shape {
        [n] => (0..*n).map(|r| Val::Int(i128::from(r))).collect(),
        [rows, cols] => (0..*rows)
            .flat_map(|r| {
                (0..*cols).map(move |c| Val::Tuple(vec![Val::Int(r as i128), Val::Int(c as i128)]))
            })
            .collect(),
        _ => vec![],
    }
}

/// The node a field of the reduction holds, when it holds one.
fn node_field(ctx: &Ctx, node: u32, field: &str) -> R<Option<u32>> {
    Ok(match ctx.field(node, field)? {
        Val::Node(found) => Some(found),
        _ => None,
    })
}

/// The index of entry `k` of a keyed payload: the key for the first, else `key + k * count`.
fn entry_index(ctx: &mut Ctx, key: u32, count: u32, k: usize) -> R<u32> {
    if k == 0 {
        return Ok(key);
    }
    let step = int_like(ctx, k as i64, count)?;
    let offset = ctx.make_op("mul", &[step, count], &Val::None)?;
    ctx.make_op("add", &[key, offset], &Val::None)
}

/// The cell sinks that store the reduction to the plane `name`: one for a scalar and one per component (planes `name_0`,
/// `name_1`, ...) for a vector or matrix, or the keyed accumulates of a reduction over some lane axes.
pub(crate) fn stored(ctx: &mut Ctx, reduction: u32, name: &str) -> R<Vec<u32>> {
    let value = ctx.nodes[reduction as usize].ops[0];
    let parts = type_parts(ctx, value);
    let scalar = ttype_val(&parts.dtype, &[], None);
    let contract = ctx.str_field(reduction, "contract")?;
    let op = ctx.str_field(reduction, "op")?;
    if let Some(key) = node_field(ctx, reduction, "key")? {
        return stored_keyed(ctx, (reduction, key), name, &contract);
    }
    if parts.shape.is_empty() {
        let given = vec![
            ("name", Val::Str(name.into())),
            ("value", Val::Node(value)),
            ("op", Val::Str(op)),
        ];
        return Ok(vec![build(ctx, "SampleSum", given)?]);
    }
    let all = components(&parts.shape);
    let columns = all.len() as i128;
    let mut out = vec![];
    for (k, index) in all.into_iter().enumerate() {
        let literal = match &index {
            Val::Tuple(pair) if parts.shape.len() == 2 => Val::Tuple(pair.clone()),
            Val::Tuple(pair) => pair[0].clone(),
            other => other.clone(),
        };
        let entry = ctx.make_op("component", &[value], &literal)?;
        let given = vec![
            ("name", Val::Str(format!("{name}_{k}"))),
            ("value", Val::Node(entry)),
            ("ttype", scalar.clone()),
            ("op", Val::Str(op.clone())),
            ("cell", Val::Str(name.into())),
            ("columns", Val::Int(columns)),
        ];
        out.push(build(ctx, "SampleSum", given)?);
    }
    Ok(out)
}

/// The accumulates of a reduction over some lane axes: entry `k` of the payload adds into the plane at `key + k * count`.
fn stored_keyed(
    ctx: &mut Ctx,
    (reduction, key): (u32, u32),
    name: &str,
    contract: &str,
) -> R<Vec<u32>> {
    let value = ctx.nodes[reduction as usize].ops[0];
    let parts = type_parts(ctx, value);
    let scalar = ttype_val(&parts.dtype, &[], None);
    let count = node_field(ctx, reduction, "count")?;
    if parts.shape.is_empty() {
        let given = vec![
            ("name", Val::Str(name.into())),
            ("value", Val::Node(value)),
            ("index", Val::Node(key)),
            ("ttype", scalar),
            ("contract", Val::Str(contract.into())),
            ("reduced", Val::Node(reduction)),
        ];
        return Ok(vec![build(ctx, "AccumWrite", given)?]);
    }
    let count = count.ok_or_else(|| Fault::Refused("a keyed vector sum without a count".into()))?;
    let mut out = vec![];
    for (k, index) in components(&parts.shape).into_iter().enumerate() {
        let literal = match (&index, parts.shape.len()) {
            (Val::Tuple(pair), 1) => pair[0].clone(),
            _ => index,
        };
        let entry = ctx.make_op("component", &[value], &literal)?;
        let at = entry_index(ctx, key, count, k)?;
        let given = vec![
            ("name", Val::Str(name.into())),
            ("value", Val::Node(entry)),
            ("index", Val::Node(at)),
            ("ttype", scalar.clone()),
            ("contract", Val::Str(contract.into())),
        ];
        out.push(build(ctx, "AccumWrite", given)?);
    }
    Ok(out)
}

/// A read of the one-cell plane `plane`, as an `At` of the given index over a `lookup` leaf of the scalar type.
fn plane_read(ctx: &mut Ctx, plane: &str, index: u32, dtype: &str) -> R<u32> {
    let scalar = ttype_val(dtype, &[], None);
    let leaf = vec![
        ("kind", Val::Str("table_read".into())),
        ("role", Val::Str("lookup".into())),
        ("name", Val::Str(plane.into())),
        ("ttype", scalar.clone()),
    ];
    let leaf = build(ctx, "Leaf", leaf)?;
    build(
        ctx,
        "At",
        vec![
            ("plane", Val::Node(leaf)),
            ("index", Val::Node(index)),
            ("ttype", scalar),
        ],
    )
}

/// The read of the finished cell of a reduction: the one-cell plane `cell` (a vector's column `k` is the plane `cell_k`), or the
/// plane of a keyed reduction at the lane's own key. A matrix is refused.
pub(crate) fn cell_read(ctx: &mut Ctx, reduction: u32, cell: &str) -> R<u32> {
    let parts = type_parts(ctx, reduction);
    if parts.shape.len() == 2 {
        return refuse(
            "a matrix reduced over samples is not read back whole: reduce each entry the kernel reads",
        );
    }
    let key = node_field(ctx, reduction, "key")?;
    let mut reads = vec![];
    for k in 0..parts.shape.first().copied().unwrap_or(1) as usize {
        let read = match key {
            Some(key) => {
                let count = node_field(ctx, reduction, "count")?
                    .ok_or_else(|| Fault::Refused("a keyed sum without a count".into()))?;
                let at = entry_index(ctx, key, count, k)?;
                plane_read(ctx, cell, at, &parts.dtype)?
            }
            None => {
                let name = if parts.shape.is_empty() {
                    cell.to_string()
                } else {
                    format!("{cell}_{k}")
                };
                let zero = vec![
                    ("value", Val::Int(0)),
                    ("label", Val::Str(PARAM_CELL.into())),
                    ("ttype", ttype_val("i32", &[], None)),
                ];
                let zero = build(ctx, "RoleConst", zero)?;
                plane_read(ctx, &name, zero, &parts.dtype)?
            }
        };
        reads.push(read);
    }
    if parts.shape.is_empty() {
        return Ok(reads[0]);
    }
    ctx.make_op("vec", &reads, &Val::None)
}
