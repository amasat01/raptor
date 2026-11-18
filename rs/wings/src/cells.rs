//! The cells of a reduction over samples: how a reduction is stored to planes (one scalar plane per component of a vector or
//! matrix, keyed accumulates for a reduction over some lane axes) and how a later launch reads the finished value back.
use crate::planes::type_parts;
use crate::segment::{strategies, Strategy};
use crate::{is_a, list_out, name_of, R};
use raptor_ir::canon::{declared_slots, Key, Tt};
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::typed::{put_pairs, text_pair, RxPair};
use raptor_ir::value::{ttype_val, Val};
use std::ffi::{c_char, c_int};

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
pub fn stored(ctx: &mut Ctx, reduction: u32, name: &str) -> R<Vec<u32>> {
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

/// The cell sinks that store the reduction over samples `reduction` to the plane `name`, into `buf`: one for a scalar, one per
/// component for a vector or matrix, or the keyed accumulates of a reduction over some lane axes.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `name` is a NUL-terminated string; `buf` is valid for
/// `cap` nodes and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_cells_stored(
    ctx: *mut Ctx,
    reduction: u32,
    name: *const c_char,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        c.node(reduction)?;
        if !crate::is_a(c, reduction, "ReduceSamples") {
            return refuse("stored takes a reduction over samples");
        }
        let sinks = stored(c, reduction, name_of(name, "name")?)?;
        list_out(buf, cap, n, &sinks)
    })
}

/// The reserved `lookup` plane a finishing kernel counts its newly finished samples into: one `uint32` cell.
pub const FINISHED_PLANE: &str = "finished_count";

/// The reserved `lookup` plane a kernel with an automatic step count reads its trip count from; the access class skips the read.
pub const FUSED_STEPS_PLANE: &str = "fused_steps";

/// The suffix of the companion plane a summing `sample_sum` keeps beside its target, from the manifest constant a front end reads
/// too (`SAMPLE_SUM_COMP_SUFFIX`; not `COMPANION_SUFFIX`, which names the companion of a keyed accumulate).
fn sample_sum_comp() -> R<&'static str> {
    rcore::STR_CONSTS
        .iter()
        .find(|c| c.0 == "SAMPLE_SUM_COMP_SUFFIX")
        .map(|c| c.1)
        .map_or_else(|| refuse("the manifest has no SAMPLE_SUM_COMP_SUFFIX"), Ok)
}

/// The type of a node as a slot type.
fn slot_type(ctx: &Ctx, node: u32) -> R<Tt> {
    let n = ctx.node(node)?;
    Ok(Tt {
        dtype: n.dtype.clone(),
        shape: n.shape.clone(),
        tag: n.tag.clone(),
    })
}

/// The compensation slot of each summing `sample_sum` among the sinks: `<name>_comp`, of the sum's role and type. Refuses when a
/// sink (or a body sink of a loop) already holds a plane of that role and name.
pub fn companion_slots(ctx: &Ctx, sinks: &[u32]) -> R<Vec<(Key, Tt)>> {
    let (mut held, mut stack) = (vec![], sinks.to_vec());
    while let Some(node) = stack.pop() {
        if is_a(ctx, node, "Loop") {
            stack.extend(raptor_ir_refs(ctx, node)?);
        } else if is_a(ctx, node, "Sink") {
            held.push((ctx.str_field(node, "role")?, ctx.str_field(node, "name")?));
        }
    }
    let mut out = vec![];
    for &sink in sinks {
        if is_a(ctx, sink, "SampleSum") && ctx.str_field(sink, "op")? == "sum" {
            let key = (
                ctx.str_field(sink, "role")?,
                format!("{}{}", ctx.str_field(sink, "name")?, sample_sum_comp()?),
            );
            if held.contains(&key) {
                return refuse(format!(
                    "sample_sum {}: its compensation plane {} collides with a plane of the same name",
                    rcore::keytext::py_str(&ctx.str_field(sink, "name")?),
                    rcore::keytext::py_str(&key.1)
                ));
            }
            out.push((key, slot_type(ctx, sink)?));
        }
    }
    Ok(out)
}

/// The summing sample sums of the walk, each as the pair (the sum's plane, its compensation plane `<name>_comp`), in the walk's order.
pub fn walk_companions(ctx: &Ctx, walk: u32) -> R<Vec<(String, String)>> {
    let mut pairs = vec![];
    for node in crate::walk_order(ctx, walk)? {
        if is_a(ctx, node, "SampleSum") && ctx.str_field(node, "op")? == "sum" {
            let row = ctx.str_field(node, "name")?;
            pairs.push((row.clone(), format!("{row}{}", sample_sum_comp()?)));
        }
    }
    Ok(pairs)
}

/// The summing sample sums of the walk as pairs (the sum's plane, its compensation plane `<name>_comp`), in the walk's order.
///
/// # Safety
/// `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_walk_companions(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> std::ffi::c_int {
    guard(ctx, |c| {
        let pairs = walk_companions(c, walk)?
            .iter()
            .map(|(row, comp)| text_pair(row, Some(comp)))
            .collect::<Result<Vec<_>, _>>()?;
        put_pairs(&pairs, buf, cap, n)
    })
}

/// The body sinks of a loop.
fn raptor_ir_refs(ctx: &Ctx, lp: u32) -> R<Vec<u32>> {
    match ctx.field(lp, "body_sinks")? {
        Val::Tuple(v) | Val::List(v) => Ok(v
            .iter()
            .filter_map(|x| if let Val::Node(n) = x { Some(*n) } else { None })
            .collect()),
        _ => Ok(vec![]),
    }
}

/// The canonical walk of a kernel's sinks over the caller's declared slots, the compensation slot of every summing sample sum, the
/// finished-count slot when a sink finishes, and the branch-mix slot of every dispatch group that counts its branch mix (a second
/// pass, as the mix slots are named by the first walk's groups); the fused-steps plane and the planes `exempt` names are
/// exempt from the access class.
pub fn kernel_walk(
    ctx: &mut Ctx,
    sinks: &[u32],
    declared: &[(Key, Tt)],
    force: Option<&str>,
    exempt: &[String],
) -> R<u32> {
    let mut slots = declared.to_vec();
    slots.extend(companion_slots(ctx, sinks)?);
    let uint32 = Tt {
        dtype: "uint32".into(),
        shape: vec![],
        tag: None,
    };
    let finishing = sinks.iter().any(|&s| is_a(ctx, s, "Finish"));
    let finished: Vec<(Key, Tt)> = if finishing {
        vec![(("lookup".into(), FINISHED_PLANE.into()), uint32.clone())]
    } else {
        vec![]
    };
    let exempt = [&[FUSED_STEPS_PLANE.to_string()], exempt].concat();
    let all = [slots.clone(), finished.clone()].concat();
    let found = ctx.canonical(sinks, &all, force, &exempt)? as u32;
    let mixes: Vec<(Key, Tt)> = strategies(ctx, found)?
        .into_iter()
        .filter(|(_, s)| *s == Strategy::Auto)
        .map(|(group, _)| {
            (
                ("lookup".into(), crate::segment::mix_name(&group)),
                uint32.clone(),
            )
        })
        .collect();
    if mixes.is_empty() {
        return Ok(found);
    }
    let all = [slots, mixes, finished].concat();
    Ok(ctx.canonical(sinks, &all, force, &exempt)? as u32)
}

/// The canonical walk of a kernel's sinks as `kernel_walk` makes it, `declared_kt` the key text of the caller's declared slots
/// (a tuple of Slot records) and `force_access` NULL for none.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `sinks` is valid for `nsinks` nodes, the texts are
/// NUL-terminated and `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_cells_walk(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    declared_kt: *const c_char,
    force_access: *const c_char,
    out: *mut u32,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() || (sinks.is_null() && nsinks > 0) {
            return Err(Fault::BadArg("out or sinks is NULL".into()));
        }
        let given = if nsinks == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        let declared = declared_slots(Some(name_of(declared_kt, "declared_kt")?))?;
        let force = if force_access.is_null() {
            None
        } else {
            Some(name_of(force_access, "force_access")?)
        };
        unsafe { *out = kernel_walk(c, given, &declared, force, &[])? };
        Ok(())
    })
}
