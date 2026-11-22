//! Block sums made plain: the sum over the lanes of a key of a block-shaped payload is kept as one block for a team of threads when
//! the build asks for it and the block is big enough, and is otherwise written as the plain sum of its payload (the per-entry
//! accumulates of the keyed sum, or the cells of the all-lane sum).
use crate::cells::{build, stored};
use crate::place::Graph;
use crate::{flag_out, is_a, list_out, name_of, R};
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::HashMap;
use std::ffi::{c_char, c_int};

/// How a build holds the linear algebra of a lane.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum LinalgForm {
    /// A block is kept for a team when it is large.
    #[default]
    Auto,
    /// A block is always kept for a team.
    Team,
    /// Every sum is worked per entry.
    Thread,
}

/// The numbers of the target a block is judged by.
#[derive(Debug, Clone, Copy)]
pub struct BlockLimits {
    /// The element count above which a block is big enough for a team under the automatic form.
    pub rolled_fold_min: i64,
    /// The largest extent a team works on.
    pub team_max_dim: i64,
}

/// The element count of a node's value.
fn size(ctx: &Ctx, node: u32) -> i64 {
    ctx.nodes[node as usize].shape.iter().product()
}

/// The rank of a node's value.
fn rank(ctx: &Ctx, node: u32) -> usize {
    ctx.nodes[node as usize].shape.len()
}

/// Whether the node's element type is floating.
fn is_float(ctx: &Ctx, node: u32) -> bool {
    matches!(ctx.nodes[node as usize].dtype.as_str(), "f64" | "f32")
}

/// The kind of an op node, or none for any other.
fn op_kind(ctx: &Ctx, node: u32) -> Option<String> {
    is_a(ctx, node, "Op")
        .then(|| ctx.str_field(node, "kind").ok())
        .flatten()
}

/// The factors of a payload that is a sum over per-row rank-1 products, as (u, v, rows): an outer product, a vector times a scalar,
/// or the product of a transposed matrix with itself or with a vector of its rows (`rows` is how many rows a lane holds).
pub fn factors(ctx: &Ctx, payload: u32) -> Option<(u32, u32, i64)> {
    let kind = op_kind(ctx, payload)?;
    let ops = ctx.nodes[payload as usize].ops.clone();
    match (kind.as_str(), ops.len()) {
        ("outer", _) => Some((ops[0], ops[1], 1)),
        ("mul", 2) => match (rank(ctx, ops[0]), rank(ctx, ops[1])) {
            (1, 0) => Some((ops[0], ops[1], 1)),
            (0, 1) => Some((ops[1], ops[0], 1)),
            _ => None,
        },
        ("mm" | "mv", 2) => {
            let (transposed, other) = (ops[0], ops[1]);
            if op_kind(ctx, transposed).as_deref() != Some("transpose") {
                return None;
            }
            let matrix = ctx.nodes[transposed as usize].ops[0];
            if rank(ctx, matrix) != 2 {
                return None;
            }
            let rows = ctx.nodes[matrix as usize].shape[0];
            let fits = if kind == "mm" {
                other == matrix
            } else {
                ctx.nodes[other as usize].shape == [rows]
            };
            fits.then_some((matrix, if kind == "mm" { matrix } else { other }, rows))
        }
        _ => None,
    }
}

/// Whether the payload is a sum over per-row rank-1 products.
fn has_factors(ctx: &Ctx, payload: u32) -> bool {
    factors(ctx, payload).is_some()
}

/// Whether the ordered contract keeps the payload (a floating scalar or vector of at most `max_dim` elements) as one block.
pub fn ordered_case(ctx: &Ctx, payload: u32, contract: &str, max_dim: Option<i64>) -> bool {
    contract == "ordered"
        && is_float(ctx, payload)
        && rank(ctx, payload) <= 1
        && ctx.nodes[payload as usize]
            .shape
            .iter()
            .all(|&extent| max_dim.is_none_or(|limit| extent <= limit))
}

/// Whether the keyed sum of the payload can be one block by its structure alone.
pub fn block_shaped(ctx: &Ctx, payload: u32, contract: &str, max_dim: Option<i64>) -> bool {
    if ordered_case(ctx, payload, contract, max_dim) {
        return true;
    }
    is_float(ctx, payload)
        && ctx.nodes[payload as usize]
            .shape
            .iter()
            .all(|&extent| max_dim.is_none_or(|limit| extent <= limit))
        && has_factors(ctx, payload)
}

/// Whether a block of the payload stays one block under the build's form.
fn block_chosen(
    ctx: &Ctx,
    payload: u32,
    (form, limits): (LinalgForm, &BlockLimits),
    contract: &str,
) -> bool {
    let limit = Some(limits.team_max_dim);
    if block_shaped(ctx, payload, contract, None) && !block_shaped(ctx, payload, contract, limit) {
        return false;
    }
    ordered_case(ctx, payload, contract, limit)
        || (form != LinalgForm::Thread
            && (form == LinalgForm::Team || size(ctx, payload) > limits.rolled_fold_min))
}

/// The plain sum of a block: the unkeyed sum for an all-lane block, the keyed one otherwise.
fn plain_sum(ctx: &mut Ctx, block: u32) -> R<u32> {
    let ops = ctx.nodes[block as usize].ops.clone();
    let contract = ctx.str_field(block, "contract")?;
    let kept = ctx.field(block, "kept")?;
    let mut given = vec![
        ("op", Val::Str("sum".into())),
        ("value", Val::Node(ops[0])),
        ("contract", Val::Str(contract)),
    ];
    if kept != Val::Tuple(vec![]) {
        given.extend([
            ("key", Val::Node(ops[1])),
            ("count", Val::Node(ops[2])),
            ("kept", kept),
        ]);
    }
    build(ctx, "ReduceSamples", given)
}

/// The sinks with every block sum the form does not keep made its plain sum, or none when every block stays. A stored block becomes
/// the cell or per-entry sinks of its plain sum, and one the body reads becomes the reduction over samples.
pub fn plain_blocks(
    ctx: &mut Ctx,
    sinks: &[u32],
    form: LinalgForm,
    limits: &BlockLimits,
) -> R<Option<Vec<u32>>> {
    let graph = Graph::build(ctx, sinks)?;
    let mut mapping: HashMap<u32, u32> = HashMap::new();
    for &node in &graph.order {
        if !is_a(ctx, node, "BlockSum") {
            continue;
        }
        let payload = ctx.nodes[node as usize].ops[0];
        let kept = ctx.field(node, "kept")? != Val::Tuple(vec![]);
        let contract = if kept {
            ctx.str_field(node, "contract")?
        } else {
            "auto".into()
        };
        if !block_chosen(ctx, payload, (form, limits), &contract) {
            let plain = plain_sum(ctx, node)?;
            mapping.insert(node, plain);
        }
    }
    if mapping.is_empty() {
        return Ok(None);
    }
    let mut out = vec![];
    for &sink in sinks {
        if is_a(ctx, sink, "BlockSumWrite") {
            let block = ctx.nodes[sink as usize].ops[0];
            let name = ctx.str_field(sink, "name")?;
            match mapping.get(&block) {
                Some(&plain) => out.extend(stored(ctx, plain, &name)?),
                None => out.push(sink),
            }
        } else if is_a(ctx, sink, "Loop") {
            out.push(sink);
        } else {
            out.extend(ctx.substitute(&[sink], &mapping)?);
        }
    }
    for &sink in &out {
        if ctx.compensates(sink)? && !sinks.contains(&sink) {
            return refuse("a block sum made plain under a compensated contract");
        }
    }
    Ok(Some(out))
}

/// The one optional extent of a caller's list: none for an empty list, else the single entry.
fn limit_of(limit: *const i64, count: usize) -> R<Option<i64>> {
    match count {
        0 => Ok(None),
        1 if !limit.is_null() => Ok(Some(unsafe { *limit })),
        _ => Err(Fault::BadArg("limit holds no entry or one".into())),
    }
}

/// The factors `u` and `v` of a payload that is a sum over per-row rank-1 products, into `buf` (none for any other payload);
/// see `raptor_wings_front.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `buf` is valid for `cap` nodes and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_block_factors(
    ctx: *mut Ctx,
    payload: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        c.node(payload)?;
        let found = factors(c, payload).map_or(vec![], |(u, v, _)| vec![u, v]);
        list_out(buf, cap, n, &found)
    })
}

/// The number of rows a lane holds of a payload that has factors; refuses a payload that has none.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_block_factor_rows(ctx: *mut Ctx, payload: u32, out: *mut i64) -> c_int {
    guard(ctx, |c| {
        c.node(payload)?;
        let Some((_, _, rows)) = factors(c, payload) else {
            return refuse("the payload is no sum over per-row rank-1 products: it has no rows");
        };
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        unsafe { *out = rows }
        Ok(())
    })
}

/// Whether the ordered contract keeps the payload as one block (`*out` is 0 or 1).
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `contract` is a NUL-terminated string; `limit` is
/// valid for `nlimit` entries and `out` for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_block_ordered_case(
    ctx: *mut Ctx,
    payload: u32,
    contract: *const c_char,
    limit: *const i64,
    nlimit: usize,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        c.node(payload)?;
        let flag = ordered_case(
            c,
            payload,
            name_of(contract, "contract")?,
            limit_of(limit, nlimit)?,
        );
        flag_out(out, flag)
    })
}

/// Whether the keyed sum of the payload can be one block by its structure alone (`*out` is 0 or 1).
///
/// # Safety
/// As `rw_block_ordered_case`.
#[no_mangle]
pub unsafe extern "C" fn rw_block_shaped(
    ctx: *mut Ctx,
    payload: u32,
    contract: *const c_char,
    limit: *const i64,
    nlimit: usize,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        c.node(payload)?;
        let flag = block_shaped(
            c,
            payload,
            name_of(contract, "contract")?,
            limit_of(limit, nlimit)?,
        );
        flag_out(out, flag)
    })
}

/// The reduction over samples that is the keyed form of a block sum: its key and count (an all-lane block is at key 0 of 1).
fn keyed_reduction(ctx: &mut Ctx, block: u32) -> R<u32> {
    let ops = ctx.nodes[block as usize].ops.clone();
    let given = vec![
        ("op", Val::Str("sum".into())),
        ("value", Val::Node(ops[0])),
        ("key", Val::Node(ops[1])),
        ("count", Val::Node(ops[2])),
        ("contract", Val::Str(ctx.str_field(block, "contract")?)),
        ("kept", ctx.field(block, "kept")?),
    ];
    build(ctx, "ReduceSamples", given)
}

/// The sinks with every block-sum write replaced by the per-entry accumulates of its keyed form; any other sink stays.
pub fn expand_stores(ctx: &mut Ctx, sinks: &[u32]) -> R<Vec<u32>> {
    let mut out = vec![];
    for &sink in sinks {
        if is_a(ctx, sink, "BlockSumWrite") {
            let block = ctx.nodes[sink as usize].ops[0];
            let name = ctx.str_field(sink, "name")?;
            let reduction = keyed_reduction(ctx, block)?;
            out.extend(stored(ctx, reduction, &name)?);
        } else {
            out.push(sink);
        }
    }
    Ok(out)
}

/// The sinks with every block-sum write expanded into its accumulates, in `buf`; see `expand_stores`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `sinks` is valid for `nsinks` nodes, `buf` for `cap`
/// nodes and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_blocks_expand_stores(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let given = if nsinks == 0 {
            &[][..]
        } else if sinks.is_null() {
            return Err(Fault::BadArg("sinks is NULL".into()));
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        for &sink in given {
            c.node(sink)?;
        }
        let done = expand_stores(c, given)?;
        list_out(buf, cap, n, &done)
    })
}
