//! Block sums made plain: the sum over the lanes of a key of a block-shaped payload is kept as one block for a team of threads when
//! the build asks for it and the block is big enough, and is otherwise written as the plain sum of its payload (the per-entry
//! accumulates of the keyed sum, or the cells of the all-lane sum).
use crate::cells::{build, stored};
use crate::place::Graph;
use crate::{is_a, R};
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::value::Val;
use std::collections::HashMap;

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
pub(crate) fn factors(ctx: &Ctx, payload: u32) -> Option<(u32, u32, i64)> {
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
fn ordered_case(ctx: &Ctx, payload: u32, contract: &str, max_dim: Option<i64>) -> bool {
    contract == "ordered"
        && is_float(ctx, payload)
        && rank(ctx, payload) <= 1
        && ctx.nodes[payload as usize]
            .shape
            .iter()
            .all(|&extent| max_dim.is_none_or(|limit| extent <= limit))
}

/// Whether the keyed sum of the payload can be one block by its structure alone.
fn block_shaped(ctx: &Ctx, payload: u32, contract: &str, max_dim: Option<i64>) -> bool {
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
        let compensated = is_a(ctx, sink, "AccumWrite")
            && matches!(
                ctx.str_field(sink, "contract")?.as_str(),
                "compensated" | "keyed_auto"
            )
            && !sinks.contains(&sink);
        if compensated {
            return refuse("a block sum made plain under a compensated contract");
        }
    }
    Ok(Some(out))
}
