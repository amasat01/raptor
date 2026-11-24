//! The team stage of a block sum: the sum over the lanes of a key of per-row rank-1 products into one dense block per key. The
//! per-lane stage before it stores the factors of every row into scratch, and the team stage adds the rows of each key in lane
//! order (one block per key), into the plane the sum is stored to, or into a block of its own that a later stage reads at the
//! lane's own key.
use crate::blocks::factors;
use crate::stages::Keyed;
use crate::team::{
    read, size_of, threads_for, Grid, Plane, Step, TeamLimits, TeamSpec, Unit, PLANE_PREFIX,
};
use crate::{is_a, R};
use raptor_ir::record::{refuse, Ctx};
use std::collections::HashMap;

/// The blocks of a partial-sum stage that add one key's rows, each owning 1024 of the 65536 virtual lanes. The writers read
/// the count back from the width of the partials plane, so this is the one place that says it.
const GROUPS: i64 = 64;

/// One block sum of a kernel and what the plan of its stages needs of it.
#[derive(Debug, Clone)]
pub struct Block {
    /// The block sum node.
    pub node: u32,
    /// The number that names its planes.
    pub index: usize,
    /// The plane a stored block commits to.
    pub out: Option<String>,
    /// Whether the body reads the block back.
    pub read_back: bool,
    /// The input planes the stage reads in place (role and leaf), when it may.
    pub in_place: Vec<(&'static str, u32)>,
}

/// The factors of a block's payload; an ordered sum of a plain scalar or vector is the payload alone.
fn factors_of(ctx: &Ctx, block: u32) -> (u32, Option<u32>, i64) {
    let payload = ctx.nodes[block as usize].ops[0];
    factors(ctx, payload).map_or((payload, None, 1), |(u, v, rows)| (u, Some(v), rows))
}

/// The last extent of a value's shape, 1 for a scalar.
fn last(ctx: &Ctx, node: u32) -> i64 {
    ctx.nodes[node as usize].shape.last().copied().unwrap_or(1)
}

/// The width of the first factor, of the second, and the rows a lane holds.
fn widths(ctx: &Ctx, block: u32) -> (i64, i64, i64) {
    let (u, v, rows) = factors_of(ctx, block);
    let Some(v) = v else {
        return (last(ctx, u), 1, rows);
    };
    let shape = &ctx.nodes[v as usize].shape;
    let wide = shape.len() == 2 || (shape.len() == 1 && rows == 1);
    (last(ctx, u), if wide { last(ctx, v) } else { 1 }, rows)
}

/// The factors of a stored sum of scalars that are a bare read of an input plane, which its stage reads in place instead of from
/// a copy in scratch (the lane's slice is one element).
fn plane_reads(
    ctx: &Ctx,
    block: u32,
    out: Option<&String>,
    read_back: bool,
) -> R<Vec<(&'static str, u32)>> {
    let (p, q, rows) = widths(ctx, block);
    if out.is_none() || read_back || p * q != 1 || rows > 1 {
        return Ok(vec![]);
    }
    let (u, v, _) = factors_of(ctx, block);
    let mut found = vec![];
    for (role, factor) in [("u", Some(u)), ("v", v)] {
        let Some(factor) = factor else { continue };
        let bare = is_a(ctx, factor, "Leaf")
            && ctx.str_field(factor, "role")? == "plane_in"
            && ctx.nodes[factor as usize].shape.is_empty();
        if bare {
            found.push((role, factor));
        }
    }
    Ok(found)
}

/// The blocks among the reductions of a kernel, each with the plane it is stored to and whether the body reads it back.
pub fn blocks(ctx: &Ctx, sinks: &[u32], order: &[u32], in_place: bool) -> R<Vec<Block>> {
    let mut stored: HashMap<u32, String> = HashMap::new();
    for &sink in sinks.iter().filter(|&&s| is_a(ctx, s, "BlockSumWrite")) {
        let (block, name) = (
            ctx.nodes[sink as usize].ops[0],
            ctx.str_field(sink, "name")?,
        );
        if stored
            .insert(block, name.clone())
            .is_some_and(|old| old != name)
        {
            return refuse(format!("one sum over the lanes of a key is stored to two planes, {:?} and {name:?}: store it once", stored[&block]));
        }
    }
    let mut found = vec![];
    for &node in order.iter().filter(|&&n| is_a(ctx, n, "BlockSum")) {
        let read_back = order.iter().any(|&user| {
            !is_a(ctx, user, "BlockSumWrite") && ctx.nodes[user as usize].ops.contains(&node)
        });
        let out = stored.get(&node).cloned();
        let reads = if in_place {
            plane_reads(ctx, node, out.as_ref(), read_back)?
        } else {
            vec![]
        };
        found.push(Block {
            node,
            index: found.len(),
            out,
            read_back,
            in_place: reads,
        });
    }
    Ok(found)
}

/// The name of one of a block's planes.
fn plane_name(block: &Block, role: &str) -> String {
    format!("{PLANE_PREFIX}b{}_{role}", block.index)
}

/// The keyed plane a block sum accumulates into: one key's payload is the block.
pub fn keyed(ctx: &Ctx, block: &Block) -> R<Keyed> {
    let node = &ctx.nodes[block.node as usize];
    let kept = crate::stages::kept_of(ctx, block.node)?;
    Ok(Keyed {
        name: plane_name(block, "g"),
        width: size_of(ctx, block.node),
        kept,
        payload: node.shape.clone(),
    })
}

/// The scratch of a block: the factors (`rows * p` and `rows * q` per lane) and, when the body reads the sum back, the block `g`.
fn planes(ctx: &Ctx, block: &Block, with_g: bool) -> Vec<Plane> {
    let (p, q, rows) = widths(ctx, block.node);
    let (u, v, _) = factors_of(ctx, block.node);
    let held = |role: &str| block.in_place.iter().any(|(r, _)| *r == role);
    let mut out = vec![];
    if !held("u") {
        out.push(Plane::per_lane(plane_name(block, "u"), "real", rows * p));
    }
    if v.is_some_and(|v| v != u) && !held("v") {
        out.push(Plane::per_lane(plane_name(block, "v"), "real", rows * q));
    }
    if with_g {
        out.push(Plane {
            name: plane_name(block, "g"),
            dtype: "real",
            per_lane: p * q,
            unit: Unit::Key,
            fill: None,
        });
    }
    out
}

/// The value of a block at the lane's own key: the block of the key in the plane `g` of its team stage.
pub fn block_read(ctx: &mut Ctx, block: &Block) -> R<u32> {
    let key = ctx.nodes[block.node as usize].ops[1];
    read(ctx, &plane_name(block, "g"), (block.node, Some(key)))
}

/// The stores of a block's factors into their scratch planes by the per-lane stage (none for a factor read in place).
pub fn factor_stores(ctx: &mut Ctx, block: &Block) -> R<Vec<u32>> {
    let (u, v, _) = factors_of(ctx, block.node);
    let mut stores = vec![("u", u)];
    if let Some(v) = v.filter(|&v| v != u) {
        stores.push(("v", v));
    }
    let mut out = vec![];
    for (role, value) in stores
        .into_iter()
        .filter(|(role, _)| !block.in_place.iter().any(|(r, _)| r == role))
    {
        out.push(crate::team::team_store(
            ctx,
            &plane_name(block, role),
            value,
        )?);
    }
    Ok(out)
}

/// The team stages of a block sum: one call that sums the rows of each key into its output plane, or, for a plain stored sum of
/// scalars, a stage of blocks that add fixed partials and one that folds them.
pub fn block_spec(ctx: &Ctx, block: &Block, limits: &TeamLimits) -> R<Vec<TeamSpec>> {
    let (p, q, rows) = widths(ctx, block.node);
    let direct = block.out.is_some() && !block.read_back;
    let scratch = planes(ctx, block, !direct);
    let mut names: HashMap<&str, String> = scratch
        .iter()
        .map(|plane| {
            (
                plane.name.rsplit('_').next().unwrap_or(""),
                plane.name.clone(),
            )
        })
        .collect();
    for (role, leaf) in &block.in_place {
        names.insert(role, ctx.str_field(*leaf, "name")?);
    }
    let ones = factors_of(ctx, block.node).1.is_none();
    let mut roles = vec![("u".to_string(), names["u"].clone())];
    if !ones {
        roles.push(("v".into(), names.get("v").unwrap_or(&names["u"]).clone()));
    }
    roles.extend(block.out.iter().map(|out| ("out".to_string(), out.clone())));
    if !direct {
        roles.push(("g".into(), names["g"].clone()));
    }
    let grid = |groups: i64| -> R<Grid> {
        Ok(Grid::Keys {
            keyed: keyed(ctx, block)?,
            groups,
        })
    };
    let reads: Vec<u32> = block.in_place.iter().map(|(_, leaf)| *leaf).collect();
    // the stages that read the lanes find a key's rows by the strides when its lanes are not one contiguous range
    let key_axes = crate::lanes::key_axes(ctx, block.node)?;
    if direct && p * q == 1 && rows <= 1 {
        let part = Plane {
            name: plane_name(block, "gp"),
            dtype: "partial",
            per_lane: GROUPS,
            unit: Unit::Key,
            fill: None,
        };
        let with = |mut roles: Vec<(String, String)>| {
            roles.push(("gp".into(), part.name.clone()));
            roles
        };
        let all: Vec<Plane> = scratch.iter().cloned().chain([part.clone()]).collect();
        let first = TeamSpec {
            threads: threads_for(1, limits),
            steps: vec![Step {
                op: "block_part",
                n: p,
                k: q,
                m: rows,
                max_chunks: 0,
                key_axes: key_axes.clone(),
                planes: with(roles.clone()),
            }],
            planes: all.clone(),
            grid: grid(GROUPS)?,
            out: None,
            reads,
        };
        let kept: Vec<(String, String)> = roles
            .into_iter()
            .filter(|(role, _)| !block.in_place.iter().any(|(r, _)| r == role))
            .collect();
        let last = TeamSpec {
            threads: threads_for(1, limits),
            steps: vec![Step {
                op: "block_sum",
                n: p,
                k: q,
                m: rows,
                max_chunks: 0,
                key_axes: None,
                planes: with(kept),
            }],
            planes: all,
            grid: grid(1)?,
            out: block.out.clone(),
            reads: vec![],
        };
        return Ok(vec![first, last]);
    }
    Ok(vec![TeamSpec {
        threads: threads_for(p.max(q), limits),
        steps: vec![Step {
            op: "block_sum",
            n: p,
            k: q,
            m: rows,
            max_chunks: 0,
            key_axes,
            planes: roles,
        }],
        planes: scratch,
        grid: grid(1)?,
        out: block.out.clone(),
        reads,
    }])
}
