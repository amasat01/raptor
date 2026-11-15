//! The split of a kernel at its reductions over samples: a value reduced over samples and read later is a launch boundary, so the
//! kernel becomes stages run in sequence. Stage `k` holds the sinks whose every reduction is a cell an earlier stage produced; the
//! stage that makes a reduction stores it as cells (rows of partial sums), and a later stage reads the finished cell as a one-cell
//! plane. A reduction over some lane axes is a keyed plane instead. A per-sample value a later stage needs is computed again there;
//! each stage records which values it recomputes.
use crate::blockteam::{self, Block};
use crate::cells::{build, cell_read, stored};
use crate::place::{kind_of, Graph};
use crate::planes::type_parts;
use crate::scatter;
use crate::team::{self, Grid, Group, TeamLimits, TeamSpec};
use crate::{is_a, R};
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::{HashMap, HashSet};

/// The prefix of a cell made for a reduction the body reads but never stores to an output of its own.
const CUT_PREFIX: &str = "cut";

/// A keyed plane the runtime owns between stages: `width` payload entries over the kept lane axes.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyed {
    /// The plane's name.
    pub name: String,
    /// The entries of one key's payload.
    pub width: i64,
    /// The kept lane axes, each a label and an extent when it has them.
    pub kept: Vec<(Option<String>, Option<i64>)>,
    /// The shape of one key's payload.
    pub payload: Vec<i64>,
}

/// A cell: the reduction stored in rows, one plane per column.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    /// The cell's name.
    pub name: String,
    /// The width of the cell.
    pub columns: i64,
    /// The reduction's operation.
    pub op: String,
    /// Whether a later stage reads the cell back (an output the kernel declares is then the caller's, not the runtime's).
    pub internal: bool,
}

/// One launch of a split kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    /// The stage's sinks.
    pub sinks: Vec<u32>,
    /// The cells it reads.
    pub cells_in: Vec<String>,
    /// The cells it stores.
    pub cells_out: Vec<String>,
    /// The kinds of the per-sample values it computes again.
    pub recomputed: Vec<String>,
    /// The team the stage runs on, when it is a team stage or a stage whose threads copy operands into team scratch.
    pub team: Option<TeamSpec>,
}

/// What the split reads besides the IR: the build's routing of keyed sums and the limits of a team.
#[derive(Debug, Clone, Copy)]
pub struct StageFacts {
    /// Whether keyed accumulates take the ordered stages.
    pub routed: bool,
    /// The numbers a team plan reads.
    pub team: TeamLimits,
    /// The most chunks an ordered scatter cuts the lanes into, and the team of each of its stages.
    pub scatter: (i64, i64),
    /// Whether a reduction's input that is a bare read of an input plane may be read in place by its team stage (an index-set build
    /// compacts its inputs, so it copies).
    pub in_place: bool,
}

/// The stages of a kernel and the cells and keyed planes between them.
#[derive(Debug, Clone, PartialEq)]
pub struct Cut {
    /// The stages in launch order.
    pub stages: Vec<Stage>,
    /// The cells the stages store, in first-stored order.
    pub cells: Vec<Cell>,
    /// The keyed planes the stages accumulate into.
    pub keyed: Vec<Keyed>,
}

/// What the split makes of a kernel's sinks.
#[derive(Debug, Clone, PartialEq)]
pub enum Stages {
    /// The body reads no reduction over samples: one kernel, unchanged.
    Whole,
    /// The kernel is cut into stages.
    Cut(Cut),
    /// The body folds an ordered log, whose team stage is not planned.
    Team,
}

/// Whether the node reduces over samples: a reduction or a block sum.
fn reduces(ctx: &Ctx, node: u32) -> bool {
    is_a(ctx, node, "ReduceSamples") || is_a(ctx, node, "BlockSum")
}

/// The depth of every node: the number of reductions over samples and members of team linear algebra on its deepest path.
fn depths(ctx: &Ctx, order: &[u32], members: &HashSet<u32>) -> R<HashMap<u32, usize>> {
    let mut depth = HashMap::new();
    for &node in order {
        let mut below = 0;
        for c in &ctx.nodes[node as usize].ops {
            below = below.max(operand_depth(ctx, &depth, *c)?);
        }
        let own = reduces(ctx, node) || members.contains(&node);
        depth.insert(node, below + usize::from(own));
    }
    Ok(depth)
}

/// The depth of one operand. A loop's placeholders (its index and its carried values) stand inside the loop body and are not in
/// the order, so they are at depth 0; any other operand without a depth is refused by name.
fn operand_depth(ctx: &Ctx, depth: &HashMap<u32, usize>, c: u32) -> R<usize> {
    match depth.get(&c) {
        Some(d) => Ok(*d),
        None if is_a(ctx, c, "LoopIndex") || is_a(ctx, c, "LoopCarry") => Ok(0),
        None => refuse(format!("node {c} is read before its depth is known")),
    }
}

/// A plane name for a cell that no node of the kernel uses.
fn fresh(taken: &mut HashSet<String>, count: usize) -> String {
    let mut name = format!("{CUT_PREFIX}{count}");
    while taken.contains(&name) {
        name.push('_');
    }
    taken.insert(name.clone());
    name
}

/// The row planes of a cell: its name, or `name_0 ..`.
fn planes_of(cell: &str, columns: usize) -> Vec<String> {
    if columns == 1 {
        vec![cell.to_string()]
    } else {
        (0..columns).map(|k| format!("{cell}_{k}")).collect()
    }
}

/// The names of the per-sample planes read under the roots.
fn reads(ctx: &Ctx, roots: &[u32]) -> R<HashSet<String>> {
    let mut found = HashSet::new();
    for node in Graph::build(ctx, roots)?.order {
        if is_a(ctx, node, "Leaf") {
            let role = ctx.str_field(node, "role")?;
            if !matches!(role.as_str(), "uniform" | "lookup" | "nsamples") {
                found.insert(ctx.str_field(node, "name")?);
            }
        }
    }
    Ok(found)
}

/// Whether the value depends on the sample.
fn per_sample(ctx: &Ctx, node: u32) -> R<bool> {
    let (mut stack, mut seen) = (vec![node], HashSet::new());
    while let Some(next) = stack.pop() {
        if !seen.insert(next) {
            continue;
        }
        let leaf = is_a(ctx, next, "Leaf")
            && !matches!(
                ctx.str_field(next, "role")?.as_str(),
                "uniform" | "nsamples" | "lookup" | "wide_in"
            );
        if is_a(ctx, next, "SampleIndex")
            || leaf
            || (is_a(ctx, next, "At") && !ctx.is_param_cell(next)?)
        {
            return Ok(true);
        }
        stack.extend(ctx.nodes[next as usize].ops.iter().copied());
    }
    Ok(false)
}

/// The kept axes of a keyed reduction, as the record states them.
pub(crate) fn kept_of(ctx: &Ctx, node: u32) -> R<Vec<(Option<String>, Option<i64>)>> {
    let Val::Tuple(axes) = ctx.field(node, "kept")? else {
        return Ok(vec![]);
    };
    let one = |axis: &Val| match axis {
        Val::Tuple(pair) if pair.len() == 2 => (
            match &pair[0] {
                Val::Str(label) => Some(label.clone()),
                _ => None,
            },
            match &pair[1] {
                Val::Int(extent) => i64::try_from(*extent).ok(),
                _ => None,
            },
        ),
        _ => (None, None),
    };
    Ok(axes.iter().map(one).collect())
}

/// Whether the sink is an ordered scatter: a keyed accumulate of the ordered contract, or of any contract when the build routes
/// them to the ordered stages.
pub(crate) fn is_ordered_sink(ctx: &Ctx, sink: u32, routed: bool) -> R<bool> {
    let keyed = is_a(ctx, sink, "AccumWrite") && ctx.nodes[sink as usize].ops.len() > 1;
    Ok(keyed
        && (ctx.str_field(sink, "contract")? == "ordered"
            || routed
                && matches!(
                    ctx.str_field(sink, "contract")?.as_str(),
                    "auto" | "compensated" | "keyed_auto"
                )))
}

/// The cells the kernel's own sample sums store, by the reduced value and the operation.
fn own_cells(ctx: &Ctx, sinks: &[u32]) -> R<HashMap<(u32, String), String>> {
    let mut found = HashMap::new();
    for &sink in sinks {
        if is_a(ctx, sink, "SampleSum") {
            let value = ctx.nodes[sink as usize].ops[0];
            let columns = ctx.field(sink, "columns")?;
            let from = if columns == Val::Int(1) {
                value
            } else {
                ctx.nodes[value as usize].ops[0]
            };
            let op = ctx.str_field(sink, "op")?;
            found.insert((from, op), ctx.str_field(sink, "cell")?);
        }
    }
    Ok(found)
}

/// The stage each sink lands in, moved later when a later stage reads the plane the sink writes.
fn settle(
    ctx: &Ctx,
    sinks: &[u32],
    cuts: &[(u32, bool)],
    depth: &HashMap<u32, usize>,
) -> R<HashMap<u32, usize>> {
    // a store into team scratch lands after the reductions its value reads (the store itself is not among the depths' nodes)
    let held = |sink: u32| match is_a(ctx, sink, "TeamStore") {
        true => ctx.nodes[sink as usize].ops[0],
        false => sink,
    };
    let mut stage_of: HashMap<u32, usize> =
        sinks.iter().map(|&s| (s, depth[&held(s)] + 1)).collect();
    let mut readers: Vec<(u32, HashSet<String>)> = vec![];
    for &sink in sinks {
        readers.push((sink, reads(ctx, &[sink])?));
    }
    for &(cut, made) in cuts {
        stage_of.insert(cut, depth[&cut]);
        if made {
            let operands = ctx.nodes[cut as usize].ops.clone();
            readers.push((cut, reads(ctx, &operands)?));
        }
    }
    loop {
        let mut last: HashMap<&str, usize> = HashMap::new();
        for (owner, names) in &readers {
            for name in names {
                let entry = last.entry(name).or_insert(0);
                *entry = (*entry).max(stage_of[owner]);
            }
        }
        let mut moved = vec![];
        for &sink in sinks {
            let name = ctx.str_field(sink, "name").unwrap_or_default();
            let later = last.get(name.as_str()).copied().unwrap_or(0);
            if !is_a(ctx, sink, "SampleSum") && later > stage_of[&sink] {
                moved.push((sink, later));
            }
        }
        if moved.is_empty() {
            return Ok(stage_of);
        }
        stage_of.extend(moved);
    }
}

/// The nodes under the roots that stop at the nodes of `stop`, children first.
fn order_under(ctx: &Ctx, roots: &[u32], stop: &HashMap<u32, u32>) -> Vec<u32> {
    let (mut out, mut seen) = (vec![], HashSet::new());
    let mut stack: Vec<(u32, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
    while let Some((node, done)) = stack.pop() {
        if seen.contains(&node) {
            continue;
        }
        if done {
            seen.insert(node);
            out.push(node);
            continue;
        }
        stack.push((node, true));
        if !stop.contains_key(&node) {
            let below = ctx.nodes[node as usize].ops.iter().rev();
            stack.extend(below.filter(|c| !seen.contains(*c)).map(|&c| (c, false)));
        }
    }
    out
}

/// The cut of a kernel's sinks into stages (including the team stages of linear algebra, block sums and ordered scatters), the
/// whole kernel when it reads no reduction over samples and needs no team, or a note that it folds an ordered log.
pub fn stages(ctx: &mut Ctx, sinks: &[u32], facts: &StageFacts) -> R<Stages> {
    let (sinks, scattered) = scatter::logged(ctx, sinks, facts.routed)?;
    let sinks = &sinks[..];
    let order = Graph::build(ctx, sinks)?.order;
    let mut cuts = vec![];
    for &node in &order {
        if reduces(ctx, node) {
            cuts.push(node);
        }
    }
    if sinks.iter().any(|&sink| is_a(ctx, sink, "LogStore")) {
        return Ok(Stages::Team);
    }
    let groups = team::groups(ctx, &order, &facts.team)?;
    if cuts.is_empty() && groups.is_empty() && scattered.is_empty() {
        return Ok(Stages::Whole);
    }
    for &cut in cuts.iter().filter(|&&cut| !is_a(ctx, cut, "BlockSum")) {
        let keyed = matches!(ctx.field(cut, "key")?, Val::Node(_));
        if keyed
            && (ctx.str_field(cut, "op")? != "sum" || ctx.str_field(cut, "contract")? == "ordered")
        {
            return refuse(
                "a reduction over some lane axes that is not a plain sum, read back in the body, is not built",
            );
        }
    }
    cut_stages(ctx, (sinks, &order, &cuts), (groups, scattered), facts).map(Stages::Cut)
}

/// The cell each cut is read through, and for each cut whether the cell is a fresh one.
type CellNames = (HashMap<u32, String>, Vec<(u32, bool)>);

/// The cells the cuts are read through: a cell the kernel's own sample sum stores, else a fresh one.
fn name_cells(ctx: &Ctx, sinks: &[u32], order: &[u32], cuts: &[u32]) -> R<CellNames> {
    let own = own_cells(ctx, sinks)?;
    let mut taken: HashSet<String> = order
        .iter()
        .filter_map(|&n| ctx.str_field(n, "name").ok())
        .collect();
    let (mut cell_of, mut marks, mut made) = (HashMap::new(), vec![], 0);
    for &cut in cuts.iter().filter(|&&cut| !is_a(ctx, cut, "BlockSum")) {
        let keyed = matches!(ctx.field(cut, "key")?, Val::Node(_));
        let found = if keyed {
            None
        } else {
            own.get(&(ctx.nodes[cut as usize].ops[0], ctx.str_field(cut, "op")?))
        };
        let fresh_cell = found.is_none();
        let name = match found {
            Some(name) => name.clone(),
            None => {
                made += 1;
                fresh(&mut taken, made - 1)
            }
        };
        cell_of.insert(cut, name);
        marks.push((cut, fresh_cell));
    }
    Ok((cell_of, marks))
}

/// The kernel's sinks with the stores the team stages need (the operands of linear algebra, the factors of block sums) added and
/// the block sum writes taken out, and for each member of team linear algebra the node that stands for its result afterwards.
fn team_inputs(
    ctx: &mut Ctx,
    given: &[u32],
    (groups, blocks): (&[Group], &[Block]),
) -> R<(Vec<u32>, HashMap<u32, u32>)> {
    let mut sinks: Vec<u32> = given
        .iter()
        .copied()
        .filter(|&s| !is_a(ctx, s, "BlockSumWrite"))
        .collect();
    let mut out_node = HashMap::new();
    for group in groups {
        for (value, plane) in team::stores(ctx, group) {
            sinks.push(team::team_store(ctx, &plane, value)?);
        }
        for &member in &group.members {
            out_node.insert(member, team::output(ctx, group, member)?);
        }
    }
    for block in blocks {
        sinks.extend(blockteam::factor_stores(ctx, block)?);
    }
    Ok((sinks, out_node))
}

/// The keyed planes of the reductions over some lane axes by cell name, and the cell each row plane of the other reductions
/// belongs to.
fn cell_planes(
    ctx: &Ctx,
    cuts: &[u32],
    cell_of: &HashMap<u32, String>,
) -> R<(HashMap<String, Keyed>, HashMap<String, String>)> {
    let mut keyed_of: HashMap<String, Keyed> = HashMap::new();
    let mut cell_plane: HashMap<String, String> = HashMap::new();
    for &cut in cuts.iter().filter(|&&cut| !is_a(ctx, cut, "BlockSum")) {
        let shape = type_parts(ctx, cut).shape;
        let width = shape.first().copied().unwrap_or(1);
        let cell = &cell_of[&cut];
        if matches!(ctx.field(cut, "key")?, Val::Node(_)) {
            let kept = kept_of(ctx, cut)?;
            let found = Keyed {
                name: cell.clone(),
                width,
                kept,
                payload: shape,
            };
            keyed_of.insert(cell.clone(), found);
        } else {
            for plane in planes_of(cell, width as usize) {
                cell_plane.insert(plane, cell.clone());
            }
        }
    }
    Ok((keyed_of, cell_plane))
}

/// Adds the three stages that add the logs of the ordered scatter into `target`, a lane holding `slots` log slots.
fn scatter_stages(
    ctx: &mut Ctx,
    (target, slots): (&str, i64),
    limits: (i64, i64),
    out: &mut Vec<Stage>,
) -> R<()> {
    for spec in scatter::stage_specs(target, slots, limits) {
        out.push(team_stage(ctx, spec)?);
    }
    Ok(())
}

/// A launch of a team stage: its sinks are markers that bind the scratch planes it works on.
fn team_stage(ctx: &mut Ctx, spec: TeamSpec) -> R<Stage> {
    let sinks = team::markers(ctx, &spec)?;
    Ok(Stage {
        sinks,
        cells_in: vec![],
        cells_out: vec![],
        recomputed: vec![],
        team: Some(spec),
    })
}

/// The stages of a kernel with plain reductions over samples.
fn cut_stages(
    ctx: &mut Ctx,
    (given, order, cuts): (&[u32], &[u32], &[u32]),
    (groups, scattered): (Vec<Group>, scatter::Slots),
    facts: &StageFacts,
) -> R<Cut> {
    let (cell_of, marks) = name_cells(ctx, given, order, cuts)?;
    let blocks = blockteam::blocks(ctx, given, order, facts.in_place)?;
    let (sinks, out_node) = team_inputs(ctx, given, (&groups, &blocks))?;
    let sinks = &sinks[..];
    let members: HashSet<u32> = out_node.keys().copied().collect();
    let depth = depths(ctx, order, &members)?;
    let (keyed_of, cell_plane) = cell_planes(ctx, cuts, &cell_of)?;
    let stage_of = settle(ctx, sinks, &marks, &depth)?;
    let count = stage_of
        .values()
        .chain(cuts.iter().map(|c| &depth[c]))
        .chain(members.iter().map(|m| &depth[m]))
        .copied()
        .max()
        .unwrap_or(0);
    let made: HashSet<u32> = marks.iter().filter(|m| m.1).map(|m| m.0).collect();
    let mut seen_before = HashSet::new();
    let (mut keys, mut factored) = (HashMap::new(), vec![]);
    let mut out = vec![];
    let (mut cells, mut keyed_planes): (Vec<Cell>, Vec<Keyed>) = (vec![], vec![]);
    for k in 1..=count {
        let mut own: Vec<u32> = sinks.iter().copied().filter(|s| stage_of[s] == k).collect();
        for &cut in cuts {
            if made.contains(&cut) && depth[&cut] == k {
                let stores = stored(ctx, cut, &cell_of[&cut])?;
                own.extend(with_companions(ctx, stores)?);
            }
        }
        let mut mapping = HashMap::new();
        for &cut in cuts {
            if depth[&cut] < k {
                let read = match blocks.iter().find(|b| b.node == cut) {
                    Some(block) => blockteam::block_read(ctx, block)?,
                    None => cell_read(ctx, cut, &cell_of[&cut])?,
                };
                mapping.insert(cut, read);
            }
        }
        mapping.extend(
            out_node
                .iter()
                .filter(|(m, _)| depth[m] < k)
                .map(|(m, n)| (*m, *n)),
        );
        let new = swap(ctx, &own, &mapping)?;
        let original = order_under(ctx, &own, &mapping);
        structural_keys(ctx, &original, &mut keys);
        let recomputed = recomputed_kinds(ctx, &original, &own, (&keys, &seen_before))?;
        seen_before.extend(original.iter().map(|n| keys[n]));
        let Edges {
            out: cells_out,
            read: cells_in,
            keyed: keyed_out,
        } = edges(ctx, &new, &cell_plane, &keyed_of)?;
        for &sink in &new {
            note_cell(ctx, sink, &mut cells)?;
        }
        for name in &keyed_out {
            if !keyed_planes.iter().any(|p| &p.name == name) {
                keyed_planes.push(keyed_of[name].clone());
            }
        }
        let (new, copy) = copy_stage(ctx, new, &facts.team)?;
        if !new.is_empty() || !cells_in.is_empty() || !cells_out.is_empty() {
            out.push(Stage {
                sinks: new,
                cells_in,
                cells_out,
                recomputed,
                team: copy,
            });
        }
        if let Some(spec) = team::spec_of(ctx, &groups, k, &depth, (&mut factored, &facts.team))? {
            out.push(team_stage(ctx, spec)?);
        }
        for block in blocks.iter().filter(|b| depth[&b.node] == k) {
            for spec in blockteam::block_spec(ctx, block, &facts.team)? {
                out.push(team_stage(ctx, spec)?);
            }
        }
    }
    for (target, slots) in &scattered {
        scatter_stages(ctx, (target, *slots), facts.scatter, &mut out)?;
    }
    for cell in &mut cells {
        cell.internal = out.iter().any(|st| st.cells_in.contains(&cell.name));
    }
    Ok(Cut {
        stages: out,
        cells,
        keyed: keyed_planes,
    })
}

/// The stores and, for each compensated keyed accumulate, the zero commit that binds its companion plane at the same key.
fn with_companions(ctx: &mut Ctx, stores: Vec<u32>) -> R<Vec<u32>> {
    let mut out = stores.clone();
    for sink in stores {
        let compensated = is_a(ctx, sink, "AccumWrite")
            && matches!(
                ctx.str_field(sink, "contract")?.as_str(),
                "compensated" | "keyed_auto"
            );
        if compensated {
            let parts = type_parts(ctx, sink);
            let tt = raptor_ir::value::ttype_val(&parts.dtype, &parts.shape, parts.tag.as_deref());
            let zero = build(
                ctx,
                "Const",
                vec![("literal", Val::Float(0.0)), ("ttype", tt.clone())],
            )?;
            let name = format!("{}_c", ctx.str_field(sink, "name")?);
            let index = ctx.nodes[sink as usize].ops[1];
            let given = vec![
                ("name", Val::Str(name)),
                ("value", Val::Node(zero)),
                ("index", Val::Node(index)),
                ("ttype", tt),
            ];
            out.push(build(ctx, "AccumWrite", given)?);
        }
    }
    Ok(out)
}

/// The sinks with the nodes of `mapping` replaced under them.
fn swap(ctx: &mut Ctx, sinks: &[u32], mapping: &HashMap<u32, u32>) -> R<Vec<u32>> {
    if mapping.is_empty() {
        return Ok(sinks.to_vec());
    }
    let mut out = vec![];
    for &sink in sinks {
        if is_a(ctx, sink, "Loop") {
            return refuse(
                "a kernel that reads a cell cannot also commit a gradient per iteration of a loop",
            );
        }
        let value = ctx.nodes[sink as usize].ops[0];
        let swapped = ctx.substitute(&[value], mapping)?;
        out.push(ctx.with_value(sink, swapped[0])?);
    }
    Ok(out)
}

/// The structural keys of the nodes (children first) added to `keys`: two nodes of one structure have one key, whichever handle
/// they were made under.
fn structural_keys(ctx: &Ctx, order: &[u32], keys: &mut HashMap<u32, u64>) {
    use std::hash::{Hash, Hasher};
    for &node in order {
        if keys.contains_key(&node) {
            continue;
        }
        let record = &ctx.nodes[node as usize];
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (record.class, &record.dtype, &record.shape, &record.tag).hash(&mut hasher);
        let fields: Vec<String> = record
            .fields
            .iter()
            .map(|f| match f {
                Val::Node(h) => format!("n{:?}", keys.get(h)),
                other => format!("{other:?}"),
            })
            .collect();
        fields.hash(&mut hasher);
        for child in &record.ops {
            keys.get(child).hash(&mut hasher);
        }
        keys.insert(node, hasher.finish());
    }
}

/// The kinds of the values a stage computes again: per-sample values under its sinks that an earlier stage computed.
fn recomputed_kinds(
    ctx: &Ctx,
    original: &[u32],
    own: &[u32],
    (keys, seen_before): (&HashMap<u32, u64>, &HashSet<u64>),
) -> R<Vec<String>> {
    let mut kinds = vec![];
    for &node in original {
        let leafy = is_a(ctx, node, "Leaf") || is_a(ctx, node, "Const");
        let interior = !leafy
            && !reduces(ctx, node)
            && !ctx.is_param_cell(node)?
            && !ctx.nodes[node as usize].ops.is_empty()
            && !own.contains(&node);
        if interior && seen_before.contains(&keys[&node]) && per_sample(ctx, node)? {
            kinds.push(kind_of(ctx, node)?.unwrap_or_default());
        }
    }
    Ok(kinds)
}

/// What a stage exchanges with the others: the cells it stores and reads and the keyed planes it accumulates into, each once in
/// first-seen order.
struct Edges {
    /// The cells it stores.
    out: Vec<String>,
    /// The cells it reads.
    read: Vec<String>,
    /// The keyed planes it accumulates into.
    keyed: Vec<String>,
}

/// The edges of a stage.
fn edges(
    ctx: &Ctx,
    sinks: &[u32],
    cell_plane: &HashMap<String, String>,
    keyed_of: &HashMap<String, Keyed>,
) -> R<Edges> {
    let (mut out, mut read) = (vec![], vec![]);
    let mut keyed = vec![];
    let push = |list: &mut Vec<String>, name: String| {
        if !list.contains(&name) {
            list.push(name);
        }
    };
    for &sink in sinks {
        if is_a(ctx, sink, "SampleSum") {
            push(&mut out, ctx.str_field(sink, "cell")?);
        }
        if is_a(ctx, sink, "AccumWrite") {
            let name = ctx.str_field(sink, "name")?;
            if keyed_of.contains_key(&name) {
                push(&mut keyed, name);
            }
        }
    }
    for node in Graph::build(ctx, sinks)?.order {
        if ctx.is_param_cell(node)? {
            let plane = ctx.nodes[node as usize].ops[0];
            if let Some(cell) = cell_plane.get(&ctx.str_field(plane, "name")?) {
                push(&mut read, cell.clone());
            }
        }
    }
    Ok(Edges { out, read, keyed })
}

/// Notes the cell of a sample-sum sink among the cells, once.
fn note_cell(ctx: &Ctx, sink: u32, cells: &mut Vec<Cell>) -> R<()> {
    if !is_a(ctx, sink, "SampleSum") {
        return Ok(());
    }
    let name = ctx.str_field(sink, "cell")?;
    if cells.iter().any(|c| c.name == name) {
        return Ok(());
    }
    let columns = match ctx.field(sink, "columns")? {
        Val::Int(n) => i64::try_from(n).map_err(|_| Fault::Refused("a cell width".into()))?,
        _ => 1,
    };
    cells.push(Cell {
        name,
        columns,
        op: ctx.str_field(sink, "op")?,
        internal: false,
    });
    Ok(())
}

/// The cells the sample sums among the nodes under the sinks store, in the order the nodes are found.
pub fn stored_cells(ctx: &Ctx, sinks: &[u32]) -> R<Vec<Cell>> {
    let mut cells = vec![];
    for node in Graph::build(ctx, sinks)?.order {
        note_cell(ctx, node, &mut cells)?;
    }
    Ok(cells)
}

/// A scratch plane the runtime makes for a placed value.
#[derive(Debug, Clone, PartialEq)]
pub struct ScratchPlane {
    /// The plane's name.
    pub name: String,
    /// The element type: `real` for a floating one, else the type's own name.
    pub dtype: String,
    /// The elements of each lane's slice.
    pub per_lane: i64,
}

/// The scratch planes the placement made for the values under the sinks, by name (a plane a kernel declares itself is not one).
pub fn scratch_planes(ctx: &Ctx, sinks: &[u32]) -> R<Vec<ScratchPlane>> {
    let order = Graph::build(ctx, sinks)?.order;
    let mut out = vec![];
    for row in crate::wide_record(ctx, &order)? {
        let name = ctx.str_field(row.plane, "name")?;
        if !name.starts_with(crate::PLANE_PREFIX) {
            continue;
        }
        let dtype = match ctx.nodes[row.plane as usize].dtype.as_str() {
            "f64" | "f32" => "real".to_string(),
            "i32" => "int32".to_string(),
            other => other.to_string(),
        };
        out.push(ScratchPlane {
            name,
            dtype,
            per_lane: row.width,
        });
    }
    Ok(out)
}

/// The names the nodes under the sinks carry, for a caller that must not reuse them.
pub fn names_under(ctx: &Ctx, sinks: &[u32]) -> R<Vec<String>> {
    let order = Graph::build(ctx, sinks)?.order;
    Ok(order
        .into_iter()
        .filter_map(|n| ctx.str_field(n, "name").ok())
        .collect())
}

/// The stage as it is launched: when it holds nothing but copies of input planes into team scratch and they are large enough for a
/// block per lane, its stores become copies and its threads copy a lane's slice together.
fn copy_stage(
    ctx: &mut Ctx,
    stage: Vec<u32>,
    limits: &TeamLimits,
) -> R<(Vec<u32>, Option<TeamSpec>)> {
    let value = |ctx: &Ctx, sink: u32| ctx.nodes[sink as usize].ops[0];
    let copies = !stage.is_empty()
        && stage
            .iter()
            .all(|&sink| is_exactly(ctx, sink, "TeamStore") && is_a(ctx, value(ctx, sink), "Leaf"));
    let elements: i64 = stage
        .iter()
        .map(|&s| {
            ctx.nodes[value(ctx, s) as usize]
                .shape
                .iter()
                .product::<i64>()
        })
        .sum();
    if !(copies && elements >= 4 * limits.copy_threads) {
        return Ok((stage, None));
    }
    let mut out = vec![];
    for sink in stage {
        let (name, index) = (
            ctx.str_field(sink, "name")?,
            ctx.nodes[sink as usize].ops[1],
        );
        let operands = Val::Tuple(vec![Val::Node(value(ctx, sink)), Val::Node(index)]);
        out.push(build(
            ctx,
            "TeamCopy",
            vec![("name", Val::Str(name)), ("operands", operands)],
        )?);
    }
    Ok((
        out,
        Some(TeamSpec {
            threads: limits.copy_threads,
            steps: vec![],
            planes: vec![],
            grid: Grid::Lanes,
            out: None,
            reads: vec![],
        }),
    ))
}

/// Whether the node is of the class itself, not of a refinement of it.
fn is_exactly(ctx: &Ctx, node: u32, class: &str) -> bool {
    ctx.table.classes[ctx.nodes[node as usize].class].name == class
}
