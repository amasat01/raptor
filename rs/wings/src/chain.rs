//! Chains of kernels: a sequence of kernels bound by name into one set of sinks, the rule that decides where a chain is cut, and
//! the sinks of each piece of a cut chain.
//!
//! Member `j`'s read of a name binds to the latest value an earlier member committed to that name, else the name is an input of
//! the chain. A name no later member reads is an output; a name several members update in turn (each reading it first) binds in
//! order and its last value is the output. The binding is a substitution in the IR: the composed body holds each intermediate as
//! one value and stores none of them. A cut chain passes a value from one piece to the next in a plane the runtime makes per lane.
use crate::planes::type_parts;
use crate::{is_a, walk_order, R};
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::{ttype_val, Val};
use std::collections::{HashMap, HashSet};

/// The roles a member may read an intermediate as: its own lane's element.
const BOUND_LEAF_ROLES: [&str; 2] = ["plane_in", "mutable"];
/// The roles an intermediate may be committed as.
const BOUND_SINK_ROLES: [&str; 2] = ["mutable", "out"];
/// The prefix of the plane a cut chain passes a value through from one piece to the next.
pub const CROSS_PREFIX: &str = "hwc_";

/// One kernel of a chain: its name (for refusals) and the sinks it writes.
#[derive(Clone, Copy)]
pub struct Member<'a> {
    /// The kernel's name.
    pub name: &'a str,
    /// The kernel's sinks.
    pub sinks: &'a [u32],
}

/// The constants of the cost rule for one target: the registers of a fixed frame, the register file, and the costs of a launch
/// and of a byte through a plane (a row whose costs are not measured fuses up to the register file).
#[derive(Debug, Clone, PartialEq)]
pub struct FusionRow {
    /// The registers a kernel uses before any value.
    pub regs_base: i64,
    /// The register file.
    pub regs_file: i64,
    /// The cost of one more launch, in microseconds.
    pub launch_us: Option<f64>,
    /// The cost of one byte through a plane, in microseconds.
    pub plane_us_per_byte: Option<f64>,
    /// The cost of a byte of local frame over a byte of plane.
    pub local_over_plane: Option<f64>,
}

/// The sinks of a composed chain and the elements per sample that cross each boundary between members.
pub struct Composed {
    /// The output sinks, in the order the chain's names were first written.
    pub sinks: Vec<u32>,
    /// Entry `b - 1`: the elements per sample of the values written before member `b` and read from it on.
    pub crossing: Vec<i64>,
}

/// One plane that carries a value across a cut.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossPlane {
    /// The plane's name.
    pub name: String,
    /// `int32` for an index value, `real` for the build's real type.
    pub dtype: &'static str,
    /// The elements one lane holds.
    pub per_lane: i64,
}

/// One piece of a chain: the sinks it writes and the planes it hands on.
pub struct Piece {
    /// The sinks of the piece.
    pub sinks: Vec<u32>,
    /// The planes the piece writes for a later piece.
    pub planes: Vec<CrossPlane>,
}

/// The elements of a node's value.
fn elements(ctx: &Ctx, node: u32) -> i64 {
    ctx.nodes[node as usize].shape.iter().product()
}

/// Whether two nodes hold values of the same type.
fn same_type(ctx: &Ctx, one: u32, other: u32) -> bool {
    let (a, b) = (&ctx.nodes[one as usize], &ctx.nodes[other as usize]);
    (&a.dtype, &a.shape, &a.tag) == (&b.dtype, &b.shape, &b.tag)
}

/// The leaves under `sinks` (in the order a depth-first walk from the last sink finds them) and the leaves an indexed read takes
/// at a computed index.
fn leaves_under(ctx: &Ctx, sinks: &[u32]) -> (Vec<u32>, HashSet<u32>) {
    let (mut leaves, mut gathered, mut seen) = (vec![], HashSet::new(), HashSet::new());
    let mut stack = sinks.to_vec();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if is_a(ctx, node, "Leaf") {
            leaves.push(node);
        } else if is_a(ctx, node, "At") {
            gathered.insert(ctx.nodes[node as usize].ops[0]);
        }
        stack.extend(ctx.nodes[node as usize].ops.iter().copied());
    }
    (leaves, gathered)
}

/// The names of the leaves under the sinks of every member from `from` on.
fn read_after(ctx: &Ctx, members: &[Member], from: usize) -> R<HashSet<String>> {
    let mut names = HashSet::new();
    for member in &members[from..] {
        for leaf in leaves_under(ctx, member.sinks).0 {
            names.insert(ctx.str_field(leaf, "name")?);
        }
    }
    Ok(names)
}

/// What the binding has seen so far.
#[derive(Default)]
struct Binding {
    /// Per name, the member that committed it last and its sink as rebuilt.
    latest: HashMap<String, (usize, u32)>,
    /// Per name read before anything committed it: the first reader, the role and a leaf of the type.
    inputs: HashMap<String, (usize, String, u32)>,
    /// The names in the order they were first committed, each with the member that committed it last.
    written: Vec<(String, usize)>,
}

impl Binding {
    /// Notes a read of a name no member has committed: every read of it must agree on role and type.
    fn input(&mut self, ctx: &Ctx, members: &[Member], reader: usize, leaf: u32) -> R<()> {
        let name = ctx.str_field(leaf, "name")?;
        let role = ctx.str_field(leaf, "role")?;
        let held = self
            .inputs
            .entry(name.clone())
            .or_insert((reader, role.clone(), leaf));
        if held.1 != role || !same_type(ctx, held.2, leaf) {
            return refuse(format!(
                "chain: {} and {} read {name} as different inputs",
                members[held.0].name, members[reader].name
            ));
        }
        Ok(())
    }

    /// The value to read in place of a leaf that reads a name an earlier member committed, or a refusal when the read is not the
    /// lane's own element of it.
    fn bound_value(
        &self,
        ctx: &Ctx,
        (members, reader): (&[Member], usize),
        leaf: u32,
        gathered: &HashSet<u32>,
    ) -> R<Option<u32>> {
        let name = ctx.str_field(leaf, "name")?;
        let Some(&(writer, sink)) = self.latest.get(&name) else {
            return Ok(None);
        };
        let (reads, writes) = (&members[reader].name, &members[writer].name);
        if !same_type(ctx, leaf, sink) {
            return refuse(format!(
                "chain: {writes} commits {name} with a type {reads} does not read it as"
            ));
        }
        let role = ctx.str_field(leaf, "role")?;
        let sink_role = ctx.str_field(sink, "role")?;
        let off_lane = ctx.nodes[sink as usize].ops.len() > 1;
        if !BOUND_LEAF_ROLES.contains(&role.as_str())
            || gathered.contains(&leaf)
            || off_lane
            || !BOUND_SINK_ROLES.contains(&sink_role.as_str())
        {
            return refuse(format!(
                "chain: {reads} reads {name}, which {writes} commits, at another sample (or off its own lane)"
            ));
        }
        Ok(Some(ctx.nodes[sink as usize].ops[0]))
    }

    /// Refuses a sink that commits a name another member committed without reading it first, or an input that is not mutable.
    fn check_commits(
        &self,
        ctx: &Ctx,
        (members, member): (&[Member], usize),
        read: &HashSet<String>,
    ) -> R<()> {
        for &sink in members[member].sinks {
            let name = ctx.str_field(sink, "name")?;
            if let Some(&(writer, _)) = self.latest.get(&name) {
                if !read.contains(&name) {
                    return refuse(format!(
                        "chain: {} and {} both commit {name} and the second does not read it first",
                        members[writer].name, members[member].name
                    ));
                }
            }
            if let Some((reader, role, _)) = self.inputs.get(&name) {
                if role != "mutable" {
                    return refuse(format!(
                        "chain: {} reads {name} as an input but {} commits it",
                        members[*reader].name, members[member].name
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Composes `members` in order. `keep` names intermediates to commit as outputs too.
pub fn bind(ctx: &mut Ctx, members: &[Member], keep: &HashSet<String>) -> R<Composed> {
    let mut binding = Binding::default();
    let mut reads: Vec<HashSet<String>> = vec![];
    for (j, member) in members.iter().enumerate() {
        let (leaves, gathered) = leaves_under(ctx, member.sinks);
        let mut names = HashSet::new();
        let mut mapping = HashMap::new();
        for &leaf in &leaves {
            names.insert(ctx.str_field(leaf, "name")?);
            match binding.bound_value(ctx, (members, j), leaf, &gathered)? {
                Some(value) => {
                    mapping.insert(leaf, value);
                }
                None => binding.input(ctx, members, j, leaf)?,
            }
        }
        binding.check_commits(ctx, (members, j), &names)?;
        reads.push(names);
        let rebuilt = match mapping.is_empty() {
            true => member.sinks.to_vec(),
            false => ctx.substitute(member.sinks, &mapping)?,
        };
        for (&old, &new) in member.sinks.iter().zip(&rebuilt) {
            let name = ctx.str_field(old, "name")?;
            binding.latest.insert(name.clone(), (j, new));
            match binding.written.iter_mut().find(|w| w.0 == name) {
                Some(slot) => slot.1 = j,
                None => binding.written.push((name, j)),
            }
        }
    }
    let outputs = binding.written.iter().filter(|(name, j)| {
        keep.contains(name) || !reads[j + 1..].iter().any(|r| r.contains(name))
    });
    let sinks = outputs.map(|(name, _)| binding.latest[name].1).collect();
    Ok(Composed {
        sinks,
        crossing: crossing(ctx, members, &reads),
    })
}

/// Per boundary, the elements per sample of the values written before it and read after it.
fn crossing(ctx: &Ctx, members: &[Member], reads: &[HashSet<String>]) -> Vec<i64> {
    let mut produced: HashMap<String, i64> = HashMap::new();
    let mut out = vec![];
    for boundary in 1..members.len() {
        for &sink in members[boundary - 1].sinks {
            if let Ok(name) = ctx.str_field(sink, "name") {
                produced.insert(name, elements(ctx, sink));
            }
        }
        let later: HashSet<&String> = reads[boundary..].iter().flatten().collect();
        out.push(
            produced
                .iter()
                .filter(|(n, _)| later.contains(n))
                .map(|(_, w)| w)
                .sum(),
        );
    }
    out
}

/// The sinks of the segment of `members` from `from` up to `to`, with every name a later member reads kept as an output.
fn segment_sinks(ctx: &mut Ctx, members: &[Member], (from, to): (usize, usize)) -> R<Vec<u32>> {
    let later = read_after(ctx, members, to)?;
    Ok(bind(ctx, &members[from..to], &later)?.sinks)
}

/// The most elements per sample live at one point of the walk's canonical order. A value is live from its node (a plane read from
/// the start) to its last reader; a sink ends nothing of its own and a literal holds no register.
fn max_live(ctx: &Ctx, walk: u32) -> R<i64> {
    let order = walk_order(ctx, walk)?;
    let mut last: HashMap<u32, usize> = HashMap::new();
    for (i, &node) in order.iter().enumerate() {
        for &operand in &ctx.nodes[node as usize].ops {
            last.insert(operand, i);
        }
    }
    let mut delta = vec![0i64; order.len() + 1];
    for (i, &node) in order.iter().enumerate() {
        let Some(&end) = last.get(&node) else {
            continue;
        };
        if is_a(ctx, node, "Sink") || is_a(ctx, node, "Const") {
            continue;
        }
        let width = elements(ctx, node);
        delta[if is_a(ctx, node, "Leaf") { 0 } else { i }] += width;
        delta[end + 1] -= width;
    }
    let (mut best, mut live) = (0, 0);
    for step in delta {
        live += step;
        best = best.max(live);
    }
    Ok(best)
}

/// The predicted registers of the segment: the row's base plus the live values, each as wide as a real takes.
fn estimate(ctx: &mut Ctx, sinks: &[u32], row: &FusionRow, single: bool) -> R<i64> {
    let walk = ctx.canonical(sinks, &[], None, &[])? as u32;
    let per = if row.regs_base == 0 || single { 1 } else { 2 };
    Ok(row.regs_base + per * max_live(ctx, walk)?)
}

/// The boundaries to cut, greedy from the left over maximal segments: boundary `b` (before member `b`) is cut when fusing it
/// costs more than the cut. `single` is whether the real type is single precision; `samples` the lanes the rule weighs a cut at.
pub fn cuts(
    ctx: &mut Ctx,
    members: &[Member],
    row: &FusionRow,
    (single, samples): (bool, i64),
) -> R<Vec<usize>> {
    let real: i64 = if single { 4 } else { 8 };
    let crossing = bind(ctx, members, &HashSet::new())?.crossing;
    let mut shapes: HashMap<String, usize> = HashMap::new();
    for member in members {
        for &sink in member.sinks {
            shapes.insert(
                ctx.str_field(sink, "name")?,
                ctx.nodes[sink as usize].shape.len(),
            );
        }
    }
    let (mut out, mut from) = (vec![], 0);
    for boundary in 1..members.len() {
        let reads = read_after(ctx, members, boundary)?;
        let passes_matrix = members[..boundary].iter().any(|m| {
            m.sinks.iter().any(|&s| {
                let name = ctx.str_field(s, "name").unwrap_or_default();
                reads.contains(&name) && shapes[&name] > 1
            })
        });
        if passes_matrix {
            continue;
        }
        let mut estimated = |a: usize, b: usize| -> R<i64> {
            let sinks = segment_sinks(ctx, members, (a, b))?;
            estimate(ctx, &sinks, row, single)
        };
        let alone = estimated(from, boundary)?.max(estimated(boundary, boundary + 1)?);
        let both = estimated(from, boundary + 1)?;
        let frame = 4 * (both - row.regs_file.max(alone)).max(0);
        let fused = match (row.local_over_plane, row.launch_us, row.plane_us_per_byte) {
            (Some(local), Some(launch), Some(plane)) => {
                let before = read_after(ctx, &members[from..boundary], 0)?;
                let reloaded = before.intersection(&reads).count() as i64 * real;
                let cost = launch
                    + samples as f64
                        * plane
                        * (2 * crossing[boundary - 1] * real + reloaded) as f64;
                samples as f64 * plane * local * 2.0 * frame as f64 <= cost
            }
            _ => frame == 0,
        };
        if !fused {
            out.push(boundary);
            from = boundary;
        }
    }
    Ok(out)
}

/// A node of the class made from its given fields.
fn make(ctx: &mut Ctx, class: &str, given: Vec<(&str, Val)>) -> R<u32> {
    let index = ctx.table.find(class);
    let index = index.ok_or_else(|| Fault::Refused(format!("no {class} class")))?;
    let given: Vec<(String, Val)> = given.into_iter().map(|(k, v)| (k.into(), v)).collect();
    ctx.new_from(index, &given)
}

/// The pieces of `members` cut before each member index in `cuts`. A value one piece computes and a later one reads crosses in a
/// plane the runtime makes per lane, shaped as a caller's plane of its width: the writer commits the value there and the reader
/// reads its own lane's.
pub fn pieces(ctx: &mut Ctx, members: &[Member], cuts: &[usize]) -> R<Vec<Piece>> {
    let bounds: Vec<usize> = [0]
        .iter()
        .chain(cuts)
        .chain(&[members.len()])
        .copied()
        .collect();
    let mut crossed: HashSet<String> = HashSet::new();
    let mut out = vec![];
    for pair in bounds.windows(2) {
        let later = read_after(ctx, members, pair[1])?;
        let bound = bind(ctx, &members[pair[0]..pair[1]], &later)?.sinks;
        let sinks = read_crossed(ctx, bound, &crossed)?;
        let (mut kept, mut planes) = (vec![], vec![]);
        for sink in sinks {
            let name = ctx.str_field(sink, "name")?;
            if !later.contains(&name) {
                kept.push(sink);
                continue;
            }
            let (dtype, shape) = {
                let node = &ctx.nodes[sink as usize];
                (node.dtype.clone(), node.shape.clone())
            };
            if shape.len() > 1 || !["f64", "f32", "i32"].contains(&dtype.as_str()) {
                return refuse(format!(
                    "chain: {name} would cross a cut as {dtype} of rank {}",
                    shape.len()
                ));
            }
            let plane = format!("{CROSS_PREFIX}{name}");
            let ty = type_parts(ctx, sink);
            let given = vec![
                ("name", Val::Str(plane.clone())),
                ("value", Val::Node(ctx.nodes[sink as usize].ops[0])),
                ("ttype", ttype_val(&ty.dtype, &ty.shape, ty.tag.as_deref())),
            ];
            kept.push(make(ctx, "Assign", given)?);
            let per_lane = shape.iter().product();
            let dtype = if dtype == "i32" { "int32" } else { "real" };
            planes.push(CrossPlane {
                name: plane,
                dtype,
                per_lane,
            });
            crossed.insert(name);
        }
        out.push(Piece {
            sinks: kept,
            planes,
        });
    }
    Ok(out)
}

/// The sinks with every read of a name an earlier piece handed on replaced by a read of its plane.
fn read_crossed(ctx: &mut Ctx, sinks: Vec<u32>, crossed: &HashSet<String>) -> R<Vec<u32>> {
    let mut mapping = HashMap::new();
    for leaf in leaves_under(ctx, &sinks).0 {
        let name = ctx.str_field(leaf, "name")?;
        if !crossed.contains(&name) {
            continue;
        }
        let ty = type_parts(ctx, leaf);
        let given = vec![
            ("kind", Val::Str("vocab_read".into())),
            ("role", Val::Str("plane_in".into())),
            ("name", Val::Str(format!("{CROSS_PREFIX}{name}"))),
            ("ttype", ttype_val(&ty.dtype, &ty.shape, ty.tag.as_deref())),
        ];
        mapping.insert(leaf, make(ctx, "Leaf", given)?);
    }
    if mapping.is_empty() {
        return Ok(sinks);
    }
    ctx.substitute(&sinks, &mapping)
}
