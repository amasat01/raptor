//! The planes of a placement: which plane every placed value and slot lives in, and the nodes
//! that bind a value to its plane.
use crate::place::{kind_of, width, Graph, ELEMENTWISE};
use crate::plan::{Plan, Slot};
use crate::{is_a, PLANE_PREFIX, RESIDENT_KIND};
use raptor_ir::record::{Ctx, Fault};
use std::collections::{HashMap, HashSet};

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// The plane of every placed value and slot, and the scratch planes made with their widths.
#[derive(Default)]
pub(crate) struct Planes {
    /// The plane of each placed value.
    pub of: HashMap<u32, String>,
    /// The plane of each placed slot.
    pub slots: HashMap<Slot, String>,
    /// The width of each scratch plane made.
    pub scratch: HashMap<String, i64>,
}

/// The names a new scratch plane must avoid, and the count of the ones tried.
struct Fresh {
    /// The names already in use.
    avoid: HashSet<String>,
    /// How many names were tried.
    count: usize,
}

impl Fresh {
    /// The next plane name not in use.
    fn next(&mut self) -> String {
        loop {
            let name = format!("{PLANE_PREFIX}{}", self.count);
            self.count += 1;
            if !self.avoid.contains(&name) {
                return name;
            }
        }
    }
}

/// Whether two nodes have the same type.
fn same_type(ctx: &Ctx, a: u32, b: u32) -> bool {
    let (x, y) = (&ctx.nodes[a as usize], &ctx.nodes[b as usize]);
    (&x.dtype, &x.shape, &x.tag) == (&y.dtype, &y.shape, &y.tag)
}

/// The type of a node, as the parts a new node is made from.
#[derive(Clone)]
pub(crate) struct NodeType {
    /// The element type name.
    pub dtype: String,
    /// The extent of each axis.
    pub shape: Vec<i64>,
    /// The tag of the node, when it has one.
    pub tag: Option<String>,
}

/// The dtype, shape and tag of a node.
pub(crate) fn type_parts(ctx: &Ctx, node: u32) -> NodeType {
    let record = &ctx.nodes[node as usize];
    NodeType {
        dtype: record.dtype.clone(),
        shape: record.shape.clone(),
        tag: record.tag.clone(),
    }
}

/// What the assignment reads while it goes.
struct Assigner<'a> {
    /// The context the kernel is read from.
    ctx: &'a Ctx,
    /// The kernel's nodes and their uses.
    graph: &'a Graph,
    /// The placement decision.
    plan: &'a Plan<'a>,
    /// The mutables the kernel reads.
    read: HashSet<String>,
    /// The planes that belong to a mutable of the kernel.
    owned: HashSet<String>,
    /// The source of new plane names.
    fresh: Fresh,
    /// The planes assigned so far.
    planes: Planes,
}

impl Assigner<'_> {
    /// A new scratch plane of the given width.
    fn scratch_plane(&mut self, size: i64) -> String {
        let name = self.fresh.next();
        self.planes.scratch.insert(name.clone(), size);
        name
    }

    /// The mutable plane `node` is stored to whole, when nothing reads that plane's prior value.
    fn own_plane(&self, node: u32) -> Outcome<Option<String>> {
        for usage in self.graph.users(node) {
            if !is_a(self.ctx, usage.user, "Assign") || !same_type(self.ctx, usage.user, node) {
                continue;
            }
            let name = self.ctx.str_field(usage.user, "name")?;
            if !self.read.contains(&name) && !self.owned.contains(&name) {
                return Ok(Some(name));
            }
        }
        Ok(None)
    }

    /// The planes of the slots: a slot's own mutable when its value ends there, else scratch; a
    /// stepping loop's mutable carry is the mutable's own plane.
    fn assign_slots(&mut self) -> Outcome<()> {
        let (ctx, plan) = (self.ctx, self.plan);
        for &lp in &self.graph.loops {
            for (j, &carry) in crate::slots::carries(ctx, lp)?.iter().enumerate() {
                if !plan.slots.contains(&(lp, j)) {
                    continue;
                }
                let value = self
                    .graph
                    .users(lp)
                    .iter()
                    .map(|u| u.user)
                    .find(|&u| is_a(ctx, u, "LoopValue") && plan.slot_of_value(u) == (lp, j));
                let own = match value {
                    Some(value) if !plan.accum.contains(&(lp, j)) => self.own_plane(value)?,
                    _ => None,
                };
                if let Some(name) = &own {
                    self.owned.insert(name.clone());
                }
                let name = own.unwrap_or_else(|| self.scratch_plane(width(ctx, carry)));
                self.planes.slots.insert((lp, j), name);
            }
        }
        for (&(lp, j), (_, plane)) in &plan.carried {
            self.planes.slots.insert((lp, j), plane.clone());
            if !plan.sunk.contains_key(&(lp, j)) {
                self.planes
                    .of
                    .insert(crate::slots::nexts(ctx, lp)?[j], plane.clone());
            }
            self.owned.insert(plane.clone());
        }
        for &lp in &self.graph.loops {
            for j in 0..crate::slots::carry_count(ctx, lp)? {
                if let (true, Some(plane)) = (
                    plan.accum.contains(&(lp, j)),
                    self.planes.slots.get(&(lp, j)),
                ) {
                    self.planes
                        .of
                        .insert(crate::slots::nexts(ctx, lp)?[j], plane.clone());
                }
            }
        }
        Ok(())
    }

    /// The scratch plane of an operand nothing else reads and of the node's own width, which the
    /// node is written into in place.
    fn reusable(&self, node: u32) -> Option<String> {
        let (ctx, graph) = (self.ctx, self.graph);
        ctx.nodes[node as usize].ops.iter().find_map(|&operand| {
            let held = self.planes.of.get(&operand).cloned().or_else(|| {
                is_a(ctx, operand, "LoopValue")
                    .then(|| {
                        self.planes
                            .slots
                            .get(&self.plan.slot_of_value(operand))
                            .cloned()
                    })
                    .flatten()
            })?;
            let alone = graph.users(operand).len() == 1
                && width(ctx, operand) == width(ctx, node)
                && graph.scope[&operand].is_none();
            (self.planes.scratch.contains_key(&held) && alone).then_some(held)
        })
    }

    /// The plane of every placed node, in walk order.
    fn assign_nodes(&mut self) -> Outcome<()> {
        let (ctx, plan) = (self.ctx, self.plan);
        let mut sunk_next = HashSet::new();
        for &(lp, j) in plan.sunk.keys() {
            sunk_next.insert(crate::slots::nexts(ctx, lp)?[j]);
        }
        for &node in &self.graph.order {
            let skip = !plan.resident.contains(&node) || self.planes.of.contains_key(&node);
            if skip || sunk_next.contains(&node) {
                continue;
            }
            if let Some(name) = self.own_plane(node)? {
                self.owned.insert(name.clone());
                self.planes.of.insert(node, name);
                continue;
            }
            let kind = kind_of(ctx, node)?.unwrap_or_default();
            let at_top = self.graph.scope[&node].is_none() && ELEMENTWISE.contains(&kind.as_str());
            let reuse = if at_top { self.reusable(node) } else { None };
            let name = reuse.unwrap_or_else(|| self.scratch_plane(width(ctx, node)));
            self.planes.of.insert(node, name);
        }
        Ok(())
    }
}

/// The plane each placed value and slot lives in: a mutable's own when the value ends there whole,
/// the plane of an operand nothing else reads, else a new scratch plane.
pub(crate) fn assign_planes(
    ctx: &Ctx,
    graph: &Graph,
    plan: &Plan,
    taken: &[String],
) -> Outcome<Planes> {
    let (mut read, mut avoid) = (
        HashSet::new(),
        taken.iter().cloned().collect::<HashSet<_>>(),
    );
    for &node in &graph.order {
        if is_a(ctx, node, "Leaf") || is_a(ctx, node, "Sink") {
            avoid.insert(ctx.str_field(node, "name")?);
        }
        let mutable = is_a(ctx, node, "Leaf") && ctx.str_field(node, "role")? == "mutable";
        if mutable && ctx.str_field(node, "kind")? != "vocab_read" {
            read.insert(ctx.str_field(node, "name")?);
        }
    }
    let fresh = Fresh { avoid, count: 0 };
    let mut assigner = Assigner {
        ctx,
        graph,
        plan,
        read,
        owned: HashSet::new(),
        fresh,
        planes: Planes::default(),
    };
    assigner.assign_slots()?;
    assigner.assign_nodes()?;
    Ok(assigner.planes)
}

/// The leaf that binds the plane `name` in the walk: a mutable of the type of `original`, or
/// (`scratch`) a flat plane of its scalar.
pub(crate) fn plane_leaf(ctx: &mut Ctx, original: u32, name: &str, scratch: bool) -> Outcome<u32> {
    let NodeType { dtype, shape, tag } = type_parts(ctx, original);
    let role = if scratch { "wide_in" } else { "mutable" };
    let (shape, tag) = if scratch {
        (vec![], None)
    } else {
        (shape, tag)
    };
    let fields = format!(
        "((\"kind\", \"vocab_read\"), (\"name\", {name:?}), (\"quantity\", None), \
         (\"role\", {role:?}), (\"sorted\", False), (\"wire_index\", 0), (\"axes\", ()))"
    );
    ctx.make("Leaf", &dtype, &shape, tag.as_deref(), &[], &fields)
}

/// The `resident` op placing `value` in the plane `name`; `original` has the type the plane's
/// leaf is made of.
pub(crate) fn place_node(
    ctx: &mut Ctx,
    value: u32,
    original: u32,
    name: &str,
    scratch: bool,
) -> Outcome<u32> {
    let leaf = plane_leaf(ctx, original, name, scratch)?;
    let NodeType { dtype, shape, tag } = type_parts(ctx, value);
    let fields = format!("((\"kind\", {RESIDENT_KIND:?}), (\"literal\", {name:?}))");
    ctx.make(
        "Op",
        &dtype,
        &shape,
        tag.as_deref(),
        &[value, leaf],
        &fields,
    )
}
