//! Groups: the typing of a `Group` scope, the numbering of its phases and the rules that keep a scratch plane honest.
//!
//! A group is a set of lanes. Its body is an ordered list of sinks; a `Barrier` sink and a group-wide reduce or scan end a
//! phase. The phase of a sink is fixed by the body alone (`phase_plan`), so a consumer asks for it (the computed fields
//! `phases` and `phase_count` of a Group) and never walks barriers and reduces itself.
use crate::dump::refs;
use crate::record::Ctx;
use crate::value::{text_val, Val};
use std::collections::{HashMap, HashSet};

/// The class names that exist only inside the body of a group.
pub(crate) const GROUP_ONLY: [&str; 5] = [
    "GroupRank",
    "GroupSize",
    "GroupReduce",
    "GroupScan",
    "Barrier",
];

/// The sinks whose meaning belongs to the whole launch, with why a group body does not hold them. Every other sink of a kernel
/// (an `Assign`, a plain sink on the roles `mutable`, `out` or `plane_out`, an `Update`) is committed from a body like any other.
const LAUNCH_SINKS: [(&str, &str); 3] = [
    (
        "SampleSum",
        "it folds the samples of the whole launch, and a group sees only its own lanes",
    ),
    (
        "Finish",
        "it ends a sample through the one mask of the launch; state it outside the group",
    ),
    (
        "TapeWrite",
        "it commits the rows of the tape of the loop it sits in, not of a group",
    ),
];

/// The facts of a `Group` as its constructor states them, read by `Ctx::group_fault`.
pub struct GroupFacts<'a> {
    /// `count`, `axes` or `key`.
    pub membership: &'a str,
    /// The number of lanes of a `count` group.
    pub count: &'a Val,
    /// The lane axes of the kernel, as (label, extent) pairs, stated by an `axes` group.
    pub grid: &'a Val,
    /// The labels of the lane axes that span one group.
    pub axes: &'a Val,
    /// The largest group a `key` group can hold.
    pub max_size: &'a Val,
    /// The key operand of a `key` group.
    pub key: Option<u32>,
    /// The scratch planes the group declares.
    pub scratch: &'a [u32],
    /// The body sinks, in order.
    pub body: &'a [u32],
}

/// The phase of every body sink and the phase each group-wide reduce or scan is evaluated in.
pub struct PhasePlan {
    /// The phase of each body sink, in body order; a barrier belongs to the phase it ends.
    pub sinks: Vec<u32>,
    /// The phase in which the operand of each reduce or scan the body reads is evaluated: the phase the reduce or scan ends, and
    /// so one before the first phase that reads its value. The rule: the evaluation is placed where the first sink that reads it
    /// is reached in body order, at the latest of (a) the floor the last barrier before that sink sets, (b) the phase of the
    /// latest store into each scratch the operand reads, and (c) one past the evaluation of every reduce or scan the operand
    /// holds. A reduce or scan evaluated once is not placed again for a later reader. A node the body does not read has no entry.
    pub evaluated: HashMap<u32, u32>,
}

impl PhasePlan {
    /// The reduces and scans the body reads, in the order of their handles: the order of the lists the door states.
    pub fn combines(&self) -> Vec<u32> {
        let mut nodes: Vec<u32> = self.evaluated.keys().copied().collect();
        nodes.sort_unstable();
        nodes
    }

    /// The number of phases: one past the last phase a sink belongs to.
    pub fn count(&self) -> u32 {
        self.sinks.iter().copied().max().map_or(0, |m| m + 1)
    }
}

impl Ctx {
    /// The class name of a node.
    pub(crate) fn class_name(&self, h: u32) -> &str {
        &self.table.classes[self.nodes[h as usize].class].name
    }

    /// The operands of a node and every node its fields hold (the body of a loop, a reduced value).
    pub(crate) fn reached_from(&self, h: u32) -> Vec<u32> {
        let node = &self.nodes[h as usize];
        let mut all = node.ops.clone();
        node.fields.iter().for_each(|f| refs(f, &mut all));
        all
    }

    /// The text of a text field, if the node has one.
    pub(crate) fn text_of(&self, h: u32, field: &str) -> Option<String> {
        match self.field(h, field) {
            Ok(Val::Str(s)) => Some(s),
            _ => None,
        }
    }

    /// Whether `h` is a scratch plane: a leaf in the role `scratch`.
    pub(crate) fn is_scratch_leaf(&self, h: u32) -> bool {
        self.is_a(h, "Leaf") && self.text_of(h, "role").as_deref() == Some("scratch")
    }

    /// Whether `h` is a sink that writes a scratch plane.
    pub(crate) fn is_scratch_sink(&self, h: u32) -> bool {
        self.is_a(h, "Sink") && self.text_of(h, "role").as_deref() == Some("scratch")
    }

    /// The name of a plane node or sink; empty when it states none.
    pub(crate) fn plane_name(&self, h: u32) -> String {
        self.text_of(h, "name").unwrap_or_default()
    }

    /// Why a group-wide reduce or scan cannot take `value`: one rank-0 value, a bool for `any` and `all`, a number otherwise.
    pub fn reduce_fault(&self, op: &str, value: u32) -> Option<String> {
        let n = &self.nodes[value as usize];
        if !n.shape.is_empty() {
            return Some(format!(
                "a value of shape {:?} is not combined in this step; combine its components one at a time",
                n.shape
            ));
        }
        let wants_bool = matches!(op, "any" | "all");
        if wants_bool != (n.dtype == "bool") {
            return Some(if wants_bool {
                format!("'{op}' takes a bool, got {}", n.dtype)
            } else {
                "a bool is combined by 'any' or 'all', not by a sum, min or max".to_string()
            });
        }
        None
    }

    /// Why a loop body may not hold what it holds: a group, a barrier, or a group-wide reduce or scan. A loop sits inside ONE phase
    /// of a group and every lane must run it the same way, so it can hold none of them.
    pub fn loop_group_fault(&self, body_sinks: &[u32], values: &[u32]) -> Option<String> {
        let mut stack: Vec<u32> = body_sinks.iter().chain(values).copied().collect();
        let mut seen = HashSet::new();
        while let Some(h) = stack.pop() {
            if !seen.insert(h) {
                continue;
            }
            let class = self.class_name(h);
            if class == "Group" || GROUP_ONLY[2..].contains(&class) {
                return Some(format!(
                    "a {class} inside a loop body is not written; a loop sits inside one phase of a group and holds no group, barrier, group reduce or group scan"
                ));
            }
            stack.extend(self.reached_from(h));
        }
        None
    }

    /// The group-only element a root reaches outside any group, named, if there is one: what a walk of a plain sink refuses.
    pub fn outside_group_fault(&self, root: u32) -> Option<String> {
        let (mut stack, mut seen) = (vec![root], HashSet::new());
        while let Some(h) = stack.pop() {
            if !seen.insert(h) {
                continue;
            }
            if let Some(name) = self.called_group_function(h) {
                return Some(format!(
                    "the call of the group function '{name}' is reached outside a Group; a group function is called only in the body of a Group"
                ));
            }
            let class = self.class_name(h);
            let thing = if GROUP_ONLY.contains(&class) {
                Some(class.to_string())
            } else if self.is_scratch_leaf(h) || self.is_scratch_sink(h) {
                Some(format!("scratch '{}'", self.plane_name(h)))
            } else {
                None
            };
            if let Some(thing) = thing {
                return Some(format!(
                    "{thing} is reached outside a Group; GroupRank, GroupSize, GroupReduce, GroupScan, Barrier and scratch exist only inside the body of a Group"
                ));
            }
            stack.extend(self.reached_from(h));
        }
        None
    }

    /// The phase of every sink of a group body, by the order of the body. A barrier ends the phase it closes and sets the floor of
    /// the phases after it. A group reduce or scan is evaluated at the highest phase of what its operand reads (the latest store
    /// into a scratch it reads, the availability of another reduce) but never below the floor, and its value is available one phase
    /// later; so reductions that do not depend on each other share a phase. A sink is in the latest of the previous sink's phase and
    /// the availability of every value it uses: phases never go backwards, so a store sits after every reduce evaluated before it.
    pub fn phase_plan(&self, body: &[u32]) -> PhasePlan {
        let mut evaluated: HashMap<u32, u32> = HashMap::new();
        let mut stored: HashMap<String, u32> = HashMap::new();
        let (mut cursor, mut floor, mut sinks) = (0u32, 0u32, vec![]);
        for &sink in body {
            if self.class_name(sink) == "Barrier" {
                sinks.push(cursor);
                cursor += 1;
                floor = cursor;
                continue;
            }
            cursor = cursor.max(self.availability(sink, floor, &stored, &mut evaluated));
            sinks.push(cursor);
            if self.is_scratch_sink(sink) {
                let latest = stored.entry(self.plane_name(sink)).or_insert(cursor);
                *latest = (*latest).max(cursor);
            }
        }
        PhasePlan { sinks, evaluated }
    }

    /// The first phase in which everything `root` reads is available. A node is as late as its latest operand, a read of a scratch
    /// waits for the stores made before, and a reduce or scan not yet evaluated is evaluated now, at `floor` or later.
    fn availability(
        &self,
        root: u32,
        floor: u32,
        stored: &HashMap<String, u32>,
        evaluated: &mut HashMap<u32, u32>,
    ) -> u32 {
        let mut level: HashMap<u32, u32> = HashMap::new();
        let mut stack: Vec<(u32, bool)> = vec![(root, false)];
        while let Some((h, expanded)) = stack.pop() {
            if level.contains_key(&h) {
                continue;
            }
            let combines = matches!(self.class_name(h), "GroupReduce" | "GroupScan");
            if let (true, Some(&done)) = (combines, evaluated.get(&h)) {
                level.insert(h, done + 1);
                continue;
            }
            let children = self.reached_from(h);
            if !expanded {
                stack.push((h, true));
                stack.extend(children.iter().map(|&c| (c, false)));
                continue;
            }
            let mut at = children
                .iter()
                .filter_map(|c| level.get(c))
                .max()
                .copied()
                .unwrap_or(0);
            if let Some(&plane) = self.nodes[h as usize].ops.first() {
                if self.class_name(h) == "At" && self.is_scratch_leaf(plane) {
                    at = at.max(stored.get(&self.plane_name(plane)).copied().unwrap_or(0));
                }
            }
            if combines {
                evaluated.insert(h, at.max(floor));
                at = at.max(floor) + 1;
            }
            level.insert(h, at);
        }
        level[&root]
    }

    /// The facts of a group that are computed from its body and scratch: `phases` (the phase of each body sink), `phase_count`,
    /// `scratch_decls` (name, element type, value shape and axes of each scratch, the part of the identity the walk's leaf
    /// key does not carry), and the two positional lists `combines` and `combine_phases`: every group-wide reduce or scan the body
    /// reads, in the order of its handle, and the phase each one is evaluated in (see `PhasePlan::evaluated`). The lists are
    /// the same length and are read together.
    pub fn group_fact(&self, group: u32, name: &str) -> Result<Val, crate::record::Fault> {
        if name == "scratch_decls" {
            let declare = |&s: &u32| -> Result<Val, crate::record::Fault> {
                let n = self.node(s)?;
                let shape = Val::Tuple(n.shape.iter().map(|&d| Val::Int(i128::from(d))).collect());
                Ok(Val::Tuple(vec![
                    Val::Str(self.plane_name(s)),
                    Val::Str(n.dtype.clone()),
                    shape,
                    self.field(s, "axes")?,
                ]))
            };
            let scratch = self.refs(group, "scratch")?;
            return Ok(Val::Tuple(
                scratch.iter().map(declare).collect::<Result<_, _>>()?,
            ));
        }
        let plan = self.phase_plan(&self.refs(group, "body_sinks")?);
        let ints = |values: Vec<u32>| {
            Val::Tuple(
                values
                    .into_iter()
                    .map(|v| Val::Int(i128::from(v)))
                    .collect(),
            )
        };
        Ok(match name {
            "phases" => ints(plan.sinks.clone()),
            "phase_count" => Val::Int(i128::from(plan.count())),
            "combines" => Val::Tuple(plan.combines().into_iter().map(Val::Node).collect()),
            _ => ints(plan.combines().iter().map(|h| plan.evaluated[h]).collect()),
        })
    }

    /// Why a group cannot be built from these facts, if it cannot: its membership, its scratch, its body and the rules of its phases.
    pub fn group_fault(&self, g: &GroupFacts) -> Option<String> {
        let bound = match self.membership_fault(g) {
            Err(fault) => return Some(fault),
            Ok(bound) => bound,
        };
        if let Some(fault) = self.scratch_fault(g, bound) {
            return Some(fault);
        }
        if g.body.is_empty() {
            return Some("a group commits nothing; its body holds no sink".into());
        }
        if let Some(&bad) = g
            .body
            .iter()
            .find(|&&s| !(self.is_a(s, "Sink") || self.class_name(s) == "Barrier"))
        {
            let class = self.class_name(bad);
            return Some(if class == "Group" {
                "a group inside a group is not written; one group scope holds the whole body"
                    .to_string()
            } else {
                format!("the body holds a {class}, which is neither a sink nor a barrier")
            });
        }
        if let Some(fault) = g.body.iter().find_map(|&s| self.launch_sink_fault(s)) {
            return Some(fault);
        }
        self.phase_fault(g)
    }

    /// Whether `sink` is a keyed `Update` in the lane order (`order` = `lane`) or an ordered accumulate: the contract the wings
    /// lower through a derived stage over the launch (the ordered scatter, the record of winners).
    fn is_ordered_keyed_update(&self, sink: u32) -> bool {
        let keyed = self.is_a(sink, "Update") && self.nodes[sink as usize].ops.len() > 1;
        let lane_order = self.text_of(sink, "order").as_deref() == Some("lane");
        (keyed && lane_order) || self.is_a(sink, "OrderedAccumWrite")
    }

    /// Why a body sink is refused when its meaning belongs to the whole launch, if it is one of those.
    fn launch_sink_fault(&self, sink: u32) -> Option<String> {
        if self.is_ordered_keyed_update(sink) {
            return Some(format!(
                "an ordered keyed update inside a group body is not written; the lane order of the launch needs a derived stage of its own, which becomes a group by key. Write the update outside the group, or use the plain or compensated contract (sink '{}')",
                self.plane_name(sink)
            ));
        }
        let (class, reason) = LAUNCH_SINKS
            .iter()
            .find(|(class, _)| self.is_a(sink, class))?;
        Some(format!(
            "a {class} inside a group body is not written; {reason} (sink '{}')",
            self.plane_name(sink)
        ))
    }

    /// The membership rules; the size bound of one group (None: not known at build) when the facts agree.
    fn membership_fault(&self, g: &GroupFacts) -> Result<Option<i128>, String> {
        let has = |v: &Val| !matches!(v, Val::None) && *v != Val::Tuple(vec![]);
        let alien = |what: &str, belongs: &str| {
            Err(format!(
                "{what} belongs to membership '{belongs}', not to '{}'",
                g.membership
            ))
        };
        if g.key.is_some() && g.membership != "key" {
            return alien("a key operand", "key");
        }
        if has(g.max_size) && g.membership != "key" {
            return alien("max_size", "key");
        }
        if has(g.count) && g.membership != "count" {
            return alien("count", "count");
        }
        if (has(g.axes) || has(g.grid)) && g.membership != "axes" {
            return alien("axes and grid", "axes");
        }
        match g.membership {
            "count" => match g.count {
                Val::Int(n) if *n >= 1 => Ok(Some(*n)),
                other => Err(format!(
                    "membership 'count' takes a positive whole number of lanes, got {}",
                    text_val(other)
                )),
            },
            "key" => {
                let Some(key) = g.key else {
                    return Err("membership 'key' takes the per-lane key as its operand".into());
                };
                let node = &self.nodes[key as usize];
                if !node.shape.is_empty() || !matches!(node.dtype.as_str(), "i32" | "i64") {
                    return Err(format!(
                        "membership 'key' takes one integer scalar per lane as the key, got {} of shape {:?}",
                        node.dtype, node.shape
                    ));
                }
                match g.max_size {
                    Val::None => Ok(None),
                    Val::Int(n) if *n >= 1 => Ok(Some(*n)),
                    other => Err(format!(
                        "max_size is a positive whole number, got {}",
                        text_val(other)
                    )),
                }
            }
            _ => self.spanned_size(g),
        }
    }

    /// The size of one group of an `axes` membership: the product of the spanned extents, None when one is known only at run time.
    fn spanned_size(&self, g: &GroupFacts) -> Result<Option<i128>, String> {
        if let Some(fault) = crate::ctor::axes_fault(g.grid) {
            return Err(format!("grid: {fault}"));
        }
        let (Val::Tuple(grid), Val::Tuple(named)) = (g.grid, g.axes) else {
            return Err(
                "membership 'axes' takes the lane axes and the labels that span a group".into(),
            );
        };
        if named.is_empty() {
            return Err(
                "membership 'axes' names no lane axis; name the axes that span one group".into(),
            );
        }
        let extent_of = |label: &str| {
            grid.iter().find_map(|axis| match axis {
                Val::Tuple(pair) => match pair.as_slice() {
                    [Val::Str(l), extent] if l == label => Some(extent.clone()),
                    _ => None,
                },
                _ => None,
            })
        };
        let labels: Vec<String> = grid
            .iter()
            .filter_map(|a| match a {
                Val::Tuple(p) => p.first().map(text_val),
                _ => None,
            })
            .collect();
        let (mut size, mut seen) = (Some(1i128), HashSet::new());
        for label in named {
            let Val::Str(label) = label else {
                return Err("an axis label is a text".into());
            };
            if !seen.insert(label.as_str()) {
                return Err(format!("the axis '{label}' is named twice"));
            }
            match extent_of(label) {
                None => {
                    return Err(format!(
                        "membership 'axes' names the axis '{label}', which the lanes do not have (lane axes: {})",
                        labels.join(", ")
                    ))
                }
                Some(Val::Int(n)) => size = size.map(|s| s * n),
                Some(_) => size = None,
            }
        }
        Ok(size)
    }

    /// The scratch rules: each is a scratch leaf with the one axis `rank` whose extent is the size bound.
    fn scratch_fault(&self, g: &GroupFacts, bound: Option<i128>) -> Option<String> {
        if g.scratch.is_empty() {
            return None;
        }
        if g.membership == "key" && bound.is_none() {
            return Some(
                "membership 'key' with scratch needs max_size, the largest group the scratch holds"
                    .into(),
            );
        }
        let mut names = HashSet::new();
        for &s in g.scratch {
            if !self.is_scratch_leaf(s) {
                return Some(format!(
                    "a scratch is a leaf in the role 'scratch', got a {}",
                    self.class_name(s)
                ));
            }
            let name = self.plane_name(s);
            if !names.insert(name.clone()) {
                return Some(format!("scratch '{name}' is declared twice"));
            }
            let Some(bound) = bound else {
                return Some(format!(
                    "scratch '{name}': the lanes of a group span an axis whose extent is known only at run time; a scratch needs a size known at build, so state the extents of the spanned axes"
                ));
            };
            let extent = match self.field(s, "axes") {
                Ok(Val::Tuple(axes)) => match axes.as_slice() {
                    [Val::Tuple(pair)] => match pair.as_slice() {
                        [Val::Str(label), Val::Int(n)] if label == "rank" => Some(*n),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            };
            match extent {
                None => {
                    return Some(format!(
                        "scratch '{name}' has one axis, 'rank', of a stated extent; it is declared with the axes {}",
                        self.field(s, "axes").map_or_else(|_| "?".into(), |a| text_val(&a))
                    ))
                }
                Some(n) if n != bound => {
                    return Some(format!(
                        "scratch '{name}' has {n} elements on its rank axis but a group holds at most {bound} lanes; the extent is the size bound of the group"
                    ))
                }
                Some(_) => {}
            }
        }
        None
    }

    /// The faults of the body's phases: the stores into scratch, the scratch reads, and what a read may see in its phase.
    fn phase_fault(&self, g: &GroupFacts) -> Option<String> {
        let plan = self.phase_plan(g.body);
        let mut written: HashSet<(u32, String)> = HashSet::new();
        for (&sink, &phase) in g.body.iter().zip(&plan.sinks) {
            if let Some(fault) = self.store_fault(sink, g.scratch) {
                return Some(fault);
            }
            if self.is_scratch_sink(sink) {
                written.insert((phase, self.plane_name(sink)));
            }
        }
        // a reduce or scan evaluates its operand once, in the phase the plan gives it; a later use does not read again
        let mut evaluated_once = HashSet::new();
        for (&sink, &phase) in g.body.iter().zip(&plan.sinks) {
            let mut stack = vec![(sink, phase)];
            let mut seen = HashSet::new();
            while let Some((h, at)) = stack.pop() {
                if !seen.insert((h, at)) {
                    continue;
                }
                if matches!(self.class_name(h), "GroupReduce" | "GroupScan") {
                    if evaluated_once.insert(h) {
                        let phase = plan.evaluated.get(&h).copied().unwrap_or(at);
                        stack.extend(self.nodes[h as usize].ops.iter().map(|&o| (o, phase)));
                    }
                    continue;
                }
                if h != sink && self.is_scratch_sink(h) {
                    return Some(format!(
                        "scratch '{}' is stored inside a loop; a store into scratch sits in the body of the group",
                        self.plane_name(h)
                    ));
                }
                if let Some(fault) = self.scratch_read_fault(h, at, &written, g.scratch) {
                    return Some(fault);
                }
                stack.extend(self.reached_from(h).into_iter().map(|c| (c, at)));
            }
        }
        None
    }

    /// Whether two plane nodes name one plane: the same name, element type and axes.
    fn same_plane(&self, a: u32, b: u32) -> bool {
        let (x, y) = (&self.nodes[a as usize], &self.nodes[b as usize]);
        self.plane_name(a) == self.plane_name(b)
            && x.dtype == y.dtype
            && x.shape == y.shape
            && self.field(a, "axes").ok() == self.field(b, "axes").ok()
    }

    /// Why a body sink may not store where it stores: a store into scratch is an `Update` of op `store` into a scratch the group
    /// declares, and no other sink of the body takes the name of one.
    fn store_fault(&self, sink: u32, scratch: &[u32]) -> Option<String> {
        if self.class_name(sink) == "Barrier" {
            return None;
        }
        let name = self.plane_name(sink);
        let declared = scratch
            .iter()
            .copied()
            .find(|&s| self.plane_name(s) == name);
        if !self.is_scratch_sink(sink) {
            return declared.map(|_| {
                format!("the sink '{name}' has the name of a scratch of this group; give one of them another name")
            });
        }
        if !self.is_a(sink, "Update") || self.text_of(sink, "op").as_deref() != Some("store") {
            return Some(format!(
                "scratch '{name}' is written by an Update of op 'store'"
            ));
        }
        let Some(plane) = declared else {
            return Some(format!(
                "the store into scratch '{name}' names no scratch of this group; a group writes only the scratch it declares"
            ));
        };
        let (target, value) = (&self.nodes[plane as usize], &self.nodes[sink as usize]);
        if target.dtype != value.dtype || target.shape != value.shape {
            return Some(format!(
                "scratch '{name}' holds {} of shape {:?}, the store gives {} of shape {:?}",
                target.dtype, target.shape, value.dtype, value.shape
            ));
        }
        (self.field(sink, "axes").ok() != self.field(plane, "axes").ok()).then(|| {
            format!("the store into scratch '{name}' states other axes than the scratch declares")
        })
    }

    /// Why the scratch operands of node `h`, evaluated in phase `at`, may not be read so: the scratch is declared by this group,
    /// is read by `At`, and a read at another lane's element does not share its phase with a store into that scratch.
    fn scratch_read_fault(
        &self,
        h: u32,
        at: u32,
        written: &HashSet<(u32, String)>,
        scratch: &[u32],
    ) -> Option<String> {
        let node = &self.nodes[h as usize];
        for (position, &operand) in node.ops.iter().enumerate() {
            if !self.is_scratch_leaf(operand) {
                continue;
            }
            let name = self.plane_name(operand);
            if !scratch.iter().any(|&s| self.same_plane(s, operand)) {
                return Some(format!(
                    "scratch '{name}' is not declared by this group; a scratch is read and written only inside the group that declares it"
                ));
            }
            match (self.class_name(h), position) {
                ("At", 0) => {
                    let own = node.ops.get(1).is_some_and(|&i| self.is_a(i, "GroupRank"));
                    if !own && written.contains(&(at, name.clone())) {
                        return Some(format!(
                            "scratch '{name}' is read at another lane's element in the phase that writes it; put a barrier between the write and the read"
                        ));
                    }
                }
                (class, _) => {
                    return Some(format!("scratch '{name}' is read by At, not by a {class}"));
                }
            }
        }
        None
    }
}
