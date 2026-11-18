//! What the placement pass decides: which wide values are written into a plane, which are written
//! element by element into the plane of their one user, and which carried slots of a loop live in a
//! plane. The analyses of loop slots are in `slots.rs`.
use crate::place::{is_a_resident, kind_of, width, Graph, Kind, Limits, Use, ELEMENTWISE};
use crate::wide::{element_kind, is_prior, is_scratch, rank};
use crate::{fold_rolled, is_a};
use raptor_ir::record::{refuse, Ctx, Fault};
use std::collections::{BTreeMap, HashSet};

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// Whether `n` exceeds the fold threshold; refused when the facts name no threshold.
pub(crate) fn exceeds(n: i64, limit: Option<i64>) -> Outcome<bool> {
    match limit {
        Some(limit) => Ok(n > limit),
        None => refuse("the facts of this build name no fold threshold (form.rolled_fold_min)"),
    }
}

/// Whether `n` exceeds the fold threshold (*out is 0 or 1); refused when `fold_min` is NULL, the facts naming no threshold.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `fold_min` is NULL or valid for one read, `out` for
/// one write.
#[no_mangle]
pub unsafe extern "C" fn rw_fold_exceeds(
    ctx: *mut Ctx,
    n: i64,
    fold_min: *const i64,
    out: *mut std::ffi::c_int,
) -> std::ffi::c_int {
    raptor_ir::door::guard(ctx, |_| {
        let limit = if fold_min.is_null() {
            None
        } else {
            Some(unsafe { *fold_min })
        };
        crate::flag_out(out, exceeds(n, limit)?)
    })
}

/// The reductions of a whole vector that read it element by element when the site is rolled.
const REDUCTIONS: [&str; 3] = ["sum", "dot", "norm"];

/// A carried slot: the loop and the slot number.
pub(crate) type Slot = (u32, usize);

/// The decision for one kernel.
pub(crate) struct Plan<'a> {
    /// The context the kernel is read from.
    pub ctx: &'a Ctx,
    /// The kernel's nodes and their uses.
    pub graph: &'a Graph,
    /// What the target keeps in registers.
    pub limits: &'a Limits,
    /// The wide values that may need a plane.
    pub candidates: Vec<u32>,
    /// The values placed in a plane.
    pub resident: HashSet<u32>,
    /// The values computed inside their consumer, needing no plane.
    pub fused: HashSet<u32>,
    /// The slots that live in a plane.
    pub slots: HashSet<Slot>,
    /// The slots whose next value is an elementwise function of the carry itself.
    pub accum: HashSet<Slot>,
    /// The slots that carry a wide mutable a step loop only overwrites: the loop and the mutable.
    pub carried: BTreeMap<Slot, (u32, String)>,
    /// The carried slots written once after the loop, with the scalar carries their value needs.
    pub sunk: BTreeMap<Slot, Vec<u32>>,
}

impl<'a> Plan<'a> {
    /// The decision for the graph: what wants a plane, what follows it, what the text can spell.
    pub fn decide(ctx: &'a Ctx, graph: &'a Graph, limits: &'a Limits) -> Outcome<Plan<'a>> {
        let mut closed = HashSet::new();
        for &lp in &graph.loops {
            if !crate::slots::is_plain(ctx, lp)?
                && !crate::construct::refs(ctx, lp, "body_sinks")?.is_empty()
            {
                closed.insert(lp);
            }
        }
        let mut candidates = vec![];
        for &node in &graph.order {
            let shaped = matches!(element_kind(ctx, node)?, "parts" | "splat" | "elementwise");
            let open = graph.scope[&node].is_none_or(|scope| !closed.contains(&scope));
            if shaped && open && (is_a(ctx, node, "Op") || is_a(ctx, node, "Select")) {
                candidates.push(node);
            }
        }
        let empty = HashSet::new;
        let mut plan = Plan {
            ctx,
            graph,
            limits,
            candidates,
            resident: empty(),
            fused: empty(),
            slots: HashSet::new(),
            accum: HashSet::new(),
            carried: BTreeMap::new(),
            sunk: BTreeMap::new(),
        };
        plan.find_slots()?;
        for node in plan.candidates.clone() {
            if plan.wanted(node)? {
                plan.resident.insert(node);
            }
        }
        plan.find_carried()?;
        plan.add_slot_values()?;
        plan.follow()?;
        plan.settle()?;
        Ok(plan)
    }

    /// The values that go into a plane because their slot does: the next values of carried and
    /// accumulating slots.
    fn add_slot_values(&mut self) -> Outcome<()> {
        let ctx = self.ctx;
        for &(lp, j) in self.carried.keys() {
            self.resident.insert(crate::slots::nexts(ctx, lp)?[j]);
        }
        for &lp in &self.graph.loops {
            for j in 0..crate::slots::carry_count(ctx, lp)? {
                if self.accum.contains(&(lp, j)) {
                    self.resident.insert(crate::slots::nexts(ctx, lp)?[j]);
                }
            }
        }
        Ok(())
    }

    /// Whether the use is an index into the value: a read or a store at a run-time index.
    pub fn indexed(&self, usage: &Use) -> Outcome<bool> {
        let Some(kind) = kind_of(self.ctx, usage.user)? else {
            return Ok(false);
        };
        if usage.kind != Kind::Operand || !is_a(self.ctx, usage.user, "Op") || usage.k != 0 {
            return Ok(false);
        }
        let index = self.ctx.nodes[usage.user as usize].ops.get(1).copied();
        let runtime = index.is_some_and(|i| !is_a(self.ctx, i, "Const"));
        Ok(kind == "component_at" || (kind == "set_component_at" && runtime))
    }

    /// Whether a rolled reduction reads the value element by element.
    pub fn reads_elements(&self, usage: &Use) -> Outcome<bool> {
        let ctx = self.ctx;
        let Some(kind) = kind_of(ctx, usage.user)? else {
            return Ok(false);
        };
        let reduces = is_a(ctx, usage.user, "Op") && REDUCTIONS.contains(&kind.as_str());
        if usage.kind != Kind::Operand || !reduces {
            return Ok(false);
        }
        let first = ctx.nodes[usage.user as usize].ops[0];
        Ok(rank(ctx, first) == 1
            && fold_rolled(ctx, usage.user, self.limits.rolled_min, &self.limits.form)?)
    }

    /// Whether an elementwise user in another scope reads the value.
    fn elsewhere(&self, node: u32, usage: &Use) -> Outcome<bool> {
        let kind = kind_of(self.ctx, usage.user)?;
        let elementwise = kind.is_some_and(|k| ELEMENTWISE.contains(&k.as_str()));
        Ok(usage.kind == Kind::Operand
            && elementwise
            && self.graph.scope[&usage.user] != self.graph.scope[&node])
    }

    /// Whether a value gets a plane: a runtime index past the register limit, an element-by-element
    /// reader, or a wide elementwise result stored to a plane or read from two places.
    fn wanted(&self, node: u32) -> Outcome<bool> {
        let (ctx, limits) = (self.ctx, self.limits);
        let (size, uses) = (width(ctx, node), self.graph.users(node));
        let mut least: Option<i64> = None;
        for usage in uses {
            if self.indexed(usage)? {
                let stores = kind_of(ctx, usage.user)?.as_deref() == Some("set_component_at");
                let limit = limits.runtime(if stores { "write" } else { "read" });
                least = Some(least.map_or(limit, |l| l.min(limit)));
            }
        }
        if let Some(limit) = least {
            return Ok(size > limit);
        }
        if !exceeds(size, limits.rolled_min)? {
            return Ok(false);
        }
        for usage in uses {
            if self.reads_elements(usage)? || self.elsewhere(node, usage)? {
                return Ok(true);
            }
        }
        let elementwise = kind_of(ctx, node)?.is_some_and(|k| ELEMENTWISE.contains(&k.as_str()));
        if elementwise && size > limits.wide_min {
            return Ok(uses.iter().any(|u| is_a(ctx, u.user, "Assign")) || uses.len() >= 2);
        }
        Ok(false)
    }

    /// Whether the node is a placed value or the result of a placed slot.
    fn placed_operand(&self, node: u32) -> bool {
        if is_a(self.ctx, node, "LoopValue") {
            return self.slots.contains(&self.slot_of_value(node));
        }
        self.resident.contains(&node)
    }

    /// The slot a loop result reads.
    pub fn slot_of_value(&self, value: u32) -> Slot {
        let slot = crate::slots::literal_int(self.ctx, value).unwrap_or(0) as usize;
        (self.ctx.nodes[value as usize].ops[0], slot)
    }

    /// An elementwise value that reads a placed value or slot is placed too, of any width.
    fn follow(&mut self) -> Outcome<()> {
        let mut grew = true;
        while grew {
            grew = false;
            for &node in &self.candidates {
                let kind = kind_of(self.ctx, node)?.unwrap_or_default();
                if self.resident.contains(&node) || !ELEMENTWISE.contains(&kind.as_str()) {
                    continue;
                }
                let operands = &self.ctx.nodes[node as usize].ops;
                if operands
                    .iter()
                    .any(|&o| rank(self.ctx, o) == 1 && self.placed_operand(o))
                {
                    self.resident.insert(node);
                    grew = true;
                }
            }
        }
        Ok(())
    }

    /// Whether an elementwise value written into a plane can read the rank-1 `node` one element
    /// at a time.
    fn readable(&self, node: u32, fused: &HashSet<u32>) -> Outcome<bool> {
        let ctx = self.ctx;
        if is_prior(ctx, node)? || is_scratch(ctx, node)? || is_a_resident(ctx, node)? {
            return Ok(true);
        }
        if self.resident.contains(&node) || fused.contains(&node) {
            return Ok(true);
        }
        if is_a(ctx, node, "LoopValue") {
            return Ok(self.slots.contains(&self.slot_of_value(node)));
        }
        if is_a(ctx, node, "LoopCarry") {
            return self.accumulating_carry(node);
        }
        Ok(element_kind(ctx, node)? == "input")
    }

    /// Whether the node is the carry of a slot that accumulates in its plane.
    fn accumulating_carry(&self, node: u32) -> Outcome<bool> {
        for &lp in &self.graph.loops {
            for (j, carry) in crate::slots::carries(self.ctx, lp)?.into_iter().enumerate() {
                if carry == node && self.accum.contains(&(lp, j)) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Whether every vector operand of the node is readable.
    pub fn supported(&self, node: u32, fused: &HashSet<u32>) -> Outcome<bool> {
        for &operand in &self.ctx.nodes[node as usize].ops {
            if rank(self.ctx, operand) == 1 && !self.readable(operand, fused)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether the use is one the text spells for a value that lives in a plane.
    fn allowed(&self, usage: &Use, node: u32) -> Outcome<bool> {
        let (ctx, user) = (self.ctx, usage.user);
        match usage.kind {
            Kind::Init => return Ok(self.slots.contains(&(user, usage.k))),
            Kind::Next => {
                let slot = (user, usage.k);
                return Ok(self.carried.contains_key(&slot) || self.accum.contains(&slot));
            }
            Kind::Body => return self.body_use_allowed(user, node),
            Kind::Boundary => return Ok(true),
            Kind::Operand => {}
        }
        if is_a(ctx, user, "Assign") {
            return Ok(true);
        }
        let Some(kind) = kind_of(ctx, user)? else {
            return Ok(false);
        };
        let kind = kind.as_str();
        if ELEMENTWISE.contains(&kind) {
            return Ok(self.resident.contains(&user) || self.fused.contains(&user));
        }
        if REDUCTIONS.contains(&kind) {
            return self.reads_elements(usage);
        }
        if kind == "set_component_at" {
            return Ok(usage.k == 0 && self.graph.users(node).len() == 1);
        }
        Ok(matches!(kind, "component_at" | "component") && usage.k == 0)
    }

    /// A tail break's exit values are the next values, which its text does not read.
    fn body_use_allowed(&self, user: u32, node: u32) -> Outcome<bool> {
        if !crate::slots::is_tail(self.ctx, user)? {
            return Ok(false);
        }
        for &(lp, j) in self.carried.keys() {
            if lp == user && crate::slots::nexts(self.ctx, lp)?[j] == node {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The values written element by element into the plane of the use that places them, in their
    /// own scope: a placed or fused elementwise value, or the initial value of a placed slot.
    fn fuse(&self) -> Outcome<HashSet<u32>> {
        let mut found = HashSet::new();
        for &node in &self.candidates {
            let uses = self.graph.users(node);
            let kind = element_kind(self.ctx, node)?;
            if self.resident.contains(&node) || uses.is_empty() {
                continue;
            }
            if uses.len() > 1 && !matches!(kind, "parts" | "splat") {
                continue;
            }
            let mut all = true;
            for usage in uses {
                all &= self.fusable(usage, node)?;
            }
            if all {
                found.insert(node);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for node in self
                .candidates
                .iter()
                .filter(|n| found.contains(*n))
                .copied()
                .collect::<Vec<_>>()
            {
                let placed = self.graph.users(node).iter().all(|u| {
                    u.kind == Kind::Init
                        || self.resident.contains(&u.user)
                        || found.contains(&u.user)
                });
                if !(placed && self.supported(node, &found)?) {
                    found.remove(&node);
                    changed = true;
                }
            }
        }
        Ok(found)
    }

    /// Whether the use can write the value element by element into its own plane.
    fn fusable(&self, usage: &Use, node: u32) -> Outcome<bool> {
        if usage.kind == Kind::Init {
            return Ok(self.slots.contains(&(usage.user, usage.k)));
        }
        let kind = kind_of(self.ctx, usage.user)?;
        let elementwise = kind.is_some_and(|k| ELEMENTWISE.contains(&k.as_str()));
        Ok(usage.kind == Kind::Operand
            && elementwise
            && self.graph.scope[&usage.user] == self.graph.scope[&node])
    }

    /// Drops what cannot be spelled until nothing changes.
    fn settle(&mut self) -> Outcome<()> {
        loop {
            self.fused = self.fuse()?;
            let mut drop = vec![];
            for &node in self
                .candidates
                .iter()
                .filter(|n| self.resident.contains(*n))
            {
                let mut allowed = self.supported(node, &self.fused)?;
                for usage in self.graph.users(node) {
                    allowed = allowed && self.allowed(usage, node)?;
                }
                if !allowed {
                    drop.push(node);
                }
            }
            let mut slots = HashSet::new();
            for &slot in &self.slots {
                if self.slot_ok(slot)? {
                    slots.insert(slot);
                }
            }
            let mut carried = BTreeMap::new();
            for (&slot, held) in &self.carried {
                let next = crate::slots::nexts(self.ctx, slot.0)?[slot.1];
                if self.resident.contains(&next) && !drop.contains(&next) {
                    carried.insert(slot, held.clone());
                }
            }
            if drop.is_empty() && slots == self.slots && carried.len() == self.carried.len() {
                return Ok(());
            }
            for node in drop {
                self.resident.remove(&node);
            }
            self.accum
                .retain(|a| slots.contains(a) || carried.contains_key(a));
            self.slots = slots;
            self.carried = carried;
        }
    }

    /// Whether the slot's initial value can be read element by element and every use of its final
    /// value is one the text spells.
    fn slot_ok(&self, slot: Slot) -> Outcome<bool> {
        let init = crate::slots::inits(self.ctx, slot.0)?[slot.1];
        if !self.supported_one(init)? {
            return Ok(false);
        }
        for usage in self.graph.users(slot.0) {
            let user = usage.user;
            if is_a(self.ctx, user, "LoopValue") && self.slot_of_value(user) == slot {
                for inner in self.graph.users(user) {
                    if !self.allowed(inner, user)? {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    /// `supported` of a node itself, as a slot's initial value is checked.
    fn supported_one(&self, node: u32) -> Outcome<bool> {
        self.supported(node, &self.fused)
    }
}
