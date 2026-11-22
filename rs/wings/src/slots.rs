//! The carried slots of loops that can live in a plane: which slots change only by stores at an
//! index into themselves or element by element from themselves, and which step loops carry a wide
//! mutable they only overwrite.
use crate::construct::refs;
use crate::place::{kind_of, one_index, width, Kind, Use, ELEMENTWISE};
use crate::plan::Plan;
use crate::wide::rank;
use crate::{is_a, wide};
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::HashSet;

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// The carried placeholders of a loop.
pub(crate) fn carries(ctx: &Ctx, lp: u32) -> Outcome<Vec<u32>> {
    refs(ctx, lp, "carries")
}

/// The number of carried slots of a loop.
pub(crate) fn carry_count(ctx: &Ctx, lp: u32) -> Outcome<usize> {
    Ok(carries(ctx, lp)?.len())
}

/// The initial values of a loop's slots.
pub(crate) fn inits(ctx: &Ctx, lp: u32) -> Outcome<Vec<u32>> {
    Ok(ctx.nodes[lp as usize].ops[..carry_count(ctx, lp)?].to_vec())
}

/// The values the slots hold at the end of an iteration.
pub(crate) fn nexts(ctx: &Ctx, lp: u32) -> Outcome<Vec<u32>> {
    refs(ctx, lp, "nexts")
}

/// A loop whose carries a plane can hold: no early exit, no committed sinks, no run-time count.
pub(crate) fn is_plain(ctx: &Ctx, lp: u32) -> Outcome<bool> {
    let no_exit = ctx.field(lp, "exit_cond")? == Val::None;
    let no_count = ctx.field(lp, "count")? == Val::None;
    Ok(no_exit && no_count && refs(ctx, lp, "body_sinks")?.is_empty())
}

/// Whether the loop's break sits at the tail of its body.
pub(crate) fn is_tail(ctx: &Ctx, lp: u32) -> Outcome<bool> {
    Ok(ctx.exit_form(lp)? == Val::Str("tail".into()))
}

/// The integer in a node's `literal` field.
pub(crate) fn literal_int(ctx: &Ctx, node: u32) -> Option<i64> {
    match ctx.field(node, "literal") {
        Ok(Val::Int(i)) => Some(i as i64),
        _ => None,
    }
}

/// The kind of any node that has one.
fn any_kind(ctx: &Ctx, node: u32) -> String {
    kind_of(ctx, node)
        .ok()
        .flatten()
        .or_else(|| ctx.str_field(node, "kind").ok())
        .unwrap_or_default()
}

impl<'a> Plan<'a> {
    /// The slots that live in a plane because of how their loop reads and writes them.
    pub fn find_slots(&mut self) -> Outcome<()> {
        let (ctx, limits) = (self.ctx, self.limits);
        for &lp in &self.graph.loops {
            if !is_plain(ctx, lp)? {
                continue;
            }
            let (all_carries, all_inits) = (carries(ctx, lp)?, inits(ctx, lp)?);
            for (j, &carry) in all_carries.iter().enumerate() {
                let pattern = self.slot_pattern(lp, j)?;
                let size = width(ctx, carry);
                let chain = self.carry_chain(lp, j)?;
                let accum = !chain && self.carry_accum(lp, j)? && size > limits.runtime("accum");
                let kept = size > limits.wide_min
                    || pattern.is_some_and(|name| size > limits.runtime(name));
                if wide::element_kind(ctx, all_inits[j])? != "none" && (chain || accum) && kept {
                    self.slots.insert((lp, j));
                    if !chain {
                        self.accum.insert((lp, j));
                    }
                }
            }
        }
        Ok(())
    }

    /// How slot `j` is read at a run-time index: `rotate` in its own loop, `slot_read` from its
    /// final value, else none.
    fn slot_pattern(&self, lp: u32, j: usize) -> Outcome<Option<&'static str>> {
        let ctx = self.ctx;
        let reads = |usage: &Use| -> Outcome<bool> {
            Ok(self.indexed(usage)?
                && kind_of(ctx, usage.user)?.as_deref() == Some("component_at")
                && one_index(ctx, usage.user))
        };
        for usage in self.graph.users(carries(ctx, lp)?[j]) {
            if reads(usage)? {
                return Ok(Some("rotate"));
            }
        }
        for node in ctx.body_nodes(lp)? {
            let runtime_read = kind_of(ctx, node)?.as_deref() == Some("component_at")
                && is_a(ctx, node, "Op")
                && one_index(ctx, node)
                && !is_a(ctx, ctx.nodes[node as usize].ops[1], "Const");
            if runtime_read {
                return Ok(Some("slot_read"));
            }
        }
        for usage in self.graph.users(lp) {
            if is_a(ctx, usage.user, "LoopValue") && self.slot_of_value(usage.user) == (lp, j) {
                for inner in self.graph.users(usage.user) {
                    if reads(inner)? {
                        return Ok(Some("slot_read"));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Whether slot `j` is only changed by stores at an index into itself, so the plane holds it in
    /// place: its next value is a chain of `set_component_at` / `add_at` over the carry, or the
    /// carry unchanged.
    fn carry_chain(&self, lp: u32, j: usize) -> Outcome<bool> {
        let ctx = self.ctx;
        let (carry, mut node, mut links) = (carries(ctx, lp)?[j], nexts(ctx, lp)?[j], vec![]);
        while matches!(
            kind_of(ctx, node)?.as_deref(),
            Some("set_component_at" | "add_at")
        ) && is_a(ctx, node, "Op")
            && one_index(ctx, node)
        {
            links.push(node);
            node = ctx.nodes[node as usize].ops[0];
        }
        if node != carry {
            return Ok(false);
        }
        let uses = self.graph.users(carry);
        if links.is_empty() {
            return Ok(uses.iter().all(|u| {
                u.kind == Kind::Next
                    || (u.kind == Kind::Operand && u.k == 0 && self.reads_component(u.user))
            }));
        }
        let inside = self.stored_values(&links, carry);
        for usage in uses {
            let own_store = usage.user == lp && usage.kind == Kind::Next;
            if own_store || (links.contains(&usage.user) && usage.k == 0) {
                continue;
            }
            let read = self.reads_component(usage.user) && usage.k == 0;
            if read
                && self
                    .graph
                    .users(usage.user)
                    .iter()
                    .all(|u| inside.contains(&u.user))
            {
                continue;
            }
            return Ok(false);
        }
        Ok(true)
    }

    /// Whether the node is an `Op` reading one component of a vector.
    fn reads_component(&self, node: u32) -> bool {
        is_a(self.ctx, node, "Op")
            && any_kind(self.ctx, node) == "component_at"
            && one_index(self.ctx, node)
    }

    /// The nodes the stored values and indices of a chain of stores are computed from.
    fn stored_values(&self, links: &[u32], carry: u32) -> HashSet<u32> {
        let mut inside = HashSet::new();
        let mut stack: Vec<u32> = links
            .iter()
            .flat_map(|&l| self.ctx.nodes[l as usize].ops[1..].to_vec())
            .collect();
        while let Some(node) = stack.pop() {
            if inside.contains(&node) || node == carry {
                continue;
            }
            inside.insert(node);
            stack.extend(self.ctx.nodes[node as usize].ops.iter().copied());
        }
        inside
    }

    /// Whether slot `j` is changed element by element from itself: its next value is an
    /// elementwise operation whose operands include the carry, which nothing else reads.
    pub fn carry_accum(&self, lp: u32, j: usize) -> Outcome<bool> {
        let ctx = self.ctx;
        let (carry, next) = (carries(ctx, lp)?[j], nexts(ctx, lp)?[j]);
        let kind = kind_of(ctx, next)?;
        let shaped = kind.is_some_and(|k| ELEMENTWISE.contains(&k.as_str()));
        if !(shaped && !self.graph.users(carry).is_empty()) {
            return Ok(false);
        }
        let (mut seen, mut stack) = (HashSet::new(), self.graph.users(carry).to_vec());
        while let Some(usage) = stack.pop() {
            let node = usage.user;
            let kind = kind_of(ctx, node)?;
            let elementwise = kind.is_some_and(|k| ELEMENTWISE.contains(&k.as_str()));
            if usage.kind != Kind::Operand || !elementwise || rank(ctx, node) != 1 {
                return Ok(false);
            }
            if !seen.insert(node) || node == next {
                continue;
            }
            let onward = self.graph.users(node);
            if onward.is_empty() {
                return Ok(false);
            }
            stack.extend(onward.iter().copied());
        }
        Ok(seen.contains(&next))
    }

    /// The slots of step loops that carry a wide mutable the loop only overwrites: the carry is the
    /// mutable's own plane and each trip writes its next value there element by element.
    pub fn find_carried(&mut self) -> Outcome<()> {
        let ctx = self.ctx;
        for &lp in &self.graph.loops {
            let tail = ctx.exit_form(lp)? == Val::Str("tail".into());
            if !tail || !refs(ctx, lp, "body_sinks")?.is_empty() {
                continue;
            }
            for (j, &carry) in carries(ctx, lp)?.iter().enumerate() {
                self.try_carried(lp, j, carry)?;
            }
        }
        Ok(())
    }

    /// Records slot `j` as carried when its loop only overwrites the mutable it starts from.
    fn try_carried(&mut self, lp: u32, j: usize, carry: u32) -> Outcome<()> {
        let ctx = self.ctx;
        let (init, next) = (inits(ctx, lp)?[j], nexts(ctx, lp)?[j]);
        let prior = is_a(ctx, init, "Leaf")
            && ctx.str_field(init, "kind")? == "prior_read"
            && ctx.str_field(init, "role")? == "mutable";
        let elementwise = kind_of(ctx, next)?.is_some_and(|k| ELEMENTWISE.contains(&k.as_str()));
        if width(ctx, carry) <= self.limits.wide_min || !prior {
            return Ok(());
        }
        if !self.candidates.contains(&next) || !elementwise {
            return Ok(());
        }
        let reads = !self.graph.users(carry).is_empty();
        if reads && !self.carry_accum(lp, j)? {
            return Ok(());
        }
        let name = ctx.str_field(init, "name")?;
        let values: Vec<u32> = self
            .graph
            .users(lp)
            .iter()
            .map(|u| u.user)
            .filter(|&u| is_a(ctx, u, "LoopValue") && self.slot_of_value(u) == (lp, j))
            .collect();
        let stores: Vec<u32> = values
            .iter()
            .flat_map(|&v| self.graph.users(v).iter().map(|u| u.user))
            .collect();
        let whole = stores.len() == 1
            && is_a(ctx, stores[0], "Assign")
            && ctx.str_field(stores[0], "name")? == name
            && self.graph.users(values[0]).len() == 1;
        if !whole {
            return Ok(());
        }
        self.carried.insert((lp, j), (lp, name));
        if !reads && ctx.field(lp, "count")? == Val::None {
            if let Some(deps) = self.sink_deps(lp, next)? {
                self.sunk.insert((lp, j), deps);
            }
        }
        if reads {
            self.accum.insert((lp, j));
        }
        Ok(())
    }

    /// The scalar carries the next value of a step loop is computed from, when it can be computed
    /// once after the loop from their values at the start of the last trip; none when it reads the
    /// loop index, a vector carry or an inner loop.
    fn sink_deps(&self, lp: u32, next: u32) -> Outcome<Option<Vec<u32>>> {
        let ctx = self.ctx;
        let body: HashSet<u32> = ctx.body_nodes(lp)?.into_iter().collect();
        let (mut deps, mut seen, mut stack): (Vec<u32>, HashSet<u32>, Vec<u32>) =
            (vec![], HashSet::new(), vec![next]);
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            if is_a(ctx, node, "LoopCarry") {
                if rank(ctx, node) > 0 {
                    return Ok(None);
                }
                if !deps.contains(&node) {
                    deps.push(node);
                }
            } else if is_a(ctx, node, "LoopIndex") {
                return Ok(None);
            } else if body.contains(&node) {
                let kind = any_kind(ctx, node);
                if is_a(ctx, node, "Loop") || kind == "tape_read" || kind == "loop_count" {
                    return Ok(None);
                }
                stack.extend(ctx.nodes[node as usize].ops.iter().copied());
            }
        }
        Ok(Some(deps))
    }
}
