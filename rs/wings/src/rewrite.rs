//! The rewrite of the placement pass: the sinks over the values of their operands, every placed
//! value
//! wrapped in the node that binds it to its plane, and every loop with placed slots made again
//! by the
//! core's loop constructor.
use crate::construct::refs;
use crate::is_a;
use crate::place::{kind_of, one_index, Graph};
use crate::plan::Slot;
use crate::planes::{place_node, plane_leaf, Planes};
use crate::slots::{carries, inits, nexts};
use crate::wide::{gather, rank};
use raptor_ir::build::LoopParts;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::{text_val, Val};
use std::collections::{BTreeMap, HashMap, HashSet};

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// The sink classes whose copy over other operands is the placed sink.
const RESINKED: [&str; 3] = ["SampleSum", "Assign", "Update"];

/// What a rewrite is told: the planes, which slots are carried or sunk, what to apply to each
/// rebuilt
/// node and to each rebuilt loop.
pub(crate) struct Rules<'a> {
    /// The plane of each placed value and slot.
    pub planes: &'a Planes,
    /// The slots that live in a plane across iterations.
    pub carried: &'a HashSet<Slot>,
    /// The slots whose value is written to a plane at the end of each iteration, with their nodes.
    pub sunk: &'a BTreeMap<Slot, Vec<u32>>,
    /// What to apply to each rebuilt node.
    pub each: fn(&mut Ctx, u32) -> Outcome<u32>,
    /// What to apply to each rebuilt loop, when anything.
    pub on_loop: Option<&'a LoopHook<'a>>,
}

/// A callback that rebuilds a loop's nodes: the context, the loop and the node to rebuild.
type LoopHook<'a> = dyn Fn(&mut Ctx, u32, u32) -> Outcome<u32> + 'a;

/// What a loop's root list holds after its carried nexts: the exit condition, the exit values and
/// the `(start, stop, step)` range.
struct LoopRoots {
    /// The loop's exit condition, when it has one.
    cond: Option<u32>,
    /// The values the loop gives when it exits early.
    values: Vec<u32>,
    /// The loop's start, stop and step.
    range: (i64, i64, i64),
}

/// A vector of scaled reads, `vec(s * a_0, s * a_1, ...)`, as `splat(s) * vec(a_0, a_1, ...)`.
pub(crate) fn scaled_gather(ctx: &mut Ctx, node: u32) -> Outcome<u32> {
    let parts = ctx.nodes[node as usize].ops.clone();
    let vector = kind_of(ctx, node)?.as_deref() == Some("vec") && is_a(ctx, node, "Op");
    if !(vector && rank(ctx, node) == 1 && parts.len() > 4) {
        return Ok(node);
    }
    for &part in &parts {
        let product = kind_of(ctx, part)?.as_deref() == Some("mul") && is_a(ctx, part, "Op");
        if !(product && ctx.nodes[part as usize].ops.len() == 2 && rank(ctx, part) == 0) {
            return Ok(node);
        }
    }
    for side in 0..2 {
        let scale = ctx.nodes[parts[0] as usize].ops[side];
        let shared = parts
            .iter()
            .all(|&p| ctx.nodes[p as usize].ops[side] == scale);
        if rank(ctx, scale) > 0 || is_a(ctx, scale, "At") || !shared {
            continue;
        }
        let reads: Vec<u32> = parts
            .iter()
            .map(|&p| ctx.nodes[p as usize].ops[1 - side])
            .collect();
        if !reads
            .iter()
            .all(|&r| is_a(ctx, r, "At") && rank(ctx, r) == 0)
        {
            continue;
        }
        let rows = ctx.make_op("vec", &reads, &Val::None)?;
        if gather(ctx, rows)?.is_none() {
            continue;
        }
        let shape = Val::Tuple(
            ctx.nodes[node as usize]
                .shape
                .iter()
                .map(|&n| Val::Int(n as _))
                .collect(),
        );
        let spread = ctx.make_op("splat", &[scale], &shape)?;
        let pair = if side == 0 {
            [spread, rows]
        } else {
            [rows, spread]
        };
        return ctx.make_op("mul", &pair, &Val::None);
    }
    Ok(node)
}

/// A rewrite in progress: what each old node became.
struct Rewriter<'a, 'r> {
    /// The context the new nodes are made in.
    ctx: &'a mut Ctx,
    /// What the rewrite was told.
    rules: &'a Rules<'r>,
    /// The new node each old node became.
    memo: HashMap<u32, u32>,
    /// The value a sunk slot is written with after its loop.
    after: HashMap<Slot, u32>,
}

impl Rewriter<'_, '_> {
    /// What the node became, else itself.
    fn done(&self, node: u32) -> u32 {
        self.memo.get(&node).copied().unwrap_or(node)
    }

    /// The nodes made again over what their operands became.
    fn rebuild_node(&mut self, node: u32) -> Outcome<()> {
        if is_a(self.ctx, node, "LoopValue") {
            let slot = (
                self.ctx.nodes[node as usize].ops[0],
                literal(self.ctx, node)?,
            );
            if let Some(&value) = self.after.get(&slot) {
                self.memo.insert(node, value);
                return Ok(());
            }
        }
        if is_a(self.ctx, node, "Loop") {
            let made = self.loop_of(node)?;
            let made = match self.rules.on_loop {
                Some(on_loop) => on_loop(self.ctx, node, made)?,
                None => made,
            };
            self.memo.insert(node, made);
            return Ok(());
        }
        let operands = self.ctx.nodes[node as usize].ops.clone();
        let now: Vec<u32> = operands.iter().map(|&o| self.done(o)).collect();
        let mut new = if now == operands {
            node
        } else {
            self.ctx.rebuild(node, &now)?
        };
        new = (self.rules.each)(self.ctx, new)?;
        if let Some(name) = self.rules.planes.of.get(&node) {
            let scratch = self.rules.planes.scratch.contains_key(name);
            new = place_node(self.ctx, new, node, name, scratch)?;
        }
        self.memo.insert(node, new);
        Ok(())
    }

    /// The loop over what its inits, body and bound became; the loop itself when nothing moved.
    fn loop_of(&mut self, lp: u32) -> Outcome<u32> {
        for node in self.ctx.body_nodes(lp)? {
            if !self.memo.contains_key(&node) {
                self.rebuild_node(node)?;
            }
        }
        let (old_inits, all_carries) = (inits(self.ctx, lp)?, carries(self.ctx, lp)?);
        let mut new_inits: Vec<u32> = old_inits.iter().map(|&i| self.done(i)).collect();
        for (j, &carry) in all_carries.iter().enumerate() {
            if let Some(name) = self.rules.planes.slots.get(&(lp, j)) {
                let scratch = self.rules.planes.scratch.contains_key(name);
                new_inits[j] = if self.rules.carried.contains(&(lp, j)) {
                    plane_leaf(self.ctx, carry, name, scratch)?
                } else {
                    place_node(self.ctx, new_inits[j], carry, name, scratch)?
                };
            }
        }
        let old_roots = self.ctx.body_roots(lp)?;
        let roots: Vec<u32> = old_roots.iter().map(|&r| self.done(r)).collect();
        let operands = self.ctx.nodes[lp as usize].ops.clone();
        let free_moved = operands[old_inits.len()..]
            .iter()
            .any(|&f| self.done(f) != f);
        if new_inits == old_inits && roots == old_roots && !free_moved {
            return Ok(lp);
        }
        let sunk: Vec<usize> = (0..all_carries.len())
            .filter(|&j| self.rules.sunk.contains_key(&(lp, j)))
            .collect();
        if sunk.is_empty() {
            self.build_plain(lp, new_inits, roots)
        } else {
            self.build_sunk(lp, new_inits, roots, &sunk)
        }
    }

    /// The bound and break of a loop over the rewritten roots.
    fn parts_of(&self, lp: u32, roots: &[u32]) -> Outcome<LoopRoots> {
        let n = carries(self.ctx, lp)?.len();
        let exit = self.ctx.field(lp, "exit_cond")? != Val::None;
        let (cond, values) = if exit {
            (Some(roots[n]), roots[n + 1..].to_vec())
        } else {
            (None, vec![])
        };
        let int = |name: &str| match self.ctx.field(lp, name) {
            Ok(Val::Int(i)) => Ok(i as i64),
            _ => refuse("a loop bound is an int"),
        };
        Ok(LoopRoots {
            cond,
            values,
            range: (int("start")?, int("stop")?, int("step")?),
        })
    }

    /// The loop made again with its slots in planes.
    fn build_plain(&mut self, lp: u32, new_inits: Vec<u32>, roots: Vec<u32>) -> Outcome<u32> {
        let n = carries(self.ctx, lp)?.len();
        let LoopRoots {
            cond,
            values,
            range,
        } = self.parts_of(lp, &roots)?;
        let has_count = self.ctx.field(lp, "count")? != Val::None;
        let count = if has_count {
            self.ctx.nodes[lp as usize]
                .ops
                .last()
                .map(|&c| self.done(c))
        } else {
            None
        };
        let dynamic = match self.ctx.field(lp, "dynamic")? {
            Val::Str(text) => Some(text),
            _ => None,
        };
        let count_tail = self.ctx.field(lp, "count_tail")? == Val::Bool(true);
        let index = refs(self.ctx, lp, "index")?[0];
        let old_carries = carries(self.ctx, lp)?;
        let parts = LoopParts {
            index,
            carries: &old_carries,
            inits: &new_inits,
            nexts: &roots[..n],
            range,
            body_sinks: &[],
            exit_cond: cond,
            exit_values: &values,
            count,
            count_tail,
            dynamic: dynamic.as_deref(),
        };
        self.ctx.build_loop(&parts)
    }

    /// The loop with the sunk slots' planes written once after it, from a shadow of each scalar
    /// carry
    /// holding its value at the start of the last trip.
    fn build_sunk(
        &mut self,
        lp: u32,
        mut new_inits: Vec<u32>,
        roots: Vec<u32>,
        sunk: &[usize],
    ) -> Outcome<u32> {
        let n = carries(self.ctx, lp)?.len();
        let LoopRoots {
            cond,
            mut values,
            range,
        } = self.parts_of(lp, &roots)?;
        let old_carries = carries(self.ctx, lp)?;
        let (mut all_carries, mut nexts_now) = (old_carries.clone(), roots[..n].to_vec());
        let last: HashMap<usize, u32> = sunk.iter().map(|&j| (j, nexts_now[j])).collect();
        let mut shadow: Vec<u32> = vec![];
        for &j in sunk {
            for &c in &self.rules.sunk[&(lp, j)] {
                if !shadow.contains(&c) {
                    shadow.push(c);
                }
            }
        }
        for &j in sunk {
            nexts_now[j] = old_carries[j];
            values[j] = old_carries[j];
        }
        let mut made = HashMap::new();
        for &c in &shadow {
            let new = self.shadow_carry(c, all_carries.len())?;
            made.insert(c, (new, all_carries.len()));
            let slot = literal(self.ctx, c)?;
            all_carries.push(new);
            new_inits.push(new_inits[slot]);
            nexts_now.push(c);
            values.push(c);
        }
        let dynamic = match self.ctx.field(lp, "dynamic")? {
            Val::Str(text) => Some(text),
            _ => None,
        };
        let index = refs(self.ctx, lp, "index")?[0];
        let parts = LoopParts {
            index,
            carries: &all_carries,
            inits: &new_inits,
            nexts: &nexts_now,
            range,
            body_sinks: &[],
            exit_cond: cond,
            exit_values: &values,
            count: None,
            count_tail: false,
            dynamic: dynamic.as_deref(),
        };
        let new = self.ctx.build_loop(&parts)?;
        self.write_after(lp, new, sunk, &last, &made)?;
        Ok(new)
    }

    /// A scalar carry of the same name with `_last` added, standing for its value at the start of
    /// the last trip, as slot `slot`.
    fn shadow_carry(&mut self, carry: u32, slot: usize) -> Outcome<u32> {
        let (depth, name) = (
            self.ctx.field(carry, "depth")?,
            self.ctx.field(carry, "name")?,
        );
        let name = match name {
            Val::Str(text) => Val::Str(format!("{text}_last")),
            other => other,
        };
        let crate::planes::NodeType { dtype, shape, tag } =
            crate::planes::type_parts(self.ctx, carry);
        let fields = format!(
            "((\"depth\", {}), (\"name\", {}), (\"slot\", {slot}))",
            text_val(&depth),
            text_val(&name)
        );
        self.ctx
            .make("LoopCarry", &dtype, &shape, tag.as_deref(), &[], &fields)
    }

    /// For each sunk slot the value its plane is written with after the loop.
    fn write_after(
        &mut self,
        lp: u32,
        new: u32,
        sunk: &[usize],
        last: &HashMap<usize, u32>,
        made: &HashMap<u32, (u32, usize)>,
    ) -> Outcome<()> {
        let (mut from, mut to) = (vec![], vec![]);
        for (&carry, &(shadow, slot)) in made {
            let crate::planes::NodeType { dtype, shape, tag } =
                crate::planes::type_parts(self.ctx, shadow);
            let fields = format!("((\"kind\", \"loop_value\"), (\"literal\", {slot}))");
            let value =
                self.ctx
                    .make("LoopValue", &dtype, &shape, tag.as_deref(), &[new], &fields)?;
            from.push(carry);
            to.push(value);
        }
        let projected: HashMap<u32, u32> = from.into_iter().zip(to).collect();
        for &j in sunk {
            let value = self.ctx.substitute(&[last[&j]], &projected)?[0];
            let name = &self.rules.planes.slots[&(lp, j)];
            let scratch = self.rules.planes.scratch.contains_key(name);
            let carry = carries(self.ctx, lp)?[j];
            let placed = place_node(self.ctx, value, carry, name, scratch)?;
            self.after.insert((lp, j), placed);
        }
        Ok(())
    }

    /// Runs the rewrite over the sinks, in the order the sinks are given.
    fn run(&mut self, sinks: &[u32]) -> Outcome<Vec<u32>> {
        let mut stack: Vec<(u32, bool)> = sinks.iter().rev().map(|&s| (s, false)).collect();
        while let Some((node, expanded)) = stack.pop() {
            if self.memo.contains_key(&node) {
                continue;
            }
            let operands = self.ctx.node(node)?.ops.clone();
            if !expanded {
                stack.push((node, true));
                stack.extend(
                    operands
                        .iter()
                        .rev()
                        .filter(|o| !self.memo.contains_key(*o))
                        .map(|&o| (o, false)),
                );
                continue;
            }
            if is_a(self.ctx, node, "Sink") {
                self.resink(node, &operands)?;
            } else {
                self.rebuild_node(node)?;
            }
        }
        Ok(sinks.iter().map(|s| self.memo[s]).collect())
    }

    /// A sink over what its operands became; a tape write stays as it is.
    fn resink(&mut self, node: u32, operands: &[u32]) -> Outcome<()> {
        let now: Vec<u32> = operands.iter().map(|&o| self.done(o)).collect();
        let copied =
            is_a(self.ctx, node, "Finish") || RESINKED.iter().any(|c| is_a(self.ctx, node, c));
        let new = if now == operands || !copied {
            node
        } else {
            self.ctx.rebuild(node, &now)?
        };
        self.memo.insert(node, new);
        Ok(())
    }
}

/// The slot number a loop value or carry names.
fn literal(ctx: &Ctx, node: u32) -> Outcome<usize> {
    let field = if is_a(ctx, node, "LoopCarry") {
        "slot"
    } else {
        "literal"
    };
    match ctx.field(node, field)? {
        Val::Int(i) => Ok(i as usize),
        _ => refuse("a slot number is an int"),
    }
}

/// The sinks over the rewritten values of their operands, by the rules.
pub(crate) fn rewritten(ctx: &mut Ctx, sinks: &[u32], rules: &Rules) -> Outcome<Vec<u32>> {
    Rewriter {
        ctx,
        rules,
        memo: HashMap::new(),
        after: HashMap::new(),
    }
    .run(sinks)
}

/// Whether carry `j` of the loop `new` (made from `old`) is a vector zeroed behind its own reads:
/// each component is read once at the loop index and the next value stores a constant there, and
/// nothing reads the vector after the loop.
fn dead_zeroing(ctx: &Ctx, graph: &Graph, old: u32, new: u32, j: usize) -> Outcome<bool> {
    let (carry, next) = (carries(ctx, new)?[j], nexts(ctx, new)?[j]);
    let index = refs(ctx, new, "index")?[0];
    let read_after = graph.users(old).iter().any(|u| {
        is_a(ctx, u.user, "LoopValue") && literal(ctx, u.user).is_ok_and(|slot| slot == j)
    });
    let ops = &ctx.nodes[next as usize].ops;
    let zeroing = is_a(ctx, next, "Op")
        && kind_of(ctx, next)?.as_deref() == Some("set_component_at")
        && one_index(ctx, next)
        && ops[..2] == [carry, index]
        && is_a(ctx, ops[2], "Const");
    if read_after || ctx.nodes[carry as usize].shape.is_empty() || !zeroing {
        return Ok(false);
    }
    let body = ctx.body_nodes(new)?;
    let mut uses = 0;
    let mut reads = vec![];
    for &node in &body {
        for (k, &operand) in ctx.nodes[node as usize].ops.iter().enumerate() {
            uses += (operand == carry && !(node == next && k == 0)) as usize;
        }
        let read = kind_of(ctx, node)?.as_deref() == Some("component_at")
            && is_a(ctx, node, "Op")
            && one_index(ctx, node);
        if read && ctx.nodes[node as usize].ops[0] == carry {
            reads.push(node);
        }
    }
    let others = nexts(ctx, new)?
        .iter()
        .enumerate()
        .any(|(k, &n)| k != j && n == carry);
    let plain = ctx.field(new, "exit_cond")? == Val::None;
    let at_index = reads.iter().all(|&r| ctx.nodes[r as usize].ops[1] == index);
    Ok(uses == reads.len()
        && at_index
        && !others
        && plain
        && refs(ctx, new, "body_sinks")?.is_empty())
}

/// The loop `new` made again with the carries in `drop` left as they are.
fn keep_carries(ctx: &mut Ctx, new: u32, drop: &[usize]) -> Outcome<u32> {
    let (all_carries, all_nexts) = (carries(ctx, new)?, nexts(ctx, new)?);
    let kept: Vec<u32> = (0..all_nexts.len())
        .map(|j| {
            if drop.contains(&j) {
                all_carries[j]
            } else {
                all_nexts[j]
            }
        })
        .collect();
    let int = |name: &str| match ctx.field(new, name) {
        Ok(Val::Int(i)) => Ok(i as i64),
        _ => refuse("a loop bound is an int"),
    };
    let range = (int("start")?, int("stop")?, int("step")?);
    let has_count = ctx.field(new, "count")? != Val::None;
    let count = if has_count {
        ctx.nodes[new as usize].ops.last().copied()
    } else {
        None
    };
    let dynamic = match ctx.field(new, "dynamic")? {
        Val::Str(text) => Some(text),
        _ => None,
    };
    let count_tail = ctx.field(new, "count_tail")? == Val::Bool(true);
    let (index, all_inits) = (refs(ctx, new, "index")?[0], inits(ctx, new)?);
    let parts = LoopParts {
        index,
        carries: &all_carries,
        inits: &all_inits,
        nexts: &kept,
        range,
        body_sinks: &[],
        exit_cond: None,
        exit_values: &[],
        count,
        count_tail,
        dynamic: dynamic.as_deref(),
    };
    ctx.build_loop(&parts)
}

/// The loop with each carried vector that is read once per component and zeroed behind its reads
/// left as it is (the zero is never read); the loop itself when there is none.
pub(crate) fn dead_take(ctx: &mut Ctx, graph: &Graph, old: u32, new: u32) -> Outcome<u32> {
    let mut drop = vec![];
    for j in 0..carries(ctx, new)?.len() {
        if dead_zeroing(ctx, graph, old, new, j)? {
            drop.push(j);
        }
    }
    if drop.is_empty() {
        Ok(new)
    } else {
        keep_carries(ctx, new, &drop)
    }
}

/// The sinks with every vector of scaled reads made a scaled run and every dead zeroing of a
/// carried vector taken out; as given when there is none.
pub(crate) fn simplified(ctx: &mut Ctx, sinks: &[u32]) -> Outcome<Vec<u32>> {
    let graph = Graph::build(ctx, sinks)?;
    let vectors = graph
        .order
        .iter()
        .any(|&n| kind_of(ctx, n).ok().flatten().as_deref() == Some("vec"));
    if !(vectors || !graph.loops.is_empty()) {
        return Ok(sinks.to_vec());
    }
    let (none, carried, sunk) = (Planes::default(), HashSet::new(), BTreeMap::new());
    let on_loop = |ctx: &mut Ctx, old: u32, new: u32| dead_take(ctx, &graph, old, new);
    let rules = Rules {
        planes: &none,
        carried: &carried,
        sunk: &sunk,
        each: scaled_gather,
        on_loop: Some(&on_loop),
    };
    rewritten(ctx, sinks, &rules)
}
