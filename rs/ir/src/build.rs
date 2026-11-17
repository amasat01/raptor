//! Building nodes from nodes: the loop constructor (its boundary and the gating of reads under a
//! dispatch arm), a node over other operands, and substitution. Owns the C doors `rx_new_loop` and
//! `rx_substitute`.
use crate::door::{guard, slice, Node};
use crate::loops::give;
use crate::record::{refuse, Ctx, Fault};
use crate::value::{ttype_val, Val};
use rcore::keytext::py_str;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_int};

/// A result that refuses with a fault.
type Fallible<T> = Result<T, Fault>;

/// The parts of a loop as a producer gives them; the loop's free operands are computed, not given.
pub struct LoopParts<'a> {
    /// The loop's index node.
    pub index: Node,
    /// The loop-carried placeholder nodes.
    pub carries: &'a [Node],
    /// The initial value of each carried placeholder.
    pub inits: &'a [Node],
    /// The next value of each carried placeholder.
    pub nexts: &'a [Node],
    /// The index range: start, stop and step.
    pub range: (i64, i64, i64),
    /// The sinks the body commits each trip.
    pub body_sinks: &'a [Node],
    /// The break condition, if the loop has one.
    pub exit_cond: Option<Node>,
    /// The values the loop yields at a break.
    pub exit_values: &'a [Node],
    /// The trip count node, when the count is not static.
    pub count: Option<Node>,
    /// Whether the count is tested at the tail.
    pub count_tail: bool,
    /// The word naming a dynamic bound, if any.
    pub dynamic: Option<&'a str>,
}

/// The node references as one tuple value.
fn nodes_val(nodes: &[Node]) -> Val {
    Val::Tuple(nodes.iter().map(|&n| Val::Node(n)).collect())
}

/// A node reference, or None.
fn opt_node_val(node: Option<Node>) -> Val {
    node.map_or(Val::None, Val::Node)
}

impl Ctx {
    /// Whether `node` is a loop index or carry placeholder.
    fn is_placeholder(&self, node: Node) -> bool {
        self.is_a(node, "LoopIndex") || self.is_a(node, "LoopCarry")
    }

    /// Whether the subtree of `node` reads this loop's index or one of its carries (`own`).
    fn depends(&self, node: Node, own: &HashSet<Node>, memo: &mut HashMap<Node, bool>) -> bool {
        if let Some(&known) = memo.get(&node) {
            return known;
        }
        let mut stack = vec![(node, false)];
        while let Some((current, expanded)) = stack.pop() {
            if memo.contains_key(&current) && !expanded {
                continue;
            }
            if self.is_placeholder(current) {
                memo.insert(current, own.contains(&current));
                continue;
            }
            let children = &self.nodes[current as usize].ops;
            if !expanded {
                stack.push((current, true));
                stack.extend(
                    children
                        .iter()
                        .filter(|c| !memo.contains_key(c))
                        .map(|&c| (c, false)),
                );
                continue;
            }
            let reads = children.iter().any(|c| *memo.get(c).unwrap_or(&false));
            memo.insert(current, reads);
        }
        memo[&node]
    }

    /// Whether the kind of `node` states the feather `name`.
    fn states_feather(&self, node: Node, name: &str) -> bool {
        matches!(self.field(node, "feathers"),
            Ok(Val::Tuple(names)) if names.iter().any(|n| matches!(n, Val::Str(s) if s == name)))
    }

    /// An index built from constants and static-index terminals only (a compile-time position).
    fn is_static_index(&self, index: Node) -> bool {
        let mut stack = vec![index];
        while let Some(current) = stack.pop() {
            if self.is_a(current, "Const") {
                continue;
            }
            let children = &self.nodes[current as usize].ops;
            if children.is_empty() && !self.states_feather(current, "static_index") {
                return false;
            }
            stack.extend(children.iter().copied());
        }
        true
    }

    /// Whether the subtree of `node` holds an `At` at a run-time index.
    fn gathers(&self, node: Node, memo: &mut HashMap<Node, bool>) -> bool {
        if let Some(&known) = memo.get(&node) {
            return known;
        }
        let mut stack = vec![(node, false)];
        while let Some((current, expanded)) = stack.pop() {
            if !expanded {
                if memo.contains_key(&current) {
                    continue;
                }
                if self.is_a(current, "At") {
                    let index = self.nodes[current as usize].ops[1];
                    memo.insert(current, !self.is_static_index(index));
                    continue;
                }
                stack.push((current, true));
                stack.extend(self.nodes[current as usize].ops.iter().map(|&c| (c, false)));
                continue;
            }
            let found = self.nodes[current as usize]
                .ops
                .iter()
                .any(|c| *memo.get(c).unwrap_or(&false));
            memo.insert(current, found);
        }
        memo[&node]
    }

    /// The loop-invariant boundary values of the roots, maximal and in deterministic order: the
    /// walk stops at the highest invariant node on each path. A plane read under a dispatch or
    /// select arm that gathers stays in the body unless it was gated.
    pub fn loop_boundary(
        &self,
        roots: &[Node],
        own: &HashSet<Node>,
        gates: &HashSet<Node>,
    ) -> Vec<Node> {
        let (mut memo, mut gather_memo) = (HashMap::new(), HashMap::new());
        let mut seen: HashMap<Node, bool> = HashMap::new();
        let mut found = vec![];
        let mut stack: Vec<(Node, bool)> = roots.iter().rev().map(|&n| (n, false)).collect();
        while let Some((node, guarded)) = stack.pop() {
            if let Some(&before) = seen.get(&node) {
                if guarded || !before {
                    continue;
                }
            }
            seen.insert(node, guarded);
            let ungated = guarded && !gates.contains(&node) && self.gathers(node, &mut gather_memo);
            if !self.depends(node, own, &mut memo) && !ungated {
                found.push(node);
                continue;
            }
            let children = self.nodes[node as usize].ops.clone();
            let branching = self.is_a(node, "Dispatch") || self.is_a(node, "Select");
            let arms: HashSet<Node> = if branching {
                children.iter().skip(1).copied().collect()
            } else {
                HashSet::new()
            };
            stack.extend(
                children
                    .iter()
                    .rev()
                    .map(|&c| (c, guarded || arms.contains(&c))),
            );
        }
        let mut held = HashSet::new();
        found.into_iter().filter(|n| held.insert(*n)).collect()
    }

    /// The invariant rank-0 gathers under one dispatch arm, in first-found order.
    fn arm_gathers(
        &self,
        arm: Node,
        own: &HashSet<Node>,
        memo: &mut HashMap<Node, bool>,
        gather_memo: &mut HashMap<Node, bool>,
    ) -> Vec<Node> {
        let (mut found, mut visited, mut walk) = (vec![], HashSet::new(), vec![arm]);
        while let Some(node) = walk.pop() {
            if !visited.insert(node) || self.is_a(node, "Loop") {
                continue;
            }
            let children = &self.nodes[node as usize].ops;
            if self.depends(node, own, memo) {
                let branching = self.is_a(node, "Dispatch") || self.is_a(node, "Select");
                if branching {
                    walk.push(children[0])
                } else {
                    walk.extend(children.iter().copied())
                }
            } else if self.gathers(node, gather_memo) && self.nodes[node as usize].shape.is_empty()
            {
                if !found.contains(&node) {
                    found.push(node);
                }
            } else {
                walk.extend(children.iter().copied());
            }
        }
        found
    }

    /// A constant node of the type of `like`.
    fn constant_like(&mut self, like: Node, literal: Val) -> Fallible<Node> {
        let ci = self
            .table
            .find("Const")
            .ok_or_else(|| Fault::Refused("no Const class".into()))?;
        let n = &self.nodes[like as usize];
        let tt = ttype_val(&n.dtype, &n.shape, n.tag.as_deref());
        self.new_from(ci, &[("literal".into(), literal), ("ttype".into(), tt)])
    }

    /// One dispatch arm with each invariant gather in it replaced by a select that reads it only
    /// when the arm is taken; the nodes made are added to `gates`.
    fn gate_arm(
        &mut self,
        selector: Node,
        arm: Node,
        position: (usize, usize),
        found: &[Node],
        gates: &mut HashSet<Node>,
    ) -> Fallible<Node> {
        let (index, last) = position;
        let kind = if index == 0 {
            "le"
        } else if index == last {
            "ge"
        } else {
            "eq"
        };
        let number = self.constant_like(selector, Val::Int(index as i128))?;
        let taken = self.make_op(kind, &[selector, number], &Val::None)?;
        let select = self
            .table
            .find("Select")
            .ok_or_else(|| Fault::Refused("no Select class".into()))?;
        let mut swaps = HashMap::new();
        for &read in found {
            let n = &self.nodes[read as usize];
            let (whole, tt) = (
                matches!(n.dtype.as_str(), "i32" | "int32"),
                ttype_val(&n.dtype, &n.shape, n.tag.as_deref()),
            );
            let zero =
                self.constant_like(read, if whole { Val::Int(0) } else { Val::Float(0.0) })?;
            let given = [
                ("cond", Val::Node(taken)),
                ("on_true", Val::Node(read)),
                ("on_false", Val::Node(zero)),
                ("ttype", tt),
            ];
            let gate = self.new_from(select, &given.map(|(k, v)| (k.to_string(), v)))?;
            gates.insert(gate);
            swaps.insert(read, gate);
        }
        Ok(self.substitute(&[arm], &swaps)?[0])
    }

    /// The nexts with each invariant gather of a dispatch arm gated by the arm's own condition,
    /// and the gates made.
    pub fn loop_gated_reads(
        &mut self,
        nexts: &[Node],
        own: &HashSet<Node>,
    ) -> Fallible<(Vec<Node>, HashSet<Node>)> {
        let (mut memo, mut gather_memo) = (HashMap::new(), HashMap::new());
        let (mut swaps, mut gates, mut seen) = (HashMap::new(), HashSet::new(), HashSet::new());
        let mut stack = nexts.to_vec();
        while let Some(node) = stack.pop() {
            if seen.contains(&node) || !self.depends(node, own, &mut memo) {
                continue;
            }
            seen.insert(node);
            let children = self.nodes[node as usize].ops.clone();
            if !self.is_a(node, "Dispatch") || self.depends(children[0], own, &mut memo) {
                stack.extend(children);
                continue;
            }
            stack.push(children[0]);
            let mut arms = children[1..].to_vec();
            for i in 0..arms.len() {
                let position = (i, arms.len() - 1);
                let found = self.arm_gathers(arms[i], own, &mut memo, &mut gather_memo);
                if !found.is_empty() {
                    arms[i] = self.gate_arm(children[0], arms[i], position, &found, &mut gates)?;
                }
            }
            if arms != children[1..] {
                let over: Vec<Node> = std::iter::once(children[0]).chain(arms).collect();
                swaps.insert(node, self.rebuild(node, &over)?);
            }
        }
        let out = if swaps.is_empty() {
            nexts.to_vec()
        } else {
            self.substitute(nexts, &swaps)?
        };
        Ok((out, gates))
    }

    /// A copy of the node over other operands, every other field kept; a loop is minted again by
    /// the loop constructor over its new operands.
    pub fn rebuild(&mut self, node: Node, operands: &[Node]) -> Fallible<Node> {
        if self.is_a(node, "Loop") {
            return self.rebuild_loop(node, operands);
        }
        let record = &self.nodes[node as usize];
        let class = self.table.classes[record.class].name.clone();
        let slots = &self.table.classes[record.class].slots;
        let fields: Vec<(String, Val)> = slots
            .iter()
            .cloned()
            .zip(record.fields.iter().cloned())
            .filter(|(name, _)| !name.starts_with('_'))
            .collect();
        let text = crate::value::text_val(&Val::Tuple(
            fields
                .into_iter()
                .map(|(k, v)| Val::Tuple(vec![Val::Str(k), v]))
                .collect(),
        ));
        let (dtype, shape, tag) = (
            record.dtype.clone(),
            record.shape.clone(),
            record.tag.clone(),
        );
        self.make(&class, &dtype, &shape, tag.as_deref(), operands, &text)
    }

    /// Rebuilds the loop `lp` over new operands, keeping its fields; the body digest cache entry is dropped.
    fn rebuild_loop(&mut self, lp: Node, operands: &[Node]) -> Fallible<Node> {
        if !self.refs(lp, "body_sinks")?.is_empty() {
            return refuse(
                "a loop that commits once per iteration (a derived loop) \
                 cannot be rewritten by substitution",
            );
        }
        let mut swapped: HashMap<Node, Node> = self.nodes[lp as usize]
            .ops
            .iter()
            .zip(operands)
            .filter(|(old, new)| old != new)
            .map(|(&o, &n)| (o, n))
            .collect();
        // the rebuilt loop owns an index and carries of its own: two loops never share a placeholder, so a template
        // instantiated twice yields loops that each read only their own
        let index = self.rebuild(self.refs(lp, "index")?[0], &[])?;
        swapped.insert(self.refs(lp, "index")?[0], index);
        let mut carries = vec![];
        for old in self.refs(lp, "carries")? {
            let fresh = self.rebuild(old, &[])?;
            swapped.insert(old, fresh);
            carries.push(fresh);
        }
        let roots = self.body_roots(lp)?;
        let done = self.substitute(&roots, &swapped)?;
        let carry_count = self.refs(lp, "carries")?.len();
        let has_exit = self.field(lp, "exit_cond")? != Val::None;
        let int = |f: &str, me: &Ctx| match me.field(lp, f) {
            Ok(Val::Int(i)) => Ok(i as i64),
            _ => refuse("a loop bound is an int"),
        };
        let count = if self.field(lp, "count")? != Val::None {
            operands.last().copied()
        } else {
            None
        };
        let dynamic = match self.field(lp, "dynamic")? {
            Val::Str(s) => Some(s),
            _ => None,
        };
        let parts = LoopParts {
            index,
            carries: &carries,
            inits: &operands[..carry_count],
            nexts: &done[..carry_count],
            range: (int("start", self)?, int("stop", self)?, int("step", self)?),
            body_sinks: &[],
            exit_cond: if has_exit {
                Some(done[carry_count])
            } else {
                None
            },
            exit_values: if has_exit {
                &done[carry_count + 1..]
            } else {
                &[]
            },
            count,
            count_tail: self.field(lp, "count_tail")? == Val::Bool(true),
            dynamic: dynamic.as_deref(),
        };
        self.build_loop(&parts)
    }

    /// `roots` with every node in `mapping` replaced; an unchanged subtree is reused, not copied.
    pub fn substitute(
        &mut self,
        roots: &[Node],
        mapping: &HashMap<Node, Node>,
    ) -> Fallible<Vec<Node>> {
        let mut memo = mapping.clone();
        let (mut order, mut seen) = (vec![], HashSet::new());
        let mut stack: Vec<(Node, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
        while let Some((node, expanded)) = stack.pop() {
            if seen.contains(&node) || mapping.contains_key(&node) {
                continue;
            }
            if !expanded {
                stack.push((node, true));
                for &child in self.node(node)?.ops.iter().rev() {
                    if !seen.contains(&child) && !mapping.contains_key(&child) {
                        stack.push((child, false));
                    }
                }
                continue;
            }
            seen.insert(node);
            order.push(node);
        }
        for node in order {
            let old = self.nodes[node as usize].ops.clone();
            let new: Vec<Node> = old.iter().map(|c| *memo.get(c).unwrap_or(c)).collect();
            let made = if new == old {
                node
            } else {
                self.rebuild(node, &new)?
            };
            memo.insert(node, made);
        }
        Ok(roots.iter().map(|r| *memo.get(r).unwrap_or(r)).collect())
    }

    /// Mints a loop and computes its boundary: the operands are the carried initial values, then
    /// the maximal loop-invariant subexpressions of the body, then the trip count.
    pub fn build_loop(&mut self, p: &LoopParts) -> Fallible<Node> {
        for &n in [p.carries, p.inits, p.nexts, p.body_sinks, p.exit_values]
            .concat()
            .iter()
            .chain(&[p.index])
            .chain(p.exit_cond.iter())
            .chain(p.count.iter())
        {
            self.node(n)?;
        }
        let own: HashSet<Node> = p.carries.iter().copied().chain([p.index]).collect();
        let (nexts, gates) = self.loop_gated_reads(p.nexts, &own)?;
        let mut roots = nexts.clone();
        for &sink in p.body_sinks {
            roots.extend(self.node(sink)?.ops.iter().copied());
        }
        if let Some(cond) = p.exit_cond {
            roots.push(cond);
            roots.extend(p.exit_values);
        }
        let held: HashSet<Node> = p.inits.iter().copied().chain(p.count).collect();
        let free: Vec<Node> = self
            .loop_boundary(&roots, &own, &gates)
            .into_iter()
            .filter(|n| !held.contains(n))
            .collect();
        let class = self
            .table
            .find("Loop")
            .ok_or_else(|| Fault::Refused("no Loop class".into()))?;
        let (start, stop, step) = p.range;
        let given: Vec<(String, Val)> = [
            ("index", Val::Node(p.index)),
            ("carries", nodes_val(p.carries)),
            ("inits", nodes_val(p.inits)),
            ("nexts", nodes_val(&nexts)),
            ("start", Val::Int(start as i128)),
            ("stop", Val::Int(stop as i128)),
            ("step", Val::Int(step as i128)),
            ("free", nodes_val(&free)),
            ("body_sinks", nodes_val(p.body_sinks)),
            ("exit_cond", opt_node_val(p.exit_cond)),
            ("exit_values", nodes_val(p.exit_values)),
            ("count", opt_node_val(p.count)),
            ("count_tail", Val::Bool(p.count_tail)),
            (
                "dynamic",
                p.dynamic.map_or(Val::None, |d| Val::Str(d.into())),
            ),
        ]
        .map(|(k, v)| (k.to_string(), v))
        .into();
        let made = self.new_from(class, &given)?;
        if let Err(fault) = self.check_placeholders(made) {
            self.nodes.pop();
            self.body_cache.borrow_mut().remove(&made);
            return Err(fault);
        }
        Ok(made)
    }

    /// Every index or carry placeholder inside the body must be this loop's (an enclosing
    /// loop's may cross as an operand).
    fn check_placeholders(&self, lp: Node) -> Fallible<()> {
        let index = self.refs(lp, "index")?[0];
        let own: HashSet<Node> = self
            .refs(lp, "carries")?
            .into_iter()
            .chain([index])
            .collect();
        let depth = |n: Node| match self.field(n, "depth") {
            Ok(Val::Int(d)) => Ok(d),
            _ => refuse("a placeholder has a depth"),
        };
        let mut crossing = HashSet::new();
        for &o in &self.nodes[lp as usize].ops {
            if self.is_placeholder(o) && depth(o)? < depth(index)? {
                crossing.insert(o);
            }
        }
        let name = |n: Node| {
            self.field(n, "name")
                .map_or("'?'".to_string(), |v| match v {
                    Val::Str(s) => py_str(&s),
                    _ => "'?'".into(),
                })
        };
        let foreign =
            |n: Node| self.is_placeholder(n) && !own.contains(&n) && !crossing.contains(&n);
        for node in self.body_nodes(lp)? {
            for &child in &self.nodes[node as usize].ops {
                if foreign(child) {
                    return refuse(format!(
                        "the loop over {} reads {} {} (depth {}), which belongs to a DIFFERENT \
                         loop. A loop body may only read its own index and its own carried slots; \
                         anything else must cross the boundary as an operand",
                        name(index),
                        self.kind_of(child)?,
                        name(child),
                        depth(child)?
                    ));
                }
            }
        }
        if self.body_roots(lp)?.into_iter().any(foreign) {
            return refuse(format!(
                "the loop over {} returns a placeholder of a different loop",
                name(index)
            ));
        }
        Ok(())
    }
}

/// Builds a Loop from its parts and computes its boundary; `exit_cond` and `count` are NULL for
/// none, `dynamic` is NULL for none. Carries, inits and nexts are `ncarries` long each.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_new_loop(
    ctx: *mut Ctx,
    index: Node,
    carries: *const Node,
    ncarries: usize,
    inits: *const Node,
    nexts: *const Node,
    start: i64,
    stop: i64,
    step: i64,
    body_sinks: *const Node,
    nsinks: usize,
    exit_cond: *const Node,
    exit_values: *const Node,
    nexit: usize,
    count: *const Node,
    count_tail: c_int,
    dynamic: *const c_char,
    out: *mut Node,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        let dynamic = if dynamic.is_null() {
            None
        } else {
            Some(crate::door::cstr(dynamic, "dynamic")?)
        };
        let optional = |p: *const Node| {
            if p.is_null() {
                None
            } else {
                Some(unsafe { *p })
            }
        };
        let parts = LoopParts {
            index,
            carries: slice(carries, ncarries, "carries")?,
            inits: slice(inits, ncarries, "inits")?,
            nexts: slice(nexts, ncarries, "nexts")?,
            range: (start, stop, step),
            body_sinks: slice(body_sinks, nsinks, "body_sinks")?,
            exit_cond: optional(exit_cond),
            exit_values: slice(exit_values, nexit, "exit_values")?,
            count: optional(count),
            count_tail: count_tail != 0,
            dynamic,
        };
        unsafe { *out = c.build_loop(&parts)? };
        Ok(())
    })
}

/// The roots with every node of the mapping (`from[i]` becomes `to[i]`, by handle) replaced.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_substitute(
    ctx: *mut Ctx,
    roots: *const Node,
    nroots: usize,
    from: *const Node,
    to: *const Node,
    nmap: usize,
    buf: *mut Node,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let mapping: HashMap<Node, Node> = slice(from, nmap, "from")?
            .iter()
            .copied()
            .zip(slice(to, nmap, "to")?.iter().copied())
            .collect();
        let done = c.substitute(slice(roots, nroots, "roots")?, &mapping)?;
        give(buf, cap, n, &done)
    })
}

/// The node over other operands, every other field kept (a loop is minted again by the loop
/// constructor).
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_rebuild(
    ctx: *mut Ctx,
    node: Node,
    operands: *const Node,
    nops: usize,
    out: *mut Node,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        c.node(node)?;
        unsafe { *out = c.rebuild(node, slice(operands, nops, "operands")?)? };
        Ok(())
    })
}
