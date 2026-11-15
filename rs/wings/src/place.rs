//! The wide placement rewrite: wide values that need a plane are wrapped in a `resident` op over
//! the value and a leaf naming the plane, carried slots of loops that can live in a plane do, and a
//! vector of scaled reads is first made a scaled run. The new nodes are made in the caller's
//! context;
//! nothing is kept between calls. The graph is in this file, the decision in `plan.rs` and
//! `slots.rs`, the planes in `planes.rs` and the rewrite in `rewrite.rs`.
use crate::construct::refs;
use crate::plan::Plan;
use crate::planes::assign_planes;
use crate::rewrite::{rewritten, simplified, Rules};
use crate::{is_a, list_out, name_of, RESIDENT_KIND};
use raptor_ir::door::guard;
use raptor_ir::record::{Ctx, Fault};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_int};

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// The operations that act on each element alone.
pub(crate) const ELEMENTWISE: [&str; 9] = [
    "add", "sub", "mul", "div", "neg", "min", "max", "abs", "select",
];

/// What the pass reads of the target: the widest vector kept in registers when accessed at a
/// run-time index, by access pattern; the widest one with only compile-time indices that stays
/// there; and the fold threshold with the build's form for deciding which sites are rolled.
pub struct Limits {
    /// `read`, `slot_read`, `rotate`, `write` and `accum`, in that order.
    pub runtime_min: [i64; 5],
    /// The widest vector with only compile-time indices that stays in registers.
    pub wide_min: i64,
    /// The fold threshold, when the facts name one.
    pub rolled_min: Option<i64>,
    /// The build's form for deciding which sites are rolled.
    pub form: String,
}

impl Limits {
    /// The widest vector kept in registers when accessed at a run-time index in the pattern.
    pub fn runtime(&self, pattern: &str) -> i64 {
        let at = ["read", "slot_read", "rotate", "write", "accum"]
            .iter()
            .position(|p| *p == pattern);
        self.runtime_min[at.unwrap_or(0)]
    }
}

/// How a user reads a node: as an operand, as the initial value or the end-of-iteration value of
/// a carried slot, as a loop's boundary value, or as a root of its body.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Read as an operand.
    Operand,
    /// Read as the initial value of a carried slot.
    Init,
    /// Read as a loop's boundary value.
    Boundary,
    /// Read as the end-of-iteration value of a carried slot.
    Next,
    /// Read as a root of a loop's body.
    Body,
}

/// One use of a node: `user` reads it as operand (or slot) `k`.
#[derive(Clone, Copy)]
pub(crate) struct Use {
    /// The node that reads.
    pub user: u32,
    /// The operand or slot number it reads at.
    pub k: usize,
    /// How it reads.
    pub kind: Kind,
}

/// Every node of the kernel in post-order with its scope (none for the top, else the loop whose
/// body holds it) and every use of it, the bodies of loops included.
pub(crate) struct Graph {
    /// The nodes in post-order.
    pub order: Vec<u32>,
    /// The uses of each node.
    pub users: HashMap<u32, Vec<Use>>,
    /// The loop whose body holds each node, none at the top.
    pub scope: HashMap<u32, Option<u32>>,
    /// The loops of the kernel.
    pub loops: Vec<u32>,
}

impl Graph {
    /// The graph under the sinks.
    pub fn build(ctx: &Ctx, sinks: &[u32]) -> Outcome<Graph> {
        let mut graph = Graph {
            order: vec![],
            users: HashMap::new(),
            scope: HashMap::new(),
            loops: vec![],
        };
        let mut seen = HashSet::new();
        let mut stack: Vec<(u32, bool)> = sinks.iter().rev().map(|&sink| (sink, false)).collect();
        while let Some((node, done)) = stack.pop() {
            if seen.contains(&node) {
                continue;
            }
            let operands = ctx.node(node)?.ops.clone();
            if !done {
                stack.push((node, true));
                stack.extend(
                    operands
                        .iter()
                        .rev()
                        .filter(|c| !seen.contains(*c))
                        .map(|&c| (c, false)),
                );
                continue;
            }
            seen.insert(node);
            graph.scope.insert(node, None);
            graph.order.push(node);
            let is_loop = is_a(ctx, node, "Loop");
            let carried = if is_loop {
                crate::slots::carry_count(ctx, node)?
            } else {
                0
            };
            for (k, &child) in operands.iter().enumerate() {
                let kind = match (is_loop, k < carried) {
                    (true, true) => Kind::Init,
                    (true, false) => Kind::Boundary,
                    _ => Kind::Operand,
                };
                graph.users.entry(child).or_default().push(Use {
                    user: node,
                    k,
                    kind,
                });
            }
            if is_loop {
                graph.loops.push(node);
                graph.body(ctx, node, &mut seen)?;
            }
        }
        Ok(graph)
    }

    /// The nodes of a loop's body, then the uses the loop makes of its carried and body values.
    fn body(&mut self, ctx: &Ctx, lp: u32, seen: &mut HashSet<u32>) -> Outcome<()> {
        for node in ctx.body_nodes(lp)? {
            if !seen.insert(node) {
                continue;
            }
            self.scope.insert(node, Some(lp));
            self.order.push(node);
            for (k, &child) in ctx.nodes[node as usize].ops.iter().enumerate() {
                self.users.entry(child).or_default().push(Use {
                    user: node,
                    k,
                    kind: Kind::Operand,
                });
            }
            if is_a(ctx, node, "Loop") {
                self.loops.push(node);
                self.body(ctx, node, seen)?;
            }
        }
        for (j, next) in refs(ctx, lp, "nexts")?.into_iter().enumerate() {
            self.users.entry(next).or_default().push(Use {
                user: lp,
                k: j,
                kind: Kind::Next,
            });
        }
        let carried = crate::slots::carry_count(ctx, lp)?;
        for root in ctx.body_roots(lp)?.into_iter().skip(carried) {
            self.users.entry(root).or_default().push(Use {
                user: lp,
                k: usize::MAX,
                kind: Kind::Body,
            });
        }
        Ok(())
    }

    /// The uses of a node, in the order the users stand in the walk.
    pub fn users(&self, node: u32) -> &[Use] {
        self.users.get(&node).map_or(&[][..], |uses| &uses[..])
    }
}

/// The kind of an op, `select` for a select, else none.
pub(crate) fn kind_of(ctx: &Ctx, node: u32) -> Outcome<Option<String>> {
    if is_a(ctx, node, "Select") {
        return Ok(Some("select".into()));
    }
    if is_a(ctx, node, "Op") {
        ctx.str_field(node, "kind").map(Some)
    } else {
        Ok(None)
    }
}

/// The elements of a rank-1 value, 0 for any other.
pub(crate) fn width(ctx: &Ctx, node: u32) -> i64 {
    let shape = &ctx.nodes[node as usize].shape;
    if shape.len() == 1 {
        shape[0]
    } else {
        0
    }
}

/// Whether the node places a value in a plane.
pub(crate) fn is_a_resident(ctx: &Ctx, node: u32) -> Outcome<bool> {
    Ok(kind_of(ctx, node)?.as_deref() == Some(RESIDENT_KIND) && is_a(ctx, node, "Op"))
}

/// The sinks with each wide value that needs a plane placed in one; as given when the kernel is
/// left alone (the thread form, no sinks, nothing that needs a plane).
fn placed_sinks(
    ctx: &mut Ctx,
    sinks: &[u32],
    thread_form: bool,
    limits: &Limits,
    taken: &[String],
) -> Outcome<Vec<u32>> {
    if thread_form || sinks.is_empty() || !sinks.iter().all(|&s| is_a(ctx, s, "Sink")) {
        return Ok(sinks.to_vec());
    }
    let simple = simplified(ctx, sinks)?;
    let graph = Graph::build(ctx, &simple)?;
    let plan = Plan::decide(ctx, &graph, limits)?;
    if plan.resident.is_empty() && plan.slots.is_empty() {
        return Ok(sinks.to_vec());
    }
    let planes = assign_planes(ctx, &graph, &plan, taken)?;
    let carried: HashSet<_> = plan.carried.keys().copied().collect();
    let sunk = plan.sunk.clone();
    let rules = Rules {
        planes: &planes,
        carried: &carried,
        sunk: &sunk,
        each: |_, node| Ok(node),
        on_loop: None,
    };
    match rewritten(ctx, &simple, &rules) {
        Err(Fault::Refused(_)) => Ok(sinks.to_vec()),
        other => other,
    }
}

/// The sinks with each wide value that needs a plane placed in one; see `raptor_wings.h`.
/// `*changed` is 0 when the sinks are returned as given.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_place(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    thread_form: c_int,
    runtime_min: *const i64,
    wide_min: i64,
    rolled_min: *const i64,
    form: *const c_char,
    taken: *const c_char,
    buf: *mut u32,
    cap: usize,
    count: *mut usize,
    changed: *mut c_int,
) -> c_int {
    guard(ctx, |ctx| {
        if (nsinks > 0 && sinks.is_null()) || changed.is_null() || runtime_min.is_null() {
            return Err(Fault::BadArg(
                "sinks, runtime_min or changed is NULL".into(),
            ));
        }
        if cap < nsinks || (buf.is_null() && nsinks > 0) {
            // nothing is made until the caller's buffer can hold the answer
            return list_out(std::ptr::null_mut(), 0, count, &vec![0; nsinks]);
        }
        let given = if nsinks == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        let table = unsafe { std::slice::from_raw_parts(runtime_min, 5) };
        let rolled_min = if rolled_min.is_null() {
            None
        } else {
            Some(unsafe { *rolled_min })
        };
        let limits = Limits {
            runtime_min: [table[0], table[1], table[2], table[3], table[4]],
            wide_min,
            rolled_min,
            form: name_of(form, "form")?.into(),
        };
        let taken: Vec<String> = name_of(taken, "taken")?.lines().map(String::from).collect();
        let made = placed_sinks(ctx, given, thread_form != 0, &limits, &taken)?;
        unsafe { *changed = (made != given) as c_int }
        list_out(buf, cap, count, &made)
    })
}
