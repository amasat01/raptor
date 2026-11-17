//! The rules about loops that the differentiation asks of the core: the terms of an `add` chain, the nested loops a
//! derivative cannot carry, and whether a loop is an accumulator or a recurrence. Owns the C doors `rx_add_terms`,
//! `rx_refuse_nesting`, `rx_loop_is_accumulator` and `rx_loop_addends`.
use crate::door::guard;
use crate::loops::give;
use crate::record::{refuse, Ctx, Fault};
use crate::value::Val;
use rcore::keytext::py_str;
use std::collections::HashSet;
use std::ffi::c_int;

/// The kind of op the terms of a sum are joined by.
const ADD: &str = "add";

/// A result that refuses with a fault.
type Fallible<T> = Result<T, Fault>;

impl Ctx {
    /// The text a node's `name` field holds, written as the source writes a string; `'?'` for a node without one.
    fn name_repr(&self, h: u32) -> String {
        match self.field(h, "name") {
            Ok(Val::Str(s)) => py_str(&s),
            _ => "'?'".into(),
        }
    }

    /// The written name of the index of loop `lp`.
    fn index_name(&self, lp: u32) -> Fallible<String> {
        let index = self.node_field(lp, "index")?;
        Ok(index.map_or_else(|| "'?'".into(), |i| self.name_repr(i)))
    }

    /// Whether the kind field of `h` is `kind` (a node that states no kind field is of no op kind).
    fn has_op_kind(&self, h: u32, kind: &str) -> bool {
        matches!(self.field(h, "kind"), Ok(Val::Str(k)) if k == kind)
    }

    /// The terms of an `add` chain, left to right; any other node is its own single term. A `sub` at the top
    /// alternates the sign of what follows, so only `add` is flattened.
    pub fn add_terms(&self, node: u32) -> Fallible<Vec<u32>> {
        self.node(node)?;
        let (mut terms, mut stack) = (vec![], vec![node]);
        while let Some(n) = stack.pop() {
            if self.is_a(n, "Op") && self.has_op_kind(n, ADD) && self.node(n)?.ops.len() == 2 {
                stack.extend(self.node(n)?.ops.iter().rev());
            } else {
                terms.push(n);
            }
        }
        Ok(terms)
    }

    /// Whether `node` reads one of `carries`, through its operands or the `nexts` of a loop it holds.
    fn touches(&self, node: u32, carries: &HashSet<u32>) -> Fallible<bool> {
        let (mut stack, mut seen) = (vec![node], HashSet::new());
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if carries.contains(&n) {
                return Ok(true);
            }
            stack.extend(self.node(n)?.ops.iter());
            if self.is_a(n, "Loop") {
                stack.extend(self.refs(n, "nexts")?);
            }
        }
        Ok(false)
    }

    /// Refuses, by name, a body whose nested `for` neither direction can carry: one that leaves early, one that
    /// commits a sink per iteration or runs a recorded trip count (a derived loop).
    pub fn refuse_nesting(&self, lp: u32) -> Fallible<()> {
        let outer = self.index_name(lp)?;
        for inner in self.body_nodes(lp)? {
            if !self.is_a(inner, "Loop") {
                continue;
            }
            let name = self.index_name(inner)?;
            if self.node_field(inner, "exit_cond")?.is_some() {
                return refuse(format!("loop over {outer} cannot be differentiated: its body contains a nested `for` over {name} with a `break`. A nested loop differentiates, but one that leaves early would need its per-sample trip count threaded through every outer iteration, which is not built. Move the `break` to the outer loop, or run the inner loop to its bound and mask"));
            }
            if !self.refs(inner, "body_sinks")?.is_empty()
                || self.node_field(inner, "count")?.is_some()
            {
                return refuse(format!("loop over {outer} cannot be differentiated: its body contains a nested loop over {name} that commits a sink or runs a recorded trip count (a derived loop); differentiating a derivative's nested loop is not built"));
            }
        }
        Ok(())
    }

    /// Refuses a loop whose body cannot be differentiated at all: a nested loop (`refuse_nesting`), a raw device
    /// block (opaque to the IR) or a sink committed per iteration (a primal body may not commit one).
    fn refuse_undifferentiable(&self, lp: u32) -> Fallible<()> {
        self.refuse_nesting(lp)?;
        let outer = self.index_name(lp)?;
        let raw = crate::consts::names("RAW_KIND").unwrap_or_default();
        if self
            .body_nodes(lp)?
            .iter()
            .any(|&n| raw.iter().any(|k| self.has_op_kind(n, k)))
        {
            return refuse(format!("loop over {outer} cannot be differentiated: its body splices a @raw_device block, whose text is opaque to the IR, so no rule can exist for it"));
        }
        if let Some(&sink) = self.refs(lp, "body_sinks")?.first() {
            let role = match self.field(sink, "role") {
                Ok(Val::Str(r)) => py_str(&r),
                _ => "'?'".into(),
            };
            return refuse(format!("loop over {outer} cannot be differentiated: its body commits the sink {} (role {role}) once per iteration. A body sink is what a DERIVED loop carries (the reverse of a gather is a scatter); a PRIMAL body may not commit one, so differentiating this loop would be differentiating a derivative", self.name_repr(sink)));
        }
        Ok(())
    }

    /// For each carry of an accumulator loop, the terms of its body result besides the carry; `None` when the loop
    /// is a recurrence. An accumulator has every carry's body result as an `add` chain holding that carry exactly
    /// once as a bare term, read nowhere else (not in another term, not by another node of the body).
    fn accumulator_terms(&self, lp: u32) -> Fallible<Option<Vec<Vec<u32>>>> {
        self.refuse_undifferentiable(lp)?;
        let carries = self.refs(lp, "carries")?;
        let carry_set: HashSet<u32> = carries.iter().copied().collect();
        let mut per = vec![];
        for (&carry, next) in carries.iter().zip(self.refs(lp, "nexts")?) {
            let terms = self.add_terms(next)?;
            let rest: Vec<u32> = terms.iter().copied().filter(|&t| t != carry).collect();
            if terms.len() - rest.len() != 1 || rest.is_empty() {
                return Ok(None);
            }
            for &t in &rest {
                if self.touches(t, &carry_set)? {
                    return Ok(None);
                }
            }
            per.push(rest);
        }
        let mut reads = 0;
        for n in self.body_nodes(lp)? {
            reads += self
                .node(n)?
                .ops
                .iter()
                .filter(|o| carry_set.contains(o))
                .count();
        }
        Ok((reads == carries.len()).then_some(per))
    }

    /// Whether loop `lp` is an accumulator (its adjoint stores nothing) rather than a recurrence (its adjoint runs
    /// in reverse over the forward loop's own carried values); refuses a loop neither direction can carry.
    pub fn is_accumulator(&self, lp: u32) -> Fallible<bool> {
        Ok(self.accumulator_terms(lp)?.is_some())
    }

    /// The addend of each carry of an accumulator loop (`c = c + a`), the carry's other terms summed in order;
    /// refuses a recurrence, which has none.
    pub fn loop_addends(&mut self, lp: u32) -> Fallible<Vec<u32>> {
        let Some(per) = self.accumulator_terms(lp)? else {
            return refuse(format!(
                "loop over {} is a recurrence: it has no addends",
                self.index_name(lp)?
            ));
        };
        let mut out = vec![];
        for rest in per {
            let mut sum = rest[0];
            for &t in &rest[1..] {
                sum = self.make_op(ADD, &[sum, t], &Val::None)?;
            }
            out.push(sum);
        }
        Ok(out)
    }
}

/// The terms of an `add` chain, left to right.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_add_terms(
    ctx: *mut Ctx,
    node: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| give(buf, cap, n, &c.add_terms(node)?))
}

/// Refuses, by name, a nested loop neither direction can carry.
///
/// # Safety
/// `ctx` must be a live context.
#[no_mangle]
pub unsafe extern "C" fn rx_refuse_nesting(ctx: *mut Ctx, lp: u32) -> c_int {
    guard(ctx, |c| {
        c.node(lp)?;
        c.refuse_nesting(lp)
    })
}

/// Whether a loop is an accumulator (`*out` is 1) or a recurrence (0).
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_loop_is_accumulator(ctx: *mut Ctx, lp: u32, out: *mut c_int) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        c.node(lp)?;
        unsafe { *out = c.is_accumulator(lp)? as c_int };
        Ok(())
    })
}

/// The addend of each carry of an accumulator loop.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_loop_addends(
    ctx: *mut Ctx,
    lp: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        c.node(lp)?;
        give(buf, cap, n, &c.loop_addends(lp)?)
    })
}
