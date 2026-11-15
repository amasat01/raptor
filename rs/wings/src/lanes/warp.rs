//! Warp-uniform values: a node is uniform when it is proven equal over every aligned run of `warp`
//! consecutive lanes, so the whole warp may read it at one address. Unknown is not uniform.
use super::{axes_of, Eval};
use crate::{is_a, konst};
use raptor_ir::door::guard;
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::HashMap;
use std::ffi::c_int;

/// The result of an analysis: its answer, or the refusal that stopped it.
type R<T> = Result<T, Fault>;

/// The number a constant node holds, if it holds one.
fn literal(ctx: &Ctx, node: u32) -> Option<f64> {
    if !is_a(ctx, node, "Const") {
        return None;
    }
    match ctx.field(node, "literal") {
        Ok(Val::Int(i)) => Some(i as f64),
        Ok(Val::Float(f)) => Some(f),
        _ => None,
    }
}

/// The proof state for one kernel: the lane evaluation and what each node was found to be.
struct Proof<'a> {
    /// The lane evaluation of the kernel's indices.
    eval: Eval<'a>,
    /// The warp width, when the target has one.
    warp: Option<i128>,
    /// Whether each node already examined is warp-uniform.
    memo: HashMap<u32, bool>,
}

impl Proof<'_> {
    /// The numerator of a run quotient with the added constant below one and the casts peeled off.
    fn peeled(&self, mut node: u32) -> u32 {
        let ctx = self.eval.c;
        let small_add = self.eval.kind(node).as_deref() == Some("add")
            && literal(ctx, self.eval.ops(node)[1]).is_some_and(|h| (0.0..1.0).contains(&h));
        if small_add {
            node = self.eval.ops(node)[0];
        }
        while self.eval.kind(node).as_deref() == Some("cast") {
            node = self.eval.ops(node)[0];
        }
        node
    }

    /// Whether `floor(numerator / divisor)` is the same for the lanes of a run: the divisor is a
    /// positive integer multiple of the warp, and the numerator is a lane form whose lanes of a run
    /// differ only in its fastest axis, with coefficient 1 there.
    fn run_quotient(&mut self, numerator: u32, divisor: f64) -> R<bool> {
        let Some(warp) = self.warp.filter(|&w| w != 0) else {
            return Ok(false);
        };
        if !(divisor > 0.0 && divisor == divisor.trunc() && (divisor as i128) % warp == 0) {
            return Ok(false);
        }
        let numerator = self.peeled(numerator);
        let v = self.eval.value(numerator)?;
        let Some(cst) = v.cst else { return Ok(false) };
        if !cst.is_int() || cst.n % warp != 0 || v.dbad != 0 || v.bad != 0 || v.data {
            return Ok(false);
        }
        let last = self.eval.axes.len() - 1;
        for (axis, coef) in &v.lin {
            let inner = self.eval.axes[*axis].stride % warp != 0;
            let whole = coef.d == 1
                && (if inner {
                    coef.n == 1
                } else {
                    coef.n % warp == 0
                });
            if !whole || (inner && *axis != last) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether `y * (1 / d)` is a run quotient, as a division by a constant folds.
    fn scaled_quotient(&mut self, node: u32) -> R<bool> {
        let operands = self.eval.ops(node).to_vec();
        if operands.len() != 2 {
            return Ok(false);
        }
        for (scale, other) in [(operands[0], operands[1]), (operands[1], operands[0])] {
            let Some(c) = literal(self.eval.c, scale) else {
                continue;
            };
            if c > 0.0
                && c < 1.0
                && c * (1.0 / c).round() == 1.0
                && self.run_quotient(other, (1.0 / c).round())?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether the value of the node is proven equal over every aligned run of the warp.
    fn uniform(&mut self, node: u32) -> R<bool> {
        if let Some(&held) = self.memo.get(&node) {
            return Ok(held);
        }
        let ctx = self.eval.c;
        let out = if is_a(ctx, node, "Const") || is_a(ctx, node, "LoopIndex") {
            true
        } else if is_a(ctx, node, "Leaf") {
            matches!(
                ctx.str_field(node, "role")?.as_str(),
                "uniform" | "nsamples"
            )
        } else if is_a(ctx, node, "At") {
            self.uniform(self.eval.raw_index(node))?
        } else if let Some(kind) = self.eval.kind(node) {
            self.uniform_op(node, &kind)?
        } else {
            false
        };
        self.memo.insert(node, out);
        Ok(out)
    }

    /// The rule for an op: a quotient by a multiple of the warp, a scaled one, else uniform
    /// operands of an op that is not random.
    fn uniform_op(&mut self, node: u32, kind: &str) -> R<bool> {
        let operands = self.eval.ops(node).to_vec();
        if kind == "div" && operands.len() == 2 {
            if let Some(divisor) = literal(self.eval.c, operands[1]) {
                if self.run_quotient(operands[0], divisor)? {
                    return Ok(true);
                }
            }
        }
        if kind == "mul" && self.scaled_quotient(node)? {
            return Ok(true);
        }
        if konst("RANDOM_OPS").contains(&kind) {
            return Ok(false);
        }
        for operand in operands {
            if !self.uniform(operand)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// For each node whether it is proven equal over every aligned run of `warp` lanes (none: the
/// target names no run width, so only what needs no run is proven) in the kernel with these sinks.
pub fn warp_uniform(ctx: &Ctx, sinks: &[u32], warp: Option<i128>, nodes: &[u32]) -> R<Vec<bool>> {
    let axes = axes_of(ctx, sinks)?;
    let all = (1u64 << axes.len()) - 1;
    let mut proof = Proof {
        eval: Eval {
            c: ctx,
            axes,
            memo: HashMap::new(),
            all,
        },
        warp,
        memo: HashMap::new(),
    };
    nodes.iter().map(|&node| proof.uniform(node)).collect()
}

/// The warp-uniform flags of nodes of the kernel with these sinks; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_lanes_warp_uniform(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    warp: *const i64,
    nodes: *const u32,
    nnodes: usize,
    flags: *mut c_int,
) -> c_int {
    guard(ctx, |ctx| {
        if (nsinks > 0 && sinks.is_null()) || (nnodes > 0 && (nodes.is_null() || flags.is_null())) {
            return Err(Fault::BadArg("sinks, nodes or flags is NULL".into()));
        }
        let given = if nsinks == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        let asked = if nnodes == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(nodes, nnodes) }
        };
        let warp = if warp.is_null() {
            None
        } else {
            Some(unsafe { *warp } as i128)
        };
        for (k, flag) in warp_uniform(ctx, given, warp, asked)?
            .into_iter()
            .enumerate()
        {
            unsafe { *flags.add(k) = flag as c_int }
        }
        Ok(())
    })
}
