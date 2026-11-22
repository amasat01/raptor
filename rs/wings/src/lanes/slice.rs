//! The warp form of a wide kernel: one warp serves a sample, and every wide value is held sliced across its lanes. A value of
//! `n` elements (`n` a whole number of warps, wide enough to pay) is held as `n / warp` elements per lane: element `j` of lane
//! `l` is element `j * warp + l` of the value. An elementwise op, a splat, a read of a plane at an arithmetic run of indices and
//! a loop carrying such values act on the slice alone; a narrow value is computed by every lane alike; a sum over a sliced
//! value is each lane's partial sum added up across the lanes. A node with no rule here leaves the kernel in its thread form.
use crate::construct::refs;
use crate::place::{kind_of, one_index, ELEMENTWISE};
use crate::planes::{type_parts, NodeType};
use crate::rewrite::simplified;
use crate::slots::{carries, inits, nexts};
use crate::wide::gather;
use crate::{is_a, list_out};
use raptor_ir::build::LoopParts;
use raptor_ir::door::guard;
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::{text_val, Val};
use std::collections::HashMap;
use std::ffi::c_int;

/// The literal of an op that reads one element of a plane the warp holds.
const WARP_READ: &str = "warp_read";
/// The kind of the op that adds one value per lane over the warp.
const WARP_SUM: &str = "warp_sum";
/// The longest short vector a fill loop unrolls into registers.
const NARROW_FILL_MAX: i64 = 32;

/// What the slicing reads of the target and of how the form was asked for.
#[derive(Debug, Clone, Copy)]
pub struct SliceSpec {
    /// The lanes of a warp.
    pub warp: i64,
    /// The narrowest vector held sliced.
    pub slice_min: i64,
    /// The shortest sliced value from which an automatic choice builds a warp.
    pub team_min: i64,
    /// Whether the form was asked for by name (else the automatic rules decline some kernels).
    pub explicit: bool,
}

/// Why the slicing stopped: a node with no rule here, or a fault of the context.
enum Stop {
    /// A node the warp form has no rule for.
    NoRule,
    /// A refusal or fault of the context.
    Fault(Fault),
}

impl From<Fault> for Stop {
    fn from(fault: Fault) -> Stop {
        Stop::Fault(fault)
    }
}

/// The result of a step of the slicing.
type R<T> = Result<T, Stop>;

/// A node of the sliced kernel and whether it is held sliced.
type Held = (u32, bool);

/// The text of a node's fields.
fn fields(pairs: &[(&str, Val)]) -> String {
    text_val(&Val::Tuple(
        pairs
            .iter()
            .map(|(key, value)| Val::Tuple(vec![Val::Str((*key).into()), value.clone()]))
            .collect(),
    ))
}

/// The slicing of one kernel.
struct Slicer<'a> {
    /// The context the new nodes are made in.
    ctx: &'a mut Ctx,
    /// The target's numbers.
    spec: SliceSpec,
    /// The node holding the lane's place in its warp.
    lane: u32,
    /// What each node became.
    memo: HashMap<u32, Held>,
    /// What each loop became and which of its slots are sliced.
    loops: HashMap<u32, (u32, Vec<bool>)>,
    /// Whether any value ended up sliced.
    used: bool,
    /// The widest value held sliced.
    widest: i64,
    /// The run-time indexed writes into a slice.
    writes: usize,
    /// The sums over the lanes.
    folds: usize,
}

impl Slicer<'_> {
    /// Whether a value of this type is held sliced: a vector of a whole number of warps, wide enough to pay.
    fn sliced(&self, ty: &NodeType) -> bool {
        let warp = self.spec.warp;
        warp > 0
            && ty.shape.len() == 1
            && ty.shape[0] % warp == 0
            && ty.shape[0] >= self.spec.slice_min
    }

    /// The type of the slice of a value of this type.
    fn slice_type(&self, ty: &NodeType) -> NodeType {
        NodeType {
            dtype: ty.dtype.clone(),
            shape: vec![ty.shape[0] / self.spec.warp],
            tag: ty.tag.clone(),
        }
    }

    /// A scalar type of the dtype.
    fn scalar(dtype: &str) -> NodeType {
        NodeType {
            dtype: dtype.into(),
            shape: vec![],
            tag: None,
        }
    }

    /// An op of the type, with no check of its operands.
    fn op(&mut self, kind: &str, ty: &NodeType, operands: &[u32], literal: Val) -> R<u32> {
        let text = fields(&[("kind", Val::Str(kind.into())), ("literal", literal)]);
        Ok(self.ctx.make(
            "Op",
            &ty.dtype,
            &ty.shape,
            ty.tag.as_deref(),
            operands,
            &text,
        )?)
    }

    /// A constant of the type.
    fn constant(&mut self, value: Val, ty: &NodeType) -> R<u32> {
        let text = fields(&[("literal", value)]);
        Ok(self
            .ctx
            .make("Const", &ty.dtype, &ty.shape, ty.tag.as_deref(), &[], &text)?)
    }

    /// An integer constant of the type.
    fn int(&mut self, value: i64, ty: &NodeType) -> R<u32> {
        self.constant(Val::Int(i128::from(value)), ty)
    }

    /// The node `lane + offset`, as an index of the type.
    fn lane_at(&mut self, offset: i64, ty: &NodeType) -> R<u32> {
        let step = self.int(offset, ty)?;
        Ok(self.ctx.make_op("add", &[self.lane, step], &Val::None)?)
    }

    /// A copy of the node over other operands and another type, every field kept.
    fn remake(&mut self, node: u32, operands: &[u32], ty: &NodeType) -> R<u32> {
        let record = &self.ctx.nodes[node as usize];
        let class = self.ctx.table.classes[record.class].name.clone();
        let slots = &self.ctx.table.classes[record.class].slots;
        let pairs: Vec<Val> = slots
            .iter()
            .zip(record.fields.iter())
            .filter(|(name, _)| !name.starts_with('_'))
            .map(|(name, value)| Val::Tuple(vec![Val::Str(name.clone()), value.clone()]))
            .collect();
        let text = text_val(&Val::Tuple(pairs));
        Ok(self.ctx.make(
            &class,
            &ty.dtype,
            &ty.shape,
            ty.tag.as_deref(),
            operands,
            &text,
        )?)
    }

    /// What the node became, and whether it is held sliced.
    fn of(&mut self, node: u32) -> R<Held> {
        if !self.memo.contains_key(&node) {
            self.operands_first(node)?;
        }
        if let Some(&held) = self.memo.get(&node) {
            return Ok(held);
        }
        let got = self.one(node)?;
        self.memo.insert(node, got);
        self.used |= got.1;
        let ty = type_parts(self.ctx, node);
        if got.1 && ty.shape.len() == 1 {
            self.widest = self.widest.max(ty.shape[0]);
        }
        Ok(got)
    }

    /// Whether the node is an op other than a vector, or a select (the chains that run deep).
    fn chained(&self, node: u32) -> bool {
        match kind_of(self.ctx, node) {
            Ok(Some(kind)) => kind != "vec" && (is_a(self.ctx, node, "Op") || kind == "select"),
            _ => false,
        }
    }

    /// Slices the operands under the node in post-order with an explicit stack, so a dependent chain of any depth costs one
    /// frame.
    fn operands_first(&mut self, root: u32) -> R<()> {
        let mut stack = vec![root];
        while let Some(&node) = stack.last() {
            if self.memo.contains_key(&node) {
                stack.pop();
                continue;
            }
            if self.chained(node) {
                let pending = self.ctx.nodes[node as usize]
                    .ops
                    .iter()
                    .copied()
                    .find(|&o| self.chained(o) && !self.memo.contains_key(&o));
                if let Some(operand) = pending {
                    stack.push(operand);
                    continue;
                }
            }
            stack.pop();
            if node != root {
                self.of(node)?;
            }
        }
        Ok(())
    }

    /// The rule of one node.
    fn one(&mut self, node: u32) -> R<Held> {
        let ctx = &*self.ctx;
        if is_a(ctx, node, "Const") || is_a(ctx, node, "LoopIndex") || node == self.lane {
            return Ok((node, false));
        }
        let ty = type_parts(ctx, node);
        if is_a(ctx, node, "LoopCarry") {
            return if self.sliced(&ty) {
                Err(Stop::NoRule)
            } else {
                Ok((node, false))
            };
        }
        if is_a(ctx, node, "Leaf") {
            return if self.sliced(&ty) {
                Ok((self.read(node)?, true))
            } else {
                Ok((node, false))
            };
        }
        if is_a(ctx, node, "LoopValue") {
            return self.loop_value(node);
        }
        if is_a(ctx, node, "Loop") {
            return Err(Stop::NoRule);
        }
        if self.is_element_read(node)? {
            let operands = ctx.nodes[node as usize].ops.clone();
            let (index, was) = self.of(operands[1])?;
            if was {
                return Err(Stop::NoRule);
            }
            let made = self.op(
                "component_at",
                &ty,
                &[operands[0], index],
                Val::Str(WARP_READ.into()),
            )?;
            return Ok((made, false));
        }
        if is_a(ctx, node, "At") {
            return self.same_in_every_lane(node);
        }
        if is_a(ctx, node, "Op") || is_a(ctx, node, "Select") {
            return self.elementwise(node);
        }
        self.same_in_every_lane(node)
    }

    /// Whether the node reads one element of a sliced input or mutable at a run-time index.
    fn is_element_read(&self, node: u32) -> R<bool> {
        let ctx = &*self.ctx;
        if !(is_a(ctx, node, "Op") && kind_of(ctx, node)?.as_deref() == Some("component_at"))
            || !one_index(ctx, node)
        {
            return Ok(false);
        }
        let leaf = ctx.nodes[node as usize].ops[0];
        if !(is_a(ctx, leaf, "Leaf") && self.sliced(&type_parts(ctx, leaf))) {
            return Ok(false);
        }
        let role = ctx.str_field(leaf, "role")?;
        let vector_input = role == "plane_in" && ctx.nodes[leaf as usize].shape.len() == 1;
        Ok(vector_input || role == "mutable")
    }

    /// A per-sample vector input read whole: lane `l` reads its elements `j * warp + l`.
    fn read(&mut self, leaf: u32) -> R<u32> {
        let ctx = &*self.ctx;
        let (role, kind) = (ctx.str_field(leaf, "role")?, ctx.str_field(leaf, "kind")?);
        let vector_input = role == "plane_in" && ctx.nodes[leaf as usize].shape.len() == 1;
        if !vector_input && !(role == "mutable" && kind == "prior_read") {
            return Err(Stop::NoRule);
        }
        let ty = type_parts(ctx, leaf);
        let index = Slicer::scalar("i32");
        let scalar = Slicer::scalar(&ty.dtype);
        let mut parts = vec![];
        for j in 0..ty.shape[0] / self.spec.warp {
            let at = self.lane_at(j * self.spec.warp, &index)?;
            parts.push(self.op(
                "component_at",
                &scalar,
                &[leaf, at],
                Val::Str(WARP_READ.into()),
            )?);
        }
        let out = self.slice_type(&ty);
        self.op("vec", &out, &parts, Val::None)
    }

    /// A node every lane computes alike: none of its operands may be a slice.
    fn same_in_every_lane(&mut self, node: u32) -> R<Held> {
        let operands = self.ctx.nodes[node as usize].ops.clone();
        let mut made = vec![];
        let mut changed = false;
        for &operand in &operands {
            let (new, was) = self.of(operand)?;
            if was {
                return Err(Stop::NoRule);
            }
            changed |= new != operand;
            made.push(new);
        }
        if !changed {
            return Ok((node, false));
        }
        Ok((self.ctx.rebuild(node, &made)?, false))
    }

    /// The rule of an op or a select.
    fn elementwise(&mut self, node: u32) -> R<Held> {
        let ty = type_parts(self.ctx, node);
        let kind = kind_of(self.ctx, node)?.unwrap_or_default();
        let operands = self.ctx.nodes[node as usize].ops.clone();
        if !self.sliced(&ty) {
            let arity = if kind == "dot" { 2 } else { 1 };
            if ["sum", "dot", "norm"].contains(&kind.as_str()) && operands.len() == arity {
                let held = operands
                    .iter()
                    .map(|&o| self.of(o))
                    .collect::<R<Vec<Held>>>()?;
                if held.iter().all(|h| h.1) {
                    return Ok((self.fold(&kind, &held, &ty)?, false));
                }
            }
            return self.same_in_every_lane(node);
        }
        let out = self.slice_type(&ty);
        match kind.as_str() {
            "splat" => {
                let (inner, was) = self.of(operands[0])?;
                if was {
                    return Err(Stop::NoRule);
                }
                Ok((self.remake(node, &[inner], &out)?, true))
            }
            "vec" => Ok((self.vector(node, &out)?, true)),
            "set_component_at" if one_index(self.ctx, node) => Ok((self.write(node, &out)?, true)),
            _ => {
                let held = operands
                    .iter()
                    .map(|&o| self.of(o))
                    .collect::<R<Vec<Held>>>()?;
                let select = kind == "select";
                if (select && !self.ctx.nodes[operands[0] as usize].shape.is_empty())
                    || !ELEMENTWISE.contains(&kind.as_str())
                {
                    return Err(Stop::NoRule);
                }
                for (&(_, was), &o) in held.iter().zip(&operands) {
                    if !was && !self.ctx.nodes[o as usize].shape.is_empty() {
                        return Err(Stop::NoRule);
                    }
                }
                let new: Vec<u32> = held.iter().map(|h| h.0).collect();
                Ok((self.remake(node, &new, &out)?, true))
            }
        }
    }

    /// A sum, dot or norm of sliced values: each lane reduces its slice, and the lanes add up.
    fn fold(&mut self, kind: &str, held: &[Held], ty: &NodeType) -> R<u32> {
        self.folds += 1;
        let first = held[0].0;
        if kind == "sum" {
            return self.op(WARP_SUM, ty, &[first], Val::None);
        }
        let last = held[held.len() - 1].0;
        let partial = self.op("dot", ty, &[first, last], Val::None)?;
        let total = self.op(WARP_SUM, ty, &[partial], Val::None)?;
        if kind == "dot" {
            Ok(total)
        } else {
            self.op("sqrt", ty, &[total], Val::None)
        }
    }

    /// A `vec` of reads of one plane at an arithmetic run of indices: lane `l` reads the run's elements `j * warp + l`; any
    /// other `vec` must be one expression of the element index.
    fn vector(&mut self, node: u32, out: &NodeType) -> R<u32> {
        let Some(run) = gather(self.ctx, node)? else {
            return self.affine(node, out);
        };
        let first = self.ctx.nodes[node as usize].ops[0];
        let first_type = type_parts(self.ctx, first);
        let index_type = type_parts(self.ctx, self.ctx.nodes[first as usize].ops[1]);
        let base = match run.base {
            Some(base) => {
                let (held, was) = self.of(base)?;
                if was {
                    return Err(Stop::NoRule);
                }
                Some(held)
            }
            None => None,
        };
        let warp = i128::from(self.spec.warp);
        let mut parts = vec![];
        for j in 0..out.shape[0] {
            let offset = run.start + i128::from(j) * warp * run.step;
            let step = self.int(run.step as i64, &index_type)?;
            let scaled = self.ctx.make_op("mul", &[self.lane, step], &Val::None)?;
            let shift = self.int(offset as i64, &index_type)?;
            let mut at = self.ctx.make_op("add", &[scaled, shift], &Val::None)?;
            if let Some(base) = base {
                at = self.ctx.make_op("add", &[base, at], &Val::None)?;
            }
            let fields = fields(&[
                ("kind", Val::Str("at".into())),
                ("literal", Val::None),
                ("edge", Val::Str("unchecked".into())),
            ]);
            parts.push(self.ctx.make(
                "At",
                &first_type.dtype,
                &first_type.shape,
                first_type.tag.as_deref(),
                &[run.plane, at],
                &fields,
            )?);
        }
        self.op("vec", out, &parts, Val::None)
    }

    /// A `vec` whose parts are one expression of the element index: the parts agree node by node, and where they differ they
    /// are constants in an arithmetic run. Lane `l` computes the parts `j * warp + l` from the run.
    fn affine(&mut self, node: u32, out: &NodeType) -> R<u32> {
        let parts = self.ctx.nodes[node as usize].ops.clone();
        let mut built = vec![];
        for j in 0..out.shape[0] {
            built.push(self.aligned(&parts, j)?);
        }
        self.op("vec", out, &built, Val::None)
    }

    /// Part `j * warp + lane` of the run `nodes`: the same expression with its constants moved along.
    fn aligned(&mut self, nodes: &[u32], j: i64) -> R<u32> {
        let first = nodes[0];
        if nodes.iter().all(|&n| n == first) {
            let shaped = !self.ctx.nodes[first as usize].shape.is_empty();
            return if shaped {
                self.plain(first)
            } else {
                Ok(self.of(first)?.0)
            };
        }
        let ty = type_parts(self.ctx, first);
        let plain_const = |ctx: &Ctx, n: u32| {
            is_a(ctx, n, "Const")
                && !is_a(ctx, n, "RoleConst")
                && type_parts(ctx, n).dtype == ty.dtype
                && ctx.nodes[n as usize].shape == ctx.nodes[first as usize].shape
        };
        if nodes.iter().all(|&n| plain_const(self.ctx, n)) {
            return self.moved_constant(nodes, j, &ty);
        }
        let ctx = &*self.ctx;
        let same = |n: u32| {
            let (a, b) = (&ctx.nodes[n as usize], &ctx.nodes[first as usize]);
            let literal = |h: u32| ctx.field(h, "literal").ok();
            let kind = |h: u32| ctx.field(h, "kind").ok();
            a.class == b.class
                && !a.ops.is_empty()
                && a.ops.len() == b.ops.len()
                && kind(n) == kind(first)
                && literal(n) == literal(first)
                && type_parts(ctx, n).dtype == ty.dtype
                && a.shape == b.shape
                && a.tag == b.tag
        };
        if !nodes.iter().all(|&n| same(n)) {
            return Err(Stop::NoRule);
        }
        let width = self.ctx.nodes[first as usize].ops.len();
        let mut operands = vec![];
        for k in 0..width {
            let column: Vec<u32> = nodes
                .iter()
                .map(|&n| self.ctx.nodes[n as usize].ops[k])
                .collect();
            operands.push(self.aligned(&column, j)?);
        }
        Ok(self.ctx.rebuild(first, &operands)?)
    }

    /// The constants of a run as `start + lane * step`, from the part `j * warp` along.
    fn moved_constant(&mut self, nodes: &[u32], j: i64, ty: &NodeType) -> R<u32> {
        let literals: Vec<Val> = nodes
            .iter()
            .map(|&n| self.ctx.field(n, "literal"))
            .collect::<Result<_, _>>()?;
        let warp = self.spec.warp;
        let (start, step) = match literals.as_slice() {
            [Val::Int(a), Val::Int(b), ..] => {
                let step = b - a;
                if literals
                    .windows(2)
                    .any(|w| matches!(w, [Val::Int(x), Val::Int(y)] if y - x != step))
                {
                    return Err(Stop::NoRule);
                }
                (Val::Int(a + i128::from(j * warp) * step), Val::Int(step))
            }
            [Val::Float(a), Val::Float(b), ..] => {
                let step = b - a;
                if literals
                    .windows(2)
                    .any(|w| matches!(w, [Val::Float(x), Val::Float(y)] if y - x != step))
                {
                    return Err(Stop::NoRule);
                }
                (Val::Float(a + (j * warp) as f64 * step), Val::Float(step))
            }
            _ => return Err(Stop::NoRule),
        };
        let lane = if ty.dtype == "i32" {
            self.lane
        } else {
            self.ctx
                .make_op("cast", &[self.lane], &Val::Str(ty.dtype.clone()))?
        };
        let step = self.constant(step, ty)?;
        let scaled = self.ctx.make_op("mul", &[lane, step], &Val::None)?;
        let start = self.constant(start, ty)?;
        Ok(self.ctx.make_op("add", &[start, scaled], &Val::None)?)
    }

    /// A node held whole by every lane.
    fn plain(&mut self, node: u32) -> R<u32> {
        let (made, was) = self.of(node)?;
        if was {
            Err(Stop::NoRule)
        } else {
            Ok(made)
        }
    }

    /// `v[k] = x` on a sliced vector: the lane whose element `k` is writes `x`, the others keep theirs. A constant `k` touches
    /// one slot; a run-time one is a select in every slot, so no slot is chosen at run time.
    fn write(&mut self, node: u32, out: &NodeType) -> R<u32> {
        let operands = self.ctx.nodes[node as usize].ops.clone();
        let (vector, was) = self.of(operands[0])?;
        let (index, index_sliced) = self.of(operands[1])?;
        let (value, value_sliced) = self.of(operands[2])?;
        if !was || index_sliced || value_sliced {
            return Err(Stop::NoRule);
        }
        let scalar = Slicer::scalar(&self.ctx.nodes[node as usize].dtype);
        let index_type = type_parts(self.ctx, index);
        let known = if is_a(self.ctx, index, "Const") {
            match self.ctx.field(index, "literal")? {
                Val::Int(i) => Some(i),
                Val::Float(f) => Some(f as i128),
                _ => None,
            }
        } else {
            None
        };
        self.writes += usize::from(known.is_none());
        let warp = i128::from(self.spec.warp);
        let mut parts = vec![];
        for j in 0..out.shape[0] {
            let old = self.op("component", &scalar, &[vector], Val::Int(i128::from(j)))?;
            if known.is_some_and(|k| k.div_euclid(warp) != i128::from(j)) {
                parts.push(old);
                continue;
            }
            let here = self.lane_at(j * self.spec.warp, &index_type)?;
            let same = self.ctx.make_op("eq", &[index, here], &Val::None)?;
            parts.push(self.ctx.make(
                "Select",
                &scalar.dtype,
                &[],
                None,
                &[same, value, old],
                "()",
            )?);
        }
        self.op("vec", out, &parts, Val::None)
    }

    /// A loop that writes element `j` of a wide vector at trip `j` from a value of `j` alone is the vector of its values: lane
    /// `l` computes the value at its own elements. Any other loop is sliced slot by slot.
    fn loop_value(&mut self, node: u32) -> R<Held> {
        let slot = match self.ctx.field(node, "literal")? {
            Val::Int(i) => i as usize,
            _ => return Err(Stop::NoRule),
        };
        let lp = self.ctx.nodes[node as usize].ops[0];
        if let Some(filled) = self.fill(lp, slot)? {
            return Ok(filled);
        }
        let (new_loop, slots) = self.slice_loop(lp)?;
        let carry = carries(self.ctx, new_loop)?[slot];
        let ty = type_parts(self.ctx, carry);
        let text = fields(&[
            ("kind", Val::Str("loop_value".into())),
            ("literal", Val::Int(slot as i128)),
        ]);
        let made = self.ctx.make(
            "LoopValue",
            &ty.dtype,
            &ty.shape,
            ty.tag.as_deref(),
            &[new_loop],
            &text,
        )?;
        Ok((made, slots[slot]))
    }

    /// The vector a fill loop stands for, when the loop is one.
    fn fill(&mut self, lp: u32, slot: usize) -> R<Option<Held>> {
        let ctx = &*self.ctx;
        let carry = carries(ctx, lp)?[slot];
        let ty = type_parts(ctx, carry);
        let int = |name: &str| match ctx.field(lp, name) {
            Ok(Val::Int(i)) => Some(i as i64),
            _ => None,
        };
        let (start, step, stop) = (int("start"), int("step"), int("stop"));
        let open = ctx.field(lp, "exit_cond")? != Val::None
            || ctx.field(lp, "count")? != Val::None
            || !refs(ctx, lp, "body_sinks")?.is_empty();
        if open
            || ty.shape.len() != 1
            || (start, step) != (Some(0), Some(1))
            || stop != Some(ty.shape[0])
        {
            return Ok(None);
        }
        let narrow = !self.sliced(&ty);
        if narrow && ty.shape[0] > NARROW_FILL_MAX {
            return Ok(None);
        }
        let next = nexts(ctx, lp)?[slot];
        let index = refs(ctx, lp, "index")?[0];
        let writes = is_a(ctx, next, "Op")
            && kind_of(ctx, next)?.as_deref() == Some("set_component_at")
            && ctx.nodes[next as usize].ops[..2] == [carry, index];
        if !writes {
            return Ok(None);
        }
        let element = ctx.nodes[next as usize].ops[2];
        let own = carries(ctx, lp)?;
        if reach(ctx, element).iter().any(|n| own.contains(n)) {
            return Ok(None);
        }
        let index_type = type_parts(ctx, index);
        if narrow {
            let mut parts = vec![];
            for r in 0..ty.shape[0] {
                let fixed = self.int(r, &index_type)?;
                let part = self
                    .ctx
                    .substitute(&[element], &HashMap::from([(index, fixed)]))?[0];
                parts.push(self.plain(part)?);
            }
            return Ok(Some((self.op("vec", &ty, &parts, Val::None)?, false)));
        }
        let mut parts = vec![];
        for j in 0..ty.shape[0] / self.spec.warp {
            let at = self.lane_at(j * self.spec.warp, &index_type)?;
            let part = self
                .ctx
                .substitute(&[element], &HashMap::from([(index, at)]))?[0];
            parts.push(self.plain(part)?);
        }
        let out = self.slice_type(&ty);
        Ok(Some((self.op("vec", &out, &parts, Val::None)?, true)))
    }

    /// The loop with its sliced carries made again, and which of its slots are sliced.
    fn slice_loop(&mut self, lp: u32) -> R<(u32, Vec<bool>)> {
        if let Some(done) = self.loops.get(&lp) {
            return Ok(done.clone());
        }
        if !refs(self.ctx, lp, "body_sinks")?.is_empty() {
            return Err(Stop::NoRule);
        }
        let old = carries(self.ctx, lp)?;
        let slots: Vec<bool> = old
            .iter()
            .map(|&c| self.sliced(&type_parts(self.ctx, c)))
            .collect();
        let mut made = vec![];
        for (&carry, &is_sliced) in old.iter().zip(&slots) {
            let new = if is_sliced {
                self.sliced_carry(carry)?
            } else {
                carry
            };
            self.memo.insert(carry, (new, is_sliced));
            made.push(new);
        }
        let (mut new_inits, mut new_nexts) = (vec![], vec![]);
        for ((init, next), &want) in inits(self.ctx, lp)?
            .into_iter()
            .zip(nexts(self.ctx, lp)?)
            .zip(&slots)
        {
            let ((i, init_sliced), (n, next_sliced)) = (self.of(init)?, self.of(next)?);
            if init_sliced != want || next_sliced != want {
                return Err(Stop::NoRule);
            }
            new_inits.push(i);
            new_nexts.push(n);
        }
        let (exit_cond, exit_values) = self.exit_of(lp, &slots)?;
        let count = match self.ctx.field(lp, "count")? {
            Val::Node(count) => Some(self.plain(count)?),
            _ => None,
        };
        let int = |name: &str| match self.ctx.field(lp, name) {
            Ok(Val::Int(i)) => Ok(i as i64),
            _ => Err(Stop::NoRule),
        };
        let dynamic = match self.ctx.field(lp, "dynamic")? {
            Val::Str(text) => Some(text),
            _ => None,
        };
        let parts = LoopParts {
            index: refs(self.ctx, lp, "index")?[0],
            carries: &made,
            inits: &new_inits,
            nexts: &new_nexts,
            range: (int("start")?, int("stop")?, int("step")?),
            body_sinks: &[],
            exit_cond,
            exit_values: &exit_values,
            count,
            count_tail: self.ctx.field(lp, "count_tail")? == Val::Bool(true),
            dynamic: dynamic.as_deref(),
        };
        let built = self.ctx.build_loop(&parts)?;
        self.loops.insert(lp, (built, slots.clone()));
        Ok((built, slots))
    }

    /// The exit condition of a loop and its exit values: the lanes leave together, so the flag may not be a slice.
    fn exit_of(&mut self, lp: u32, slots: &[bool]) -> R<(Option<u32>, Vec<u32>)> {
        let Val::Node(cond) = self.ctx.field(lp, "exit_cond")? else {
            return Ok((None, vec![]));
        };
        let cond = self.plain(cond)?;
        let mut values = vec![];
        for (j, value) in refs(self.ctx, lp, "exit_values")?.into_iter().enumerate() {
            let (made, was) = self.of(value)?;
            if was != slots.get(j).copied().unwrap_or(false) {
                return Err(Stop::NoRule);
            }
            values.push(made);
        }
        Ok((Some(cond), values))
    }

    /// A carry holding a slice in place of the whole value.
    fn sliced_carry(&mut self, carry: u32) -> R<u32> {
        let ty = self.slice_type(&type_parts(self.ctx, carry));
        let text = fields(&[
            ("depth", self.ctx.field(carry, "depth")?),
            ("name", self.ctx.field(carry, "name")?),
            ("slot", self.ctx.field(carry, "slot")?),
        ]);
        Ok(self.ctx.make(
            "LoopCarry",
            &ty.dtype,
            &ty.shape,
            ty.tag.as_deref(),
            &[],
            &text,
        )?)
    }
}

/// Every node under the root.
fn reach(ctx: &Ctx, root: u32) -> Vec<u32> {
    let (mut seen, mut stack) = (std::collections::HashSet::new(), vec![root]);
    while let Some(node) = stack.pop() {
        if seen.insert(node) {
            stack.extend(ctx.nodes[node as usize].ops.iter().copied());
        }
    }
    seen.into_iter().collect()
}

/// The sinks of a kernel in the warp form, or none when a node has no rule there. A commit becomes a warp store; a finish keeps
/// its flag, which every lane computes alike. A kernel the automatic choice declines is one whose only wide work is writes at
/// run-time indices, or whose widest sliced value is under the team minimum.
pub fn slice_sinks(
    ctx: &mut Ctx,
    sinks: &[u32],
    spec: SliceSpec,
) -> Result<Option<Vec<u32>>, Fault> {
    let exact = |ctx: &Ctx, sink: u32, class: &str| {
        ctx.table.classes[ctx.nodes[sink as usize].class].name == class
    };
    if sinks.is_empty()
        || !sinks
            .iter()
            .all(|&s| exact(ctx, s, "Assign") || is_a(ctx, s, "Finish"))
    {
        return Ok(None);
    }
    let simple = simplified(ctx, sinks)?;
    let sinks = &simple[..];
    let lane = ctx.make("WarpLane", "i32", &[], None, &[], "()")?;
    let mut slicer = Slicer {
        ctx,
        spec,
        lane,
        memo: HashMap::new(),
        loops: HashMap::new(),
        used: false,
        widest: 0,
        writes: 0,
        folds: 0,
    };
    match slicer.all(sinks) {
        Ok(out) => {
            let declined = !slicer.used
                || (!spec.explicit && slicer.writes > 0 && slicer.folds == 0)
                || (!spec.explicit && slicer.widest < spec.team_min);
            Ok((!declined).then_some(out))
        }
        Err(Stop::NoRule) => Ok(None),
        Err(Stop::Fault(fault)) => Err(fault),
    }
}

impl Slicer<'_> {
    /// The warp form of each sink, in order.
    fn all(&mut self, sinks: &[u32]) -> R<Vec<u32>> {
        let mut out = vec![];
        for &sink in sinks {
            let value = self.ctx.nodes[sink as usize].ops[0];
            let (value, was) = self.of(value)?;
            if is_a(self.ctx, sink, "Finish") {
                if was {
                    return Err(Stop::NoRule);
                }
                out.push(self.ctx.rebuild(sink, &[value])?);
                continue;
            }
            let ty = type_parts(self.ctx, sink);
            let name = self.ctx.field(sink, "name")?;
            let role = self.ctx.field(sink, "role")?;
            let text = fields(&[
                ("kind", Val::Str("assign".into())),
                ("name", name),
                ("role", role),
            ]);
            out.push(self.ctx.make(
                "WarpStore",
                &ty.dtype,
                &ty.shape,
                ty.tag.as_deref(),
                &[value],
                &text,
            )?);
        }
        Ok(out)
    }
}

/// The sinks of a kernel in the warp form, in the caller's context; see `raptor_wings.h`. `*found` is 0 when the kernel keeps
/// its thread form.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_lanes_slice_sinks(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    warp: i64,
    slice_min: i64,
    team_min: i64,
    explicit: c_int,
    buf: *mut u32,
    cap: usize,
    count: *mut usize,
    found: *mut c_int,
) -> c_int {
    guard(ctx, |ctx| {
        if (nsinks > 0 && sinks.is_null()) || found.is_null() {
            return Err(Fault::BadArg("sinks or found is NULL".into()));
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
        let spec = SliceSpec {
            warp,
            slice_min,
            team_min,
            explicit: explicit != 0,
        };
        let made = slice_sinks(ctx, given, spec)?;
        unsafe { *found = c_int::from(made.is_some()) }
        list_out(buf, cap, count, &made.unwrap_or_default())
    })
}
