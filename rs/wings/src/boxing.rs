//! The thread box: one thread computes `T` lanes along a non-fastest axis. The sink set of a boxed
//! entry holds each sink once per lane, as a copy of its value graph in which every node that may
//! move along the box axis is copied for lane `k` (the lane `i + k * stride`) and every other
//! node is lane 0's, shared by all. The copies are made in the caller's context.
use crate::construct::refs;
use crate::lanes::{box_candidate, reachable, varying_nodes};
use crate::planes::{type_parts, NodeType};
use crate::{in_kinds, index_operands, is_a, is_update_of, konst, list_out, op_kind};
use raptor_ir::build::LoopParts;
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::{text_val, Val};
use std::collections::{HashMap, HashSet};
use std::ffi::c_int;

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// The box: `lanes` lanes along lane axis `axis`, `stride` flat lanes apart.
struct BoxSpec {
    /// The lane axis the box runs along.
    axis: usize,
    /// How many lanes the box holds.
    lanes: usize,
    /// How many flat lanes apart two neighbouring box lanes are.
    stride: i64,
}

/// Per-lane copies of a node graph, with the memo of what was copied.
struct Copy<'a> {
    /// The context the copies are made in.
    ctx: &'a mut Ctx,
    /// The nodes whose value differs between the box's lanes.
    varies: HashSet<u32>,
    /// The box being copied.
    boxed: BoxSpec,
    /// The node holding each lane's number in the box.
    lane_nodes: Vec<u32>,
    /// The copy already made of a node for a lane.
    memo: HashMap<(u32, usize), u32>,
    /// The fused copy already made of a node.
    fused: HashMap<u32, u32>,
}

impl Copy<'_> {
    /// A new op of the given type over the operands.
    fn op(&mut self, kind: &str, ty: &NodeType, operands: &[u32]) -> Outcome<u32> {
        let fields = format!("((\"kind\", {kind:?}), (\"literal\", None))");
        self.ctx.make(
            "Op",
            &ty.dtype,
            &ty.shape,
            ty.tag.as_deref(),
            operands,
            &fields,
        )
    }

    /// A new integer constant of the given type.
    fn constant(&mut self, value: i64, ty: &NodeType) -> Outcome<u32> {
        let fields = format!("((\"literal\", {value}),)");
        self.ctx.make(
            "Const",
            &ty.dtype,
            &ty.shape,
            ty.tag.as_deref(),
            &[],
            &fields,
        )
    }

    /// `lane / divisor` with the divisor a positive integer dividing the stride: the divisor.
    fn divides_stride(&self, node: u32) -> Outcome<Option<i64>> {
        let ctx = &*self.ctx;
        let operands = &ctx.nodes[node as usize].ops;
        let is_division = op_kind(ctx, node)?.as_deref() == Some("div")
            && operands.len() == 2
            && is_a(ctx, operands[0], "SampleIndex")
            && is_a(ctx, operands[1], "Const")
            && matches!(ctx.nodes[node as usize].dtype.as_str(), "i32" | "i64");
        if !is_division {
            return Ok(None);
        }
        let divisor = match ctx.field(operands[1], "literal")? {
            Val::Int(i) => i as i64,
            _ => return Ok(None),
        };
        Ok((divisor > 0 && self.boxed.stride % divisor == 0).then_some(divisor))
    }

    /// The operands a copy of the node needs: a read's raw index (the plane is never copied), none
    /// for a leaf-like node, else the node's own operands.
    fn operands_to_copy(&self, node: u32) -> Vec<u32> {
        let ctx = &*self.ctx;
        if is_a(ctx, node, "At") {
            return index_operands(self.ctx, node);
        }
        if self.fusing(node) {
            return vec![];
        }
        let leaf_like = ["Leaf", "Const", "SampleIndex", "LoopIndex", "LoopCarry"];
        if leaf_like.iter().any(|class| is_a(ctx, node, class)) {
            return vec![];
        }
        ctx.nodes[node as usize].ops.clone()
    }

    /// Whether the node is the result of a loop that moves along the box axis: it is read off the
    /// one loop made for the whole box.
    fn fusing(&self, node: u32) -> bool {
        is_a(self.ctx, node, "LoopValue")
            && self.varies.contains(&self.ctx.nodes[node as usize].ops[0])
    }

    /// Component `lane` of slot `slot` of the fused loop.
    fn component_of_slot(&mut self, fused: u32, slot: i64, lane: usize) -> Outcome<u32> {
        let carry = refs(self.ctx, fused, "carries")?[slot as usize];
        let carry_type = type_parts(self.ctx, carry);
        let fields = format!("((\"kind\", \"loop_value\"), (\"literal\", {slot}))");
        let value = self.ctx.make(
            "LoopValue",
            &carry_type.dtype,
            &carry_type.shape,
            None,
            &[fused],
            &fields,
        )?;
        let fields = format!("((\"kind\", \"component\"), (\"literal\", {lane}))");
        self.ctx
            .make("Op", &carry_type.dtype, &[], None, &[value], &fields)
    }

    /// The loop once for the whole box: carried slot `j` becomes a vector of the lanes' values,
    /// read back per lane by component.
    fn fuse(&mut self, loop_node: u32) -> Outcome<u32> {
        if let Some(&held) = self.fused.get(&loop_node) {
            return Ok(held);
        }
        let lanes = self.boxed.lanes;
        let old_carries = refs(self.ctx, loop_node, "carries")?;
        let mut carries = vec![];
        for (slot, &old) in old_carries.iter().enumerate() {
            let dtype = self.ctx.nodes[old as usize].dtype.clone();
            let (depth, name) = (self.ctx.field(old, "depth")?, self.ctx.field(old, "name")?);
            let fields = format!(
                "((\"depth\", {}), (\"name\", {}), (\"slot\", {slot}))",
                text_val(&depth),
                text_val(&name)
            );
            let new = self
                .ctx
                .make("LoopCarry", &dtype, &[lanes as i64], None, &[], &fields)?;
            for lane in 0..lanes {
                let part = self.component(new, lane, &dtype)?;
                self.memo.insert((old, lane), part);
            }
            carries.push(new);
        }
        let index = refs(self.ctx, loop_node, "index")?[0];
        for lane in 0..lanes {
            self.memo.insert((index, lane), index);
        }
        let inits = self.ctx.nodes[loop_node as usize].ops.clone();
        let nexts = refs(self.ctx, loop_node, "nexts")?;
        let inits = self.lane_vectors(&inits[..carries.len()], &carries)?;
        let nexts = self.lane_vectors(&nexts[..carries.len()], &carries)?;
        let int = |name: &str| match self.ctx.field(loop_node, name) {
            Ok(Val::Int(i)) => Ok(i as i64),
            _ => refuse("a loop bound is an int"),
        };
        let range = (int("start")?, int("stop")?, int("step")?);
        let parts = LoopParts {
            index,
            carries: &carries,
            inits: &inits,
            nexts: &nexts,
            range,
            body_sinks: &[],
            exit_cond: None,
            exit_values: &[],
            count: None,
            count_tail: false,
            dynamic: None,
        };
        let fused = self.ctx.build_loop(&parts)?;
        self.fused.insert(loop_node, fused);
        Ok(fused)
    }

    /// For each node, the vector of its copies for every lane, of the type of the matching carry.
    fn lane_vectors(&mut self, nodes: &[u32], carries: &[u32]) -> Outcome<Vec<u32>> {
        let mut vectors = vec![];
        for (&node, &carry) in nodes.iter().zip(carries) {
            let mut parts = vec![];
            for lane in 0..self.boxed.lanes {
                parts.push(self.copy(node, lane)?);
            }
            let ty = type_parts(self.ctx, carry);
            let fields = "((\"kind\", \"vec\"), (\"literal\", None))";
            vectors.push(self.ctx.make(
                "Op",
                &ty.dtype,
                &ty.shape,
                ty.tag.as_deref(),
                &parts,
                fields,
            )?);
        }
        Ok(vectors)
    }

    /// Component `lane` of a vector node, a scalar of the same dtype.
    fn component(&mut self, vector: u32, lane: usize, dtype: &str) -> Outcome<u32> {
        let fields = format!("((\"kind\", \"component\"), (\"literal\", {lane}))");
        self.ctx.make("Op", dtype, &[], None, &[vector], &fields)
    }

    /// The node for lane `lane`: lane 0's when the node does not move along the axis.
    fn copy(&mut self, node: u32, lane: usize) -> Outcome<u32> {
        if let Some(&done) = self.memo.get(&(node, lane)) {
            return Ok(done);
        }
        let result = if lane > 0 && !self.varies.contains(&node) {
            self.copy(node, 0)?
        } else {
            for child in self.operands_to_copy(node) {
                self.copy(child, lane)?;
            }
            self.build(node, lane)?
        };
        self.memo.insert((node, lane), result);
        Ok(result)
    }

    /// Builds the copy of one node for a lane, its operands' copies being in the memo.
    fn build(&mut self, node: u32, lane: usize) -> Outcome<u32> {
        let ty = type_parts(self.ctx, node);
        if is_a(self.ctx, node, "SampleIndex") {
            return Ok(if lane == 0 {
                node
            } else {
                self.lane_nodes[lane]
            });
        }
        if lane > 0 {
            if let Some(divisor) = self.divides_stride(node)? {
                let at_zero = self.copy(node, 0)?;
                let step = self.constant(lane as i64 * self.boxed.stride / divisor, &ty)?;
                return self.op("add", &ty, &[at_zero, step]);
            }
        }
        if is_a(self.ctx, node, "Leaf") {
            let per_sample = self.ctx.str_field(node, "role")? == "plane_in"
                && self.ctx.nodes[node as usize].shape.is_empty();
            if lane == 0 || !per_sample {
                return Ok(node);
            }
            let fields = "((\"kind\", \"at\"), (\"literal\", None), (\"edge\", \"unchecked\"))";
            return self.ctx.make(
                "At",
                &ty.dtype,
                &ty.shape,
                ty.tag.as_deref(),
                &[node, self.lane_nodes[lane]],
                fields,
            );
        }
        if is_a(self.ctx, node, "Const") || is_a(self.ctx, node, "LoopIndex") {
            return Ok(node);
        }
        if self.fusing(node) {
            let loop_node = self.ctx.nodes[node as usize].ops[0];
            let slot = match self.ctx.field(node, "literal")? {
                Val::Int(i) => i as i64,
                _ => return refuse("a loop value without a slot"),
            };
            let fused = self.fuse(loop_node)?;
            return self.component_of_slot(fused, slot, lane);
        }
        let mut kids: Vec<u32> = vec![];
        for child in self.operands_to_copy(node) {
            kids.push(self.memo[&(child, lane)]);
        }
        if is_a(self.ctx, node, "At") {
            kids.insert(0, self.ctx.nodes[node as usize].ops[0]);
        }
        if kids == self.ctx.nodes[node as usize].ops {
            return Ok(node);
        }
        self.ctx.rebuild(node, &kids)
    }
}

/// The sink set of the box's entry: each sink once per lane of the box.
fn boxed_sinks(ctx: &mut Ctx, sinks: &[u32], boxed: BoxSpec) -> Outcome<Vec<u32>> {
    let varies = varying_nodes(ctx, sinks, boxed.axis)?;
    let lane = ctx.make("SampleIndex", "i32", &[], None, &[], "()")?;
    let lane_type = type_parts(ctx, lane);
    let mut copy = Copy {
        ctx,
        varies,
        boxed,
        lane_nodes: vec![lane],
        memo: HashMap::new(),
        fused: HashMap::new(),
    };
    for k in 1..copy.boxed.lanes {
        let step = copy.constant(k as i64 * copy.boxed.stride, &lane_type)?;
        let lane_k = copy.op("add", &lane_type, &[lane, step])?;
        copy.lane_nodes.push(lane_k);
    }
    let mut stores = vec![];
    for &sink in sinks {
        let value = copy.ctx.nodes[sink as usize].ops[0];
        let (name, role) = (
            copy.ctx.str_field(sink, "name")?,
            copy.ctx.str_field(sink, "role")?,
        );
        for k in 0..copy.boxed.lanes {
            let copied = copy.copy(value, k)?;
            let ty = type_parts(copy.ctx, copied);
            let fields =
                format!("((\"kind\", \"box_store\"), (\"name\", {name:?}), (\"role\", {role:?}))");
            let lane_node = copy.lane_nodes[k];
            stores.push(copy.ctx.make(
                "BoxStore",
                &ty.dtype,
                &ty.shape,
                ty.tag.as_deref(),
                &[copied, lane_node],
                &fields,
            )?);
        }
    }
    Ok(stores)
}

/// The box a kernel's plain entry is written with.
pub struct ChosenBox {
    /// The lane axis the box runs along.
    pub axis: usize,
    /// How many lanes the box holds.
    pub lanes: i64,
    /// How many flat lanes apart two neighbouring box lanes are.
    pub stride: i64,
}

/// Whether the kernel can be written boxed: every commit is a scalar write of the lane's own column, and every node is local to
/// its lane (no read of a per-lane input but a scalar one, no loop that exits, counts, commits or nests, no carried value but a
/// scalar real, no dispatch or random draw).
fn boxable(ctx: &Ctx, sinks: &[u32]) -> Outcome<bool> {
    for &sink in sinks {
        let commit = is_a(ctx, sink, "Assign") || is_update_of(ctx, sink, "store")?;
        if !commit
            || ctx.field(sink, "index")? != Val::None
            || !ctx.nodes[sink as usize].shape.is_empty()
        {
            return Ok(false);
        }
    }
    for node in reachable(ctx, sinks)? {
        if ["Dispatch", "TapeRead", "LoopCount"]
            .iter()
            .any(|class| is_a(ctx, node, class))
            || in_kinds(ctx, node, |kind| konst("RANDOM_OPS").contains(&kind))?
            || (is_a(ctx, node, "Leaf") && !lane_local_leaf(ctx, node)?)
            || (is_a(ctx, node, "Loop") && !lane_local_loop(ctx, node)?)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether a leaf is one a box can share: not a per-lane plane (an input holding a value of rank one or more, a mutable)
/// or a terminated flag.
fn lane_local_leaf(ctx: &Ctx, leaf: u32) -> Outcome<bool> {
    let role = ctx.str_field(leaf, "role")?;
    let value_plane = role == "plane_in" && !ctx.nodes[leaf as usize].shape.is_empty();
    Ok(!(value_plane || matches!(role.as_str(), "mutable" | "terminated")))
}

/// Whether a loop is one a box can fuse: it runs a fixed count, commits nothing, holds no loop inside and carries scalar reals.
fn lane_local_loop(ctx: &Ctx, lp: u32) -> Outcome<bool> {
    if ctx.field(lp, "exit_cond")? != Val::None
        || ctx.field(lp, "count")? != Val::None
        || !refs(ctx, lp, "body_sinks")?.is_empty()
    {
        return Ok(false);
    }
    for inner in ctx.body_nodes(lp)? {
        if is_a(ctx, inner, "Loop") {
            return Ok(false);
        }
    }
    for carry in refs(ctx, lp, "carries")? {
        let node = &ctx.nodes[carry as usize];
        if !node.shape.is_empty() || !matches!(node.dtype.as_str(), "f64" | "f32") {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The box for a kernel whose sinks are `sinks`, or none. `card_lanes` is the host's lanes per thread for the scalar type and
/// `group` the lanes the launch compacts by. A box needs an axis other than the fastest along which a plane is read invariantly,
/// room for at least two lanes that divide the axis, and a group the box does not straddle.
pub fn choose_box(
    ctx: &Ctx,
    sinks: &[u32],
    group: i64,
    card_lanes: i64,
) -> Outcome<Option<ChosenBox>> {
    if !boxable(ctx, sinks)? {
        return Ok(None);
    }
    let Some(found) = box_candidate(ctx, sinks)? else {
        return Ok(None);
    };
    let lanes = found
        .extent
        .map_or(card_lanes, |e| card_lanes.min(e as i64));
    let stride = found.stride as i64;
    let divides_axis = found.extent.is_none_or(|e| e % i128::from(lanes) == 0);
    if lanes < 2 || !divides_axis || (group > 1 && group % (lanes * stride) != 0) {
        return Ok(None);
    }
    Ok(Some(ChosenBox {
        axis: found.axis,
        lanes,
        stride,
    }))
}

/// The sinks of a boxed unit, in the caller's context; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_box_boxed_sinks(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    axis: usize,
    lanes: usize,
    stride: i64,
    buf: *mut u32,
    cap: usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |ctx| {
        if nsinks > 0 && sinks.is_null() {
            return Err(Fault::BadArg("sinks is NULL".into()));
        }
        let given = if nsinks == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        let expected = nsinks * lanes;
        if cap < expected || (buf.is_null() && expected > 0) {
            // nothing is made until the caller's buffer can hold the answer
            list_out(std::ptr::null_mut(), 0, count, &vec![0; expected])?;
        }
        let made = boxed_sinks(
            ctx,
            given,
            BoxSpec {
                axis,
                lanes,
                stride,
            },
        )?;
        list_out(buf, cap, count, &made)
    })
}
