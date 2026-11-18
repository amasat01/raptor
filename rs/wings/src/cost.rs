//! The operations a set of nodes runs once, counted in the IR by class (arithmetic or library call, 32 or 64 bits, loads,
//! gathers). A step of a finishing kernel is priced by its 64-bit total.
use super::{is_a, konst, library_kind, op_kind, raw_index, R};
use crate::construct::refs;
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::typing::all_kinds;
use raptor_ir::value::Val;
use std::collections::{HashMap, HashSet};

/// Ops that only move or re-label values.
const FREE: [&str; 10] = [
    "component",
    "component_at",
    "set_component_at",
    "add_at",
    "vec",
    "splat",
    "transpose",
    "as_pure",
    "as_vec3",
    "quat_conj",
];
/// The operations of one lane of the ray ops, by kind.
const RAYS: [(&str, f64); 3] = [("ray_tri", 70.0), ("ray_aabb", 60.0), ("ray_sphere", 45.0)];
/// The operations of the quaternion ops, by kind.
const QUATS: [(&str, f64); 3] = [
    ("quat_mul", 28.0),
    ("quat_rotate", 56.0),
    ("quat_recip", 8.0),
];

/// The number of elements of the node's value.
fn elements(c: &Ctx, h: u32) -> f64 {
    c.nodes[h as usize].shape.iter().product::<i64>() as f64
}

/// Whether the node computes in 64 bits: a float result, or an operand that is one, and not an integer op.
fn wide(c: &Ctx, h: u32) -> R<bool> {
    let node = &c.nodes[h as usize];
    let int_dtype = konst("INT_PLANE_DTYPES").contains(&node.dtype.as_str());
    let int_kind = super::in_kinds(c, h, |k| {
        konst("INT_BINARY").contains(&k) || konst("INT_UNARY").contains(&k)
    })?;
    if int_dtype || int_kind {
        return Ok(false);
    }
    Ok(node.dtype == "f64" || node.ops.iter().any(|&o| c.nodes[o as usize].dtype == "f64"))
}

/// The operands' shapes the linear-algebra count reads: the matrix side and the right-hand side's columns.
fn geometry(c: &Ctx, h: u32, kind: &str) -> (f64, f64) {
    let ops = &c.nodes[h as usize].ops;
    let n = *c.nodes[ops[0] as usize].shape.last().unwrap_or(&1) as f64;
    let rhs = if konst("LINALG_CONSUMERS").contains(&kind) {
        &c.nodes[ops[1] as usize].shape
    } else {
        &vec![]
    };
    (n, if rhs.len() == 2 { rhs[1] as f64 } else { 1.0 })
}

/// The operations of a linear-algebra op.
fn linalg_ops(c: &Ctx, h: u32, kind: &str) -> f64 {
    let (n, k) = geometry(c, h, kind);
    let m = c.nodes[c.nodes[h as usize].ops[0] as usize].shape[0] as f64;
    let sweeps = match n as i64 {
        ..=2 => 2.0,
        3..=4 => 6.0,
        5..=8 => 8.0,
        9..=16 => 12.0,
        _ => 15.0,
    };
    match kind {
        "qr" => m * n * n,
        "qr_solve" => m * n * k + n * n * k,
        "lu" => n.powi(3) / 3.0,
        "chol" => n.powi(3) / 6.0,
        "eigh" => sweeps * n * (n - 1.0) / 2.0 * (6.0 * n + 10.0),
        _ if konst("LINALG_CONSUMERS").contains(&kind) => {
            n * n * k
                / if matches!(kind, "tri_lower" | "tri_upper_t") {
                    2.0
                } else {
                    1.0
                }
        }
        _ => 0.0,
    }
}

/// The operations an op of this kind runs, in whichever class.
fn op_ops(c: &Ctx, h: u32, kind: &str) -> f64 {
    let ops = &c.nodes[h as usize].ops;
    let table = |t: &[(&str, f64)]| t.iter().find(|x| x.0 == kind).map(|x| x.1);
    if [
        "LINALG_FACTORISATIONS",
        "LINALG_CONSUMERS",
        "LINALG_PROJECTIONS",
    ]
    .iter()
    .any(|g| konst(g).contains(&kind))
    {
        return linalg_ops(c, h, kind);
    }
    if let Some(n) = table(&RAYS).or_else(|| table(&QUATS)) {
        return n;
    }
    if library_kind(kind) {
        return elements(c, h);
    }
    let shape = |i: usize| &c.nodes[ops[i] as usize].shape;
    match kind {
        "dot" | "norm" => 2.0 * elements(c, ops[0]) + f64::from(u8::from(kind == "norm")),
        "sum" => elements(c, ops[0]),
        "cross" => 9.0,
        "outer" => elements(c, h),
        "mv" | "mm" => {
            let cols = if kind == "mm" {
                *shape(1).last().unwrap_or(&1)
            } else {
                1
            };
            2.0 * shape(0).iter().product::<i64>() as f64 * cols as f64
        }
        _ => elements(c, h).max(1.0),
    }
}

/// Slot of the loads in a node's six counts; the slots before it are the arithmetic and library-call classes.
const LOADS: usize = 4;
/// Slot of the gathers in a node's six counts.
const GATHERS: usize = 5;
/// The six counts of a set of nodes: arithmetic and library calls in 32 and 64 bits (see `op_slot`), loads, gathers.
pub(crate) type Counts = [f64; 6];
/// Roles of a leaf that is a scalar parameter or a count, not a plane read.
const NOT_A_PLANE: [&str; 2] = ["uniform", "nsamples"];

/// The iterations a loop is counted at: its trip count, and one when the bound is a run-time value with no maximum.
fn trips(c: &Ctx, h: u32) -> R<f64> {
    if matches!(c.field(h, "dynamic")?, Val::Str(word) if word == "open") {
        return Ok(1.0);
    }
    match c.field(h, "trip")? {
        Val::Int(n) => u32::try_from(n).map(f64::from).or_else(|_| {
            refuse(format!(
                "a loop's trip {n} is not a count that fits 32 bits"
            ))
        }),
        _ => refuse("a loop's trip is not a whole number"),
    }
}

/// The class and amount of the work an op, select or dispatch node adds itself: the index into the first four counts
/// (arithmetic or library call, 32 or 64 bits) and the operations it runs. None for a node that only moves values.
pub(crate) fn op_slot(c: &Ctx, h: u32) -> R<Option<(usize, f64)>> {
    if is_a(c, h, "Select") || is_a(c, h, "Dispatch") {
        return Ok(Some((usize::from(wide(c, h)?), elements(c, h).max(1.0))));
    }
    let Some(kind) = op_kind(c, h)? else {
        return Ok(None);
    };
    let node = &c.nodes[h as usize];
    let widens =
        kind == "cast" && node.dtype == "i32" && c.nodes[node.ops[0] as usize].dtype == "int32";
    let free =
        FREE.contains(&kind.as_str()) || konst("LINALG_PROJECTIONS").contains(&kind.as_str());
    if !all_kinds().contains(&kind.as_str()) || free || widens {
        return Ok(None);
    }
    let class = 2 * usize::from(library_kind(&kind)) + usize::from(wide(c, h)?);
    Ok(Some((class, op_ops(c, h, &kind))))
}

/// The six counts the node adds itself, not counting its operands; a loop adds its body once per trip.
fn own_counts(c: &Ctx, h: u32) -> R<Counts> {
    let mut out = [0.0; 6];
    if is_a(c, h, "Leaf") {
        if !NOT_A_PLANE.contains(&c.str_field(h, "role")?.as_str()) {
            out[LOADS] = elements(c, h);
        }
    } else if is_a(c, h, "At") {
        out[GATHERS] = elements(c, h).max(1.0);
    } else if is_a(c, h, "Loop") {
        let trips = trips(c, h)?;
        for (slot, body) in out.iter_mut().zip(work(c, &c.body_nodes(h)?)?) {
            *slot = trips * body;
        }
    } else if let Some((class, amount)) = op_slot(c, h)? {
        out[class] = amount;
    }
    Ok(out)
}

/// The six counts of `nodes`, each counted once (a constant counts nothing).
pub(crate) fn work(c: &Ctx, nodes: &[u32]) -> R<Counts> {
    let mut total = [0.0; 6];
    for &h in nodes {
        c.node(h)?;
        if !is_a(c, h, "Const") {
            for (sum, own) in total.iter_mut().zip(own_counts(c, h)?) {
                *sum += own;
            }
        }
    }
    Ok(total)
}

/// The 64-bit floating-point operations `nodes` run once: the 64-bit arithmetic and library calls of the nodes, a loop body
/// counted once per trip (a library call counts once per element).
///
/// # Errors
/// Refuses a handle the context does not hold and a loop whose trip count is not a count.
pub fn fp64_ops(c: &Ctx, nodes: &[u32]) -> R<i64> {
    let counts = work(c, nodes)?;
    Ok((counts[1] + counts[3]) as i64)
}

/// The 64-bit floating-point operations of one trip of the walk's one top-level loop (a finishing kernel's step); none when the
/// walk has no such loop or more than one.
pub fn step_fp64_ops(c: &Ctx, walk: u32) -> R<Option<i64>> {
    let order = c.walk(walk)?.order.clone();
    let mut loops = order.iter().copied().filter(|&n| is_a(c, n, "Loop"));
    match (loops.next(), loops.next()) {
        (Some(lp), None) => fp64_ops(c, &c.body_nodes(lp)?).map(Some),
        _ => Ok(None),
    }
}

/// The longest dependent path to a value, in operations, per class: arithmetic 32, arithmetic 64, library call 32, library
/// call 64 (the order of `op_slot`'s class).
type Path = [f64; 4];
/// The path of a value that depends on no operation.
const NO_PATH: Path = [0.0; 4];

/// The sum of a path's classes: the length by which paths are ranked.
fn length(path: &Path) -> f64 {
    path.iter().sum()
}

/// The nodes `h` waits for: its operands, or for a gather its index alone (the indexed plane is not read).
fn dependencies(c: &Ctx, h: u32) -> Vec<u32> {
    if is_a(c, h, "At") {
        vec![raw_index(c, h)]
    } else {
        c.nodes[h as usize].ops.clone()
    }
}

/// The path to node `h` given the paths of what it waits for, or None when `carried` and no carried value reaches it.
fn path_through(c: &Ctx, h: u32, reached: &[Option<Path>], carried: bool) -> R<Option<Path>> {
    let mut best = if carried { None } else { Some(NO_PATH) };
    for path in reached.iter().flatten() {
        // the first of equally long paths wins, so the class split is deterministic
        if best.as_ref().is_none_or(|b| length(path) > length(b)) {
            best = Some(*path);
        }
    }
    let Some(mut path) = best else {
        return Ok(None);
    };
    if is_a(c, h, "Loop") {
        let boundary: HashSet<u32> = c.nodes[h as usize].ops.iter().copied().collect();
        let body = critical_path(c, &refs(c, h, "nexts")?, &boundary, true)?;
        let trips = trips(c, h)?;
        for (sum, step) in path.iter_mut().zip(body) {
            *sum += trips * step;
        }
    } else if let Some((class, _)) = op_slot(c, h)? {
        path[class] += 1.0;
    }
    Ok(Some(path))
}

/// The critical path to the longest of `roots`: nodes in `zero` and the loop placeholders and constants cost nothing. With
/// `carried`, only the paths that start at a loop-carried read count (what one iteration adds to the recurrence): work that
/// does not depend on a carry runs alongside it.
fn critical_path(c: &Ctx, roots: &[u32], zero: &HashSet<u32>, carried: bool) -> R<Path> {
    let mut memo: HashMap<u32, Option<Path>> = HashMap::new();
    let mut stack: Vec<(u32, bool)> = roots.iter().map(|&r| (r, false)).collect();
    while let Some((h, expanded)) = stack.pop() {
        if memo.contains_key(&h) {
            continue;
        }
        if carried && is_a(c, h, "LoopCarry") {
            memo.insert(h, Some(NO_PATH));
        } else if zero.contains(&h)
            || ["Leaf", "Const", "LoopIndex", "LoopCarry"]
                .iter()
                .any(|class| is_a(c, h, class))
        {
            memo.insert(h, if carried { None } else { Some(NO_PATH) });
        } else if expanded {
            let reached: Vec<Option<Path>> = dependencies(c, h).iter().map(|d| memo[d]).collect();
            memo.insert(h, path_through(c, h, &reached, carried)?);
        } else {
            stack.push((h, true));
            let waiting = dependencies(c, h)
                .into_iter()
                .filter(|d| !memo.contains_key(d));
            stack.extend(waiting.map(|d| (d, false)));
        }
    }
    let mut longest = NO_PATH;
    let mut seen_any = false;
    for path in roots.iter().filter_map(|r| memo[r]) {
        if !seen_any || length(&path) > length(&longest) {
            longest = path;
            seen_any = true;
        }
    }
    Ok(longest)
}

/// The nodes `root` evaluates, each once, leaving out `skip` and what only `skip` reaches. A gather reaches its index, never its
/// plane.
fn closure(c: &Ctx, root: u32, skip: &HashSet<u32>) -> Vec<u32> {
    let mut seen = skip.clone();
    let (mut out, mut stack) = (vec![], vec![root]);
    while let Some(h) = stack.pop() {
        if seen.insert(h) {
            out.push(h);
            stack.extend(dependencies(c, h));
        }
    }
    out
}

/// What one branch of a dispatch group costs one sample that takes it, by class. The counts are operations; the chains are
/// the longest dependent paths in operations (independent work adds to the counts, not to the path). Whatever the selector
/// evaluates is left out, and a value two branches share is counted in each: a split unit runs only its own branch.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchCost {
    /// Arithmetic operations on 32-bit values (integer work included).
    pub arith32: f64,
    /// Arithmetic operations on 64-bit floats.
    pub arith64: f64,
    /// Library calls (transcendentals, random draws) on 32-bit values.
    pub special32: f64,
    /// Library calls on 64-bit floats.
    pub special64: f64,
    /// Plane elements read.
    pub loads: f64,
    /// Indexed reads.
    pub gathers: f64,
    /// Critical path of the 32-bit arithmetic.
    pub chain32: f64,
    /// Critical path of the 64-bit arithmetic.
    pub chain64: f64,
    /// Critical path of the 32-bit library calls.
    pub chain_special32: f64,
    /// Critical path of the 64-bit library calls.
    pub chain_special64: f64,
    /// Plane elements the branch writes; None when no sink takes the group's value directly.
    pub stores: Option<f64>,
}

/// One dispatch group: the cost of each branch and the planes its selector reads.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupCost {
    /// One cost per branch, in branch order.
    pub branches: Vec<BranchCost>,
    /// The sorted `(role, name)` slots of the selector's reads.
    pub selector_reads: Vec<(String, String)>,
}

/// The cost of the branch `branch` of a group whose selector evaluates `zero`; `stores` as `BranchCost::stores`.
fn branch_cost(c: &Ctx, branch: u32, zero: &HashSet<u32>, stores: Option<f64>) -> R<BranchCost> {
    let [arith32, arith64, special32, special64, loads, gathers] =
        work(c, &closure(c, branch, zero))?;
    let [chain32, chain64, chain_special32, chain_special64] =
        critical_path(c, &[branch], zero, false)?;
    Ok(BranchCost {
        arith32,
        arith64,
        special32,
        special64,
        loads,
        gathers,
        chain32,
        chain64,
        chain_special32,
        chain_special64,
        stores,
    })
}

/// The cost of the group the dispatch node `group` heads; `written` is whether a sink of the walk takes its value.
fn group_cost(c: &Ctx, group: u32, written: bool) -> R<GroupCost> {
    let (selector, branches) = c.nodes[group as usize]
        .ops
        .split_first()
        .map_or_else(|| refuse("a dispatch has no selector"), Ok)?;
    let zero: HashSet<u32> = closure(c, *selector, &HashSet::new()).into_iter().collect();
    let branches = branches
        .iter()
        .map(|&b| branch_cost(c, b, &zero, written.then(|| elements(c, b))))
        .collect::<R<Vec<_>>>()?;
    Ok(GroupCost {
        branches,
        selector_reads: c.selector_reads(group)?.into_iter().collect(),
    })
}

/// The cost of every dispatch group of a walk, in the order of the walk's dispatch listing (the top-level groups, then those
/// inside loop bodies); `order` is the walk's node order.
///
/// # Errors
/// Refuses a handle the context does not hold and a loop whose trip count is not a whole number.
pub fn dispatch_costs(c: &Ctx, order: &[u32]) -> R<Vec<GroupCost>> {
    let sink_values: HashSet<u32> = order
        .iter()
        .filter(|&&h| is_a(c, h, "Sink"))
        .map(|&h| c.nodes[h as usize].ops[0])
        .collect();
    c.dispatch_nodes(order)?
        .into_iter()
        .map(|group| group_cost(c, group, sink_values.contains(&group)))
        .collect()
}

#[cfg(test)]
mod tests {
    //! The counts, the critical path and the stores of a branch, each on a small hand-made kernel; every expected vector is
    //! counted by hand from the rules stated in the type docs above (`work`: arithmetic 32, 64, library call 32, 64, loads,
    //! gathers; `Path`: the same four classes).
    use super::*;

    /// A context with the node builders of the tests.
    struct Graph {
        c: Ctx,
    }

    impl Graph {
        fn new() -> Graph {
            Graph {
                c: Ctx::new().unwrap(),
            }
        }

        /// A per-sample plane read of the element type.
        fn plane(&mut self, name: &str, dtype: &str) -> u32 {
            let fields = format!(
                "(('kind', 'vocab_read'), ('role', 'per_sample'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))"
            );
            self.c.make("Leaf", dtype, &[], None, &[], &fields).unwrap()
        }

        fn constant(&mut self, literal: &str, dtype: &str) -> u32 {
            let fields = format!("(('literal', {literal}),)");
            self.c
                .make("Const", dtype, &[], None, &[], &fields)
                .unwrap()
        }

        fn op(&mut self, kind: &str, dtype: &str, ops: &[u32]) -> u32 {
            let fields = format!("(('kind', '{kind}'), ('literal', None))");
            self.c.make("Op", dtype, &[], None, ops, &fields).unwrap()
        }

        fn counts(&self, nodes: &[u32]) -> Counts {
            work(&self.c, nodes).unwrap()
        }

        fn path(&self, root: u32) -> Path {
            critical_path(&self.c, &[root], &HashSet::new(), false).unwrap()
        }
    }

    #[test]
    fn a_plain_kernel_counts_its_arithmetic_and_loads_and_chains_them_in_sequence() {
        // y = x * x + 1.0: two 64-bit operations, one plane read, and the add waits for the multiply
        let mut g = Graph::new();
        let x = g.plane("x", "f64");
        let one = g.constant("1.0", "f64");
        let square = g.op("mul", "f64", &[x, x]);
        let sum = g.op("add", "f64", &[square, one]);
        assert_eq!(
            g.counts(&[x, one, square, sum]),
            [0.0, 2.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(g.path(sum), [0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn work_on_32_bit_values_lands_in_the_32_bit_fields() {
        let mut g = Graph::new();
        let a = g.plane("a", "f32");
        let (product, sine) = (g.op("mul", "f32", &[a, a]), g.op("sin", "f32", &[a]));
        assert_eq!(g.counts(&[a, product]), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(g.counts(&[a, sine]), [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_library_call_lands_in_the_special_chain_beside_the_arithmetic_path() {
        // log(x * x + 1.0) * b: the arithmetic path is multiply, add, multiply = 3 and the one library call is on it
        let mut g = Graph::new();
        let (x, b) = (g.plane("x", "f64"), g.plane("b", "f64"));
        let one = g.constant("1.0", "f64");
        let square = g.op("mul", "f64", &[x, x]);
        let shifted = g.op("add", "f64", &[square, one]);
        let log = g.op("log", "f64", &[shifted]);
        let scaled = g.op("mul", "f64", &[log, b]);
        assert_eq!(g.path(scaled), [0.0, 3.0, 0.0, 1.0]);
        assert_eq!(
            g.counts(&[x, b, one, square, shifted, log, scaled]),
            [0.0, 3.0, 0.0, 1.0, 2.0, 0.0]
        );
    }

    #[test]
    fn a_gather_waits_for_its_index_and_is_one_step_of_its_own_width() {
        // table.at(lane + 1): the path is the 32-bit index add, then the read as one step of its own 64-bit width; the plane,
        // a leaf, adds nothing (the recorded model gives a lone gather branch a 64-bit chain of 1 and one gather)
        let mut g = Graph::new();
        let lane =
            g.c.make("SampleIndex", "i32", &[], None, &[], "()")
                .unwrap();
        let (one, table) = (g.constant("1", "i32"), g.plane("table", "f64"));
        let index = g.op("add", "i32", &[lane, one]);
        let read =
            g.c.make("At", "f64", &[], None, &[table, index], "()")
                .unwrap();
        assert_eq!(g.path(read), [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(g.counts(&[table, read]), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    /// The branch costs of the one dispatch group of `x * x + 1.0` and `x * x + 2.0`, the group's value kept by a sink or
    /// added to a constant first.
    fn shared_branch_stores(kept_directly: bool) -> Vec<Option<f64>> {
        let mut g = Graph::new();
        let (k, x) = (g.plane("k", "i32"), g.plane("x", "f64"));
        let (one, two) = (g.constant("1.0", "f64"), g.constant("2.0", "f64"));
        let square = g.op("mul", "f64", &[x, x]);
        let (a, b) = (
            g.op("add", "f64", &[square, one]),
            g.op("add", "f64", &[square, two]),
        );
        let group =
            g.c.make("Dispatch", "f64", &[], None, &[k, a, b], "()")
                .unwrap();
        let value = if kept_directly {
            group
        } else {
            g.op("add", "f64", &[group, one])
        };
        let fields = "(('kind', 'assign'), ('name', 'y'), ('role', 'mutable'))";
        let sink =
            g.c.make("Assign", "f64", &[], None, &[value], fields)
                .unwrap();
        let mut order = vec![k, x, one, two, square, a, b, group];
        order.extend((value != group).then_some(value));
        order.push(sink);
        let costs = dispatch_costs(&g.c, &order).unwrap();
        costs[0].branches.iter().map(|b| b.stores).collect()
    }

    #[test]
    fn each_branch_of_a_group_a_sink_keeps_stores_the_value_it_hands_over() {
        assert_eq!(shared_branch_stores(true), [Some(1.0), Some(1.0)]);
    }

    #[test]
    fn a_group_no_sink_keeps_directly_states_no_store_count() {
        assert_eq!(shared_branch_stores(false), [None, None]);
    }
}
