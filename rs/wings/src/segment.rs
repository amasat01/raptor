//! The segmented split: a kernel with one dispatch that splits by branch becomes one kernel per branch, each with the dispatch
//! replaced by its own branch, plus (for a selector the kernel itself never writes) one kernel that writes each sample's class.
//! A unit whose selector can change as the kernel runs keeps each commit only where the selector picks its branch, and records
//! the class and whether the sample went stale. This module makes the IR of those kernels; the caller takes their walks.
use crate::planes::{type_parts, NodeType};
use crate::{is_a, walk_order, R};
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::{ttype_val, Val};

/// The dispatch a kernel is split by.
pub struct SegmentedDispatch {
    /// The dispatch node.
    pub node: u32,
    /// Its place among the walk's dispatches: the name of its group is `dispatch<index>`.
    pub index: usize,
    /// How many branches it has.
    pub branches: usize,
}

/// How a dispatch group is lowered, decided by the feathers the kind of its node states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strategy {
    /// Every branch is evaluated and the result selected.
    Predicated,
    /// A plain dispatch: a switch.
    Switch,
    /// The branches are split apart, one unit each.
    Segmented,
    /// The branches are counted and the form is chosen at run time.
    Auto,
}

/// The strategy of each dispatch group of the walk, by the group's name, in the walk's order.
pub fn strategies(ctx: &Ctx, walk: u32) -> R<Vec<(String, Strategy)>> {
    let order = walk_order(ctx, walk)?;
    let mut out = vec![];
    for (index, node) in ctx.dispatch_nodes(&order)?.into_iter().enumerate() {
        let strategy = if ctx.has_feather(node, "splits_by_branch")? {
            Strategy::Segmented
        } else if ctx.has_feather(node, "counts_branch_mix")? {
            Strategy::Auto
        } else if ctx.has_feather(node, "evaluates_all_branches")? {
            Strategy::Predicated
        } else {
            Strategy::Switch
        };
        out.push((group_name(index), strategy));
    }
    Ok(out)
}

/// The name of the dispatch group at `index`.
pub fn group_name(index: usize) -> String {
    format!("dispatch{index}")
}

/// The name of the plane that holds a group's run offsets.
pub fn offsets_name(group: &str) -> String {
    format!("{group}_offsets")
}

/// The name of the plane a split unit writes each sample's branch class to.
pub fn class_name(group: &str) -> String {
    format!("{group}_class")
}

/// The name of the plane a split unit marks each sample on whose selector no longer picks the unit's branch.
pub fn stale_name(group: &str) -> String {
    format!("{group}_stale")
}

/// The dispatch of the walk that splits by branch, or none. A kernel with two of them, or whose one sits only inside a loop body,
/// is refused.
pub fn segmented_dispatch(ctx: &Ctx, walk: u32) -> R<Option<SegmentedDispatch>> {
    let order = walk_order(ctx, walk)?;
    let mut found = vec![];
    for (index, node) in ctx.dispatch_nodes(&order)?.into_iter().enumerate() {
        if ctx.has_feather(node, "splits_by_branch")? {
            if !order.contains(&node) {
                return refuse("a segmented dispatch inside a loop body");
            }
            found.push((index, node));
        }
    }
    match found[..] {
        [] => Ok(None),
        [(index, node)] => Ok(Some(SegmentedDispatch {
            node,
            index,
            branches: ctx.nodes[node as usize].ops.len() - 1,
        })),
        _ => refuse(format!(
            "{} segmented dispatches in one kernel",
            found.len()
        )),
    }
}

/// The nodes the split makes, in the caller's context.
struct Maker<'a> {
    /// The context the nodes are made in.
    ctx: &'a mut Ctx,
}

impl Maker<'_> {
    /// A node of the class built from its parameters.
    fn build(&mut self, class: &str, given: Vec<(&str, Val)>) -> R<u32> {
        let index = self.ctx.table.find(class);
        let index = index.ok_or_else(|| Fault::Refused(format!("no {class} class")))?;
        let given: Vec<(String, Val)> =
            given.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        self.ctx.new_from(index, &given)
    }

    /// A constant of the type.
    fn constant(&mut self, literal: Val, ty: &NodeType) -> R<u32> {
        let tt = ttype_val(&ty.dtype, &ty.shape, ty.tag.as_deref());
        self.build("Const", vec![("literal", literal), ("ttype", tt)])
    }

    /// An integer constant.
    fn int(&mut self, value: i64) -> R<u32> {
        let ty = NodeType {
            dtype: "i32".into(),
            shape: vec![],
            tag: None,
        };
        self.constant(Val::Int(i128::from(value)), &ty)
    }

    /// An op of the kind over the operands, its type inferred.
    fn op(&mut self, kind: &str, ops: &[u32], literal: &Val) -> R<u32> {
        self.ctx.make_op(kind, ops, literal)
    }

    /// A choice between two values of the given type.
    fn select(&mut self, cond: u32, on_true: u32, on_false: u32, ty: &NodeType) -> R<u32> {
        let tt = ttype_val(&ty.dtype, &ty.shape, ty.tag.as_deref());
        let given = vec![
            ("cond", Val::Node(cond)),
            ("on_true", Val::Node(on_true)),
            ("on_false", Val::Node(on_false)),
            ("ttype", tt),
        ];
        self.build("Select", given)
    }

    /// A write of the value to the named plane.
    fn assign(&mut self, name: &str, value: u32, ty: &NodeType) -> R<u32> {
        let tt = ttype_val(&ty.dtype, &ty.shape, ty.tag.as_deref());
        let given = vec![
            ("name", Val::Str(name.into())),
            ("value", Val::Node(value)),
            ("ttype", tt),
        ];
        self.build("Assign", given)
    }

    /// The selector clamped into the range of branches.
    fn clamped(&mut self, selector: u32, branches: usize) -> R<u32> {
        let ty = NodeType {
            dtype: "i32".into(),
            shape: vec![],
            tag: None,
        };
        let (zero, last) = (self.int(0)?, self.int(branches as i64 - 1)?);
        let over = self.op("gt", &[selector, last], &Val::None)?;
        let upper = self.select(over, last, selector, &ty)?;
        let under = self.op("lt", &[selector, zero], &Val::None)?;
        self.select(under, zero, upper, &ty)
    }

    /// The sink kept only where `mine` holds: a commit keeps the value it had elsewhere, a finish does not fire elsewhere.
    fn guarded(&mut self, sink: u32, mine: u32) -> R<u32> {
        let ty = type_parts(self.ctx, sink);
        let value = self.ctx.nodes[sink as usize].ops[0];
        let name = self.ctx.str_field(sink, "name")?;
        if is_a(self.ctx, sink, "Finish") {
            let fires = self.op("land", &[mine, value], &Val::None)?;
            return self.build(
                "Finish",
                vec![("name", Val::Str(name)), ("value", Val::Node(fires))],
            );
        }
        if is_a(self.ctx, sink, "Assign") {
            let kind = ("kind", Val::Str("prior_read".into()));
            let tt = ttype_val(&ty.dtype, &ty.shape, ty.tag.as_deref());
            let given = vec![
                kind,
                ("role", Val::Str("mutable".into())),
                ("name", Val::Str(name.clone())),
                ("ttype", tt),
            ];
            let old = self.build("Leaf", given)?;
            let kept = self.select(mine, value, old, &ty)?;
            return self.assign(&name, kept, &ty);
        }
        if is_a(self.ctx, sink, "AccumWrite") {
            let scalar = NodeType {
                dtype: ty.dtype.clone(),
                shape: vec![],
                tag: None,
            };
            let mut zero = self.constant(Val::Float(0.0), &scalar)?;
            if !ty.shape.is_empty() {
                let extents =
                    Val::Tuple(ty.shape.iter().map(|&d| Val::Int(i128::from(d))).collect());
                zero = self.op("splat", &[zero], &extents)?;
            }
            let kept = self.select(mine, value, zero, &ty)?;
            let tt = ttype_val(&ty.dtype, &ty.shape, ty.tag.as_deref());
            let index = self.ctx.nodes[sink as usize]
                .ops
                .get(1)
                .map_or(Val::None, |&i| Val::Node(i));
            let contract = self.ctx.str_field(sink, "contract")?;
            let given = vec![
                ("name", Val::Str(name)),
                ("value", Val::Node(kept)),
                ("index", index),
                ("ttype", tt),
                ("contract", Val::Str(contract)),
            ];
            return self.build("AccumWrite", given);
        }
        refuse(format!(
            "a segmented unit guards each commit by the selector, and the sink {name} has no old value to keep"
        ))
    }
}

/// The sinks of the unit for branch `branch`: the kernel's sinks with the dispatch replaced by that branch. With `guard` (a
/// selector the kernel writes) every commit is kept only where the selector picks the branch, and the class and stale planes are
/// written too.
pub fn branch_sinks(
    ctx: &mut Ctx,
    sinks: &[u32],
    split: &SegmentedDispatch,
    branch: usize,
    guard: bool,
) -> R<Vec<u32>> {
    let chosen = ctx.nodes[split.node as usize].ops[1 + branch];
    let selector = ctx.nodes[split.node as usize].ops[0];
    let swapped = ctx.substitute(sinks, &[(split.node, chosen)].into_iter().collect())?;
    if !guard {
        return Ok(swapped);
    }
    let mut maker = Maker { ctx };
    let selected = maker.clamped(selector, split.branches)?;
    let this = maker.int(branch as i64)?;
    let mine = maker.op("eq", &[selected, this], &Val::None)?;
    let mut out = vec![];
    for sink in swapped {
        out.push(maker.guarded(sink, mine)?);
    }
    let group = group_name(split.index);
    let int = NodeType {
        dtype: "i32".into(),
        shape: vec![],
        tag: None,
    };
    out.push(maker.assign(&class_name(&group), selected, &int)?);
    let other = maker.int(branch as i64)?;
    let moved = maker.op("ne", &[selected, other], &Val::None)?;
    let flag = NodeType {
        dtype: "bool".into(),
        shape: vec![],
        tag: None,
    };
    out.push(maker.assign(&stale_name(&group), moved, &flag)?);
    Ok(out)
}

/// The sinks of the unit that writes each sample's class, for a selector no unit writes.
pub fn class_sinks(ctx: &mut Ctx, split: &SegmentedDispatch) -> R<Vec<u32>> {
    let selector = ctx.nodes[split.node as usize].ops[0];
    let mut maker = Maker { ctx };
    let selected = maker.clamped(selector, split.branches)?;
    let int = NodeType {
        dtype: "i32".into(),
        shape: vec![],
        tag: None,
    };
    let name = class_name(&group_name(split.index));
    Ok(vec![maker.assign(&name, selected, &int)?])
}

/// The strategy words of the walk's dispatch groups in the walk's order, as the names protocol.
///
/// # Safety
/// `buf` is valid for `cap` bytes; `len` and `count` for one write each.
#[no_mangle]
pub unsafe extern "C" fn rw_walk_strategies(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut std::ffi::c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> std::ffi::c_int {
    raptor_ir::door::guard(ctx, |c| {
        let words: Vec<&str> = strategies(c, walk)?
            .into_iter()
            .map(|(_, s)| match s {
                Strategy::Predicated => "predicated",
                Strategy::Switch => "switch",
                Strategy::Segmented => "segmented",
                Strategy::Auto => "auto",
            })
            .collect();
        raptor_ir::consts::put_names(&words, buf, cap, len, count)
    })
}

/// The strategies a dispatch can be pinned to: `switch` (the core kind itself) and one for each wing kind that refines `dispatch`
/// (`<strategy>_dispatch`), in the order the wing manifests declare them.
pub fn dispatch_strategies() -> Vec<String> {
    let mut out = vec!["switch".to_string()];
    for (_, manifest) in crate::declare::MANIFESTS {
        let Ok(parsed) = rcore::json::parse(manifest) else {
            continue;
        };
        for row in parsed.get("kinds").map_or(&[][..], rcore::json::J::arr) {
            let refines = row.get("refines").and_then(rcore::json::J::str);
            let name = row.get("name").and_then(rcore::json::J::str).unwrap_or("");
            if let (Some("dispatch"), Some(word)) = (refines, name.strip_suffix("_dispatch")) {
                out.push(word.to_string());
            }
        }
    }
    out
}

/// The dispatch strategies, as the names protocol; see `dispatch_strategies`.
///
/// # Safety
/// `buf` is valid for `cap` bytes; `len` and `count` for one write each.
#[no_mangle]
pub unsafe extern "C" fn rw_dispatch_strategies(
    ctx: *mut Ctx,
    buf: *mut std::ffi::c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> std::ffi::c_int {
    raptor_ir::door::guard(ctx, |_| {
        raptor_ir::consts::put_names(&dispatch_strategies(), buf, cap, len, count)
    })
}
