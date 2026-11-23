//! Constructors of group kernels for tests and for the writers that consume them: the pieces (a scratch plane, a group
//! scope, a barrier, a group-wide reduce) and three reference kernels, `tile`, `rows` and `ring`, whose documents are stored
//! under `rs/support/tests/data/groups/` in the dump format.
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::{ttype_val, Val};

/// The text of a refusal, or of any other fault.
fn message(fault: Fault) -> String {
    match fault {
        Fault::Refused(m) | Fault::BadArg(m) => m,
        other => format!("{other:?}"),
    }
}

/// The node of class `class` built from constructor parameters, or the refusal text.
fn build(ctx: &mut Ctx, class: &str, given: Vec<(&str, Val)>) -> Result<u32, String> {
    let index = ctx.table.find(class).ok_or(format!("no class {class}"))?;
    let given: Vec<(String, Val)> = given.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    ctx.new_from(index, &given).map_err(message)
}

/// A text value.
fn text(s: &str) -> Val {
    Val::Str(s.to_string())
}

/// A tuple of node references.
fn nodes(list: &[u32]) -> Val {
    Val::Tuple(list.iter().map(|&n| Val::Node(n)).collect())
}

/// An axis list: pairs of a label and an extent, None for an extent known at run time.
pub fn axes(list: &[(&str, Option<i128>)]) -> Val {
    Val::Tuple(
        list.iter()
            .map(|(label, extent)| {
                Val::Tuple(vec![text(label), extent.map_or(Val::None, Val::Int)])
            })
            .collect(),
    )
}

/// The type value of a node.
fn type_of(ctx: &Ctx, node: u32) -> Val {
    let n = ctx.node(node).expect("a node of this context");
    ttype_val(&n.dtype, &n.shape, n.tag.as_deref())
}

/// A per-lane input plane `name` of element type `dtype`.
pub fn plane_in(ctx: &mut Ctx, name: &str, dtype: &str) -> Result<u32, String> {
    let given = vec![
        ("kind", text("vocab_read")),
        ("role", text("plane_in")),
        ("name", text(name)),
        ("ttype", ttype_val(dtype, &[], None)),
    ];
    build(ctx, "Leaf", given)
}

/// A scratch plane `name` of element type `dtype` over the one axis `rank` of `extent` elements.
pub fn scratch(ctx: &mut Ctx, name: &str, dtype: &str, extent: i128) -> Result<u32, String> {
    let given = vec![
        ("name", text(name)),
        ("ttype", ttype_val(dtype, &[], None)),
        ("axes", axes(&[("rank", Some(extent))])),
    ];
    build(ctx, "Scratch", given)
}

/// The lane's place in its group.
pub fn rank(ctx: &mut Ctx) -> Result<u32, String> {
    build(ctx, "GroupRank", vec![])
}

/// The number of lanes of the lane's group.
pub fn size(ctx: &mut Ctx) -> Result<u32, String> {
    build(ctx, "GroupSize", vec![])
}

/// A whole-number constant of type `dtype`.
pub fn whole(ctx: &mut Ctx, dtype: &str, value: i128) -> Result<u32, String> {
    let (dtype, literal) = (text(dtype), Val::Int(value));
    let (shape, tag) = (Val::Tuple(vec![]), Val::None);
    let ttype = Val::Rec(
        "TensorType".into(),
        vec![
            ("shape".into(), shape),
            ("dtype".into(), dtype),
            ("tag".into(), tag),
        ],
    );
    build(ctx, "Const", vec![("literal", literal), ("ttype", ttype)])
}

/// The operation `kind` over `ops`, with the type its operands give.
pub fn apply(ctx: &mut Ctx, kind: &str, ops: &[u32]) -> Result<u32, String> {
    ctx.make_op(kind, ops, &Val::None).map_err(message)
}

/// A read of `scratch` at `index` under the edge rule `edge`.
pub fn read(ctx: &mut Ctx, scratch: u32, index: u32, edge: &str) -> Result<u32, String> {
    let given = vec![
        ("plane", Val::Node(scratch)),
        ("indices", nodes(&[index])),
        ("ttype", type_of(ctx, scratch)),
        ("edge", text(edge)),
    ];
    build(ctx, "At", given)
}

/// A store of `value` into the scratch plane `name` (of `extent` elements) at `index`.
pub fn store(
    ctx: &mut Ctx,
    name: &str,
    extent: i128,
    value: u32,
    index: u32,
) -> Result<u32, String> {
    let given = vec![
        ("name", text(name)),
        ("value", Val::Node(value)),
        ("indices", nodes(&[index])),
        ("role", text("scratch")),
        ("axes", axes(&[("rank", Some(extent))])),
    ];
    build(ctx, "Update", given)
}

/// A store of `value` into the lane's own element of the output plane `name`.
pub fn commit(ctx: &mut Ctx, name: &str, value: u32) -> Result<u32, String> {
    build(
        ctx,
        "Update",
        vec![("name", text(name)), ("value", Val::Node(value))],
    )
}

/// The group-wide reduce `op` of `value`, added in `order` when it is a sum.
pub fn reduce(ctx: &mut Ctx, op: &str, order: &str, value: u32) -> Result<u32, String> {
    let given = vec![
        ("op", text(op)),
        ("value", Val::Node(value)),
        ("order", text(order)),
    ];
    build(ctx, "GroupReduce", given)
}

/// The group-wide scan `op` of `value`, counting the lane's own value when `inclusive`.
pub fn scan(ctx: &mut Ctx, op: &str, inclusive: bool, value: u32) -> Result<u32, String> {
    let given = vec![
        ("op", text(op)),
        ("value", Val::Node(value)),
        ("inclusive", Val::Bool(inclusive)),
    ];
    build(ctx, "GroupScan", given)
}

/// The barrier that ends a phase.
pub fn barrier(ctx: &mut Ctx) -> Result<u32, String> {
    build(ctx, "Barrier", vec![])
}

/// How the lanes of a group are chosen: the facts of the membership beside the body.
#[derive(Default)]
pub struct Membership {
    /// `count`, `axes` or `key`.
    pub kind: &'static str,
    /// The lanes of a `count` group.
    pub count: Option<i128>,
    /// The lane axes of the kernel (an `axes` group states them).
    pub grid: Vec<(&'static str, Option<i128>)>,
    /// The lane axes that span one group.
    pub axes: Vec<&'static str>,
    /// The largest group a `key` group holds.
    pub max_size: Option<i128>,
    /// The per-lane key of a `key` group.
    pub key: Option<u32>,
}

impl Membership {
    /// Groups of `lanes` consecutive lanes.
    pub fn count(lanes: i128) -> Self {
        Membership {
            kind: "count",
            count: Some(lanes),
            ..Default::default()
        }
    }
}

/// A group of `membership` declaring `scratch` over the sinks `body`.
pub fn group(ctx: &mut Ctx, m: &Membership, scratch: &[u32], body: &[u32]) -> Result<u32, String> {
    let labels = m.axes.iter().map(|l| text(l)).collect();
    let given = vec![
        ("membership", text(m.kind)),
        ("body_sinks", nodes(body)),
        ("key", m.key.map_or(Val::None, Val::Node)),
        ("count", m.count.map_or(Val::None, Val::Int)),
        ("grid", axes(&m.grid)),
        ("axes", Val::Tuple(labels)),
        ("max_size", m.max_size.map_or(Val::None, Val::Int)),
        ("scratch", nodes(scratch)),
    ];
    build(ctx, "Group", given)
}

/// A reference kernel: its Group node and the names of its planes.
pub struct Fixture {
    /// The Group node, the root of the document.
    pub group: u32,
    /// The names of the per-lane input planes.
    pub inputs: &'static [&'static str],
    /// The names of the scratch planes the group declares.
    pub scratch: &'static [&'static str],
    /// The names of the output planes the body commits to.
    pub outputs: &'static [&'static str],
}

/// `tile`: lanes in groups of 8; input `x`. Phase 0 stores sqrt(x) into the scratch `t` at the lane's rank and sums `t[rank]`
/// over the group with the fixed tree; the sum ends the phase. Phase 1 commits `y = t[rank] / s`.
/// Phases of the body (store, commit): [0, 1].
pub fn tile(ctx: &mut Ctx) -> Result<Fixture, String> {
    let x = plane_in(ctx, "x", "f64")?;
    let t = scratch(ctx, "t", "f64", 8)?;
    let place = rank(ctx)?;
    let root = apply(ctx, "sqrt", &[x])?;
    let fill = store(ctx, "t", 8, root, place)?;
    let own = read(ctx, t, place, "clamp")?;
    let total = reduce(ctx, "sum", "tree", own)?;
    let share = apply(ctx, "div", &[own, total])?;
    let out = commit(ctx, "y", share)?;
    let group = group(ctx, &Membership::count(8), &[t], &[fill, out])?;
    Ok(Fixture {
        group,
        inputs: &["x"],
        scratch: &["t"],
        outputs: &["y"],
    })
}

/// `rows`: lanes on a grid `row` (run-time extent) by `col` (4); a group is one row (membership by the axis `col`); input `x`.
/// Outputs per lane: `total` (sum in rank order), `least` (minimum) and `place` (the lane's rank). No scratch.
/// Phases of the body (total, least, place): [1, 1, 1]; the two reductions are independent and share phase 0.
pub fn rows(ctx: &mut Ctx) -> Result<Fixture, String> {
    let x = plane_in(ctx, "x", "f64")?;
    let sum = reduce(ctx, "sum", "lane", x)?;
    let total = commit(ctx, "total", sum)?;
    let min = reduce(ctx, "min", "tree", x)?;
    let least = commit(ctx, "least", min)?;
    let place = rank(ctx)?;
    let rank_out = commit(ctx, "place", place)?;
    let by_col = Membership {
        kind: "axes",
        grid: vec![("row", None), ("col", Some(4))],
        axes: vec!["col"],
        ..Default::default()
    };
    let group = group(ctx, &by_col, &[], &[total, least, rank_out])?;
    Ok(Fixture {
        group,
        inputs: &["x"],
        scratch: &[],
        outputs: &["total", "least", "place"],
    })
}

/// `ring`: lanes in groups of 8; input `x`. Phase 0 stores x into the scratch `t` at the lane's rank; a barrier; phase 1
/// commits `y = t[rank] + t[rank + 1]` with the edge rule `wrap`, a neighbour read of another lane's element.
/// Phases of the body (store, barrier, commit): [0, 0, 1].
pub fn ring(ctx: &mut Ctx) -> Result<Fixture, String> {
    ring_with(ctx, true)
}

/// `ring`, with its barrier or without it (the refused form: the neighbour read shares the phase that writes `t`).
pub fn ring_with(ctx: &mut Ctx, with_barrier: bool) -> Result<Fixture, String> {
    let x = plane_in(ctx, "x", "f64")?;
    let t = scratch(ctx, "t", "f64", 8)?;
    let place = rank(ctx)?;
    let fill = store(ctx, "t", 8, x, place)?;
    let one = whole(ctx, "i32", 1)?;
    let next = apply(ctx, "add", &[place, one])?;
    let own = read(ctx, t, place, "clamp")?;
    let neighbour = read(ctx, t, next, "wrap")?;
    let sum = apply(ctx, "add", &[own, neighbour])?;
    let out = commit(ctx, "y", sum)?;
    let body = if with_barrier {
        vec![fill, barrier(ctx)?, out]
    } else {
        vec![fill, out]
    };
    let group = group(ctx, &Membership::count(8), &[t], &body)?;
    Ok(Fixture {
        group,
        inputs: &["x"],
        scratch: &["t"],
        outputs: &["y"],
    })
}
