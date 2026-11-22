//! Team-form linear algebra: what a kernel is cut at when a matrix is too large for one lane, and the scratch planes between its
//! stages. A factorisation of a matrix above the lane's limit (or any size when the build asks for teams) is worked by a team of
//! threads, one team per lane, on a matrix that lives in scratch the runtime owns. That makes it a launch boundary: the stage
//! before it stores the operands into planes of `lanes x size` elements, the team stage factors the matrix in place and runs every
//! solve of it, and the stages after it read each result at the lane's own slice. A solve, a determinant or a field of the factor
//! is a member of the group its factorisation makes, so one factorisation runs once however many results are read.
use crate::blocks::LinalgForm;
use crate::cells::build;
use crate::planes::type_parts;
use crate::stages::Keyed;
use crate::{is_a, R};
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::value::{ttype_val, Val};
use std::collections::{HashMap, HashSet};

/// The prefix of a scratch plane made for team stages.
pub const PLANE_PREFIX: &str = "tm";

/// The kinds of the factorisations, each one node whose value is the whole factor.
const FACTORISATIONS: [&str; 5] = ["lu", "chol", "eigh", "chol_from", "qr"];
/// The kinds that solve against a factor.
const CONSUMERS: [&str; 7] = [
    "lu_solve",
    "lu_solve_t",
    "cho_solve",
    "tri_lower",
    "tri_upper_t",
    "qr_solve",
    "qr_gram_solve",
];
/// The kinds that read one field of a factor.
const PROJECTIONS: [&str; 8] = [
    "lu_info",
    "lu_det",
    "chol_info",
    "chol_factor",
    "eigh_w",
    "eigh_v",
    "eigh_info",
    "qr_info",
];
/// The kinds of the QR least-squares family, which exists only as a team stage.
const QR_OPS: [&str; 4] = ["qr", "qr_solve", "qr_gram_solve", "qr_info"];

/// A factorisation the device works with one warp per lane instead of a team: its kind, order, the scalar type name it is stated for
/// and the outputs read (`v` for the vectors; none for the values only).
#[derive(Debug, Clone, PartialEq)]
pub struct WarpEntry {
    /// The factorisation's kind (`eigh`).
    pub kind: String,
    /// The matrix order.
    pub n: i64,
    /// The scalar type name, `float32` or `float64`.
    pub scalar: String,
    /// The outputs read besides the info word: `v` or nothing.
    pub outputs: Vec<String>,
}

/// The numbers a team plan reads: the build's settings and the target's limits.
#[derive(Debug, Clone)]
pub struct TeamLimits {
    /// How the build holds the linear algebra of a lane.
    pub form: LinalgForm,
    /// The largest dimension one lane factors alone.
    pub max_dim: i64,
    /// The largest dimension a team of threads works on.
    pub team_max_dim: i64,
    /// The threads of the block that copies operands into team scratch.
    pub copy_threads: i64,
    /// The threads of a team for a matrix of order up to this many.
    pub team_threads: i64,
    /// The threads of a team for a larger matrix.
    pub team_threads_large: i64,
    /// The team size the caller forces, or 0 for the default.
    pub forced: i64,
    /// The lanes of a warp, the team size of a stage that holds only warp steps; none for a target without warps.
    pub warp_threads: Option<i64>,
    /// The factorisations a warp takes per lane in this build (kind, order, outputs read), already narrowed by the caller to the
    /// scalar type of the build, to a build whose every target is a device, and, for a double row, to a device whose FP64 rate is
    /// high; empty for any other build.
    pub warp_eigh: Vec<WarpEntry>,
}

/// What one lane's slice of a scratch plane stands for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unit {
    /// A lane.
    Lane,
    /// A key of a block sum.
    Key,
    /// The cells of the plane an ordered scatter adds into, per chunk.
    Target,
    /// One slice for the whole launch.
    Once,
}

/// One scratch plane: `per_lane` elements of `dtype` for each unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    /// The plane's name.
    pub name: String,
    /// `real` (the kernel's scalar type), `int32` or `partial`.
    pub dtype: &'static str,
    /// The elements of each unit's slice.
    pub per_lane: i64,
    /// What a slice stands for.
    pub unit: Unit,
    /// What the plane holds at the start of every run, when something is promised.
    pub fill: Option<i64>,
}

impl Plane {
    /// A plane of one unit per lane.
    pub(crate) fn per_lane(name: String, dtype: &'static str, per_lane: i64) -> Plane {
        Plane {
            name,
            dtype,
            per_lane,
            unit: Unit::Lane,
            fill: None,
        }
    }
}

/// One collective call of a team stage.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// The operation, by name.
    pub op: &'static str,
    /// The matrix order or element count.
    pub n: i64,
    /// The right-hand columns.
    pub k: i64,
    /// The rows of a tall factor.
    pub m: i64,
    /// The most chunks an ordered scatter's first stage cuts the lanes into (0 for any other call).
    pub max_chunks: i64,
    /// The scratch planes the call works on, as (role, plane name).
    pub planes: Vec<(String, String)>,
}

/// What one block of a team stage's launch is.
#[derive(Debug, Clone, PartialEq)]
pub enum Grid {
    /// A lane: the launch's grid is the lane count.
    Lanes,
    /// A key of a block sum: the grid is the key count, in `groups` blocks per key.
    Keys {
        /// The keyed plane the keys are those of.
        keyed: Keyed,
        /// The blocks of a key.
        groups: i64,
    },
    /// A chunk of lanes of an ordered scatter into the plane `target`.
    Chunks {
        /// The plane the scatter adds into.
        target: String,
    },
    /// A cell of the plane `target` of an ordered scatter.
    Targets {
        /// The plane the scatter adds into.
        target: String,
    },
}

/// A team stage: its size, the calls in order and the scratch planes they touch.
#[derive(Debug, Clone, PartialEq)]
pub struct TeamSpec {
    /// The threads of one team.
    pub threads: i64,
    /// The calls in order.
    pub steps: Vec<Step>,
    /// The scratch planes the calls touch.
    pub planes: Vec<Plane>,
    /// What a block of the launch is.
    pub grid: Grid,
    /// The plane the stage commits its result to, which is not scratch of its own.
    pub out: Option<String>,
    /// The input planes (leaf nodes) the stage reads in place.
    pub reads: Vec<u32>,
}

/// The factorisation of a group and every node of the body that reads it.
#[derive(Debug)]
pub struct Group {
    /// The group's number, which names its planes.
    pub index: usize,
    /// The factorisation node.
    pub factor: u32,
    /// The nodes that read the factor, in the order they appear.
    pub members: Vec<u32>,
    /// The matrix order.
    pub n: i64,
    /// The rows of the matrix a QR factors.
    pub m: i64,
    /// The group's scratch, by role.
    pub planes: Vec<(&'static str, Plane)>,
    /// The plane each member's result lives in.
    pub member_planes: HashMap<u32, Plane>,
    /// The plane each member's right-hand side is stored in and solved in place.
    pub rhs_planes: HashMap<u32, Plane>,
    /// Whether one warp per lane works the factorisation (eigenvalues of an order-32 matrix) in place of a team.
    pub warp: bool,
}

impl Group {
    /// The group's plane in a role (the roles a factor always has).
    fn plane(&self, role: &str) -> &Plane {
        &self
            .planes
            .iter()
            .find(|(r, _)| *r == role)
            .expect("a plane of the group")
            .1
    }
}

/// Whether the node is one of the linear-algebra kinds.
fn is_linalg(ctx: &Ctx, node: u32) -> bool {
    is_a(ctx, node, "Op")
        && ctx.str_field(node, "kind").is_ok_and(|kind| {
            [&FACTORISATIONS[..], &CONSUMERS[..], &PROJECTIONS[..]]
                .iter()
                .any(|list| list.contains(&kind.as_str()))
        })
}

/// The kind of an op node.
fn kind(ctx: &Ctx, node: u32) -> R<String> {
    ctx.str_field(node, "kind")
}

/// The static order of a linear-algebra node: the matrix it factors, or the factor it reads.
fn dimension(ctx: &Ctx, node: u32) -> i64 {
    let first = ctx.nodes[node as usize].ops[0];
    ctx.nodes[first as usize].shape.last().copied().unwrap_or(1)
}

/// The elements of one lane's slice of a node's value.
pub(crate) fn size_of(ctx: &Ctx, node: u32) -> i64 {
    ctx.nodes[node as usize].shape.iter().product()
}

/// Whether a team of threads works on the node (else one lane holds its matrix), refused by name when the build forbids it.
fn is_team(ctx: &Ctx, node: u32, limits: &TeamLimits) -> R<bool> {
    let kind = kind(ctx, node)?;
    let n = dimension(ctx, node);
    if QR_OPS.contains(&kind.as_str()) {
        if limits.form == LinalgForm::Thread {
            return refuse(format!(
                "op {kind:?}: least squares by QR is worked by a team of threads, so linalg_form='thread' refuses it; build with linalg_form='auto' or 'team'"
            ));
        }
        return Ok(true);
    }
    if limits.form == LinalgForm::Thread && n > limits.max_dim {
        return refuse(format!(
            "op {kind:?} at n={n}: linalg_form='thread' keeps the matrix in one lane, which holds n <= {}; build with linalg_form='auto' or 'team'",
            limits.max_dim
        ));
    }
    Ok(limits.form == LinalgForm::Team || n > limits.max_dim)
}

/// The refusal text for a team-form op inside a loop of the body, or None when no such op sits in a loop: a team works in a launch of
/// its own, which cannot be one per iteration.
pub(crate) fn team_in_loop(ctx: &Ctx, order: &[u32], limits: &TeamLimits) -> R<Option<String>> {
    for &node in order {
        if !is_a(ctx, node, "Loop") {
            continue;
        }
        let mut loops = vec![node];
        while let Some(lp) = loops.pop() {
            for inner in ctx.body_nodes(lp)? {
                if is_a(ctx, inner, "Loop") {
                    loops.push(inner);
                } else if is_linalg(ctx, inner) && is_team(ctx, inner, limits)? {
                    return Ok(Some(format!(
                        "op {:?} at n={} is worked by a team of threads, which is a launch of its own and cannot sit inside a loop of the body; iterate by replaying the plan (one launch per iteration)",
                        kind(ctx, inner)?,
                        dimension(ctx, inner)
                    )));
                }
            }
        }
    }
    Ok(None)
}

/// The factorisations that take the warp form: a build with a warp row for the factorisation's kind, order and the outputs read
/// (the vectors of an `eigh` make the row's outputs `v`), under `linalg_form` auto, whose stage can be a launch of its own: not in a
/// loop body, and not reachable only through an arm of a dispatch (only the selector of a dispatch is followed).
fn warp_factors(ctx: &Ctx, order: &[u32], sinks: &[u32], limits: &TeamLimits) -> R<HashSet<u32>> {
    if limits.form != LinalgForm::Auto || limits.warp_eigh.is_empty() {
        return Ok(HashSet::new());
    }
    let mut vectors: HashSet<u32> = HashSet::new();
    let mut found: Vec<u32> = vec![];
    for &node in order {
        if !is_a(ctx, node, "Op") {
            continue;
        }
        let name = kind(ctx, node)?;
        if name == "eigh_v" {
            vectors.insert(ctx.nodes[node as usize].ops[0]);
        } else if FACTORISATIONS.contains(&name.as_str()) {
            found.push(node);
        }
    }
    found.retain(|&node| {
        let name = kind(ctx, node).unwrap_or_default();
        let outputs: Vec<&str> = if vectors.contains(&node) {
            vec!["v"]
        } else {
            vec![]
        };
        let n = dimension(ctx, node);
        limits.warp_eigh.iter().any(|row| {
            row.kind == name
                && row.n == n
                && row
                    .outputs
                    .iter()
                    .map(String::as_str)
                    .eq(outputs.iter().copied())
        })
    });
    if found.is_empty() {
        return Ok(HashSet::new());
    }
    let mut inner: HashSet<u32> = HashSet::new();
    for &node in order {
        if is_a(ctx, node, "Loop") {
            inner.extend(ctx.body_nodes(node)?);
        }
    }
    let mut seen: HashSet<u32> = HashSet::new();
    let mut stack = sinks.to_vec();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        let ops = &ctx.nodes[node as usize].ops;
        stack.extend(if is_a(ctx, node, "Dispatch") {
            &ops[..1]
        } else {
            &ops[..]
        });
    }
    Ok(found
        .into_iter()
        .filter(|node| seen.contains(node) && !inner.contains(node))
        .collect())
}

/// The groups of team-form linear algebra among the nodes (children first): one per factorisation, its members in the order they
/// appear.
pub fn groups(ctx: &Ctx, order: &[u32], sinks: &[u32], limits: &TeamLimits) -> R<Vec<Group>> {
    let warp = warp_factors(ctx, order, sinks, limits)?;
    let mut found: Vec<Group> = vec![];
    for &node in order {
        if !is_linalg(ctx, node) {
            continue;
        }
        let kind = kind(ctx, node)?;
        let factor = if FACTORISATIONS.contains(&kind.as_str()) {
            node
        } else {
            ctx.nodes[node as usize].ops[0]
        };
        if !is_team(ctx, node, limits)? && !warp.contains(&factor) {
            continue;
        }
        if !(is_a(ctx, factor, "Op") && FACTORISATIONS.contains(&self::kind(ctx, factor)?.as_str()))
        {
            return refuse(format!(
                "op {kind:?} reads a factor that no factorisation made"
            ));
        }
        let at = match found.iter().position(|g| g.factor == factor) {
            Some(at) => at,
            None => {
                let n = dimension(ctx, factor);
                if n > limits.team_max_dim {
                    return refuse(format!(
                        "op {:?}: the dimension is static and n <= {} (the target's team limit), got n={n}",
                        self::kind(ctx, factor)?,
                        limits.team_max_dim
                    ));
                }
                let first = ctx.nodes[factor as usize].ops[0];
                let m = if self::kind(ctx, factor)? == "qr" {
                    ctx.nodes[first as usize].shape[0]
                } else {
                    0
                };
                found.push(Group {
                    index: found.len(),
                    factor,
                    members: vec![],
                    n,
                    m,
                    planes: vec![],
                    member_planes: HashMap::new(),
                    rhs_planes: HashMap::new(),
                    warp: warp.contains(&factor),
                });
                found.len() - 1
            }
        };
        if node != factor {
            found[at].members.push(node);
        }
    }
    for group in &mut found {
        name_planes(ctx, group)?;
    }
    Ok(found)
}

/// Whether the node is the constant identity matrix (an inverse solves against it).
fn identity(ctx: &Ctx, node: u32) -> R<bool> {
    let record = &ctx.nodes[node as usize];
    if !(is_a(ctx, node, "Op") && kind(ctx, node)? == "vec" && record.shape.len() == 2) {
        return Ok(false);
    }
    let order = record.shape[0] as usize;
    for (at, &operand) in record.ops.iter().enumerate() {
        if !is_a(ctx, operand, "Const") {
            return Ok(false);
        }
        let want = if at / order == at % order { 1.0 } else { 0.0 };
        let held = match ctx.field(operand, "literal")? {
            Val::Float(x) => x,
            Val::Int(x) => x as f64,
            _ => return Ok(false),
        };
        if held != want {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Adds a plane to the group under its role and returns it.
fn add_plane(
    group: &mut Group,
    role: &'static str,
    tag: &str,
    dtype: &'static str,
    per_lane: i64,
) -> Plane {
    let name = format!("{PLANE_PREFIX}{}_{tag}", group.index);
    let plane = Plane::per_lane(name, dtype, per_lane);
    group.planes.push((role, plane.clone()));
    plane
}

/// The right-hand side columns of a consumer.
fn columns(ctx: &Ctx, member: u32) -> i64 {
    let rhs = ctx.nodes[member as usize].ops[1];
    match ctx.nodes[rhs as usize].shape.as_slice() {
        [_, columns] => *columns,
        _ => 1,
    }
}

/// The scratch elements a QR needs as work space for `cols` columns of `m` rows: one partial sum per column per tile of 64 rows, and
/// one.
fn qr_work(m: i64, cols: i64) -> i64 {
    cols * ((m + 63) / 64) + 1
}

/// The scratch of a group: the factor's planes, then one per member that has a result of its own.
fn name_planes(ctx: &Ctx, group: &mut Group) -> R<()> {
    let (n, factor_kind) = (group.n, kind(ctx, group.factor)?);
    let operand = if factor_kind == "qr" {
        group.m * n
    } else {
        n * n
    };
    add_plane(group, "a", "a", "real", operand);
    if factor_kind == "lu" {
        add_plane(group, "piv", "piv", "int32", n);
    }
    add_plane(group, "info", "info", "int32", 1);
    if factor_kind == "eigh" {
        add_plane(group, "w", "w", "real", n);
        // a warp works the values in its own scratch and has neither vectors nor a work plane
        if !group.warp {
            add_plane(group, "v", "v", "real", n * n);
            add_plane(group, "work", "work", "real", n * n + 4);
        }
    }
    if factor_kind == "qr" {
        let mut widest = n;
        for &member in &group.members {
            if CONSUMERS.contains(&kind(ctx, member)?.as_str()) {
                widest = widest.max(columns(ctx, member));
            }
        }
        add_plane(group, "tau", "tau", "real", n);
        add_plane(group, "work", "work", "real", qr_work(group.m, widest));
    }
    for (at, &member) in group.members.clone().iter().enumerate() {
        let member_kind = kind(ctx, member)?;
        let (own, rhs) = (
            size_of(ctx, member),
            ctx.nodes[member as usize].ops.get(1).copied(),
        );
        if member_kind == "qr_solve" {
            let rhs_plane = add_plane(
                group,
                "r",
                &format!("r{at}"),
                "real",
                size_of(ctx, rhs.expect("a right-hand side")),
            );
            let out = add_plane(group, "x", &format!("x{at}"), "real", own);
            group.member_planes.insert(member, out);
            group.rhs_planes.insert(member, rhs_plane);
        } else if CONSUMERS.contains(&member_kind.as_str()) {
            let out = add_plane(group, "x", &format!("x{at}"), "real", own);
            group.member_planes.insert(member, out.clone());
            if !(member_kind == "lu_solve" && identity(ctx, rhs.expect("a right-hand side"))?) {
                group.rhs_planes.insert(member, out);
            }
        } else if member_kind == "lu_det" {
            let out = add_plane(group, "x", &format!("d{at}"), "real", 1);
            group.member_planes.insert(member, out);
        }
    }
    Ok(())
}

/// The read of a team result held in plane `plane`, at the lane's own slice or at slice `at`: a scalar is a plain read of the lane's element, a vector or matrix a team read.
pub(crate) fn read(ctx: &mut Ctx, plane: &str, (like, at): (u32, Option<u32>)) -> R<u32> {
    let parts = type_parts(ctx, like);
    let scalar = ttype_val(&parts.dtype, &[], None);
    let leaf = vec![
        ("kind", Val::Str("table_read".into())),
        ("role", Val::Str("lookup".into())),
        ("name", Val::Str(plane.into())),
        ("ttype", scalar.clone()),
    ];
    let leaf = build(ctx, "Leaf", leaf)?;
    let index = match at {
        Some(index) => index,
        None => sample_index(ctx)?,
    };
    let class = if parts.shape.is_empty() {
        "At"
    } else {
        "TeamRead"
    };
    let ttype = if parts.shape.is_empty() {
        scalar
    } else {
        ttype_val(&parts.dtype, &parts.shape, parts.tag.as_deref())
    };
    let operands = Val::Tuple(vec![Val::Node(leaf), Val::Node(index)]);
    if class == "At" {
        return build(
            ctx,
            class,
            vec![
                ("plane", Val::Node(leaf)),
                ("indices", Val::Tuple(vec![Val::Node(index)])),
                ("edge", Val::Str("unchecked".into())),
                ("ttype", ttype),
            ],
        );
    }
    build(
        ctx,
        class,
        vec![
            ("operands", operands),
            ("edge", Val::Str("unchecked".into())),
            ("ttype", ttype),
        ],
    )
}

/// The lane's own index into a scratch plane (64-bit, since `lanes x size` passes 2^31).
pub(crate) fn sample_index(ctx: &mut Ctx) -> R<u32> {
    build(
        ctx,
        "SampleIndex",
        vec![("ttype", ttype_val("i64", &[], None))],
    )
}

/// The node that stands for a member's value in the stages after the team stage.
pub fn output(ctx: &mut Ctx, group: &Group, member: u32) -> R<u32> {
    let member_kind = kind(ctx, member)?;
    let plane = if CONSUMERS.contains(&member_kind.as_str()) || member_kind == "lu_det" {
        group.member_planes[&member].name.clone()
    } else if member_kind.ends_with("_info") {
        group.plane("info").name.clone()
    } else if member_kind == "chol_factor" {
        group.plane("a").name.clone()
    } else if member_kind == "eigh_w" {
        group.plane("w").name.clone()
    } else {
        group.plane("v").name.clone()
    };
    read(ctx, &plane, (member, None))
}

/// The values the per-lane stages store for a group, as (node, plane name): the operand and each right-hand side that is not the
/// identity.
pub fn stores(ctx: &Ctx, group: &Group) -> Vec<(u32, String)> {
    let mut out = vec![(
        ctx.nodes[group.factor as usize].ops[0],
        group.plane("a").name.clone(),
    )];
    for &member in &group.members {
        if let Some(plane) = group.rhs_planes.get(&member) {
            out.push((ctx.nodes[member as usize].ops[1], plane.name.clone()));
        }
    }
    out
}

/// The store of a value into the lane's own slice of the scratch plane `name`.
pub fn team_store(ctx: &mut Ctx, name: &str, value: u32) -> R<u32> {
    let index = sample_index(ctx)?;
    let operands = Val::Tuple(vec![Val::Node(value), Val::Node(index)]);
    build(
        ctx,
        "TeamStore",
        vec![("name", Val::Str(name.into())), ("operands", operands)],
    )
}

/// The calls a group makes at team level `level`: the factorisation when it is the first level with a member, then each member.
fn step_list(
    ctx: &Ctx,
    group: &Group,
    level: usize,
    depth: &HashMap<u32, usize>,
    first: bool,
) -> R<Vec<Step>> {
    let (factor_kind, n) = (kind(ctx, group.factor)?, group.n);
    let here: Vec<u32> = group
        .members
        .iter()
        .copied()
        .filter(|m| depth[m] == level)
        .collect();
    let mut out = vec![];
    if here.is_empty() {
        return Ok(out);
    }
    let name = |role: &str| (role.to_string(), group.plane(role).name.clone());
    let mut base = vec![name("a"), name("info")];
    if factor_kind == "lu" {
        base.push(name("piv"));
    }
    let mut roles = base.clone();
    if factor_kind == "eigh" {
        roles.push(name("w"));
        if !group.warp {
            roles.extend([name("v"), name("work")]);
        }
    }
    if factor_kind == "qr" {
        base.extend([name("tau"), name("work")]);
        roles.extend([name("tau"), name("work")]);
    }
    if first {
        let op = if group.warp {
            "eigh_warp"
        } else {
            FACTORISATIONS
                .iter()
                .find(|k| **k == factor_kind)
                .copied()
                .unwrap_or("lu")
        };
        out.push(Step {
            op,
            n,
            k: 1,
            m: group.m,
            max_chunks: 0,
            planes: roles,
        });
    }
    for member in here {
        let member_kind = kind(ctx, member)?;
        let consumer = CONSUMERS.contains(&member_kind.as_str());
        let k = if consumer { columns(ctx, member) } else { 1 };
        let mut planes = base.clone();
        let own = |plane: &Plane| plane.name.clone();
        if member_kind == "qr_solve" {
            planes.push(("r".into(), own(&group.rhs_planes[&member])));
            planes.push(("x".into(), own(&group.member_planes[&member])));
            out.push(Step {
                op: "qr_solve",
                n,
                k,
                m: group.m,
                max_chunks: 0,
                planes,
            });
            continue;
        }
        let op = if consumer {
            if member_kind == "lu_solve" && !group.rhs_planes.contains_key(&member) {
                planes.push(("x".into(), own(&group.member_planes[&member])));
                out.push(Step {
                    op: "lu_inverse",
                    n,
                    k: 1,
                    m: 0,
                    max_chunks: 0,
                    planes,
                });
                continue;
            }
            planes.push(("x".into(), own(&group.member_planes[&member])));
            CONSUMERS
                .iter()
                .find(|c| **c == member_kind)
                .copied()
                .unwrap_or("lu_solve")
        } else if member_kind == "lu_det" {
            planes.push(("x".into(), own(&group.member_planes[&member])));
            "lu_det"
        } else {
            continue;
        };
        out.push(Step {
            op,
            n,
            k,
            m: 0,
            max_chunks: 0,
            planes,
        });
    }
    Ok(out)
}

/// The team size for a matrix of order `n`: the card's smaller team up to that many, its larger team above, or the caller's.
pub(crate) fn threads_for(n: i64, limits: &TeamLimits) -> i64 {
    if limits.forced != 0 {
        limits.forced
    } else if n <= limits.team_threads {
        limits.team_threads
    } else {
        limits.team_threads_large
    }
}

/// The team stage at `level`: every group's calls, or none when no group has a member at that level. `factored` holds the groups
/// already factored.
pub fn spec_of(
    ctx: &Ctx,
    groups: &[Group],
    level: usize,
    depth: &HashMap<u32, usize>,
    (factored, limits): (&mut Vec<usize>, &TeamLimits),
) -> R<Option<TeamSpec>> {
    let (mut steps, mut planes, mut size) = (vec![], vec![], 0);
    for group in groups {
        let calls = step_list(ctx, group, level, depth, !factored.contains(&group.index))?;
        if calls.is_empty() {
            continue;
        }
        factored.push(group.index);
        size = size.max(group.n);
        let used: Vec<&String> = calls
            .iter()
            .flat_map(|s| s.planes.iter().map(|p| &p.1))
            .collect();
        for (_, plane) in &group.planes {
            if used.contains(&&plane.name) && !planes.iter().any(|p: &Plane| p.name == plane.name) {
                planes.push(plane.clone());
            }
        }
        steps.extend(calls);
    }
    // a warp per lane is a team of exactly one warp
    let warp_only = steps.iter().all(|s: &Step| s.op == "eigh_warp");
    let spec = TeamSpec {
        threads: match limits.warp_threads {
            Some(warp) if warp_only && !steps.is_empty() => warp,
            _ => threads_for(size, limits),
        },
        steps,
        planes,
        grid: Grid::Lanes,
        out: None,
        reads: vec![],
    };
    Ok((!spec.steps.is_empty()).then_some(spec))
}

/// The sinks of a team stage: one marker per plane it works on, which binds the plane as an argument.
pub fn markers(ctx: &mut Ctx, spec: &TeamSpec) -> R<Vec<u32>> {
    let mut out = vec![];
    let names = spec
        .planes
        .iter()
        .map(|p| (&p.name, p.dtype))
        .chain(spec.out.iter().map(|o| (o, "real")));
    for (name, dtype) in names {
        let (literal, dtype) = if dtype == "int32" {
            (Val::Int(0), "int32")
        } else {
            (Val::Float(0.0), "f64")
        };
        let zero = build(
            ctx,
            "Const",
            vec![("literal", literal), ("ttype", ttype_val(dtype, &[], None))],
        )?;
        out.push(build(
            ctx,
            "Update",
            vec![("name", Val::Str(name.clone())), ("value", Val::Node(zero))],
        )?);
    }
    Ok(out)
}
