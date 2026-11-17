//! Whether the sinks of a kernel commit a dense contraction: one loop folding sums of products of reads of one or two operands.
//! The answer is the old code generator's recogniser (`recognize`) over the context's nodes: the loop's header, how each operand
//! walks the reduce axis, the shape of the terms, and the lanes that share one slot of a staged operand. The caller tells the
//! names of the staged planes, which the nodes do not carry.
use crate::is_a;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::{HashMap, HashSet};

/// The result of an analysis: its answer, or the refusal that stopped it.
type R<T> = Result<T, Fault>;

/// How one step of the reduce axis moves an operand's flat index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stride {
    /// A compile-time number of elements.
    Int(i64),
    /// A run-time stride: the launch's sample count (`n_samples`).
    Symbol(String),
}

/// The shape of a term of the fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermForm {
    /// One operand, each term a bare read of it.
    BareRead,
    /// Two operands, each term the product of a read of each.
    ProductOfReads,
    /// Any other term: a matrix product does not compute it.
    Transformed,
}

/// What kind of value a read takes its operand from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandRole {
    /// A plane the caller declared staged.
    Staged,
    /// Any other plane.
    Plane,
    /// The components of a vector input.
    Wide,
    /// The components of a value built in the kernel.
    Computed,
}

/// One operand of the contraction.
#[derive(Debug, Clone, PartialEq)]
pub struct Operand {
    /// The plane's name, the vector input's name, or `computed#k` for the k-th computed value read.
    pub name: String,
    /// What the operand is.
    pub role: OperandRole,
    /// How one step of the reduce axis moves its index; none when its reads disagree.
    pub reduce_stride: Option<Stride>,
    /// Whether its walk can be taken as a matrix.
    pub dense: bool,
    /// Why not, one sentence per reason.
    pub refusals: Vec<String>,
    /// The kinds of the operations the reads' indices apply directly to the lane index, sorted: a lone `div` is the major half
    /// of a lane split, `div`, `mul` and `sub` together its minor half.
    pub lane_ops: Vec<String>,
    /// The lanes that read one slot of this operand, when its address states it (see `staged_fanin`).
    pub fanin: Option<i64>,
}

/// A recognised contraction.
#[derive(Debug, Clone, PartialEq)]
pub struct Contraction {
    /// The first output's name.
    pub carry: String,
    /// The reduce variable's name.
    pub reduce_var: String,
    /// The trip count of the loop.
    pub reduce_extent: i64,
    /// The first value of the reduce variable.
    pub reduce_start: i64,
    /// The step of the reduce variable.
    pub reduce_step: i64,
    /// The operands in first-read order: one (a matrix-vector product) or two (a matrix product).
    pub operands: Vec<Operand>,
    /// The shape of the terms.
    pub term_form: TermForm,
    /// The outputs, each with its rank (0 scalar, 1 vector, 2 matrix).
    pub outputs: Vec<(String, usize)>,
    /// Whether every operand is dense.
    pub dense: bool,
    /// The operands' refusals, in operand order.
    pub refusals: Vec<String>,
    /// For each staged operand, the lanes that read one of its slots, when the address states it.
    pub staged_fanin: Vec<(String, Option<i64>)>,
}

/// One read of an operand inside a term.
struct Read {
    /// The read node: an `At` or a component read.
    node: u32,
    /// Whether it is a component read of a wide value.
    component: bool,
}

/// The nodes of the expression, this one included.
fn subtree(ctx: &Ctx, node: u32) -> R<Vec<u32>> {
    let (mut out, mut seen, mut stack) = (vec![], HashSet::new(), vec![node]);
    while let Some(n) = stack.pop() {
        if seen.insert(n) {
            out.push(n);
            stack.extend(ctx.node(n)?.ops.iter().copied());
        }
    }
    Ok(out)
}

/// Whether the node is a component read of a wide value.
fn is_component(ctx: &Ctx, node: u32) -> R<bool> {
    Ok(is_a(ctx, node, "Op") && ctx.str_field(node, "kind")? == "component_at")
}

/// Whether the node is a read a term multiplies: a plane read or a component read.
fn is_read(ctx: &Ctx, node: u32) -> R<bool> {
    Ok(is_a(ctx, node, "At") || is_component(ctx, node)?)
}

/// A literal zero of rank 0, the identity a sum starts from.
fn is_zero(ctx: &Ctx, node: u32) -> R<bool> {
    if !is_a(ctx, node, "Const") || !ctx.node(node)?.shape.is_empty() {
        return Ok(false);
    }
    Ok(match ctx.field(node, "literal")? {
        Val::Float(x) => x == 0.0,
        Val::Int(n) => n == 0,
        _ => false,
    })
}

/// Every read of a term, first seen first: a component read counts as one read of its value, of which only the index is walked.
fn reads_of(ctx: &Ctx, term: u32) -> R<Vec<Read>> {
    let (mut out, mut stack) = (vec![], vec![term]);
    while let Some(n) = stack.pop() {
        let ops = &ctx.node(n)?.ops;
        if is_component(ctx, n)? {
            out.push(Read {
                node: n,
                component: true,
            });
            stack.push(ops[1]);
        } else {
            if is_a(ctx, n, "At") {
                out.push(Read {
                    node: n,
                    component: false,
                });
            }
            stack.extend(ops.iter().rev());
        }
    }
    Ok(out)
}

/// The flat index a read walks.
fn read_index(ctx: &Ctx, read: &Read) -> R<u32> {
    Ok(ctx.node(read.node)?.ops[1])
}

/// The operand a read belongs to; a value built in the kernel is named `computed#k` by first read, through `computed`.
fn read_name(ctx: &Ctx, read: &Read, computed: &mut HashMap<u32, usize>) -> R<String> {
    let source = ctx.node(read.node)?.ops[0];
    if !read.component {
        return ctx.str_field(source, "name");
    }
    if is_a(ctx, source, "Leaf") {
        return ctx.str_field(source, "name");
    }
    let next = computed.len();
    Ok(format!(
        "computed#{}",
        computed.entry(source).or_insert(next)
    ))
}

/// The coefficient, offset and constants of an affine index form in `var`, or none.
struct Form {
    /// The coefficient, truncated toward zero.
    coefficient: i64,
    /// Whether a bare literal scales the index.
    anonymous: bool,
    /// The run-time symbol that scales the index, if any.
    symbol: Option<String>,
    /// The compile-time offset, when the index has one.
    offset: Option<i64>,
}

/// A rational `n/d` or `n` as an integer, truncated toward zero.
fn truncated(text: &str) -> R<i64> {
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    match (n.parse::<i64>(), d.parse::<i64>()) {
        (Ok(n), Ok(d)) if d != 0 => Ok(n / d),
        _ => refuse(format!("an index form number {text:?}")),
    }
}

/// The affine form of `index` in `var` (none: the sample index), or none when no form is claimed.
fn form_of(ctx: &Ctx, index: u32, var: Option<u32>) -> R<Option<Form>> {
    let texts = ctx.index_form(index, var, false)?;
    if texts.first().map(String::as_str) != Some("form") || texts.len() < 5 {
        return Ok(None);
    }
    Ok(Some(Form {
        coefficient: truncated(&texts[1])?,
        offset: if texts[2] == "-" {
            None
        } else {
            Some(truncated(&texts[2])?)
        },
        anonymous: texts[3] == "1",
        symbol: (texts[4] != "-").then(|| texts[4].clone()),
    }))
}

/// How one operand's reads walk the reduce axis, and every reason the walk is not dense.
fn describe(
    ctx: &Ctx,
    name: &str,
    indices: &[u32],
    loop_index: u32,
) -> R<(Option<Stride>, Vec<String>)> {
    let mut refusals: Vec<String> = vec![];
    for &index in indices {
        if subtree(ctx, index)?.iter().any(|&n| is_a(ctx, n, "At")) {
            refusals.push(format!(
                "{name}: the index depends on a value read from a plane, so the row it walks is chosen at runtime"
            ));
            break;
        }
    }
    let mut strides: HashSet<Stride> = HashSet::new();
    for &index in indices {
        let Some(form) = form_of(ctx, index, Some(loop_index))? else {
            refusals.push(format!(
                "{name}: the index is not affine in the reduce axis"
            ));
            continue;
        };
        if let Some(symbol) = form.symbol {
            if form.coefficient == 1 {
                strides.insert(Stride::Symbol(symbol));
            } else {
                refusals.push(format!(
                    "{name}: the reduce-axis stride is a multiple of the sample count"
                ));
            }
            continue;
        }
        if form.anonymous && form.coefficient != 0 {
            refusals.push(format!(
                "{name}: the reduce-axis stride is built from an anonymous integer literal, so it is not derivable from the declaration"
            ));
        }
        if form.coefficient != 0 {
            strides.insert(Stride::Int(form.coefficient));
        }
    }
    if strides.len() > 1 {
        let mut shown: Vec<String> = strides
            .iter()
            .map(|s| match s {
                Stride::Int(n) => n.to_string(),
                Stride::Symbol(t) => format!("'{t}'"),
            })
            .collect();
        shown.sort();
        refusals.push(format!(
            "{name}: its reads disagree on the reduce-axis stride ([{}])",
            shown.join(", ")
        ));
    }
    let stride = match strides.len() {
        0 => Some(Stride::Int(0)),
        1 => strides.into_iter().next(),
        _ => None,
    };
    let mut seen = HashSet::new();
    refusals.retain(|r| seen.insert(r.clone()));
    Ok((stride, refusals))
}

/// How many lanes read one slot of a plane, when the address states it: every use of the lane index is the dividend of a
/// division by one compile-time number, which is that number.
fn fanin(ctx: &Ctx, indices: &[u32]) -> R<Option<i64>> {
    let (mut divisors, mut covered, mut reached) = (HashSet::new(), HashSet::new(), HashSet::new());
    for &index in indices {
        for n in subtree(ctx, index)? {
            if is_a(ctx, n, "SampleIndex") {
                reached.insert(n);
            }
            if is_a(ctx, n, "Op") && ctx.str_field(n, "kind")? == "div" {
                let ops = &ctx.node(n)?.ops;
                if !is_a(ctx, ops[0], "SampleIndex") {
                    continue;
                }
                if let Some(Form {
                    coefficient: 0,
                    offset: Some(divisor),
                    ..
                }) = form_of(ctx, ops[1], None)?
                {
                    divisors.insert(divisor);
                    covered.insert(ops[0]);
                }
            }
        }
    }
    let one = divisors.len() == 1;
    Ok((one && !reached.is_empty() && reached.is_subset(&covered))
        .then(|| divisors.into_iter().next())
        .flatten())
}

/// The kinds of the operations the indices apply directly to the lane index, sorted and without repeats.
fn lane_ops(ctx: &Ctx, indices: &[u32]) -> R<Vec<String>> {
    let mut kinds = std::collections::BTreeSet::new();
    for &index in indices {
        for n in subtree(ctx, index)? {
            if is_a(ctx, n, "Op")
                && ctx
                    .node(n)?
                    .ops
                    .iter()
                    .any(|&o| is_a(ctx, o, "SampleIndex"))
            {
                kinds.insert(ctx.str_field(n, "kind")?);
            }
        }
    }
    Ok(kinds.into_iter().collect())
}

/// The shape of the terms, given the operand names in first-read order.
fn term_form(
    ctx: &Ctx,
    terms: &[u32],
    operands: &[String],
    computed: &mut HashMap<u32, usize>,
) -> R<TermForm> {
    if operands.len() == 1
        && terms
            .iter()
            .try_fold(true, |all, &t| Ok::<_, Fault>(all && is_read(ctx, t)?))?
    {
        return Ok(TermForm::BareRead);
    }
    if operands.len() != 2 {
        return Ok(TermForm::Transformed);
    }
    for &term in terms {
        if !(is_a(ctx, term, "Op") && ctx.str_field(term, "kind")? == "mul") {
            return Ok(TermForm::Transformed);
        }
        let mut names = HashSet::new();
        for &factor in &ctx.node(term)?.ops {
            if !is_read(ctx, factor)? {
                return Ok(TermForm::Transformed);
            }
            let read = Read {
                node: factor,
                component: is_component(ctx, factor)?,
            };
            names.insert(read_name(ctx, &read, computed)?);
        }
        if names != operands.iter().cloned().collect::<HashSet<_>>() {
            return Ok(TermForm::Transformed);
        }
    }
    Ok(TermForm::ProductOfReads)
}

/// The loop, and the addends beside the carry, of a sink that commits one additive fold: a carried slot seeded with zero and
/// advanced as `carry + terms`, the carry once as a bare addend and nothing else carried feeding it; or none.
fn fold(ctx: &Ctx, sink: u32) -> R<Option<(u32, Vec<u32>)>> {
    let mut value = ctx.node(sink)?.ops[0];
    if is_a(ctx, value, "Op") && ctx.str_field(value, "kind")? == "vec" {
        let mut live = vec![];
        for &c in &ctx.node(value)?.ops {
            if !is_zero(ctx, c)? {
                live.push(c);
            }
        }
        let [only] = live[..] else { return Ok(None) };
        value = only;
    }
    if !is_a(ctx, value, "LoopValue") {
        return Ok(None);
    }
    let lp = ctx.node(value)?.ops[0];
    let Val::Int(slot) = ctx.field(value, "literal")? else {
        return Ok(None);
    };
    let slot = usize::try_from(slot).or_else(|_| refuse("a loop value slot"))?;
    let nodes = |field: &str| -> R<Vec<u32>> {
        match ctx.field(lp, field)? {
            Val::Tuple(items) | Val::List(items) => Ok(items
                .iter()
                .filter_map(|v| if let Val::Node(n) = v { Some(*n) } else { None })
                .collect()),
            _ => refuse(format!("loop field {field:?} is not a list")),
        }
    };
    let (carries, nexts) = (nodes("carries")?, nodes("nexts")?);
    let count = |field: &str| -> R<i64> {
        if let Val::Int(n) = ctx.field(lp, field)? {
            Ok(i64::try_from(n).unwrap_or(0))
        } else {
            Ok(0)
        }
    };
    if count("trip")? < 2
        || ctx.field(lp, "exit_cond")? != Val::None
        || ctx.field(lp, "count")? != Val::None
    {
        return Ok(None);
    }
    if !is_zero(ctx, ctx.node(lp)?.ops[slot])? {
        return Ok(None);
    }
    let carry = carries[slot];
    let terms = ctx.add_terms(nexts[slot])?;
    if terms.iter().filter(|&&t| t == carry).count() != 1 {
        return Ok(None);
    }
    let rest: Vec<u32> = terms.into_iter().filter(|&t| t != carry).collect();
    let touches = |node: u32, other: u32| -> R<bool> { Ok(subtree(ctx, node)?.contains(&other)) };
    for &t in &rest {
        if touches(t, carry)? {
            return Ok(None);
        }
        for &other in &carries {
            if other != carry && touches(t, other)? {
                return Ok(None);
            }
        }
    }
    Ok((!rest.is_empty()).then_some((lp, rest)))
}

/// The contraction the `sinks` commit, or none for anything outside the family: sinks that are plain or wide writes of folds
/// over one loop, reading one or two operands. `staged` names the planes the caller declared staged.
///
/// # Errors
/// A node the context does not hold, or a field the loop lacks.
pub fn recognize(ctx: &mut Ctx, sinks: &[u32], staged: &[String]) -> R<Option<Contraction>> {
    let ctx = &*ctx;
    if sinks.is_empty()
        || !sinks
            .iter()
            .all(|&s| is_a(ctx, s, "Assign") || is_a(ctx, s, "WideWrite"))
    {
        return Ok(None);
    }
    let mut folds = vec![];
    for &sink in sinks {
        match fold(ctx, sink)? {
            Some(f) => folds.push(f),
            None => return Ok(None),
        }
    }
    let lp = folds[0].0;
    if folds.iter().any(|f| f.0 != lp) {
        return Ok(None);
    }
    let terms: Vec<u32> = folds.iter().flat_map(|f| f.1.iter().copied()).collect();
    let mut computed: HashMap<u32, usize> = HashMap::new();
    let mut order: Vec<String> = vec![];
    let mut per_plane: HashMap<String, Vec<u32>> = HashMap::new();
    let mut wide: HashMap<String, OperandRole> = HashMap::new();
    for &term in &terms {
        for read in reads_of(ctx, term)? {
            let name = read_name(ctx, &read, &mut computed)?;
            if !per_plane.contains_key(&name) {
                order.push(name.clone());
            }
            per_plane
                .entry(name.clone())
                .or_default()
                .push(read_index(ctx, &read)?);
            if read.component {
                let source = ctx.node(read.node)?.ops[0];
                wide.insert(
                    name,
                    if is_a(ctx, source, "Leaf") {
                        OperandRole::Wide
                    } else {
                        OperandRole::Computed
                    },
                );
            }
        }
    }
    if !(1..=2).contains(&order.len()) {
        return Ok(None);
    }
    let loop_index = match ctx.field(lp, "index")? {
        Val::Node(n) => n,
        _ => return refuse("a loop without an index"),
    };
    let int = |field: &str| -> R<i64> {
        if let Val::Int(n) = ctx.field(lp, field)? {
            Ok(i64::try_from(n).unwrap_or(0))
        } else {
            refuse(format!("loop {field} is not a number"))
        }
    };
    let mut operands = vec![];
    for name in &order {
        let (stride, refusals) = describe(ctx, name, &per_plane[name], loop_index)?;
        let role = wide.get(name).copied().unwrap_or(if staged.contains(name) {
            OperandRole::Staged
        } else {
            OperandRole::Plane
        });
        operands.push(Operand {
            name: name.clone(),
            role,
            reduce_stride: stride,
            dense: refusals.is_empty(),
            refusals,
            lane_ops: lane_ops(ctx, &per_plane[name])?,
            fanin: fanin(ctx, &per_plane[name])?,
        });
    }
    let form = term_form(ctx, &terms, &order, &mut computed)?;
    let mut outputs = vec![];
    for &sink in sinks {
        outputs.push((ctx.str_field(sink, "name")?, ctx.node(sink)?.shape.len()));
    }
    let mut staged_fanin = vec![];
    for name in order.iter().filter(|n| staged.contains(n)) {
        staged_fanin.push((name.clone(), fanin(ctx, &per_plane[name])?));
    }
    Ok(Some(Contraction {
        carry: ctx.str_field(sinks[0], "name")?,
        reduce_var: ctx.str_field(loop_index, "name")?,
        reduce_extent: int("trip")?,
        reduce_start: int("start")?,
        reduce_step: int("step")?,
        dense: operands.iter().all(|o| o.dense),
        refusals: operands
            .iter()
            .flat_map(|o| o.refusals.iter().cloned())
            .collect(),
        operands,
        term_form: form,
        outputs,
        staged_fanin,
    }))
}
