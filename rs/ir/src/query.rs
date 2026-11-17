//! Small questions and rewrites the writers ask the core: what a node stands for, whether two index
//! expressions are one value, a node's feathers and kinds, a sink over a new value. Owns the C
//! doors//! `rx_through`, `rx_same_index`, `rx_is_param_cell`, `rx_has_feather`,
//! `rx_sample_sums`, `rx_is_a`, `rx_with_value` and `rx_dispatch_nodes`.
use crate::door::{cstr, guard, Node};
use crate::loops::give;
use crate::record::{refuse, Ctx, Fault};
use crate::value::Val;
use std::collections::HashMap;
use std::ffi::{c_char, c_int};

/// A result that refuses with a fault.
type Fallible<T> = Result<T, Fault>;

impl Ctx {
    /// The node a reference to `node` is a reference to: a primitive stands for its forward.
    pub fn through(&self, mut node: Node) -> Fallible<Node> {
        self.node(node)?;
        while self.is_a(node, "Primitive") {
            node = match self.field(node, "forward")? {
                Val::Node(forward) => forward,
                _ => return refuse("a primitive has no forward node"),
            };
        }
        Ok(node)
    }

    /// Whether two index expressions are the same value: one node, or equal class, type, kind,
    /// literal and operands. `seen` keeps a shared subexpression from being compared once per path.
    pub fn same_index(
        &self,
        a: Node,
        b: Node,
        seen: &mut HashMap<(Node, Node), bool>,
    ) -> Fallible<bool> {
        if a == b {
            return Ok(true);
        }
        if let Some(&known) = seen.get(&(a, b)) {
            return Ok(known);
        }
        let (na, nb) = (self.node(a)?, self.node(b)?);
        let mut same = na.class == nb.class
            && (&na.dtype, &na.shape, &na.tag) == (&nb.dtype, &nb.shape, &nb.tag)
            && na.ops.len() == nb.ops.len();
        if same && self.is_a(a, "Const") {
            same = self.field(a, "literal")? == self.field(b, "literal")?;
        } else if same {
            same = (self.is_a(a, "Op") || self.is_a(a, "At"))
                && self.field(a, "kind")? == self.field(b, "kind")?
                && self.field(a, "literal")? == self.field(b, "literal")?;
            for (&x, &y) in na.ops.iter().zip(&nb.ops) {
                same = same && self.same_index(self.through(x)?, self.through(y)?, seen)?;
            }
        }
        seen.insert((a, b), same);
        Ok(same)
    }

    /// Whether `node` reads a parameter cell: an `At` whose index is a role constant so labelled.
    pub fn is_param_cell(&self, node: Node) -> Fallible<bool> {
        self.node(node)?;
        if !self.is_a(node, "At") {
            return Ok(false);
        }
        let index = self.field(node, "index")?;
        let Val::Node(index) = index else {
            return Ok(false);
        };
        if !self.is_a(index, "RoleConst") {
            return Ok(false);
        }
        let label = crate::consts::constant("PARAM_CELL").unwrap_or_default();
        Ok(self.field(index, "label")? == Val::Str(label.trim_matches('"').to_string()))
    }

    /// Whether the kind of `node` states the feather `name`; an unknown feather is refused.
    pub fn has_feather(&self, node: Node, name: &str) -> Fallible<bool> {
        let known = crate::consts::constant("KIND_FEATHERS").unwrap_or_default();
        if !known.contains(&format!("\"{name}: ")) {
            return refuse(format!("unknown feather {}", rcore::keytext::py_str(name)));
        }
        let declared = matches!(self.field(node, "feathers")?,
            Val::Tuple(held) if held.contains(&Val::Str(name.into())));
        // a core kind states its feathers in the manifest; a refining kind inherits them through its class chain
        let chain = &self.table.classes[self.node(node)?.class].chain;
        let core = chain.iter().any(|&c| {
            let class = &self.table.classes[c].name;
            rcore::CLASSES
                .iter()
                .any(|spec| spec.name == class && spec.feathers.contains(&name))
        });
        Ok(declared || core)
    }

    /// The sample-sum sinks of `sinks`, in order.
    pub fn sample_sums(&self, sinks: &[Node]) -> Fallible<Vec<Node>> {
        let mut out = vec![];
        for &s in sinks {
            self.node(s)?;
            if self.is_a(s, "SampleSum") {
                out.push(s);
            }
        }
        Ok(out)
    }

    /// Whether the class of `node` is the class of the kind `name` or refines it (a kind name, as
    /// the manifest lists it); a name that is no kind is false.
    pub fn is_kind(&self, node: Node, name: &str) -> bool {
        let chain = &self.table.classes[self.nodes[node as usize].class].chain;
        self.table
            .kind(name)
            .is_some_and(|class| chain.contains(&class))
    }

    /// `sink` committing `value`: a commit, sample sum, keyed accumulate or finish, or a sink that
    /// states the `commits_value` feather.
    pub fn with_value(&mut self, sink: Node, value: Node) -> Fallible<Node> {
        self.node(value)?;
        let kinds = ["Finish", "SampleSum", "AccumWrite", "Assign"];
        if !kinds.iter().any(|k| self.is_a(sink, k)) && !self.has_feather(sink, "commits_value")? {
            let name = match self.field(sink, "name") {
                Ok(Val::Str(s)) => rcore::keytext::py_str(&s),
                _ => "'?'".into(),
            };
            return refuse(format!(
                "a promotion rebuilds Mutable commits, sample sums, keyed accumulates and a \
                 finish; {} {name} is none of them",
                self.kind_of(sink)?
            ));
        }
        let mut operands = self.nodes[sink as usize].ops.clone();
        operands[0] = value;
        self.rebuild(sink, &operands)
    }
}

/// Writes a flag (0 or 1) through the caller's pointer.
fn flag(out: *mut c_int, value: bool) -> Fallible<()> {
    if out.is_null() {
        return Err(Fault::BadArg("out is NULL".into()));
    }
    unsafe { *out = value as c_int };
    Ok(())
}

/// The node a reference to `node` stands for.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_through(ctx: *mut Ctx, node: Node, out: *mut Node) -> c_int {
    guard(ctx, |c| {
        let through = c.through(node)?;
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        unsafe { *out = through };
        Ok(())
    })
}

/// Whether two index expressions are the same value.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_same_index(ctx: *mut Ctx, a: Node, b: Node, out: *mut c_int) -> c_int {
    guard(ctx, |c| flag(out, c.same_index(a, b, &mut HashMap::new())?))
}

/// Whether a node is of the kind named (a kind name) or of a kind that refines it.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_is_a(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        c.node(node)?;
        flag(out, c.is_kind(node, cstr(name, "name")?))
    })
}

/// The dispatch nodes of a walk, the top-level ones first, then those inside lowered loop bodies:
/// the nodes the walk's `dispatches` listing names, in its order.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_dispatch_nodes(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut Node,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let order = c.walk(walk)?.order.clone();
        let v = c.dispatch_nodes(&order)?;
        give(buf, cap, n, &v)
    })
}

/// Whether the kind of a node states the feather `name`; an unknown feather is a refusal.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_has_feather(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        flag(out, c.has_feather(node, cstr(name, "name")?)?)
    })
}

/// Whether a node reads a parameter cell (an `At` of a role constant labelled so).
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_is_param_cell(ctx: *mut Ctx, node: Node, out: *mut c_int) -> c_int {
    guard(ctx, |c| flag(out, c.is_param_cell(node)?))
}

/// The sink `sink` committing `value` instead of the value it commits.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_with_value(
    ctx: *mut Ctx,
    sink: Node,
    value: Node,
    out: *mut Node,
) -> c_int {
    guard(ctx, |c| {
        let rebuilt = c.with_value(sink, value)?;
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        unsafe { *out = rebuilt };
        Ok(())
    })
}

/// The node where a matrix is wanted: a Cholesky factor becomes its projection, any other factor is refused.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_as_matrix(ctx: *mut Ctx, node: Node, out: *mut Node) -> c_int {
    guard(ctx, |c| {
        c.node(node)?;
        let matrix = c.as_matrix(node)?;
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        unsafe { *out = matrix };
        Ok(())
    })
}
