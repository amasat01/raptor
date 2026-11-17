//! The checks a caller's derivative template passes before it is kept: closure of the graph (what it may read freely, what it may
//! hold) and whether a kind name is taken. Owns the C doors `rx_rule_template_check` and `rx_kind_reserved`.
use crate::door::{cstr, guard, slice};
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::value::Val;
use rcore::keytext::py_str;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_int};

/// A result that refuses with a fault.
type Res<T> = Result<T, Fault>;

/// The kind string of a tape read.
const TAPE_READ: &str = "tape_read";

/// The kind string of a primitive node.
const PRIMITIVE_KIND: &str = "primitive";

/// A node and the loops whose bodies the walk is inside (outermost first).
type Visit = (u32, Vec<u32>);

impl Ctx {
    /// Whether `name` is taken as a kind name: the primitive kind, a core kind a declaration may refine, or a kind declared in
    /// this context. A primitive is registered under a name outside these.
    pub fn kind_reserved(&self, name: &str) -> bool {
        let declared = self
            .table
            .kind(name)
            .is_some_and(|c| self.table.classes[c].decl.is_some());
        name == PRIMITIVE_KIND || self.tables.refinable.iter().any(|k| k == name) || declared
    }

    /// The nodes a loop holds outside its operands: the body's results and the break (its index and carries are its own).
    fn loop_edges(&self, lp: u32) -> Res<Vec<u32>> {
        let mut v = self.refs(lp, "nexts")?;
        v.extend(self.refs(lp, "exit_values")?);
        v.extend(self.node_field(lp, "exit_cond")?);
        Ok(v)
    }

    /// Whether a node of a class that is not a loop holds a node in one of its edge fields.
    fn holds_edges(&self, h: u32) -> Res<bool> {
        let class = &self.table.classes[self.node(h)?.class];
        for field in &class.edges {
            match self.field(h, field)? {
                Val::None => {}
                Val::Tuple(v) | Val::List(v) if v.is_empty() => {}
                _ => return Ok(true),
            }
        }
        Ok(false)
    }

    /// Checks that the graph under `roots` is a CLOSED rule template of the primitive `name`: its free leaves are among `allowed`
    /// (inside a loop body too), a loop's index and carries are read only inside that loop, nothing in it holds edges outside a
    /// loop, writes a plane or reads a tape. Refuses with the first violation found.
    pub fn rule_template_check(&self, name: &str, roots: &[u32], allowed: &[String]) -> Res<()> {
        let who = format!("primitive {}", py_str(name));
        let mut owned: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut seen: HashSet<Visit> = HashSet::new();
        let mut stack: Vec<Visit> = roots.iter().map(|&r| (r, vec![])).collect();
        while let Some((n, loops)) = stack.pop() {
            if !seen.insert((n, loops.clone())) {
                continue;
            }
            let is_loop = self.is_a(n, "Loop");
            if !is_loop && self.holds_edges(n)? {
                return refuse(format!(
                    "{who}: a rule template may not hold a {} node (it holds edges outside a loop)",
                    self.kind_of(n).unwrap_or_default()
                ));
            }
            self.check_leaf(&who, n, !loops.is_empty(), allowed)?;
            let own_read = self.is_a(n, "LoopIndex") || self.is_a(n, "LoopCarry");
            if own_read
                && !loops
                    .iter()
                    .any(|lp| owned.get(lp).is_some_and(|o| o.contains(&n)))
            {
                return refuse(format!(
                    "{who}: a rule template reads the index or a carry of a loop outside that loop"
                ));
            }
            if self.kind_of(n).is_ok_and(|k| k == TAPE_READ) {
                return refuse(format!("{who}: a rule template may not read a tape (it belongs to a loop outside the template)"));
            }
            stack.extend(self.node(n)?.ops.iter().map(|&c| (c, loops.clone())));
            if is_loop {
                self.enter_loop(&who, n, &loops, &mut owned, &mut stack)?;
            }
        }
        Ok(())
    }

    /// Refuses a free leaf whose name is not among `allowed`.
    fn check_leaf(&self, who: &str, n: u32, in_loop: bool, allowed: &[String]) -> Res<()> {
        if !self.is_a(n, "Leaf") {
            return Ok(());
        }
        let leaf = self.str_field(n, "name")?;
        if allowed.contains(&leaf) {
            return Ok(());
        }
        let mut sorted = allowed.to_vec();
        sorted.sort();
        let shown: Vec<String> = sorted.iter().map(|s| py_str(s)).collect();
        let place = if in_loop { " inside a loop body" } else { "" };
        refuse(format!("{who}: a rule template reads {} {}{place}: only its inputs, the incoming adjoint and the input tangents may be free in a rule ([{}])", self.str_field(n, "role")?, py_str(&leaf), shown.join(", ")))
    }

    /// Records the index and carries loop `lp` owns and queues its body, which is walked as inside the loop.
    fn enter_loop(
        &self,
        who: &str,
        lp: u32,
        loops: &[u32],
        owned: &mut HashMap<u32, HashSet<u32>>,
        stack: &mut Vec<Visit>,
    ) -> Res<()> {
        if !self.refs(lp, "body_sinks")?.is_empty() {
            return refuse(format!(
                "{who}: a loop in a rule template may not write a plane (it has body sinks)"
            ));
        }
        let mut own: HashSet<u32> = self.refs(lp, "index")?.into_iter().collect();
        own.extend(self.refs(lp, "carries")?);
        owned.insert(lp, own);
        let mut inner = loops.to_vec();
        inner.push(lp);
        stack.extend(self.loop_edges(lp)?.into_iter().map(|c| (c, inner.clone())));
        Ok(())
    }
}

/// Checks that the graph under `roots` is a closed rule template of the primitive `name` whose free leaves are the names in
/// `allowed` (separated by white space).
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_rule_template_check(
    ctx: *mut Ctx,
    name: *const c_char,
    roots: *const u32,
    nroots: usize,
    allowed: *const c_char,
    allowed_len: usize,
) -> c_int {
    guard(ctx, |c| {
        let bytes = slice(allowed.cast::<u8>(), allowed_len, "allowed")?;
        let text = std::str::from_utf8(bytes).or_else(|_| badarg("allowed is not UTF-8"))?;
        let names: Vec<String> = text.split_whitespace().map(String::from).collect();
        c.rule_template_check(cstr(name, "name")?, slice(roots, nroots, "roots")?, &names)
    })
}

/// Whether a name is taken as a kind name (`*out` is 0 or 1).
///
/// # Safety
/// Every pointer must be valid, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_kind_reserved(
    ctx: *mut Ctx,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return badarg("out is NULL");
        }
        unsafe { *out = c_int::from(c.kind_reserved(cstr(name, "name")?)) };
        Ok(())
    })
}
