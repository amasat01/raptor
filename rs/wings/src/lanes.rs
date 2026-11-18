//! The key run of a keyed sink: how its key moves over runs of consecutive lanes, from an affine algebra over the axes of the lane space.
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::collections::{BTreeMap, HashMap};

/// The result of an analysis: its answer, or the refusal that stopped it.
type R<T> = Result<T, Fault>;

/// An exact fraction, lowest terms, positive denominator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Frac {
    /// The numerator.
    pub n: i128,
    /// The denominator, above zero.
    pub d: i128,
}
/// The greatest common divisor of two integers.
fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}
impl Frac {
    /// The fraction `n / d` in lowest terms.
    pub fn new(n: i128, d: i128) -> Frac {
        let g = gcd(n, d).max(1);
        let s = if d < 0 { -1 } else { 1 };
        Frac {
            n: s * n / g,
            d: s * d / g,
        }
    }
    /// The whole number `n` as a fraction.
    fn int(n: i128) -> Frac {
        Frac { n, d: 1 }
    }
    /// The sum of two fractions.
    fn add(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }
    /// The difference of two fractions.
    fn sub(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d - o.n * self.d, self.d * o.d)
    }
    /// The product of two fractions.
    fn mul(self, o: Frac) -> Frac {
        Frac::new(self.n * o.n, self.d * o.d)
    }
    /// The quotient of two fractions.
    fn div(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d, self.d * o.n)
    }
    /// Whether this fraction is less than `o`.
    fn lt(self, o: Frac) -> bool {
        self.n * o.d < o.n * self.d
    }
    /// The largest whole number not above the fraction.
    fn floor(self) -> i128 {
        self.n.div_euclid(self.d)
    }
    /// The fraction with its fractional part dropped, towards zero.
    fn trunc(self) -> i128 {
        self.n / self.d
    }
    /// Whether the fraction is a whole number.
    fn is_int(self) -> bool {
        self.d == 1
    }
    /// Whether the fraction is zero.
    fn zero(self) -> bool {
        self.n == 0
    }
    /// The remainder of a fraction, with the sign of the divisor.
    fn rem(self, o: Frac) -> Frac {
        self.sub(o.mul(Frac::int(self.div(o).floor())))
    }
}

/// The form of an index that does not move along an axis.
pub const INVARIANT: &str = "invariant";
/// The form of an index that is a whole-number multiple of the lane.
const AFFINE: &str = "affine";
/// The form of an index that is a fraction of the lane, floored.
const QUOTIENT: &str = "quotient";
/// The form of an index that depends on loaded data.
const DATA_DEPENDENT: &str = "data_dependent";
/// The form of an index that is none of the others.
const NON_AFFINE: &str = "non_affine";

/// An axis: its extent (None for the runtime first axis) and the flat-lane distance of one step.
pub(crate) struct Axis {
    /// How many steps the axis has; none for the run-time first axis.
    extent: Option<i128>,
    /// The flat-lane distance of one step.
    stride: i128,
}

/// An index's value over the axes: affine coefficients, constant, axes it varies along non-affinely, data it carries.
#[derive(Clone, Default)]
pub(crate) struct V {
    /// The coefficient of each axis the value is affine along.
    lin: Vec<(usize, Frac)>,
    /// The constant term, when it is known.
    cst: Option<Frac>,
    /// The axes the value varies along non-affinely, one bit each.
    bad: u64,
    /// Whether the value carries loaded data.
    data: bool,
    /// The axes the value's data part varies along, one bit each.
    dbad: u64,
}

/// The axes of the coefficients as one bit each.
fn bits(lin: &[(usize, Frac)]) -> u64 {
    lin.iter().fold(0, |m, (a, _)| m | 1 << a)
}
/// A value from its parts, dropping zero coefficients.
fn mk(coefs: BTreeMap<usize, Frac>, cst: Option<Frac>, bad: u64, data: bool, dbad: u64) -> V {
    V {
        lin: coefs.into_iter().filter(|(_, c)| !c.zero()).collect(),
        cst,
        bad,
        data,
        dbad,
    }
}
impl V {
    /// The axes the value varies along, in any way.
    fn varies(&self) -> u64 {
        bits(&self.lin) | self.bad | self.dbad
    }
    /// The axes the value varies along through its coefficients or non-affinely.
    fn lin_axes(&self) -> u64 {
        bits(&self.lin) | self.bad
    }
    /// The coefficients by axis.
    fn coefs(&self) -> BTreeMap<usize, Frac> {
        self.lin.iter().cloned().collect()
    }
}

/// The evaluation of nodes as values over the lane axes, with what it found so far.
pub(crate) struct Eval<'a> {
    /// The context the nodes are read from.
    c: &'a Ctx,
    /// The axes the lanes run along.
    axes: Vec<Axis>,
    /// The value already found for a node.
    memo: HashMap<u32, V>,
    /// The set of all axes, one bit each.
    all: u64,
}

/// The whole number a literal holds, if it holds one.
fn lit_int(v: &Val) -> Option<i128> {
    match v {
        Val::Int(i) => Some(*i),
        Val::Float(f) => Some(*f as i128),
        Val::Bool(b) => Some(*b as i128),
        _ => None,
    }
}
/// The number a literal holds as a float, if it holds one.
fn lit_f(v: &Val) -> Option<f64> {
    match v {
        Val::Int(i) => Some(*i as f64),
        Val::Float(f) => Some(*f),
        _ => None,
    }
}

impl<'a> Eval<'a> {
    /// Whether the node is of the class.
    fn is_a(&self, h: u32, k: &str) -> bool {
        self.c.table.is_a(self.c.nodes[h as usize].class, k)
    }
    /// The operands of the node.
    fn ops(&self, h: u32) -> &'a [u32] {
        &self.c.nodes[h as usize].ops
    }
    /// The kind of the node when it is an op.
    fn kind(&self, h: u32) -> Option<String> {
        if self.is_a(h, "Op") {
            self.c.str_field(h, "kind").ok()
        } else {
            None
        }
    }
    /// The index of a read, before its clamp.
    fn raw_index(&self, at: u32) -> u32 {
        self.peel(self.ops(at)[1])
    }
    /// The node without a clamp around it.
    fn peel(&self, h: u32) -> u32 {
        if self.kind(h).as_deref() == Some("clamp_index") {
            self.ops(h)[0]
        } else {
            h
        }
    }
    /// The element type name of the node.
    fn dtype(&self, h: u32) -> &str {
        &self.c.nodes[h as usize].dtype
    }
    /// Whether the node holds integers.
    fn is_int_dtype(&self, h: u32) -> bool {
        matches!(self.dtype(h), "i32" | "i64")
    }
    /// The value of the lane number itself.
    fn lane(&self) -> V {
        mk(
            self.axes
                .iter()
                .enumerate()
                .map(|(i, a)| (i, Frac::int(a.stride)))
                .collect(),
            Some(Frac::int(0)),
            0,
            false,
            0,
        )
    }

    /// The nodes whose values the node's value is made from.
    fn children(&self, n: u32) -> Vec<u32> {
        if self.is_a(n, "At") {
            vec![self.raw_index(n)]
        } else if self.is_a(n, "LoopIndex") || self.is_a(n, "LoopCarry") || self.is_a(n, "Leaf") {
            vec![]
        } else {
            self.ops(n).to_vec()
        }
    }

    /// The value of a node over the axes, evaluating what it needs below it first.
    fn value(&mut self, root: u32) -> R<V> {
        let mut stack = vec![(root, false)];
        while let Some((n, expanded)) = stack.pop() {
            if self.memo.contains_key(&n) {
                continue;
            }
            if !expanded {
                stack.push((n, true));
                for ch in self.children(n) {
                    if !self.memo.contains_key(&ch) {
                        stack.push((ch, false))
                    }
                }
                continue;
            }
            let v = self.eval(n)?;
            self.memo.insert(n, v);
        }
        Ok(self.memo[&root].clone())
    }

    /// The value of one node from the values of its children.
    fn eval(&self, n: u32) -> R<V> {
        let c = self.c;
        if self.is_a(n, "SampleIndex") {
            return Ok(self.lane());
        }
        if self.is_a(n, "Const") {
            if self.is_int_dtype(n) {
                let l = lit_int(&c.field(n, "literal")?)
                    .ok_or(Fault::Refused("const literal is not a number".into()))?;
                return Ok(mk(Default::default(), Some(Frac::int(l)), 0, false, 0));
            }
            return Ok(mk(Default::default(), None, 0, false, 0));
        }
        if self.is_a(n, "LoopIndex") {
            return Ok(mk(Default::default(), None, 0, false, 0));
        }
        if self.is_a(n, "LoopCarry") {
            return Ok(mk(Default::default(), None, self.all, false, 0));
        }
        if self.is_a(n, "Leaf") {
            let role = c.str_field(n, "role")?;
            return Ok(if role == "uniform" || role == "nsamples" {
                mk(Default::default(), None, 0, false, 0)
            } else {
                mk(Default::default(), None, 0, true, self.all)
            });
        }
        if self.is_a(n, "At") {
            let idx = &self.memo[&self.raw_index(n)];
            return Ok(mk(Default::default(), None, 0, true, idx.varies()));
        }
        if let Some(k) = self.kind(n) {
            if k == "clamp_index" {
                return Ok(self.memo[&self.ops(n)[0]].clone());
            }
            let ops: Vec<&V> = self.ops(n).iter().map(|o| &self.memo[o]).collect();
            if (k == "add" || k == "sub") && ops.len() == 2 {
                return Ok(self.addsub(k == "add", ops[0], ops[1]));
            }
            if k == "mul" && ops.len() == 2 {
                return Ok(self.mul(ops[0], ops[1]));
            }
            if k == "div" && ops.len() == 2 && self.is_int_dtype(n) {
                return Ok(self.div(ops[0], ops[1]));
            }
        }
        let ops: Vec<&V> = self
            .ops(n)
            .iter()
            .filter_map(|o| self.memo.get(o))
            .collect();
        Ok(mk(
            Default::default(),
            None,
            ops.iter().fold(0, |m, o| m | o.lin_axes()),
            ops.iter().any(|o| o.data),
            ops.iter().fold(0, |m, o| m | o.dbad),
        ))
    }

    /// The sum (or difference) of two values.
    fn addsub(&self, add: bool, a: &V, b: &V) -> V {
        let mut coefs = a.coefs();
        for (ax, c) in &b.lin {
            let cur = coefs.get(ax).copied().unwrap_or(Frac::int(0));
            coefs.insert(*ax, if add { cur.add(*c) } else { cur.sub(*c) });
        }
        let cst = match (a.cst, b.cst) {
            (Some(x), Some(y)) => Some(if add { x.add(y) } else { x.sub(y) }),
            _ => None,
        };
        if coefs.values().any(|c| !c.is_int()) && !a.lin.is_empty() && !b.lin.is_empty() {
            return mk(
                Default::default(),
                None,
                a.lin_axes() | b.lin_axes(),
                a.data || b.data,
                a.dbad | b.dbad,
            );
        }
        mk(coefs, cst, a.bad | b.bad, a.data || b.data, a.dbad | b.dbad)
    }

    /// The product of two values.
    fn mul(&self, a: &V, b: &V) -> V {
        for (x, y) in [(a, b), (b, a)] {
            if x.varies() == 0 {
                if let Some(k) = x.cst {
                    if k.zero() {
                        return mk(Default::default(), Some(Frac::int(0)), 0, false, 0);
                    }
                    if y.lin.iter().any(|(_, c)| !c.is_int()) && !k.is_int() {
                        break;
                    }
                    return mk(
                        y.lin.iter().map(|(ax, c)| (*ax, c.mul(k))).collect(),
                        y.cst.map(|c| c.mul(k)),
                        y.bad,
                        y.data || x.data,
                        y.dbad,
                    );
                }
            }
        }
        if a.varies() | b.varies() == 0 {
            let cst = match (a.cst, b.cst) {
                (Some(x), Some(y)) => Some(x.mul(y)),
                _ => None,
            };
            return mk(
                Default::default(),
                cst,
                0,
                a.data || b.data,
                a.dbad | b.dbad,
            );
        }
        mk(
            Default::default(),
            None,
            a.lin_axes() | b.lin_axes(),
            a.data || b.data,
            a.dbad | b.dbad,
        )
    }

    /// The quotient of two values.
    fn div(&self, a: &V, b: &V) -> V {
        let zero = Frac::int(0);
        let k = match b.cst {
            Some(k) if b.varies() == 0 && !k.lt(Frac::int(0)) && !k.zero() => k,
            _ => {
                return mk(
                    Default::default(),
                    None,
                    a.lin_axes() | b.lin_axes(),
                    a.data || b.data,
                    a.dbad | b.dbad,
                )
            }
        };
        if a.varies() == 0 {
            let cst = match a.cst {
                Some(x) if !x.lt(zero) => Some(Frac::int(x.trunc().div_euclid(k.trunc()))),
                _ => None,
            };
            return mk(Default::default(), cst, 0, a.data, a.dbad);
        }
        let ac = match a.cst {
            Some(x) if a.bad == 0 && a.dbad == 0 && !x.lt(zero) => x,
            _ => return mk(Default::default(), None, a.lin_axes(), a.data, a.dbad),
        };
        let (mut whole, mut rest_max): (Option<BTreeMap<usize, Frac>>, Frac) =
            (Some(BTreeMap::new()), zero);
        for (ax, c) in &a.lin {
            if c.is_int() && c.rem(k).zero() {
                whole.as_mut().unwrap().insert(*ax, c.div(k));
                continue;
            }
            match self.axes[*ax].extent {
                Some(e) if !c.lt(zero) && c.is_int() => {
                    rest_max = rest_max.add(c.mul(Frac::int(e - 1)))
                }
                _ => {
                    whole = None;
                    break;
                }
            }
        }
        if let Some(w) = whole {
            if ac.add(rest_max).lt(k) {
                return mk(
                    w,
                    if ac.lt(k) { Some(zero) } else { None },
                    0,
                    a.data,
                    a.dbad,
                );
            }
        }
        if a.lin.len() == 1 {
            let (ax, c) = a.lin[0];
            return mk(
                [(ax, c.div(k))].into(),
                Some(Frac::int(ac.trunc().div_euclid(k.trunc()))),
                0,
                a.data,
                a.dbad,
            );
        }
        mk(Default::default(), None, a.lin_axes(), a.data, a.dbad)
    }

    /// `G` when the node is the clamp `cast_i32(min(max(_, 0), G-1))`.
    fn clamp_bound(&self, n: u32) -> R<Option<i128>> {
        let c = self.c;
        if self.kind(n).as_deref() != Some("cast")
            || c.field(n, "literal")? != Val::Str("i32".into())
        {
            return Ok(None);
        }
        let mn = self.ops(n)[0];
        if self.kind(mn).as_deref() != Some("min") || !self.is_a(self.ops(mn)[1], "Const") {
            return Ok(None);
        }
        let mx = self.ops(mn)[0];
        if self.kind(mx).as_deref() != Some("max")
            || !self.is_a(self.ops(mx)[1], "Const")
            || lit_f(&c.field(self.ops(mx)[1], "literal")?) != Some(0.0)
        {
            return Ok(None);
        }
        let top = lit_f(&c.field(self.ops(mn)[1], "literal")?).unwrap_or(f64::NAN);
        Ok(if top == top.trunc() && top >= 0.0 {
            Some(top as i128 + 1)
        } else {
            None
        })
    }

    /// The plane leaf a read of an index plane goes through.
    fn stored(&self, n: u32) -> u32 {
        if self.kind(n).as_deref() == Some("cast")
            && self.dtype(n) == "i32"
            && self.dtype(self.ops(n)[0]) == "int32"
        {
            self.ops(n)[0]
        } else {
            n
        }
    }

    /// Whether the node reads an index plane.
    fn index_plane_read(&mut self, n: u32) -> R<bool> {
        let s = self.stored(n);
        let own = s == n;
        let ok = |d: &str| {
            if own {
                d == "i32" || d == "i64"
            } else {
                d == "int32"
            }
        };
        if self.is_a(s, "Leaf") {
            return Ok(self.c.str_field(s, "role")? == "per_sample" && ok(self.dtype(s)));
        }
        if !self.is_a(s, "At") {
            return Ok(false);
        }
        let plane = self.ops(s)[0];
        if !(ok(self.dtype(plane)) && self.c.str_field(plane, "role")? == "lookup") {
            return Ok(false);
        }
        let ri = self.raw_index(s);
        Ok(!self.value(ri)?.data)
    }

    /// The additive terms of an index: data-free values with their signs, clamp slots, index-plane reads, and a count of the rest.
    fn split_terms(&mut self, key: u32) -> R<IndexTerms> {
        let (mut free, mut slots, mut planes, mut other) = (vec![], vec![], vec![], 0);
        let mut stack = vec![(key, 1i32)];
        while let Some((n, sign)) = stack.pop() {
            let n = self.peel(n);
            let k = self.kind(n);
            let ops = self.ops(n);
            if k.as_deref() == Some("add") && ops.len() == 2 {
                stack.push((ops[1], sign));
                stack.push((ops[0], sign));
                continue;
            }
            if k.as_deref() == Some("sub") && ops.len() == 2 && !self.value(ops[1])?.data {
                stack.push((ops[1], -sign));
                stack.push((ops[0], sign));
                continue;
            }
            let v = self.value(n)?;
            let g = self.clamp_bound(n)?;
            if !v.data {
                free.push((v, sign))
            } else if let Some(g) = g {
                slots.push((g, sign))
            } else if self.index_plane_read(n)? {
                planes.push((self.stored(n), sign))
            } else {
                other += 1
            }
        }
        Ok(IndexTerms {
            free,
            slots,
            planes,
            other,
        })
    }

    /// Whether the key is read from a plane declared sorted .
    fn key_origin_sorted(&mut self, key: u32) -> R<bool> {
        let k = self.peel(key);
        let IndexTerms {
            free,
            slots,
            planes,
            other,
        } = self.split_terms(k)?;
        let mut base = mk(Default::default(), Some(Frac::int(0)), 0, false, 0);
        for (v, sign) in &free {
            base = self.addsub(*sign > 0, &base, v)
        }
        if !(planes.len() == 1 && slots.is_empty() && other == 0 && base.bad == 0) {
            return Ok(false);
        }
        let (leaf, sign) = planes[0];
        let invariant = axis_forms(&base, self.axes.len())
            .iter()
            .all(|f| f.0 == INVARIANT);
        Ok(self.is_a(leaf, "Leaf")
            && self.c.field(leaf, "sorted")? == Val::Bool(true)
            && sign > 0
            && invariant)
    }
}

/// How an index moves along one axis: its form name and its step when it has one.
pub type Form = (&'static str, Option<Frac>);

/// The additive terms of an index key, as split by `split_terms`.
struct IndexTerms {
    /// Data-free values with their signs.
    free: Vec<(V, i32)>,
    /// Clamp slots with their signs.
    slots: Vec<(i128, i32)>,
    /// Index-plane reads (the leaf) with their signs.
    planes: Vec<(u32, i32)>,
    /// How many terms are none of the above.
    other: usize,
}

/// The form of the value along each of the first `n` axes.
fn axis_forms(v: &V, n: usize) -> Vec<Form> {
    let coefs = v.coefs();
    (0..n)
        .map(|i| {
            if v.dbad >> i & 1 == 1 {
                (DATA_DEPENDENT, None)
            } else if v.bad >> i & 1 == 1 {
                (NON_AFFINE, None)
            } else if let Some(c) = coefs.get(&i) {
                (if c.is_int() { AFFINE } else { QUOTIENT }, Some(*c))
            } else {
                (INVARIANT, None)
            }
        })
        .collect()
}

/// How a keyed sink's key moves over runs of consecutive lanes: `form` is "proven", "sorted" or "tested".
#[derive(Debug, PartialEq)]
pub struct KeyRun {
    /// How the key moves: proven, sorted or tested.
    pub form: &'static str,
    /// How many consecutive lanes the run covers.
    pub width: i128,
    /// How the key moves along each axis.
    pub forms: Vec<Form>,
    /// Whether the key is constant over a whole block of lanes.
    pub block: bool,
}

/// Every node reachable from the sinks, loop bodies included.
pub(crate) fn reachable(c: &Ctx, sinks: &[u32]) -> R<Vec<u32>> {
    let (mut seen, mut stack, mut found) = (
        std::collections::HashSet::new(),
        sinks.iter().rev().copied().collect::<Vec<_>>(),
        vec![],
    );
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        c.node(n)?;
        found.push(n);
        let nd = &c.nodes[n as usize];
        stack.extend(nd.ops.iter().copied());
        if c.table.is_a(nd.class, "Loop") {
            let body_sinks = match c.field(n, "body_sinks")? {
                Val::Tuple(t) | Val::List(t) => t,
                _ => vec![],
            };
            stack.extend(body_sinks.iter().filter_map(|v| {
                if let Val::Node(x) = v {
                    Some(*x)
                } else {
                    None
                }
            }));
            stack.extend(c.body_nodes(n)?);
        }
    }
    Ok(found)
}

/// The nodes of the kernel whose value may differ between lanes that differ only in lane axis `axis` (unknown means it may).
pub fn varying_nodes(c: &Ctx, sinks: &[u32], axis: usize) -> R<std::collections::HashSet<u32>> {
    let axes = axes_of(c, sinks)?;
    let n = axes.len();
    let mut ev = Eval {
        c,
        axes,
        memo: HashMap::new(),
        all: (1u64 << n) - 1,
    };
    let mut varying = std::collections::HashSet::new();
    for node in reachable(c, sinks)? {
        if ev.value(node)?.varies() >> axis & 1 == 1 {
            varying.insert(node);
        }
    }
    Ok(varying)
}

/// The axes of the lane space from the kernel's `lane / k` divisors: a chain of divisors gives one axis each, anything else one flat axis.
pub(crate) fn axes_of(c: &Ctx, sinks: &[u32]) -> R<Vec<Axis>> {
    let mut ks: Vec<i128> = vec![];
    for n in reachable(c, sinks)? {
        let nd = &c.nodes[n as usize];
        if c.table.is_a(nd.class, "Op")
            && c.str_field(n, "kind")? == "div"
            && nd.ops.len() == 2
            && c.table
                .is_a(c.nodes[nd.ops[0] as usize].class, "SampleIndex")
        {
            let d = nd.ops[1];
            if c.table.is_a(c.nodes[d as usize].class, "Const")
                && matches!(c.nodes[d as usize].dtype.as_str(), "i32" | "i64")
            {
                if let Some(k) = lit_int(&c.field(d, "literal")?) {
                    if k > 0 && !ks.contains(&k) {
                        ks.push(k)
                    }
                }
            }
        }
    }
    ks.sort_by(|a, b| b.cmp(a));
    let chain = ks.windows(2).all(|w| w[0] % w[1] == 0);
    if ks.is_empty() || !chain {
        return Ok(vec![Axis {
            extent: None,
            stride: 1,
        }]);
    }
    let strides: Vec<i128> = ks.iter().copied().chain([1]).collect();
    Ok((0..strides.len())
        .map(|i| Axis {
            extent: if i == 0 {
                None
            } else {
                Some(strides[i - 1] / strides[i])
            },
            stride: strides[i],
        })
        .collect())
}

/// A lane axis along which some plane is read at the same place by lanes that differ only in that axis, so one thread can serve
/// several of them from one read.
pub(crate) struct BoxCandidate {
    /// The axis's position among the lane axes.
    pub(crate) axis: usize,
    /// How many steps the axis has; none for the run-time first axis.
    pub(crate) extent: Option<i128>,
    /// The flat-lane distance of one step.
    pub(crate) stride: i128,
}

/// The first axis, other than the fastest, along which every read of some plane is invariant; none when there is no such axis.
pub(crate) fn box_candidate(c: &Ctx, sinks: &[u32]) -> R<Option<BoxCandidate>> {
    let axes = axes_of(c, sinks)?;
    let n = axes.len();
    let mut ev = Eval {
        c,
        axes,
        memo: HashMap::new(),
        all: (1u64 << n) - 1,
    };
    // for each plane read, the axes along which every read site so far is invariant
    let mut planes: Vec<(String, Vec<bool>)> = vec![];
    for node in reachable(c, sinks)? {
        if !c.table.is_a(c.nodes[node as usize].class, "At") {
            continue;
        }
        let forms = axis_forms(&ev.value(ev.raw_index(node))?, n);
        let plane = ev.ops(node)[0];
        let name = c
            .str_field(plane, "name")
            .unwrap_or_else(|_| format!("#{plane}"));
        let held: Vec<bool> = forms.iter().map(|f| f.0 == INVARIANT).collect();
        match planes.iter_mut().find(|p| p.0 == name) {
            Some(known) => known.1.iter_mut().zip(held).for_each(|(a, b)| *a &= b),
            None => planes.push((name, held)),
        }
    }
    let found = (0..n - 1).find(|&i| planes.iter().any(|p| p.1[i]));
    Ok(found.map(|axis| BoxCandidate {
        axis,
        extent: ev.axes[axis].extent,
        stride: ev.axes[axis].stride,
    }))
}

/// The key run of one keyed sink of the kernel whose sinks are `sinks`; None for an own-column sink or a key with no run.
pub fn key_run(
    c: &Ctx,
    sinks: &[u32],
    sink: u32,
    run_max: i128,
    block_run: i128,
) -> R<Option<KeyRun>> {
    c.node(sink)?;
    let key = match c.field(sink, "index") {
        Ok(Val::Node(k)) => k,
        Ok(Val::None) => return Ok(None),
        Ok(_) => return refuse("a sink's index is not a node"),
        Err(e) => return Err(e),
    };
    let axes = axes_of(c, sinks)?;
    let n = axes.len();
    let strides: Vec<i128> = axes.iter().map(|a| a.stride).collect();
    let mut ev = Eval {
        c,
        axes,
        memo: HashMap::new(),
        all: (1u64 << n) - 1,
    };
    let v = ev.value(key)?;
    let forms = axis_forms(&v, n);
    if ev.key_origin_sorted(key)? {
        return Ok(Some(KeyRun {
            form: "sorted",
            width: run_max,
            forms,
            block: false,
        }));
    }
    if v.dbad != 0 || (v.data && v.varies() != 0) {
        return Ok(None);
    }
    let run_axes = |w: i128| (0..n).filter(|&i| strides[i] % w != 0).collect::<Vec<_>>();
    let mut width = run_max;
    while width >= 2 {
        if run_axes(width).iter().all(|&i| forms[i].0 == INVARIANT) {
            let block =
                width == run_max && run_axes(block_run).iter().all(|&i| forms[i].0 == INVARIANT);
            return Ok(Some(KeyRun {
                form: "proven",
                width,
                forms,
                block,
            }));
        }
        width /= 2;
    }
    Ok(if matches!(forms[n - 1].0, INVARIANT | QUOTIENT) {
        Some(KeyRun {
            form: "tested",
            width: run_max,
            forms,
            block: false,
        })
    } else {
        None
    })
}

pub mod slice;
pub mod warp;

#[cfg(test)]
mod tests {
    //! The axes, the forms and the sorted-key test of the lane algebra, each on a small hand-made kernel over `lane`.
    //! A split of the lane by `n` is `lane / n` for the slow part and `lane - (lane / n) * n` for the fast part.
    use super::*;

    /// A context and the node builders the kernels below share.
    struct Kernel {
        c: Ctx,
        lane: u32,
    }

    impl Kernel {
        fn new() -> Kernel {
            let mut c = Ctx::new().unwrap();
            let lane = c.make("SampleIndex", "i32", &[], None, &[], "()").unwrap();
            Kernel { c, lane }
        }

        fn int(&mut self, n: i64) -> u32 {
            let fields = format!("(('literal', {n}),)");
            self.c
                .make("Const", "i32", &[], None, &[], &fields)
                .unwrap()
        }

        fn op(&mut self, kind: &str, ops: &[u32]) -> u32 {
            let fields = format!("(('kind', '{kind}'), ('literal', None))");
            self.c.make("Op", "i32", &[], None, ops, &fields).unwrap()
        }

        fn leaf(&mut self, name: &str, sorted: bool) -> u32 {
            let sorted = if sorted { "True" } else { "False" };
            let fields = format!(
                "(('kind', 'vocab_read'), ('role', 'per_sample'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', {sorted}))"
            );
            self.c.make("Leaf", "i32", &[], None, &[], &fields).unwrap()
        }

        fn quotient(&mut self, n: i64) -> u32 {
            let n = self.int(n);
            self.op("div", &[self.lane, n])
        }

        /// A sink that stores a read of a plane at `index`.
        fn read_at(&mut self, index: u32) -> u32 {
            let plane = self.leaf("c", false);
            let read = self
                .c
                .make("At", "i32", &[], None, &[plane, index], "()")
                .unwrap();
            let fields = "(('kind', 'assign'), ('name', 'y'), ('role', 'mutable'))";
            self.c
                .make("Assign", "i32", &[], None, &[read], fields)
                .unwrap()
        }

        fn eval(&self, sinks: &[u32]) -> Eval<'_> {
            let axes = axes_of(&self.c, sinks).unwrap();
            let all = (1u64 << axes.len()) - 1;
            Eval {
                c: &self.c,
                axes,
                memo: HashMap::new(),
                all,
            }
        }

        /// The (extent, stride) of every axis of the kernel whose sinks are `sinks`.
        fn axes(&self, sinks: &[u32]) -> Vec<(Option<i128>, i128)> {
            let axes = axes_of(&self.c, sinks).unwrap();
            axes.iter().map(|a| (a.extent, a.stride)).collect()
        }

        /// The form of `index` along each axis of the kernel whose sinks are `sinks`.
        fn forms(&self, sinks: &[u32], index: u32) -> Vec<Form> {
            let mut ev = self.eval(sinks);
            let value = ev.value(index).unwrap();
            axis_forms(&value, ev.axes.len())
        }
    }

    #[test]
    fn a_kernel_with_no_divisor_has_one_run_time_axis_of_stride_one() {
        let mut k = Kernel::new();
        let one = k.int(1);
        let index = k.op("add", &[k.lane, one]);
        let sink = k.read_at(index);
        assert_eq!(k.axes(&[sink]), [(None, 1)]);
    }

    #[test]
    fn a_split_by_n_makes_a_run_time_slow_axis_and_a_fast_axis_of_n_lanes() {
        let mut k = Kernel::new();
        let slow = k.quotient(256);
        let sink = k.read_at(slow);
        assert_eq!(k.axes(&[sink]), [(None, 256), (Some(256), 1)]);
    }

    #[test]
    fn a_chain_of_divisors_makes_axes_whose_extents_are_the_ratios_of_neighbouring_strides() {
        // strides 32, 8, 1: the axis of stride 8 has 32 / 8 = 4 steps and the fastest has 8 / 1 = 8
        let mut k = Kernel::new();
        let (a, b) = (k.quotient(32), k.quotient(8));
        let both = k.op("add", &[a, b]);
        let sink = k.read_at(both);
        assert_eq!(k.axes(&[sink]), [(None, 32), (Some(4), 8), (Some(8), 1)]);
    }

    #[test]
    fn divisors_that_do_not_divide_each_other_leave_one_flat_axis() {
        let mut k = Kernel::new();
        let (a, b) = (k.quotient(4), k.quotient(6));
        let both = k.op("add", &[a, b]);
        let sink = k.read_at(both);
        assert_eq!(k.axes(&[sink]), [(None, 1)]);
    }

    #[test]
    fn a_lane_divided_by_a_lone_divisor_is_affine_along_the_slow_axis_and_invariant_along_the_fast_one(
    ) {
        let mut k = Kernel::new();
        let index = k.quotient(4);
        let sink = k.read_at(index);
        let one = Some(Frac::int(1));
        assert_eq!(k.forms(&[sink], index), [(AFFINE, one), (INVARIANT, None)]);
    }

    #[test]
    fn a_fraction_of_the_flat_lane_is_a_quotient_with_that_coefficient() {
        // lane / 4 and lane / 6 do not form a chain, so each is read along the one flat axis with its own fraction
        let mut k = Kernel::new();
        let (by_four, by_six) = (k.quotient(4), k.quotient(6));
        let both = k.op("add", &[by_four, by_six]);
        let sink = k.read_at(both);
        assert_eq!(
            k.forms(&[sink], by_four),
            [(QUOTIENT, Some(Frac::new(1, 4)))]
        );
        assert_eq!(
            k.forms(&[sink], by_six),
            [(QUOTIENT, Some(Frac::new(1, 6)))]
        );
    }

    #[test]
    fn a_product_of_lane_terms_is_non_affine() {
        let mut k = Kernel::new();
        let square = k.op("mul", &[k.lane, k.lane]);
        let sink = k.read_at(square);
        assert_eq!(k.forms(&[sink], square), [(NON_AFFINE, None)]);
    }

    #[test]
    fn a_lane_shifted_by_a_constant_is_affine_with_the_lane_coefficient() {
        let mut k = Kernel::new();
        let one = k.int(1);
        let (up, down) = (k.op("add", &[k.lane, one]), k.op("sub", &[k.lane, one]));
        let sink = k.read_at(up);
        let coefficient = Some(Frac::int(1));
        assert_eq!(k.forms(&[sink], up), [(AFFINE, coefficient)]);
        assert_eq!(k.forms(&[sink], down), [(AFFINE, coefficient)]);
    }

    #[test]
    fn an_index_read_from_a_plane_is_data_dependent() {
        let mut k = Kernel::new();
        let key = k.leaf("i", false);
        let sink = k.read_at(key);
        assert_eq!(k.forms(&[sink], key), [(DATA_DEPENDENT, None)]);
    }

    /// Whether the key `key` is known to be a sorted plane in the kernel that stores a read at it.
    fn sorted(k: &mut Kernel, key: u32) -> bool {
        let sink = k.read_at(key);
        k.eval(&[sink]).key_origin_sorted(key).unwrap()
    }

    #[test]
    fn a_plane_declared_sorted_is_a_sorted_key_and_an_undeclared_one_is_not() {
        let mut k = Kernel::new();
        let (declared, undeclared) = (k.leaf("key", true), k.leaf("other", false));
        assert!(sorted(&mut k, declared));
        assert!(!sorted(&mut k, undeclared));
    }

    #[test]
    fn a_sorted_plane_moved_by_a_constant_is_still_a_sorted_key() {
        let mut k = Kernel::new();
        let (plane, three) = (k.leaf("key", true), k.int(3));
        let (up, down) = (k.op("add", &[plane, three]), k.op("sub", &[plane, three]));
        assert!(sorted(&mut k, up));
        assert!(sorted(&mut k, down));
    }

    #[test]
    fn a_sorted_plane_scaled_or_reversed_is_not_a_sorted_key() {
        // a product is opaque plane data and `3 - key` runs the order backwards
        let mut k = Kernel::new();
        let (plane, two, three) = (k.leaf("key", true), k.int(2), k.int(3));
        let doubled = k.op("mul", &[plane, two]);
        let reversed = k.op("sub", &[three, plane]);
        assert!(!sorted(&mut k, doubled));
        assert!(!sorted(&mut k, reversed));
    }

    #[test]
    fn a_sorted_plane_plus_a_lane_term_is_not_a_sorted_key() {
        let mut k = Kernel::new();
        let plane = k.leaf("key", true);
        let fast = {
            let (q, n) = (k.quotient(8192), k.int(8192));
            let rows = k.op("mul", &[q, n]);
            k.op("sub", &[k.lane, rows])
        };
        let key = k.op("add", &[plane, fast]);
        assert!(!sorted(&mut k, key));
    }
}
