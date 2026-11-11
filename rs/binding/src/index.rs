//! access.index_form: the affine (and one quotient) form of an index expression in a loop index or the sample index, over exact rationals.
use super::*;

#[derive(Clone, Copy, PartialEq)]
struct Q { n: i128, d: i128 }
fn gcd(a: i128, b: i128) -> i128 { if b == 0 { a.abs() } else { gcd(b, a % b) } }
impl Q {
    fn new(n: i128, d: i128) -> Q { let g = gcd(n, d).max(1); let s = if d < 0 { -1 } else { 1 }; Q { n: s * n / g, d: s * d / g } }
    fn int(n: i128) -> Q { Q { n, d: 1 } }
    fn add(self, o: Q) -> Q { Q::new(self.n * o.d + o.n * self.d, self.d * o.d) }
    fn neg(self) -> Q { Q { n: -self.n, d: self.d } }
    fn mul(self, o: Q) -> Q { Q::new(self.n * o.n, self.d * o.d) }
    fn exact(self) -> bool { self.d == 1 }
    fn lt0(self) -> bool { self.n < 0 }
}
struct F { coef: Q, off: Option<Q>, labels: Vec<Py<PyAny>>, anon: bool, symbol: String }
fn labels_plus(py: Python<'_>, a: &[Py<PyAny>], b: &[Py<PyAny>]) -> Vec<Py<PyAny>> { a.iter().chain(b).map(|x| x.clone_ref(py)).collect() }

struct Ctx<'py, 'a> { py: Python<'py>, var: Option<&'a Bound<'py, PyAny>>, peel: Bound<'py, PyAny> }
impl<'py, 'a> Ctx<'py, 'a> {
    fn is_var(&self, n: &Bound<'py, PyAny>) -> PyResult<bool> { match self.var { None => canon::is_a(n, "SampleIndex"), Some(v) => Ok(n.is(v)) } }
    fn carries(&self, node: &Bound<'py, PyAny>) -> PyResult<bool> {
        let mut pending = vec![node.clone()];
        while let Some(cur) = pending.pop() {
            if self.is_var(&cur)? { return Ok(true) }
            for o in cur.getattr("operands")?.try_iter()? { pending.push(o?) }
        }
        Ok(false)
    }
    /// (value, label) of a compile-time integer node.
    fn int_factor(&self, n: &Bound<'py, PyAny>) -> PyResult<Option<(i128, Option<Py<PyAny>>)>> {
        if canon::is_a(n, "RoleConst")? { let l = n.getattr("label")?; return Ok(Some((n.getattr("literal")?.extract::<i128>()?, if l.is_none() { None } else { Some(l.unbind()) }))) }
        if canon::is_a(n, "Const")? { let dt: String = n.getattr("ttype")?.getattr("dtype")?.extract()?; if dt == "i32" || dt == "i64" { return Ok(Some((n.getattr("literal")?.extract::<i128>()?, None))) } }
        Ok(None)
    }
    fn form(&self, node: &Bound<'py, PyAny>, quotients: bool) -> PyResult<Option<F>> {
        let node = self.peel.call1((node,))?;
        if !self.carries(&node)? {
            return Ok(Some(F { coef: Q::int(0), off: self.int_factor(&node)?.map(|v| Q::int(v.0)), labels: vec![], anon: false, symbol: String::new() }))
        }
        if self.is_var(&node)? { return Ok(Some(F { coef: Q::int(1), off: Some(Q::int(0)), labels: vec![], anon: false, symbol: String::new() })) }
        if !canon::is_a(&node, "Op")? { return Ok(None) }
        let kind: String = node.getattr("kind")?.extract()?;
        let ops: Vec<Bound<'py, PyAny>> = node.getattr("operands")?.try_iter()?.collect::<PyResult<_>>()?;
        match kind.as_str() {
            "add" | "sub" => self.sum_form(&kind, &ops, quotients),
            "mul" => self.product_form(&ops, quotients),
            "div" if quotients => self.quotient_form(&node, &ops),
            _ => Ok(None),
        }
    }
    fn sum_form(&self, kind: &str, ops: &[Bound<'py, PyAny>], quotients: bool) -> PyResult<Option<F>> {
        let (Some(l), Some(r)) = (self.form(&ops[0], quotients)?, self.form(&ops[1], quotients)?) else { return Ok(None) };
        let sign = if kind == "add" { 1 } else { -1 };
        if l.coef.exact() && r.coef.exact() {
            let off = match (l.off, r.off) { (Some(a), Some(b)) => Some(a.add(b.mul(Q::int(sign)))), _ => None };
            let sym = if l.symbol.is_empty() { r.symbol.clone() } else { l.symbol.clone() };
            return Ok(Some(F { coef: l.coef.add(r.coef.mul(Q::int(sign))), off, labels: labels_plus(self.py, &l.labels, &r.labels), anon: l.anon || r.anon, symbol: sym }))
        }
        let right_q = if l.coef.n == 0 { true } else if r.coef.n == 0 { false } else { return Ok(None) };
        if sign == -1 && right_q { return Ok(None) }
        Ok(Some(if right_q { r } else { l }))
    }
    fn product_form(&self, ops: &[Bound<'py, PyAny>], quotients: bool) -> PyResult<Option<F>> {
        let (a, b) = (&ops[0], &ops[1]);
        let (a_has, b_has) = (self.carries(a)?, self.carries(b)?);
        if a_has && b_has { return Ok(None) }
        let (term, fnode) = if a_has { (a, b) } else { (b, a) };
        if canon::is_a(fnode, "Leaf")? && fnode.getattr("kind")?.extract::<String>()? == "nsamples" {
            let Some(inner) = self.form(term, quotients)? else { return Ok(None) };
            if !inner.coef.exact() || !inner.symbol.is_empty() { return Ok(None) }
            return Ok(Some(F { coef: inner.coef, off: None, labels: inner.labels, anon: inner.anon, symbol: "n_samples".into() }))
        }
        let Some((factor, label)) = self.int_factor(fnode)? else { return Ok(None) };
        let Some(inner) = self.form(term, quotients)? else { return Ok(None) };
        let anon = inner.anon || label.is_none();
        let mut labels = inner.labels.iter().map(|x| x.clone_ref(self.py)).collect::<Vec<_>>();
        if let Some(l) = label { labels.push(l) }
        if inner.coef.exact() {
            return Ok(Some(F { coef: inner.coef.mul(Q::int(factor)), off: inner.off.map(|o| o.mul(Q::int(factor))), labels, anon, symbol: String::new() }))
        }
        if factor == 0 { return Ok(Some(F { coef: Q::int(0), off: Some(Q::int(0)), labels, anon, symbol: String::new() })) }
        Ok(Some(F { coef: inner.coef, off: inner.off, labels, anon, symbol: String::new() }))
    }
    fn quotient_form(&self, node: &Bound<'py, PyAny>, ops: &[Bound<'py, PyAny>]) -> PyResult<Option<F>> {
        let dt: String = node.getattr("ttype")?.getattr("dtype")?.extract()?;
        if dt != "i32" && dt != "i64" { return Ok(None) }
        let (num, div) = (&ops[0], &ops[1]);
        if self.carries(div)? { return Ok(None) }
        let Some((k, label)) = self.int_factor(div)? else { return Ok(None) };
        if k <= 0 { return Ok(None) }
        let Some(inner) = self.form(num, true)? else { return Ok(None) };
        let Some(off) = inner.off else { return Ok(None) };
        if !inner.coef.exact() || inner.coef.n <= 0 || off.lt0() { return Ok(None) }
        let anon = inner.anon || label.is_none();
        let mut labels = inner.labels.iter().map(|x| x.clone_ref(self.py)).collect::<Vec<_>>();
        if let Some(l) = label { labels.push(l) }
        let coef = Q::new(inner.coef.n, inner.coef.d * k);
        if coef.exact() { return Ok(Some(F { coef, off: Some(Q::int((off.n / off.d) / k)), labels, anon, symbol: String::new() })) }
        Ok(Some(F { coef, off: Some(Q::new(off.n, off.d * k)), labels, anon, symbol: String::new() }))
    }
}

/// The affine form of `node` in `var` (None: the sample index), or None: an `IndexForm`.
#[pyfunction]
#[pyo3(signature = (node, var=None, quotients=true))]
pub fn index_form<'py>(py: Python<'py>, node: &Bound<'py, PyAny>, var: Option<Bound<'py, PyAny>>, quotients: bool) -> PyResult<Option<Bound<'py, PyAny>>> {
    let peel = py.import("raptor.ir.nodes")?.getattr("peel_clamp")?;
    let ctx = Ctx { py, var: var.as_ref(), peel };
    let Some(f) = ctx.form(node, quotients)? else { return Ok(None) };
    let frac = py.import("fractions")?.getattr("Fraction")?;
    let q = |x: Q| frac.call1((x.n as i64, x.d as i64));
    let off = match f.off { Some(o) => q(o)?, None => py.None().into_bound(py) };
    let cls = py.import("raptor.ir.access")?.getattr("IndexForm")?;
    Ok(Some(cls.call1((q(f.coef)?, off, PyTuple::new(py, f.labels.iter().map(|l| l.bind(py).clone()))?, f.anon, f.symbol))?))
}
