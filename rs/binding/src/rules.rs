//! Derivative factors as templates (core.json `rules`), instantiated over IR nodes: the same nodes the Python rule lambdas built.
//! A template is a name (a bound node), `["c", x]` (an f64 scalar constant), `["let", name, e, body]`, `["zero_like", e]`, `["sel", c, a, b]`,
//! `["ramp", c, like]`, `["entrywise", kind, e...]`, `["abs_d", x]`, or `[kind, e...]` (an op minted by `make`).
use super::*;
use rcore::json::{self, J};

static RULES: OnceLock<J> = OnceLock::new();
fn rules() -> &'static J { RULES.get_or_init(|| json::parse(rcore::MANIFEST).ok().and_then(|m| m.get("rules").cloned()).unwrap_or(J::Null)) }

fn ty<'py>(py: Python<'py>, shape: Vec<i64>, dtype: &str) -> PyResult<Bound<'py, PyAny>> { reg().ttype.bind(py).call1((PyTuple::new(py, shape)?, dtype, py.None())) }
fn shape_of(n: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> { n.getattr("ttype")?.getattr("shape")?.try_iter()?.map(|x| x?.extract::<i64>()).collect() }
fn dtype_of(n: &Bound<'_, PyAny>) -> PyResult<String> { n.getattr("ttype")?.getattr("dtype")?.extract() }
fn cls<'py>(py: Python<'py>, n: &str) -> Bound<'py, PyAny> { reg().by_name.get(n).unwrap().bind(py).clone() }
fn mk<'py>(py: Python<'py>, kind: &str, args: Vec<Bound<'py, PyAny>>, lit: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    typing::make(py, kind, PyTuple::new(py, args)?.as_any(), lit)
}
fn konst<'py>(py: Python<'py>, v: f64, t: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { cls(py, "Const").call1((v, t)) }
fn select<'py>(py: Python<'py>, c: &Bound<'py, PyAny>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>, t: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { cls(py, "Select").call1((c, a, b, t)) }
fn zero_like<'py>(py: Python<'py>, e: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let z = konst(py, 0.0, ty(py, vec![], &dtype_of(e)?)?)?;
    let sh = shape_of(e)?;
    if sh.is_empty() { Ok(z) } else { mk(py, "splat", vec![z], Some(PyTuple::new(py, sh)?.into_any())) }
}
fn entrywise<'py>(py: Python<'py>, kind: &str, nodes: &[Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> {
    let mut shape = vec![];
    for x in nodes { let s = shape_of(x)?; if !s.is_empty() { shape = s; break } }
    if shape.is_empty() { return mk(py, kind, nodes.to_vec(), None) }
    let entries: Vec<Bound<'py, PyAny>> = if shape.len() == 1 { (0..shape[0]).map(|i| i.into_pyobject(py).unwrap().into_any()).collect() }
        else { let mut v = vec![]; for r in 0..shape[0] { for c in 0..shape[1] { v.push(PyTuple::new(py, [r, c])?.into_any()) } } v };
    let mut parts = vec![];
    for k in entries {
        let args = nodes.iter().map(|x| if shape_of(x)?.is_empty() { Ok(x.clone()) } else { mk(py, "component", vec![x.clone()], Some(k.clone())) }).collect::<PyResult<Vec<_>>>()?;
        parts.push(mk(py, kind, args, None)?);
    }
    let lit = if shape.len() == 2 { Some(PyTuple::new(py, shape)?.into_any()) } else { None };
    mk(py, "vec", parts, lit)
}
#[derive(Clone)]
enum Val<'py> { N(Bound<'py, PyAny>), P(Bound<'py, PyAny>), L(Vec<Val<'py>>), R(Vec<(String, Val<'py>)>), I(i64), Nil }
type Env<'py> = Vec<(String, Val<'py>)>;
fn node<'py>(v: Val<'py>) -> PyResult<Bound<'py, PyAny>> { match v { Val::N(n) => Ok(n), _ => Err(he("rules: a node was expected".into())) } }
fn nil_or<'py>(py: Python<'py>, v: Val<'py>) -> Bound<'py, PyAny> { match v { Val::N(n) => n, _ => py.None().into_bound(py) } }
impl<'py> Val<'py> {
    fn node_lit(&self) -> PyResult<Bound<'py, PyAny>> { match self { Val::N(n) => n.getattr("literal"), _ => Err(he("rules: lit of a non-node".into())) } }
}
fn pyobj<'py>(py: Python<'py>, v: Val<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(match v { Val::P(p) => Some(p), Val::I(i) => Some(i.into_pyobject(py)?.into_any()), Val::Nil => None, _ => return Err(he("rules: a literal was expected".into())) })
}
fn entry_list<'py>(py: Python<'py>, shape: &[i64]) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if shape.len() == 1 { return (0..shape[0]).map(|i| Ok(i.into_pyobject(py)?.into_any())).collect() }
    let mut v = vec![]; for r in 0..shape[0] { for c in 0..shape[1] { v.push(PyTuple::new(py, [r, c])?.into_any()) } } Ok(v)
}
fn idx_i(v: &Val<'_>) -> PyResult<i64> { if let Val::I(i) = v { Ok(*i) } else { Err(he("rules: an integer was expected".into())) } }
fn idx(v: &Val<'_>) -> PyResult<usize> { if let Val::I(i) = v { Ok(*i as usize) } else { Err(he("rules: an index was expected".into())) } }
fn lookup<'py>(env: &Env<'py>, name: &str) -> PyResult<Val<'py>> {
    let mut parts = name.split('.');
    let head = parts.next().unwrap_or("");
    let mut cur = env.iter().rev().find(|e| e.0 == head).map(|e| e.1.clone()).ok_or_else(|| he(format!("rules: unbound name {name}")))?;
    for p in parts {
        cur = match cur { Val::R(f) => f.into_iter().find(|x| x.0 == p).map(|x| x.1).ok_or_else(|| he(format!("rules: no field {p} in {name}")))?, _ => return Err(he(format!("rules: {name} is not a record"))) };
    }
    Ok(cur)
}
fn nd<'py>(py: Python<'py>, t: &J, env: &mut Env<'py>) -> PyResult<Bound<'py, PyAny>> { node(ev(py, t, env)?) }
fn ev<'py>(py: Python<'py>, t: &J, env: &mut Env<'py>) -> PyResult<Val<'py>> {
    match t {
        J::Str(n) => lookup(env, n),
        J::Num(v) => Ok(Val::I(*v as i64)),
        J::Null => Ok(Val::Nil),
        J::Arr(a) => {
            let head = a[0].str().unwrap_or("");
            let args = |env: &mut Env<'py>, from: usize| -> PyResult<Vec<Bound<'py, PyAny>>> {
                let mut out = vec![];
                for x in &a[from..] {
                    if let J::Arr(sp) = x { if sp.first().and_then(|h| h.str()) == Some("...") {
                        match ev(py, &sp[1], env)? { Val::L(l) => { for v in l { out.push(node(v)?) } } _ => return Err(he("rules: ... spreads a list".into())) }
                        continue
                    } }
                    out.push(nd(py, x, env)?)
                }
                Ok(out)
            };
            let n = |x: Bound<'py, PyAny>| Ok(Val::N(x));
            match head {
                "c" => { let J::Num(v) = a[1] else { return Err(he("rules: c takes a number".into())) }; n(konst(py, v, ty(py, vec![], "f64")?)?) }
                "let" => { let v = ev(py, &a[2], env)?; env.push((a[1].str().unwrap_or("").to_string(), v)); let r = ev(py, &a[3], env); env.pop(); r }
                "lets" => {
                    let mut pushed = 0;
                    for p in a[1].arr() { let v = ev(py, &p.arr()[1], env)?; env.push((p.arr()[0].str().unwrap_or("").to_string(), v)); pushed += 1 }
                    let r = ev(py, &a[2], env); for _ in 0..pushed { env.pop(); } r
                }
                "list" => Ok(Val::L(a[1..].iter().map(|x| ev(py, x, env)).collect::<PyResult<_>>()?)),
                "rec" => Ok(Val::R(a[1..].iter().map(|p| Ok((p.arr()[0].str().unwrap_or("").to_string(), ev(py, &p.arr()[1], env)?))).collect::<PyResult<_>>()?)),
                "map" => {
                    let cnt = idx_i(&ev(py, &a[2], env)?)?;
                    let mut out = vec![];
                    for i in 0..cnt { env.push((a[1].str().unwrap_or("").to_string(), Val::I(i))); let r = ev(py, &a[3], env); env.pop(); out.push(r?) }
                    Ok(Val::L(out))
                }
                "nth" => { let l = ev(py, &a[1], env)?; let i = idx(&ev(py, &a[2], env)?)?; match l { Val::L(v) => v.get(i).cloned().ok_or_else(|| he("rules: index out of range".into())), _ => Err(he("rules: nth of a non-list".into())) } }
                "k" => { let v = nd(py, &a[1], env)?; let i = idx(&ev(py, &a[2], env)?)?; n(mk(py, "component", vec![v], Some((i as i64).into_pyobject(py)?.into_any()))?) }
                "live" => {   // the tangent of operand x (the callable `t` in the environment), a structural f64 zero when it has none
                    let x = nd(py, &a[1], env)?; let t = lookup(env, "t")?;
                    let Val::N(tf) = t else { return Err(he("rules: live needs a tangent function".into())) };
                    let d = tf.call1((&x,))?;
                    if !d.is_none() { return n(d) }
                    let z = konst(py, 0.0, ty(py, vec![], "f64")?)?; let sh = shape_of(&x)?;
                    n(if sh.is_empty() { z } else { mk(py, "splat", vec![z], Some(PyTuple::new(py, sh)?.into_any()))? })
                }
                "t" => { let x = nd(py, &a[1], env)?; let Val::N(tf) = lookup(env, "t")? else { return Err(he("rules: t needs a tangent function".into())) }; let d = tf.call1((&x,))?; Ok(if d.is_none() { Val::Nil } else { Val::N(d) }) }
                "or" => { let v = ev(py, &a[1], env)?; if matches!(v, Val::Nil) { ev(py, &a[2], env) } else { Ok(v) } }
                "when_any" => { let mut any = false; for x in a[1].arr() { if !matches!(ev(py, x, env)?, Val::Nil) { any = true } } if any { ev(py, &a[2], env) } else { Ok(Val::Nil) } }
                "sumn" => {
                    let mut live = vec![]; for x in &a[1..] { if let Val::N(v) = ev(py, x, env)? { live.push(v) } }
                    if live.is_empty() { return Ok(Val::Nil) }
                    let mut it = live.into_iter(); let mut acc = it.next().unwrap();
                    for x in it { acc = mk(py, "add", vec![acc, x], None)? }
                    n(acc)
                }
                "opl" => { let kind = a[1].str().unwrap_or("").to_string(); let lit = ev(py, &a[2], env)?; let v = args(env, 3)?; n(mk(py, &kind, v, pyobj(py, lit)?)?) }
                "lit" => Ok(Val::P(lookup(env, "n")?.node_lit()?)),
                "shape" => { let x = nd(py, &a[1], env)?; Ok(Val::P(PyTuple::new(py, shape_of(&x)?)?.into_any())) }
                "vlit" => { let Val::P(sh) = ev(py, &a[1], env)? else { return Err(he("rules: vlit takes a shape".into())) }; Ok(Val::P(if sh.len()? == 2 { sh } else { py.None().into_bound(py) })) }
                "entries" => { let Val::P(sh) = ev(py, &a[1], env)? else { return Err(he("rules: entries takes a shape".into())) }; let shape: Vec<i64> = sh.try_iter()?.map(|x| x?.extract()).collect::<PyResult<_>>()?; Ok(Val::L(entry_list(py, &shape)?.into_iter().map(Val::P).collect())) }
                "ifeq" => { let (x, y) = (pyobj(py, ev(py, &a[1], env)?)?, pyobj(py, ev(py, &a[2], env)?)?); let eq = match (x, y) { (Some(x), Some(y)) => x.eq(y)?, (None, None) => true, _ => false }; ev(py, if eq { &a[3] } else { &a[4] }, env) }
                "map_list" => { let Val::L(l) = ev(py, &a[2], env)? else { return Err(he("rules: map_list takes a list".into())) }; let mut out = vec![]; for v in l { env.push((a[1].str().unwrap_or("").to_string(), v)); let r = ev(py, &a[3], env); env.pop(); out.push(r?) } Ok(Val::L(out)) }
                "zipmap" => {
                    let (Val::L(l1), Val::L(l2)) = (ev(py, &a[2], env)?, ev(py, &a[4], env)?) else { return Err(he("rules: zipmap takes two lists".into())) };
                    let mut out = vec![];
                    for (x, y) in l1.into_iter().zip(l2) { env.push((a[1].str().unwrap_or("").to_string(), x)); env.push((a[3].str().unwrap_or("").to_string(), y)); let r = ev(py, &a[5], env); env.pop(); env.pop(); out.push(r?) }
                    Ok(Val::L(out))
                }
                "pair" => Ok(Val::L(vec![ev(py, &a[1], env)?, ev(py, &a[2], env)?])),
                "scalar_zero" => { let e = nd(py, &a[1], env)?; n(konst(py, 0.0, ty(py, vec![], &dtype_of(&e)?)?)?) }
                "call" => {
                    let name = a[1].str().unwrap_or("");
                    let def = rules().get("defs").and_then(|d| d.get(name)).ok_or_else(|| he(format!("rules: no def {name}")))?;
                    let mut fresh: Env<'py> = vec![];
                    for (p, x) in def.get("params").map(|x| x.strs()).unwrap_or_default().into_iter().zip(&a[2..]) { fresh.push((p, ev(py, x, env)?)) }
                    ev(py, def.get("body").unwrap_or(&J::Null), &mut fresh)
                }
                "grid" => {
                    let f = nd(py, &a[1], env)?; let m = shape_of(&f)?[0];
                    let mut parts = vec![];
                    for i in 0..m { for j in 0..m {
                        env.push((a[2].str().unwrap_or("").to_string(), Val::I(i))); env.push((a[3].str().unwrap_or("").to_string(), Val::I(j)));
                        let r = nd(py, &a[4], env); env.pop(); env.pop(); parts.push(r?)
                    } }
                    n(mk(py, "vec", parts, Some(PyTuple::new(py, [m, m])?.into_any()))?)
                }
                "ifcmp" => {
                    let (x, y) = (idx_i(&ev(py, &a[2], env)?)?, idx_i(&ev(py, &a[3], env)?)?);
                    let ok = match a[1].str().unwrap_or("") { "eq" => x == y, "lt" => x < y, "le" => x <= y, "gt" => x > y, _ => x >= y };
                    ev(py, if ok { &a[4] } else { &a[5] }, env)
                }
                "tup" => { let (x, y) = (idx_i(&ev(py, &a[1], env)?)?, idx_i(&ev(py, &a[2], env)?)?); Ok(Val::P(PyTuple::new(py, [x, y])?.into_any())) }
                "rank1" => { let x = nd(py, &a[1], env)?; ev(py, if shape_of(&x)?.len() == 1 { &a[2] } else { &a[3] }, env) }
                "dim" => { let x = nd(py, &a[1], env)?; Ok(Val::I(shape_of(&x)?[0])) }
                "operand" => { let x = nd(py, &a[1], env)?; let i = idx(&ev(py, &a[2], env)?)?; n(x.getattr("operands")?.get_item(i)?) }
                "foldl" => {
                    let kind = a[1].str().unwrap_or("").to_string();
                    let Val::L(l) = ev(py, &a[2], env)? else { return Err(he("rules: foldl takes a list".into())) };
                    let mut it = l.into_iter(); let mut acc = node(it.next().ok_or_else(|| he("rules: foldl of an empty list".into()))?)?;
                    for x in it { acc = mk(py, &kind, vec![acc, node(x)?], None)? }
                    n(acc)
                }
                "ci" => { let J::Num(v) = a[1] else { return Err(he("rules: ci takes an integer".into())) }; n(cls(py, "Const").call1((v as i64, ty(py, vec![], "int32")?))?) }
                "zero_f64" => { let e = nd(py, &a[1], env)?; let z = konst(py, 0.0, ty(py, vec![], "f64")?)?; let sh = shape_of(&e)?; n(if sh.is_empty() { z } else { mk(py, "splat", vec![z], Some(PyTuple::new(py, sh)?.into_any()))? }) }
                "zero_like" => { let e = nd(py, &a[1], env)?; n(zero_like(py, &e)?) }
                "sel" | "where" => { let v = args(env, 1)?; let t = v[1].getattr("ttype")?; n(select(py, &v[0], &v[1], &v[2], t)?) }
                "ramp" => { let v = args(env, 1)?; let t = ty(py, vec![], &dtype_of(&v[1])?)?; n(select(py, &v[0], &konst(py, 1.0, t.clone())?, &konst(py, 0.0, t.clone())?, t)?) }
                "entrywise" => { let kind = a[1].str().unwrap_or("").to_string(); let v = args(env, 2)?; n(entrywise(py, &kind, &v)?) }
                "when" => { let v = ev(py, &a[1], env)?; if matches!(v, Val::Nil) { Ok(v) } else { ev(py, &a[2], env) } }
                "pick" => {
                    let (lo, hi, g) = (nd(py, &a[1], env)?, nd(py, &a[2], env)?, nd(py, &a[3], env)?);
                    let first = matches!(a[4], J::Bool(true));
                    let cond = entrywise(py, "le", &[lo, hi])?;
                    let cs = shape_of(&cond)?;
                    let g = if !cs.is_empty() && shape_of(&g)?.is_empty() { mk(py, "splat", vec![g], Some(PyTuple::new(py, cs)?.into_any()))? } else { g };
                    let z = zero_like(py, &g)?; let t = g.getattr("ttype")?;
                    n(if first { select(py, &cond, &g, &z, t)? } else { select(py, &cond, &z, &g, t)? })
                }
                "abs_d" => {
                    let x = nd(py, &a[1], env)?; let sh = shape_of(&x)?; let f64t = || ty(py, vec![], "f64");
                    if sh.is_empty() {
                        let cond = mk(py, "ge", vec![x.clone(), zero_like(py, &x)?], None)?;
                        return n(select(py, &cond, &konst(py, 1.0, f64t()?)?, &konst(py, -1.0, f64t()?)?, f64t()?)?)
                    }
                    let cond = entrywise(py, "ge", &[x.clone(), konst(py, 0.0, f64t()?)?])?;
                    let splat = |v: f64| -> PyResult<Bound<'py, PyAny>> { mk(py, "splat", vec![konst(py, v, f64t()?)?], Some(PyTuple::new(py, sh.clone())?.into_any())) };
                    n(select(py, &cond, &splat(1.0)?, &splat(-1.0)?, ty(py, sh.clone(), "f64")?)?)
                }
                kind => {
                    let v = args(env, 1)?;
                    if (kind == "add" || kind == "mul") && v.len() > 2 {   // a variadic sum or product folds from the left
                        let mut it = v.into_iter(); let mut acc = it.next().unwrap();
                        for x in it { acc = mk(py, kind, vec![acc, x], None)? }
                        return n(acc)
                    }
                    n(mk(py, kind, v, None)?)
                }
            }
        }
        _ => Err(he("rules: a template is a name, a number, null or an array".into())),
    }
}
fn bind<'py>(env: &Bound<'py, PyDict>) -> PyResult<Env<'py>> { env.iter().map(|(k, v)| Ok((k.extract::<String>()?, if v.is_none() { Val::Nil } else { Val::N(v) }))).collect() }

/// The factor `table`/`kind` names, over the nodes in `env`: one node, or (table `partials`) a pair whose entries may be None.
#[pyfunction]
pub fn rule_factor<'py>(py: Python<'py>, table: &str, kind: &str, env: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
    let t = rules().get(table).and_then(|t| t.get(kind)).ok_or_else(|| he(format!("rules: no {table} template for {kind}")))?;
    let mut e = bind(env)?;
    let out = |v: Vec<Bound<'py, PyAny>>| -> PyResult<Bound<'py, PyAny>> { Ok(PyTuple::new(py, v)?.into_any()) };
    match table {
        "partials" | "bin_jvp" => out(t.arr().iter().map(|x| Ok(nil_or(py, ev(py, x, &mut e)?))).collect::<PyResult<Vec<_>>>()?),
        "bin_vjp" => out(t.arr().iter().map(|p| Ok(PyTuple::new(py, [nd(py, &p.arr()[0], &mut e)?, nd(py, &p.arr()[1], &mut e)?])?.into_any())).collect::<PyResult<Vec<_>>>()?),
        _ => nd(py, t, &mut e),
    }
}
/// A rule program: the operands of `n` bound to the names the row lists, its `lets` run in order, then the `vjp` pairs (arg = the cotangent) or
/// the `jvp` expression (arg = the tangent function).
#[pyfunction]
pub fn rule_run<'py>(py: Python<'py>, table: &str, kind: &str, mode: &str, n: &Bound<'py, PyAny>, arg: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let t = rules().get(table).and_then(|t| t.get(kind)).ok_or_else(|| he(format!("rules: no {table} program for {kind}")))?;
    let mut e: Env<'py> = vec![("n".into(), Val::N(n.clone())), (if mode == "vjp" { "g" } else { "t" }.into(), Val::N(arg.clone()))];
    let ops: Vec<Bound<'py, PyAny>> = n.getattr("operands")?.try_iter()?.collect::<PyResult<_>>()?;
    e.push(("ops".into(), Val::L(ops.iter().map(|o| Val::N(o.clone())).collect())));
    for (name, o) in t.get("operands").map(|x| x.strs()).unwrap_or_default().into_iter().zip(ops) { e.push((name, Val::N(o))) }
    for p in t.get("lets").map(|x| x.arr()).unwrap_or(&[]) { let v = ev(py, &p.arr()[1], &mut e)?; e.push((p.arr()[0].str().unwrap_or("").to_string(), v)) }
    for p in t.get(&format!("{mode}_lets")).map(|x| x.arr()).unwrap_or(&[]) { let v = ev(py, &p.arr()[1], &mut e)?; e.push((p.arr()[0].str().unwrap_or("").to_string(), v)) }
    if mode == "vjp" && t.get("vjp_expr").is_some() {
        let Val::L(l) = ev(py, t.get("vjp_expr").unwrap(), &mut e)? else { return Err(he("rules: vjp_expr is a list of pairs".into())) };
        let v = l.into_iter().map(|p| { let Val::L(p) = p else { return Err(he("rules: a pair".into())) }; Ok(PyTuple::new(py, p.into_iter().map(node).collect::<PyResult<Vec<_>>>()?)?.into_any()) }).collect::<PyResult<Vec<_>>>()?;
        return Ok(PyTuple::new(py, v)?.into_any())
    }
    if mode == "vjp" {
        let v = t.get("vjp").map(|x| x.arr()).unwrap_or(&[]).iter().map(|p| Ok(PyTuple::new(py, [nd(py, &p.arr()[0], &mut e)?, nd(py, &p.arr()[1], &mut e)?])?.into_any())).collect::<PyResult<Vec<_>>>()?;
        return Ok(PyTuple::new(py, v)?.into_any())
    }
    Ok(nil_or(py, ev(py, t.get("jvp").unwrap_or(&J::Null), &mut e)?))
}
/// The kinds a rule table has templates for, in manifest order.
#[pyfunction]
pub fn rule_kinds(table: &str) -> Vec<String> { rules().get(table).map(|t| t.obj().iter().map(|x| x.0.clone()).collect()).unwrap_or_default() }
/// The kinds that carry no derivative (`zero`) or are left to the transform (`walker`), in table order; `@NAME` names a manifest constant.
#[pyfunction]
pub fn rule_modes() -> Vec<(String, String)> {
    let mut out = vec![];
    for (mode, list) in rules().get("modes").map(|t| t.obj().iter().map(|x| (x.0.clone(), x.1.clone())).collect::<Vec<_>>()).unwrap_or_default() {
        for k in list.arr() {
            let k = k.str().unwrap_or("");
            match k.strip_prefix('@') {
                Some(c) => out.extend(const_of(c).unwrap_or(&[]).iter().map(|x| (x.to_string(), mode.clone()))),
                None => out.push((k.to_string(), mode.clone())),
            }
        }
    }
    out
}
/// The derivative of a sum over samples read in the body (`reduce`): the same sum of the adjoint (`vjp`, `x` = the adjoint) or of the
/// operand's tangent (`jvp`, `x` = the tangent function). Any other reduction has no derivative.
#[pyfunction]
pub fn rule_reduce<'py>(py: Python<'py>, direction: &str, n: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let op: String = n.getattr("op")?.extract()?;
    if op != "sum" {
        return Err(he(format!("no derivative for the {op} over samples read in the body: its subgradient picks one sample, so it is not in this release; reduce with a sum")))
    }
    let child = n.getattr("operands")?.get_item(0)?;
    if direction == "vjp" { return Ok(PyTuple::new(py, [PyTuple::new(py, [child, n.call_method1("rebuilt", (PyTuple::new(py, [x])?,))?])?])?.into_any()) }
    let d = x.call1((child,))?;
    if d.is_none() { Ok(d) } else { n.call_method1("rebuilt", (PyTuple::new(py, [d])?,)) }
}
/// `kind` over `nodes` entry by entry when any is not rank-0, re-assembled with `vec`.
#[pyfunction]
pub fn entrywise_op<'py>(py: Python<'py>, kind: &str, nodes: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> { entrywise(py, kind, &nodes) }

/// Whether `policy` is `"ordered"` and `payload` is a floating scalar or vector within the team size: a block kept whole, rows added in lane order.
#[pyfunction]
pub fn ordered_case(payload: &Bound<'_, PyAny>, policy: &str) -> PyResult<bool> {
    let shape = shape_of(payload)?;
    Ok(policy == "ordered" && matches!(dtype_of(payload)?.as_str(), "f64" | "f32") && shape.len() <= 1 && shape.iter().all(|e| *e <= typing::team_max_dim()))
}
/// Whether the keyed sum of `payload` CAN be one `BlockSum`, by structure alone: a product of per-row rank-1 values of a float type within
/// the team size; under `policy == "ordered"` any floating scalar or vector payload is.
#[pyfunction]
#[pyo3(signature = (payload, policy="auto"))]
pub fn block_shaped<'py>(py: Python<'py>, payload: &Bound<'py, PyAny>, policy: &str) -> PyResult<bool> {
    if ordered_case(payload, policy)? { return Ok(true) }
    let float = matches!(dtype_of(payload)?.as_str(), "f64" | "f32");
    Ok(float && shape_of(payload)?.iter().all(|e| *e <= typing::team_max_dim()) && !block_factors(py, payload)?.is_none())
}
/// `sinks` with every `BlockSumWrite` replaced by its per-entry accumulates.
#[pyfunction]
pub fn expand_stores<'py>(py: Python<'py>, sinks: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let mut out: Vec<Bound<'py, PyAny>> = vec![];
    for sink in sinks.try_iter()? {
        let sink = sink?;
        if canon::is_a(&sink, "BlockSumWrite")? { for x in sink.call_method0("expanded")?.try_iter()? { out.push(x?) } } else { out.push(sink) }
    }
    PyTuple::new(py, out)
}
/// ``(u, v, rows)`` when ``payload`` is a sum over per-lane rows of rank-1 products (``outer(u, v)``; ``u * v`` of a rank-1 and a rank-0;
/// ``transpose(M) @ M``; ``transpose(M) @ r``), else None.
#[pyfunction]
pub fn block_factors<'py>(py: Python<'py>, payload: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if !canon::is_a(payload, "Op")? { return Ok(None) }
    let kind: String = payload.getattr("kind")?.extract()?;
    let ops: Vec<Bound<'py, PyAny>> = payload.getattr("operands")?.try_iter()?.collect::<PyResult<_>>()?;
    let rank = |x: &Bound<'py, PyAny>| shape_of(x).map(|s| s.len());
    let one = 1i64.into_pyobject(py)?.into_any();
    let tup = |v: Vec<Bound<'py, PyAny>>| -> PyResult<Option<Bound<'py, PyAny>>> { Ok(Some(PyTuple::new(py, v)?.into_any())) };
    match kind.as_str() {
        "outer" => return tup(vec![ops[0].clone(), ops[1].clone(), one]),
        "mul" => {
            let (a, b) = (&ops[0], &ops[1]);
            if rank(a)? == 1 && rank(b)? == 0 { return tup(vec![a.clone(), b.clone(), one]) }
            if rank(b)? == 1 && rank(a)? == 0 { return tup(vec![b.clone(), a.clone(), one]) }
        }
        "mm" | "mv" => {
            let (t, other) = (&ops[0], &ops[1]);
            if canon::is_a(t, "Op")? && t.getattr("kind")?.extract::<String>()? == "transpose" {
                let m = t.getattr("operands")?.get_item(0)?;
                if rank(&m)? == 2 {
                    let rows = shape_of(&m)?[0];
                    if kind == "mm" && other.is(&m) { return tup(vec![m.clone(), m, rows.into_pyobject(py)?.into_any()]) }
                    if kind == "mv" && shape_of(other)? == vec![rows] { return tup(vec![m, other.clone(), rows.into_pyobject(py)?.into_any()]) }
                }
            }
        }
        _ => {}
    }
    Ok(None)
}

/// The factors of a BlockSum's payload; an ordered sum of a plain scalar or vector is `(payload, None, 1)`.
#[pyfunction]
pub fn block_factors_of<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let payload = node.getattr("operands")?.get_item(0)?;
    match block_factors(py, &payload)? { Some(t) => Ok(t), None => Ok(PyTuple::new(py, [payload, py.None().into_bound(py), 1i64.into_pyobject(py)?.into_any()])?.into_any()) }
}
