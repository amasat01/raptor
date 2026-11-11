//! The op vocabulary's result-type rules and `make` (raptor/ir/ops.py, linalg.result_type and rays.result_type, ported). The vocabularies are
//! the manifest's consts (core.json); every refusal text is the one the Python rules gave.
use super::*;
use rcore::keytext::{self, V};

#[derive(Clone, PartialEq)]
pub struct TT { shape: Vec<i64>, dtype: String, tag: Option<String> }
fn tt(shape: &[i64], dtype: &str, tag: Option<&str>) -> TT { TT { shape: shape.to_vec(), dtype: dtype.to_string(), tag: tag.map(|s| s.to_string()) } }
fn c(name: &str) -> &'static [&'static str] { const_of(name).unwrap_or(&[]) }
fn inn(name: &str, k: &str) -> bool { c(name).contains(&k) }
fn pairs(name: &str, k: &str) -> Option<&'static str> { c(name).chunks(2).find(|p| p[0] == k).map(|p| p[1]) }
fn q(s: &str) -> String { keytext::py_str(s) }
fn shp(s: &[i64]) -> String { keytext::text(&V::Tuple(s.iter().map(|x| V::Int(*x as i128)).collect())) }
fn r(t: &TT) -> String { format!("TensorType(shape={}, dtype={}, tag={})", shp(&t.shape), q(&t.dtype), t.tag.as_deref().map(q).unwrap_or("None".into())) }
fn rl(ts: &[TT]) -> String { format!("[{}]", ts.iter().map(r).collect::<Vec<_>>().join(", ")) }
fn err<T>(m: String) -> PyResult<T> { Err(he(m)) }
fn tt_of(o: &Bound<'_, PyAny>) -> PyResult<TT> {
    let t = o.getattr("ttype")?;
    let tag = t.getattr("tag")?;
    Ok(TT { shape: t.getattr("shape")?.try_iter()?.map(|x| x?.extract::<i64>()).collect::<PyResult<_>>()?, dtype: t.getattr("dtype")?.extract()?,
            tag: if tag.is_none() { None } else { Some(tag.extract()?) } })
}
fn tt_py<'py>(py: Python<'py>, t: &TT) -> PyResult<Bound<'py, PyAny>> {
    let shape = PyTuple::new(py, t.shape.iter().copied())?;
    reg().ttype.bind(py).call1((shape, &t.dtype, t.tag.as_deref()))
}
fn all_kinds() -> Vec<&'static str> {
    let mut v = vec![];
    for n in ["ELEMENTWISE_BINARY", "UNARY_MATH", "COMPARE", "LOGICAL", "VECTOR_OPS", "MATRIX_OPS", "LINALG_FACTORISATIONS", "LINALG_CONSUMERS", "LINALG_PROJECTIONS", "LSTSQ_OPS",
              "RAY_OPS", "QUAT_OPS", "RANDOM_OPS"] { v.extend_from_slice(c(n)) }
    v.extend(["select", "dispatch", "at"]);
    for n in ["ELEMENTWISE_TERNARY", "PREDICATE", "INT_BINARY", "INT_UNARY"] { v.extend_from_slice(c(n)) }
    v.push("cast");
    v
}
const RANKS: [&str; 5] = ["bool", "i32", "i64", "f32", "f64"];
fn mix_message(a: &str, b: &str) -> String {
    let (plane, other) = if inn("INT_PLANE_DTYPES", a) { (a, b) } else { (b, a) };
    let fix = if inn("INT_PLANE_DTYPES", other) { format!("convert one side with hawk.math.{plane}(...) or hawk.math.{other}(...)") }
        else if other == "f64" || other == "f32" { format!("convert explicitly: hawk.math.real(n) for the float result, or hawk.math.{plane}(x) for the integer one") }
        else { format!("convert explicitly with hawk.math.{plane}(...)") };
    format!("an {plane} value does not mix with {other} without a cast (no silent promotion): {fix}")
}
fn promote(a: &str, b: &str) -> PyResult<String> {
    if inn("INT_PLANE_DTYPES", a) || inn("INT_PLANE_DTYPES", b) {
        if a != b { return err(mix_message(a, b)) }
        return Ok(a.to_string())
    }
    for d in [a, b] { if !RANKS.contains(&d) { return err(format!("unknown dtype {}; declared are ('bool', 'i32', 'i64', 'f32', 'f64')", q(d))) } }
    let rk = |d: &str| RANKS.iter().position(|x| *x == d).unwrap();
    Ok(if rk(a) >= rk(b) { a } else { b }.to_string())
}
fn rank(kind: &str, t: &TT, rk: usize, extent: Option<i64>) -> PyResult<()> {
    if t.shape.len() != rk || extent.map_or(false, |e| t.shape[0] != e) {
        return err(format!("op {}: operand typed {} is not the expected rank-{rk}{} value", q(kind), r(t), extent.map(|e| format!(" extent-{e}")).unwrap_or_default()))
    }
    Ok(())
}
fn same(kind: &str, a: &TT, b: &TT, rk: usize, extent: Option<i64>) -> PyResult<()> {
    rank(kind, a, rk, extent)?; rank(kind, b, rk, extent)?;
    if a.shape != b.shape { return err(format!("op {}: operand shapes {} and {} disagree", q(kind), shp(&a.shape), shp(&b.shape))) }
    Ok(())
}
fn broadcast(kind: &str, a: &TT, b: &TT) -> PyResult<Vec<i64>> {
    if a.shape == b.shape || b.shape.is_empty() { return Ok(a.shape.clone()) }
    if a.shape.is_empty() { return Ok(b.shape.clone()) }
    err(format!("op {}: shapes {} and {} do not broadcast — an elementwise op takes equal shapes or one rank-0 operand", q(kind), shp(&a.shape), shp(&b.shape)))
}
fn tag_of(kind: &str, ts: &[TT]) -> Option<String> {
    let mut tags: Vec<&Option<String>> = vec![];
    for t in ts { if !t.shape.is_empty() && !tags.contains(&&t.tag) { tags.push(&t.tag) } }
    // a tag survives only where a tagged value stays that thing
    if !["add", "sub", "neg", "mul", "div", "select", "dispatch"].contains(&kind) || tags.len() != 1 { return None }
    if (kind == "mul" || kind == "div") && ts.iter().all(|t| !t.shape.is_empty()) { return None }
    tags[0].clone()
}
fn arity(kind: &str, n: usize) -> PyResult<()> {
    if kind == "vec" { if n == 0 { return err("vec: at least one component is required".into()) } return Ok(()) }
    if kind == "dispatch" { if n < 3 { return err(format!("dispatch: a kind plus at least 2 branches is required, got {n} operand(s)")) } return Ok(()) }
    let want: usize = match kind {
        "lnot" | "cast" | "invert" | "isnan" | "isinf" | "isfinite" | "component" | "transpose" | "norm" | "sum" | "splat" | "quat_conj" | "quat_recip" | "as_pure" | "as_vec3" => 1,
        "select" | "fma" | "clip" | "set_component_at" | "add_at" => 3,
        "at" | "component_at" => 2,
        k if inn("LINALG_FACTORISATIONS", k) || inn("LINALG_PROJECTIONS", k) || k == "lstsq_info" => 1,
        k if inn("LINALG_CONSUMERS", k) || k == "lstsq" || k == "lstsq_gram_solve" => 2,
        k if inn("RAY_OPS", k) => pairs("RAY_ARITY_PAIRS", k).unwrap().parse().unwrap(),
        k => if inn("UNARY_MATH", k) { 1 } else { 2 },
    };
    if n != want { return err(format!("op {}: takes {want} operand(s), got {n}", q(kind)).replacen("op ", "op ", 1).replace(&format!(": takes {want}"), &format!(" takes {want}"))) }
    Ok(())
}
fn is_int(o: &Bound<'_, PyAny>) -> bool { use pyo3::types::{PyBool, PyInt}; o.is_instance_of::<PyInt>() && !o.is_instance_of::<PyBool>() }

fn quat(kind: &str, ts: &[TT]) -> PyResult<TT> {
    let qt = tt(&[4], &ts[0].dtype, Some("quaternion"));
    if ["quat_mul", "quat_conj", "quat_recip", "as_pure"].contains(&kind) {
        let want = if kind == "as_pure" { 3 } else { 4 };
        for t in ts { rank(kind, t, 1, Some(want))? }
        return Ok(qt)
    }
    if kind == "quat_rotate" {
        rank(kind, &ts[0], 1, Some(4))?; rank(kind, &ts[1], 1, Some(3))?;
        return Ok(tt(&[3], &promote(&ts[0].dtype, &ts[1].dtype)?, None))
    }
    rank(kind, &ts[0], 1, Some(4))?;
    Ok(tt(&[3], &ts[0].dtype, None))
}
fn int_index(t: &TT) -> bool { t.shape.is_empty() && (t.dtype == "i32" || t.dtype == "i64") }
fn shaped(kind: &str, ts: &[TT], lit: &Bound<'_, PyAny>) -> PyResult<TT> {
    use pyo3::types::PyTuple as T;
    let lit_t = lit.cast::<T>().ok();
    match kind {
        "dot" => { same(kind, &ts[0], &ts[1], 1, None)?; Ok(tt(&[], &promote(&ts[0].dtype, &ts[1].dtype)?, None)) }
        "cross" => { same(kind, &ts[0], &ts[1], 1, Some(3))?; Ok(tt(&[3], &promote(&ts[0].dtype, &ts[1].dtype)?, None)) }
        "norm" => { rank(kind, &ts[0], 1, None)?; Ok(tt(&[], &ts[0].dtype, None)) }
        "sum" => { if ![1, 2].contains(&ts[0].shape.len()) { return err(format!("sum: expected a rank-1 or rank-2 operand, got {}", r(&ts[0]))) } Ok(tt(&[], &ts[0].dtype, None)) }
        "splat" => {
            rank(kind, &ts[0], 0, None)?;
            match lit_t { Some(t) if t.len() > 0 => Ok(TT { shape: t.iter().map(|x| x.extract::<i64>()).collect::<PyResult<_>>()?, dtype: ts[0].dtype.clone(), tag: None }),
                _ => err(format!("splat: the literal must be a target shape, got {}", rep(lit))) }
        }
        "component" => {
            let (index, extent): (Vec<Bound<'_, PyAny>>, Vec<i64>) = if let Some(t) = lit_t { rank(kind, &ts[0], 2, None)?; (t.iter().collect(), ts[0].shape.clone()) }
                else { rank(kind, &ts[0], 1, None)?; (vec![lit.clone()], ts[0].shape.clone()) };
            let ok = index.len() == extent.len() && index.iter().zip(&extent).all(|(k, e)| is_int(k) && k.extract::<i64>().map_or(false, |k| 0 <= k && k < *e));
            if !ok { return err(format!("component: index {} is out of the static extent {} (an extent is a type, not a runtime value)", rep(lit),
                if extent.len() > 1 { shp(&extent) } else { extent[0].to_string() })) }
            Ok(tt(&[], &ts[0].dtype, None))
        }
        "component_at" => {
            rank(kind, &ts[0], 1, None)?;
            if !int_index(&ts[1]) { return err(format!("component_at: the index must be a rank-0 integer value, got {}. A runtime component index is an ADDRESS — typed f64 it would reach the emitter as a Real cast into a std::size_t (the same rule hawk.trace.value.index_node applies to a table subscript)", r(&ts[1]))) }
            Ok(tt(&[], &ts[0].dtype, None))
        }
        "set_component_at" | "add_at" => {
            rank(kind, &ts[0], 1, None)?;
            if !int_index(&ts[1]) { return err(format!("{kind}: the index must be a rank-0 integer value, got {} (see component_at)", r(&ts[1]))) }
            rank(kind, &ts[2], 0, None)?;
            Ok(TT { shape: ts[0].shape.clone(), dtype: promote(&ts[0].dtype, &ts[2].dtype)?, tag: ts[0].tag.clone() })
        }
        "vec" => {
            for t in ts { rank(kind, t, 0, None)? }
            let mut dtype = ts[0].dtype.clone();
            for t in &ts[1..] { dtype = promote(&dtype, &t.dtype)? }
            if lit.is_none() { return Ok(tt(&[ts.len() as i64], &dtype, None)) }
            let shape: Option<Vec<i64>> = lit_t.as_ref().filter(|t| t.len() == 2).and_then(|t| t.iter().map(|x| x.extract::<i64>().ok()).collect());
            match shape { Some(s) if s[0] * s[1] == ts.len() as i64 => Ok(TT { shape: s, dtype, tag: None }),
                _ => err(format!("vec: a matrix shape literal must be (rows, cols) with rows * cols == {} entries, got {}", ts.len(), rep(lit))) }
        }
        "mv" => {
            rank(kind, &ts[0], 2, None)?; rank(kind, &ts[1], 1, None)?;
            if ts[0].shape[1] != ts[1].shape[0] { return err(format!("mv: {} does not contract with {}", shp(&ts[0].shape), shp(&ts[1].shape))) }
            Ok(tt(&[ts[0].shape[0]], &promote(&ts[0].dtype, &ts[1].dtype)?, None))
        }
        "mm" => {
            rank(kind, &ts[0], 2, None)?; rank(kind, &ts[1], 2, None)?;
            if ts[0].shape[1] != ts[1].shape[0] { return err(format!("mm: {} does not contract with {}", shp(&ts[0].shape), shp(&ts[1].shape))) }
            Ok(tt(&[ts[0].shape[0], ts[1].shape[1]], &promote(&ts[0].dtype, &ts[1].dtype)?, None))
        }
        "outer" => { rank(kind, &ts[0], 1, None)?; rank(kind, &ts[1], 1, None)?; Ok(tt(&[ts[0].shape[0], ts[1].shape[0]], &promote(&ts[0].dtype, &ts[1].dtype)?, None)) }
        "transpose" => { rank(kind, &ts[0], 2, None)?; Ok(tt(&[ts[0].shape[1], ts[0].shape[0]], &ts[0].dtype, None)) }
        _ => quat(kind, ts),
    }
}

pub(crate) fn team_max_dim() -> i64 { str_const("TEAM_MAX_DIM").parse().expect("TEAM_MAX_DIM is an integer") }
fn floats(d: &str) -> bool { d == "f64" || d == "f32" }
fn linalg_type(kind: &str, ts: &[TT]) -> PyResult<TT> {
    let tag_of_kind = |k: &str| pairs("FACTOR_TAG_PAIRS", k).unwrap_or("");
    if kind == "qr" || inn("LSTSQ_OPS", kind) {
        let t = &ts[0];
        if t.shape.len() != 2 || t.shape[0] < t.shape[1] { return err(format!("op {}: a tall matrix (rows >= columns) is factored, got {}", q(kind), r(t))) }
        if !floats(&t.dtype) { return err(format!("op {}: an f64/f32 matrix is factored, got {}", q(kind), r(t))) }
        let (m, n) = (t.shape[0], t.shape[1]);
        if !(1 <= n && n <= team_max_dim()) { return err(format!("op {}: the columns are static and 1 <= n <= {}, got n={n}", q(kind), team_max_dim())) }
        if kind == "qr" { return Ok(TT { shape: t.shape.clone(), dtype: t.dtype.clone(), tag: Some(tag_of_kind(kind).into()) }) }
        if kind == "lstsq_info" { return Ok(tt(&[], "int32", None)) }
        let (b, rows) = (&ts[1], if kind == "lstsq" { m } else { n });
        if b.dtype != t.dtype || ![1, 2].contains(&b.shape.len()) || b.shape[0] != rows {
            return err(format!("op {}: the right-hand side is a {} vector or matrix of {rows} rows against the {m}x{n} matrix, got {}", q(kind), t.dtype, r(b)))
        }
        let mut s = vec![n]; s.extend_from_slice(&b.shape[1..]);
        return Ok(TT { shape: s, dtype: t.dtype.clone(), tag: None })
    }
    if inn("LINALG_FACTORISATIONS", kind) {
        let t = &ts[0];
        if t.shape.len() != 2 || t.shape[0] != t.shape[1] { return err(format!("op {}: a square matrix is factored, got {}", q(kind), r(t))) }
        let n = t.shape[0];
        if !floats(&t.dtype) { return err(format!("op {}: an f64/f32 matrix is factored, got {}", q(kind), r(t))) }
        if !(1 <= n && n <= team_max_dim()) { return err(format!("op {}: the dimension is static and 1 <= n <= {}, got n={n}", q(kind), team_max_dim())) }
        return Ok(TT { shape: vec![n, n], dtype: t.dtype.clone(), tag: Some(tag_of_kind(kind).into()) })
    }
    let want = pairs("FACTOR_NEEDS_PAIRS", kind).unwrap_or("");
    let t0 = &ts[0];
    if t0.tag.as_deref() != Some(want) {
        let producer = c("FACTOR_TAG_PAIRS").chunks(2).find(|p| p[1] == want).map(|p| p[0]).unwrap_or("");
        return err(format!("op {} reads the result of {} (a {want}), got {}", q(kind), q(producer), r(t0)))
    }
    let n = *t0.shape.last().unwrap();
    let dtype = t0.dtype.clone();
    if kind == "qr_solve" {
        let (m, b) = (t0.shape[0], &ts[1]);
        if b.dtype != dtype || ![1, 2].contains(&b.shape.len()) || b.shape[0] != m {
            return err(format!("op {}: the right-hand side is a {dtype} vector ({m},) or matrix ({m}, k) against the {m}x{n} factor, got {}", q(kind), r(b)))
        }
        let mut s = vec![n]; s.extend_from_slice(&b.shape[1..]);
        return Ok(TT { shape: s, dtype, tag: None })
    }
    if inn("LINALG_CONSUMERS", kind) {
        let b = &ts[1];
        if b.dtype != dtype || ![1, 2].contains(&b.shape.len()) || b.shape[0] != n {
            return err(format!("op {}: the right-hand side is a {dtype} vector ({n},) or matrix ({n}, k) against the {n}x{n} factor, got {}", q(kind), r(b)))
        }
        return Ok(TT { shape: b.shape.clone(), dtype, tag: None })
    }
    if kind.ends_with("_info") { return Ok(tt(&[], "int32", None)) }
    if kind == "lu_det" { return Ok(TT { shape: vec![], dtype, tag: None }) }
    if kind == "eigh_w" { return Ok(TT { shape: vec![n], dtype, tag: None }) }
    Ok(TT { shape: vec![n, n], dtype, tag: None })
}
fn rays_type(kind: &str, ts: &[TT]) -> PyResult<TT> {
    let scalars: Vec<usize> = pairs("RAY_SCALAR_PAIRS", kind).unwrap().split(',').map(|x| x.parse().unwrap()).collect();
    for (k, t) in ts.iter().enumerate() {
        let want: Vec<i64> = if scalars.contains(&k) { vec![] } else { vec![3] };
        if t.shape != want || !floats(&t.dtype) {
            return err(format!("op {}: operand {k} is {} f64/f32 value, got {}", q(kind), if want.is_empty() { "a rank-0" } else { "a length-3" }, r(t)))
        }
    }
    Ok(tt(&[4], if ts.iter().any(|t| t.dtype == "f64") { "f64" } else { "f32" }, None))
}

fn compute(kind: &str, ts: &[TT], lit: &Bound<'_, PyAny>) -> PyResult<TT> {
    let kinds = all_kinds();
    if !kinds.contains(&kind) {
        let v = V::Tuple(kinds.iter().map(|k| V::Str(k.to_string())).collect());
        return err(format!("unknown op kind {}; the vocabulary is {}", q(kind), keytext::text(&v)))
    }
    arity(kind, ts.len())?;
    if kind == "cast" {
        let ok = lit.extract::<&str>().map_or(false, |s| inn("CAST_TARGETS", s));
        if !ok { return err(format!("cast: the target dtype must be one of {}, got {}", keytext::text(&V::Tuple(c("CAST_TARGETS").iter().map(|s| V::Str(s.to_string())).collect())), rep(lit))) }
        if !ts[0].shape.is_empty() || ts[0].dtype == "bool" { return err(format!("cast: converts a rank-0 number, got {} — a bool picks a value with hawk.math.where(cond, a, b)", r(&ts[0]))) }
        return Ok(tt(&[], lit.extract::<&str>()?, None))
    }
    if inn("LINALG_FACTORISATIONS", kind) || inn("LINALG_CONSUMERS", kind) || inn("LINALG_PROJECTIONS", kind) || inn("LSTSQ_OPS", kind) { return linalg_type(kind, ts) }
    if inn("RAY_OPS", kind) { return rays_type(kind, ts) }
    let int_op = inn("INT_BINARY", kind) || inn("INT_UNARY", kind);
    if int_op {
        for t in ts {
            if !t.shape.is_empty() || !(inn("INT_PLANE_DTYPES", &t.dtype) || t.dtype == "i32" || t.dtype == "i64") {
                return err(format!("op {} is an integer operator and takes rank-0 integer operands, got {}; for floats use hawk.math.floor(a / b) or hawk.math.remainder(a, b), or convert with hawk.math.int32(...)", q(kind), r(t)))
            }
        }
    } else if ts.iter().any(|t| inn("INT_PLANE_DTYPES", &t.dtype)) {
        let plane = ts.iter().find(|t| inn("INT_PLANE_DTYPES", &t.dtype)).unwrap().dtype.clone();
        let allowed = ["add", "sub", "mul", "min", "max", "neg", "abs", "select"].contains(&kind) || inn("COMPARE", kind) || inn("INT_BINARY", kind) || inn("INT_UNARY", kind);
        if !allowed {
            let hint = if kind == "div" { " — integer division is `//`" } else { "" };
            return err(format!("op {} is not defined on an {plane} value{hint}; convert it first with hawk.math.real(...) to compute in floating point", q(kind)))
        }
        if ts.iter().any(|t| !t.shape.is_empty()) { return err(format!("op {}: an {plane} value is rank-0 (a per-sample integer plane), got operands {}", q(kind), rl(ts))) }
        if kind != "select" { for t in ts { if t.dtype != plane { return err(mix_message(&plane, &t.dtype)) } } }
    }
    let ebin = inn("ELEMENTWISE_BINARY", kind); let ter = inn("ELEMENTWISE_TERNARY", kind); let um = inn("UNARY_MATH", kind);
    if ts.iter().any(|t| t.dtype == "bool") && (ebin || um || ter || int_op) {
        return err(format!("op {} takes numbers, not a bool condition: use & | ^ ~ (or land, lor, lnot) for logic, or m.select(cond, x, 0.0) for a number", q(kind)))
    }
    let binary_logical = inn("LOGICAL", kind) && kind != "lnot";
    if ebin || inn("COMPARE", kind) || binary_logical {
        let shape = broadcast(kind, &ts[0], &ts[1])?;
        let boolean = inn("COMPARE", kind) || inn("LOGICAL", kind);
        let dtype = if boolean { "bool".to_string() } else { promote(&ts[0].dtype, &ts[1].dtype)? };
        return Ok(TT { shape, dtype, tag: tag_of(kind, ts) })
    }
    if um || kind == "lnot" || inn("PREDICATE", kind) {
        let dtype = if kind == "lnot" || inn("PREDICATE", kind) { "bool".to_string() } else { ts[0].dtype.clone() };
        return Ok(TT { shape: ts[0].shape.clone(), dtype, tag: tag_of(kind, ts) })
    }
    if ter {
        let b01 = TT { shape: broadcast(kind, &ts[0], &ts[1])?, dtype: "f64".into(), tag: None };
        let shape = broadcast(kind, &b01, &ts[2])?;
        let dtype = promote(&promote(&ts[0].dtype, &ts[1].dtype)?, &ts[2].dtype)?;
        return Ok(TT { shape, dtype, tag: None })
    }
    match kind {
        "select" => {
            if ts[0].dtype != "bool" { return err(format!("select: the condition must be bool-typed, got {}", r(&ts[0]))) }
            if ts[1] != ts[2] { return err(format!("select: the branches disagree — {} vs {}", r(&ts[1]), r(&ts[2]))) }
            return Ok(ts[1].clone())
        }
        _ if inn("INT_BINARY", kind) => return Ok(tt(&[], &promote(&ts[0].dtype, &ts[1].dtype)?, None)),
        _ if inn("INT_UNARY", kind) => return Ok(ts[0].clone()),
        "dispatch" => {
            if !ts[0].shape.is_empty() || ts[0].dtype != "i32" { return err(format!("dispatch: kind must be a rank-0 i32 value, got {}", r(&ts[0]))) }
            for t in &ts[2..] { if *t != ts[1] { return err(format!("dispatch: the branches disagree — {} vs {}", r(&ts[1]), r(t))) } }
            return Ok(ts[1].clone())
        }
        "at" => return Ok(ts[0].clone()),
        _ => {}
    }
    if inn("RANDOM_OPS", kind) {
        for (pname, t) in ["seed", "counter"].iter().zip(ts) {
            if !t.shape.is_empty() || !(t.dtype == "i32" || t.dtype == "i64") {
                return err(format!("{kind}: {pname} must be a rank-0 i32/i64 value (a literal, a by-value uniform wire, or a plane read), got {}", r(t)))
            }
        }
        return Ok(tt(&[], "f64", None))
    }
    shaped(kind, ts, lit)
}

/// The type `kind` returns over `operands` (a TensorType), with its refusal.
#[pyfunction]
#[pyo3(signature = (kind, operands, literal=None))]
pub fn result_type<'py>(py: Python<'py>, kind: &str, operands: &Bound<'py, PyAny>, literal: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let ts = operands.try_iter()?.map(|o| tt_of(&o?)).collect::<PyResult<Vec<_>>>()?;
    let lit = literal.unwrap_or_else(|| py.None().into_bound(py));
    tt_py(py, &compute(kind, &ts, &lit)?)
}
/// The dtype of an elementwise combination of two dtypes.
#[pyfunction]
pub fn promote_dtype(a: &str, b: &str) -> PyResult<String> { promote(a, b) }

/// `node` where a matrix is wanted: a Cholesky factor becomes its `chol_factor` projection, another factor is refused by name.
fn as_matrix<'py>(py: Python<'py>, node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let t = tt_of(node)?;
    match t.tag.as_deref() {
        Some("chol_factor") => {
            let op = reg().by_name.get("Op").unwrap().bind(py);
            op.call1(("chol_factor", PyTuple::new(py, [node.clone()])?, tt_py(py, &TT { shape: t.shape, dtype: t.dtype, tag: None })?))
        }
        Some(tag @ ("lu_factor" | "eigh_factor" | "qr_factor")) => {
            let (what, reads) = match tag { "lu_factor" => ("an LU", "lu_solve, lu_det or lu_info"), "eigh_factor" => ("an eigh", "eigh_w, eigh_v or eigh_info"), _ => ("a QR", "qr_solve or qr_info") };
            err(format!("{what} factor is not a matrix: it carries pivots and a status, so it is read with {reads}, never as an operand of arithmetic"))
        }
        _ => Ok(node.clone()),
    }
}
/// Mint an Op of `kind` with its inferred type (factorisations used as matrices are replaced by what a matrix use means).
#[pyfunction]
#[pyo3(signature = (kind, operands, literal=None))]
pub fn make<'py>(py: Python<'py>, kind: &str, operands: &Bound<'py, PyAny>, literal: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let mut ops: Vec<Bound<'py, PyAny>> = operands.try_iter()?.collect::<PyResult<_>>()?;
    if all_kinds().contains(&kind) {
        let keep = if inn("LINALG_CONSUMERS", kind) || inn("LINALG_PROJECTIONS", kind) { 1 } else { 0 };
        for o in ops.iter_mut().skip(keep) { *o = as_matrix(py, o)? }
    }
    let lit = literal.unwrap_or_else(|| py.None().into_bound(py));
    let ts = ops.iter().map(tt_of).collect::<PyResult<Vec<_>>>()?;
    let ty = tt_py(py, &compute(kind, &ts, &lit)?)?;
    reg().by_name.get("Op").unwrap().bind(py).call1((kind, PyTuple::new(py, ops)?, ty, lit))
}
