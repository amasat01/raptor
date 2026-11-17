//! The op vocabulary's result-type rules and the typed op constructor: the type of an op is inferred from its operands. The
//! vocabularies are the manifest's consts (core.json); every refusal text is the one the rules gave before.
use crate::record::{refuse, Ctx, Fault};
use crate::value::{text_val, Val};
use rcore::keytext;

/// A tensor type: extents, element type and tag.
#[derive(Clone, PartialEq)]
pub struct TT {
    /// The extents.
    pub shape: Vec<i64>,
    /// The element type name.
    pub dtype: String,
    /// The tag, if any.
    pub tag: Option<String>,
}
/// A result that refuses with a fault.
type Checked<T> = Result<T, Fault>;
/// A type from its parts.
fn tt(shape: &[i64], dtype: &str, tag: Option<&str>) -> TT {
    TT {
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
        tag: tag.map(|s| s.to_string()),
    }
}
/// The core list constant `name`.
fn c(name: &str) -> &'static [&'static str] {
    rcore::CONSTS
        .iter()
        .find(|x| x.0 == name)
        .map_or(&[][..], |x| x.1)
}
/// Whether `k` is in the core list `name`.
fn inn(name: &str, k: &str) -> bool {
    c(name).contains(&k)
}
/// The value paired with `k` in the core pair list `name`.
fn pairs(name: &str, k: &str) -> Option<&'static str> {
    c(name).chunks(2).find(|p| p[0] == k).map(|p| p[1])
}
/// An extent list as text.
fn shp(s: &[i64]) -> String {
    text_val(&Val::Tuple(
        s.iter().map(|x| Val::Int(*x as i128)).collect(),
    ))
}
/// A type as text.
fn r(t: &TT) -> String {
    format!(
        "TensorType(shape={}, dtype={}, tag={})",
        shp(&t.shape),
        keytext::py_str(&t.dtype),
        t.tag
            .as_deref()
            .map(keytext::py_str)
            .unwrap_or("None".into())
    )
}
/// A list of types as text.
fn rl(ts: &[TT]) -> String {
    format!("[{}]", ts.iter().map(r).collect::<Vec<_>>().join(", "))
}
/// A refusal with the message `m`.
fn err<T>(m: String) -> Checked<T> {
    refuse(m)
}
/// The integer a value holds, if it is one.
fn int_of(v: &Val) -> Option<i64> {
    if let Val::Int(i) = v {
        i64::try_from(*i).ok()
    } else {
        None
    }
}
/// Whether a value is an integer.
fn is_int(v: &Val) -> bool {
    matches!(v, Val::Int(_))
}
/// The names of every op kind.
pub fn all_kinds() -> Vec<&'static str> {
    let mut v = vec![];
    for n in [
        "ELEMENTWISE_BINARY",
        "UNARY_MATH",
        "COMPARE",
        "LOGICAL",
        "VECTOR_OPS",
        "MATRIX_OPS",
        "LINALG_FACTORISATIONS",
        "LINALG_CONSUMERS",
        "LINALG_PROJECTIONS",
        "LSTSQ_OPS",
        "RAY_OPS",
        "QUAT_OPS",
        "RANDOM_OPS",
    ] {
        v.extend_from_slice(c(n))
    }
    v.extend(["select", "dispatch", "at"]);
    for n in [
        "ELEMENTWISE_TERNARY",
        "PREDICATE",
        "INT_BINARY",
        "INT_UNARY",
    ] {
        v.extend_from_slice(c(n))
    }
    v.push("cast");
    v
}
/// The numeric dtypes from lowest to highest rank.
const RANKS: [&str; 5] = ["bool", "i32", "i64", "f32", "f64"];
/// The refusal text for mixing an integer plane value with another dtype.
fn mix_message(a: &str, b: &str) -> String {
    let (plane, other) = if inn("INT_PLANE_DTYPES", a) {
        (a, b)
    } else {
        (b, a)
    };
    let fix = if inn("INT_PLANE_DTYPES", other) {
        "convert one side to the other's type explicitly".to_string()
    } else if other == "f64" || other == "f32" {
        format!("convert explicitly: the float side to {plane} for the integer result, or the {plane} side to a float for the float one")
    } else {
        format!("convert the other side to {plane} explicitly")
    };
    format!(
        "[mixed scalar types] an {plane} value does not mix with {other} without a cast (no silent promotion): {fix}"
    )
}
/// The dtype two operands promote to; refuses a mix that needs a cast.
fn promote(a: &str, b: &str) -> Checked<String> {
    if inn("INT_PLANE_DTYPES", a) || inn("INT_PLANE_DTYPES", b) {
        if a != b {
            return err(mix_message(a, b));
        }
        return Ok(a.to_string());
    }
    for d in [a, b] {
        if !RANKS.contains(&d) {
            return err(format!(
                "unknown dtype {}; declared are ('bool', 'i32', 'i64', 'f32', 'f64')",
                keytext::py_str(d)
            ));
        }
    }
    let rk = |d: &str| RANKS.iter().position(|x| *x == d).unwrap();
    Ok(if rk(a) >= rk(b) { a } else { b }.to_string())
}
/// Checks that `t` has rank `rk` (and extent `extent` when given).
fn rank(kind: &str, t: &TT, rk: usize, extent: Option<i64>) -> Checked<()> {
    if t.shape.len() != rk || extent.is_some_and(|e| t.shape[0] != e) {
        return err(format!(
            "op {}: operand typed {} is not the expected rank-{rk}{} value",
            keytext::py_str(kind),
            r(t),
            extent.map(|e| format!(" extent-{e}")).unwrap_or_default()
        ));
    }
    Ok(())
}
/// Checks that `a` and `b` have rank `rk` and the same extents.
fn same(kind: &str, a: &TT, b: &TT, rk: usize, extent: Option<i64>) -> Checked<()> {
    rank(kind, a, rk, extent)?;
    rank(kind, b, rk, extent)?;
    if a.shape != b.shape {
        return err(format!(
            "op {}: operand shapes {} and {} disagree",
            keytext::py_str(kind),
            shp(&a.shape),
            shp(&b.shape)
        ));
    }
    Ok(())
}
/// The broadcast extents of two operands; refuses unequal non-scalar extents.
fn broadcast(kind: &str, a: &TT, b: &TT) -> Checked<Vec<i64>> {
    if a.shape == b.shape || b.shape.is_empty() {
        return Ok(a.shape.clone());
    }
    if a.shape.is_empty() {
        return Ok(b.shape.clone());
    }
    err(format!("op {}: shapes {} and {} do not broadcast — an elementwise op takes equal shapes or one rank-0 operand", keytext::py_str(kind), shp(&a.shape), shp(&b.shape)))
}
/// The tag the result of `kind` carries, from its operands.
fn tag_of(kind: &str, ts: &[TT]) -> Option<String> {
    let mut tags: Vec<&Option<String>> = vec![];
    for t in ts {
        if !t.shape.is_empty() && !tags.contains(&&t.tag) {
            tags.push(&t.tag)
        }
    }
    // a tag survives only where a tagged value stays that thing
    if !["add", "sub", "neg", "mul", "div", "select", "dispatch"].contains(&kind) || tags.len() != 1
    {
        return None;
    }
    if (kind == "mul" || kind == "div") && ts.iter().all(|t| !t.shape.is_empty()) {
        return None;
    }
    tags[0].clone()
}
/// Checks the operand count of `kind`.
fn arity(kind: &str, n: usize) -> Checked<()> {
    if kind == "vec" {
        if n == 0 {
            return err("vec: at least one component is required".into());
        }
        return Ok(());
    }
    if kind == "dispatch" {
        if n < 3 {
            return err(format!(
                "dispatch: a kind plus at least 2 branches is required, got {n} operand(s)"
            ));
        }
        return Ok(());
    }
    let want: usize = match kind {
        "lnot" | "cast" | "invert" | "isnan" | "isinf" | "isfinite" | "component" | "transpose"
        | "norm" | "sum" | "splat" | "quat_conj" | "quat_recip" | "as_pure" | "as_vec3" => 1,
        "select" | "fma" | "clip" | "set_component_at" | "add_at" => 3,
        "at" | "component_at" => 2,
        k if inn("LINALG_FACTORISATIONS", k)
            || inn("LINALG_PROJECTIONS", k)
            || k == "lstsq_info" =>
        {
            1
        }
        k if inn("LINALG_CONSUMERS", k) || k == "lstsq" || k == "lstsq_gram_solve" => 2,
        k if inn("RAY_OPS", k) => pairs("RAY_ARITY_PAIRS", k).unwrap().parse().unwrap(),
        k => {
            if inn("UNARY_MATH", k) {
                1
            } else {
                2
            }
        }
    };
    if n != want {
        return err(format!(
            "op {}: takes {want} operand(s), got {n}",
            keytext::py_str(kind)
        )
        .replacen("op ", "op ", 1)
        .replace(&format!(": takes {want}"), &format!(" takes {want}")));
    }
    Ok(())
}

/// The type of a quaternion op.
fn quat(kind: &str, ts: &[TT]) -> Checked<TT> {
    let qt = tt(&[4], &ts[0].dtype, Some("quaternion"));
    if ["quat_mul", "quat_conj", "quat_recip", "as_pure"].contains(&kind) {
        let want = if kind == "as_pure" { 3 } else { 4 };
        for t in ts {
            rank(kind, t, 1, Some(want))?
        }
        return Ok(qt);
    }
    if kind == "quat_rotate" {
        rank(kind, &ts[0], 1, Some(4))?;
        rank(kind, &ts[1], 1, Some(3))?;
        return Ok(tt(&[3], &promote(&ts[0].dtype, &ts[1].dtype)?, None));
    }
    rank(kind, &ts[0], 1, Some(4))?;
    Ok(tt(&[3], &ts[0].dtype, None))
}
/// Whether `t` is a rank-0 integer index.
fn int_index(t: &TT) -> bool {
    t.shape.is_empty() && (t.dtype == "i32" || t.dtype == "i64")
}
/// The type of a vector-shaped or matrix-shaped op (dot, cross, norm, sum, splat, component, vec, mv, mm, outer,
/// transpose, and the quaternion kinds); refuses operands of the wrong rank or extent.
fn shaped(kind: &str, ts: &[TT], lit: &Val) -> Checked<TT> {
    match kind {
        "dot" | "cross" | "norm" | "sum" | "splat" | "component" | "component_at"
        | "set_component_at" | "add_at" | "vec" => vector_shaped(kind, ts, lit),
        "mv" | "mm" | "outer" | "transpose" => matrix_shaped(kind, ts),
        _ => quat(kind, ts),
    }
}

/// The type of the vector ops: dot, cross, norm, sum, splat; the component ops go to `component_shaped`.
fn vector_shaped(kind: &str, ts: &[TT], lit: &Val) -> Checked<TT> {
    let lit_t = if let Val::Tuple(t) = lit {
        Some(t)
    } else {
        None
    };
    match kind {
        "dot" => {
            same(kind, &ts[0], &ts[1], 1, None)?;
            Ok(tt(&[], &promote(&ts[0].dtype, &ts[1].dtype)?, None))
        }
        "cross" => {
            same(kind, &ts[0], &ts[1], 1, Some(3))?;
            Ok(tt(&[3], &promote(&ts[0].dtype, &ts[1].dtype)?, None))
        }
        "norm" => {
            rank(kind, &ts[0], 1, None)?;
            Ok(tt(&[], &ts[0].dtype, None))
        }
        "sum" => {
            if ![1, 2].contains(&ts[0].shape.len()) {
                return err(format!(
                    "sum: expected a rank-1 or rank-2 operand, got {}",
                    r(&ts[0])
                ));
            }
            Ok(tt(&[], &ts[0].dtype, None))
        }
        "splat" => {
            rank(kind, &ts[0], 0, None)?;
            match lit_t {
                Some(t) if !t.is_empty() && t.iter().all(is_int) => Ok(TT {
                    shape: t.iter().filter_map(int_of).collect(),
                    dtype: ts[0].dtype.clone(),
                    tag: None,
                }),
                _ => err(format!(
                    "splat: the literal must be a target shape, got {}",
                    text_val(lit)
                )),
            }
        }
        _ => component_shaped(kind, ts, lit),
    }
}

/// The type of the component and vec ops: component, component_at, set_component_at, add_at, vec.
fn component_shaped(kind: &str, ts: &[TT], lit: &Val) -> Checked<TT> {
    let lit_t = if let Val::Tuple(t) = lit {
        Some(t)
    } else {
        None
    };
    match kind {
        "component" => {
            let (index, extent): (Vec<Val>, Vec<i64>) = if let Some(t) = lit_t {
                rank(kind, &ts[0], 2, None)?;
                (t.clone(), ts[0].shape.clone())
            } else {
                rank(kind, &ts[0], 1, None)?;
                (vec![lit.clone()], ts[0].shape.clone())
            };
            let ok = index.len() == extent.len()
                && index
                    .iter()
                    .zip(&extent)
                    .all(|(k, e)| is_int(k) && int_of(k).is_some_and(|k| 0 <= k && k < *e));
            if !ok {
                return err(format!("component: index {} is out of the static extent {} (an extent is a type, not a runtime value)", text_val(lit),
                if extent.len() > 1 { shp(&extent) } else { extent[0].to_string() }));
            }
            Ok(tt(&[], &ts[0].dtype, None))
        }
        "component_at" => {
            rank(kind, &ts[0], 1, None)?;
            if !int_index(&ts[1]) {
                return err(format!("[component index type] component_at: the index must be a rank-0 integer value, got {}. A runtime component index is an address, so it must be an integer value, never a float", r(&ts[1])));
            }
            Ok(tt(&[], &ts[0].dtype, None))
        }
        "set_component_at" | "add_at" => {
            rank(kind, &ts[0], 1, None)?;
            if !int_index(&ts[1]) {
                return err(format!(
                    "{kind}: the index must be a rank-0 integer value, got {} (see component_at)",
                    r(&ts[1])
                ));
            }
            rank(kind, &ts[2], 0, None)?;
            Ok(TT {
                shape: ts[0].shape.clone(),
                dtype: promote(&ts[0].dtype, &ts[2].dtype)?,
                tag: ts[0].tag.clone(),
            })
        }
        "vec" => {
            for t in ts {
                rank(kind, t, 0, None)?
            }
            let mut dtype = ts[0].dtype.clone();
            for t in &ts[1..] {
                dtype = promote(&dtype, &t.dtype)?
            }
            if *lit == Val::None {
                return Ok(tt(&[ts.len() as i64], &dtype, None));
            }
            let shape: Option<Vec<i64>> = lit_t
                .as_ref()
                .filter(|t| t.len() == 2)
                .and_then(|t| t.iter().map(int_of).collect());
            match shape { Some(s) if s[0] * s[1] == ts.len() as i64 => Ok(TT { shape: s, dtype, tag: None }),
                _ => err(format!("vec: a matrix shape literal must be (rows, cols) with rows * cols == {} entries, got {}", ts.len(), text_val(lit))) }
        }
        _ => err(format!("op {kind} is not a vector op")),
    }
}

/// The type of the matrix ops: mv, mm, outer, transpose.
fn matrix_shaped(kind: &str, ts: &[TT]) -> Checked<TT> {
    match kind {
        "mv" => {
            rank(kind, &ts[0], 2, None)?;
            rank(kind, &ts[1], 1, None)?;
            if ts[0].shape[1] != ts[1].shape[0] {
                return err(format!(
                    "mv: {} does not contract with {}",
                    shp(&ts[0].shape),
                    shp(&ts[1].shape)
                ));
            }
            Ok(tt(
                &[ts[0].shape[0]],
                &promote(&ts[0].dtype, &ts[1].dtype)?,
                None,
            ))
        }
        "mm" => {
            rank(kind, &ts[0], 2, None)?;
            rank(kind, &ts[1], 2, None)?;
            if ts[0].shape[1] != ts[1].shape[0] {
                return err(format!(
                    "mm: {} does not contract with {}",
                    shp(&ts[0].shape),
                    shp(&ts[1].shape)
                ));
            }
            Ok(tt(
                &[ts[0].shape[0], ts[1].shape[1]],
                &promote(&ts[0].dtype, &ts[1].dtype)?,
                None,
            ))
        }
        "outer" => {
            rank(kind, &ts[0], 1, None)?;
            rank(kind, &ts[1], 1, None)?;
            Ok(tt(
                &[ts[0].shape[0], ts[1].shape[0]],
                &promote(&ts[0].dtype, &ts[1].dtype)?,
                None,
            ))
        }
        "transpose" => {
            rank(kind, &ts[0], 2, None)?;
            Ok(tt(&[ts[0].shape[1], ts[0].shape[0]], &ts[0].dtype, None))
        }
        _ => err(format!("op {kind} is not a matrix op")),
    }
}

/// Whether `d` is a floating dtype.
fn floats(d: &str) -> bool {
    d == "f64" || d == "f32"
}
/// The tag a factorisation kind gives its result ("" when it gives none).
fn tag_of_kind(kind: &str) -> &'static str {
    pairs("FACTOR_TAG_PAIRS", kind).unwrap_or("")
}

/// The type of the linear algebra ops: factorisations, least squares, and what consumes a factor.
fn linalg_type(kind: &str, ts: &[TT]) -> Checked<TT> {
    if kind == "qr" || inn("LSTSQ_OPS", kind) {
        return tall_factor_type(kind, ts);
    }
    if inn("LINALG_FACTORISATIONS", kind) {
        return square_factor_type(kind, ts);
    }
    factor_consumer_type(kind, ts)
}

/// The type of qr and of the least squares ops over a tall matrix; refuses a non-tall or non-float matrix.
fn tall_factor_type(kind: &str, ts: &[TT]) -> Checked<TT> {
    let t = &ts[0];
    if t.shape.len() != 2 || t.shape[0] < t.shape[1] {
        return err(format!(
            "op {}: a tall matrix (rows >= columns) is factored, got {}",
            keytext::py_str(kind),
            r(t)
        ));
    }
    if !floats(&t.dtype) {
        return err(format!(
            "op {}: an f64/f32 matrix is factored, got {}",
            keytext::py_str(kind),
            r(t)
        ));
    }
    let (m, n) = (t.shape[0], t.shape[1]);
    if n < 1 {
        return err(format!(
            "op {}: the columns are static and n >= 1, got n={n}",
            keytext::py_str(kind)
        ));
    }
    if kind == "qr" {
        return Ok(TT {
            shape: t.shape.clone(),
            dtype: t.dtype.clone(),
            tag: Some(tag_of_kind(kind).into()),
        });
    }
    if kind == "lstsq_info" {
        return Ok(tt(&[], "int32", None));
    }
    let (b, rows) = (&ts[1], if kind == "lstsq" { m } else { n });
    if b.dtype != t.dtype || ![1, 2].contains(&b.shape.len()) || b.shape[0] != rows {
        return err(format!("op {}: the right-hand side is a {} vector or matrix of {rows} rows against the {m}x{n} matrix, got {}", keytext::py_str(kind), t.dtype, r(b)));
    }
    let mut s = vec![n];
    s.extend_from_slice(&b.shape[1..]);
    Ok(TT {
        shape: s,
        dtype: t.dtype.clone(),
        tag: None,
    })
}

/// The type of a factorisation of a square matrix; refuses a non-square or non-float matrix.
fn square_factor_type(kind: &str, ts: &[TT]) -> Checked<TT> {
    let t = &ts[0];
    if t.shape.len() != 2 || t.shape[0] != t.shape[1] {
        return err(format!(
            "op {}: a square matrix is factored, got {}",
            keytext::py_str(kind),
            r(t)
        ));
    }
    let n = t.shape[0];
    if !floats(&t.dtype) {
        return err(format!(
            "op {}: an f64/f32 matrix is factored, got {}",
            keytext::py_str(kind),
            r(t)
        ));
    }
    if n < 1 {
        return err(format!(
            "op {}: the dimension is static and n >= 1, got n={n}",
            keytext::py_str(kind)
        ));
    }
    Ok(TT {
        shape: vec![n, n],
        dtype: t.dtype.clone(),
        tag: Some(tag_of_kind(kind).into()),
    })
}

/// The type of an op that reads the result of a factorisation (a solve, a determinant, an info word, a projection).
fn factor_consumer_type(kind: &str, ts: &[TT]) -> Checked<TT> {
    let want = pairs("FACTOR_NEEDS_PAIRS", kind).unwrap_or("");
    let t0 = &ts[0];
    if t0.tag.as_deref() != Some(want) {
        let producer = c("FACTOR_TAG_PAIRS")
            .chunks(2)
            .find(|p| p[1] == want)
            .map(|p| p[0])
            .unwrap_or("");
        return err(format!(
            "op {} reads the result of {} (a {want}), got {}",
            keytext::py_str(kind),
            keytext::py_str(producer),
            r(t0)
        ));
    }
    let n = *t0.shape.last().unwrap();
    let dtype = t0.dtype.clone();
    if kind == "qr_solve" {
        let (m, b) = (t0.shape[0], &ts[1]);
        if b.dtype != dtype || ![1, 2].contains(&b.shape.len()) || b.shape[0] != m {
            return err(format!("op {}: the right-hand side is a {dtype} vector ({m},) or matrix ({m}, k) against the {m}x{n} factor, got {}", keytext::py_str(kind), r(b)));
        }
        let mut s = vec![n];
        s.extend_from_slice(&b.shape[1..]);
        return Ok(TT {
            shape: s,
            dtype,
            tag: None,
        });
    }
    if inn("LINALG_CONSUMERS", kind) {
        let b = &ts[1];
        if b.dtype != dtype || ![1, 2].contains(&b.shape.len()) || b.shape[0] != n {
            return err(format!("op {}: the right-hand side is a {dtype} vector ({n},) or matrix ({n}, k) against the {n}x{n} factor, got {}", keytext::py_str(kind), r(b)));
        }
        return Ok(TT {
            shape: b.shape.clone(),
            dtype,
            tag: None,
        });
    }
    if kind.ends_with("_info") {
        return Ok(tt(&[], "int32", None));
    }
    if kind == "lu_det" {
        return Ok(TT {
            shape: vec![],
            dtype,
            tag: None,
        });
    }
    if kind == "eigh_w" {
        return Ok(TT {
            shape: vec![n],
            dtype,
            tag: None,
        });
    }
    Ok(TT {
        shape: vec![n, n],
        dtype,
        tag: None,
    })
}
/// The type of a ray op.
fn rays_type(kind: &str, ts: &[TT]) -> Checked<TT> {
    let scalars: Vec<usize> = pairs("RAY_SCALAR_PAIRS", kind)
        .unwrap()
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    for (k, t) in ts.iter().enumerate() {
        let want: Vec<i64> = if scalars.contains(&k) {
            vec![]
        } else {
            vec![3]
        };
        if t.shape != want || !floats(&t.dtype) {
            return err(format!(
                "op {}: operand {k} is {} f64/f32 value, got {}",
                keytext::py_str(kind),
                if want.is_empty() {
                    "a rank-0"
                } else {
                    "a length-3"
                },
                r(t)
            ));
        }
    }
    Ok(tt(
        &[4],
        if ts.iter().any(|t| t.dtype == "f64") {
            "f64"
        } else {
            "f32"
        },
        None,
    ))
}

/// The result type of the op `kind` over operands of types `ts` and the literal `lit`: checks the vocabulary, the arity,
/// the operand types and the broadcast, and refuses what the op's rules do not allow.
pub fn compute(kind: &str, ts: &[TT], lit: &Val) -> Checked<TT> {
    let kinds = all_kinds();
    if !kinds.contains(&kind) {
        let v = Val::Tuple(kinds.iter().map(|k| Val::Str(k.to_string())).collect());
        return err(format!(
            "unknown op kind {}; the vocabulary is {}",
            keytext::py_str(kind),
            text_val(&v)
        ));
    }
    arity(kind, ts.len())?;
    if kind == "cast" {
        return cast_type(ts, lit);
    }
    if inn("LINALG_FACTORISATIONS", kind)
        || inn("LINALG_CONSUMERS", kind)
        || inn("LINALG_PROJECTIONS", kind)
        || inn("LSTSQ_OPS", kind)
    {
        return linalg_type(kind, ts);
    }
    if inn("RAY_OPS", kind) {
        return rays_type(kind, ts);
    }
    let int_op = inn("INT_BINARY", kind) || inn("INT_UNARY", kind);
    check_integer_use(kind, ts, int_op)?;
    if let Some(done) = elementwise_type(kind, ts, int_op)? {
        return Ok(done);
    }
    if let Some(done) = selection_type(kind, ts)? {
        return Ok(done);
    }
    if inn("RANDOM_OPS", kind) {
        return random_type(kind, ts);
    }
    shaped(kind, ts, lit)
}

/// The type of a cast: the literal names the target dtype; refuses a target outside the list or a non-number operand.
fn cast_type(ts: &[TT], lit: &Val) -> Checked<TT> {
    let ok = matches!(lit, Val::Str(s) if inn("CAST_TARGETS", s));
    if !ok {
        return err(format!(
            "cast: the target dtype must be one of {}, got {}",
            text_val(&Val::Tuple(
                c("CAST_TARGETS")
                    .iter()
                    .map(|s| Val::Str(s.to_string()))
                    .collect()
            )),
            text_val(lit)
        ));
    }
    if !ts[0].shape.is_empty() || ts[0].dtype == "bool" {
        return err(format!("[cast operand] cast: converts a rank-0 number, got {}; a bool is not a number: choose between two values with a select", r(&ts[0])));
    }
    if let Val::Str(s) = lit {
        return Ok(tt(&[], s, None));
    }
    err(format!(
        "cast: the target must be a dtype name, got {}",
        text_val(lit)
    ))
}

/// Checks the integer rules: an integer operator takes rank-0 integer operands, and an integer plane value is rank-0, of
/// one dtype, and only in the ops that allow it.
fn check_integer_use(kind: &str, ts: &[TT], int_op: bool) -> Checked<()> {
    if int_op {
        for t in ts {
            if !t.shape.is_empty()
                || !(inn("INT_PLANE_DTYPES", &t.dtype) || t.dtype == "i32" || t.dtype == "i64")
            {
                return err(format!("[integer operator operands] op {} is an integer operator and takes rank-0 integer operands, got {}; for floats use a floor or remainder op, or convert the operands to integers", keytext::py_str(kind), r(t)));
            }
        }
    } else if ts.iter().any(|t| inn("INT_PLANE_DTYPES", &t.dtype)) {
        let plane = ts
            .iter()
            .find(|t| inn("INT_PLANE_DTYPES", &t.dtype))
            .unwrap()
            .dtype
            .clone();
        let allowed = ["add", "sub", "mul", "min", "max", "neg", "abs", "select"].contains(&kind)
            || inn("COMPARE", kind)
            || inn("INT_BINARY", kind)
            || inn("INT_UNARY", kind);
        if !allowed {
            let hint = if kind == "div" {
                " (an integer value divides with the integer division op)"
            } else {
                ""
            };
            return err(format!("[integer plane value] op {} is not defined on an {plane} value{hint}; convert it to a float first to compute in floating point", keytext::py_str(kind)));
        }
        if ts.iter().any(|t| !t.shape.is_empty()) {
            return err(format!(
                "op {}: an {plane} value is rank-0 (a per-sample integer plane), got operands {}",
                keytext::py_str(kind),
                rl(ts)
            ));
        }
        if kind != "select" {
            for t in ts {
                if t.dtype != plane {
                    return err(mix_message(&plane, &t.dtype));
                }
            }
        }
    }
    Ok(())
}

/// The type of an elementwise op (arithmetic, comparison, logic, unary math, ternary), or None when `kind` is not one;
/// refuses a bool operand where a number is needed.
fn elementwise_type(kind: &str, ts: &[TT], int_op: bool) -> Checked<Option<TT>> {
    let ebin = inn("ELEMENTWISE_BINARY", kind);
    let ter = inn("ELEMENTWISE_TERNARY", kind);
    let um = inn("UNARY_MATH", kind);
    if ts.iter().any(|t| t.dtype == "bool") && (ebin || um || ter || int_op) {
        return err(format!("[bool operand] op {} takes numbers, not a bool condition: use land, lor, lnot for logic, or a select between values for a number", keytext::py_str(kind)));
    }
    let binary_logical = inn("LOGICAL", kind) && kind != "lnot";
    if ebin || inn("COMPARE", kind) || binary_logical {
        let shape = broadcast(kind, &ts[0], &ts[1])?;
        let boolean = inn("COMPARE", kind) || inn("LOGICAL", kind);
        let dtype = if boolean {
            "bool".to_string()
        } else {
            promote(&ts[0].dtype, &ts[1].dtype)?
        };
        return Ok(Some(TT {
            shape,
            dtype,
            tag: tag_of(kind, ts),
        }));
    }
    if um || kind == "lnot" || inn("PREDICATE", kind) {
        let dtype = if kind == "lnot" || inn("PREDICATE", kind) {
            "bool".to_string()
        } else {
            ts[0].dtype.clone()
        };
        return Ok(Some(TT {
            shape: ts[0].shape.clone(),
            dtype,
            tag: tag_of(kind, ts),
        }));
    }
    if ter {
        let b01 = TT {
            shape: broadcast(kind, &ts[0], &ts[1])?,
            dtype: "f64".into(),
            tag: None,
        };
        let shape = broadcast(kind, &b01, &ts[2])?;
        let dtype = promote(&promote(&ts[0].dtype, &ts[1].dtype)?, &ts[2].dtype)?;
        return Ok(Some(TT {
            shape,
            dtype,
            tag: None,
        }));
    }
    Ok(None)
}

/// The type of select, dispatch, at and the integer ops, or None when `kind` is none of them.
fn selection_type(kind: &str, ts: &[TT]) -> Checked<Option<TT>> {
    match kind {
        "select" => {
            if ts[0].dtype != "bool" {
                return err(format!(
                    "select: the condition must be bool-typed, got {}",
                    r(&ts[0])
                ));
            }
            if ts[1] != ts[2] {
                return err(format!(
                    "select: the branches disagree — {} vs {}",
                    r(&ts[1]),
                    r(&ts[2])
                ));
            }
            return Ok(Some(ts[1].clone()));
        }
        _ if inn("INT_BINARY", kind) => {
            return Ok(Some(tt(&[], &promote(&ts[0].dtype, &ts[1].dtype)?, None)))
        }
        _ if inn("INT_UNARY", kind) => return Ok(Some(ts[0].clone())),
        "dispatch" => {
            if !ts[0].shape.is_empty() || ts[0].dtype != "i32" {
                return err(format!(
                    "dispatch: kind must be a rank-0 i32 value, got {}",
                    r(&ts[0])
                ));
            }
            for t in &ts[2..] {
                if *t != ts[1] {
                    return err(format!(
                        "dispatch: the branches disagree — {} vs {}",
                        r(&ts[1]),
                        r(t)
                    ));
                }
            }
            return Ok(Some(ts[1].clone()));
        }
        "at" => return Ok(Some(ts[0].clone())),
        _ => {}
    }
    Ok(None)
}

/// The type of a random draw: a seed and a counter that are rank-0 integers give an f64.
fn random_type(kind: &str, ts: &[TT]) -> Checked<TT> {
    for (pname, t) in ["seed", "counter"].iter().zip(ts) {
        if !t.shape.is_empty() || !(t.dtype == "i32" || t.dtype == "i64") {
            return err(format!("{kind}: {pname} must be a rank-0 i32/i64 value (a literal, a by-value uniform wire, or a plane read), got {}", r(t)));
        }
    }
    Ok(tt(&[], "f64", None))
}

impl Ctx {
    /// The type of node `h`.
    fn tt_of(&self, h: u32) -> TT {
        let n = &self.nodes[h as usize];
        TT {
            shape: n.shape.clone(),
            dtype: n.dtype.clone(),
            tag: n.tag.clone(),
        }
    }

    /// `node` where a matrix is wanted: a Cholesky factor becomes its `chol_factor` projection, another factor is refused by name.
    pub fn as_matrix(&mut self, node: u32) -> Checked<u32> {
        let t = self.tt_of(node);
        match t.tag.as_deref() {
            Some("chol_factor") => self.make(
                "Op",
                &t.dtype,
                &t.shape,
                None,
                &[node],
                "(('kind', 'chol_factor'), ('literal', None))",
            ),
            Some(tag @ ("lu_factor" | "eigh_factor" | "qr_factor")) => {
                let (what, reads) = match tag {
                    "lu_factor" => ("an LU", "lu_solve, lu_det or lu_info"),
                    "eigh_factor" => ("an eigh", "eigh_w, eigh_v or eigh_info"),
                    _ => ("a QR", "qr_solve or qr_info"),
                };
                err(format!("{what} factor is not a matrix: it carries pivots and a status, so it is read with {reads}, never as an operand of arithmetic"))
            }
            _ => Ok(node),
        }
    }

    /// The type `kind` returns over the operands, with its refusal.
    pub fn result_type(&self, kind: &str, ops: &[u32], lit: &Val) -> Checked<TT> {
        let ts: Vec<TT> = ops
            .iter()
            .map(|&o| self.node(o).map(|_| self.tt_of(o)))
            .collect::<Checked<_>>()?;
        compute(kind, &ts, lit)
    }

    /// Mints an Op of `kind` with its inferred type (a factorisation used as a matrix is replaced by what a matrix use means).
    pub fn make_op(&mut self, kind: &str, ops: &[u32], lit: &Val) -> Checked<u32> {
        let mut ops: Vec<u32> = ops.to_vec();
        for &o in &ops {
            self.node(o)?;
        }
        if all_kinds().contains(&kind) {
            let keep = if inn("LINALG_CONSUMERS", kind) || inn("LINALG_PROJECTIONS", kind) {
                1
            } else {
                0
            };
            for op in &mut ops[keep..] {
                *op = self.as_matrix(*op)?
            }
        }
        let t = self.result_type(kind, &ops, lit)?;
        let fields = text_val(&Val::Tuple(vec![
            Val::Tuple(vec![Val::Str("kind".into()), Val::Str(kind.into())]),
            Val::Tuple(vec![Val::Str("literal".into()), lit.clone()]),
        ]));
        self.make("Op", &t.dtype, &t.shape, t.tag.as_deref(), &ops, &fields)
    }
}
