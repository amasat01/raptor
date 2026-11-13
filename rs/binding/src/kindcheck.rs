//! The checks of a declared kind's node (kinds.py `_make`): arity, operand families, the result rule, the field set and field types.
use super::*;

type B<'py> = Bound<'py, PyAny>;
fn refuse(name: &str, why: String) -> PyErr { he(format!("kind {}: {}", rcore::keytext::py_str(name), why)) }
fn exact<'py>(v: &B<'py>, names: &[&str]) -> PyResult<bool> {
    let t = v.get_type().name()?.to_string();
    Ok(v.get_type().getattr("__module__")?.to_string() == "builtins" && names.contains(&t.as_str()))
}
/// kinds.py `_has_type`: whether `v` is a value of the declared field type.
fn has_type<'py>(py: Python<'py>, v: &B<'py>, t: &str) -> PyResult<bool> {
    match t {
        "int" => exact(v, &["int"]),
        "float" => exact(v, &["int", "float"]),
        "bool" => exact(v, &["bool"]),
        "str" => exact(v, &["str"]),
        "none" => Ok(v.is_none()),
        _ => {
            if py.import("raptor.ir.nodes")?.getattr("plain_out")?.call1((v,)).is_err() { return Ok(false) }
            Ok(v.is_instance_of::<PyTuple>())
        }
    }
}
/// The fields of every declaration along the class chain, nearest first (the first of a name wins).
fn all_fields<'py>(cls: &B<'py>) -> PyResult<Vec<(String, B<'py>, B<'py>, bool)>> {
    let mut out: Vec<(String, B<'py>, B<'py>, bool)> = vec![];
    for c in cls.getattr("__mro__")?.try_iter()? {
        let c = c?;
        let d = c.getattr("__dict__")?;
        if !d.contains("decl")? { continue }
        for f in d.get_item("decl")?.getattr("fields")?.try_iter()? {
            let f = f?;
            let n: String = f.get_item(0)?.extract()?;
            if out.iter().all(|o| o.0 != n) { out.push((n, f.get_item(1)?, f.get_item(2)?, f.get_item(3)?.extract()?)) }
        }
    }
    Ok(out)
}

/// Checks `operands` and `fields` against the declaration of `cls`; returns (the result type, the operands as a tuple, the fields with
/// the defaults filled in). `core_role` is the role the refined core sink kind fixes (None when it fixes none).
#[pyfunction]
pub fn kind_check<'py>(py: Python<'py>, cls: &B<'py>, ttype: &B<'py>, operands: &B<'py>, fields: &Bound<'py, PyDict>, core_role: &B<'py>)
    -> PyResult<(B<'py>, B<'py>, Bound<'py, PyDict>)> {
    let d = cls.getattr("decl")?;
    let name: String = d.getattr("name")?.extract()?;
    let ops: Vec<B<'py>> = operands.try_iter()?.collect::<PyResult<_>>()?;
    let spec = d.getattr("operands")?;
    let lo: usize = spec.get_item(0)?.extract()?;
    let hi: Option<usize> = spec.get_item(1)?.extract()?;
    let fams: Vec<String> = spec.get_item(2)?.extract()?;
    if ops.len() < lo || hi.is_some_and(|h| ops.len() > h) {
        return Err(refuse(&name, format!("takes {lo}..{} operand(s), got {}", hi.map_or("many".to_string(), |h| h.to_string()), ops.len())))
    }
    for (i, fam) in fams.iter().take(ops.len()).enumerate() {
        if fam == "plane-leaf" && !canon::is_a(&ops[i], "Leaf")? {
            return Err(refuse(&name, format!("operand {i} must be a plane leaf, got {}", rep(&ops[i]))))
        }
    }
    // the result rule
    let rule = d.getattr("result")?;
    let head: String = rule.get_item(0)?.extract()?;
    let tt_cls = reg().ttype.bind(py);
    let mut ttype = ttype.clone();
    let want = match head.as_str() {
        "same_as" => ops[rule.get_item(1)?.extract::<usize>()?].getattr("ttype")?,
        "scalar_of" => tt_cls.call1((PyTuple::empty(py), ops[rule.get_item(1)?.extract::<usize>()?].getattr("ttype")?.getattr("dtype")?))?,
        "fixed" => tt_cls.call1((PyTuple::new(py, rule.get_item(2)?.try_iter()?.collect::<PyResult<Vec<_>>>()?)?, rule.get_item(1)?))?,
        _ => {
            if ttype.is_none() && d.getattr("family")?.extract::<String>()? == "sink" && !ops.is_empty() { ttype = ops[0].getattr("ttype")? }
            if !ttype.is_instance(tt_cls)? { return Err(refuse(&name, format!("carries its own type, got {}", rep(&ttype)))) }
            ttype.clone()
        }
    };
    if head != "carried" && !ttype.is_none() && ttype.ne(&want)? {
        return Err(refuse(&name, format!("type {} is not what its result rule derives, {}", rep(&ttype), rep(&want))))
    }
    let ttype = want;
    // the fields
    let allf = all_fields(cls)?;
    let sink = cls.getattr("decl_root")?.extract::<String>()? == "sink";
    let given: Vec<String> = fields.keys().iter().map(|k| k.extract()).collect::<PyResult<_>>()?;
    let mut extra: Vec<&String> = given.iter().filter(|g| !allf.iter().any(|f| &f.0 == *g) && !(sink && (*g == "name" || *g == "role"))).collect();
    extra.sort();
    if !extra.is_empty() {
        return Err(refuse(&name, format!("unknown field(s) [{}]", extra.iter().map(|e| rcore::keytext::py_str(e)).collect::<Vec<_>>().join(", "))))
    }
    for (fname, ftype, default, required) in &allf {
        match fields.get_item(fname)? {
            None => {
                if *required { return Err(refuse(&name, format!("field {} is required", rcore::keytext::py_str(fname)))) }
                fields.set_item(fname, default)?;
            }
            Some(v) => {
                if !has_type(py, &v, &ftype.extract::<String>()?)? {
                    return Err(refuse(&name, format!("field {} must be a {}, got {}", rcore::keytext::py_str(fname), ftype.extract::<String>()?, rep(&v))))
                }
            }
        }
    }
    if sink {
        let role = match fields.get_item("role")? { Some(r) => r, None => core_role.clone() };
        let named = fields.get_item("name")?.is_some_and(|n| n.is_instance_of::<pyo3::types::PyString>());
        if role.is_none() || !named {
            return Err(refuse(&name, "a sink needs `name` (and `role` unless the refined core kind fixes it)".into()))
        }
    }
    Ok((ttype, PyTuple::new(py, ops)?.into_any(), fields.clone()))
}

const KEYS: [&str; 14] = ["name", "dialect", "ext", "family", "refines", "wire", "operands", "fields", "result", "identity", "spelling", "diff", "no_writer", "feathers"];
const FIELD_TYPES: [&str; 6] = ["int", "float", "bool", "str", "none", "tuple"];
fn tup<'py>(py: Python<'py>, v: Vec<B<'py>>) -> PyResult<B<'py>> { Ok(PyTuple::new(py, v)?.into_any()) }
fn text<'py>(o: &B<'py>) -> Option<String> { if o.is_instance_of::<pyo3::types::PyString>() { o.extract().ok() } else { None } }
fn get<'py>(d: &B<'py>, k: &str) -> PyResult<B<'py>> { d.call_method1("get", (k,)) }
/// The fields of a declaration and of every kind it refines, nearest first.
fn inherited<'py>(kinds: &B<'py>, from: &B<'py>) -> PyResult<Vec<(String, B<'py>)>> {
    let mut out: Vec<(String, B<'py>)> = vec![];
    let mut cur = from.clone();
    while !cur.is_none() {
        for f in cur.getattr("fields")?.try_iter()? { let f = f?; let n: String = f.get_item(0)?.extract()?; if out.iter().all(|o| o.0 != n) { out.push((n, f)) } }
        cur = kinds.call_method1("get", (cur.getattr("refines")?,))?;
    }
    Ok(out)
}

/// kinds.py `declare`: validates the plain-data declaration `data` against the declared `kinds` and the `core` kinds a kind may refine
/// (name -> (family, class, role, wire)); returns the `KindDecl` (built with `decl_cls`) and the root of its class (`node`, `sink` or `at`).
#[pyfunction]
pub fn kind_declare<'py>(py: Python<'py>, kinds: &B<'py>, core: &B<'py>, data: &B<'py>, first_party: &B<'py>, decl_cls: &B<'py>)
    -> PyResult<(B<'py>, String)> {
    let none = py.None().into_bound(py);
    let nm = get(data, "name")?;
    let name = match text(&nm) { Some(n) if !n.is_empty() => n, _ => return Err(he("kind declaration: `name` is required".into())) };
    let bi = py.import("builtins")?;
    let sorted_repr = |l: Vec<B<'py>>| -> PyResult<String> { Ok(rep(&bi.getattr("sorted")?.call1((PyList::new(py, l)?,))?)) };
    let unknown: Vec<B<'py>> = data.try_iter()?.collect::<PyResult<Vec<_>>>()?.into_iter().filter(|k| text(k).map_or(true, |t| !KEYS.contains(&t.as_str()))).collect();
    if !unknown.is_empty() { return Err(refuse(&name, format!("unknown key(s) {}", sorted_repr(unknown)?))) }
    let ext_o = get(data, "ext")?;
    let ext = match text(&ext_o) {
        Some(e) if { let e = e.strip_suffix('\n').unwrap_or(&e); let mut c = e.chars(); c.next().is_some_and(|f| f.is_ascii_lowercase()) && c.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '_' || x == '-') } => e,
        _ => return Err(refuse(&name, format!("extension id {} is not [a-z][a-z0-9_-]*", rep(&ext_o)))),
    };
    let dotted = ext.replace('-', "_");
    if !first_party.contains(&ext_o)? && !name.starts_with(&format!("{dotted}.")) {
        return Err(refuse(&name, format!("a third-party kind is spelled '{dotted}.<name>': unqualified names are reserved to the core and first-party extensions")))
    }
    if core.contains(&name)? { return Err(refuse(&name, "a core kind name is reserved".into())) }
    let fam_o = get(data, "family")?;
    let family = match text(&fam_o) { Some(f) if ["expr", "sink", "terminal"].contains(&f.as_str()) => f, _ => return Err(refuse(&name, format!("family {} is not one of ('expr', 'sink', 'terminal')", rep(&fam_o)))) };
    let sink = family == "sink";
    let refines = get(data, "refines")?;
    let mut parent = none.clone();
    if !refines.is_none() {
        let disagree = || refuse(&name, format!("family {family} does not agree with the family of {}", rep(&refines)));
        if core.contains(&refines)? {
            let cf: String = core.get_item(&refines)?.get_item(0)?.extract()?;
            if cf != if sink { "sink" } else { "expr" } { return Err(disagree()) }
        } else if kinds.contains(&refines)? {
            parent = kinds.get_item(&refines)?;
            if (parent.getattr("family")?.extract::<String>()? == "sink") != sink { return Err(disagree()) }
        } else { return Err(refuse(&name, format!("refines {}, which is not declared", rep(&refines)))) }
    }
    let (mut chain, mut cur) = (1, parent.clone());
    while !cur.is_none() { chain += 1; cur = kinds.call_method1("get", (cur.getattr("refines")?,))? }
    if chain > 3 { return Err(refuse(&name, "refinement deeper than 3".into())) }
    let mut rootname = if refines.is_none() { if sink { "sink".to_string() } else { "node".to_string() } } else { refines.extract::<String>()? };
    while !core.contains(&rootname)? {
        let k = kinds.get_item(&rootname)?;
        let r = k.getattr("refines")?;
        rootname = if !r.is_none() && r.is_truthy()? { r.extract()? } else if k.getattr("family")?.extract::<String>()? == "sink" { "sink".into() } else { "node".into() };
    }
    let root = if rootname == "at" { "at" } else if core.get_item(&rootname)?.get_item(0)?.extract::<String>()? == "sink" { "sink" } else { "node" };
    // operands
    let ops = get(data, "operands")?;
    let operands = if ops.is_none() && !parent.is_none() { parent.getattr("operands")? } else {
        let ops = if ops.is_truthy()? { ops } else { let d = PyDict::new(py); d.set_item("min", 0)?; d.set_item("max", 0)?; d.into_any() };
        let mx = get(&ops, "max")?;
        let mx = if mx.is_none() || text(&mx).as_deref() == Some("many") { none.clone() } else { bi.getattr("int")?.call1((mx,))? };
        let fams = get(&ops, "families")?;
        tup(py, vec![bi.getattr("int")?.call1((ops.get_item("min")?,))?, mx, bi.getattr("tuple")?.call1((if fams.is_none() { tup(py, vec![])? } else { fams },))?])?
    };
    let (olo, ohi) = (operands.get_item(0)?, operands.get_item(1)?);
    if family == "terminal" && !ohi.eq(0)? { return Err(refuse(&name, "a terminal takes no operands".into())) }
    // fields
    let fl = data.call_method1("get", ("fields", tup(py, vec![])?))?;
    let mut own: Vec<B<'py>> = vec![];
    for f in fl.try_iter()? {
        let f = f?;
        let has_default = f.contains("default")?;
        own.push(tup(py, vec![f.get_item("name")?, f.get_item("type")?, get(&f, "default")?, pyo3::types::PyBool::new(py, !has_default).to_owned().into_any()])?);
    }
    let inh = inherited(kinds, &parent)?;
    let node_cls = reg().node.bind(py);
    for f in &own {
        let (fname, ftype, default) = (f.get_item(0)?, f.get_item(1)?, f.get_item(2)?);
        let ft = text(&ftype);
        if !ft.as_deref().is_some_and(|t| FIELD_TYPES.contains(&t)) { return Err(refuse(&name, format!("field {} has type {}, not one of ('int', 'float', 'bool', 'str', 'none', 'tuple')", rep(&fname), rep(&ftype)))) }
        let fs = text(&fname).unwrap_or_default();
        if ["name", "role", "kind", "ttype", "operands", "decl"].contains(&fs.as_str()) || inh.iter().any(|i| i.0 == fs) || node_cls.hasattr(fs.as_str())? {
            return Err(refuse(&name, format!("field {} collides with a builtin or inherited name", rep(&fname))))
        }
        if !default.is_none() && !has_type(py, &default, ft.as_deref().unwrap())? { return Err(refuse(&name, format!("field {}: default {} is not a {}", rep(&fname), rep(&default), ft.unwrap()))) }
    }
    // result
    let rd = get(data, "result")?;
    let result = if !rd.is_none() { bi.getattr("tuple")?.call1((rd,))? } else if !parent.is_none() { parent.getattr("result")? } else { tup(py, vec![s(py, "carried")?])? };
    let r0 = result.get_item(0)?;
    let rn = text(&r0).unwrap_or_default();
    if !["same_as", "scalar_of", "fixed", "carried"].contains(&rn.as_str()) { return Err(refuse(&name, format!("result rule {} is unknown", rep(&r0)))) }
    if rn == "same_as" || rn == "scalar_of" {
        let i = result.get_item(1)?;
        if !(i.ge(0)? && i.lt(&olo)?) { return Err(refuse(&name, format!("result operand index {} is outside the {} required operand(s)", rep(&i), rep(&olo)))) }
    }
    let mut names: Vec<String> = own.iter().map(|f| f.get_item(0).and_then(|n| n.extract())).collect::<PyResult<_>>()?;
    names.extend(inh.iter().map(|i| i.0.clone()));
    if sink { names.push("name".into()); names.push("role".into()) }
    let idd = get(data, "identity")?;
    let identity = if !idd.is_none() { bi.getattr("tuple")?.call1((idd,))? } else if !parent.is_none() { parent.getattr("identity")? } else { none.clone() };
    if !identity.is_none() {
        let bad: Vec<B<'py>> = identity.try_iter()?.collect::<PyResult<Vec<_>>>()?.into_iter().filter(|i| text(i).map_or(true, |t| !names.contains(&t))).collect();
        if !bad.is_empty() {
            let uniq = bi.getattr("set")?.call1((PyList::new(py, bad)?,))?;
            return Err(refuse(&name, format!("identity names unknown field(s) {}", sorted_repr(uniq.try_iter()?.collect::<PyResult<_>>()?)?)))
        }
    }
    // feathers: named properties raptor's table lists (KIND_FEATHERS, "name: meaning"); a row states its own, a refining kind inherits by class
    let mut feathers: Vec<String> = vec![];
    let fd = get(data, "feathers")?;
    if fd.is_truthy()? {
        for f in fd.try_iter()? {
            let f = f?;
            let known = text(&f).filter(|x| const_of("KIND_FEATHERS").unwrap_or(&[]).iter().any(|l| l.split(':').next() == Some(x.as_str())));
            match known { Some(x) => if !feathers.contains(&x) { feathers.push(x) }, None => return Err(refuse(&name, format!("unknown feather {}", rep(&f)))) }
        }
        feathers.sort();
    }
    // spelling
    let spell = get(data, "spelling")?;
    let spelling = if spell.is_none() { if parent.is_none() { none.clone() } else { parent.getattr("spelling")? } } else {
        let a = data.get_item("spelling")?.call_method1("get", ("args", tup(py, vec![])?))?;
        tup(py, vec![spell.get_item("dialect")?, spell.get_item("name")?, bi.getattr("tuple")?.call1((a,))?])?
    };
    if !spelling.is_none() {
        let args = spelling.get_item(2)?;
        let hi: i64 = if ohi.is_none() { 0 } else { ohi.extract()? };
        let lo: i64 = olo.extract()?;
        let mut ok = true;
        for a in args.try_iter()? {
            let a = text(&a?).unwrap_or_default();
            let good = a == "node" || a == "target" || (a.starts_with("ref") && a.len() > 3 && a[3..].chars().all(|c| c.is_ascii_digit()) && {
                let i: i64 = a[3..].parse().unwrap_or(i64::MAX); i < lo.max(1) || i < hi });
            if !good { ok = false; break }
        }
        if !ok { return Err(refuse(&name, format!("spelling args {} are not node|target|ref<i> within the operands", rep(&args)))) }
    }
    let wire = get(data, "wire")?;
    let wire = if wire.is_truthy()? { wire } else if !parent.is_none() { parent.getattr("wire")? } else {
        let cw = if !refines.is_none() && core.contains(&refines)? { core.get_item(&refines)?.get_item(3)? } else { none.clone() };
        if cw.is_truthy()? { cw } else { s(py, &name)? }
    };
    let flag = pyo3::types::PyBool::new(py, get(data, "no_writer")?.is_truthy()?).to_owned().into_any();
    let all = vec![s(py, &name)?, data.call_method1("get", ("dialect", ""))?, s(py, &ext)?, s(py, &family)?, refines, wire, operands, tup(py, own)?, result, identity, tup(py, feathers.iter().map(|f| s(py, f)).collect::<PyResult<Vec<_>>>()?)?, spelling,
        data.call_method1("get", ("diff", "none"))?, flag];
    let decl = decl_cls.call1(PyTuple::new(py, all)?)?;
    Ok((decl, root.to_string()))
}
