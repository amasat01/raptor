//! primitive.py's algorithms: the closure check of a rule template, its structural key and its instantiation (loop bodies included).
use super::*;
use std::collections::{HashMap, HashSet};

type B<'py> = Bound<'py, PyAny>;
fn id(n: &B<'_>) -> usize { n.as_ptr() as usize }
fn ops<'py>(n: &B<'py>) -> PyResult<Vec<B<'py>>> { n.getattr("operands")?.try_iter()?.collect() }
/// The nodes a loop holds outside its operands: the body's results and the break (its index and carries are its own).
fn edge_nodes<'py>(n: &B<'py>) -> PyResult<Vec<B<'py>>> {
    if !canon::is_loop(n)? { return Ok(vec![]) }
    let mut v: Vec<B<'py>> = n.getattr("nexts")?.try_iter()?.collect::<PyResult<_>>()?;
    for x in n.getattr("exit_values")?.try_iter()? { v.push(x?) }
    let c = n.getattr("exit_cond")?;
    if !c.is_none() { v.push(c) }
    Ok(v)
}
fn pyrepr(o: &B<'_>) -> PyResult<String> { Ok(o.repr()?.to_string()) }

/// The structural key of a template node: kind, type, extra, leaf name, loop state and the keys of its operands and loop edges.
fn key<'py>(py: Python<'py>, node: &B<'py>) -> PyResult<B<'py>> {
    let mut memo: HashMap<usize, B<'py>> = HashMap::new();
    let mut stack = vec![node.clone()];
    while let Some(n) = stack.last().cloned() {
        if memo.contains_key(&id(&n)) { stack.pop(); continue }
        let kids: Vec<B<'py>> = ops(&n)?.into_iter().chain(edge_nodes(&n)?).collect();
        let todo: Vec<B<'py>> = kids.iter().filter(|c| !memo.contains_key(&id(c))).cloned().collect();
        if !todo.is_empty() { stack.extend(todo); continue }
        stack.pop();
        let name = if canon::is_a(&n, "Leaf")? { n.getattr("name")? } else { py.None().into_bound(py) };
        let state = if canon::is_loop(&n)? {
            let items = n.call_method0("state")?.call_method0("items")?;
            s(py, &pyrepr(&py.import("builtins")?.getattr("sorted")?.call1((items,))?)?)?
        } else { py.None().into_bound(py) };
        let ok: Vec<B<'py>> = ops(&n)?.iter().map(|c| memo[&id(c)].clone()).collect();
        let ek: Vec<B<'py>> = edge_nodes(&n)?.iter().map(|c| memo[&id(c)].clone()).collect();
        let k = PyTuple::new(py, [n.getattr("kind")?, n.getattr("ttype")?, n.getattr("dedup_extra")?, name, state, PyTuple::new(py, ok)?.into_any(), PyTuple::new(py, ek)?.into_any()])?;
        memo.insert(id(&n), k.into_any());
    }
    Ok(memo[&id(node)].clone())
}

/// A template is CLOSED: its free leaves are the allowed placeholders (inside a loop body too), a loop's index and carries are read only
/// inside that loop, and nothing in it writes a plane or reads a tape of a loop outside it.
fn check_template<'py>(py: Python<'py>, name: &str, node: &B<'py>, allowed: &B<'py>) -> PyResult<()> {
    let mut owned: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut stack: Vec<(B<'py>, Vec<usize>)> = vec![(node.clone(), vec![])];
    let mut seen: HashSet<(usize, Vec<usize>)> = HashSet::new();
    while let Some((n, loops)) = stack.pop() {
        if !seen.insert((id(&n), loops.clone())) { continue }
        let loop_ = canon::is_loop(&n)?;
        if !loop_ && !n.call_method0("edges")?.is_empty()? {
            return Err(he(format!("primitive {}: a rule template may not hold a {} node (it holds edges outside a loop)", rep_s(name), canon::st(&n, "kind")?)))
        }
        if canon::is_a(&n, "Leaf")? && !allowed.contains(n.getattr("name")?)? {
            let sorted = pyrepr(&py.import("builtins")?.getattr("sorted")?.call1((allowed,))?)?;
            return Err(he(format!("primitive {}: a rule template reads {} {}{}: only its inputs, the incoming adjoint and the input tangents may be free in a rule ({})",
                rep_s(name), canon::st(&n, "role")?, pyrepr(&n.getattr("name")?)?, if loops.is_empty() { "" } else { " inside a loop body" }, sorted)))
        }
        if (canon::is_a(&n, "LoopIndex")? || canon::is_a(&n, "LoopCarry")?) && !loops.iter().any(|lp| owned.get(lp).is_some_and(|o| o.contains(&id(&n)))) {
            return Err(he(format!("primitive {}: a rule template reads the index or a carry of a loop outside that loop", rep_s(name))))
        }
        if canon::st(&n, "kind")? == str_const("TAPE_READ_KIND") {
            return Err(he(format!("primitive {}: a rule template may not read a tape (it belongs to a loop outside the template)", rep_s(name))))
        }
        for c in ops(&n)? { stack.push((c, loops.clone())) }
        if loop_ {
            if !n.getattr("body_sinks")?.is_empty()? {
                return Err(he(format!("primitive {}: a loop in a rule template may not write a plane (it has body sinks)", rep_s(name))))
            }
            let mut own: HashSet<usize> = HashSet::new();
            own.insert(id(&n.getattr("index")?));
            for c in n.getattr("carries")?.try_iter()? { own.insert(id(&c?)); }
            owned.insert(id(&n), own);
            let mut inner = loops.clone(); inner.push(id(&n));
            for c in edge_nodes(&n)? { stack.push((c, inner.clone())) }
        }
    }
    Ok(())
}
fn rep_s(name: &str) -> String { rcore::keytext::py_str(name) }

fn subst_into<'py>(py: Python<'py>, roots: Vec<B<'py>>, mapping: &B<'py>, memo: &mut HashMap<usize, B<'py>>) -> PyResult<()> {
    let mut stack = roots;
    while let Some(n) = stack.last().cloned() {
        if memo.contains_key(&id(&n)) { stack.pop(); continue }
        if canon::is_a(&n, "Leaf")? {
            let got = mapping.call_method1("get", (n.getattr("name")?, &n))?;
            memo.insert(id(&n), got); stack.pop(); continue
        }
        let cs = ops(&n)?;
        let todo: Vec<B<'py>> = cs.iter().filter(|c| !memo.contains_key(&id(c))).cloned().collect();
        if !todo.is_empty() { stack.extend(todo); continue }
        stack.pop();
        let new: Vec<B<'py>> = cs.iter().map(|c| memo[&id(c)].clone()).collect();
        let out = if canon::is_loop(&n)? {
            let cls = |c: &str| reg().by_name.get(c).unwrap().bind(py).clone();
            let ix = n.getattr("index")?;
            let index = cls("LoopIndex").call1((ix.getattr("depth")?, ix.getattr("name")?, ix.getattr("ttype")?, ix.getattr("bounds")?))?;
            let mut carries = vec![];
            for c in n.getattr("carries")?.try_iter()? {
                let c = c?;
                let nc = cls("LoopCarry").call1((c.getattr("depth")?, c.getattr("slot")?, c.getattr("name")?, c.getattr("ttype")?))?;
                memo.insert(id(&c), nc.clone()); carries.push(nc);
            }
            memo.insert(id(&ix), index.clone());
            subst_into(py, edge_nodes(&n)?, mapping, memo)?;
            let at = |a: &str| -> PyResult<Vec<B<'py>>> { n.getattr(a)?.try_iter()?.map(|x| Ok(memo[&id(&x?)].clone())).collect() };
            let d = PyDict::new(py);
            d.set_item("index", index)?; d.set_item("carries", PyTuple::new(py, carries)?)?;
            d.set_item("nexts", PyTuple::new(py, at("nexts")?)?)?; d.set_item("body_sinks", PyTuple::empty(py))?;
            let ec = n.getattr("exit_cond")?;
            d.set_item("exit_cond", if ec.is_none() { ec } else { memo[&id(&ec)].clone() })?;
            d.set_item("exit_values", PyTuple::new(py, at("exit_values")?)?)?;
            n.call_method1("rebuilt", (PyTuple::new(py, new)?, d))?
        } else if new.iter().zip(cs.iter()).all(|(a, b)| id(a) == id(b)) { n.clone() } else { n.call_method1("rebuilt", (PyTuple::new(py, new)?,))? };
        memo.insert(id(&n), out);
    }
    Ok(())
}

/// `template` with each placeholder leaf replaced by `mapping[name]`, through operands and loop bodies, in one traversal.
#[pyfunction]
pub fn rule_substitute<'py>(py: Python<'py>, template: &B<'py>, mapping: &B<'py>) -> PyResult<B<'py>> {
    let mut memo = HashMap::new();
    subst_into(py, vec![template.clone()], mapping, &mut memo)?;
    Ok(memo[&id(template)].clone())
}

/// The templates of one rule checked against the arity and the types they owe and for closure; returns their keys.
/// `signature` is (input types, output type); `prefix` is the placeholder prefix.
#[pyfunction]
pub fn rule_prepare<'py>(py: Python<'py>, name: &str, signature: &B<'py>, direction: &str, templates: &B<'py>, prefix: &str) -> PyResult<(B<'py>, B<'py>)> {
    let ins = signature.get_item(0)?; let out = signature.get_item(1)?;
    let n = ins.len()?;
    let allowed = pyo3::types::PySet::empty(py)?;
    let add = |t: String| allowed.add(format!("{prefix}{t}"));
    let (tpls, owed): (Vec<B<'py>>, Vec<B<'py>>) = match direction {
        "vjp" => {
            let t: Vec<B<'py>> = templates.try_iter()?.collect::<PyResult<_>>()?;
            if t.len() != n {
                return Err(he(format!("primitive {}: its vjp returned {} gradient(s) for {} input(s). A reverse rule returns ONE adjoint per declared input, in call order \u{2014} a short tuple would silently leave an input's gradient at zero", rep_s(name), t.len(), n)))
            }
            add("out".into())?; add("cot".into())?; for i in 0..n { add(format!("in{i}"))? }
            (t, ins.try_iter()?.collect::<PyResult<_>>()?)
        }
        "jvp" => {
            add("out".into())?; for i in 0..n { add(format!("in{i}"))? } for i in 0..n { add(format!("dot{i}"))? }
            (vec![templates.clone()], vec![out])
        }
        _ => return Err(he(format!("primitive {}: unknown direction {}", rep_s(name), pyrepr(&s(py, direction)?)?))),
    };
    let mut keys = vec![];
    for (tpl, want) in tpls.iter().zip(owed.iter()) {
        let tt = tpl.getattr("ttype")?;
        if tt.ne(want)? {
            return Err(he(format!("primitive {}: its {} produced a value typed {} where {} was owed. A supplied rule is checked against the type it stands in for, because a rank or dtype mismatch here reaches the emitter as a shape error far from the declaration that caused it", rep_s(name), direction, pyrepr(&tt)?, pyrepr(want)?)))
        }
        check_template(py, name, tpl, allowed.as_any())?;
        keys.push(key(py, tpl)?);
    }
    Ok((PyTuple::new(py, tpls)?.into_any(), PyTuple::new(py, keys)?.into_any()))
}
