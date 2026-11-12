//! loops.py ported: the loop boundary, the gating of a dispatch arm's invariant reads, the placeholder check and the loop classifier.
use super::*;
use std::collections::{HashMap, HashSet};

type B<'py> = Bound<'py, PyAny>;
fn id(n: &B<'_>) -> usize { n.as_ptr() as usize }
fn ops<'py>(n: &B<'py>) -> PyResult<Vec<B<'py>>> { n.getattr("operands")?.try_iter()?.collect() }
fn list<'py>(n: &B<'py>, a: &str) -> PyResult<Vec<B<'py>>> { n.getattr(a)?.try_iter()?.collect() }
fn placeholder(n: &B<'_>) -> PyResult<bool> { Ok(canon::is_a(n, "LoopIndex")? || canon::is_a(n, "LoopCarry")?) }
fn kind_of(n: &B<'_>) -> Option<String> { n.getattr("kind").ok().and_then(|k| k.extract().ok()) }

/// Whether `node`'s subtree reads THIS loop's index or one of its carries (`own`), memoised on identity.
fn depends(node: &B<'_>, own: &HashSet<usize>, memo: &mut HashMap<usize, bool>) -> PyResult<bool> {
    if let Some(h) = memo.get(&id(node)) { return Ok(*h) }
    let mut stack = vec![(node.clone(), false)];
    while let Some((cur, expanded)) = stack.pop() {
        if memo.contains_key(&id(&cur)) && !expanded { continue }
        if placeholder(&cur)? { memo.insert(id(&cur), own.contains(&id(&cur))); continue }
        let cs = ops(&cur)?;
        if !expanded {
            stack.push((cur.clone(), true));
            for c in cs { if !memo.contains_key(&id(&c)) { stack.push((c, false)) } }
            continue
        }
        let v = cs.iter().any(|c| *memo.get(&id(c)).unwrap_or(&false));
        memo.insert(id(&cur), v);
    }
    Ok(memo[&id(node)])
}
/// A terminal whose kind declares the `static_index` trait: a position known at compile time.
fn static_terminal(n: &B<'_>) -> bool { n.getattr("static_index").ok().and_then(|t| t.extract::<bool>().ok()).unwrap_or(false) }
/// An index built from constants and static-index terminals only (a compile-time read position).
fn is_static(index: &B<'_>) -> PyResult<bool> {
    let mut stack = vec![index.clone()];
    while let Some(cur) = stack.pop() {
        if canon::is_a(&cur, "Const")? { continue }
        let cs = ops(&cur)?;
        if cs.is_empty() { if !static_terminal(&cur) { return Ok(false) } continue }
        stack.extend(cs);
    }
    Ok(true)
}
/// Whether `node`'s subtree contains an `At` at a runtime index (memoised on identity).
fn gathers(node: &B<'_>, memo: &mut HashMap<usize, bool>) -> PyResult<bool> {
    if let Some(h) = memo.get(&id(node)) { return Ok(*h) }
    let mut stack = vec![(node.clone(), false)];
    while let Some((cur, expanded)) = stack.pop() {
        if !expanded {
            if memo.contains_key(&id(&cur)) { continue }
            if canon::is_a(&cur, "At")? { let s = !is_static(&cur.getattr("index")?)?; memo.insert(id(&cur), s); continue }
            stack.push((cur.clone(), true));
            for c in ops(&cur)? { stack.push((c, false)) }
            continue
        }
        let v = ops(&cur)?.iter().any(|c| *memo.get(&id(c)).unwrap_or(&false));
        memo.insert(id(&cur), v);
    }
    Ok(memo[&id(node)])
}
fn id_set(o: &B<'_>) -> PyResult<HashSet<usize>> { o.try_iter()?.map(|x| x?.extract::<usize>()).collect() }

/// The body's loop-invariant boundary values, in deterministic order.
#[pyfunction]
#[pyo3(signature = (nexts, own, gates))]
pub fn loop_boundary<'py>(py: Python<'py>, nexts: &B<'py>, own: &B<'py>, gates: &B<'py>) -> PyResult<Bound<'py, PyTuple>> {
    let (own, gates) = (id_set(own)?, id_set(gates)?);
    let (mut memo, mut gmemo): (HashMap<usize, bool>, HashMap<usize, bool>) = (HashMap::new(), HashMap::new());
    let mut seen: HashMap<usize, bool> = HashMap::new();
    let mut out: Vec<B<'py>> = vec![];
    let ns: Vec<B<'py>> = nexts.try_iter()?.collect::<PyResult<_>>()?;
    let mut stack: Vec<(B<'py>, bool)> = ns.into_iter().rev().map(|n| (n, false)).collect();
    while let Some((node, guarded)) = stack.pop() {
        if let Some(prev) = seen.get(&id(&node)) { if guarded || !*prev { continue } }
        seen.insert(id(&node), guarded);
        if !depends(&node, &own, &mut memo)? && !(guarded && !gates.contains(&id(&node)) && gathers(&node, &mut gmemo)?) { out.push(node); continue }
        let cs = ops(&node)?;
        let arms: HashSet<usize> = if canon::is_a(&node, "Dispatch")? || canon::is_a(&node, "Select")? { cs.iter().skip(1).map(id).collect() } else { HashSet::new() };
        for c in cs.into_iter().rev() { let g = guarded || arms.contains(&id(&c)); stack.push((c, g)) }
    }
    let mut held = HashSet::new();
    let uniq: Vec<B<'py>> = out.into_iter().filter(|n| held.insert(id(n))).collect();
    PyTuple::new(py, uniq)
}

/// `nexts` with each loop-invariant plane read of a dispatch arm gated by the arm's own condition, and the ids of the gates made.
#[pyfunction]
pub fn loop_gated_reads<'py>(py: Python<'py>, nexts: &B<'py>, own: &B<'py>) -> PyResult<(B<'py>, Vec<usize>)> {
    let own = id_set(own)?;
    let nexts_l: Vec<B<'py>> = nexts.try_iter()?.collect::<PyResult<_>>()?;
    let substitute = py.import("raptor.ir.walk")?.getattr("substitute")?;
    let (mut memo, mut gmemo): (HashMap<usize, bool>, HashMap<usize, bool>) = (HashMap::new(), HashMap::new());
    let swaps = PyDict::new(py);
    let mut gates: Vec<usize> = vec![];
    let mut seen: HashSet<usize> = HashSet::new();
    let mut stack = nexts_l.clone();
    while let Some(node) = stack.pop() {
        if seen.contains(&id(&node)) || !depends(&node, &own, &mut memo)? { continue }
        seen.insert(id(&node));
        let cs = ops(&node)?;
        if !canon::is_a(&node, "Dispatch")? || depends(&cs[0], &own, &mut memo)? { stack.extend(cs); continue }
        stack.push(cs[0].clone());
        let sel = cs[0].clone(); let mut arms: Vec<B<'py>> = cs[1..].to_vec();
        for i in 0..arms.len() {
            let arm = arms[i].clone();
            let mut found: Vec<(usize, B<'py>)> = vec![];
            let mut walk = vec![arm.clone()]; let mut visited: HashSet<usize> = HashSet::new();
            while let Some(n) = walk.pop() {
                if !visited.insert(id(&n)) { continue }
                if canon::is_loop(&n)? { continue }
                if !depends(&n, &own, &mut memo)? {
                    if gathers(&n, &mut gmemo)? && shape_len(&n)? == 0 { if !found.iter().any(|f| f.0 == id(&n)) { found.push((id(&n), n.clone())) } continue }
                    walk.extend(ops(&n)?);
                } else if canon::is_a(&n, "Dispatch")? || canon::is_a(&n, "Select")? { walk.push(ops(&n)?[0].clone()) } else { walk.extend(ops(&n)?) }
            }
            if found.is_empty() { continue }
            let last = arms.len() - 1;
            let kind = if i == 0 { "le" } else if i == last { "ge" } else { "eq" };
            let taken = typing::make(py, kind, PyTuple::new(py, [sel.clone(), reg().by_name.get("Const").unwrap().bind(py).call1((i as i64, sel.getattr("ttype")?))?])?.as_any(), None)?;
            let gate = PyDict::new(py);
            for (key, g) in &found {
                let tt = g.getattr("ttype")?; let dt: String = tt.getattr("dtype")?.extract()?;
                let zero = if dt == "i32" || dt == "int32" { 0i64.into_pyobject(py)?.into_any() } else { 0.0f64.into_pyobject(py)?.into_any() };
                let zc = reg().by_name.get("Const").unwrap().bind(py).call1((zero, &tt))?;
                let sel_n = reg().by_name.get("Select").unwrap().bind(py).call1((&taken, g, zc, &tt))?;
                gates.push(id(&sel_n)); gate.set_item(*key, sel_n)?;
            }
            arms[i] = substitute.call1((PyList::new(py, [arm])?, gate))?.get_item(0)?;
        }
        let orig = &cs[1..];
        if arms.iter().zip(orig).any(|(a, b)| !a.is(b)) {
            let d = reg().by_name.get("Dispatch").unwrap().bind(py).call1((&sel, PyList::new(py, arms)?, node.getattr("policy")?, node.getattr("ttype")?))?;
            swaps.set_item(id(&node), d)?;
        }
    }
    let out = if swaps.len() > 0 { substitute.call1((nexts, swaps))? } else { PyTuple::new(py, nexts_l)?.into_any() };
    Ok((out, gates))
}
fn shape_len(n: &B<'_>) -> PyResult<usize> { Ok(n.getattr("ttype")?.getattr("shape")?.len()?) }

/// Every index/carry placeholder inside the body must be this loop's (an enclosing loop's may cross as an operand).
#[pyfunction]
pub fn loop_check_placeholders<'py>(lp: &B<'py>) -> PyResult<()> {
    let index = lp.getattr("index")?;
    let mut own: HashSet<usize> = HashSet::new(); own.insert(id(&index));
    for c in list(lp, "carries")? { own.insert(id(&c)); }
    let depth = index.getattr("depth")?.extract::<i64>()?;
    let mut crossing: HashSet<usize> = HashSet::new();
    for o in ops(lp)? { if placeholder(&o)? && o.getattr("depth")?.extract::<i64>()? < depth { crossing.insert(id(&o)); } }
    let iname = rep(&index.getattr("name")?);
    for node in lp.call_method0("body_nodes")?.try_iter()? {
        for child in ops(&node?)? {
            if placeholder(&child)? && !own.contains(&id(&child)) && !crossing.contains(&id(&child)) {
                let nm = child.getattr("name").map(|x| rep(&x)).unwrap_or_else(|_| "'?'".into());
                return Err(he(format!("the loop over {iname} reads {} {nm} (depth {}), which belongs to a DIFFERENT loop. A loop body may only read its own index and its own carried slots; anything else must cross the boundary as an operand",
                    kind_of(&child).unwrap_or_default(), child.getattr("depth")?.extract::<i64>()?)))
            }
        }
    }
    for root in lp.call_method0("body_roots")?.try_iter()? {
        let root = root?;
        if placeholder(&root)? && !own.contains(&id(&root)) && !crossing.contains(&id(&root)) {
            return Err(he(format!("the loop over {iname} returns a placeholder of a different loop")))
        }
    }
    Ok(())
}

fn addends<'py>(n: &B<'py>) -> PyResult<Vec<B<'py>>> {
    if kind_of(n).as_deref() == Some("add") { let o = ops(n)?; let mut a = addends(&o[0])?; a.extend(addends(&o[1])?); return Ok(a) }
    Ok(vec![n.clone()])
}
fn touches(node: &B<'_>, carry_ids: &HashSet<usize>) -> PyResult<bool> {
    let mut stack = vec![node.clone()]; let mut seen = HashSet::new();
    while let Some(cur) = stack.pop() {
        if !seen.insert(id(&cur)) { continue }
        if carry_ids.contains(&id(&cur)) { return Ok(true) }
        stack.extend(ops(&cur)?);
        if canon::is_loop(&cur)? { stack.extend(list(&cur, "nexts")?) }
    }
    Ok(false)
}
fn refuse_nesting(lp: &B<'_>) -> PyResult<()> {
    let iname = rep(&lp.getattr("index")?.getattr("name")?);
    for inner in lp.call_method0("body_nodes")?.try_iter()? {
        let inner = inner?;
        if !canon::is_loop(&inner)? { continue }
        let inm = rep(&inner.getattr("index")?.getattr("name")?);
        if !inner.getattr("exit_cond")?.is_none() {
            return Err(he(format!("loop over {iname} cannot be differentiated: its body contains a nested `for` over {inm} with a `break`. A nested loop differentiates, but one that leaves early would need its per-sample trip count threaded through every outer iteration, which is not built. Move the `break` to the outer loop, or run the inner loop to its bound and mask")))
        }
        if inner.getattr("body_sinks")?.is_truthy()? || !inner.getattr("count")?.is_none() {
            return Err(he(format!("loop over {iname} cannot be differentiated: its body contains a nested loop over {inm} that commits a sink or runs a recorded trip count (a derived loop); differentiating a derivative's nested loop is not built")))
        }
    }
    Ok(())
}
/// Which class a loop is (an accumulator with its per-carry addends, or a recurrence), or a refusal naming why.
#[pyfunction]
pub fn loop_classify<'py>(py: Python<'py>, lp: &B<'py>, loop_class: &B<'py>) -> PyResult<B<'py>> {
    refuse_nesting(lp)?;
    let iname = rep(&lp.getattr("index")?.getattr("name")?);
    for inner in lp.call_method0("body_nodes")?.try_iter()? {
        if kind_of(&inner?).as_deref() == Some(str_const("RAW_KIND")) {
            return Err(he(format!("loop over {iname} cannot be differentiated: its body splices a @raw_device block, whose text is opaque to the IR, so no rule can exist for it")))
        }
    }
    for sink in list(lp, "body_sinks")? {
        return Err(he(format!("loop over {iname} cannot be differentiated: its body commits the sink {} (role {}) once per iteration. A body sink is what a DERIVED loop carries (the reverse of a gather is a scatter); a PRIMAL body may not commit one, so differentiating this loop would be differentiating a derivative",
            rep(&sink.getattr("name")?), rep(&sink.getattr("role")?))))
    }
    let rec = || loop_class.call1(("recurrence",));
    let carries = list(lp, "carries")?; let nexts = list(lp, "nexts")?;
    let carry_ids: HashSet<usize> = carries.iter().map(id).collect();
    let mut per: Vec<Option<B<'py>>> = vec![];
    for (carry, nxt) in carries.iter().zip(&nexts) {
        let terms = addends(nxt)?;
        let bare = terms.iter().filter(|t| t.is(carry)).count();
        let rest: Vec<B<'py>> = terms.iter().filter(|t| !t.is(carry)).cloned().collect();
        if bare != 1 { return rec() }
        for t in &rest { if touches(t, &carry_ids)? { return rec() } }
        per.push(if rest.is_empty() { None } else {
            let mut it = rest.into_iter(); let mut out = it.next().unwrap();
            for t in it { out = typing::make(py, "add", PyTuple::new(py, [out, t])?.as_any(), None)? }
            Some(out)
        });
    }
    let mut reads = 0;
    for node in lp.call_method0("body_nodes")?.try_iter()? { for c in ops(&node?)? { if carry_ids.contains(&id(&c)) { reads += 1 } } }
    if reads != carries.len() || per.iter().any(|p| p.is_none()) { return rec() }
    loop_class.call1(("accumulator", PyTuple::new(py, per.into_iter().map(|p| p.unwrap()))?))
}
/// The terms of an `add` chain (`add` only).
#[pyfunction]
pub fn loop_addends<'py>(node: &B<'py>) -> PyResult<Vec<B<'py>>> { addends(node) }
/// Refuse, by name, a body whose nested `for` neither direction can carry.
#[pyfunction]
pub fn loop_refuse_nesting(lp: &B<'_>) -> PyResult<()> { refuse_nesting(lp) }
