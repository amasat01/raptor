//! Dialect classes: kinds declared from a manifest into one context, and the checks a node of such a class passes at make.
use crate::class::{Class, Decl};
use crate::door::{cstr, guard, slice};
use crate::record::{badarg, refuse, Ctx, Fault, Node};
use crate::value::{parse, Val};
use rcore::json::{self, J};
use rcore::keytext::py_str;
use rcore::Rule;
use std::ffi::{c_char, c_int};

const KEYS: [&str; 14] = ["name", "dialect", "ext", "family", "refines", "wire", "operands", "fields", "result", "identity", "spelling", "diff", "no_writer", "feathers"];
const FIELD_TYPES: [&str; 6] = ["int", "float", "bool", "str", "none", "tuple"];
const RESERVED: [&str; 6] = ["name", "role", "kind", "ttype", "operands", "decl"];

fn bad<T>(name: &str, why: String) -> Result<T, Fault> { refuse(format!("kind {}: {why}", py_str(name))) }

/// A manifest value as a plain field value (numbers without a fraction are ints).
fn plain(j: &J) -> Val {
    match j {
        J::Null => Val::None, J::Bool(b) => Val::Bool(*b),
        J::Num(n) if n.fract() == 0.0 && n.abs() < 1e15 => Val::Int(*n as i128), J::Num(n) => Val::Float(*n),
        J::Str(s) => Val::Str(s.clone()), J::Arr(a) => Val::Tuple(a.iter().map(plain).collect()),
        J::Obj(_) => Val::None,
    }
}
fn text(j: Option<&J>) -> Option<&str> { j.and_then(J::str) }

/// kinds.py `_has_type`.
fn has_type(v: &Val, t: &str) -> bool {
    match t { "int" => matches!(v, Val::Int(_)), "float" => matches!(v, Val::Int(_) | Val::Float(_)), "bool" => matches!(v, Val::Bool(_)),
              "str" => matches!(v, Val::Str(_)), "none" => *v == Val::None, _ => matches!(v, Val::Tuple(_)) }
}

impl Ctx {
    /// Declares the kinds of a manifest into this context. Rows naming a core class are skipped. An equal re-declaration is accepted.
    pub fn declare_dialect(&mut self, manifest: &str, first_party: &str) -> Result<(), Fault> {
        let m = json::parse(manifest).map_err(|e| Fault::Refused(format!("manifest: {e}")))?;
        let Some(ext) = text(m.get("name")) else { return refuse("manifest: `name` is required") };
        let first: Vec<String> = match parse(first_party).map_err(Fault::Refused)? {
            Val::Tuple(v) | Val::List(v) => v.into_iter().filter_map(|x| if let Val::Str(s) = x { Some(s) } else { None }).collect(),
            _ => return refuse("first_party: expected a tuple of str"),
        };
        for row in m.get("kinds").map_or(&[][..], |k| k.arr()) {
            if row.get("class").is_some() { continue }
            self.declare_row(row, ext, text(m.get("dialect")).unwrap_or(""), first.iter().any(|f| f == ext))?;
        }
        Ok(())
    }

    fn declare_row(&mut self, row: &J, ext: &str, dialect: &str, first_party: bool) -> Result<(), Fault> {
        let Some(name) = text(row.get("name")).filter(|n| !n.is_empty()) else { return refuse("kind declaration: `name` is required") };
        let mut unknown: Vec<&str> = row.obj().iter().map(|k| k.0.as_str()).filter(|k| !KEYS.contains(k) && *k != "class").collect();
        unknown.sort();
        if !unknown.is_empty() { return bad(name, format!("unknown key(s) {unknown:?}")) }
        let _ = dialect;
        let mut ec = ext.strip_suffix('\n').unwrap_or(ext).chars();
        if !(ec.next().is_some_and(|f| f.is_ascii_lowercase()) && ec.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '_' || x == '-')) {
            return bad(name, format!("extension id {} is not [a-z][a-z0-9_-]*", py_str(ext)))
        }
        let dotted = ext.replace('-', "_");
        if !first_party && !name.starts_with(&format!("{dotted}.")) { return bad(name, format!("a third-party kind is spelled '{dotted}.<name>': unqualified names are reserved to the core and first-party extensions")) }
        let t = &self.table;
        if t.kind(name).is_some_and(|i| t.classes[i].decl.is_none()) { return bad(name, "a core kind name is reserved".into()) }
        let family = text(row.get("family")).unwrap_or("");
        if !["expr", "sink", "terminal"].contains(&family) { return bad(name, format!("family {} is not one of ('expr', 'sink', 'terminal')", py_str(family))) }
        let sink = family == "sink";
        let refines = text(row.get("refines"));
        let mut parent: Option<usize> = None;
        let mut base = t.find(if sink { "Sink" } else { "Node" }).unwrap();
        if let Some(r) = refines {
            let Some(i) = t.kind(r) else { return bad(name, format!("refines {}, which is not declared", py_str(r))) };
            if t.is_a(i, "Sink") != sink { return bad(name, format!("family {family} does not agree with the family of {}", py_str(r))) }
            if t.classes[i].decl.is_some() { parent = Some(i) }
            base = i;
        }
        let depth = 1 + parent.map_or(0, |p| t.classes[p].chain.iter().filter(|&&c| t.classes[c].decl.is_some()).count());
        if depth > 3 { return bad(name, "refinement deeper than 3".into()) }
        let root = if t.is_a(base, "At") { "at" } else if sink { "sink" } else { "node" };
        let pd = parent.and_then(|p| t.classes[p].decl.clone());
        // operands
        let ops = match (row.get("operands").filter(|o| !matches!(o, J::Null)), &pd) {
            (None, Some(p)) => p.ops.clone(),
            (o, _) => {
                let o = o.filter(|o| !o.obj().is_empty());
                let lo = o.and_then(|o| o.get("min")).map_or(0, |x| x.int()) as usize;
                let hi = match o.and_then(|o| o.get("max")) { None | Some(J::Null) => Some(0), Some(J::Str(s)) if s == "many" => None, Some(x) => Some(x.int() as usize) };
                (lo, if o.is_none() { Some(0) } else { hi }, o.and_then(|o| o.get("families")).map_or(vec![], |f| f.strs()))
            }
        };
        if family == "terminal" && ops.1 != Some(0) { return bad(name, "a terminal takes no operands".into()) }
        // fields
        let inherited: Vec<String> = t.classes[base].state_fields.clone();
        let mut fields: Vec<(String, String, Val, bool)> = vec![];
        for f in row.get("fields").map_or(&[][..], |f| f.arr()) {
            let (fname, ftype) = (text(f.get("name")).unwrap_or(""), text(f.get("type")).unwrap_or(""));
            if !FIELD_TYPES.contains(&ftype) { return bad(name, format!("field {} has type {}, not one of {FIELD_TYPES:?}", py_str(fname), py_str(ftype))) }
            if RESERVED.contains(&fname) || t.classes[base].slots.iter().any(|s| s == fname) || inherited.iter().any(|s| s == fname) || fields.iter().any(|x| x.0 == fname) {
                return bad(name, format!("field {} collides with a builtin or inherited name", py_str(fname)))
            }
            let def = f.get("default");
            let dv = def.map_or(Val::None, plain);
            if def.is_some() && dv != Val::None && !has_type(&dv, ftype) { return bad(name, format!("field {}: default is not a {ftype}", py_str(fname))) }
            fields.push((fname.into(), ftype.into(), dv, def.is_none()));
        }
        // result
        let result: Vec<Val> = match (row.get("result").filter(|r| !matches!(r, J::Null)), &pd) {
            (Some(r), _) => r.arr().iter().map(plain).collect(),
            (None, Some(p)) => p.result.clone(),
            _ => vec![Val::Str("carried".into())],
        };
        let rn: String = if let Some(Val::Str(s)) = result.first() { s.clone() } else { String::new() };
        let rn = rn.as_str();
        if !["same_as", "scalar_of", "fixed", "carried"].contains(&rn) { return bad(name, format!("result rule {rn:?} is unknown")) }
        if rn == "same_as" || rn == "scalar_of" {
            let i = if let Some(Val::Int(i)) = result.get(1) { *i } else { -1 };
            if i < 0 || i as usize >= ops.0 { return bad(name, format!("result operand index {i} is outside the {} required operand(s)", ops.0)) }
        }
        let identity: Option<Vec<String>> = match (row.get("identity").filter(|i| !matches!(i, J::Null)), &pd) {
            (Some(i), _) => Some(i.strs()), (None, Some(p)) => p.identity.clone(), _ => None,
        };
        if let Some(id) = &identity {
            let mut known: Vec<&str> = fields.iter().map(|f| f.0.as_str()).chain(inherited.iter().map(String::as_str)).collect();
            if sink { known.extend(["name", "role"]) }
            let mut miss: Vec<&String> = id.iter().filter(|i| !known.contains(&i.as_str())).collect();
            miss.sort();
            if !miss.is_empty() { return bad(name, format!("identity names unknown field(s) {miss:?}")) }
        }
        // feathers: named properties core.json lists; a refining kind inherits them by class
        let known = rcore::CONSTS.iter().find(|c| c.0 == "KIND_FEATHERS").map_or(&[][..], |c| c.1);
        let mut feathers: Vec<String> = vec![];
        for f in row.get("feathers").map_or(&[][..], |f| f.arr()) {
            match f.str().filter(|x| known.iter().any(|l| l.split(':').next() == Some(*x))) {
                Some(x) => if !feathers.iter().any(|y| y == x) { feathers.push(x.into()) },
                None => return bad(name, format!("unknown feather {}", crate::value::text_val(&plain(f)))),
            }
        }
        feathers.sort();
        // spelling: how the kind is written in a dialect's text; its arguments name operands
        let spelling = match row.get("spelling").filter(|x| !matches!(x, J::Null)) {
            None => pd.as_ref().and_then(|p| p.spelling.clone()),
            Some(sp) => Some((text(sp.get("dialect")).unwrap_or("").into(), text(sp.get("name")).unwrap_or("").into(), sp.get("args").map_or(vec![], |a| a.strs()))),
        };
        if let Some((_, _, args)) = &spelling {
            let hi = ops.1.unwrap_or(0);
            let ok = args.iter().all(|a| a == "node" || a == "target" || (a.starts_with("ref") && a.len() > 3 && a[3..].chars().all(|c| c.is_ascii_digit()) && {
                let i: usize = a[3..].parse().unwrap_or(usize::MAX); i < ops.0.max(1) || i < hi }));
            if !ok { return bad(name, format!("spelling args {} are not node|target|ref<i> within the operands", crate::value::text_val(&Val::Tuple(args.iter().map(|a| Val::Str(a.clone())).collect())))) }
        }
        let wire = match text(row.get("wire")).filter(|w| !w.is_empty()) {
            Some(w) => w.to_string(),
            None => match &pd { Some(p) => p.wire.clone(), None => refines.and_then(|r| t.kind(r)).and_then(|i| t.wire_of(&t.classes[i].name)).unwrap_or(name).to_string() },
        };
        let decl = Decl { name: name.into(), ext: ext.into(), family: family.into(), refines: refines.map(String::from), wire: wire.clone(), ops, fields, result, identity, feathers, spelling };
        if let Some(i) = t.kind(name) {
            return if t.classes[i].decl.as_ref() == Some(&decl) { Ok(()) } else { bad(name, "already declared with a different declaration".into()) }
        }
        let b = &t.classes[base];
        let class_name: String = name.split(['.', '_']).map(|p| { let mut c = p.chars(); c.next().map_or(String::new(), |f| f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase()) }).collect();
        let mut chain = vec![t.classes.len()];
        chain.extend(&b.chain);
        let mut slots = b.slots.clone();
        slots.extend(decl.fields.iter().map(|f| f.0.clone()));
        let mut fixed = b.fixed.clone();
        let has_kind = b.slots.iter().any(|s| s == "kind");
        if has_kind { fixed.retain(|f| f.0 != "kind"); fixed.push(("kind".into(), Val::Str(wire.clone()))) }
        let mut rule = Rule { min: decl.ops.0 as u32, max: decl.ops.1.map_or(-1, |h| h as i32), ..Default::default() };
        for (i, f) in decl.ops.2.iter().enumerate() { if f == "plane-leaf" && i < 32 { rule.plane |= 1 << i } }
        match (rn, decl.result.get(1), decl.result.get(2)) {
            ("same_as", Some(Val::Int(i)), _) => { rule.res = 1; rule.idx = *i as u32 }
            ("scalar_of", Some(Val::Int(i)), _) => { rule.res = 2; rule.idx = *i as u32 }
            ("fixed", Some(Val::Str(d)), Some(Val::Tuple(sh))) => { rule.res = 3; rule.dtype = d.clone(); rule.shape = sh.iter().map(|x| if let Val::Int(i) = x { *i as i64 } else { 0 }).collect() }
            _ => {}
        }
        let mut defaults = b.defaults.clone();
        defaults.extend(decl.fields.iter().filter(|f| !f.3).map(|f| (f.0.clone(), f.2.clone())));
        let mut state_fields = b.state_fields.clone();
        state_fields.extend(decl.fields.iter().map(|f| f.0.clone()));
        let c = Class { name: class_name, base: Some(base), chain, slots, edges: b.edges.clone(), fixed, kind: if has_kind { None } else { Some(wire) }, props: b.props.clone(),
                        spec: b.spec, rule, binds: b.binds, watch: true, decl: Some(decl), state_fields, defaults };
        self.table.push(name, c);
        Ok(())
    }
}

/// The checks of kinds.py `_make`: arity, operand families, the result rule, field types and a sink's name and role.
pub fn kind_check(ctx: &Ctx, node: &Node) -> Result<(), String> {
    let c = &ctx.table.classes[node.class];
    let Some(d) = &c.decl else { return Ok(()) };
    let why = |m: String| Err(format!("kind {}: {m}", py_str(&d.name)));
    let n = node.ops.len();
    if n < d.ops.0 || d.ops.1.is_some_and(|h| n > h) {
        return why(format!("takes {}..{} operand(s), got {n}", d.ops.0, d.ops.1.map_or("many".into(), |h| h.to_string())))
    }
    for (i, fam) in d.ops.2.iter().take(n).enumerate() {
        if fam == "plane-leaf" && !ctx.table.is_a(ctx.nodes[node.ops[i] as usize].class, "Leaf") { return why(format!("operand {i} must be a plane leaf")) }
    }
    let t = |h: u32| { let o = &ctx.nodes[h as usize]; (o.dtype.clone(), o.shape.clone(), o.tag.clone()) };
    let want = match (d.result.first(), d.result.get(1), d.result.get(2)) {
        (Some(Val::Str(r)), Some(Val::Int(i)), _) if r == "same_as" => Some(t(node.ops[*i as usize])),
        (Some(Val::Str(r)), Some(Val::Int(i)), _) if r == "scalar_of" => Some((ctx.nodes[node.ops[*i as usize] as usize].dtype.clone(), vec![], None)),
        (Some(Val::Str(r)), Some(Val::Str(dt)), Some(Val::Tuple(sh))) if r == "fixed" => Some((dt.clone(), sh.iter().map(|x| if let Val::Int(i) = x { *i as i64 } else { 0 }).collect(), None)),
        _ => None,
    };
    if let Some(w) = want { if w != (node.dtype.clone(), node.shape.clone(), node.tag.clone()) { return why(format!("type {} {:?} is not what its result rule derives, {} {:?}", node.dtype, node.shape, w.0, w.1)) } }
    for k in c.chain.iter().rev() {
        for (fname, ftype, _, _) in ctx.table.classes[*k].decl.iter().flat_map(|d| d.fields.iter()) {
            let i = c.slots.iter().position(|s| s == fname).unwrap();
            if !has_type(&node.fields[i], ftype) { return why(format!("field {} must be a {ftype}, got {}", py_str(fname), crate::value::text_val(&node.fields[i]))) }
        }
    }
    if d.family == "sink" {
        let named = |f: &str| c.slots.iter().position(|s| s == f).map(|i| &node.fields[i]);
        if !matches!(named("name"), Some(Val::Str(_))) || matches!(named("role"), None | Some(Val::None)) { return why("a sink needs `name` (and `role` unless the refined core kind fixes it)".into()) }
    }
    Ok(())
}

#[no_mangle]
pub extern "C" fn rx_declare_dialect(ctx: *mut Ctx, manifest_json: *const c_char, n: usize, first_party_kt: *const c_char) -> c_int {
    guard(ctx, |c| {
        let m = std::str::from_utf8(slice(manifest_json as *const u8, n, "manifest_json")?).map_err(|_| Fault::BadArg("manifest_json is not UTF-8".into()))?;
        let fp = if first_party_kt.is_null() { "()" } else { cstr(first_party_kt, "first_party_kt")? };
        if m.is_empty() { return badarg("manifest_json is empty") }
        c.declare_dialect(m, fp)
    })
}
