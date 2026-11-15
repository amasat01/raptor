//! Dialect classes: kinds declared from a manifest into one context, and the checks a node of such a class passes at make.
use crate::class::{Class, Decl, Table};
use crate::door::{cstr, guard, slice};
use crate::record::{badarg, refuse, Ctx, Fault, Node};
use crate::value::{parse, Val};
use rcore::json::{self, J};
use rcore::keytext::py_str;
use rcore::Rule;
use std::ffi::{c_char, c_int};

/// The keys a kind declaration may state.
const KEYS: [&str; 14] = [
    "name",
    "dialect",
    "ext",
    "family",
    "refines",
    "wire",
    "operands",
    "fields",
    "result",
    "identity",
    "spelling",
    "diff",
    "no_writer",
    "feathers",
];
/// The field types a declared field may have.
const FIELD_TYPES: [&str; 6] = ["int", "float", "bool", "str", "none", "tuple"];
/// The field names a declaration may not reuse.
const RESERVED: [&str; 6] = ["name", "role", "kind", "ttype", "operands", "decl"];

/// A refusal of the kind declaration `name`.
fn bad<T>(name: &str, why: String) -> Result<T, Fault> {
    refuse(format!("kind {}: {why}", py_str(name)))
}

/// A manifest value as a plain field value (numbers without a fraction are ints).
fn plain(j: &J) -> Val {
    match j {
        J::Null => Val::None,
        J::Bool(b) => Val::Bool(*b),
        J::Num(n) if n.fract() == 0.0 && n.abs() < 1e15 => Val::Int(*n as i128),
        J::Num(n) => Val::Float(*n),
        J::Str(s) => Val::Str(s.clone()),
        J::Arr(a) => Val::Tuple(a.iter().map(plain).collect()),
        J::Obj(_) => Val::None,
    }
}
/// The text of a JSON string value, if it is one.
fn text(j: Option<&J>) -> Option<&str> {
    j.and_then(J::str)
}

/// Whether the value `v` has the declared field type `t` (int, float, bool, str, none or tuple).
fn has_type(v: &Val, t: &str) -> bool {
    match t {
        "int" => matches!(v, Val::Int(_)),
        "float" => matches!(v, Val::Int(_) | Val::Float(_)),
        "bool" => matches!(v, Val::Bool(_)),
        "str" => matches!(v, Val::Str(_)),
        "none" => *v == Val::None,
        _ => matches!(v, Val::Tuple(_)),
    }
}

impl Ctx {
    /// Declares the kinds of a manifest into this context. Rows naming a core class are skipped. An equal re-declaration is accepted.
    pub fn declare_dialect(&mut self, manifest: &str, first_party: &str) -> Result<(), Fault> {
        let m = json::parse(manifest).map_err(|e| Fault::Refused(format!("manifest: {e}")))?;
        let Some(ext) = text(m.get("name")) else {
            return refuse("manifest: `name` is required");
        };
        let first: Vec<String> = match parse(first_party).map_err(Fault::Refused)? {
            Val::Tuple(v) | Val::List(v) => v
                .into_iter()
                .filter_map(|x| if let Val::Str(s) = x { Some(s) } else { None })
                .collect(),
            _ => return refuse("first_party: expected a tuple of str"),
        };
        for row in m.get("kinds").map_or(&[][..], |k| k.arr()) {
            if row.get("class").is_some() {
                continue;
            }
            self.declare_row(row, ext, first.iter().any(|f| f == ext))?;
        }
        Ok(())
    }

    /// Declares one kind row of a manifest as a class.
    fn declare_row(&mut self, row: &J, ext: &str, first_party: bool) -> Result<(), Fault> {
        let Some(name) = text(row.get("name")).filter(|n| !n.is_empty()) else {
            return refuse("kind declaration: `name` is required");
        };
        let t = &self.table;
        check_header(t, row, ext, first_party, name)?;
        let family = family_of(row, name)?;
        let sink = family == "sink";
        let refines = text(row.get("refines"));
        let (parent, base) = lineage(t, name, family, refines, sink)?;
        let pd = parent.and_then(|p| t.classes[p].decl.clone());
        let ops = operand_rule(row, &pd, family, name)?;
        let inherited: Vec<String> = t.classes[base].state_fields.clone();
        let fields = declared_fields(t, row, name, base, &inherited)?;
        let result = result_rule(row, &pd, name, &ops)?;
        let identity = identity_rule(row, &pd, &fields, &inherited, sink, name)?;
        let feathers = feather_list(row, name)?;
        let spelling = spelling_of(row, &pd, &ops, name)?;
        let wire = wire_of_kind(row, &pd, t, refines, name);
        let decl = Decl {
            name: name.into(),
            ext: ext.into(),
            family: family.into(),
            refines: refines.map(String::from),
            wire: wire.clone(),
            ops,
            fields,
            result,
            identity,
            feathers,
            spelling,
        };
        if let Some(i) = t.kind(name) {
            return if t.classes[i].decl.as_ref() == Some(&decl) {
                Ok(())
            } else {
                bad(name, "already declared with a different declaration".into())
            };
        }
        let c = build_class(t, decl, base, wire);
        self.table.push(name, c);
        Ok(())
    }
}

/// The operand rule of a declared kind: the minimum, the maximum (None for many) and the operand families.
type Operands = (usize, Option<usize>, Vec<String>);
/// How a declared kind is written in a dialect's text: the dialect, the name and the argument names.
type Spelling = (String, String, Vec<String>);

/// Checks the kind's keys, its extension id, the third-party spelling of its name and that it does not take a core name.
fn check_header(t: &Table, row: &J, ext: &str, first_party: bool, name: &str) -> Result<(), Fault> {
    let mut unknown: Vec<&str> = row
        .obj()
        .iter()
        .map(|k| k.0.as_str())
        .filter(|k| !KEYS.contains(k) && *k != "class")
        .collect();
    unknown.sort();
    if !unknown.is_empty() {
        return bad(name, format!("unknown key(s) {unknown:?}"));
    }
    let mut ec = ext.strip_suffix('\n').unwrap_or(ext).chars();
    if !(ec.next().is_some_and(|f| f.is_ascii_lowercase())
        && ec.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '_' || x == '-'))
    {
        return bad(
            name,
            format!("extension id {} is not [a-z][a-z0-9_-]*", py_str(ext)),
        );
    }
    let dotted = ext.replace('-', "_");
    if !first_party && !name.starts_with(&format!("{dotted}.")) {
        return bad(name, format!("a third-party kind is spelled '{dotted}.<name>': unqualified names are reserved to the core and first-party extensions"));
    }
    if t.kind(name).is_some_and(|i| t.classes[i].decl.is_none()) {
        return bad(name, "a core kind name is reserved".into());
    }
    Ok(())
}

/// The family of the kind (`expr`, `sink` or `terminal`); refuses another word.
fn family_of<'a>(row: &'a J, name: &str) -> Result<&'a str, Fault> {
    let family = text(row.get("family")).unwrap_or("");
    if !["expr", "sink", "terminal"].contains(&family) {
        return bad(
            name,
            format!(
                "family {} is not one of ('expr', 'sink', 'terminal')",
                py_str(family)
            ),
        );
    }
    Ok(family)
}

/// The class a kind refines (as its declared parent, if that is itself declared) and the class it is built on; refuses a
/// refinement across families or deeper than 3.
fn lineage(
    t: &Table,
    name: &str,
    family: &str,
    refines: Option<&str>,
    sink: bool,
) -> Result<(Option<usize>, usize), Fault> {
    let mut parent: Option<usize> = None;
    let mut base = t.find(if sink { "Sink" } else { "Node" }).unwrap();
    if let Some(r) = refines {
        let Some(i) = t.kind(r) else {
            return bad(
                name,
                format!("refines {}, which is not declared", py_str(r)),
            );
        };
        if t.is_a(i, "Sink") != sink {
            return bad(
                name,
                format!(
                    "family {family} does not agree with the family of {}",
                    py_str(r)
                ),
            );
        }
        if t.classes[i].decl.is_some() {
            parent = Some(i)
        }
        base = i;
    }
    let depth = 1 + parent.map_or(0, |p| {
        t.classes[p]
            .chain
            .iter()
            .filter(|&&c| t.classes[c].decl.is_some())
            .count()
    });
    if depth > 3 {
        return bad(name, "refinement deeper than 3".into());
    }
    Ok((parent, base))
}

/// The operand rule: the parent's unless the row states one; refuses operands on a terminal.
fn operand_rule(row: &J, pd: &Option<Decl>, family: &str, name: &str) -> Result<Operands, Fault> {
    // operands
    let ops = match (row.get("operands").filter(|o| !matches!(o, J::Null)), &pd) {
        (None, Some(p)) => p.ops.clone(),
        (o, _) => {
            let o = o.filter(|o| !o.obj().is_empty());
            let lo = o.and_then(|o| o.get("min")).map_or(0, |x| x.int()) as usize;
            let hi = match o.and_then(|o| o.get("max")) {
                None | Some(J::Null) => Some(0),
                Some(J::Str(s)) if s == "many" => None,
                Some(x) => Some(x.int() as usize),
            };
            (
                lo,
                if o.is_none() { Some(0) } else { hi },
                o.and_then(|o| o.get("families"))
                    .map_or(vec![], |f| f.strs()),
            )
        }
    };
    if family == "terminal" && ops.1 != Some(0) {
        return bad(name, "a terminal takes no operands".into());
    }
    Ok(ops)
}

/// The declared fields (name, type, default, required); refuses an unknown type, a colliding name or a bad default.
fn declared_fields(
    t: &Table,
    row: &J,
    name: &str,
    base: usize,
    inherited: &[String],
) -> Result<Vec<(String, String, Val, bool)>, Fault> {
    let mut fields: Vec<(String, String, Val, bool)> = vec![];
    for f in row.get("fields").map_or(&[][..], |f| f.arr()) {
        let (fname, ftype) = (
            text(f.get("name")).unwrap_or(""),
            text(f.get("type")).unwrap_or(""),
        );
        if !FIELD_TYPES.contains(&ftype) {
            return bad(
                name,
                format!(
                    "field {} has type {}, not one of {FIELD_TYPES:?}",
                    py_str(fname),
                    py_str(ftype)
                ),
            );
        }
        if RESERVED.contains(&fname)
            || t.classes[base].slots.iter().any(|s| s == fname)
            || inherited.iter().any(|s| s == fname)
            || fields.iter().any(|x| x.0 == fname)
        {
            return bad(
                name,
                format!(
                    "field {} collides with a builtin or inherited name",
                    py_str(fname)
                ),
            );
        }
        let def = f.get("default");
        let dv = def.map_or(Val::None, plain);
        if def.is_some() && dv != Val::None && !has_type(&dv, ftype) {
            return bad(
                name,
                format!("field {}: default is not a {ftype}", py_str(fname)),
            );
        }
        fields.push((fname.into(), ftype.into(), dv, def.is_none()));
    }
    Ok(fields)
}

/// The result rule: the row's, else the parent's, else `carried`; refuses an unknown rule or an operand index outside the
/// required operands.
fn result_rule(row: &J, pd: &Option<Decl>, name: &str, ops: &Operands) -> Result<Vec<Val>, Fault> {
    // result
    let result: Vec<Val> = match (row.get("result").filter(|r| !matches!(r, J::Null)), &pd) {
        (Some(r), _) => r.arr().iter().map(plain).collect(),
        (None, Some(p)) => p.result.clone(),
        _ => vec![Val::Str("carried".into())],
    };
    let rn: String = if let Some(Val::Str(s)) = result.first() {
        s.clone()
    } else {
        String::new()
    };
    let rn = rn.as_str();
    if !["same_as", "scalar_of", "fixed", "carried"].contains(&rn) {
        return bad(name, format!("result rule {rn:?} is unknown"));
    }
    if rn == "same_as" || rn == "scalar_of" {
        let i = if let Some(Val::Int(i)) = result.get(1) {
            *i
        } else {
            -1
        };
        if i < 0 || i as usize >= ops.0 {
            return bad(
                name,
                format!(
                    "result operand index {i} is outside the {} required operand(s)",
                    ops.0
                ),
            );
        }
    }
    Ok(result)
}

/// The first word of a result rule (empty when it has none).
fn result_word(result: &[Val]) -> String {
    if let Some(Val::Str(s)) = result.first() {
        s.clone()
    } else {
        String::new()
    }
}

/// The identity fields: the row's, else the parent's; refuses a name that is no field of the kind.
fn identity_rule(
    row: &J,
    pd: &Option<Decl>,
    fields: &[(String, String, Val, bool)],
    inherited: &[String],
    sink: bool,
    name: &str,
) -> Result<Option<Vec<String>>, Fault> {
    let identity: Option<Vec<String>> =
        match (row.get("identity").filter(|i| !matches!(i, J::Null)), &pd) {
            (Some(i), _) => Some(i.strs()),
            (None, Some(p)) => p.identity.clone(),
            _ => None,
        };
    if let Some(id) = &identity {
        let mut known: Vec<&str> = fields
            .iter()
            .map(|f| f.0.as_str())
            .chain(inherited.iter().map(String::as_str))
            .collect();
        if sink {
            known.extend(["name", "role"])
        }
        let mut miss: Vec<&String> = id.iter().filter(|i| !known.contains(&i.as_str())).collect();
        miss.sort();
        if !miss.is_empty() {
            return bad(name, format!("identity names unknown field(s) {miss:?}"));
        }
    }
    Ok(identity)
}

/// The feathers the row names, each once, sorted; refuses a name the manifest does not list.
fn feather_list(row: &J, name: &str) -> Result<Vec<String>, Fault> {
    // feathers: named properties core.json lists; a refining kind inherits them by class
    let known = rcore::CONSTS
        .iter()
        .find(|c| c.0 == "KIND_FEATHERS")
        .map_or(&[][..], |c| c.1);
    let mut feathers: Vec<String> = vec![];
    for f in row.get("feathers").map_or(&[][..], |f| f.arr()) {
        match f
            .str()
            .filter(|x| known.iter().any(|l| l.split(':').next() == Some(*x)))
        {
            Some(x) => {
                if !feathers.iter().any(|y| y == x) {
                    feathers.push(x.into())
                }
            }
            None => {
                return bad(
                    name,
                    format!("unknown feather {}", crate::value::text_val(&plain(f))),
                )
            }
        }
    }
    feathers.sort();
    Ok(feathers)
}

/// The spelling of the kind: the row's, else the parent's; refuses arguments that are not node, target or ref<i> within
/// the operands.
fn spelling_of(
    row: &J,
    pd: &Option<Decl>,
    ops: &Operands,
    name: &str,
) -> Result<Option<Spelling>, Fault> {
    // spelling: how the kind is written in a dialect's text; its arguments name operands
    let spelling = match row.get("spelling").filter(|x| !matches!(x, J::Null)) {
        None => pd.as_ref().and_then(|p| p.spelling.clone()),
        Some(sp) => Some((
            text(sp.get("dialect")).unwrap_or("").into(),
            text(sp.get("name")).unwrap_or("").into(),
            sp.get("args").map_or(vec![], |a| a.strs()),
        )),
    };
    if let Some((_, _, args)) = &spelling {
        let hi = ops.1.unwrap_or(0);
        let ok = args.iter().all(|a| {
            a == "node"
                || a == "target"
                || (a.starts_with("ref")
                    && a.len() > 3
                    && a[3..].chars().all(|c| c.is_ascii_digit())
                    && {
                        let i: usize = a[3..].parse().unwrap_or(usize::MAX);
                        i < ops.0.max(1) || i < hi
                    })
        });
        if !ok {
            return bad(
                name,
                format!(
                    "spelling args {} are not node|target|ref<i> within the operands",
                    crate::value::text_val(&Val::Tuple(
                        args.iter().map(|a| Val::Str(a.clone())).collect()
                    ))
                ),
            );
        }
    }
    Ok(spelling)
}

/// The wire name of the kind: the row's, else the parent's, else that of the kind it refines, else its own name.
fn wire_of_kind(
    row: &J,
    pd: &Option<Decl>,
    t: &Table,
    refines: Option<&str>,
    name: &str,
) -> String {
    match text(row.get("wire")).filter(|w| !w.is_empty()) {
        Some(w) => w.to_string(),
        None => match &pd {
            Some(p) => p.wire.clone(),
            None => refines
                .and_then(|r| t.kind(r))
                .and_then(|i| t.wire_of(&t.classes[i].name))
                .unwrap_or(name)
                .to_string(),
        },
    }
}

/// The class a declaration makes: its chain, slots, fixed values, result rule and defaults over the class it is built on.
fn build_class(t: &Table, decl: Decl, base: usize, wire: String) -> Class {
    let name = decl.name.clone();
    let result_name = result_word(&decl.result);
    let rn = result_name.as_str();
    let b = &t.classes[base];
    let class_name: String = name
        .split(['.', '_'])
        .map(|p| {
            let mut c = p.chars();
            c.next().map_or(String::new(), |f| {
                f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase()
            })
        })
        .collect();
    let mut chain = vec![t.classes.len()];
    chain.extend(&b.chain);
    let mut slots = b.slots.clone();
    slots.extend(decl.fields.iter().map(|f| f.0.clone()));
    let mut fixed = b.fixed.clone();
    // a declared sink may state its own role; the role its core kind fixes is only the default
    let role_default = fixed
        .iter()
        .position(|f| f.0 == "role")
        .map(|i| fixed.remove(i));
    let has_kind = b.slots.iter().any(|s| s == "kind");
    if has_kind {
        fixed.retain(|f| f.0 != "kind");
        fixed.push(("kind".into(), Val::Str(wire.clone())))
    }
    let mut rule = Rule {
        min: decl.ops.0 as u32,
        max: decl.ops.1.map_or(-1, |h| h as i32),
        ..Default::default()
    };
    for (i, f) in decl.ops.2.iter().enumerate() {
        if f == "plane-leaf" && i < 32 {
            rule.plane |= 1 << i
        }
    }
    match (rn, decl.result.get(1), decl.result.get(2)) {
        ("same_as", Some(Val::Int(i)), _) => {
            rule.res = 1;
            rule.idx = *i as u32
        }
        ("scalar_of", Some(Val::Int(i)), _) => {
            rule.res = 2;
            rule.idx = *i as u32
        }
        ("fixed", Some(Val::Str(d)), Some(Val::Tuple(sh))) => {
            rule.res = 3;
            rule.dtype = d.clone();
            rule.shape = sh
                .iter()
                .map(|x| if let Val::Int(i) = x { *i as i64 } else { 0 })
                .collect()
        }
        _ => {}
    }
    let mut defaults = b.defaults.clone();
    defaults.extend(role_default);
    defaults.extend(
        decl.fields
            .iter()
            .filter(|f| !f.3)
            .map(|f| (f.0.clone(), f.2.clone())),
    );
    let mut state_fields = b.state_fields.clone();
    state_fields.extend(decl.fields.iter().map(|f| f.0.clone()));

    Class {
        name: class_name,
        base: Some(base),
        chain,
        slots,
        edges: b.edges.clone(),
        fixed,
        kind: if has_kind { None } else { Some(wire) },
        props: b.props.clone(),
        spec: b.spec,
        rule,
        binds: b.binds,
        watch: true,
        decl: Some(decl),
        state_fields,
        defaults,
    }
}

/// A dialect node from constructor parameters: `ttype` (optional: the result rule gives it), `operands`, and the declared fields.
pub fn forward(
    ctx: &Ctx,
    class: usize,
    given: &[(String, Val)],
) -> Result<crate::ctor::New, String> {
    let c = &ctx.table.classes[class];
    let d = c.decl.as_ref().ok_or("not a dialect class")?;
    if let Some((k, _)) = given
        .iter()
        .find(|g| g.0 != "ttype" && g.0 != "operands" && !c.slots.contains(&g.0))
    {
        return Err(format!(
            "kind {}: unknown field(s) [{}]",
            py_str(&d.name),
            py_str(k)
        ));
    }
    let get = |k: &str| given.iter().find(|g| g.0 == k).map(|g| g.1.clone());
    let ops: Vec<u32> = match get("operands") {
        Some(Val::Tuple(v)) | Some(Val::List(v)) => v
            .iter()
            .map(|x| {
                if let Val::Node(h) = x {
                    Ok(*h)
                } else {
                    Err("an operand is not a node".to_string())
                }
            })
            .collect::<Result<_, _>>()?,
        None => vec![],
        _ => return Err("operands: expected a tuple of nodes".into()),
    };
    let typed = forward_type(ctx, d, get("ttype"), &ops)?;
    Ok(crate::ctor::New {
        dtype: typed.dtype,
        shape: typed.shape,
        tag: typed.tag,
        ops,
        fields: given
            .iter()
            .filter(|g| g.0 != "ttype" && g.0 != "operands")
            .cloned()
            .collect(),
    })
}

/// A node's type as a dialect constructor carries it.
struct NodeType {
    /// The element type name.
    dtype: String,
    /// The extents.
    shape: Vec<i64>,
    /// The tag, if any.
    tag: Option<String>,
}

/// The type of a dialect node: the `ttype` the caller gave, else what the kind's result rule says from its operands; a
/// sink takes its first operand's type. Refuses a kind that carries its own type when none is given.
fn forward_type(ctx: &Ctx, d: &Decl, ttype: Option<Val>, ops: &[u32]) -> Result<NodeType, String> {
    let ty = |h: u32| {
        ctx.nodes
            .get(h as usize)
            .map(|n| NodeType {
                dtype: n.dtype.clone(),
                shape: n.shape.clone(),
                tag: n.tag.clone(),
            })
            .ok_or(format!("node handle {h} is not in this context"))
    };
    let typed = match ttype {
        Some(Val::Rec(_, f)) => {
            let g = |k: &str| {
                f.iter()
                    .find(|x| x.0 == k)
                    .map(|x| x.1.clone())
                    .unwrap_or(Val::None)
            };
            NodeType {
                dtype: if let Val::Str(s) = g("dtype") {
                    s
                } else {
                    return Err("ttype: dtype is a string".into());
                },
                shape: if let Val::Tuple(v) = g("shape") {
                    v.iter()
                        .filter_map(|x| {
                            if let Val::Int(i) = x {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    vec![]
                },
                tag: if let Val::Str(s) = g("tag") {
                    Some(s)
                } else {
                    None
                },
            }
        }
        _ => match (d.result.first(), d.result.get(1), d.result.get(2)) {
            (Some(Val::Str(r)), Some(Val::Int(i)), _) if r == "same_as" => {
                ty(*ops.get(*i as usize).ok_or("a result operand is missing")?)?
            }
            (Some(Val::Str(r)), Some(Val::Int(i)), _) if r == "scalar_of" => NodeType {
                dtype: ty(*ops.get(*i as usize).ok_or("a result operand is missing")?)?.dtype,
                shape: vec![],
                tag: None,
            },
            (Some(Val::Str(r)), Some(Val::Str(dt)), Some(Val::Tuple(sh))) if r == "fixed" => {
                NodeType {
                    dtype: dt.clone(),
                    shape: sh
                        .iter()
                        .filter_map(|x| {
                            if let Val::Int(i) = x {
                                Some(*i as i64)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    tag: None,
                }
            }
            _ if d.family == "sink" && !ops.is_empty() => ty(ops[0])?,
            _ => {
                return Err(format!(
                    "kind {}: carries its own type, got None",
                    py_str(&d.name)
                ))
            }
        },
    };
    Ok(typed)
}

/// The checks of a node of a declared kind: arity, operand families, the result rule, field types and a sink's name and role.
pub fn kind_check(ctx: &Ctx, node: &Node) -> Result<(), String> {
    let c = &ctx.table.classes[node.class];
    let Some(d) = &c.decl else { return Ok(()) };
    let why = |m: String| Err(format!("kind {}: {m}", py_str(&d.name)));
    let n = node.ops.len();
    if n < d.ops.0 || d.ops.1.is_some_and(|h| n > h) {
        return why(format!(
            "takes {}..{} operand(s), got {n}",
            d.ops.0,
            d.ops.1.map_or("many".into(), |h| h.to_string())
        ));
    }
    for (i, fam) in d.ops.2.iter().take(n).enumerate() {
        if fam == "plane-leaf" && !ctx.is_a(node.ops[i], "Leaf") {
            return why(format!("operand {i} must be a plane leaf"));
        }
    }
    let t = |h: u32| {
        let o = &ctx.nodes[h as usize];
        (o.dtype.clone(), o.shape.clone(), o.tag.clone())
    };
    let want = match (d.result.first(), d.result.get(1), d.result.get(2)) {
        (Some(Val::Str(r)), Some(Val::Int(i)), _) if r == "same_as" => {
            Some(t(node.ops[*i as usize]))
        }
        (Some(Val::Str(r)), Some(Val::Int(i)), _) if r == "scalar_of" => Some((
            ctx.nodes[node.ops[*i as usize] as usize].dtype.clone(),
            vec![],
            None,
        )),
        (Some(Val::Str(r)), Some(Val::Str(dt)), Some(Val::Tuple(sh))) if r == "fixed" => Some((
            dt.clone(),
            sh.iter()
                .map(|x| if let Val::Int(i) = x { *i as i64 } else { 0 })
                .collect(),
            None,
        )),
        _ => None,
    };
    if let Some(w) = want {
        if w != (node.dtype.clone(), node.shape.clone(), node.tag.clone()) {
            return why(format!(
                "type {} {:?} is not what its result rule derives, {} {:?}",
                node.dtype, node.shape, w.0, w.1
            ));
        }
    }
    for k in c.chain.iter().rev() {
        for (fname, ftype, _, _) in ctx.table.classes[*k]
            .decl
            .iter()
            .flat_map(|d| d.fields.iter())
        {
            let i = c.slots.iter().position(|s| s == fname).unwrap();
            if !has_type(&node.fields[i], ftype) {
                return why(format!(
                    "field {} must be a {ftype}, got {}",
                    py_str(fname),
                    crate::value::text_val(&node.fields[i])
                ));
            }
        }
    }
    if d.family == "sink" {
        let named = |f: &str| c.slots.iter().position(|s| s == f).map(|i| &node.fields[i]);
        if !matches!(named("name"), Some(Val::Str(_)))
            || matches!(named("role"), None | Some(Val::None))
        {
            return why(
                "a sink needs `name` (and `role` unless the refined core kind fixes it)".into(),
            );
        }
        sink_role(named("name"), named("role"))?;
    }
    Ok(())
}

/// The role of a sink must be one the manifest lists as a sink role.
fn sink_role(name: Option<&Val>, role: Option<&Val>) -> Result<(), String> {
    let (Some(Val::Str(name)), Some(Val::Str(role))) = (name, role) else {
        return Ok(());
    };
    let listed = crate::consts::constant("SINK_ROLES").unwrap_or_default();
    let roles: Vec<&str> = listed
        .trim_matches(['[', ']'])
        .split(',')
        .map(|r| r.trim_matches('"'))
        .collect();
    if roles.contains(&role.as_str()) {
        return Ok(());
    }
    let shown: Vec<String> = roles.iter().map(|r| py_str(r)).collect();
    let shown = shown.join(", ");
    let (name, role) = (py_str(name), py_str(role));
    Err(format!(
        "sink {name}: unknown sink role {role}, expected one of ({shown})"
    ))
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_declare_dialect(
    ctx: *mut Ctx,
    manifest_json: *const c_char,
    n: usize,
    first_party_kt: *const c_char,
) -> c_int {
    guard(ctx, |c| {
        let m = std::str::from_utf8(slice(manifest_json as *const u8, n, "manifest_json")?)
            .map_err(|_| Fault::BadArg("manifest_json is not UTF-8".into()))?;
        let fp = if first_party_kt.is_null() {
            "()"
        } else {
            cstr(first_party_kt, "first_party_kt")?
        };
        if m.is_empty() {
            return badarg("manifest_json is empty");
        }
        c.declare_dialect(m, fp)
    })
}
