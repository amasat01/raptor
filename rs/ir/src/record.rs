//! The node record and the arena. One record per make call: equal nodes are never merged, a handle is an object.
use crate::class::Table;
use crate::value::{fields_of, text_val, Val};
use rcore::{Dedup, Item, Rec, Rule};
use std::ffi::CString;

/// One node: its class, type, operands and field values.
pub struct Node {
    /// The index of the node's class.
    pub class: usize,
    /// The element type name.
    pub dtype: String,
    /// The extents.
    pub shape: Vec<i64>,
    /// The tag, if any.
    pub tag: Option<String>,
    /// the operands, in order
    pub ops: Vec<u32>,
    /// every slot beyond the type and the operands, in the class's declared order
    pub fields: Vec<Val>,
    /// The operand rule of the node's kind, when it has one of its own.
    pub rule: Option<Rule>,
}

/// Why a call failed: a refusal of the request, an argument that is not a valid value of its C type, or a caller buffer too small.
#[derive(Debug)]
pub enum Fault {
    /// The request was refused, with the reason.
    Refused(String),
    /// An argument was not a valid value, with the reason.
    BadArg(String),
    /// The caller's buffer was too small.
    Truncated,
    /// The question is well formed and the answer is that there is none (a field holding nothing).
    Absent,
}
/// A refusal with the message `m`.
pub fn refuse<T>(m: impl Into<String>) -> Result<T, Fault> {
    Err(Fault::Refused(m.into()))
}
/// A bad-argument fault with the message `m`.
pub fn badarg<T>(m: impl Into<String>) -> Result<T, Fault> {
    Err(Fault::BadArg(m.into()))
}

/// Names the role of a slot as a digest recorded before the planes shared roles took it, from the role, the name, the rank of the
/// value and whether the runtime zeroes the plane.
#[cfg(feature = "replay-key-form")]
pub type RoleForm = fn(&str, &str, usize, bool) -> String;

/// Everything a context owns: the class table, the arena, the last error text.
pub struct Ctx {
    /// The class table.
    pub table: Table,
    /// The nodes, in creation order; a handle is a position.
    pub nodes: Vec<Node>,
    /// The text of the last refusal.
    pub err: CString,
    /// The code of the last refusal when it has one, else empty. A code is a promise to callers, who may match on it; the text is not.
    pub code: CString,
    /// The walks computed so far.
    pub walks: Vec<crate::canon::WalkRes>,
    /// The structural digest of each loop's body, by loop handle. It is correct to keep: records are only appended, so
    /// a loop's body never changes under its handle; the one place that rewrites a loop record drops its entry (build.rs).
    pub body_cache: std::cell::RefCell<std::collections::HashMap<u32, String>>,
    /// The compiled constructor programs of the manifest.
    pub progs: crate::ctor::Progs,
    /// The manifest's rule tables and refinable kinds, read when the context is made.
    pub tables: std::sync::Arc<crate::rules::ManifestTables>,
    /// Rewrites the key line of each node before a walk digest or a loop body hash takes it in. Only the replay tools set it,
    /// to compare a walk with a digest recorded before a field was added to a node: the rewrite removes that field.
    #[cfg(feature = "replay-key-form")]
    pub key_form: Option<fn(&str) -> String>,
    /// Names the role of a slot the way a digest recorded before the read planes shared one role took it, from the role, the name
    /// and the rank of the value. Only the replay tools set it, together with `key_form`.
    #[cfg(feature = "replay-key-form")]
    pub role_form: Option<RoleForm>,
}

impl Ctx {
    /// A new context with the core classes and constructor programs.
    pub fn new() -> Result<Ctx, String> {
        Ok(Ctx {
            table: Table::core()?,
            nodes: vec![],
            err: CString::default(),
            code: CString::default(),
            walks: vec![],
            body_cache: Default::default(),
            progs: crate::ctor::Progs::compile()?,
            tables: std::sync::Arc::new(crate::rules::ManifestTables::load()?),
            #[cfg(feature = "replay-key-form")]
            key_form: None,
            #[cfg(feature = "replay-key-form")]
            role_form: None,
        })
    }
    /// A node key as the digests take it: rewritten by `key_form` when a replay tool set one. The rewrite reads a numbered line
    /// (`<number>|<key>`); a key without a number is numbered for it and unnumbered after.
    #[cfg(feature = "replay-key-form")]
    pub(crate) fn keyed(&self, key: &str) -> String {
        match self.key_form {
            None => key.to_string(),
            Some(form) => form(&format!("0|{key}"))
                .split_once('|')
                .map_or(key.to_string(), |(_, rest)| rest.to_string()),
        }
    }
    /// The role of a slot of a value of rank `rank`, which the runtime zeroes when `zeroed`, as the digests take it: rewritten by
    /// `role_form` when a replay tool set one.
    #[cfg(feature = "replay-key-form")]
    pub(crate) fn digest_role(&self, role: &str, name: &str, rank: usize, zeroed: bool) -> String {
        self.role_form
            .map_or_else(|| role.to_string(), |form| form(role, name, rank, zeroed))
    }
    /// The role of a slot as the digests take it: as it is.
    #[cfg(not(feature = "replay-key-form"))]
    pub(crate) fn digest_role(
        &self,
        role: &str,
        _name: &str,
        _rank: usize,
        _zeroed: bool,
    ) -> String {
        role.to_string()
    }
    /// A numbered key line as the digests take it.
    #[cfg(feature = "replay-key-form")]
    pub(crate) fn keyed_line(&self, line: &str) -> String {
        self.key_form.map_or(line.to_string(), |form| form(line))
    }
    /// Stores `code` as the code of the refusal about to be returned; it holds until the next call through the door begins.
    pub fn set_code(&mut self, code: &str) {
        self.code = CString::new(code).unwrap_or_default()
    }
    /// Stores `m` as the last refusal text.
    pub fn set_err(&mut self, m: &str) {
        self.err = CString::new(m.replace('\0', " ")).unwrap_or_default()
    }

    /// Whether node `h` is of the class `class` or of a class that refines it; `h` must be a handle of this context.
    pub(crate) fn is_a(&self, h: u32, class: &str) -> bool {
        self.table.is_a(self.nodes[h as usize].class, class)
    }

    /// The node with handle `h`; refuses a handle the context does not hold.
    pub fn node(&self, h: u32) -> Result<&Node, Fault> {
        self.nodes.get(h as usize).map_or_else(
            || refuse(format!("node handle {h} is not in this context")),
            Ok,
        )
    }

    /// Appends one record. Refuses an unknown class, a handle out of range, an unknown or missing field and unparsable value text.
    pub fn make(
        &mut self,
        cls: &str,
        dtype: &str,
        shape: &[i64],
        tag: Option<&str>,
        ops: &[u32],
        fields: &str,
    ) -> Result<u32, Fault> {
        let Some(ci) = self.table.find(cls) else {
            return refuse(format!("unknown class {cls:?}"));
        };
        if !rcore::facts::DTYPES.contains(&dtype) {
            return badarg(format!("unknown dtype {dtype:?}"));
        }
        if shape.iter().any(|&d| d < 0) {
            return badarg(format!("negative extent in shape {shape:?}"));
        }
        if let Some(&h) = ops.iter().find(|&&h| h as usize >= self.nodes.len()) {
            return refuse(format!("operand handle {h} is not in this context"));
        }
        let given = fields_of(fields).map_err(Fault::Refused)?;
        let class = &self.table.classes[ci];
        let mut vals: Vec<Option<Val>> = vec![None; class.slots.len()];
        for (k, v) in given {
            let Some(i) = class.slots.iter().position(|s| *s == k) else {
                return refuse(format!("{cls} has no field {k:?}"));
            };
            if vals[i].is_some() {
                return refuse(format!("{cls}: field {k:?} is given twice"));
            }
            if let Some(h) = first_ref(&v, self.nodes.len()) {
                return refuse(format!("{cls}.{k}: node handle {h} is not in this context"));
            }
            vals[i] = Some(v);
        }
        for (k, d) in &class.defaults {
            let i = class.slots.iter().position(|s| s == k).unwrap();
            if vals[i].is_none() {
                vals[i] = Some(d.clone())
            }
        }
        for (i, slot) in class.slots.iter().enumerate() {
            if vals[i].is_none() {
                vals[i] = crate::ctor::slot_default(self, ci, slot);
            }
        }
        for (k, f) in &class.fixed {
            let i = class.slots.iter().position(|s| s == k).unwrap();
            match &vals[i] {
                Some(v) if v != f && !k.starts_with('_') => {
                    return refuse(format!("{cls}: field {k:?} is fixed to {}", text_val(f)))
                }
                _ => vals[i] = Some(f.clone()), // a slot named with a leading underscore is a cache: whatever was given is dropped
            }
        }
        let mut fields = Vec::with_capacity(vals.len());
        for (i, v) in vals.into_iter().enumerate() {
            match v {
                Some(v) => fields.push(v),
                None => {
                    return refuse(match &class.decl {
                        Some(d) => {
                            format!("kind {}: field {:?} is required", d.name, class.slots[i])
                        }
                        None => format!("{cls}: missing field {:?}", class.slots[i]),
                    })
                }
            }
        }
        let node = Node {
            class: ci,
            dtype: dtype.into(),
            shape: shape.to_vec(),
            tag: tag.map(String::from),
            ops: ops.to_vec(),
            fields,
            rule: Some(class.rule.clone()),
        };
        crate::dialect::kind_check(self, &node).map_err(Fault::Refused)?;
        self.nodes.push(node);
        let h = self.nodes.len() as u32 - 1;
        // a declared check that refuses has the last word; when the program cannot run or disagrees with the record, the rule of the
        // kind (operand count, plane leaves, result type) says what is wrong
        let checked = match crate::ctor::check(self, &self.nodes[h as usize]) {
            Err(f) if f.declared => Err(f.text),
            Err(f) => Err(self.rule_check(h).err().unwrap_or(f.text)),
            Ok(()) => self.rule_check(h),
        };
        if let Err(m) = checked {
            self.nodes.pop();
            return refuse(m);
        }
        Ok(h)
    }

    /// Builds a node from constructor parameters (a tuple of (name, value) pairs); `#i` in a value is `refs[i]`.
    pub fn new_node(&mut self, cls: &str, params: &str, refs: &[u32]) -> Result<u32, Fault> {
        let Some(ci) = self.table.find(cls) else {
            return refuse(format!("unknown class {cls:?}"));
        };
        fn sub(v: Val, refs: &[u32]) -> Result<Val, Fault> {
            Ok(match v {
                Val::Node(i) => Val::Node(*refs.get(i as usize).ok_or_else(|| {
                    Fault::Refused(format!(
                        "node reference #{i} is outside the {} reference(s) given",
                        refs.len()
                    ))
                })?),
                Val::Tuple(t) => Val::Tuple(
                    t.into_iter()
                        .map(|x| sub(x, refs))
                        .collect::<Result<_, _>>()?,
                ),
                Val::List(t) => Val::List(
                    t.into_iter()
                        .map(|x| sub(x, refs))
                        .collect::<Result<_, _>>()?,
                ),
                Val::Rec(n, f) => Val::Rec(
                    n,
                    f.into_iter()
                        .map(|(k, x)| Ok((k, sub(x, refs)?)))
                        .collect::<Result<_, Fault>>()?,
                ),
                x => x,
            })
        }
        let given: Vec<(String, Val)> = fields_of(params)
            .map_err(Fault::Refused)?
            .into_iter()
            .map(|(k, v)| Ok((k, sub(v, refs)?)))
            .collect::<Result<_, Fault>>()?;
        self.new_from(ci, &given)
    }

    /// Builds a node of the class at index `ci` from parameter values (a node value is a
    /// handle): the declared program runs forward.
    pub fn new_from(&mut self, ci: usize, given: &[(String, Val)]) -> Result<u32, Fault> {
        let cls = self.table.classes[ci].name.clone();
        let cls = cls.as_str();
        let new = if self.table.classes[ci].decl.is_some() {
            crate::dialect::forward(self, ci, given)
        } else {
            crate::ctor::forward(self, ci, given)
        }
        .map_err(Fault::Refused)?;
        let fields = text_val(&Val::Tuple(
            new.fields
                .into_iter()
                .map(|(k, v)| Val::Tuple(vec![Val::Str(k), v]))
                .collect(),
        ));
        self.make(
            cls,
            &new.dtype,
            &new.shape,
            new.tag.as_deref(),
            &new.ops,
            &fields,
        )
    }

    /// The refusal text of a node that breaks the operand-count or result-type rule of its class, if it does.
    fn rule_check(&self, h: u32) -> Result<(), String> {
        let n = &self.nodes[h as usize];
        let Some(rule) = &n.rule else { return Ok(()) };
        let mut r = self.rec_head(h).map_err(|e| format!("{e:?}"))?;
        let recs: Vec<Rec> = n
            .ops
            .iter()
            .map(|&o| {
                let x = &self.nodes[o as usize];
                Rec {
                    leaf: self.table.is_a(x.class, "Leaf"),
                    dtype: x.dtype.clone(),
                    shape: x.shape.clone(),
                    tag: x.tag.clone(),
                    ..Default::default()
                }
            })
            .collect();
        r.ops = (0..recs.len() as u32).collect();
        match rcore::rule_fault(&r, rule, &recs) {
            Some(m) => Err(format!("{}: {m}", r.kind)),
            None => Ok(()),
        }
    }

    /// The head of the key record of node `h`: its class, type and kind.
    fn rec_head(&self, h: u32) -> Result<Rec, Fault> {
        let n = self.node(h)?;
        Ok(Rec {
            kind: self.kind_of(h)?,
            dtype: n.dtype.clone(),
            shape: n.shape.clone(),
            tag: n.tag.clone(),
            ..Default::default()
        })
    }

    /// The value of a slot, or of the type parts `kind`, `dtype`, `shape`, `tag`.
    pub fn field(&self, h: u32, name: &str) -> Result<Val, Fault> {
        let n = self.node(h)?;
        let c = &self.table.classes[n.class];
        if let Some(i) = c.slots.iter().position(|s| s == name) {
            return Ok(n.fields[i].clone());
        }
        match name {
            "feathers" => Ok(Val::Tuple(
                c.chain
                    .iter()
                    .filter_map(|&k| self.table.classes[k].decl.as_ref())
                    .flat_map(|d| d.feathers.iter())
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .map(|f| Val::Str(f.clone()))
                    .collect(),
            )),
            "kind" => c
                .kind
                .clone()
                .map(Val::Str)
                .map_or_else(|| refuse(format!("{} has no field \"kind\"", c.name)), Ok),
            "dtype" => Ok(Val::Str(n.dtype.clone())),
            "shape" => Ok(Val::Tuple(
                n.shape.iter().map(|&d| Val::Int(d as i128)).collect(),
            )),
            "tag" => Ok(n.tag.clone().map_or(Val::None, Val::Str)),
            "phases" | "phase_count" | "scratch_decls" | "combines" | "combine_phases"
                if self.table.is_a(n.class, "Group") =>
            {
                self.group_fact(h, name)
            }
            "native" if self.table.is_a(n.class, "Call") => self.call_is_native(h).map(Val::Bool),
            "callee" if self.table.is_a(n.class, "Call") => {
                self.called_function(h).map(Val::Node).ok_or_else(|| {
                    Fault::Refused("a call of a function that the table does not hold".into())
                })
            }
            "digest" if self.table.is_a(n.class, "Function") => {
                self.function_digest(h).map(Val::Str)
            }
            _ => self.prop(h, name),
        }
    }

    /// A property a class declares beyond its slots (core.json `props`): an operand, a slice of operands, an attribute of an operand.
    fn prop(&self, h: u32, name: &str) -> Result<Val, Fault> {
        let n = self.node(h)?;
        let c = &self.table.classes[n.class];
        let Some(p) = c
            .chain
            .iter()
            .find_map(|&k| self.table.classes[k].props.iter().find(|p| p[0] == name))
        else {
            return refuse(format!("{} has no field {name:?}", c.name));
        };
        let op = |i: usize| {
            n.ops
                .get(i)
                .copied()
                .map_or_else(|| refuse(format!("{}.{name}: no operand {i}", c.name)), Ok)
        };
        let idx = |s: &str| {
            s.parse::<usize>().map_err(|_| {
                Fault::Refused(format!("not in this build: property {name} of {}", c.name))
            })
        };
        match p[1].as_str() {
            "op" => Ok(Val::Node(op(idx(&p[2])?)?)),
            "opq" => Ok(n.ops.get(idx(&p[2])?).map_or(Val::None, |&o| Val::Node(o))),
            "ops" => Ok(Val::Tuple(
                n.ops
                    .iter()
                    .skip(idx(p[2].trim_end_matches(':'))?)
                    .map(|&o| Val::Node(o))
                    .collect(),
            )),
            "opattr" => self.field(op(idx(&p[2])?)?, &p[3]),
            "nops-1" => Ok(Val::Int(n.ops.len() as i128 - 1)),
            "none" => Ok(Val::None),
            "intlit" => self.field(h, "literal"),
            "slice" => {
                let end = |t: &str| -> Result<usize, Fault> {
                    match t.strip_prefix("self.") {
                        Some(f) => match self.field(h, f)? {
                            Val::Int(i) => Ok(i as usize),
                            _ => refuse("a slice bound is an int"),
                        },
                        None => idx(t),
                    }
                };
                let (a, b) = (end(&p[2])?, end(&p[3])?);
                Ok(Val::Tuple(
                    n.ops
                        .get(a..b.min(n.ops.len()))
                        .unwrap_or(&[])
                        .iter()
                        .map(|&o| Val::Node(o))
                        .collect(),
                ))
            }
            "core" => self.exit_form(h),
            _ => refuse(format!("not in this build: property {name} of {}", c.name)),
        }
    }

    /// Whether `h` is an update that commits through a companion plane: an add under the compensated summation.
    pub fn compensates(&self, h: u32) -> Result<bool, Fault> {
        Ok(self.is_a(h, "Update")
            && self.str_field(h, "op")? == "add"
            && self.str_field(h, "summation")? == "compensated")
    }

    /// The text a field holds; refuses a field of another kind.
    pub fn str_field(&self, h: u32, name: &str) -> Result<String, Fault> {
        match self.field(h, name)? {
            Val::Str(s) => Ok(s),
            _ => refuse(format!("field {name:?} is not a string")),
        }
    }

    /// The walk's record of one node: kind, role, name and the dedup text derived from its fields.
    pub fn rec_of(&self, h: u32) -> Result<Rec, Fault> {
        let n = self.node(h)?;
        let leaf = self.table.is_a(n.class, "Leaf");
        let mut r = Rec {
            rule: n.rule.clone(),
            leaf,
            kind: self.kind_of(h)?,
            dtype: n.dtype.clone(),
            shape: n.shape.clone(),
            tag: n.tag.clone(),
            ops: n.ops.clone(),
            ..Default::default()
        };
        if leaf {
            r.role = self.str_field(h, "role")?;
            r.name = self.str_field(h, "name")?;
        } else {
            r.extra = text_val(&self.dedup_val(h)?);
        }
        Ok(r)
    }

    /// The value a node writes as its dedup identity, by the recipe of its class.
    pub fn dedup_val(&self, h: u32) -> Result<Val, Fault> {
        let n = self.node(h)?;
        let c = &self.table.classes[n.class];
        let Some(spec) = c.spec else {
            return refuse("no dedup recipe");
        };
        let identity = c.decl.as_ref().and_then(|d| d.identity.as_ref());
        let v = if let Some(id) = identity {
            Val::Tuple(
                id.iter()
                    .map(|f| self.field(h, f))
                    .collect::<Result<_, _>>()?,
            )
        } else {
            match &spec.dedup {
                Dedup::None => Val::None,
                Dedup::Py if c.state_fields.is_empty() && c.name == "Node" => Val::None, // the state of a bare node is empty
                Dedup::Py if c.decl.is_some() || !c.state_fields.is_empty() => {
                    // the state of a declared kind or of a core kind that lists its state: its fields, by name
                    let mut st: Vec<(&String, Val)> = c
                        .state_fields
                        .iter()
                        .map(|f| Ok((f, self.field(h, f)?)))
                        .collect::<Result<_, Fault>>()?;
                    st.sort_by(|a, b| a.0.cmp(b.0));
                    if st.is_empty() {
                        Val::None
                    } else {
                        Val::Str(text_val(&Val::Tuple(
                            st.into_iter()
                                .map(|(k, v)| Val::Tuple(vec![Val::Str(k.clone()), v]))
                                .collect(),
                        )))
                    }
                }
                Dedup::Py => {
                    return refuse(format!("not in this build: the state text of a {}", c.name))
                }
                Dedup::One(it) => self.item(h, it)?.unwrap(),
                Dedup::Tup(its) => {
                    let mut v = vec![];
                    for it in its.iter() {
                        if let Some(x) = self.item(h, it)? {
                            v.push(x)
                        }
                    }
                    Val::Tuple(v)
                }
            }
        };
        Ok(v)
    }

    /// The value of one item of a record's key: a slot, a field or a type part.
    fn item(&self, h: u32, it: &Item) -> Result<Option<Val>, Fault> {
        Ok(Some(match it {
            Item::Field(f) => self.field(h, f)?,
            Item::ReprOf(f) => Val::Str(text_val(&self.field(h, f)?)),
            Item::Lit(l) => Val::Str(l.to_string()),
            Item::K => Val::Int(self.node(h)?.ops.len() as i128 - 1),
            Item::Names(f) => Val::Tuple(
                self.refs(h, f)?
                    .into_iter()
                    .map(|x| self.field(x, "name"))
                    .collect::<Result<_, _>>()?,
            ),
            Item::SinkKeys(f) => Val::Tuple(
                self.refs(h, f)?
                    .into_iter()
                    .map(|x| {
                        Ok(Val::Tuple(vec![
                            Val::Str(self.kind_of(x)?),
                            self.field(x, "role")?,
                            self.field(x, "name")?,
                        ]))
                    })
                    .collect::<Result<_, Fault>>()?,
            ),
            Item::Body => Val::Str(self.body_digest(h)?),
            Item::LitIfSet(l, f) => {
                if self.field(h, f)? == Val::None {
                    return Ok(None);
                }
                Val::Str(l.to_string())
            }
            Item::FieldIfSet(f, g) => {
                if self.field(h, g)? == Val::None {
                    return Ok(None);
                }
                self.field(h, f)?
            }
            Item::FieldUnless(f, set) => {
                let v = self.field(h, f)?;
                if matches!(&v, Val::Str(s) if set.contains(&s.as_str())) {
                    return Ok(None);
                }
                v
            }
        }))
    }

    /// The kind string of the walk's key line: the `kind` slot, else the class's own.
    pub(crate) fn kind_of(&self, h: u32) -> Result<String, Fault> {
        match self.field(h, "kind")? {
            Val::Str(s) => Ok(s),
            _ => refuse("field \"kind\" is not a string"),
        }
    }

    /// The node handles held by a field (a reference or a sequence of them).
    pub(crate) fn refs(&self, h: u32, f: &str) -> Result<Vec<u32>, Fault> {
        match self.field(h, f)? {
            Val::Tuple(v) | Val::List(v) => v
                .iter()
                .map(|x| {
                    if let Val::Node(n) = x {
                        Ok(*n)
                    } else {
                        refuse(format!("field {f:?} holds a non-node"))
                    }
                })
                .collect(),
            Val::Node(n) => Ok(vec![n]),
            _ => refuse(format!("field {f:?} holds no nodes")),
        }
    }
}

/// The first node reference at or past `len` in a value, if any.
fn first_ref(v: &Val, len: usize) -> Option<u32> {
    match v {
        Val::Node(h) if *h as usize >= len => Some(*h),
        Val::Tuple(t) | Val::List(t) => t.iter().find_map(|x| first_ref(x, len)),
        Val::Rec(_, f) => f.iter().find_map(|(_, x)| first_ref(x, len)),
        _ => None,
    }
}
