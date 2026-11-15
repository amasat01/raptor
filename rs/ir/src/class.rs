//! The class table of one context: the core classes from rcore's generated specs, plus the edges, the fixed slots and the kind
//! string that core.json states per class. Dialect classes are appended by a declaration (dialect.rs).
use rcore::json::{self, J};
use rcore::keytext::Val;
use rcore::{ClassSpec, Rule, CLASSES};
use std::collections::HashMap;

/// One node class: its name, place in the hierarchy, slots, fixed values, operand rule and flags.
pub struct Class {
    /// The class name.
    pub name: String,
    /// the nearest base class (an index into the table)
    pub base: Option<usize>,
    /// this class and its ancestors, nearest first
    pub chain: Vec<usize>,
    /// every slot beyond the type and the operands: inherited ones first, in declared order
    pub slots: Vec<String>,
    /// the slots that hold node references
    pub edges: Vec<String>,
    /// slots whose value the class fixes, with that value
    pub fixed: Vec<(String, Val)>,
    /// the kind string of a class without a `kind` slot
    pub kind: Option<String>,
    /// declared properties (core.json `props`): name, form and its arguments; nearest class first is found through the chain
    pub props: Vec<Vec<String>>,
    /// the dedup recipe: the class's own, or of its nearest core ancestor
    pub spec: Option<&'static ClassSpec>,
    /// The operand-count and result-type rule.
    pub rule: Rule,
    /// whether a node of this class binds a slot, and whether the access and dispatch passes look at it
    pub binds: bool,
    /// Whether the access and dispatch passes watch nodes of this class.
    pub watch: bool,
    /// the declaration of a dialect class
    pub decl: Option<Decl>,
    /// the fields a dialect declares, inherited ones first (the state of a `py` recipe), and their defaults
    pub state_fields: Vec<String>,
    /// The default of each slot that has one.
    pub defaults: Vec<(String, Val)>,
}

/// A dialect kind's declaration, resolved: operands (min, max or None for many, families), fields (name, type, default, required),
/// the result rule, the identity fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Decl {
    /// The kind name.
    pub name: String,
    /// The extension that declared it.
    pub ext: String,
    /// The family: expr, sink or terminal.
    pub family: String,
    /// The kind it refines, if any.
    pub refines: Option<String>,
    /// The wire name the kind is spelled with.
    pub wire: String,
    /// The operand rule: minimum, maximum (None for many) and families.
    pub ops: (usize, Option<usize>, Vec<String>),
    /// The declared fields: name, type, default and whether it is required.
    pub fields: Vec<(String, String, Val, bool)>,
    /// The result rule words.
    pub result: Vec<Val>,
    /// The identity fields, if stated.
    pub identity: Option<Vec<String>>,
    /// the declared feathers, sorted; the spelling (dialect, name, args)
    pub feathers: Vec<String>,
    /// The spelling: dialect, name and argument names.
    pub spelling: Option<(String, String, Vec<String>)>,
}

/// The class table of a context: every class, found by name or by kind.
#[derive(Default)]
pub struct Table {
    /// The classes, in declaration order.
    pub classes: Vec<Class>,
    /// The index of each class by class name.
    by_name: HashMap<String, usize>,
    /// The index of each class by kind name.
    kinds: HashMap<String, usize>,
    /// The wire name of each kind.
    wires: HashMap<String, String>,
}

impl Table {
    /// The 26 core classes.
    pub fn core() -> Result<Table, String> {
        let m = json::parse(rcore::MANIFEST)?;
        let kinds = m.get("kinds").map(|k| k.arr()).unwrap_or(&[]);
        let mut t = Table::default();
        for (i, sp) in CLASSES.iter().enumerate() {
            let row = kinds
                .iter()
                .find(|k| k.get("class").and_then(J::str) == Some(sp.name))
                .ok_or("core.json has no row for a class")?;
            let base = if sp.base.is_empty() {
                None
            } else {
                Some(
                    *t.by_name
                        .get(sp.base)
                        .ok_or("a base class comes after its subclass")?,
                )
            };
            let mut chain = vec![i];
            let mut slots = vec![];
            if let Some(b) = base {
                chain.extend(&t.classes[b].chain);
                slots = t.classes[b].slots.clone()
            }
            slots.extend(
                sp.slots
                    .iter()
                    .filter(|s| **s != "ttype" && **s != "operands")
                    .map(|s| s.to_string()),
            );
            let mut fixed = base.map_or(vec![], |b| t.classes[b].fixed.clone());
            // a constructor's `out` lists the value of each assigned slot; a literal there is a fixed value
            let (assigns, out) = (
                row.get("assigns").map_or(&[][..], J::arr),
                row.get("ctor")
                    .and_then(|c| c.get("out"))
                    .map_or(&[][..], J::arr),
            );
            for (a, o) in assigns.iter().zip(out) {
                let (Some(a), Some(o)) = (a.str(), o.str()) else {
                    continue;
                };
                let v = if o == "None" {
                    Val::None
                } else if o.len() > 1 && o.starts_with('\'') && o.ends_with('\'') {
                    Val::Str(o[1..o.len() - 1].into())
                } else {
                    continue;
                };
                fixed.retain(|f| f.0 != a);
                if slots.iter().any(|s| s == a) {
                    fixed.push((a.into(), v))
                }
            }
            let own = row.get("wire_attr").and_then(J::str).map(String::from);
            let kind = own.or_else(|| base.and_then(|b| t.classes[b].kind.clone()));
            let edges = row.get("edges").map_or(vec![], |e| {
                e.strs()
                    .iter()
                    .map(|s| s.trim_end_matches('?').to_string())
                    .collect()
            });
            let props = row
                .get("props")
                .map_or(vec![], |p| p.arr().iter().map(|r| r.strs()).collect());
            if let Some(n) = row.get("name").and_then(J::str) {
                t.kinds.insert(n.into(), i);
            }
            // A refining kind inherits a wire name only from a core row that states one; otherwise it keeps its own name.
            let wire = row.get("wire").and_then(J::str);
            if let Some(w) = wire {
                t.wires.insert(sp.name.into(), w.into());
            }
            t.by_name.insert(sp.name.to_string(), i);
            t.classes.push(Class {
                name: sp.name.into(),
                base,
                chain,
                slots,
                edges,
                fixed,
                kind,
                props,
                spec: Some(sp),
                rule: rcore::core_rule(i),
                binds: sp.binds,
                watch: sp.watch == 1 || sp.watch == 2,
                decl: None,
                state_fields: vec![],
                defaults: vec![],
            });
        }
        Ok(t)
    }
    /// The index of the class named `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }
    /// Whether `class` is `ancestor` or refines it.
    pub fn is_a(&self, class: usize, ancestor: &str) -> bool {
        self.classes[class]
            .chain
            .iter()
            .any(|&c| self.classes[c].name == ancestor)
    }
    /// The class a declaration's `refines` names: a core kind or a dialect kind.
    pub fn kind(&self, name: &str) -> Option<usize> {
        self.kinds.get(name).copied()
    }
    /// The wire a core class states, if any.
    pub fn wire_of(&self, class: &str) -> Option<&str> {
        self.wires.get(class).map(String::as_str)
    }
    /// Adds the class `c` for kind `kind` and returns its index.
    pub fn push(&mut self, kind: &str, c: Class) -> usize {
        let i = self.classes.len();
        self.by_name.insert(c.name.clone(), i);
        self.kinds.insert(kind.into(), i);
        self.classes.push(c);
        i
    }
}
