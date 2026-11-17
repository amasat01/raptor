#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! raptor's dependency-free core: the canonical form (key text), the digest, the manifest shape validation and the specs generated
//! from raptor/ir/core.json by build.rs (the classes, their dedup recipes, the operand and result rules, the constants and `facts`).
pub mod keytext;
pub mod nonprintable;
pub mod sha256;
use std::collections::HashMap;

/// The manifest of raptor's core kinds (raptor/ir/core.json), compiled in; the generated specs below come from it by build.rs.
pub const MANIFEST: &str = include_str!("../../../raptor/ir/core.json");
pub mod json;
pub mod manifest;
pub mod pjson;
/// One piece of a class's dedup recipe. A piece may yield nothing, and then it is left out of the identity.
pub enum Item {
    /// The value of a field.
    Field(&'static str),
    /// The key text of a field's value, as a string.
    ReprOf(&'static str),
    /// A fixed text.
    Lit(&'static str),
    /// The operand count minus one.
    K,
    /// A field's value, left out when it is a string in the given list.
    FieldUnless(&'static str, &'static [&'static str]),
    /// a tuple of the `name` of each element of a field
    Names(&'static str),
    /// a tuple of (kind, role, name) of each sink of a field
    SinkKeys(&'static str),
    /// the structural digest of a loop body
    Body,
    /// a literal, only when a field is set (not None)
    LitIfSet(&'static str, &'static str),
    /// a field's value, only when a gate field is set
    FieldIfSet(&'static str, &'static str),
}
/// How a class forms the dedup identity of its nodes.
pub enum Dedup {
    /// No identity beyond the kind and operands.
    None,
    /// The default: the state of the class, taken from its fields.
    Py,
    /// The identity is one item.
    One(Item),
    /// The identity is the tuple of these items.
    Tup(&'static [Item]),
}
/// One node class of the core manifest.
pub struct ClassSpec {
    /// The class name.
    pub name: &'static str,
    /// The class it refines, or empty.
    pub base: &'static str,
    /// The names of its own slots.
    pub slots: &'static [&'static str],
    /// Whether a node of the class binds a value in a slot.
    pub binds: bool,
    /// Its dedup recipe.
    pub dedup: Dedup,
    /// Which nodes the access and dispatch passes look at: 0 none, 1 all, 2 raw ones.
    pub watch: u8,
    /// The feathers the kind states itself; a refining kind inherits them through its class chain.
    pub feathers: &'static [&'static str],
    /// The slots a node of the class writes as its identity state when its dedup recipe is the generic one.
    pub state: &'static [&'static str],
}
include!(concat!(env!("OUT_DIR"), "/specs.rs"));

/// The operand-count and result-type rule of one kind (core.json rows and dialect manifests alike): operands min..max (max -1 = many),
/// result code 0 carried, 1 same_as(idx), 2 scalar_of(idx), 3 fixed(dtype, shape); `plane` bit i = operand i must be a plane leaf.
#[derive(Clone, Default)]
pub struct Rule {
    /// The fewest operands.
    pub min: u32,
    /// The most operands, or -1 for any number.
    pub max: i32,
    /// The result code: 0 carried, 1 same as an operand, 2 scalar of an operand, 3 fixed.
    pub res: u8,
    /// The operand that codes 1 and 2 refer to.
    pub idx: u32,
    /// The element type of the result: the fixed one when the result is `fixed`.
    pub dtype: String,
    /// The extents of the result when it is `fixed`.
    pub shape: Vec<i64>,
    /// Bit i set: operand i must be a plane leaf.
    pub plane: u32,
}
/// The rule of the core kind at row `i` of the generated table.
pub fn core_rule(i: usize) -> Rule {
    let r = &RULES[i];
    Rule {
        min: r.0,
        max: r.1,
        res: r.2,
        idx: r.3,
        dtype: r.4.to_string(),
        shape: r.5.to_vec(),
        plane: 0,
    }
}

#[derive(Clone, Default)]
/// One node as the walk reads it: its kind, type, operands and the text of its extra identity.
pub struct Rec {
    /// The rule of its kind, when it is checked against one.
    pub rule: Option<Rule>,
    /// Whether the node is a leaf (a named input with a role).
    pub leaf: bool,
    /// The kind name.
    pub kind: String,
    /// The element type of its result.
    pub dtype: String,
    /// The extents of its result; empty for a scalar.
    pub shape: Vec<i64>,
    /// The type tag of its result, when it has one.
    pub tag: Option<String>,
    /// The records of its operands, as indices into the record list.
    pub ops: Vec<u32>,
    /// The key text of its extra identity, already written.
    pub extra: String,
    /// For a leaf, its role.
    pub role: String,
    /// For a leaf, its name.
    pub name: String,
}

pub use keytext::str_repr as repr_str;

/// Appends integers as a key text tuple, `(1, 2)` or `(1,)`.
fn repr_ints(v: &[i64], out: &mut String) {
    out.push('(');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            out.push_str(", ")
        }
        out.push_str(&x.to_string());
    }
    if v.len() == 1 {
        out.push(',')
    }
    out.push(')');
}

/// Appends the key text of one record; operands are written as their positions in the walk order `pos`.
fn key_of(r: &Rec, pos: &[u32], out: &mut String) {
    if r.leaf {
        out.push_str("('leaf', ");
        repr_str(&r.role, out);
        out.push_str(", ");
        repr_str(&r.name, out);
        out.push(')');
        return;
    }
    out.push('(');
    repr_str(&r.kind, out);
    out.push_str(", ");
    repr_str(&r.dtype, out);
    out.push_str(", ");
    repr_ints(&r.shape, out);
    out.push_str(", ");
    match &r.tag {
        None => out.push_str("None"),
        Some(t) => repr_str(t, out),
    }
    out.push_str(", ");
    let p: Vec<i64> = r.ops.iter().map(|&c| pos[c as usize] as i64).collect();
    repr_ints(&p, out);
    out.push_str(", ");
    out.push_str(&r.extra);
    out.push(')');
}

/// What a walk returns: the records in canonical order, the text of the node lines (the digest's input), the position each record
/// took and the post-order of visits.
pub struct WalkOut {
    /// The records in canonical order.
    pub order: Vec<u32>,
    /// The text of the node lines, in order, that the digest is taken over.
    pub pre: String,
    /// For each record, its position in the canonical order.
    pub pos: Vec<u32>,
    /// The records in the order the walk finished them.
    pub visits: Vec<u32>,
}

/// The walk of `roots` over `recs`: the records in canonical order and the text the digest is taken over.
pub fn walk(recs: &[Rec], roots: &[u32]) -> Result<(Vec<u32>, String), String> {
    let w = walk_full(recs, roots)?;
    Ok((w.order, w.pre))
}

/// The key text of a tuple of strings: `('a', 'b')`, or `('a',)` for one.
pub fn repr_tuple_str(v: &[&str]) -> String {
    let mut o = String::from("(");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            o.push_str(", ")
        }
        repr_str(x, &mut o)
    }
    if v.len() == 1 {
        o.push(',')
    }
    o.push(')');
    o
}

/// The walk with its working tables: post-order, one entry per distinct key text, and a cycle is refused.
pub fn walk_full(recs: &[Rec], roots: &[u32]) -> Result<WalkOut, String> {
    verify(recs)?;
    let mut live = vec![false; recs.len()];
    /// Marks a record the walk has not placed yet.
    const UNSET: u32 = u32::MAX;
    let mut pos = vec![UNSET; recs.len()];
    let mut by_key: HashMap<String, u32> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut pre = String::new();
    let mut visits: Vec<u32> = Vec::new();
    let mut stack: Vec<(u32, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();
    let mut key = String::new();
    while let Some((n, expanded)) = stack.pop() {
        if pos[n as usize] != UNSET {
            continue;
        }
        let r = &recs[n as usize];
        if !expanded {
            stack.push((n, true));
            live[n as usize] = true;
            for &c in r.ops.iter().rev() {
                if live[c as usize] {
                    return Err(format!(
                        "the graph has a cycle through a {} node",
                        recs[c as usize].kind
                    ));
                }
                if pos[c as usize] == UNSET {
                    stack.push((c, false))
                }
            }
            continue;
        }
        live[n as usize] = false;
        key.clear();
        key_of(r, &pos, &mut key);
        let p = match by_key.get(&key) {
            Some(&p) => p,
            None => {
                let p = order.len() as u32;
                order.push(n);
                pre.push_str(&p.to_string());
                pre.push('|');
                pre.push_str(&key);
                pre.push('\n');
                by_key.insert(key.clone(), p);
                p
            }
        };
        pos[n as usize] = p;
        visits.push(n);
    }
    Ok(WalkOut {
        order,
        pre,
        pos,
        visits,
    })
}

/// The hex digest of the walk text `pre` followed by `tail`, under the digest's version tag.
pub fn digest(pre: &str, tail: &[u8]) -> String {
    let mut b = Vec::with_capacity(pre.len() + tail.len() + 16);
    b.extend_from_slice(b"raptor.ir.walk/2\n");
    b.extend_from_slice(pre.as_bytes());
    b.extend_from_slice(tail);
    sha256::hex(&b)
}

/// The checks the Rust side enforces at the boundary, on the records of ONE crossing.
pub fn verify(recs: &[Rec]) -> Result<(), String> {
    for (i, r) in recs.iter().enumerate() {
        if r.kind.is_empty() {
            return Err(format!("node {i}: empty kind"));
        }
        if !facts::DTYPES.contains(&r.dtype.as_str()) {
            return Err(format!(
                "node {i} ({}): unknown dtype {:?}",
                r.kind, r.dtype
            ));
        }
        if r.shape.iter().any(|&d| d < 0) {
            return Err(format!(
                "node {i} ({}): negative extent in shape {:?}",
                r.kind, r.shape
            ));
        }
        if r.ops.iter().any(|&c| c as usize >= recs.len()) {
            return Err(format!("node {i}: operand out of range"));
        }
        if r.leaf && (!r.ops.is_empty() || r.role.is_empty() || r.name.is_empty()) {
            return Err(format!(
                "node {i}: a leaf has no operands and carries a role and a name"
            ));
        }
        if let Some(u) = &r.rule {
            check_rule(i, r, u, recs)?
        }
    }
    Ok(())
}

/// The operand count, the plane-leaf families and the result-type rule of one record (the declared rule of its kind).
fn check_rule(i: usize, r: &Rec, u: &Rule, recs: &[Rec]) -> Result<(), String> {
    match rule_fault(r, u, recs) {
        Some(m) => Err(format!("node {i} ({}): {m}", r.kind)),
        None => Ok(()),
    }
}

/// What a record breaks of its kind's rule, as text without the position prefix; `recs` holds its operands (`r.ops` index into it).
pub fn rule_fault(r: &Rec, u: &Rule, recs: &[Rec]) -> Option<String> {
    let n = r.ops.len();
    if (n as u32) < u.min || (u.max >= 0 && n as i32 > u.max) {
        let hi = if u.max < 0 {
            "many".to_string()
        } else {
            u.max.to_string()
        };
        return Some(format!("takes {}..{hi} operand(s), got {n}", u.min));
    }
    for (j, &c) in r.ops.iter().enumerate() {
        if j < 32 && u.plane >> j & 1 == 1 && !recs[c as usize].leaf {
            return Some(format!("operand {j} must be a plane leaf"));
        }
    }
    let same = |o: &Rec| o.dtype == r.dtype && o.shape == r.shape && o.tag == r.tag;
    let ok = match u.res {
        1 => same(&recs[r.ops[u.idx as usize] as usize]),
        2 => {
            let o = &recs[r.ops[u.idx as usize] as usize];
            r.shape.is_empty() && o.dtype == r.dtype
        }
        3 => r.dtype == u.dtype && r.shape == u.shape,
        _ => true,
    };
    if ok {
        return None;
    }
    let want = match u.res {
        1 => format!("the type of operand {}", u.idx),
        2 => format!("a scalar of operand {}'s dtype", u.idx),
        _ => format!("{} {:?}", u.dtype, u.shape),
    };
    Some(format!(
        "type {} {:?} is not what its result rule derives, {want}",
        r.dtype, r.shape
    ))
}
