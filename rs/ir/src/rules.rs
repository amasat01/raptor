//! Derivative factors as templates (core.json `rules`), instantiated over the nodes of a context. A template is a name (a bound
//! item), `["c", x]` (an f64 scalar constant), `["let", name, e, body]`, `["zero_like", e]`, `["sel", c, a, b]`,
//! `["ramp", c, like]`, `["entrywise", kind, e...]`, `["abs_d", x]`, or `[kind, e...]` (an op minted by `make_op`); the other
//! forms are the ones `form` dispatches on. Owns the C doors `rx_rule_factor`, `rx_rule_vjp`, `rx_rule_jvp`, `rx_reduce_vjp`,
//! `rx_reduce_jvp`, `rx_rule_kinds`, `rx_rule_modes`, `rx_entrywise` and `rx_zero_of`.
use crate::consts::put_names;
use crate::door::{cstr, guard, slice};
use crate::loops::give;
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::typing::{is_integer_dtype, is_real_dtype};
use crate::value::{ttype_val, Val as Lit};
use rcore::json::{self, J};
use std::ffi::{c_char, c_int};
use std::sync::Arc;

/// A result that refuses with a fault.
type Res<T> = Result<T, Fault>;

/// The manifest data the rule interpreter and the template checks read, parsed once when a context is made.
pub struct ManifestTables {
    /// The `rules` section: the rule tables, the defs and the modes.
    pub rules: J,
    /// The names of the core kinds a kind declaration may refine (the `kinds` rows marked `refinable`).
    pub refinable: Vec<String>,
}

impl ManifestTables {
    /// Reads the tables from the manifest the crate embeds.
    pub fn load() -> Result<Self, String> {
        let manifest = json::parse_without(rcore::MANIFEST, &["consts", "records", "checks"])?;
        let rows = manifest.get("kinds").map_or(&[][..], J::arr);
        let marked = rows
            .iter()
            .filter(|r| matches!(r.get("refinable"), Some(J::Bool(true))));
        Ok(ManifestTables {
            rules: manifest.get("rules").cloned().unwrap_or(J::Null),
            refinable: marked
                .filter_map(|r| r.get("name").and_then(J::str).map(String::from))
                .collect(),
        })
    }

    /// The template of `table`/`kind`, or the refusal that names them.
    fn template(&self, table: &str, kind: &str) -> Res<&J> {
        let found = self.rules.get(table).and_then(|t| t.get(kind));
        found.map_or_else(|| bad(&format!("no {table} template for {kind}")), Ok)
    }
}

/// What a template evaluates to.
#[derive(Clone)]
enum Item {
    /// A node of the context.
    Node(u32),
    /// A literal of an op or a node (a shape, an entry index, a field value).
    Lit(Lit),
    /// A list of items.
    List(Vec<Item>),
    /// A record of named items.
    Rec(Vec<(String, Item)>),
    /// An integer used to count and index.
    Int(i64),
    /// The tangents handed to the run, asked through `t` and `live`.
    Tangents,
    /// No value: an operand without a tangent, a branch that yields nothing.
    Nil,
}

/// A refusal of the template evaluator.
fn bad<T>(what: &str) -> Res<T> {
    refuse(format!("rules: {what}"))
}

/// The part `i` of a template form.
fn part(a: &[J], i: usize) -> Res<&J> {
    a.get(i)
        .map_or_else(|| bad(&format!("a form lacks its part {i}")), Ok)
}

/// The text of part `i` of a form.
fn text_part(a: &[J], i: usize) -> Res<String> {
    Ok(part(a, i)?.str().unwrap_or("").to_string())
}

/// The shape as a literal.
fn shape_lit(shape: &[i64]) -> Lit {
    Lit::Tuple(shape.iter().map(|&d| Lit::Int(i128::from(d))).collect())
}

/// The entry indices of a shape in row-major order: an integer each for a vector, one tuple of an index per axis each for
/// a value of rank two or more.
fn entry_list(shape: &[i64]) -> Vec<Lit> {
    let int = |i: i64| Lit::Int(i128::from(i));
    if shape.len() == 1 {
        return (0..shape[0]).map(int).collect();
    }
    let mut out = vec![vec![]];
    for &extent in shape {
        out = out
            .into_iter()
            .flat_map(|head| {
                (0..extent).map(move |k| {
                    let mut next = head.clone();
                    next.push(Lit::Int(i128::from(k)));
                    next
                })
            })
            .collect();
    }
    out.into_iter().map(Lit::Tuple).collect()
}

/// The interpreter state: the context the nodes are made in, the names in scope, and what the run asks of its caller.
struct Interp<'a> {
    /// The context the nodes live in.
    ctx: &'a mut Ctx,
    /// The names in scope, the innermost last.
    env: Vec<(String, Item)>,
    /// The operands of the rule's node, the only nodes `t` answers for.
    operands: Vec<u32>,
    /// The (operand, tangent) pairs the caller handed in; an operand not listed has no tangent.
    tangents: &'a [(u32, u32)],
}

impl<'a> Interp<'a> {
    /// An interpreter over `ctx` with an empty scope.
    fn new(ctx: &'a mut Ctx, tangents: &'a [(u32, u32)]) -> Self {
        Interp {
            ctx,
            env: vec![],
            operands: vec![],
            tangents,
        }
    }

    /// The shape of node `n`.
    fn shape(&self, n: u32) -> Res<Vec<i64>> {
        Ok(self.ctx.node(n)?.shape.clone())
    }

    /// The element type name of node `n`.
    fn dtype(&self, n: u32) -> Res<String> {
        Ok(self.ctx.node(n)?.dtype.clone())
    }

    /// The type of node `n` as a type literal.
    fn type_lit(&self, n: u32) -> Res<Lit> {
        let node = self.ctx.node(n)?;
        Ok(ttype_val(&node.dtype, &node.shape, node.tag.as_deref()))
    }

    /// A rank-0 constant of `dtype` holding `value`.
    fn konst(&mut self, value: Lit, dtype: &str) -> Res<u32> {
        let ci = self
            .ctx
            .table
            .find("Const")
            .map_or_else(|| bad("no Const class"), Ok)?;
        let given = [
            ("literal".to_string(), value),
            ("ttype".to_string(), ttype_val(dtype, &[], None)),
        ];
        self.ctx.new_from(ci, &given)
    }

    /// An op of `kind` over `args` carrying `lit`.
    fn make(&mut self, kind: &str, args: &[u32], lit: Lit) -> Res<u32> {
        self.ctx.make_op(kind, args, &lit)
    }

    /// `cond ? a : b` typed as `ty`.
    fn select(&mut self, cond: u32, a: u32, b: u32, ty: Lit) -> Res<u32> {
        let given = vec![
            ("cond".to_string(), Lit::Node(cond)),
            ("on_true".to_string(), Lit::Node(a)),
            ("on_false".to_string(), Lit::Node(b)),
            ("ttype".to_string(), ty),
        ];
        let ci = self
            .ctx
            .table
            .find("Select")
            .map_or_else(|| bad("no Select class"), Ok)?;
        self.ctx.new_from(ci, &given)
    }

    /// An f64 value `v` of `shape`: a rank-0 constant, else a splat of it.
    fn filled(&mut self, v: f64, shape: &[i64]) -> Res<u32> {
        let k = self.konst(Lit::Float(v), "f64")?;
        if shape.is_empty() {
            Ok(k)
        } else {
            self.make("splat", &[k], shape_lit(shape))
        }
    }

    /// A zero of the dtype of `e`, of its shape.
    fn zero_like(&mut self, e: u32) -> Res<u32> {
        let (shape, dtype) = (self.shape(e)?, self.dtype(e)?);
        let z = self.konst(Lit::Float(0.0), &dtype)?;
        if shape.is_empty() {
            Ok(z)
        } else {
            self.make("splat", &[z], shape_lit(&shape))
        }
    }

    /// `kind` over `nodes` entry by entry when any is not rank-0, re-assembled with `vec`.
    fn entrywise(&mut self, kind: &str, nodes: &[u32]) -> Res<u32> {
        let mut shape = vec![];
        for &x in nodes {
            let s = self.shape(x)?;
            if !s.is_empty() {
                shape = s;
                break;
            }
        }
        if shape.is_empty() {
            return self.make(kind, nodes, Lit::None);
        }
        let mut parts = vec![];
        for k in entry_list(&shape) {
            let mut args = vec![];
            for &x in nodes {
                args.push(if self.shape(x)?.is_empty() {
                    x
                } else {
                    self.make("component", &[x], k.clone())?
                });
            }
            parts.push(self.make(kind, &args, Lit::None)?);
        }
        let lit = if shape.len() >= 2 {
            shape_lit(&shape)
        } else {
            Lit::None
        };
        self.make("vec", &parts, lit)
    }

    /// The value a name is bound to; `a.b` reads the field `b` of the record `a`.
    fn lookup(&self, name: &str) -> Res<Item> {
        let mut parts = name.split('.');
        let head = parts.next().unwrap_or("");
        let found = self.env.iter().rev().find(|e| e.0 == head);
        let mut cur = found
            .map(|e| e.1.clone())
            .map_or_else(|| bad(&format!("unbound name {name}")), Ok)?;
        for p in parts {
            cur = match cur {
                Item::Rec(f) => f
                    .into_iter()
                    .find(|x| x.0 == p)
                    .map(|x| x.1)
                    .map_or_else(|| bad(&format!("no field {p} in {name}")), Ok)?,
                _ => return bad(&format!("{name} is not a record")),
            };
        }
        Ok(cur)
    }

    /// Runs `f` with the names it binds dropped on return.
    fn scoped<T>(&mut self, f: impl FnOnce(&mut Self) -> Res<T>) -> Res<T> {
        let depth = self.env.len();
        let out = f(self);
        self.env.truncate(depth);
        out
    }

    /// The node a template stands for.
    fn node(&mut self, t: &J) -> Res<u32> {
        match self.ev(t)? {
            Item::Node(n) => Ok(n),
            _ => bad("a node was expected"),
        }
    }

    /// The tangent of operand `x`: `None` when the run gave it none; a node that is not an operand is refused.
    fn tangent(&self, x: u32) -> Res<Option<u32>> {
        if !matches!(self.lookup("t")?, Item::Tangents) {
            return bad("t needs the tangents of the run");
        }
        if !self.operands.contains(&x) {
            return bad("t asks the tangent of a node that is not an operand of the rule's node");
        }
        Ok(self.tangents.iter().find(|p| p.0 == x).map(|p| p.1))
    }

    /// The nodes of the arguments `a[from..]`, a `["...", list]` part spreading a list.
    fn args(&mut self, a: &[J], from: usize) -> Res<Vec<u32>> {
        let mut out = vec![];
        for x in a.iter().skip(from) {
            let spread = match x {
                J::Arr(sp) if sp.first().and_then(J::str) == Some("...") => Some(part(sp, 1)?),
                _ => None,
            };
            match spread {
                Some(list) => match self.ev(list)? {
                    Item::List(l) => {
                        for v in l {
                            out.push(into_node(v)?)
                        }
                    }
                    _ => return bad("... spreads a list"),
                },
                None => out.push(self.node(x)?),
            }
        }
        Ok(out)
    }

    /// Evaluates a template.
    fn ev(&mut self, t: &J) -> Res<Item> {
        match t {
            J::Str(n) => self.lookup(n),
            J::Num(v) => Ok(Item::Int(*v as i64)),
            J::Null => Ok(Item::Nil),
            J::Arr(a) => self.form(a),
            _ => bad("a template is a name, a number, null or an array"),
        }
    }

    /// Evaluates a form by its head.
    fn form(&mut self, a: &[J]) -> Res<Item> {
        let head = a.first().and_then(J::str).unwrap_or("");
        match head {
            "let" | "lets" | "list" | "rec" | "map" | "map_list" | "zipmap" | "nth" | "pair"
            | "call" | "grid" | "foldl" => self.binding_form(head, a),
            "ifeq" | "ifcmp" | "rank1" | "when" | "or" | "when_any" => self.branch_form(head, a),
            "lit" | "shape" | "vlit" | "entries" | "tup" | "dim" | "ci" | "c" => {
                self.value_form(head, a)
            }
            "t" | "live" | "sumn" => self.tangent_form(head, a),
            _ => self.node_form(head, a).map(Item::Node),
        }
    }

    /// The forms that bind names, build lists and records, and call a def.
    fn binding_form(&mut self, head: &str, a: &[J]) -> Res<Item> {
        match head {
            "let" => {
                let v = self.ev(part(a, 2)?)?;
                let name = text_part(a, 1)?;
                self.scoped(|s| {
                    s.env.push((name, v));
                    s.ev(part(a, 3)?)
                })
            }
            "lets" => self.scoped(|s| {
                for p in part(a, 1)?.arr() {
                    let v = s.ev(part(p.arr(), 1)?)?;
                    s.env.push((text_part(p.arr(), 0)?, v));
                }
                s.ev(part(a, 2)?)
            }),
            "list" => Ok(Item::List(
                a[1..].iter().map(|x| self.ev(x)).collect::<Res<_>>()?,
            )),
            "rec" => {
                let mut fields = vec![];
                for p in &a[1..] {
                    fields.push((text_part(p.arr(), 0)?, self.ev(part(p.arr(), 1)?)?));
                }
                Ok(Item::Rec(fields))
            }
            "pair" => Ok(Item::List(vec![
                self.ev(part(a, 1)?)?,
                self.ev(part(a, 2)?)?,
            ])),
            "nth" => {
                let list = self.ev(part(a, 1)?)?;
                let i = self.index(part(a, 2)?)?;
                match list {
                    Item::List(v) => v
                        .get(i)
                        .cloned()
                        .map_or_else(|| bad("index out of range"), Ok),
                    _ => bad("nth of a non-list"),
                }
            }
            "call" => self.call(a),
            "foldl" => self.foldl(a),
            _ => self.repeat_form(head, a),
        }
    }

    /// The forms that evaluate a body once per element: `map`, `map_list`, `zipmap` and `grid`.
    fn repeat_form(&mut self, head: &str, a: &[J]) -> Res<Item> {
        match head {
            "map" => {
                let count = self.int(part(a, 2)?)?;
                let name = text_part(a, 1)?;
                let mut out = vec![];
                for i in 0..count {
                    out.push(self.scoped(|s| {
                        s.env.push((name.clone(), Item::Int(i)));
                        s.ev(part(a, 3)?)
                    })?)
                }
                Ok(Item::List(out))
            }
            "map_list" => {
                let Item::List(l) = self.ev(part(a, 2)?)? else {
                    return bad("map_list takes a list");
                };
                let name = text_part(a, 1)?;
                let mut out = vec![];
                for v in l {
                    out.push(self.scoped(|s| {
                        s.env.push((name.clone(), v));
                        s.ev(part(a, 3)?)
                    })?)
                }
                Ok(Item::List(out))
            }
            "zipmap" => {
                let (Item::List(l1), Item::List(l2)) =
                    (self.ev(part(a, 2)?)?, self.ev(part(a, 4)?)?)
                else {
                    return bad("zipmap takes two lists");
                };
                let (n1, n2) = (text_part(a, 1)?, text_part(a, 3)?);
                let mut out = vec![];
                for (x, y) in l1.into_iter().zip(l2) {
                    out.push(self.scoped(|s| {
                        s.env.push((n1.clone(), x));
                        s.env.push((n2.clone(), y));
                        s.ev(part(a, 5)?)
                    })?)
                }
                Ok(Item::List(out))
            }
            _ => self.grid(a),
        }
    }

    /// `["grid", f, i, j, body]`: the m by m matrix of `body` over the indices, m the extent of `f`.
    fn grid(&mut self, a: &[J]) -> Res<Item> {
        let f = self.node(part(a, 1)?)?;
        let m = self.shape(f)?.first().copied().unwrap_or(0);
        let (ni, nj) = (text_part(a, 2)?, text_part(a, 3)?);
        let mut parts = vec![];
        for i in 0..m {
            for j in 0..m {
                parts.push(self.scoped(|s| {
                    s.env.push((ni.clone(), Item::Int(i)));
                    s.env.push((nj.clone(), Item::Int(j)));
                    s.node(part(a, 4)?)
                })?)
            }
        }
        Ok(Item::Node(self.make("vec", &parts, shape_lit(&[m, m]))?))
    }

    /// `["call", def, args...]`: the body of the manifest def over fresh names bound to the arguments.
    fn call(&mut self, a: &[J]) -> Res<Item> {
        let name = text_part(a, 1)?;
        let tables = Arc::clone(&self.ctx.tables);
        let def = tables
            .rules
            .get("defs")
            .and_then(|d| d.get(&name))
            .map_or_else(|| bad(&format!("no def {name}")), Ok)?;
        let mut fresh = vec![];
        for (p, x) in def
            .get("params")
            .map(J::strs)
            .unwrap_or_default()
            .into_iter()
            .zip(&a[2..])
        {
            fresh.push((p, self.ev(x)?))
        }
        let saved = std::mem::replace(&mut self.env, fresh);
        let out = self.ev(def.get("body").unwrap_or(&J::Null));
        self.env = saved;
        out
    }

    /// `["foldl", kind, list]`: the op `kind` folded over the nodes of the list from the left.
    fn foldl(&mut self, a: &[J]) -> Res<Item> {
        let kind = text_part(a, 1)?;
        let Item::List(l) = self.ev(part(a, 2)?)? else {
            return bad("foldl takes a list");
        };
        let mut it = l.into_iter();
        let Some(first) = it.next() else {
            return bad("foldl of an empty list");
        };
        let mut acc = into_node(first)?;
        for x in it {
            acc = self.make(&kind, &[acc, into_node(x)?], Lit::None)?
        }
        Ok(Item::Node(acc))
    }

    /// The integer a template stands for.
    fn int(&mut self, t: &J) -> Res<i64> {
        match self.ev(t)? {
            Item::Int(i) => Ok(i),
            _ => bad("an integer was expected"),
        }
    }

    /// The non-negative integer a template stands for, as an index.
    fn index(&mut self, t: &J) -> Res<usize> {
        usize::try_from(self.int(t)?).or_else(|_| bad("an index was expected"))
    }

    /// The literal a template stands for (an absent value is the literal None).
    fn literal(&mut self, t: &J) -> Res<Option<Lit>> {
        match self.ev(t)? {
            Item::Lit(l) => Ok(Some(l)),
            Item::Int(i) => Ok(Some(Lit::Int(i128::from(i)))),
            Item::Nil => Ok(None),
            _ => bad("a literal was expected"),
        }
    }

    /// The forms that choose between two templates.
    fn branch_form(&mut self, head: &str, a: &[J]) -> Res<Item> {
        match head {
            "ifeq" => {
                let eq = self.literal(part(a, 1)?)? == self.literal(part(a, 2)?)?;
                self.ev(part(a, if eq { 3 } else { 4 })?)
            }
            "ifcmp" => {
                let (x, y) = (self.int(part(a, 2)?)?, self.int(part(a, 3)?)?);
                let ok = match text_part(a, 1)?.as_str() {
                    "eq" => x == y,
                    "lt" => x < y,
                    "le" => x <= y,
                    "gt" => x > y,
                    _ => x >= y,
                };
                self.ev(part(a, if ok { 4 } else { 5 })?)
            }
            "rank1" => {
                let x = self.node(part(a, 1)?)?;
                let rank1 = self.shape(x)?.len() == 1;
                self.ev(part(a, if rank1 { 2 } else { 3 })?)
            }
            "when" => match self.ev(part(a, 1)?)? {
                Item::Nil => Ok(Item::Nil),
                _ => self.ev(part(a, 2)?),
            },
            "or" => match self.ev(part(a, 1)?)? {
                Item::Nil => self.ev(part(a, 2)?),
                v => Ok(v),
            },
            _ => {
                let mut any = false;
                for x in part(a, 1)?.arr() {
                    any |= !matches!(self.ev(x)?, Item::Nil);
                }
                if any {
                    self.ev(part(a, 2)?)
                } else {
                    Ok(Item::Nil)
                }
            }
        }
    }

    /// The forms that read a literal, a shape or an extent, and the scalar constants.
    fn value_form(&mut self, head: &str, a: &[J]) -> Res<Item> {
        match head {
            "c" => {
                let J::Num(v) = part(a, 1)? else {
                    return bad("c takes a number");
                };
                Ok(Item::Node(self.konst(Lit::Float(*v), "f64")?))
            }
            "ci" => {
                let J::Num(v) = part(a, 1)? else {
                    return bad("ci takes an integer");
                };
                Ok(Item::Node(self.konst(Lit::Int(*v as i128), "int32")?))
            }
            "lit" => match self.lookup("n")? {
                Item::Node(n) => Ok(Item::Lit(self.ctx.field(n, "literal")?)),
                _ => bad("lit of a non-node"),
            },
            "shape" => {
                let x = self.node(part(a, 1)?)?;
                Ok(Item::Lit(shape_lit(&self.shape(x)?)))
            }
            "dim" => {
                let x = self.node(part(a, 1)?)?;
                Ok(Item::Int(self.shape(x)?.first().copied().unwrap_or(0)))
            }
            "tup" => {
                let (x, y) = (self.int(part(a, 1)?)?, self.int(part(a, 2)?)?);
                Ok(Item::Lit(shape_lit(&[x, y])))
            }
            _ => self.shape_form(head, a),
        }
    }

    /// The forms over a shape literal: `vlit` keeps a shape of rank two or more, `entries` lists the entry indices.
    fn shape_form(&mut self, head: &str, a: &[J]) -> Res<Item> {
        let Item::Lit(Lit::Tuple(dims)) = self.ev(part(a, 1)?)? else {
            return bad(&format!("{head} takes a shape"));
        };
        if head == "vlit" {
            return Ok(Item::Lit(if dims.len() >= 2 {
                Lit::Tuple(dims)
            } else {
                Lit::None
            }));
        }
        let shape: Vec<i64> = dims
            .iter()
            .map(|d| match d {
                Lit::Int(i) => i64::try_from(*i).or_else(|_| bad("an extent is too large")),
                _ => bad("a shape holds integers"),
            })
            .collect::<Res<_>>()?;
        Ok(Item::List(
            entry_list(&shape).into_iter().map(Item::Lit).collect(),
        ))
    }

    /// The forms over the tangents of the run: `t`, `live` and the sum of the live terms.
    fn tangent_form(&mut self, head: &str, a: &[J]) -> Res<Item> {
        match head {
            "t" => {
                let x = self.node(part(a, 1)?)?;
                Ok(self.tangent(x)?.map_or(Item::Nil, Item::Node))
            }
            "live" => {
                // the tangent of the operand, a structural f64 zero when it has none
                let x = self.node(part(a, 1)?)?;
                if let Some(d) = self.tangent(x)? {
                    return Ok(Item::Node(d));
                }
                let shape = self.shape(x)?;
                Ok(Item::Node(self.filled(0.0, &shape)?))
            }
            _ => {
                let mut live = vec![];
                for x in &a[1..] {
                    if let Item::Node(v) = self.ev(x)? {
                        live.push(v)
                    }
                }
                let Some((&first, rest)) = live.split_first() else {
                    return Ok(Item::Nil);
                };
                let mut acc = first;
                for &x in rest {
                    acc = self.make("add", &[acc, x], Lit::None)?
                }
                Ok(Item::Node(acc))
            }
        }
    }

    /// The forms that make one node.
    fn node_form(&mut self, head: &str, a: &[J]) -> Res<u32> {
        match head {
            "opl" => {
                let lit = self.literal(part(a, 2)?)?.unwrap_or(Lit::None);
                let v = self.args(a, 3)?;
                self.make(&text_part(a, 1)?, &v, lit)
            }
            "k" => {
                let v = self.node(part(a, 1)?)?;
                let i = self.int(part(a, 2)?)?;
                self.make("component", &[v], Lit::Int(i128::from(i)))
            }
            "operand" => {
                let x = self.node(part(a, 1)?)?;
                let i = self.index(part(a, 2)?)?;
                let ops = &self.ctx.node(x)?.ops;
                ops.get(i)
                    .copied()
                    .map_or_else(|| bad("operand out of range"), Ok)
            }
            "scalar_zero" => {
                let e = self.node(part(a, 1)?)?;
                let dtype = self.dtype(e)?;
                self.konst(Lit::Float(0.0), &dtype)
            }
            "zero_f64" => {
                let e = self.node(part(a, 1)?)?;
                let shape = self.shape(e)?;
                self.filled(0.0, &shape)
            }
            "zero_like" => {
                let e = self.node(part(a, 1)?)?;
                self.zero_like(e)
            }
            "entrywise" => {
                let v = self.args(a, 2)?;
                self.entrywise(&text_part(a, 1)?, &v)
            }
            _ => self.select_form(head, a),
        }
    }

    /// The forms built on a select: `sel`, `where`, `ramp`, `pick` and `abs_d`; any other head is an op of that kind.
    fn select_form(&mut self, head: &str, a: &[J]) -> Res<u32> {
        match head {
            "sel" | "where" => {
                let v = self.args(a, 1)?;
                let (Some(&c), Some(&x), Some(&y)) = (v.first(), v.get(1), v.get(2)) else {
                    return bad("sel takes three nodes");
                };
                let ty = self.type_lit(x)?;
                self.select(c, x, y, ty)
            }
            "ramp" => {
                let v = self.args(a, 1)?;
                let (Some(&c), Some(&like)) = (v.first(), v.get(1)) else {
                    return bad("ramp takes a condition and a node");
                };
                let dtype = self.dtype(like)?;
                let (one, zero) = (
                    self.konst(Lit::Float(1.0), &dtype)?,
                    self.konst(Lit::Float(0.0), &dtype)?,
                );
                self.select(c, one, zero, ttype_val(&dtype, &[], None))
            }
            "pick" => self.pick(a),
            "abs_d" => self.abs_d(a),
            kind => {
                let v = self.args(a, 1)?;
                if (kind == "add" || kind == "mul") && v.len() > 2 {
                    // a variadic sum or product folds from the left
                    let mut acc = v[0];
                    for &x in &v[1..] {
                        acc = self.make(kind, &[acc, x], Lit::None)?
                    }
                    return Ok(acc);
                }
                self.make(kind, &v, Lit::None)
            }
        }
    }

    /// `["pick", lo, hi, g, first]`: `g` where `lo <= hi` (the first arm) or where it is not (the second), zero elsewhere.
    fn pick(&mut self, a: &[J]) -> Res<u32> {
        let (lo, hi, g) = (
            self.node(part(a, 1)?)?,
            self.node(part(a, 2)?)?,
            self.node(part(a, 3)?)?,
        );
        let first = matches!(part(a, 4)?, J::Bool(true));
        let cond = self.entrywise("le", &[lo, hi])?;
        let cond_shape = self.shape(cond)?;
        let g = if !cond_shape.is_empty() && self.shape(g)?.is_empty() {
            self.make("splat", &[g], shape_lit(&cond_shape))?
        } else {
            g
        };
        let z = self.zero_like(g)?;
        let ty = self.type_lit(g)?;
        if first {
            self.select(cond, g, z, ty)
        } else {
            self.select(cond, z, g, ty)
        }
    }

    /// `["abs_d", x]`: the derivative of `abs`, +1 where `x >= 0` and -1 elsewhere, per entry.
    fn abs_d(&mut self, a: &[J]) -> Res<u32> {
        let x = self.node(part(a, 1)?)?;
        let shape = self.shape(x)?;
        if shape.is_empty() {
            let zero = self.zero_like(x)?;
            let cond = self.make("ge", &[x, zero], Lit::None)?;
            let (up, down) = (self.filled(1.0, &[])?, self.filled(-1.0, &[])?);
            return self.select(cond, up, down, ttype_val("f64", &[], None));
        }
        let zero = self.filled(0.0, &[])?;
        let cond = self.entrywise("ge", &[x, zero])?;
        let (up, down) = (self.filled(1.0, &shape)?, self.filled(-1.0, &shape)?);
        self.select(cond, up, down, ttype_val("f64", &shape, None))
    }

    /// The program of `table`/`kind` run on the operands of `n`: its `lets` in order, then the pairs `vjp` (`g` the cotangent) or
    /// the expression `jvp`. The result is the nodes it yields: pairs flattened for `vjp`, none or one for `jvp`.
    fn run(&mut self, table: &str, kind: &str, vjp: bool, n: u32, arg: Item) -> Res<Vec<u32>> {
        let mode = if vjp { "vjp" } else { "jvp" };
        let tables = Arc::clone(&self.ctx.tables);
        let t = tables
            .rules
            .get(table)
            .and_then(|t| t.get(kind))
            .map_or_else(|| bad(&format!("no {table} program for {kind}")), Ok)?;
        self.operands = self.ctx.node(n)?.ops.clone();
        let ops: Vec<Item> = self.operands.iter().map(|&o| Item::Node(o)).collect();
        self.env = vec![
            ("n".into(), Item::Node(n)),
            (if vjp { "g" } else { "t" }.into(), arg),
            ("ops".into(), Item::List(ops.clone())),
        ];
        let names = t.get("operands").map(J::strs).unwrap_or_default();
        self.env.extend(bind_operands(&names, ops));
        for key in ["lets".to_string(), format!("{mode}_lets")] {
            for p in t.get(&key).map_or(&[][..], J::arr) {
                let v = self.ev(part(p.arr(), 1)?)?;
                self.env.push((text_part(p.arr(), 0)?, v));
            }
        }
        if !vjp {
            return match self.ev(t.get("jvp").unwrap_or(&J::Null))? {
                Item::Node(d) => Ok(vec![d]),
                _ => Ok(vec![]),
            };
        }
        match t.get("vjp_expr") {
            Some(expr) => {
                let Item::List(pairs) = self.ev(expr)? else {
                    return bad("vjp_expr is a list of pairs");
                };
                let mut out = vec![];
                for p in pairs {
                    let Item::List(p) = p else {
                        return bad("a pair");
                    };
                    for x in p {
                        out.push(into_node(x)?)
                    }
                }
                Ok(out)
            }
            None => self.pairs(t.get("vjp").map_or(&[][..], J::arr)),
        }
    }

    /// The pairs `[child, contribution]` of a template list, flattened.
    fn pairs(&mut self, list: &[J]) -> Res<Vec<u32>> {
        let mut out = vec![];
        for p in list {
            out.push(self.node(part(p.arr(), 0)?)?);
            out.push(self.node(part(p.arr(), 1)?)?);
        }
        Ok(out)
    }
}

/// Binds the operand names of a rule to the operands of its node. A name written `...rest` takes every operand
/// the other names leave over, as a list: the element accesses carry one index per axis of their value.
fn bind_operands(names: &[String], ops: Vec<Item>) -> Vec<(String, Item)> {
    let Some(at) = names.iter().position(|n| n.starts_with("...")) else {
        return names.iter().cloned().zip(ops).collect();
    };
    let take = ops.len().saturating_sub(names.len() - 1);
    let mut out: Vec<(String, Item)> = names[..at]
        .iter()
        .cloned()
        .zip(ops.iter().cloned())
        .collect();
    out.push((
        names[at][3..].to_string(),
        Item::List(ops[at..at + take].to_vec()),
    ));
    out.extend(
        names[at + 1..]
            .iter()
            .cloned()
            .zip(ops[at + take..].iter().cloned()),
    );
    out
}

/// The node an item holds.
fn into_node(v: Item) -> Res<u32> {
    match v {
        Item::Node(n) => Ok(n),
        _ => bad("a node was expected"),
    }
}

impl Ctx {
    /// The factor `table`/`kind` names over the nodes bound in `bound`, the names in `absent` standing for no value. The nodes it
    /// yields in order: one for a factor, the live entries for `bin_jvp`, the pairs flattened for `bin_vjp`.
    pub fn rule_factor(
        &mut self,
        table: &str,
        kind: &str,
        bound: &[(String, u32)],
        absent: &[String],
    ) -> Res<Vec<u32>> {
        let tables = Arc::clone(&self.tables);
        let t = tables.template(table, kind)?;
        let mut it = Interp::new(self, &[]);
        for (name, n) in bound {
            it.env.push((name.clone(), Item::Node(*n)));
        }
        for name in absent {
            it.env.push((name.clone(), Item::Nil));
        }
        match table {
            "partials" | "bin_jvp" => {
                let mut out = vec![];
                for x in t.arr() {
                    if let Item::Node(n) = it.ev(x)? {
                        out.push(n)
                    }
                }
                Ok(out)
            }
            "bin_vjp" => it.pairs(t.arr()),
            _ => Ok(vec![it.node(t)?]),
        }
    }

    /// The reverse contributions of node `n` under the program `table`/`kind`, `cot` the incoming adjoint: (child, adjoint) pairs
    /// flattened.
    pub fn rule_vjp(&mut self, table: &str, kind: &str, n: u32, cot: u32) -> Res<Vec<u32>> {
        self.node(cot)?;
        Interp::new(self, &[]).run(table, kind, true, n, Item::Node(cot))
    }

    /// The tangent of node `n` under the program `table`/`kind`, `tangents` the (operand, tangent) pairs: none or one node.
    pub fn rule_jvp(
        &mut self,
        table: &str,
        kind: &str,
        n: u32,
        tangents: &[(u32, u32)],
    ) -> Res<Vec<u32>> {
        Interp::new(self, tangents).run(table, kind, false, n, Item::Tangents)
    }

    /// The kinds a rule table has templates for, in manifest order.
    pub fn rule_kinds(&self, table: &str) -> Vec<String> {
        self.tables
            .rules
            .get(table)
            .map(|t| t.obj().iter().map(|x| x.0.clone()).collect())
            .unwrap_or_default()
    }

    /// The kinds of rule mode `mode` (`zero`: no derivative; `walker`: left to the transform), in table order; `@NAME` in the
    /// table names a manifest constant.
    pub fn rule_modes(&self, mode: &str) -> Vec<String> {
        let mut out = vec![];
        let list = self.tables.rules.get("modes").and_then(|m| m.get(mode));
        for k in list.map_or(&[][..], J::arr) {
            let k = k.str().unwrap_or("");
            match k.strip_prefix('@') {
                Some(c) => out.extend(
                    crate::consts::names(c)
                        .unwrap_or_default()
                        .into_iter()
                        .map(String::from),
                ),
                None => out.push(k.to_string()),
            }
        }
        out
    }

    /// The operand of the sum over samples `n`; any other reduction over samples has no derivative and is refused.
    fn reduce_parts(&self, n: u32) -> Res<u32> {
        let op = self.str_field(n, "op")?;
        if op != "sum" {
            return refuse(format!("no derivative for the {op} over samples read in the body: its subgradient picks one sample, so it is not in this release; reduce with a sum"));
        }
        self.node(n)?
            .ops
            .first()
            .copied()
            .map_or_else(|| bad("a reduction without an operand"), Ok)
    }

    /// The reverse contributions of a sum over samples: the pair (operand, the sum rebuilt over the adjoint `x`).
    pub fn reduce_vjp(&mut self, n: u32, x: u32) -> Res<Vec<u32>> {
        let child = self.reduce_parts(n)?;
        Ok(vec![child, self.rebuild(n, &[x])?])
    }

    /// The tangent of a sum over samples: the sum rebuilt over the operand's tangent; none when the operand has none.
    pub fn reduce_jvp(&mut self, n: u32, tangents: &[(u32, u32)]) -> Res<Vec<u32>> {
        let child = self.reduce_parts(n)?;
        match tangents.iter().find(|p| p.0 == child) {
            Some(p) => Ok(vec![self.rebuild(n, &[p.1])?]),
            None => Ok(vec![]),
        }
    }

    /// `kind` over `nodes` entry by entry when any is not rank-0, re-assembled with `vec`.
    pub fn entrywise_op(&mut self, kind: &str, nodes: &[u32]) -> Res<u32> {
        for &x in nodes {
            self.node(x)?;
        }
        Interp::new(self, &[]).entrywise(kind, nodes)
    }

    /// A structural zero of `dtype` and `shape`: a rank-0 constant, else a splat of it. The literal is the dtype's own: real for a
    /// floating dtype, integer for an integer one, false for `bool`; any other dtype has no zero and is refused by name.
    pub fn zero_of(&mut self, dtype: &str, shape: &[i64]) -> Res<u32> {
        let zero = if is_real_dtype(dtype) {
            Lit::Float(0.0)
        } else if is_integer_dtype(dtype) {
            Lit::Int(0)
        } else if dtype == "bool" {
            Lit::Bool(false)
        } else {
            return bad(&format!("zero_of: dtype {dtype:?} has no zero"));
        };
        let mut it = Interp::new(self, &[]);
        let z = it.konst(zero, dtype)?;
        if shape.is_empty() {
            Ok(z)
        } else {
            it.make("splat", &[z], shape_lit(shape))
        }
    }
}

/// The names of a text, separated by white space.
fn words(p: *const c_char, n: usize, what: &str) -> Res<Vec<String>> {
    let bytes = slice(p.cast::<u8>(), n, what)?;
    let text = std::str::from_utf8(bytes).or_else(|_| badarg(format!("{what} is not UTF-8")))?;
    Ok(text.split_whitespace().map(String::from).collect())
}

/// The (operand, tangent) pairs of two parallel node arrays.
fn tangent_pairs(of: *const u32, tangent: *const u32, n: usize) -> Res<Vec<(u32, u32)>> {
    let of = slice(of, n, "tangent_of")?;
    let tangent = slice(tangent, n, "tangent")?;
    Ok(of.iter().copied().zip(tangent.iter().copied()).collect())
}

/// The factor `table`/`kind` over the nodes named in `bound` (names separated by white space, one node each in `nodes`), the
/// names in `absent` standing for no value. The nodes come back in `buf`: one for a factor, the live entries for `bin_jvp`,
/// the pairs (child, contribution) flattened for `bin_vjp`.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_rule_factor(
    ctx: *mut Ctx,
    table: *const c_char,
    kind: *const c_char,
    bound: *const c_char,
    bound_len: usize,
    nodes: *const u32,
    nnodes: usize,
    absent: *const c_char,
    absent_len: usize,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let names = words(bound, bound_len, "bound")?;
        let nodes = slice(nodes, nnodes, "nodes")?;
        if names.len() != nodes.len() {
            return badarg("bound names a different number of nodes than nodes holds");
        }
        let env: Vec<(String, u32)> = names.into_iter().zip(nodes.iter().copied()).collect();
        let out = c.rule_factor(
            cstr(table, "table")?,
            cstr(kind, "kind")?,
            &env,
            &words(absent, absent_len, "absent")?,
        )?;
        give(buf, cap, n, &out)
    })
}

/// The reverse contributions of `node` under the program `table`/`kind`, `cot` the incoming adjoint: (child, adjoint) pairs
/// flattened into `buf`.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_rule_vjp(
    ctx: *mut Ctx,
    table: *const c_char,
    kind: *const c_char,
    node: u32,
    cot: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let out = c.rule_vjp(cstr(table, "table")?, cstr(kind, "kind")?, node, cot)?;
        give(buf, cap, n, &out)
    })
}

/// The tangent of `node` under the program `table`/`kind`: `tangent_of[i]` has the tangent `tangent[i]`, an operand not listed
/// has none. Zero or one node into `buf`.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_rule_jvp(
    ctx: *mut Ctx,
    table: *const c_char,
    kind: *const c_char,
    node: u32,
    tangent_of: *const u32,
    tangent: *const u32,
    count: usize,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let pairs = tangent_pairs(tangent_of, tangent, count)?;
        let out = c.rule_jvp(cstr(table, "table")?, cstr(kind, "kind")?, node, &pairs)?;
        give(buf, cap, n, &out)
    })
}

/// The reverse contributions of the sum over samples `node`: the pair (operand, sum over the adjoint `cot`) into `buf`.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_reduce_vjp(
    ctx: *mut Ctx,
    node: u32,
    cot: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        c.node(cot)?;
        let out = c.reduce_vjp(node, cot)?;
        give(buf, cap, n, &out)
    })
}

/// The tangent of the sum over samples `node`, as `rx_rule_jvp` takes its tangents: zero or one node into `buf`.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_reduce_jvp(
    ctx: *mut Ctx,
    node: u32,
    tangent_of: *const u32,
    tangent: *const u32,
    count: usize,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let out = c.reduce_jvp(node, &tangent_pairs(tangent_of, tangent, count)?)?;
        give(buf, cap, n, &out)
    })
}

/// The kinds of a rule table in manifest order, as the names protocol.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_rule_kinds(
    ctx: *mut Ctx,
    table: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let kinds = c.rule_kinds(cstr(table, "table")?);
        put_names(&kinds, buf, cap, len, count)
    })
}

/// The kinds of a rule mode (`zero` or `walker`) in table order, as the names protocol.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_rule_modes(
    ctx: *mut Ctx,
    mode: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let kinds = c.rule_modes(cstr(mode, "mode")?);
        put_names(&kinds, buf, cap, len, count)
    })
}

/// `kind` over `nodes` entry by entry when any is not rank-0, re-assembled with `vec`.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_entrywise(
    ctx: *mut Ctx,
    kind: *const c_char,
    nodes: *const u32,
    nnodes: usize,
    out: *mut u32,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return badarg("out is NULL");
        }
        let made = c.entrywise_op(cstr(kind, "kind")?, slice(nodes, nnodes, "nodes")?)?;
        unsafe { *out = made };
        Ok(())
    })
}

/// A structural zero of a dtype and shape.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_zero_of(
    ctx: *mut Ctx,
    dtype: *const c_char,
    shape: *const i64,
    rank: usize,
    out: *mut u32,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return badarg("out is NULL");
        }
        let made = c.zero_of(cstr(dtype, "dtype")?, slice(shape, rank, "shape")?)?;
        unsafe { *out = made };
        Ok(())
    })
}
