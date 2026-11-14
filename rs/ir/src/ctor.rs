//! The declared constructor checks over records. The programs of the manifest are compiled once per context; a node's state is mapped
//! back to constructor parameters, the program runs forward over them, and the state it produces must equal the record's state.
use crate::record::{Ctx, Node};
use crate::value::{text_val, ttype_val, Val};
use rcore::json::{self, J};
use rcore::keytext::py_str;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum Cmp { Eq, Ne, Lt, Gt, In, NotIn, Is, IsNot }
enum E { Var(usize), Const(usize), Class(String), Int(i128), Str(String), Nil, True, False, Tuple(Vec<E>), Attr(Box<E>, String), Index(Box<E>, Box<E>),
         Not(Box<E>), And(Box<E>, Box<E>), Or(Box<E>, Box<E>), Cmp(Box<E>, Cmp, Box<E>), Call(String, Vec<E>) }
enum Piece { Lit(String), Val(E, bool) }
enum Step { Let(usize, E), Check(E, Vec<Piece>), Node(E, E), Call(usize, Vec<E>) }
struct Prog { params: usize, nvars: usize, steps: Vec<Step>, out: Vec<E>, slots: Vec<String>, from: Vec<(Option<usize>, String)> }
/// The compiled programs of one context.
pub struct Progs { named: Vec<Prog>, named_idx: HashMap<String, usize>, by_class: HashMap<String, usize>, progs: Vec<Prog>, consts: Vec<(String, Val)> }

const BUILTINS: &[&str] = &["len", "isinstance", "tt", "width_of", "rem", "int", "tuple", "cat", "opt", "head", "first_ne_ttype", "first_not_node", "coalesce", "orelse",
    "name_of", "bool", "trip", "first_mismatch", "first_not_instance", "carry_fault", "ttype_fault", "literal_fault"];
const RESERVED: &[&str] = &["_cls", "_t", "_o"];

struct P<'a> { c: Vec<char>, i: usize, vars: &'a [String], consts: &'a [(String, Val)], classes: &'a [String] }
type R<T> = Result<T, String>;
impl P<'_> {
    fn ws(&mut self) { while self.c.get(self.i) == Some(&' ') { self.i += 1 } }
    fn at(&self, w: &str) -> bool { w.chars().enumerate().all(|(k, ch)| self.c.get(self.i + k) == Some(&ch)) }
    fn sym(&mut self, w: &str) -> bool { self.ws(); if self.at(w) { self.i += w.chars().count(); true } else { false } }
    fn kw(&mut self, w: &str) -> bool {
        self.ws();
        let n = w.chars().count();
        if self.at(w) && !self.c.get(self.i + n).is_some_and(|x| x.is_alphanumeric() || *x == '_') { self.i += n; true } else { false }
    }
    fn or(&mut self) -> R<E> { let mut l = self.and()?; while self.kw("or") { l = E::Or(l.into(), self.and()?.into()) } Ok(l) }
    fn and(&mut self) -> R<E> { let mut l = self.not()?; while self.kw("and") { l = E::And(l.into(), self.not()?.into()) } Ok(l) }
    fn not(&mut self) -> R<E> { if self.kw("not") { Ok(E::Not(self.not()?.into())) } else { self.cmp() } }
    fn cmp(&mut self) -> R<E> {
        let l = self.post()?;
        let op = if self.sym("==") { Cmp::Eq } else if self.sym("!=") { Cmp::Ne } else if self.sym("<") { Cmp::Lt } else if self.sym(">") { Cmp::Gt }
            else if self.kw("not") { if self.kw("in") { Cmp::NotIn } else { return Err("expected `in` after `not`".into()) } }
            else if self.kw("in") { Cmp::In } else if self.kw("is") { if self.kw("not") { Cmp::IsNot } else { Cmp::Is } } else { return Ok(l) };
        Ok(E::Cmp(l.into(), op, self.post()?.into()))
    }
    fn post(&mut self) -> R<E> {
        let mut e = self.atom()?;
        loop {
            if self.sym(".") { e = E::Attr(e.into(), self.name()?) }
            else if self.sym("[") { let i = self.or()?; if !self.sym("]") { return Err("expected ]".into()) } e = E::Index(e.into(), i.into()) }
            else { return Ok(e) }
        }
    }
    fn name(&mut self) -> R<String> {
        self.ws();
        let st = self.i;
        while self.c.get(self.i).is_some_and(|x| x.is_alphanumeric() || *x == '_') { self.i += 1 }
        if st == self.i { return Err(format!("name expected at {}", self.i)) }
        Ok(self.c[st..self.i].iter().collect())
    }
    fn args(&mut self, close: &str) -> R<(Vec<E>, bool)> {
        let (mut v, mut comma) = (vec![], false);
        loop {
            if self.sym(close) { return Ok((v, comma)) }
            v.push(self.or()?);
            if self.sym(",") { comma = true } else if self.sym(close) { return Ok((v, comma)) } else { return Err(format!("expected , or {close}")) }
        }
    }
    fn atom(&mut self) -> R<E> {
        self.ws();
        match self.c.get(self.i).copied() {
            Some('(') => { self.i += 1; let (mut v, comma) = self.args(")")?; if v.len() == 1 && !comma { Ok(v.pop().unwrap()) } else { Ok(E::Tuple(v)) } }
            Some('\'') => {
                self.i += 1;
                let st = self.i;
                while self.c.get(self.i).is_some_and(|x| *x != '\'') { self.i += 1 }
                self.i += 1;
                Ok(E::Str(self.c[st..self.i - 1].iter().collect()))
            }
            Some(c) if c.is_ascii_digit() => {
                let st = self.i;
                while self.c.get(self.i).is_some_and(|x| x.is_ascii_digit()) { self.i += 1 }
                self.c[st..self.i].iter().collect::<String>().parse().map(E::Int).map_err(|_| "bad integer".to_string())
            }
            _ => {
                let n = self.name()?;
                match n.as_str() { "None" => return Ok(E::Nil), "True" => return Ok(E::True), "False" => return Ok(E::False), _ => {} }
                if self.sym("(") {
                    if !BUILTINS.contains(&n.as_str()) { return Err(format!("unknown function {n}")) }
                    return Ok(E::Call(n, self.args(")")?.0));
                }
                if let Some(i) = self.vars.iter().position(|v| *v == n) { Ok(E::Var(i)) }
                else if let Some(i) = self.consts.iter().position(|c| c.0 == n) { Ok(E::Const(i)) }
                else if n == "TensorType" || n == "Node" || self.classes.contains(&n) { Ok(E::Class(n)) }
                else { Err(format!("unknown name {n}")) }
            }
        }
    }
}

struct Scope<'a> { vars: Vec<String>, consts: &'a [(String, Val)], classes: &'a [String] }
impl Scope<'_> {
    fn expr(&self, t: &str) -> R<E> {
        let mut p = P { c: t.chars().collect(), i: 0, vars: &self.vars, consts: self.consts, classes: self.classes };
        let e = p.or()?;
        p.ws();
        if p.i != p.c.len() { return Err(format!("trailing text in `{t}`")) }
        Ok(e)
    }
    fn tmpl(&self, t: &str) -> R<Vec<Piece>> {
        let (mut out, mut lit, c, mut i) = (vec![], String::new(), t.chars().collect::<Vec<_>>(), 0);
        while i < c.len() {
            if c[i] != '{' { lit.push(c[i]); i += 1; continue }
            let (mut j, mut d) = (i + 1, 1);
            while j < c.len() { if c[j] == '{' { d += 1 } if c[j] == '}' { d -= 1; if d == 0 { break } } j += 1 }
            let mut inner: String = c[i + 1..j.min(c.len())].iter().collect();
            let r = inner.ends_with("!r");
            if r { inner.truncate(inner.len() - 2) }
            if !lit.is_empty() { out.push(Piece::Lit(std::mem::take(&mut lit))) }
            out.push(Piece::Val(self.expr(&inner)?, r));
            i = j + 1;
        }
        if !lit.is_empty() { out.push(Piece::Lit(lit)) }
        Ok(out)
    }
}

fn names(p: &str) -> Vec<String> {
    p.split(',').map(str::trim).filter(|x| !x.is_empty() && *x != "*").map(|x| x.split('=').next().unwrap().split(':').next().unwrap().trim().to_string()).collect()
}

fn prog(who: &str, params: Vec<String>, steps: &[J], out: &[String], slots: Vec<String>, from: &[String], named: &HashMap<String, (usize, usize)>, consts: &[(String, Val)], classes: &[String]) -> R<Prog> {
    let mut sc = Scope { vars: params.clone(), consts, classes };
    sc.vars.extend(RESERVED.iter().map(|s| s.to_string()));
    let mut cs = vec![];
    for st in steps {
        let a = st.arr();
        let t = |i: usize| a.get(i).and_then(J::str).unwrap_or("");
        let w = |e: String| format!("{who}: {e}");
        cs.push(match t(0) {
            "let" => { let e = sc.expr(t(2)).map_err(w)?; sc.vars.push(t(1).to_string()); Step::Let(sc.vars.len() - 1, e) }
            "check" => Step::Check(sc.expr(t(1)).map_err(w)?, sc.tmpl(t(2)).map_err(w)?),
            "node" => Step::Node(sc.expr(t(1)).map_err(w)?, sc.expr(t(2)).map_err(w)?),
            "call" => {
                let args = (2..a.len()).map(|i| sc.expr(t(i))).collect::<R<Vec<_>>>().map_err(w)?;
                match named.get(t(1)) { Some((ix, np)) if *np == args.len() => Step::Call(*ix, args), _ => return Err(format!("{who}: bad call of check {}", t(1))) }
            }
            x => return Err(format!("{who}: unknown step {x}")),
        });
    }
    let outs = out.iter().map(|e| sc.expr(e)).collect::<R<Vec<_>>>().map_err(|e| format!("{who}: {e}"))?;
    let from = from.iter().map(|t| match t.split_once('=') {
        Some((k, v)) if !(t.starts_with('@') || t.starts_with("edge") || t.starts_with("op")) => (params.iter().position(|p| p == k), v.to_string()),
        _ => (None, t.to_string()),
    }).collect();
    Ok(Prog { params: params.len(), nvars: sc.vars.len(), steps: cs, out: outs, slots, from })
}

impl Progs {
    pub fn compile() -> Result<Progs, String> {
        let m = json::parse(rcore::MANIFEST)?;
        let consts: Vec<(String, Val)> = m.get("consts").map_or(&[][..], J::obj).iter()
            .filter(|(k, v)| matches!(v, J::Arr(_)) && !k.contains('.')).map(|(k, v)| (k.clone(), Val::Tuple(v.strs().into_iter().map(Val::Str).collect()))).collect();
        let kinds = m.get("kinds").ok_or("no kinds")?.arr();
        let classes: Vec<String> = kinds.iter().filter_map(|k| k.get("class").and_then(J::str).map(String::from)).collect();
        let chk = m.get("checks").map_or(&[][..], J::obj);
        let named_map: HashMap<String, (usize, usize)> = chk.iter().enumerate().map(|(i, (k, v))| (k.clone(), (i, v.get("params").map_or(0, |p| p.arr().len())))).collect();
        let mut named = vec![];
        for (k, v) in chk { named.push(prog(k, v.get("params").map_or(vec![], J::strs), v.get("steps").map_or(&[][..], J::arr), &[], vec![], &[], &named_map, &consts, &classes)?) }
        let (mut progs, mut by_class) = (vec![], HashMap::new());
        for k in kinds {
            let Some(c) = k.get("ctor") else { continue };
            let cls = k.get("class").and_then(J::str).ok_or("a kind without a class")?.to_string();
            let mut slots = vec!["ttype".to_string(), "operands".to_string()];
            slots.extend(k.get("assigns").map_or(vec![], J::strs));
            let from = k.get("from").map_or(vec![], J::strs);
            by_class.insert(cls.clone(), progs.len());
            progs.push(prog(&cls, names(k.get("params").and_then(J::str).unwrap_or("")), c.get("steps").map_or(&[][..], J::arr), &c.get("out").map_or(vec![], J::strs), slots, &from, &named_map, &consts, &classes)?);
        }
        let named_idx = chk.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect::<HashMap<_, _>>();
        if !named_idx.contains_key("node_base") { return Err("the manifest declares no `node_base` check".into()) }
        Ok(Progs { named, named_idx, by_class, progs, consts })
    }
}

fn num(v: &Val) -> Option<f64> { match v { Val::Int(i) => Some(*i as f64), Val::Bool(b) => Some(*b as u8 as f64), Val::Float(f) => Some(*f), _ => None } }
/// Python `==` over values.
fn veq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Tuple(x), Val::Tuple(y)) | (Val::List(x), Val::List(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| veq(p, q)),
        _ => match (num(a), num(b)) { (Some(x), Some(y)) => x == y, _ => text_val(a) == text_val(b) },
    }
}
fn truthy(v: &Val) -> bool {
    match v { Val::None => false, Val::Bool(b) => *b, Val::Int(i) => *i != 0, Val::Float(f) => *f != 0.0, Val::Str(s) => !s.is_empty(), Val::Tuple(t) | Val::List(t) => !t.is_empty(), _ => true }
}
fn seq(v: &Val) -> R<&[Val]> { match v { Val::Tuple(t) | Val::List(t) => Ok(t), _ => Err(format!("{} is not a sequence", text_val(v))) } }
fn int(v: &Val) -> R<i128> { match v { Val::Int(i) => Ok(*i), Val::Bool(b) => Ok(*b as i128), Val::Float(f) => Ok(*f as i128), _ => Err(format!("{} is not an integer", text_val(v))) } }
fn has_list(v: &Val) -> bool { match v { Val::List(_) => true, Val::Tuple(t) => t.iter().any(has_list), _ => false } }
fn plain(v: &Val) -> bool { match v { Val::Node(_) => false, Val::Rec(n, _) => n == "RawBlock", Val::Tuple(t) | Val::List(t) => t.iter().all(plain), _ => true } }
fn literal_fault(v: &Val) -> Option<String> {
    if has_list(v) { return Some("a list is not a literal (it can change after the node is built); use a tuple".into()) }
    if plain(v) { None } else { Some(format!("{} is not a plain literal", text_val(v))) }
}
fn ttype_fault(t: &Val) -> Option<String> {
    let Val::Rec(_, fs) = t else { return Some("not a TensorType".into()) };
    let get = |k: &str| fs.iter().find(|f| f.0 == k).map_or(Val::None, |f| f.1.clone());
    let dt = get("dtype");
    if !matches!(&dt, Val::Str(d) if rcore::facts::DTYPES.contains(&d.as_str())) {
        return Some(format!("TensorType.dtype {} is not one of {}", text_val(&dt), rcore::repr_tuple_str(&rcore::facts::DTYPES)));
    }
    let sh = get("shape");
    let Val::Tuple(es) = &sh else { return Some(format!("TensorType.shape {} is not a tuple", text_val(&sh))) };
    for e in es {
        let Val::Int(x) = e else { return Some(format!("TensorType.shape {} holds {}, which is not an int", text_val(&sh), text_val(e))) };
        if *x < 0 || *x > i64::MAX as i128 { return Some(format!("TensorType.shape {} has a negative extent", text_val(&sh))) }
    }
    None
}
fn cls(n: &str) -> Val { Val::Rec("$class".into(), vec![("name".into(), Val::Str(n.into()))]) }

struct Ev<'a> { c: &'a Ctx, ps: &'a Progs }
impl Ev<'_> {
    fn class_of(&self, h: u32) -> R<usize> { self.c.nodes.get(h as usize).map(|n| n.class).ok_or(format!("node handle {h} is not in this context")) }
    fn ttype(&self, v: &Val) -> R<Val> {
        match v { Val::Node(h) => { let n = self.c.nodes.get(*h as usize).ok_or("node handle is not in this context")?; Ok(ttype_val(&n.dtype, &n.shape, n.tag.as_deref())) } t @ Val::Rec(..) => Ok(t.clone()), _ => Err("not a node".into()) }
    }
    fn rep(&self, v: &Val) -> String {
        match v { Val::Node(h) => format!("<{} #{h}>", self.class_of(*h).map_or("Node", |c| &self.c.table.classes[c].name)), x => text_val(x) }
    }
    fn is_a(&self, x: &Val, k: &Val) -> R<bool> {
        let Val::Rec(_, f) = k else { return Err("isinstance: not a class".into()) };
        let Some((_, Val::Str(k))) = f.first() else { return Err("isinstance: not a class".into()) };
        Ok(match x { Val::Node(h) => k == "Node" || self.c.table.is_a(self.class_of(*h)?, k), Val::Rec(n, _) => n == k, _ => false })
    }
    fn eval(&self, e: &E, env: &[Option<Val>]) -> R<Val> {
        Ok(match e {
            E::Var(i) => env[*i].clone().ok_or("a name is used before it is set")?,
            E::Const(i) => self.ps.consts[*i].1.clone(),
            E::Class(n) => cls(n),
            E::Int(i) => Val::Int(*i), E::Str(s) => Val::Str(s.clone()), E::Nil => Val::None, E::True => Val::Bool(true), E::False => Val::Bool(false),
            E::Tuple(v) => Val::Tuple(v.iter().map(|x| self.eval(x, env)).collect::<R<_>>()?),
            E::Attr(a, n) => match self.eval(a, env)? {
                Val::Rec(_, fs) => fs.into_iter().find(|f| f.0 == *n).map(|f| f.1).ok_or(format!("no attribute {n}"))?,
                v @ Val::Node(_) if n == "ttype" => self.ttype(&v)?,
                Val::Node(h) => self.c.field(h, n).map_err(|e| format!("{e:?}"))?,
                v => return Err(format!("{} has no attribute {n}", text_val(&v))),
            },
            E::Index(a, i) => { let (v, i) = (self.eval(a, env)?, int(&self.eval(i, env)?)?); seq(&v)?.get(i as usize).cloned().ok_or("index out of range")? }
            E::Not(a) => Val::Bool(!truthy(&self.eval(a, env)?)),
            E::And(a, b) => Val::Bool(truthy(&self.eval(a, env)?) && truthy(&self.eval(b, env)?)),
            E::Or(a, b) => Val::Bool(truthy(&self.eval(a, env)?) || truthy(&self.eval(b, env)?)),
            E::Cmp(a, op, b) => {
                let (l, r) = (self.eval(a, env)?, self.eval(b, env)?);
                Val::Bool(match op {
                    Cmp::Eq => veq(&l, &r), Cmp::Ne => !veq(&l, &r),
                    Cmp::Lt | Cmp::Gt => { let (x, y) = (num(&l).ok_or("comparison of a non-number")?, num(&r).ok_or("comparison of a non-number")?); if *op == Cmp::Lt { x < y } else { x > y } }
                    Cmp::In | Cmp::NotIn => (match &r { Val::Str(s) => matches!(&l, Val::Str(x) if s.contains(x.as_str())), t => seq(t)?.iter().any(|x| veq(x, &l)) }) == (*op == Cmp::In),
                    Cmp::Is | Cmp::IsNot => (matches!(&l, Val::None) && matches!(&r, Val::None) || veq(&l, &r) && !matches!(&l, Val::None)) == (*op == Cmp::Is),
                })
            }
            E::Call(n, a) => self.call(n, a, env)?,
        })
    }
    fn call(&self, n: &str, a: &[E], env: &[Option<Val>]) -> R<Val> {
        if n == "coalesce" || n == "orelse" {
            let x = self.eval(&a[0], env)?;
            return if (n == "coalesce" && !matches!(x, Val::None)) || (n == "orelse" && truthy(&x)) { Ok(x) } else { self.eval(&a[1], env) };
        }
        let v = a.iter().map(|x| self.eval(x, env)).collect::<R<Vec<_>>>()?;
        let nodes = |x: &Val| -> R<Vec<u32>> { seq(x)?.iter().map(|y| if let Val::Node(h) = y { Ok(*h) } else { Err("not a node".to_string()) }).collect() };
        Ok(match n {
            "len" => Val::Int(match &v[0] { Val::Str(s) => s.chars().count(), x => seq(x)?.len() } as i128),
            "isinstance" => Val::Bool(self.is_a(&v[0], &v[1])?),
            "tt" => { let Val::Str(d) = &v[0] else { return Err("tt: a dtype".into()) }; ttype_val(d, &[], None) }
            "width_of" => { let mut w = 0i128; for h in nodes(&v[0])? { let Val::Tuple(s) = self.field_of(&self.ttype(&Val::Node(h))?, "shape")? else { return Err("shape".into()) }; w += s.iter().map(|d| int(d).unwrap_or(1)).product::<i128>() } Val::Int(w) }
            "rem" => { let (x, y) = (int(&v[0])?, int(&v[1])?); if y == 0 { return Err("integer modulo by zero".into()) } let r = x % y; Val::Int(if r != 0 && (r < 0) != (y < 0) { r + y } else { r }) }
            "int" => Val::Int(int(&v[0])?),
            "tuple" => Val::Tuple(seq(&v[0])?.to_vec()),
            "cat" => Val::Tuple(v.iter().map(|x| seq(x).map(|s| s.to_vec())).collect::<R<Vec<_>>>()?.concat()),
            "opt" => if matches!(v[0], Val::None) { Val::Tuple(vec![]) } else { Val::Tuple(vec![v[0].clone()]) },
            "head" => { let Val::Str(t) = &v[0] else { return Err("head: a string".into()) }; Val::Str(t.split('_').next().unwrap_or("").into()) }
            "first_ne_ttype" => { let hs = nodes(&v[0])?; let first = self.ttype(&Val::Node(*hs.first().ok_or("empty")?))?; let mut r = Val::None; for h in &hs[1..] { if !veq(&self.ttype(&Val::Node(*h))?, &first) { r = Val::Node(*h); break } } r }
            "first_not_node" => seq(&v[0])?.iter().find(|x| !matches!(x, Val::Node(_))).cloned().unwrap_or(Val::None),
            "name_of" => match &v[0] { Val::Rec(_, f) => f[0].1.clone(), _ => return Err("name_of: a class".into()) },
            "bool" => Val::Bool(truthy(&v[0])),
            "trip" => { let (a, b, c) = (int(&v[0])?, int(&v[1])?, int(&v[2])?); Val::Int(if c > 0 { if b > a { (b - a + c - 1) / c } else { 0 } } else if c < 0 { if a > b { (a - b - c - 1) / (-c) } else { 0 } } else { 0 }) }
            "first_mismatch" => { let (x, y) = (nodes(&v[0])?, nodes(&v[1])?); Val::Int(x.iter().zip(&y).position(|(p, q)| self.differ(*p, *q)).map_or(-1, |j| j as i128)) }
            "first_not_instance" => { let mut at = -1; for (j, x) in seq(&v[0])?.iter().enumerate() { if !self.is_a(x, &v[1])? { at = j as i128; break } } Val::Int(at) }
            "carry_fault" => {
                let (c, i, x) = (nodes(&v[0])?, nodes(&v[1])?, nodes(&v[2])?);
                for j in 0..c.len() {
                    let code = if self.differ(c[j], i[j]) { 0 } else if self.differ(c[j], x[j]) { 1 } else if self.c.nodes[c[j] as usize].shape.len() > 2 { 2 } else { continue };
                    return Ok(Val::Tuple(vec![Val::Int(j as i128), Val::Int(code)]));
                }
                Val::None
            }
            "ttype_fault" => ttype_fault(&v[0]).map_or(Val::None, Val::Str),
            "literal_fault" => literal_fault(&v[0]).map_or(Val::None, Val::Str),
            _ => return Err(format!("unknown function {n}")),
        })
    }
    fn field_of(&self, t: &Val, k: &str) -> R<Val> { if let Val::Rec(_, f) = t { f.iter().find(|x| x.0 == k).map(|x| x.1.clone()).ok_or(format!("no {k}")) } else { Err("not a record".into()) } }
    fn differ(&self, a: u32, b: u32) -> bool { let (x, y) = (&self.c.nodes[a as usize], &self.c.nodes[b as usize]); (&x.dtype, &x.shape, &x.tag) != (&y.dtype, &y.shape, &y.tag) }

    fn run(&self, p: &Prog, env: &mut Vec<Option<Val>>) -> R<()> {
        for st in &p.steps {
            match st {
                Step::Let(i, e) => env[*i] = Some(self.eval(e, env)?),
                Step::Check(c, t) => if truthy(&self.eval(c, env)?) {
                    let mut m = String::new();
                    for pc in t { match pc { Piece::Lit(x) => m += x, Piece::Val(e, r) => { let v = self.eval(e, env)?; m += &match (&v, r) { (Val::Str(s), false) => s.clone(), (v, _) => self.rep(v) } } } }
                    return Err(format!("\u{1}{m}"));
                },
                Step::Node(tt, ops) => {
                    let (ttype, o) = (self.eval(tt, env)?, self.eval(ops, env)?);
                    let ops = Val::Tuple(seq(&o)?.to_vec());
                    let base = &self.ps.named[self.ps.named_idx["node_base"]];
                    let mut e2 = vec![None; base.nvars];
                    (e2[0], e2[1], e2[2]) = (env[p.params].clone(), Some(ttype.clone()), Some(ops.clone()));
                    self.run(base, &mut e2)?;
                    (env[p.params + 1], env[p.params + 2]) = (Some(ttype), Some(ops));
                }
                Step::Call(ix, args) => {
                    let c = &self.ps.named[*ix];
                    let mut e2 = vec![None; c.nvars];
                    for (k, a) in args.iter().enumerate() { e2[k] = Some(self.eval(a, env)?) }
                    e2[c.params] = env[p.params].clone();
                    self.run(c, &mut e2)?;
                    for off in [1, 2] { if e2[c.params + off].is_some() { env[p.params + off] = e2[c.params + off].clone() } }
                }
            }
        }
        Ok(())
    }

    /// The constructor parameters of a record, by the class's `from` list.
    fn params(&self, node: &Node, p: &Prog, class: &crate::class::Class) -> R<Vec<Val>> {
        let slot = |k: &str| class.slots.iter().position(|s| s == k).map(|i| node.fields[i].clone()).ok_or(format!("{} has no slot {k:?}", class.name));
        let op = |n: usize| node.ops.get(n).map(|h| Val::Node(*h)).ok_or(format!("{}: operand {n} is missing", class.name));
        let ops = |n: usize| Val::Tuple(node.ops.iter().skip(n).map(|h| Val::Node(*h)).collect());
        let ncar = |s: &dyn Fn(&str) -> R<Val>| -> R<usize> { Ok(seq(&s("carries")?)?.len()) };
        let (mut out, mut pos) = (vec![Val::None; p.params], 0);
        for (named, t) in &p.from {
            let vals = match t.as_str() {
                "ttype" => vec![ttype_val(&node.dtype, &node.shape, node.tag.as_deref())],
                "operands" => vec![ops(0)],
                "*operands" => node.ops.iter().map(|h| Val::Node(*h)).collect(),
                "carry_inits" => { let n = ncar(&slot)?; vec![Val::Tuple(node.ops.iter().take(n).map(|h| Val::Node(*h)).collect())] }
                "free_ops" => {
                    let (n, end) = (ncar(&slot)?, int(&slot("_free_end")?)? as usize);
                    vec![Val::Tuple(node.ops.iter().take(end).skip(n).map(|h| Val::Node(*h)).collect())]
                }
                "count_op" => vec![if matches!(slot("count")?, Val::None) { Val::None } else { op(node.ops.len().wrapping_sub(1))? }],
                t if t.starts_with("opq:") => vec![node.ops.get(t[4..].parse::<usize>().map_err(|_| "bad token")?).map_or(Val::None, |h| Val::Node(*h))],
                t if t.starts_with("op:") => vec![op(t[3..].parse().map_err(|_| "bad token")?)?],
                t if t.starts_with("ops:") => vec![ops(t[4..].trim_end_matches(':').parse().map_err(|_| "bad token")?)],
                t if t.starts_with("edge:") => vec![slot(&t[5..])?],
                t if t.starts_with('@') => vec![slot(t[1..].split(':').next().unwrap())?],
                t => return Err(format!("{}: unknown token {t}", class.name)),
            };
            match named { Some(i) => out[*i] = vals.into_iter().next().unwrap_or(Val::None), None => for v in vals { if pos < out.len() { out[pos] = v } pos += 1 } }
        }
        if pos > p.params { return Err(format!("{}: {} operands given, the constructor takes {}", class.name, pos, p.params)) }
        Ok(out)
    }
}

fn quantity_check(ps: &Progs, f: &[(String, Val)]) -> R<()> {
    let get = |k: &str| f.iter().find(|x| x.0 == k).map_or(Val::None, |x| x.1.clone());
    let konst = |k: &str| ps.consts.iter().find(|c| c.0 == k).map_or(vec![], |c| seq(&c.1).unwrap().iter().filter_map(|v| if let Val::Str(s) = v { Some(s.clone()) } else { None }).collect::<Vec<_>>());
    let (slug, wires, pack) = (get("slug"), get("wires"), get("pack"));
    let sl = text_val(&slug);
    let packs = konst("QUANTITY_PACKS");
    if !matches!(&pack, Val::Str(p) if packs.contains(p)) {
        return Err(format!("Quantity {sl}: pack {} is not one of {}", text_val(&pack), text_val(&Val::Tuple(packs.into_iter().map(Val::Str).collect()))));
    }
    if !truthy(&slug) { return Err("Quantity: slug is required \u{2014} it is the namespace's first segment".into()) }
    let ws = seq(&wires)?;
    if ws.is_empty() { return Err(format!("Quantity {sl}: declares no wires")) }
    let roles: Vec<String> = konst("QUANTITY_ROLE_PAIRS").chunks(2).map(|p| p[0].clone()).collect();
    let mut seen: Vec<Val> = vec![];
    for w in ws {
        let Val::Rec(n, wf) = w else { return Err(format!("Quantity {sl}: {} is not a Wire", text_val(w))) };
        if n != "Wire" { return Err(format!("Quantity {sl}: {} is not a Wire", text_val(w))) }
        let g = |k: &str| wf.iter().find(|x| x.0 == k).map_or(Val::None, |x| x.1.clone());
        let (role, name) = (g("role"), g("name"));
        if !matches!(&role, Val::Str(r) if roles.contains(r)) {
            let mut s = roles.clone();
            s.sort();
            return Err(format!("Quantity {sl}: wire {} has role {}, which no leaf kind mints; declared read roles are [{}]", text_val(&name), text_val(&role), s.iter().map(|r| py_str(r)).collect::<Vec<_>>().join(", ")));
        }
        if seen.contains(&name) { return Err(format!("Quantity {sl}: duplicate wire name {}", text_val(&name))) }
        seen.push(name);
    }
    Ok(())
}

fn quantities(ps: &Progs, v: &Val) -> R<()> {
    match v {
        Val::Rec(n, f) if n == "Quantity" => quantity_check(ps, f),
        Val::Rec(_, f) => f.iter().try_for_each(|x| quantities(ps, &x.1)),
        Val::Tuple(t) | Val::List(t) => t.iter().try_for_each(|x| quantities(ps, x)),
        _ => Ok(()),
    }
}

/// The one call site of the checks: every make passes its record here before it is appended. A class that is not a core class is not checked here.
pub fn check(ctx: &Ctx, node: &Node) -> Result<(), CtorFault> {
    // a refusal of a declared check carries a marker; any other failure is the evaluator's, and is never a refusal text
    match check_raw(ctx, node) {
        Err(e) => match e.strip_prefix('\u{1}') {
            Some(m) => Err(CtorFault { declared: true, text: m.to_string() }),
            None if e.contains("does not match its constructor") => Err(CtorFault { declared: false, text: e }),
            None => Err(CtorFault { declared: false, text: format!("{}: its state cannot be read by its constructor ({e})", ctx.table.classes[node.class].name) }),
        },
        Ok(()) => Ok(()),
    }
}

/// A failure of the constructor checks: a declared check that refused (its text is final), or the program could not run or its result
/// differs from the record (the rule of the kind has the better text).
pub struct CtorFault { pub declared: bool, pub text: String }

fn check_raw(ctx: &Ctx, node: &Node) -> Result<(), String> {
    let class = &ctx.table.classes[node.class];
    if class.spec.is_none() { return Ok(()) }
    let ps = &ctx.progs;
    node.fields.iter().try_for_each(|f| quantities(ps, f)).map_err(|e| format!("\u{1}{e}"))?;
    let ev = Ev { c: ctx, ps };
    let ttype = ttype_val(&node.dtype, &node.shape, node.tag.as_deref());
    let ops = Val::Tuple(node.ops.iter().map(|h| Val::Node(*h)).collect());
    let Some(&pi) = ps.by_class.get(&class.name) else {
        let base = &ps.named[ps.named_idx["node_base"]];
        let mut env = vec![None; base.nvars];
        (env[0], env[1], env[2]) = (Some(cls(&class.name)), Some(ttype), Some(ops));
        return ev.run(base, &mut env);
    };
    let p = &ps.progs[pi];
    let mut env = vec![None; p.nvars];
    for (i, v) in ev.params(node, p, class)?.into_iter().enumerate() { env[i] = Some(v) }
    env[p.params] = Some(cls(&class.name));
    ev.run(p, &mut env)?;
    for (k, (slot, e)) in p.slots.iter().zip(std::iter::once(None).chain(std::iter::once(None)).chain(p.out.iter().map(Some))).enumerate() {
        let (got, want) = match (k, e) {
            (0, _) => (ttype.clone(), env[p.params + 1].clone().unwrap_or(Val::None)),
            (1, _) => (ops.clone(), env[p.params + 2].clone().unwrap_or(Val::None)),
            (_, e) => {
                let Some(i) = class.slots.iter().position(|s| s == slot) else { continue };
                (node.fields[i].clone(), ev.eval(e.unwrap(), &env)?)
            }
        };
        if text_val(&got) != text_val(&want) { return Err(format!("{}: {slot} does not match its constructor (got {}, constructor gives {})", class.name, text_val(&got), text_val(&want))) }
    }
    Ok(())
}
