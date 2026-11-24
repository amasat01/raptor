//! The declared constructor checks over records. The programs of the manifest are compiled once per context; a node's state is mapped
//! back to constructor parameters, the program runs forward over them, and the state it produces must equal the record's state.
use crate::record::{Ctx, Node};
use crate::value::{text_val, ttype_val, Val};
use rcore::json::{self, J};
use rcore::keytext::py_str;
use std::collections::HashMap;

/// A comparison operator of the check language.
#[derive(Clone, Copy, PartialEq)]
enum Cmp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Greater than.
    Gt,
    /// Membership.
    In,
    /// Non-membership.
    NotIn,
    /// Identity (both None, or equal and not None).
    Is,
    /// Negated identity.
    IsNot,
}
/// An expression of the check language.
enum E {
    /// A program variable, by slot.
    Var(usize),
    /// A manifest constant, by position.
    Const(usize),
    /// A class, by name.
    Class(String),
    /// An integer literal.
    Int(i128),
    /// A text literal.
    Str(String),
    /// The value None.
    Nil,
    /// The flag true.
    True,
    /// The flag false.
    False,
    /// A tuple of expressions.
    Tuple(Vec<E>),
    /// An attribute of a value.
    Attr(Box<E>, String),
    /// An element of a sequence.
    Index(Box<E>, Box<E>),
    /// Negation.
    Not(Box<E>),
    /// Conjunction.
    And(Box<E>, Box<E>),
    /// Disjunction.
    Or(Box<E>, Box<E>),
    /// A comparison.
    Cmp(Box<E>, Cmp, Box<E>),
    /// A call of a built-in function.
    Call(String, Vec<E>),
}
/// A piece of a check's message: literal text or a value to print.
enum Piece {
    /// Literal text.
    Lit(String),
    /// A value to print, and whether it prints as text.
    Val(E, bool),
}
/// One statement of a program.
enum Step {
    /// Binds a variable to the value of an expression.
    Let(usize, E),
    /// Refuses with the message pieces when the condition holds.
    Check(E, Vec<Piece>),
    /// Asserts that the first node equals the second.
    Node(E, E),
    /// Calls a named check with arguments.
    Call(usize, Vec<E>),
}
/// A compiled program: parameters, statements, outputs and the slots it assigns.
struct Prog {
    /// The parameter names.
    pnames: Vec<String>,
    /// The default of each parameter, if any.
    pdef: Vec<Option<Val>>,
    /// The number of parameters.
    params: usize,
    /// The number of variables, parameters included.
    nvars: usize,
    /// The statements, in order.
    steps: Vec<Step>,
    /// The output expressions.
    out: Vec<E>,
    /// The slot each output is assigned to.
    slots: Vec<String>,
    /// The slot each parameter is read from: an output position, or None, with a name.
    from: Vec<(Option<usize>, String)>,
}
/// The compiled programs of one context.
pub struct Progs {
    /// The named checks.
    named: Vec<Prog>,
    /// The position of each named check, by name.
    named_idx: HashMap<String, usize>,
    /// The position of each class's constructor program, by class name.
    by_class: HashMap<String, usize>,
    /// The constructor programs.
    progs: Vec<Prog>,
    /// The manifest constants.
    consts: Vec<(String, Val)>,
}

/// The built-in functions of the check language.
const BUILTINS: &[&str] = &[
    "len",
    "isinstance",
    "tt",
    "width_of",
    "rem",
    "int",
    "tuple",
    "cat",
    "opt",
    "head",
    "first_ne_ttype",
    "first_not_node",
    "coalesce",
    "orelse",
    "name_of",
    "bool",
    "trip",
    "first_mismatch",
    "first_not_instance",
    "carry_fault",
    "index_fault",
    "sample_fault",
    "group_fault",
    "reduce_fault",
    "loop_group_fault",
    "function_fault",
    "call_fault",
    "call_result",
    "axis_fault",
    "axes_fault",
    "store_index_fault",
    "address_of",
    "expected_of",
    "ttype_fault",
    "literal_fault",
];
/// Variable names the language reserves.
const RESERVED: &[&str] = &["_cls", "_t", "_o"];

/// The parser of the check language: a cursor over the characters of one expression.
struct P<'a> {
    /// The characters of the text.
    c: Vec<char>,
    /// The cursor position.
    i: usize,
    /// The variable names in scope.
    vars: &'a [String],
    /// The manifest constants in scope.
    consts: &'a [(String, Val)],
    /// The class names in scope.
    classes: &'a [String],
}
/// A result that fails with a message.
type R<T> = Result<T, String>;
impl P<'_> {
    /// Skips white space.
    fn ws(&mut self) {
        while self.c.get(self.i) == Some(&' ') {
            self.i += 1
        }
    }
    /// Whether the next token is the word `w`.
    fn at(&self, w: &str) -> bool {
        w.chars()
            .enumerate()
            .all(|(k, ch)| self.c.get(self.i + k) == Some(&ch))
    }
    /// Takes the symbol `w` if it comes next.
    fn sym(&mut self, w: &str) -> bool {
        self.ws();
        if self.at(w) {
            self.i += w.chars().count();
            true
        } else {
            false
        }
    }
    /// Takes the keyword `w` if it comes next.
    fn kw(&mut self, w: &str) -> bool {
        self.ws();
        let n = w.chars().count();
        if self.at(w)
            && !self
                .c
                .get(self.i + n)
                .is_some_and(|x| x.is_alphanumeric() || *x == '_')
        {
            self.i += n;
            true
        } else {
            false
        }
    }
    /// Parses a disjunction.
    fn or(&mut self) -> R<E> {
        let mut l = self.and()?;
        while self.kw("or") {
            l = E::Or(l.into(), self.and()?.into())
        }
        Ok(l)
    }
    /// Parses a conjunction.
    fn and(&mut self) -> R<E> {
        let mut l = self.not()?;
        while self.kw("and") {
            l = E::And(l.into(), self.not()?.into())
        }
        Ok(l)
    }
    /// Parses a negation.
    fn not(&mut self) -> R<E> {
        if self.kw("not") {
            Ok(E::Not(self.not()?.into()))
        } else {
            self.cmp()
        }
    }
    /// Parses a comparison.
    fn cmp(&mut self) -> R<E> {
        let l = self.post()?;
        let op = if self.sym("==") {
            Cmp::Eq
        } else if self.sym("!=") {
            Cmp::Ne
        } else if self.sym("<") {
            Cmp::Lt
        } else if self.sym(">") {
            Cmp::Gt
        } else if self.kw("not") {
            if self.kw("in") {
                Cmp::NotIn
            } else {
                return Err("expected `in` after `not`".into());
            }
        } else if self.kw("in") {
            Cmp::In
        } else if self.kw("is") {
            if self.kw("not") {
                Cmp::IsNot
            } else {
                Cmp::Is
            }
        } else {
            return Ok(l);
        };
        Ok(E::Cmp(l.into(), op, self.post()?.into()))
    }
    /// Parses a postfix chain: attributes, indexes and calls.
    fn post(&mut self) -> R<E> {
        let mut e = self.atom()?;
        loop {
            if self.sym(".") {
                e = E::Attr(e.into(), self.name()?)
            } else if self.sym("[") {
                let i = self.or()?;
                if !self.sym("]") {
                    return Err("expected ]".into());
                }
                e = E::Index(e.into(), i.into())
            } else {
                return Ok(e);
            }
        }
    }
    /// Parses a name.
    fn name(&mut self) -> R<String> {
        self.ws();
        let st = self.i;
        while self
            .c
            .get(self.i)
            .is_some_and(|x| x.is_alphanumeric() || *x == '_')
        {
            self.i += 1
        }
        if st == self.i {
            return Err(format!("name expected at {}", self.i));
        }
        Ok(self.c[st..self.i].iter().collect())
    }
    /// Parses call arguments up to `close`; also reports whether a trailing comma was seen.
    fn args(&mut self, close: &str) -> R<(Vec<E>, bool)> {
        let (mut v, mut comma) = (vec![], false);
        loop {
            if self.sym(close) {
                return Ok((v, comma));
            }
            v.push(self.or()?);
            if self.sym(",") {
                comma = true
            } else if self.sym(close) {
                return Ok((v, comma));
            } else {
                return Err(format!("expected , or {close}"));
            }
        }
    }
    /// Parses an atom: a literal, a name, a tuple or a parenthesised expression.
    fn atom(&mut self) -> R<E> {
        self.ws();
        match self.c.get(self.i).copied() {
            Some('(') => {
                self.i += 1;
                let (mut v, comma) = self.args(")")?;
                if v.len() == 1 && !comma {
                    Ok(v.pop().unwrap())
                } else {
                    Ok(E::Tuple(v))
                }
            }
            Some('\'') => {
                self.i += 1;
                let st = self.i;
                while self.c.get(self.i).is_some_and(|x| *x != '\'') {
                    self.i += 1
                }
                self.i += 1;
                Ok(E::Str(self.c[st..self.i - 1].iter().collect()))
            }
            Some(c) if c.is_ascii_digit() => {
                let st = self.i;
                while self.c.get(self.i).is_some_and(|x| x.is_ascii_digit()) {
                    self.i += 1
                }
                self.c[st..self.i]
                    .iter()
                    .collect::<String>()
                    .parse()
                    .map(E::Int)
                    .map_err(|_| "bad integer".to_string())
            }
            _ => {
                let n = self.name()?;
                match n.as_str() {
                    "None" => return Ok(E::Nil),
                    "True" => return Ok(E::True),
                    "False" => return Ok(E::False),
                    _ => {}
                }
                if self.sym("(") {
                    if !BUILTINS.contains(&n.as_str()) {
                        return Err(format!("unknown function {n}"));
                    }
                    return Ok(E::Call(n, self.args(")")?.0));
                }
                if let Some(i) = self.vars.iter().position(|v| *v == n) {
                    Ok(E::Var(i))
                } else if let Some(i) = self.consts.iter().position(|c| c.0 == n) {
                    Ok(E::Const(i))
                } else if n == "TensorType" || n == "Node" || self.classes.contains(&n) {
                    Ok(E::Class(n))
                } else {
                    Err(format!("unknown name {n}"))
                }
            }
        }
    }
}

/// The names a template or expression may use.
struct Scope<'a> {
    /// The variables in scope.
    vars: Vec<String>,
    /// The manifest constants.
    consts: &'a [(String, Val)],
    /// The class names.
    classes: &'a [String],
}
impl Scope<'_> {
    /// Compiles the expression text `t`.
    fn expr(&self, t: &str) -> R<E> {
        let mut p = P {
            c: t.chars().collect(),
            i: 0,
            vars: &self.vars,
            consts: self.consts,
            classes: self.classes,
        };
        let e = p.or()?;
        p.ws();
        if p.i != p.c.len() {
            return Err(format!("trailing text in `{t}`"));
        }
        Ok(e)
    }
    /// Compiles a message template into pieces.
    fn tmpl(&self, t: &str) -> R<Vec<Piece>> {
        let (mut out, mut lit, c, mut i) =
            (vec![], String::new(), t.chars().collect::<Vec<_>>(), 0);
        while i < c.len() {
            if c[i] != '{' {
                lit.push(c[i]);
                i += 1;
                continue;
            }
            let (mut j, mut d) = (i + 1, 1);
            while j < c.len() {
                if c[j] == '{' {
                    d += 1
                }
                if c[j] == '}' {
                    d -= 1;
                    if d == 0 {
                        break;
                    }
                }
                j += 1
            }
            let mut inner: String = c[i + 1..j.min(c.len())].iter().collect();
            let r = inner.ends_with("!r");
            if r {
                inner.truncate(inner.len() - 2)
            }
            if !lit.is_empty() {
                out.push(Piece::Lit(std::mem::take(&mut lit)))
            }
            out.push(Piece::Val(self.expr(&inner)?, r));
            i = j + 1;
        }
        if !lit.is_empty() {
            out.push(Piece::Lit(lit))
        }
        Ok(out)
    }
}

/// The parameter names of a parameter text.
fn names(p: &str) -> Vec<String> {
    p.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty() && *x != "*")
        .map(|x| {
            x.split('=')
                .next()
                .unwrap()
                .split(':')
                .next()
                .unwrap()
                .trim()
                .to_string()
        })
        .collect()
}

/// The default of each constructor parameter, in order (None: the parameter is required).
fn defaults(p: &str) -> Vec<Option<Val>> {
    let (mut parts, mut cur, mut depth, mut quote) = (vec![], String::new(), 0i32, None::<char>);
    for ch in p.chars() {
        match quote {
            Some(q) => {
                cur.push(ch);
                if ch == q {
                    quote = None
                }
            }
            None => match ch {
                '\'' | '"' => {
                    quote = Some(ch);
                    cur.push(ch)
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    cur.push(ch)
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    cur.push(ch)
                }
                ',' if depth == 0 => parts.push(std::mem::take(&mut cur)),
                _ => cur.push(ch),
            },
        }
    }
    parts.push(cur);
    parts
        .iter()
        .map(|x| x.trim())
        .filter(|x| !x.is_empty() && *x != "*")
        .map(|x| {
            let mut depth = 0i32;
            let eq = x.char_indices().find(|&(_, ch)| {
                match ch {
                    '[' | '(' => depth += 1,
                    ']' | ')' => depth -= 1,
                    _ => {}
                }
                ch == '=' && depth == 0
            });
            eq.and_then(|(i, _)| crate::value::parse(x[i + 1..].trim()).ok())
        })
        .collect()
}

/// What a program's steps may refer to besides its own variables: the named checks, the constants and the class names.
struct Known<'a> {
    /// Each named check's position and parameter count, by name.
    named: &'a HashMap<String, (usize, usize)>,
    /// The manifest's constants, by name.
    consts: &'a [(String, Val)],
    /// The class names of the manifest.
    classes: &'a [String],
}

/// The parameters of a program and their defaults (None for a parameter without one).
struct Params {
    /// The parameter names, in order.
    names: Vec<String>,
    /// The default of each parameter, in the same order.
    defaults: Vec<Option<Val>>,
}

/// Compiles one program from its manifest rows.
fn prog(
    who: &str,
    given: Params,
    steps: &[J],
    out: &[String],
    slots: Vec<String>,
    from: &[String],
    known: &Known,
) -> R<Prog> {
    let Params {
        names: params,
        defaults: pdef,
    } = given;
    let (named, consts, classes) = (known.named, known.consts, known.classes);
    let mut sc = Scope {
        vars: params.clone(),
        consts,
        classes,
    };
    sc.vars.extend(RESERVED.iter().map(|s| s.to_string()));
    let mut cs = vec![];
    for st in steps {
        let a = st.arr();
        let t = |i: usize| a.get(i).and_then(J::str).unwrap_or("");
        let w = |e: String| format!("{who}: {e}");
        cs.push(match t(0) {
            "let" => {
                let e = sc.expr(t(2)).map_err(w)?;
                sc.vars.push(t(1).to_string());
                Step::Let(sc.vars.len() - 1, e)
            }
            "check" => Step::Check(sc.expr(t(1)).map_err(w)?, sc.tmpl(t(2)).map_err(w)?),
            "node" => Step::Node(sc.expr(t(1)).map_err(w)?, sc.expr(t(2)).map_err(w)?),
            "call" => {
                let args = (2..a.len())
                    .map(|i| sc.expr(t(i)))
                    .collect::<R<Vec<_>>>()
                    .map_err(w)?;
                match named.get(t(1)) {
                    Some((ix, np)) if *np == args.len() => Step::Call(*ix, args),
                    _ => return Err(format!("{who}: bad call of check {}", t(1))),
                }
            }
            x => return Err(format!("{who}: unknown step {x}")),
        });
    }
    let outs = out
        .iter()
        .map(|e| sc.expr(e))
        .collect::<R<Vec<_>>>()
        .map_err(|e| format!("{who}: {e}"))?;
    let from = from
        .iter()
        .map(|t| match t.split_once('=') {
            Some((k, v))
                if !(t.starts_with('@') || t.starts_with("edge") || t.starts_with("op")) =>
            {
                (params.iter().position(|p| p == k), v.to_string())
            }
            _ => (None, t.to_string()),
        })
        .collect();
    Ok(Prog {
        pnames: params.clone(),
        pdef,
        params: params.len(),
        nvars: sc.vars.len(),
        steps: cs,
        out: outs,
        slots,
        from,
    })
}

impl Progs {
    /// Compiles every program of the manifest.
    pub fn compile() -> Result<Progs, String> {
        let m = json::parse(rcore::MANIFEST)?;
        let consts: Vec<(String, Val)> = m
            .get("consts")
            .map_or(&[][..], J::obj)
            .iter()
            .filter(|(k, v)| matches!(v, J::Arr(_)) && !k.contains('.'))
            .map(|(k, v)| {
                (
                    k.clone(),
                    Val::Tuple(v.strs().into_iter().map(Val::Str).collect()),
                )
            })
            .collect();
        let kinds = m.get("kinds").ok_or("no kinds")?.arr();
        let classes: Vec<String> = kinds
            .iter()
            .filter_map(|k| k.get("class").and_then(J::str).map(String::from))
            .collect();
        let chk = m.get("checks").map_or(&[][..], J::obj);
        let named_map: HashMap<String, (usize, usize)> = chk
            .iter()
            .enumerate()
            .map(|(i, (k, v))| (k.clone(), (i, v.get("params").map_or(0, |p| p.arr().len()))))
            .collect();
        let known = Known {
            named: &named_map,
            consts: &consts,
            classes: &classes,
        };
        let mut named = vec![];
        for (k, v) in chk {
            named.push(prog(
                k,
                Params {
                    names: v.get("params").map_or(vec![], J::strs),
                    defaults: vec![None; v.get("params").map_or(0, |p| p.arr().len())],
                },
                v.get("steps").map_or(&[][..], J::arr),
                &[],
                vec![],
                &[],
                &known,
            )?)
        }
        let (mut progs, mut by_class) = (vec![], HashMap::new());
        for k in kinds {
            let Some(c) = k.get("ctor") else { continue };
            let cls = k
                .get("class")
                .and_then(J::str)
                .ok_or("a kind without a class")?
                .to_string();
            let mut slots = vec!["ttype".to_string(), "operands".to_string()];
            slots.extend(k.get("assigns").map_or(vec![], J::strs));
            let from = k.get("from").map_or(vec![], J::strs);
            by_class.insert(cls.clone(), progs.len());
            progs.push(prog(
                &cls,
                Params {
                    names: names(k.get("params").and_then(J::str).unwrap_or("")),
                    defaults: defaults(k.get("params").and_then(J::str).unwrap_or("")),
                },
                c.get("steps").map_or(&[][..], J::arr),
                &c.get("out").map_or(vec![], J::strs),
                slots,
                &from,
                &known,
            )?);
        }
        let named_idx = chk
            .iter()
            .enumerate()
            .map(|(i, (k, _))| (k.clone(), i))
            .collect::<HashMap<_, _>>();
        if !named_idx.contains_key("node_base") {
            return Err("the manifest declares no `node_base` check".into());
        }
        Ok(Progs {
            named,
            named_idx,
            by_class,
            progs,
            consts,
        })
    }
}

/// The number a value holds, if it is numeric.
fn num(v: &Val) -> Option<f64> {
    match v {
        Val::Int(i) => Some(*i as f64),
        Val::Bool(b) => Some(*b as u8 as f64),
        Val::Float(f) => Some(*f),
        _ => None,
    }
}
/// Value equality: equal numbers, texts, flags, None and nodes; sequences and records element by element.
fn veq(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Tuple(x), Val::Tuple(y)) | (Val::List(x), Val::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| veq(p, q))
        }
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => x == y,
            _ => text_val(a) == text_val(b),
        },
    }
}
/// Whether a value is true.
fn truthy(v: &Val) -> bool {
    match v {
        Val::None => false,
        Val::Bool(b) => *b,
        Val::Int(i) => *i != 0,
        Val::Float(f) => *f != 0.0,
        Val::Str(s) => !s.is_empty(),
        Val::Tuple(t) | Val::List(t) => !t.is_empty(),
        _ => true,
    }
}
/// The operands of an `Update` after its value, split into the expected value (a `compare_exchange` takes one right after the
/// value) and the index list.
fn update_operands<'a>(op: &Val, rest: &'a Val) -> R<(&'a [Val], &'a [Val])> {
    let rest = seq(rest)?;
    let taken =
        usize::from(matches!(op, Val::Str(s) if s == "compare_exchange") && !rest.is_empty());
    Ok(rest.split_at(taken))
}
/// The elements of a tuple or list value; refuses another value.
fn seq(v: &Val) -> R<&[Val]> {
    match v {
        Val::Tuple(t) | Val::List(t) => Ok(t),
        _ => Err(format!("{} is not a sequence", text_val(v))),
    }
}
/// The integer a value holds; refuses another value.
fn int(v: &Val) -> R<i128> {
    match v {
        Val::Int(i) => Ok(*i),
        Val::Bool(b) => Ok(*b as i128),
        Val::Float(f) => Ok(*f as i128),
        _ => Err(format!("{} is not an integer", text_val(v))),
    }
}
/// Whether a value holds a list.
fn has_list(v: &Val) -> bool {
    match v {
        Val::List(_) => true,
        Val::Tuple(t) => t.iter().any(has_list),
        _ => false,
    }
}
/// Whether a value is plain data: no list inside.
fn plain(v: &Val) -> bool {
    match v {
        Val::Node(_) => false,
        Val::Rec(n, _) => n == "RawBlock",
        Val::Tuple(t) | Val::List(t) => t.iter().all(plain),
        _ => true,
    }
}
/// The reason a literal is invalid, if it is.
fn literal_fault(v: &Val) -> Option<String> {
    if has_list(v) {
        return Some(
            "a list is not a literal (it can change after the node is built); use a tuple".into(),
        );
    }
    if plain(v) {
        None
    } else {
        Some(format!("{} is not a plain literal", text_val(v)))
    }
}
/// Why `axes` is not an axis list: a tuple of at most eight (label, extent) pairs, each label a distinct non-empty text and each
/// extent a positive whole number or none (an extent known at run time).
pub(crate) fn axes_fault(axes: &Val) -> Option<String> {
    let Val::Tuple(list) = axes else {
        return Some("axes is a tuple of (label, extent) pairs".into());
    };
    if list.len() > 8 {
        return Some(format!("a plane has at most 8 axes, got {}", list.len()));
    }
    let mut seen: Vec<&str> = vec![];
    for axis in list {
        let Val::Tuple(pair) = axis else {
            return Some("an axis is a (label, extent) pair".into());
        };
        match pair.as_slice() {
            [Val::Str(label), extent] if !label.is_empty() => {
                if seen.contains(&label.as_str()) {
                    return Some(format!("the axis label {label:?} is given twice"));
                }
                seen.push(label);
                if !matches!(extent, Val::None) && !matches!(extent, Val::Int(n) if *n >= 1) {
                    return Some(format!(
                        "axis {label:?}: the extent is a positive whole number or none, got {}",
                        text_val(extent)
                    ));
                }
            }
            _ => return Some("an axis is a (label, extent) pair with a non-empty label".into()),
        }
    }
    None
}

/// The reason a type value is invalid, if it is.
fn ttype_fault(t: &Val) -> Option<String> {
    let Val::Rec(_, fs) = t else {
        return Some("not a TensorType".into());
    };
    let get = |k: &str| {
        fs.iter()
            .find(|f| f.0 == k)
            .map_or(Val::None, |f| f.1.clone())
    };
    let dt = get("dtype");
    if !matches!(&dt, Val::Str(d) if rcore::facts::DTYPES.contains(&d.as_str())) {
        return Some(format!(
            "TensorType.dtype {} is not one of {}",
            text_val(&dt),
            rcore::repr_tuple_str(&rcore::facts::DTYPES)
        ));
    }
    let sh = get("shape");
    let Val::Tuple(es) = &sh else {
        return Some(format!("TensorType.shape {} is not a tuple", text_val(&sh)));
    };
    for e in es {
        let Val::Int(x) = e else {
            return Some(format!(
                "TensorType.shape {} holds {}, which is not an int",
                text_val(&sh),
                text_val(e)
            ));
        };
        if *x < 0 || *x > i64::MAX as i128 {
            return Some(format!(
                "TensorType.shape {} has a negative extent",
                text_val(&sh)
            ));
        }
    }
    None
}
/// A class value from its name.
fn cls(n: &str) -> Val {
    Val::Rec("$class".into(), vec![("name".into(), Val::Str(n.into()))])
}

/// The evaluator: a context and its compiled programs.
struct Ev<'a> {
    /// The context whose nodes the programs read.
    c: &'a Ctx,
    /// The compiled programs.
    ps: &'a Progs,
}
/// The handles in a sequence of node values; refuses an element that is not a node.
fn node_list(x: &Val) -> R<Vec<u32>> {
    seq(x)?
        .iter()
        .map(|y| {
            if let Val::Node(h) = y {
                Ok(*h)
            } else {
                Err("not a node".to_string())
            }
        })
        .collect()
}

impl Ev<'_> {
    /// The class of node `h`; refuses an unknown handle.
    fn class_of(&self, h: u32) -> R<usize> {
        self.c
            .nodes
            .get(h as usize)
            .map(|n| n.class)
            .ok_or(format!("node handle {h} is not in this context"))
    }
    /// The handles in a sequence of node values, each of them a node of this context.
    fn live_nodes(&self, x: &Val) -> R<Vec<u32>> {
        let hs = node_list(x)?;
        hs.iter().try_for_each(|&h| self.class_of(h).map(|_| ()))?;
        Ok(hs)
    }
    /// The type value of a node, or of a type value.
    fn ttype(&self, v: &Val) -> R<Val> {
        match v {
            Val::Node(h) => {
                let n = self
                    .c
                    .nodes
                    .get(*h as usize)
                    .ok_or("node handle is not in this context")?;
                Ok(ttype_val(&n.dtype, &n.shape, n.tag.as_deref()))
            }
            t @ Val::Rec(..) => Ok(t.clone()),
            _ => Err("not a node".into()),
        }
    }
    /// A value as text for messages.
    fn rep(&self, v: &Val) -> String {
        match v {
            Val::Node(h) => format!(
                "<{} #{h}>",
                self.class_of(*h)
                    .map_or("Node", |c| &self.c.table.classes[c].name)
            ),
            x => text_val(x),
        }
    }
    /// Whether a value is an instance of a class value.
    fn is_a(&self, x: &Val, k: &Val) -> R<bool> {
        let Val::Rec(_, f) = k else {
            return Err("isinstance: not a class".into());
        };
        let Some((_, Val::Str(k))) = f.first() else {
            return Err("isinstance: not a class".into());
        };
        Ok(match x {
            Val::Node(h) => k == "Node" || self.c.table.is_a(self.class_of(*h)?, k),
            Val::Rec(n, _) => n == k,
            _ => false,
        })
    }
    /// Evaluates an expression over the variable values.
    fn eval(&self, e: &E, env: &[Option<Val>]) -> R<Val> {
        Ok(match e {
            E::Var(i) => env[*i].clone().ok_or("a name is used before it is set")?,
            E::Const(i) => self.ps.consts[*i].1.clone(),
            E::Class(n) => cls(n),
            E::Int(i) => Val::Int(*i),
            E::Str(s) => Val::Str(s.clone()),
            E::Nil => Val::None,
            E::True => Val::Bool(true),
            E::False => Val::Bool(false),
            E::Tuple(v) => Val::Tuple(v.iter().map(|x| self.eval(x, env)).collect::<R<_>>()?),
            E::Attr(a, n) => match self.eval(a, env)? {
                Val::Rec(_, fs) => fs
                    .into_iter()
                    .find(|f| f.0 == *n)
                    .map(|f| f.1)
                    .ok_or(format!("no attribute {n}"))?,
                v @ Val::Node(_) if n == "ttype" => self.ttype(&v)?,
                Val::Node(h) => self.c.field(h, n).map_err(|e| format!("{e:?}"))?,
                v => return Err(format!("{} has no attribute {n}", text_val(&v))),
            },
            E::Index(a, i) => {
                let (v, i) = (self.eval(a, env)?, int(&self.eval(i, env)?)?);
                seq(&v)?
                    .get(i as usize)
                    .cloned()
                    .ok_or("index out of range")?
            }
            E::Not(a) => Val::Bool(!truthy(&self.eval(a, env)?)),
            E::And(a, b) => Val::Bool(truthy(&self.eval(a, env)?) && truthy(&self.eval(b, env)?)),
            E::Or(a, b) => Val::Bool(truthy(&self.eval(a, env)?) || truthy(&self.eval(b, env)?)),
            E::Cmp(a, op, b) => {
                let (l, r) = (self.eval(a, env)?, self.eval(b, env)?);
                Val::Bool(match op {
                    Cmp::Eq => veq(&l, &r),
                    Cmp::Ne => !veq(&l, &r),
                    Cmp::Lt | Cmp::Gt => {
                        let (x, y) = (
                            num(&l).ok_or("comparison of a non-number")?,
                            num(&r).ok_or("comparison of a non-number")?,
                        );
                        if *op == Cmp::Lt {
                            x < y
                        } else {
                            x > y
                        }
                    }
                    Cmp::In | Cmp::NotIn => {
                        (match &r {
                            Val::Str(s) => matches!(&l, Val::Str(x) if s.contains(x.as_str())),
                            t => seq(t)?.iter().any(|x| veq(x, &l)),
                        }) == (*op == Cmp::In)
                    }
                    Cmp::Is | Cmp::IsNot => {
                        (matches!(&l, Val::None) && matches!(&r, Val::None)
                            || veq(&l, &r) && !matches!(&l, Val::None))
                            == (*op == Cmp::Is)
                    }
                })
            }
            E::Call(n, a) => self.call(n, a, env)?,
        })
    }
    /// Calls a built-in function; `coalesce` and `orelse` evaluate their arguments lazily.
    fn call(&self, n: &str, a: &[E], env: &[Option<Val>]) -> R<Val> {
        if n == "coalesce" || n == "orelse" {
            let x = self.eval(&a[0], env)?;
            return if (n == "coalesce" && !matches!(x, Val::None)) || (n == "orelse" && truthy(&x))
            {
                Ok(x)
            } else {
                self.eval(&a[1], env)
            };
        }
        let v = a.iter().map(|x| self.eval(x, env)).collect::<R<Vec<_>>>()?;
        match n {
            "width_of" | "first_ne_ttype" | "first_mismatch" | "first_not_instance"
            | "carry_fault" | "index_fault" | "sample_fault" | "group_fault" | "reduce_fault"
            | "loop_group_fault" | "function_fault" | "call_fault" | "call_result"
            | "axis_fault" | "store_index_fault" => self.call_over_nodes(n, &v),
            _ => self.call_plain(n, &v),
        }
    }

    /// The built-in functions over numbers, text, sequences and types (everything but the functions over node lists).
    fn call_plain(&self, n: &str, v: &[Val]) -> R<Val> {
        Ok(match n {
            "len" => Val::Int(match &v[0] {
                Val::Str(s) => s.chars().count(),
                x => seq(x)?.len(),
            } as i128),
            "isinstance" => Val::Bool(self.is_a(&v[0], &v[1])?),
            "tt" => {
                let Val::Str(d) = &v[0] else {
                    return Err("tt: a dtype".into());
                };
                ttype_val(d, &[], None)
            }
            "rem" => {
                let (x, y) = (int(&v[0])?, int(&v[1])?);
                if y == 0 {
                    return Err("integer modulo by zero".into());
                }
                let r = x % y;
                Val::Int(if r != 0 && (r < 0) != (y < 0) {
                    r + y
                } else {
                    r
                })
            }
            "int" => Val::Int(int(&v[0])?),
            "tuple" => Val::Tuple(seq(&v[0])?.to_vec()),
            "cat" => Val::Tuple(
                v.iter()
                    .map(|x| seq(x).map(|s| s.to_vec()))
                    .collect::<R<Vec<_>>>()?
                    .concat(),
            ),
            "opt" => {
                if matches!(v[0], Val::None) {
                    Val::Tuple(vec![])
                } else {
                    Val::Tuple(vec![v[0].clone()])
                }
            }
            "address_of" => Val::Tuple(update_operands(&v[0], &v[1])?.1.to_vec()),
            "expected_of" => Val::Tuple(update_operands(&v[0], &v[1])?.0.to_vec()),
            "head" => {
                let Val::Str(t) = &v[0] else {
                    return Err("head: a string".into());
                };
                Val::Str(t.split('_').next().unwrap_or("").into())
            }
            "first_not_node" => seq(&v[0])?
                .iter()
                .find(|x| !matches!(x, Val::Node(_)))
                .cloned()
                .unwrap_or(Val::None),
            "name_of" => match &v[0] {
                Val::Rec(_, f) => f[0].1.clone(),
                _ => return Err("name_of: a class".into()),
            },
            "bool" => Val::Bool(truthy(&v[0])),
            "trip" => {
                let (a, b, c) = (int(&v[0])?, int(&v[1])?, int(&v[2])?);
                Val::Int(if c > 0 {
                    if b > a {
                        (b - a + c - 1) / c
                    } else {
                        0
                    }
                } else if c < 0 {
                    if a > b {
                        (a - b - c - 1) / (-c)
                    } else {
                        0
                    }
                } else {
                    0
                })
            }
            "axes_fault" => axes_fault(&v[0]).map_or(Val::None, Val::Str),
            "ttype_fault" => ttype_fault(&v[0]).map_or(Val::None, Val::Str),
            "literal_fault" => literal_fault(&v[0]).map_or(Val::None, Val::Str),
            _ => return Err(format!("unknown function {n}")),
        })
    }

    /// The built-in functions whose first arguments are lists of nodes: width, first type mismatch, carry faults.
    fn call_over_nodes(&self, n: &str, v: &[Val]) -> R<Val> {
        Ok(match n {
            "width_of" => {
                let mut w = 0i128;
                for h in node_list(&v[0])? {
                    let Val::Tuple(s) = self.field_of(&self.ttype(&Val::Node(h))?, "shape")? else {
                        return Err("shape".into());
                    };
                    w += s.iter().map(|d| int(d).unwrap_or(1)).product::<i128>()
                }
                Val::Int(w)
            }
            "first_ne_ttype" => {
                let hs = node_list(&v[0])?;
                let first = self.ttype(&Val::Node(*hs.first().ok_or("empty")?))?;
                let mut r = Val::None;
                for h in &hs[1..] {
                    if !veq(&self.ttype(&Val::Node(*h))?, &first) {
                        r = Val::Node(*h);
                        break;
                    }
                }
                r
            }
            "first_mismatch" => {
                let (x, y) = (node_list(&v[0])?, node_list(&v[1])?);
                Val::Int(
                    x.iter()
                        .zip(&y)
                        .position(|(p, q)| self.differ(*p, *q))
                        .map_or(-1, |j| j as i128),
                )
            }
            "first_not_instance" => {
                let mut at = -1;
                for (j, x) in seq(&v[0])?.iter().enumerate() {
                    if !self.is_a(x, &v[1])? {
                        at = j as i128;
                        break;
                    }
                }
                Val::Int(at)
            }
            "index_fault" => {
                let (plane, indices) = (
                    node_list(&Val::Tuple(vec![v[0].clone()]))?[0],
                    node_list(&v[1])?,
                );
                self.index_fault(plane, &indices)?
                    .map_or(Val::None, Val::Str)
            }
            "sample_fault" => {
                let (plane, coordinates) = (
                    node_list(&Val::Tuple(vec![v[0].clone()]))?[0],
                    node_list(&v[1])?,
                );
                self.sample_fault(plane, &coordinates, &v[2])?
                    .map_or(Val::None, Val::Str)
            }
            "group_fault" => {
                let Val::Str(membership) = &v[0] else {
                    return Err("group_fault: a membership name".into());
                };
                let key = node_list(&v[5])?;
                let (scratch, body) = (node_list(&v[6])?, node_list(&v[7])?);
                let facts = crate::groups::GroupFacts {
                    membership,
                    count: &v[1],
                    grid: &v[2],
                    axes: &v[3],
                    max_size: &v[4],
                    key: key.first().copied(),
                    scratch: &scratch,
                    body: &body,
                };
                self.c.group_fault(&facts).map_or(Val::None, Val::Str)
            }
            "reduce_fault" => {
                let operand = node_list(&Val::Tuple(vec![v[1].clone()]))?;
                let (Val::Str(op), [value]) = (&v[0], operand.as_slice()) else {
                    return Err("reduce_fault: an op name and one value".into());
                };
                self.c.reduce_fault(op, *value).map_or(Val::None, Val::Str)
            }
            "loop_group_fault" => {
                let (body, values) = (node_list(&v[0])?, node_list(&v[1])?);
                self.c
                    .loop_group_fault(&body, &values)
                    .map_or(Val::None, Val::Str)
            }
            "function_fault" => {
                let (Val::Str(name), Val::Str(scope)) = (&v[0], &v[1]) else {
                    return Err("function_fault: a name and a scope".into());
                };
                let (params, body) = (self.live_nodes(&v[2])?, self.live_nodes(&v[3])?);
                let (jvp, vjp) = (self.live_nodes(&v[6])?, self.live_nodes(&v[7])?);
                let facts = crate::functions::FunctionFacts {
                    name,
                    scope,
                    params: &params,
                    body: body.first().copied(),
                    natives: &v[4],
                    returns: &v[5],
                    jvp: jvp.first().copied(),
                    vjp: &vjp,
                };
                self.c.function_fault(&facts).map_or(Val::None, Val::Str)
            }
            "call_fault" => {
                let args = self.live_nodes(&v[0])?;
                self.c.call_fault(&args, &v[1]).map_or(Val::None, Val::Str)
            }
            "call_result" => self.c.call_result(&v[0])?,
            "store_index_fault" => {
                let Val::Str(name) = &v[0] else {
                    return Err("store_index_fault: a plane name".into());
                };
                let indices = node_list(&v[2])?;
                self.store_index_fault(name, &v[1], &indices)?
                    .map_or(Val::None, Val::Str)
            }
            "axis_fault" => {
                let plane = node_list(&Val::Tuple(vec![v[0].clone()]))?[0];
                self.axis_fault(plane, int(&v[1])?)?
                    .map_or(Val::None, Val::Str)
            }
            "carry_fault" => {
                let (c, i, x) = (node_list(&v[0])?, node_list(&v[1])?, node_list(&v[2])?);
                for j in 0..c.len() {
                    let code = if self.differ(c[j], i[j]) {
                        0
                    } else if self.differ(c[j], x[j]) {
                        1
                    } else if self.c.nodes[c[j] as usize].shape.len() > 2 {
                        2
                    } else {
                        continue;
                    };
                    return Ok(Val::Tuple(vec![Val::Int(j as i128), Val::Int(code)]));
                }
                Val::None
            }
            _ => return Err(format!("unknown function {n}")),
        })
    }
    /// The axes a plane leaf declares as (label, extent) pairs; a leaf without axes has the one runtime axis of a flat plane.
    fn plane_axes(&self, plane: u32) -> R<Vec<(String, bool)>> {
        let declared = match self.c.field(plane, "axes").map_err(|e| format!("{e:?}"))? {
            Val::Tuple(axes) => axes,
            _ => vec![],
        };
        if declared.is_empty() {
            return Ok(vec![(String::new(), true)]);
        }
        declared
            .iter()
            .map(|axis| {
                let pair = seq(axis)?;
                let label = match &pair[0] {
                    Val::Str(s) => s.clone(),
                    _ => String::new(),
                };
                Ok((label, matches!(pair[1], Val::None)))
            })
            .collect()
    }

    /// The name a refusal gives the axis at `position` of `plane`: its label, else its position.
    fn axis_name(axes: &[(String, bool)], position: usize) -> String {
        match axes.get(position) {
            Some((label, _)) if !label.is_empty() => format!("axis {label:?}"),
            _ => format!("axis {position}"),
        }
    }

    /// Why `coordinates` cannot sample `plane` with `filter`: one to three of them, one per axis of the plane, each a rank-0 real, and a real table for a linear sample.
    fn sample_fault(&self, plane: u32, coordinates: &[u32], filter: &Val) -> R<Option<String>> {
        let axes = self.plane_axes(plane)?;
        let Val::Str(name) = self.c.field(plane, "name").map_err(|e| format!("{e:?}"))? else {
            return Err("a plane leaf has no name".into());
        };
        if !(1..=3).contains(&coordinates.len()) {
            return Ok(Some(format!(
                "a sample takes one to three coordinates, got {}",
                coordinates.len()
            )));
        }
        if coordinates.len() != axes.len() {
            return Ok(Some(format!(
                "plane {name:?} has {} axes, so a sample takes one coordinate per axis, got {}",
                axes.len(),
                coordinates.len()
            )));
        }
        let value = &self.c.nodes[plane as usize].dtype;
        if *filter == Val::Str("linear".into()) && !(value == "f32" || value == "f64") {
            return Ok(Some(format!(
                "plane {name:?} holds {value}: a linear sample weighs values, so it takes a real table (use filter 'nearest')"
            )));
        }
        Ok(coordinates.iter().enumerate().find_map(|(k, &h)| {
            let node = &self.c.nodes[h as usize];
            (!node.shape.is_empty() || !(node.dtype == "f32" || node.dtype == "f64")).then(|| {
                format!(
                    "plane {name:?} {}: the coordinate must be a rank-0 real (f32 or f64), got {} of shape {:?}",
                    Self::axis_name(&axes, k),
                    node.dtype,
                    node.shape
                )
            })
        }))
    }

    /// Why `indices` do not address `plane`: the count against the plane's axes, then each index being a rank-0 integer.
    fn index_fault(&self, plane: u32, indices: &[u32]) -> R<Option<String>> {
        let axes = self.plane_axes(plane)?;
        let Val::Str(name) = self.c.field(plane, "name").map_err(|e| format!("{e:?}"))? else {
            return Err("a plane leaf has no name".into());
        };
        Ok(self.address_fault("plane", &name, &axes, indices))
    }

    /// Why a store at `indices` into the plane `name` that declares `axes` is refused; a store with no index commits at the
    /// lane's own cell and states no address.
    fn store_index_fault(&self, name: &str, axes: &Val, indices: &[u32]) -> R<Option<String>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let declared = match axes {
            Val::Tuple(list) if !list.is_empty() => list
                .iter()
                .map(|axis| {
                    let pair = seq(axis)?;
                    let label = match &pair[0] {
                        Val::Str(s) => s.clone(),
                        _ => String::new(),
                    };
                    Ok((label, matches!(pair[1], Val::None)))
                })
                .collect::<R<Vec<_>>>()?,
            _ => vec![(String::new(), true)],
        };
        Ok(self.address_fault("sink", name, &declared, indices))
    }

    /// The count and type faults of `indices` against the axes of the named plane or sink.
    fn address_fault(
        &self,
        what: &str,
        name: &str,
        axes: &[(String, bool)],
        indices: &[u32],
    ) -> Option<String> {
        if indices.len() != axes.len() {
            let labels: Vec<String> = (0..axes.len()).map(|k| Self::axis_name(axes, k)).collect();
            return Some(format!(
                "{what} {name:?} has {} axes ({}), so an access takes one index per axis, got {}",
                axes.len(),
                labels.join(", "),
                indices.len()
            ));
        }
        indices.iter().enumerate().find_map(|(k, &h)| {
            let node = &self.c.nodes[h as usize];
            (!node.shape.is_empty() || !(node.dtype == "i32" || node.dtype == "i64")).then(|| {
                format!(
                    "{what} {name:?} {}: the index must be a rank-0 integer (i32 or i64), got {} of shape {:?}",
                    Self::axis_name(axes, k),
                    node.dtype,
                    node.shape
                )
            })
        })
    }

    /// Why `axis` is not an axis of `plane`.
    fn axis_fault(&self, plane: u32, axis: i128) -> R<Option<String>> {
        let axes = self.plane_axes(plane)?;
        let Val::Str(name) = self.c.field(plane, "name").map_err(|e| format!("{e:?}"))? else {
            return Err("a plane leaf has no name".into());
        };
        let known = usize::try_from(axis).is_ok_and(|a| a < axes.len());
        Ok((!known).then(|| {
            format!(
                "plane {name:?} has {} axes, there is no axis {axis}",
                axes.len()
            )
        }))
    }

    /// The field `k` of a record value.
    fn field_of(&self, t: &Val, k: &str) -> R<Val> {
        if let Val::Rec(_, f) = t {
            f.iter()
                .find(|x| x.0 == k)
                .map(|x| x.1.clone())
                .ok_or(format!("no {k}"))
        } else {
            Err("not a record".into())
        }
    }
    /// Whether two nodes differ in type.
    fn differ(&self, a: u32, b: u32) -> bool {
        let (x, y) = (&self.c.nodes[a as usize], &self.c.nodes[b as usize]);
        (&x.dtype, &x.shape, &x.tag) != (&y.dtype, &y.shape, &y.tag)
    }

    /// Runs the statements of a program over its variables.
    fn run(&self, p: &Prog, env: &mut [Option<Val>]) -> R<()> {
        for st in &p.steps {
            match st {
                Step::Let(i, e) => env[*i] = Some(self.eval(e, env)?),
                Step::Check(c, t) => {
                    if truthy(&self.eval(c, env)?) {
                        let mut m = String::new();
                        for pc in t {
                            match pc {
                                Piece::Lit(x) => m += x,
                                Piece::Val(e, r) => {
                                    let v = self.eval(e, env)?;
                                    m += &match (&v, r) {
                                        (Val::Str(s), false) => s.clone(),
                                        (v, _) => self.rep(v),
                                    }
                                }
                            }
                        }
                        return Err(format!("\u{1}{m}"));
                    }
                }
                Step::Node(tt, ops) => {
                    let (ttype, o) = (self.eval(tt, env)?, self.eval(ops, env)?);
                    let ops = Val::Tuple(seq(&o)?.to_vec());
                    let base = &self.ps.named[self.ps.named_idx["node_base"]];
                    let mut e2 = vec![None; base.nvars];
                    (e2[0], e2[1], e2[2]) = (
                        env[p.params].clone(),
                        Some(ttype.clone()),
                        Some(ops.clone()),
                    );
                    self.run(base, &mut e2)?;
                    (env[p.params + 1], env[p.params + 2]) = (Some(ttype), Some(ops));
                }
                Step::Call(ix, args) => {
                    let c = &self.ps.named[*ix];
                    let mut e2 = vec![None; c.nvars];
                    for (k, a) in args.iter().enumerate() {
                        e2[k] = Some(self.eval(a, env)?)
                    }
                    e2[c.params] = env[p.params].clone();
                    self.run(c, &mut e2)?;
                    for off in [1, 2] {
                        if e2[c.params + off].is_some() {
                            env[p.params + off] = e2[c.params + off].clone()
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The constructor parameters of a record, by the class's `from` list.
    fn params(&self, node: &Node, p: &Prog, class: &crate::class::Class) -> R<Vec<Val>> {
        let slot = |k: &str| {
            class
                .slots
                .iter()
                .position(|s| s == k)
                .map(|i| node.fields[i].clone())
                .ok_or(format!("{} has no slot {k:?}", class.name))
        };
        let op = |n: usize| {
            node.ops
                .get(n)
                .map(|h| Val::Node(*h))
                .ok_or(format!("{}: operand {n} is missing", class.name))
        };
        let ops = |n: usize| Val::Tuple(node.ops.iter().skip(n).map(|h| Val::Node(*h)).collect());
        let ncar = |s: &dyn Fn(&str) -> R<Val>| -> R<usize> { Ok(seq(&s("carries")?)?.len()) };
        let (mut out, mut pos) = (vec![Val::None; p.params], 0);
        for (named, t) in &p.from {
            let vals = match t.as_str() {
                "ttype" => vec![ttype_val(&node.dtype, &node.shape, node.tag.as_deref())],
                "operands" => vec![ops(0)],
                "*operands" => node.ops.iter().map(|h| Val::Node(*h)).collect(),
                "carry_inits" => {
                    let n = ncar(&slot)?;
                    vec![Val::Tuple(
                        node.ops.iter().take(n).map(|h| Val::Node(*h)).collect(),
                    )]
                }
                "free_ops" => {
                    let (n, end) = (ncar(&slot)?, int(&slot("_free_end")?)? as usize);
                    vec![Val::Tuple(
                        node.ops
                            .iter()
                            .take(end)
                            .skip(n)
                            .map(|h| Val::Node(*h))
                            .collect(),
                    )]
                }
                "count_op" => vec![if matches!(slot("count")?, Val::None) {
                    Val::None
                } else {
                    op(node.ops.len().wrapping_sub(1))?
                }],
                t if t.starts_with("opq:") => vec![node
                    .ops
                    .get(t[4..].parse::<usize>().map_err(|_| "bad token")?)
                    .map_or(Val::None, |h| Val::Node(*h))],
                t if t.starts_with("op:") => vec![op(t[3..].parse().map_err(|_| "bad token")?)?],
                t if t.starts_with("ops:") => vec![ops(t[4..]
                    .trim_end_matches(':')
                    .parse()
                    .map_err(|_| "bad token")?)],
                t if t.starts_with("edge:") => vec![slot(&t[5..])?],
                t if t.starts_with('@') => vec![slot(t[1..].split(':').next().unwrap())?],
                t => return Err(format!("{}: unknown token {t}", class.name)),
            };
            match named {
                Some(i) => out[*i] = vals.into_iter().next().unwrap_or(Val::None),
                None => {
                    for v in vals {
                        if pos < out.len() {
                            out[pos] = v
                        }
                        pos += 1
                    }
                }
            }
        }
        if pos > p.params {
            return Err(format!(
                "{}: {} operands given, the constructor takes {}",
                class.name, pos, p.params
            ));
        }
        Ok(out)
    }
}

/// Checks the quantity records among a node's fields.
fn quantity_check(ps: &Progs, f: &[(String, Val)]) -> R<()> {
    let get = |k: &str| {
        f.iter()
            .find(|x| x.0 == k)
            .map_or(Val::None, |x| x.1.clone())
    };
    let konst = |k: &str| {
        ps.consts.iter().find(|c| c.0 == k).map_or(vec![], |c| {
            seq(&c.1)
                .unwrap()
                .iter()
                .filter_map(|v| {
                    if let Val::Str(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
    };
    let (slug, wires, pack) = (get("slug"), get("wires"), get("pack"));
    let sl = text_val(&slug);
    let packs = konst("QUANTITY_PACKS");
    if !matches!(&pack, Val::Str(p) if packs.contains(p)) {
        return Err(format!(
            "Quantity {sl}: pack {} is not one of {}",
            text_val(&pack),
            text_val(&Val::Tuple(packs.into_iter().map(Val::Str).collect()))
        ));
    }
    if !truthy(&slug) {
        return Err(
            "Quantity: slug is required \u{2014} it is the namespace's first segment".into(),
        );
    }
    let ws = seq(&wires)?;
    if ws.is_empty() {
        return Err(format!("Quantity {sl}: declares no wires"));
    }
    let roles: Vec<String> = konst("QUANTITY_ROLE_PAIRS")
        .chunks(2)
        .map(|p| p[0].clone())
        .collect();
    let mut seen: Vec<Val> = vec![];
    for w in ws {
        let Val::Rec(n, wf) = w else {
            return Err(format!("Quantity {sl}: {} is not a Wire", text_val(w)));
        };
        if n != "Wire" {
            return Err(format!("Quantity {sl}: {} is not a Wire", text_val(w)));
        }
        let g = |k: &str| {
            wf.iter()
                .find(|x| x.0 == k)
                .map_or(Val::None, |x| x.1.clone())
        };
        let (role, name) = (g("role"), g("name"));
        if !matches!(&role, Val::Str(r) if roles.contains(r)) {
            let mut s = roles.clone();
            s.sort();
            return Err(format!("Quantity {sl}: wire {} has role {}, which no leaf kind mints; declared read roles are [{}]", text_val(&name), text_val(&role), s.iter().map(|r| py_str(r)).collect::<Vec<_>>().join(", ")));
        }
        if seen.contains(&name) {
            return Err(format!(
                "Quantity {sl}: duplicate wire name {}",
                text_val(&name)
            ));
        }
        seen.push(name);
    }
    Ok(())
}

/// Checks one quantity value.
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
            Some(m) => Err(CtorFault {
                declared: true,
                text: m.to_string(),
            }),
            None if e.contains("does not match its constructor") => Err(CtorFault {
                declared: false,
                text: e,
            }),
            None => Err(CtorFault {
                declared: false,
                text: format!(
                    "{}: its state cannot be read by its constructor ({e})",
                    ctx.table.classes[node.class].name
                ),
            }),
        },
        Ok(()) => Ok(()),
    }
}

/// A failure of the constructor checks: a declared check that refused (its text is final), or the program could not run or its result
/// differs from the record (the rule of the kind has the better text).
pub struct CtorFault {
    /// True when a declared check refused.
    pub declared: bool,
    /// The message.
    pub text: String,
}

/// Checks a raw block node.
fn check_raw(ctx: &Ctx, node: &Node) -> Result<(), String> {
    let class = &ctx.table.classes[node.class];
    if class.spec.is_none() {
        return Ok(());
    }
    let ps = &ctx.progs;
    node.fields
        .iter()
        .try_for_each(|f| quantities(ps, f))
        .map_err(|e| format!("\u{1}{e}"))?;
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
    for (i, v) in ev.params(node, p, class)?.into_iter().enumerate() {
        env[i] = Some(v)
    }
    env[p.params] = Some(cls(&class.name));
    ev.run(p, &mut env)?;
    for (k, (slot, e)) in p
        .slots
        .iter()
        .zip(
            std::iter::once(None)
                .chain(std::iter::once(None))
                .chain(p.out.iter().map(Some)),
        )
        .enumerate()
    {
        let (got, want) = match (k, e) {
            (0, _) => (
                ttype.clone(),
                env[p.params + 1].clone().unwrap_or(Val::None),
            ),
            (1, _) => (ops.clone(), env[p.params + 2].clone().unwrap_or(Val::None)),
            (_, e) => {
                let Some(i) = class.slots.iter().position(|s| s == slot) else {
                    continue;
                };
                (node.fields[i].clone(), ev.eval(e.unwrap(), &env)?)
            }
        };
        if text_val(&got) != text_val(&want) {
            return Err(format!(
                "{}: {slot} does not match its constructor (got {}, constructor gives {})",
                class.name,
                text_val(&got),
                text_val(&want)
            ));
        }
    }
    Ok(())
}

/// A node built from constructor parameters: its type, operands and every slot the program assigned.
pub struct New {
    /// The element type name.
    pub dtype: String,
    /// The extents.
    pub shape: Vec<i64>,
    /// The tag, if any.
    pub tag: Option<String>,
    /// The operands.
    pub ops: Vec<u32>,
    /// The slots the program assigned, by name.
    pub fields: Vec<(String, Val)>,
}

/// The constructor program of a class, if it has one.
fn program_of(ctx: &Ctx, class: usize) -> Option<&Prog> {
    ctx.table.classes[class]
        .chain
        .iter()
        .find_map(|&k| ctx.progs.by_class.get(&ctx.table.classes[k].name))
        .map(|&i| &ctx.progs.progs[i])
}

/// The default the constructor of a core class states for the parameter named like `slot`, if it states one.
pub fn slot_default(ctx: &Ctx, class: usize, slot: &str) -> Option<Val> {
    let p = program_of(ctx, class)?;
    let at = p.pnames.iter().position(|n| n == slot)?;
    p.pdef[at].clone()
}

/// Runs the declared program of a core class forward over constructor parameters (name, value); a node reference in a value is a handle.
pub fn forward(ctx: &Ctx, class: usize, given: &[(String, Val)]) -> Result<New, String> {
    let c = &ctx.table.classes[class];
    let Some(p) = program_of(ctx, class) else {
        return Err(format!("{}: has no constructor program", c.name));
    };
    if let Some((k, _)) = given.iter().find(|g| !p.pnames.contains(&g.0)) {
        return Err(format!("{}: has no parameter {k:?}", c.name));
    }
    let mut env: Vec<Option<Val>> = vec![None; p.nvars];
    for (i, name) in p.pnames.iter().enumerate() {
        match given
            .iter()
            .find(|g| g.0 == *name)
            .map(|g| g.1.clone())
            .or_else(|| p.pdef[i].clone())
        {
            Some(v) => env[i] = Some(v),
            None => return Err(format!("{}: parameter {name:?} is required", c.name)),
        }
    }
    env[p.params] = Some(cls(&c.name));
    let ev = Ev {
        c: ctx,
        ps: &ctx.progs,
    };
    let run = |env: &mut Vec<Option<Val>>| -> R<New> {
        ev.run(p, env)?;
        let Some(Val::Rec(_, tf)) = env[p.params + 1].clone() else {
            return Err("the program set no type".into());
        };
        let get = |k: &str| {
            tf.iter()
                .find(|x| x.0 == k)
                .map(|x| x.1.clone())
                .unwrap_or(Val::None)
        };
        let shape = match get("shape") {
            Val::Tuple(v) => v
                .iter()
                .map(|x| int(x).map(|i| i as i64))
                .collect::<R<Vec<_>>>()?,
            _ => vec![],
        };
        let Val::Str(dtype) = get("dtype") else {
            return Err("the program set no dtype".into());
        };
        let tag = if let Val::Str(t) = get("tag") {
            Some(t)
        } else {
            None
        };
        let Some(ops) = env[p.params + 2].clone() else {
            return Err("the program set no operands".into());
        };
        let ops = seq(&ops)?
            .iter()
            .map(|o| {
                if let Val::Node(h) = o {
                    Ok(*h)
                } else {
                    Err("an operand is not a node".to_string())
                }
            })
            .collect::<R<Vec<_>>>()?;
        let mut fields = vec![];
        for (slot, e) in p.slots.iter().skip(2).zip(&p.out) {
            if c.slots.iter().any(|s| s == slot) {
                fields.push((slot.clone(), ev.eval(e, env)?))
            }
        }
        Ok(New {
            dtype,
            shape,
            tag,
            ops,
            fields,
        })
    };
    run(&mut env).map_err(|e| match e.strip_prefix('\u{1}') {
        Some(m) => m.to_string(),
        None => format!(
            "{}: its state cannot be read by its constructor ({e})",
            c.name
        ),
    })
}

/// The constructor parameters of a node, as value text and the handles its references name in order (`#i` is `refs[i]`).
pub fn params_kt(ctx: &Ctx, h: u32, refs: &mut Vec<u32>) -> Result<String, String> {
    let node = ctx.nodes.get(h as usize).ok_or("no such node")?;
    let class = &ctx.table.classes[node.class];
    let ev = Ev {
        c: ctx,
        ps: &ctx.progs,
    };
    let vals: Vec<(String, Val)> = if class.decl.is_some() {
        // a declared kind's constructor takes the type, the operands and every field its class does not fix
        let mut v = vec![
            (
                "ttype".to_string(),
                ttype_val(&node.dtype, &node.shape, node.tag.as_deref()),
            ),
            (
                "operands".to_string(),
                Val::Tuple(node.ops.iter().map(|&o| Val::Node(o)).collect()),
            ),
        ];
        v.extend(
            class
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !class.fixed.iter().any(|f| f.0 == **s))
                .map(|(i, s)| (s.clone(), node.fields[i].clone())),
        );
        v
    } else {
        match program_of(ctx, node.class) {
            Some(p) => ev
                .params(node, p, class)?
                .into_iter()
                .enumerate()
                .map(|(i, v)| (p.pnames[i].clone(), v))
                .collect(),
            None => return Err(format!("{}: has no constructor program", class.name)),
        }
    };
    Ok(text_val(&Val::Tuple(
        vals.into_iter()
            .map(|(k, v)| Val::Tuple(vec![Val::Str(k), crate::value::lift_refs(v, refs)]))
            .collect(),
    )))
}
