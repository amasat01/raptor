//! The one-way pseudo-code text of a node: readable, deterministic, never parsed back.
use raptor_ir::dump::refs;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::{text_val, Val};
use std::collections::HashMap;

/// The (target, text) pairs of a native map.
fn refs_pairs(v: &Val, out: &mut Vec<(String, String)>) {
    if let Val::Tuple(items) = v {
        for item in items {
            if let Val::Tuple(p) = item {
                if let [Val::Str(target), Val::Str(text)] = p.as_slice() {
                    out.push((target.clone(), text.clone()))
                }
            }
        }
    }
}

struct R<'a> {
    c: &'a Ctx,
    root: u32,
    uses: HashMap<u32, usize>,
    names: HashMap<u32, String>,
    lines: Vec<String>,
    ind: usize,
}

impl R<'_> {
    fn is(&self, h: u32, class: &str) -> bool {
        self.c.table.is_a(self.c.nodes[h as usize].class, class)
    }
    fn s(&self, h: u32, f: &str) -> Result<String, Fault> {
        self.c.str_field(h, f)
    }
    fn kind(&self, h: u32) -> String {
        self.s(h, "kind").unwrap_or_else(|_| {
            self.c.table.classes[self.c.nodes[h as usize].class]
                .name
                .clone()
        })
    }
    fn atom(&self, h: u32) -> bool {
        self.is(h, "Leaf")
            || self.is(h, "Param")
            || self.is(h, "Const")
            || self.is(h, "LoopIndex")
            || self.is(h, "LoopCarry")
    }
    fn sink(&self, h: u32) -> bool {
        self.is(h, "Sink") && (1..=2).contains(&self.c.nodes[h as usize].ops.len())
    }
    fn push(&mut self, l: String) {
        self.lines.push(format!("{}{l}", "  ".repeat(self.ind)))
    }

    /// A field value: a node reference renders as its expression, anything else as its text.
    fn val(&mut self, v: &Val) -> Result<String, Fault> {
        Ok(match v {
            Val::Node(h) => self.expr(*h)?,
            Val::Tuple(t) | Val::List(t) => {
                let p = t
                    .iter()
                    .map(|x| self.val(x))
                    .collect::<Result<Vec<_>, _>>()?;
                format!("({})", p.join(", "))
            }
            x => text_val(x),
        })
    }

    fn expr(&mut self, h: u32) -> Result<String, Fault> {
        if let Some(n) = self.names.get(&h) {
            return Ok(n.clone());
        }
        let c = self.c;
        let n = &c.nodes[h as usize];
        if self.is(h, "Const") {
            return Ok(text_val(&c.field(h, "literal")?));
        }
        if self.atom(h) {
            return self.s(h, "name");
        }
        let named = !self.sink(h)
            && (self.uses.get(&h).copied().unwrap_or(0) > 1
                || (self.is(h, "Loop") && h != self.root));
        let ops = n.ops.clone();
        let rhs = if self.is(h, "Loop") {
            self.lp(h)?;
            String::new()
        } else {
            let a = ops
                .iter()
                .map(|&o| self.expr(o))
                .collect::<Result<Vec<_>, _>>()?;
            let k = if self.is(h, "Call") {
                self.callee_name(h)?
            } else {
                self.kind(h)
            };
            match (
                ["add", "sub", "mul", "div"].iter().position(|x| *x == k),
                a.as_slice(),
            ) {
                (Some(i), [l, r]) if self.is(h, "Op") => {
                    format!("({l} {} {r})", ["+", "-", "*", "/"][i])
                }
                _ => {
                    let mut a = a;
                    if let Ok(l) = c.field(h, "literal") {
                        if l != Val::None {
                            a.push(format!("literal={}", text_val(&l)))
                        }
                    }
                    format!("{k}({})", a.join(", "))
                }
            }
        };
        if !named {
            return Ok(rhs);
        }
        let name = format!("t{}", self.names.len());
        if self.is(h, "Loop") {
            self.names.insert(h, name.clone());
            return Ok(name);
        }
        self.push(format!("{name} = {rhs}"));
        self.names.insert(h, name.clone());
        Ok(name)
    }

    /// The name of the function a call names.
    fn callee_name(&self, h: u32) -> Result<String, Fault> {
        match self.c.field(h, "function")? {
            Val::Tuple(pair) => match pair.first() {
                Some(Val::Str(name)) => Ok(name.clone()),
                _ => refuse("a call names its function"),
            },
            _ => refuse("a call names its function"),
        }
    }

    /// The type of a node as the header of a function states it: the element type, then the shape when there is one.
    fn type_text(&self, h: u32) -> String {
        let n = &self.c.nodes[h as usize];
        if n.shape.is_empty() {
            n.dtype.clone()
        } else {
            format!("{}{:?}", n.dtype, n.shape)
        }
    }

    /// A function of the table: its header, then its traced body or its native texts, then the names of its rules.
    fn function(&mut self, h: u32) -> Result<(), Fault> {
        let c = self.c;
        let mut params = vec![];
        let mut held = vec![];
        refs(&c.field(h, "params")?, &mut held);
        for p in held {
            params.push(format!("{}: {}", self.s(p, "name")?, self.type_text(p)))
        }
        let scope = match self.s(h, "scope")?.as_str() {
            "group" => "group ",
            _ => "",
        };
        self.push(format!(
            "{scope}function {}({}) -> {}",
            self.s(h, "name")?,
            params.join(", "),
            self.type_text(h)
        ));
        self.ind += 1;
        if let Val::Node(body) = c.field(h, "body")? {
            let e = self.expr(body)?;
            self.push(format!("return {e}"));
        }
        let mut texts = vec![];
        refs_pairs(&c.field(h, "natives")?, &mut texts);
        for (target, text) in texts {
            self.push(format!("native {target}: {}", text_val(&Val::Str(text))));
        }
        let mut rules = vec![];
        for role in ["jvp", "vjp"] {
            let mut held = vec![];
            refs(&c.field(h, role)?, &mut held);
            rules.extend(held.into_iter().map(|r| (role, r)));
        }
        for (role, rule) in rules {
            self.push(format!("{role} {}", self.s(rule, "name")?));
        }
        self.ind -= 1;
        Ok(())
    }

    /// A statement: a sink, a loop with its body, or a bare expression.
    fn stmt(&mut self, h: u32) -> Result<(), Fault> {
        if self.is(h, "Function") {
            return self.function(h);
        }
        if self.is(h, "Loop") {
            return self.lp(h);
        }
        if !self.sink(h) {
            let e = self.expr(h)?;
            self.push(e);
            return Ok(());
        }
        let ops = self.c.nodes[h as usize].ops.clone();
        let v = self.expr(ops[0])?;
        let mut l = self.s(h, "name")?;
        if let Some(&i) = ops.get(1) {
            l = format!("{l}[{}]", self.expr(i)?)
        }
        let k = self.kind(h);
        self.push(if k == "assign" {
            format!("{l} = {v}")
        } else {
            format!("{k} {l} = {v}")
        });
        Ok(())
    }

    fn lp(&mut self, h: u32) -> Result<(), Fault> {
        let c = self.c;
        let names = |r: &Self, v: Val| -> Result<Vec<String>, Fault> {
            let mut o = vec![];
            refs(&v, &mut o);
            o.into_iter().map(|x| r.s(x, "name")).collect()
        };
        let carries = names(self, c.field(h, "carries")?)?;
        let (mut bounds, mut inits) = (vec![], vec![]);
        for f in ["start", "stop", "step"] {
            let v = c.field(h, f)?;
            bounds.push(self.val(&v)?)
        }
        let ops = c.nodes[h as usize].ops.clone();
        for (n, &o) in carries.iter().zip(&ops) {
            inits.push(format!("{n} = {}", self.expr(o)?))
        }
        let ix = match c.field(h, "index")? {
            Val::Node(i) => self.s(i, "name")?,
            _ => "_".into(),
        };
        let name = self
            .names
            .get(&h)
            .cloned()
            .unwrap_or_else(|| format!("t{}", self.names.len()));
        self.names.insert(h, name.clone());
        let head = if h == self.root {
            String::new()
        } else {
            format!("{name} = ")
        };
        self.push(format!(
            "{head}loop {ix} in {}..{} step {} carry [{}]",
            bounds[0],
            bounds[1],
            bounds[2],
            inits.join(", ")
        ));
        self.ind += 1;
        let mut nexts = vec![];
        refs(&c.field(h, "nexts")?, &mut nexts);
        for (n, x) in carries.iter().zip(nexts) {
            let e = self.expr(x)?;
            self.push(format!("{n} <- {e}"))
        }
        let mut sinks = vec![];
        refs(&c.field(h, "body_sinks")?, &mut sinks);
        for s in sinks {
            self.stmt(s)?
        }
        self.ind -= 1;
        Ok(())
    }
}

/// The pseudo-code text of the node `root`: one line per node, in the order the walk finds them.
pub fn pseudo(c: &Ctx, root: u32) -> Result<String, Fault> {
    c.node(root)?;
    let (mut uses, mut seen, mut stack) = (HashMap::new(), vec![false; c.nodes.len()], vec![root]);
    while let Some(h) = stack.pop() {
        if std::mem::replace(&mut seen[h as usize], true) {
            continue;
        }
        let n = &c.nodes[h as usize];
        let mut kids = n.ops.clone();
        n.fields.iter().for_each(|f| refs(f, &mut kids));
        for k in kids {
            *uses.entry(k).or_insert(0usize) += 1;
            stack.push(k)
        }
    }
    let mut r = R {
        c,
        root,
        uses,
        names: HashMap::new(),
        lines: vec![],
        ind: 0,
    };
    r.stmt(root)?;
    Ok(r.lines.join("\n") + "\n")
}
