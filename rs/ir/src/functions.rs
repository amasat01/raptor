//! Functions: the typing of a `Function` declaration and of a `Call`, and the digest that names a function in a call.
//!
//! The function table of a document is the set of `Function` nodes of its context. A function is named by its `name` and its
//! digest; a `Call` holds both, so a call never meets a function other than the one it was typed against. A traced body is a
//! sub-graph whose only leaves are the parameters (`Param` nodes); a native body is a map from a target name to a text that the IR
//! carries unread. The rules of derivatives are functions of the same table, held by reference.
use crate::record::{Ctx, Fault};
use crate::value::{text_val, ttype_val, Val};
use std::collections::{HashMap, HashSet};

/// The number of hex digits of a function digest and of a body key.
const DIGEST_HEX: usize = 32;

/// The facts of a `Function` as its constructor states them, read by `Ctx::function_fault`.
pub struct FunctionFacts<'a> {
    /// The name the table and the calls know the function by.
    pub name: &'a str,
    /// `lane` or `group`.
    pub scope: &'a str,
    /// The `Param` nodes, in the order a call passes its arguments.
    pub params: &'a [u32],
    /// The root of a traced body; none for a native function.
    pub body: Option<u32>,
    /// The native texts: a tuple of (target name, text) pairs.
    pub natives: &'a Val,
    /// The type of the result.
    pub returns: &'a Val,
    /// The rule of the forward derivative.
    pub jvp: Option<u32>,
    /// The rules of the reverse derivative: one per read, in parameter order, rule i giving the gradient of read i.
    pub vjp: &'a [u32],
}

/// The dtype, shape and tag a type value states.
type Typed = (String, Vec<i64>, Option<String>);

/// The parts of a type value, or none when it is not a well formed one.
fn typed_of(t: &Val) -> Option<Typed> {
    let Val::Rec(_, fields) = t else { return None };
    let get = |k: &str| fields.iter().find(|f| f.0 == k).map(|f| &f.1);
    let (Some(Val::Str(dtype)), Some(Val::Tuple(shape))) = (get("dtype"), get("shape")) else {
        return None;
    };
    let shape = shape
        .iter()
        .map(|d| match d {
            Val::Int(x) => i64::try_from(*x).ok(),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let tag = match get("tag") {
        Some(Val::Str(s)) => Some(s.clone()),
        _ => None,
    };
    Some((dtype.clone(), shape, tag))
}

/// The text of a type, for messages.
fn type_text((dtype, shape, tag): &Typed) -> String {
    text_val(&ttype_val(dtype, shape, tag.as_deref()))
}

/// The (target, text) pairs of a native map, or why it is not one.
fn native_pairs(natives: &Val) -> Result<Vec<(String, String)>, String> {
    let Val::Tuple(items) = natives else {
        return Err("the native texts are a tuple of (target name, text) pairs".into());
    };
    items
        .iter()
        .map(|item| match item {
            Val::Tuple(p) => match p.as_slice() {
                [Val::Str(target), Val::Str(text)] if !target.is_empty() => {
                    Ok((target.clone(), text.clone()))
                }
                _ => Err("a native text is a (target name, text) pair with a target name".into()),
            },
            _ => Err("a native text is a (target name, text) pair with a target name".into()),
        })
        .collect()
}

impl Ctx {
    /// The type of node `h`.
    fn typed_node(&self, h: u32) -> Typed {
        let n = &self.nodes[h as usize];
        (n.dtype.clone(), n.shape.clone(), n.tag.clone())
    }

    /// The `Function` nodes of the table that carry `name`.
    fn functions_named(&self, name: &str) -> Vec<u32> {
        (0..self.nodes.len() as u32)
            .filter(|&h| {
                self.is_a(h, "Function") && self.text_of(h, "name").as_deref() == Some(name)
            })
            .collect()
    }

    /// The function of the table with this name and digest.
    pub fn function_named(&self, name: &str, digest: &str) -> Option<u32> {
        self.functions_named(name)
            .into_iter()
            .find(|&h| self.function_digest(h).is_ok_and(|d| d == digest))
    }

    /// The name and digest a `Call` holds, or none when `h` is not a call or holds anything else.
    fn callee_of(&self, h: u32) -> Option<(String, String)> {
        if !self.is_a(h, "Call") {
            return None;
        }
        match self.field(h, "function").ok()? {
            Val::Tuple(p) => match p.as_slice() {
                [Val::Str(name), Val::Str(digest)] => Some((name.clone(), digest.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    /// The function of the table that the call `h` names; none when `h` is not a call or the table holds no such function.
    pub(crate) fn called_function(&self, h: u32) -> Option<u32> {
        let (name, digest) = self.callee_of(h)?;
        self.function_named(&name, &digest)
    }

    /// Whether the function the call `h` names has a native body (texts per target) and no traced one.
    pub(crate) fn call_is_native(&self, h: u32) -> Result<bool, Fault> {
        let function = self.called_function(h).ok_or_else(|| {
            Fault::Refused("a call of a function that the table does not hold".into())
        })?;
        Ok(matches!(self.field(function, "natives")?, Val::Tuple(texts) if !texts.is_empty()))
    }

    /// The name of the group function a node calls, if it is a call of one.
    pub(crate) fn called_group_function(&self, h: u32) -> Option<String> {
        let (name, _) = self.callee_of(h)?;
        let f = self.called_function(h)?;
        (self.text_of(f, "scope").as_deref() == Some("group")).then_some(name)
    }

    /// The digest of the function `h`: equal for equal declarations, different when its name, scope, a parameter, the result type,
    /// the body, one native text or a rule differs. A traced body is hashed by structure, its leaves keyed by the position of the
    /// parameter; the order of the native texts does not matter.
    pub fn function_digest(&self, h: u32) -> Result<String, Fault> {
        let node = self.node(h)?;
        if !self.is_a(h, "Function") {
            return Err(Fault::Refused(format!("node {h} is not a Function")));
        }
        let (name, scope) = (self.str_field(h, "name")?, self.str_field(h, "scope")?);
        let params = self.refs(h, "params")?;
        let node_of = |field: &str| self.node_field(h, field);
        let facts = FunctionFacts {
            name: &name,
            scope: &scope,
            params: &params,
            body: node_of("body")?,
            natives: &self.field(h, "natives")?,
            returns: &ttype_val(&node.dtype, &node.shape, node.tag.as_deref()),
            jvp: node_of("jvp")?,
            vjp: &self.refs(h, "vjp")?,
        };
        self.digest_of(&facts)
            .map_err(|m| Fault::Refused(format!("function '{name}': {m}")))
    }

    /// The digest of a function stated by its facts.
    fn digest_of(&self, f: &FunctionFacts) -> Result<String, String> {
        let param_line = |&p: &u32| {
            let access = self.text_of(p, "access").unwrap_or_default();
            Val::Tuple(vec![
                Val::Str(self.plane_name(p)),
                Val::Str(access),
                ttype_val(
                    &self.nodes[p as usize].dtype,
                    &self.nodes[p as usize].shape,
                    self.nodes[p as usize].tag.as_deref(),
                ),
            ])
        };
        let mut natives = native_pairs(f.natives)?;
        natives.sort();
        let natives = Val::Tuple(
            natives
                .into_iter()
                .map(|(target, text)| Val::Tuple(vec![Val::Str(target), Val::Str(text)]))
                .collect(),
        );
        let body = match f.body {
            Some(b) => Val::Str(self.body_key(b, f.params)?),
            None => Val::None,
        };
        let rule = |r: Option<u32>| -> Result<Val, String> {
            r.map_or(Ok(Val::None), |h| {
                self.function_digest(h)
                    .map(Val::Str)
                    .map_err(|e| format!("{e:?}"))
            })
        };
        let lines = [
            ("name", Val::Str(f.name.into())),
            ("scope", Val::Str(f.scope.into())),
            (
                "params",
                Val::Tuple(f.params.iter().map(param_line).collect()),
            ),
            ("returns", f.returns.clone()),
            ("body", body),
            ("natives", natives),
            ("jvp", rule(f.jvp)?),
            (
                "vjp",
                Val::Tuple(
                    f.vjp
                        .iter()
                        .map(|&r| rule(Some(r)))
                        .collect::<Result<_, _>>()?,
                ),
            ),
        ];
        let mut data = b"raptor.ir.function/1\n".to_vec();
        for (key, value) in lines {
            data.extend_from_slice(format!("{key}|{}\n", text_val(&value)).as_bytes());
        }
        Ok(rcore::sha256::hex(&data)[..DIGEST_HEX].to_string())
    }

    /// The structural key of a traced body: every node is keyed by a hash of its kind, type, child keys and identity; a parameter
    /// of the function by its position, so the names of the parameters of a body that never reads them do not matter.
    fn body_key(&self, root: u32, params: &[u32]) -> Result<String, String> {
        let mut key_of: HashMap<u32, String> = params
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, format!("param{i}")))
            .collect();
        let mut stack = vec![(root, false)];
        while let Some((n, expanded)) = stack.pop() {
            if key_of.contains_key(&n) {
                continue;
            }
            let node = &self.nodes[n as usize];
            if !expanded {
                stack.push((n, true));
                stack.extend(node.ops.iter().rev().map(|&c| (c, false)));
                continue;
            }
            let children = node
                .ops
                .iter()
                .map(|c| Val::Str(key_of[c].clone()))
                .collect();
            let kind = self.kind_of(n).map_err(|e| format!("{e:?}"))?;
            let identity = self.dedup_val(n).map_err(|e| format!("{e:?}"))?;
            let raw = format!(
                "({}, {}, {}, {}, {}, {})",
                text_val(&Val::Str(kind)),
                text_val(&Val::Str(node.dtype.clone())),
                text_val(&Val::Tuple(
                    node.shape
                        .iter()
                        .map(|&d| Val::Int(i128::from(d)))
                        .collect()
                )),
                text_val(&node.tag.clone().map_or(Val::None, Val::Str)),
                text_val(&Val::Tuple(children)),
                text_val(&identity)
            );
            key_of.insert(
                n,
                rcore::sha256::hex(raw.as_bytes())[..DIGEST_HEX].to_string(),
            );
        }
        Ok(key_of[&root].clone())
    }

    /// Why a function stated by `f` cannot be declared, if it cannot: the first of a wrong parameter list, a body that is not
    /// exactly one of traced and native, an impure traced body, a call that reaches the function itself, a rule that does not
    /// fit, and a name the table already holds with another digest.
    pub fn function_fault(&self, f: &FunctionFacts) -> Option<String> {
        let who = format!("function '{}'", f.name);
        if f.name.is_empty() {
            return Some("a function has a name".into());
        }
        let mut seen = HashSet::new();
        for &p in f.params {
            if !self.is_a(p, "Param") {
                return Some(format!(
                    "{who}: a parameter is a Param, got a {}",
                    self.class_name(p)
                ));
            }
            if !seen.insert(self.plane_name(p)) {
                return Some(format!(
                    "{who}: parameter '{}' is declared twice",
                    self.plane_name(p)
                ));
            }
        }
        let natives = match native_pairs(f.natives) {
            Ok(n) => n,
            Err(m) => return Some(format!("{who}: {m}")),
        };
        let mut targets = HashSet::new();
        if let Some((target, _)) = natives.iter().find(|(t, _)| !targets.insert(t.as_str())) {
            return Some(format!("{who}: the target '{target}' has two texts"));
        }
        let Some(returns) = typed_of(f.returns) else {
            return Some(format!("{who}: the result is a TensorType"));
        };
        match (f.body, natives.is_empty()) {
            (Some(_), false) => {
                return Some(format!("{who}: the body is traced or native, not both"));
            }
            (None, true) => {
                return Some(format!(
                    "native function '{}' holds no text: its map from a target name to a text is empty",
                    f.name
                ));
            }
            (Some(body), true) => {
                if let Some(fault) = self.body_fault(&who, f, body, &returns) {
                    return Some(fault);
                }
            }
            (None, false) => {}
        }
        if let Some(fault) = self.rules_fault(&who, f, &returns) {
            return Some(fault);
        }
        if let Some(fault) = self.recursion_fault(&who, f) {
            return Some(fault);
        }
        let digest = match self.digest_of(f) {
            Ok(d) => d,
            Err(m) => return Some(format!("{who}: {m}")),
        };
        self.functions_named(f.name)
            .into_iter()
            .find(|&h| self.function_digest(h).is_ok_and(|d| d != digest))
            .map(|_| format!("{who} is already declared with another body; a name holds one function in the table"))
    }

    /// Why the traced body `body` is not pure: it reads a plane, holds an update or another sink, a group or a sample reduction,
    /// the index or a carry of a loop it is not in, reaches a parameter of another function, calls a group function from a lane function, or has another type than the result.
    fn body_fault(
        &self,
        who: &str,
        f: &FunctionFacts,
        body: u32,
        returns: &Typed,
    ) -> Option<String> {
        if self.typed_node(body) != *returns {
            return Some(format!(
                "{who}: the body has the type {}, the result is declared {}",
                type_text(&self.typed_node(body)),
                type_text(returns)
            ));
        }
        let (mut stack, mut seen) = (vec![body], HashSet::new());
        // the index and the carries of the loops the body holds: a placeholder is the body's own only through its loop
        let mut own = HashSet::new();
        while let Some(h) = stack.pop() {
            if !seen.insert(h) {
                continue;
            }
            if self.is_a(h, "Loop") {
                for field in ["index", "carries"] {
                    own.extend(self.refs(h, field).unwrap_or_default());
                }
            }
            let class = self.class_name(h);
            let held = if self.is_a(h, "LoopIndex") || self.is_a(h, "LoopCarry") {
                if own.contains(&h) {
                    stack.extend(self.reached_from(h));
                    continue;
                }
                "the index or a carry of a loop it is not in".to_string()
            } else if self.is_a(h, "Param") {
                if !f.params.contains(&h) {
                    return Some(format!(
                        "{who}: the body reaches the parameter '{}', which the function does not declare",
                        self.plane_name(h)
                    ));
                }
                continue;
            } else if class == "Group" || crate::groups::GROUP_ONLY.contains(&class) {
                format!("a {class}, a group element")
            } else if class == "ReduceSamples" || class == "SampleSum" {
                format!("a {class}, a sample reduction")
            } else if self.is_a(h, "Sink") {
                format!("an {class}, which writes")
            } else if self.is_a(h, "Leaf") {
                format!("a read of the plane '{}'", self.plane_name(h))
            } else if self.called_group_function(h).is_some() && f.scope == "lane" {
                format!(
                    "a call of the group function '{}'",
                    self.called_group_function(h).unwrap_or_default()
                )
            } else {
                stack.extend(self.reached_from(h));
                continue;
            };
            return Some(format!(
                "{who}: a traced body is pure and its body holds {held}"
            ));
        }
        None
    }

    /// Why the rules of the function stated by `f` do not fit. The forward rule takes the reads and then one tangent per read and
    /// returns the tangent of the result. The reverse rules are one per read, in parameter order; rule i takes the reads and then
    /// the adjoint of the result and returns the gradient of read i, so a function with reverse rules has as many as it has reads.
    fn rules_fault(&self, who: &str, f: &FunctionFacts, returns: &Typed) -> Option<String> {
        let reads: Vec<Typed> = f.params.iter().map(|&p| self.typed_node(p)).collect();
        if let Some(rule) = f.jvp {
            let fault = self.rule_fault(
                who,
                "jvp",
                f,
                rule,
                (&reads, returns),
                "one tangent per read",
            );
            if fault.is_some() {
                return fault;
            }
        }
        if !f.vjp.is_empty() && f.vjp.len() != reads.len() {
            return Some(format!(
                "{who}: the vjp rules are one per read: {} given, the function has {} read(s)",
                f.vjp.len(),
                reads.len()
            ));
        }
        f.vjp.iter().enumerate().find_map(|(i, &rule)| {
            let label = format!("vjp {i}");
            let tail = std::slice::from_ref(returns);
            self.rule_fault(
                who,
                &label,
                f,
                rule,
                (tail, &reads[i]),
                "the adjoint of the result",
            )
        })
    }

    /// Why `rule`, called `label`, cannot be a rule of the function stated by `f`: it is a function of the same scope whose
    /// parameters are the reads and then `tail`, and whose result has the type `result`. `tail_hint` says what the tail is.
    fn rule_fault(
        &self,
        who: &str,
        label: &str,
        f: &FunctionFacts,
        rule: u32,
        (tail, result): (&[Typed], &Typed),
        tail_hint: &str,
    ) -> Option<String> {
        if !self.is_a(rule, "Function") {
            return Some(format!(
                "{who}: the {label} rule is a Function, got a {}",
                self.class_name(rule)
            ));
        }
        let rule_scope = self.text_of(rule, "scope").unwrap_or_default();
        if rule_scope != f.scope {
            return Some(format!("{who}: the {label} rule is a {rule_scope} function; a rule has the scope of its function"));
        }
        let want: Vec<Typed> = f
            .params
            .iter()
            .map(|&p| self.typed_node(p))
            .chain(tail.iter().cloned())
            .collect();
        let have = self.refs(rule, "params").unwrap_or_default();
        if have.len() != want.len() {
            return Some(format!(
                "{who}: the {label} rule takes {} parameter(s), it needs {}: the reads, then {tail_hint}",
                have.len(),
                want.len()
            ));
        }
        for (i, (&p, want)) in have.iter().zip(&want).enumerate() {
            if self.typed_node(p) != *want {
                return Some(format!(
                    "{who}: parameter {i} of the {label} rule has the type {}, it needs {}",
                    type_text(&self.typed_node(p)),
                    type_text(want)
                ));
            }
        }
        let got = self.typed_node(rule);
        (got != *result).then(|| {
            format!(
                "{who}: the {label} rule returns {}, it needs {}",
                type_text(&got),
                type_text(result)
            )
        })
    }

    /// Why the function stated by `f` reaches itself through the calls of its body and rules: the call graph is acyclic.
    fn recursion_fault(&self, who: &str, f: &FunctionFacts) -> Option<String> {
        let mut stack: Vec<u32> = f
            .body
            .into_iter()
            .chain(f.jvp)
            .chain(f.vjp.iter().copied())
            .collect();
        let mut seen = HashSet::new();
        while let Some(h) = stack.pop() {
            if !seen.insert(h) {
                continue;
            }
            if let Some((callee, digest)) = self.callee_of(h) {
                if callee == f.name {
                    return Some(format!("{who} calls itself; recursion is not written"));
                }
                stack.extend(self.function_named(&callee, &digest));
            }
            stack.extend(self.reached_from(h));
        }
        None
    }

    /// Why a call of the function `function` (a (name, digest) pair) cannot take `args`: no such function in the table, a digest
    /// the table does not hold, another number of arguments, or an argument of another type than its parameter.
    pub fn call_fault(&self, args: &[u32], function: &Val) -> Option<String> {
        let target = match self.callee_pair(function) {
            Ok(t) => t,
            Err(m) => return Some(m),
        };
        let (name, digest) = target;
        let named = self.functions_named(&name);
        if named.is_empty() {
            return Some(format!("call of unknown function '{name}': the function table holds no function of that name"));
        }
        let Some(f) = self.function_named(&name, &digest) else {
            return Some(format!("call of function '{name}' names the digest {digest}, which the function table does not hold for that name"));
        };
        let params = self.refs(f, "params").unwrap_or_default();
        if params.len() != args.len() {
            return Some(format!(
                "call of function '{name}' passes {} argument(s), the function takes {}",
                args.len(),
                params.len()
            ));
        }
        params.iter().zip(args).enumerate().find_map(|(i, (&p, &a))| {
            (self.typed_node(p) != self.typed_node(a)).then(|| {
                format!(
                    "call of function '{name}': argument {i} ('{}') has the type {}, the parameter takes {}",
                    self.plane_name(p),
                    type_text(&self.typed_node(a)),
                    type_text(&self.typed_node(p))
                )
            })
        })
    }

    /// The name and digest of a call's `function` value.
    fn callee_pair(&self, function: &Val) -> Result<(String, String), String> {
        match function {
            Val::Tuple(p) => match p.as_slice() {
                [Val::Str(name), Val::Str(digest)] => Ok((name.clone(), digest.clone())),
                _ => Err("a call names its function as a (name, digest) pair".into()),
            },
            _ => Err("a call names its function as a (name, digest) pair".into()),
        }
    }

    /// The type of the result of the call that names `function`.
    pub fn call_result(&self, function: &Val) -> Result<Val, String> {
        let (name, digest) = self.callee_pair(function)?;
        let f = self
            .function_named(&name, &digest)
            .ok_or_else(|| format!("call of unknown function '{name}'"))?;
        let (dtype, shape, tag) = self.typed_node(f);
        Ok(ttype_val(&dtype, &shape, tag.as_deref()))
    }
}
