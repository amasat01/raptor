//! Loop bodies: the nodes of a body, their structural keys and the body digest that a loop's dedup identity holds.
use crate::door::guard;
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::value::{text_val, Val};
use std::collections::{HashMap, HashSet};
use std::ffi::c_int;

/// The number of hex digits of a body key.
const BODY_KEY_HEX: usize = rcore::facts::BODY_KEY_HEX as usize;

impl Ctx {
    /// The node a field holds, or None when it holds none.
    fn node_field(&self, h: u32, f: &str) -> Result<Option<u32>, Fault> {
        match self.field(h, f)? {
            Val::Node(n) => Ok(Some(n)),
            Val::None => Ok(None),
            _ => refuse(format!("field {f:?} is not a node")),
        }
    }

    /// What the body evaluates, in emission order: the carried results, every sink operand, then the break condition and values.
    pub fn body_roots(&self, lp: u32) -> Result<Vec<u32>, Fault> {
        let mut roots = self.refs(lp, "nexts")?;
        for sk in self.refs(lp, "body_sinks")? {
            roots.extend(self.node(sk)?.ops.iter().copied())
        }
        if let Some(ec) = self.node_field(lp, "exit_cond")? {
            roots.push(ec);
            roots.extend(self.refs(lp, "exit_values")?)
        }
        Ok(roots)
    }

    /// Where the break sits, from its values: "head", "tail", "middle", or None without a break.
    pub fn exit_form(&self, lp: u32) -> Result<Val, Fault> {
        if self.node_field(lp, "exit_cond")?.is_none() {
            return Ok(Val::None);
        }
        let vals = self.refs(lp, "exit_values")?;
        let same = |f: &str| -> Result<bool, Fault> {
            Ok(vals.iter().zip(self.refs(lp, f)?).all(|(v, c)| *v == c))
        };
        Ok(Val::Str(
            if same("carries")? {
                "head"
            } else if same("nexts")? {
                "tail"
            } else {
                "middle"
            }
            .into(),
        ))
    }

    /// Every node of the body, post-order, by record identity; the loop's operands and its own placeholders are excluded.
    pub fn body_nodes(&self, lp: u32) -> Result<Vec<u32>, Fault> {
        let boundary: HashSet<u32> = self.node(lp)?.ops.iter().copied().collect();
        let (mut seen, mut order) = (HashSet::new(), vec![]);
        let mut stack: Vec<(u32, bool)> = self
            .body_roots(lp)?
            .into_iter()
            .rev()
            .map(|n| (n, false))
            .collect();
        while let Some((n, expanded)) = stack.pop() {
            if seen.contains(&n) || boundary.contains(&n) {
                continue;
            }
            self.node(n)?;
            if self.is_a(n, "LoopIndex") || self.is_a(n, "LoopCarry") {
                continue;
            }
            if !expanded {
                stack.push((n, true));
                for &c in self.nodes[n as usize].ops.iter().rev() {
                    if !seen.contains(&c) {
                        stack.push((c, false))
                    }
                }
                continue;
            }
            seen.insert(n);
            order.push(n);
        }
        Ok(order)
    }

    /// The key of a node reached from a loop body: placeholders are keyed by position.
    fn key_of_child(&self, c: u32, key_of: &mut HashMap<u32, String>) -> Result<String, Fault> {
        if let Some(k) = key_of.get(&c) {
            return Ok(k.clone());
        }
        let held = if self.is_a(c, "LoopIndex") {
            "index".to_string()
        } else if self.is_a(c, "LoopCarry") {
            match self.field(c, "slot")? {
                Val::Int(i) => format!("carry{i}"),
                _ => return refuse("a carry's slot is an int"),
            }
        } else {
            return refuse(format!("Loop body: a {} is neither a boundary value, a placeholder nor a body node - the boundary computation and the body traversal disagree", self.table.classes[self.nodes[c as usize].class].name));
        };
        key_of.insert(c, held.clone());
        Ok(held)
    }

    /// Boundary values by position in the loop's operands (the first position wins), placeholders by slot, every other node by a
    /// fixed-width hash of (kind, dtype, shape, tag, child keys, dedup identity).
    fn body_keys(&self, lp: u32) -> Result<(HashMap<u32, String>, Vec<u32>), Fault> {
        let mut key_of: HashMap<u32, String> = HashMap::new();
        for (k, &o) in self.node(lp)?.ops.iter().enumerate() {
            key_of.entry(o).or_insert(format!("free{k}"));
        }
        let body = self.body_nodes(lp)?;
        for &n in &body {
            let mut parts = vec![];
            for &c in &self.nodes[n as usize].ops {
                parts.push(Val::Str(self.key_of_child(c, &mut key_of)?))
            }
            let node = &self.nodes[n as usize];
            let raw = format!(
                "({}, {}, {}, {}, {}, {})",
                text_val(&Val::Str(self.kind_of(n)?)),
                text_val(&Val::Str(node.dtype.clone())),
                text_val(&Val::Tuple(
                    node.shape.iter().map(|&d| Val::Int(d as i128)).collect()
                )),
                text_val(&node.tag.clone().map_or(Val::None, Val::Str)),
                text_val(&Val::Tuple(parts)),
                text_val(&self.dedup_val(n)?)
            );
            key_of.insert(
                n,
                rcore::sha256::hex(raw.as_bytes())[..BODY_KEY_HEX].to_string(),
            );
        }
        Ok((key_of, body))
    }

    /// A structural hash of the loop's body, held per context.
    pub fn body_digest(&self, lp: u32) -> Result<String, Fault> {
        if let Some(d) = self.body_cache.borrow().get(&lp) {
            return Ok(d.clone());
        }
        let (mut key_of, body) = self.body_keys(lp)?;
        let mut roots = vec![];
        for r in self.body_roots(lp)? {
            roots.push(match key_of.get(&r) {
                Some(k) => Val::Str(k.clone()),
                None => {
                    if self.is_a(r, "LoopIndex") || self.is_a(r, "LoopCarry") {
                        Val::Str(self.key_of_child(r, &mut key_of)?)
                    } else {
                        Val::None
                    }
                }
            });
        }
        let mut data = b"raptor.ir.loop/3\n".to_vec();
        for n in &body {
            data.extend_from_slice(key_of[n].as_bytes());
            data.push(b'\n')
        }
        data.extend_from_slice(format!("roots|{}\n", text_val(&Val::List(roots))).as_bytes());
        let d = rcore::sha256::hex(&data)[..32].to_string();
        self.body_cache.borrow_mut().insert(lp, d.clone());
        Ok(d)
    }

    /// The nodes the access and dispatch passes see: each node, a loop's body sinks and, recursively, the nodes of its body.
    pub fn with_loop_bodies(&self, order: &[u32], out: &mut Vec<u32>) -> Result<(), Fault> {
        for &n in order {
            out.push(n);
            if self.is_a(n, "Loop") {
                out.extend(self.refs(n, "body_sinks")?);
                let body = self.body_nodes(n)?;
                self.with_loop_bodies(&body, out)?;
            }
        }
        Ok(())
    }

    /// The At node that reads an exempt `lookup` plane at a constant index inside a loop's count: the loop's own trip count.
    pub fn step_word(&self, count: Option<u32>, exempt: &[String]) -> Result<Option<u32>, Fault> {
        let Some(c) = count else { return Ok(None) };
        let (mut stack, mut seen) = (vec![c], HashSet::new());
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if self.is_a(n, "At") {
                let plane = self.nodes[n as usize].ops[0];
                if self.str_field(plane, "role")? == "lookup"
                    && exempt.contains(&self.str_field(plane, "name")?)
                    && self.is_a(self.nodes[n as usize].ops[1], "Const")
                {
                    return Ok(Some(n));
                }
            }
            stack.extend(self.nodes[n as usize].ops.iter().copied());
        }
        Ok(None)
    }

    /// The Dispatch nodes of an order: the top-level ones, then those inside lowered loop bodies (first in, first out).
    pub fn dispatch_nodes(&self, order: &[u32]) -> Result<Vec<u32>, Fault> {
        let (mut top, mut seen) = (vec![], HashSet::new());
        let mut loops: std::collections::VecDeque<u32> = Default::default();
        for &n in order {
            if self.is_a(n, "Dispatch") {
                seen.insert(n);
                top.push(n)
            }
            if self.is_a(n, "Loop") {
                loops.push_back(n)
            }
        }
        while let Some(l) = loops.pop_front() {
            for n in self.body_nodes(l)? {
                if self.is_a(n, "Dispatch") && !seen.contains(&n) {
                    seen.insert(n);
                    top.push(n)
                } else if self.is_a(n, "Loop") {
                    loops.push_back(n)
                }
            }
        }
        Ok(top)
    }
}

/// Copies handles to the caller's array; sets the count and refuses a short buffer.
pub(crate) fn give(buf: *mut u32, cap: usize, n: *mut usize, v: &[u32]) -> Result<(), Fault> {
    if n.is_null() {
        return badarg("n is NULL");
    }
    unsafe { *n = v.len() }
    if cap < v.len() || (buf.is_null() && !v.is_empty()) {
        return Err(Fault::Truncated);
    }
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), buf, v.len()) }
    Ok(())
}

impl Ctx {
    /// The loop node `h`; refuses a node that is not a loop.
    fn loop_of(&self, h: u32) -> Result<u32, Fault> {
        self.node(h)?;
        if self.is_a(h, "Loop") {
            Ok(h)
        } else {
            refuse(format!(
                "node {h} is a {}, not a Loop",
                self.table.classes[self.nodes[h as usize].class].name
            ))
        }
    }
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_body_nodes(
    ctx: *mut Ctx,
    lp: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let l = c.loop_of(lp)?;
        let v = c.body_nodes(l)?;
        give(buf, cap, n, &v)
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_body_roots(
    ctx: *mut Ctx,
    lp: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let l = c.loop_of(lp)?;
        let v = c.body_roots(l)?;
        give(buf, cap, n, &v)
    })
}
