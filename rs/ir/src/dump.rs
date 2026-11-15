//! The text dump of a node graph and its loader.
//!
//! Format `raptor.ir.dump/1`: the header line, then one line `n|<Class>|<dtype>|<shape>|<tag>|<operands>|<fields>` per node of the
//! reachable set (operands and every node reference held in a field), in ascending handle order, then one line `root|#p` per root.
//! Every node reference is written as the position p of the node among the `n|` lines. The dump holds the state of each node; keys
//! and digests are recomputed by whoever loads.
//!
//! A failed load leaves the arena as it was before the call: the nodes appended so far are dropped.
use crate::door::{guard, put};
use crate::record::{refuse, Ctx, Fault};
use crate::value::{parse, text_val, Val};
use std::ffi::{c_char, c_int};

/// The first line of a dump: the format and its version.
const HEADER: &str = "raptor.ir.dump/1";

/// Every node reference inside a value.
pub fn refs(v: &Val, out: &mut Vec<u32>) {
    match v {
        Val::Node(h) => out.push(*h),
        Val::Tuple(t) | Val::List(t) => t.iter().for_each(|x| refs(x, out)),
        Val::Rec(_, f) => f.iter().for_each(|(_, x)| refs(x, out)),
        _ => {}
    }
}

/// The value with every node reference replaced by `f` of it.
fn remap(v: &Val, f: &mut dyn FnMut(u32) -> Result<u32, String>) -> Result<Val, String> {
    Ok(match v {
        Val::Node(h) => Val::Node(f(*h)?),
        Val::Tuple(t) => Val::Tuple(t.iter().map(|x| remap(x, f)).collect::<Result<_, _>>()?),
        Val::List(t) => Val::List(t.iter().map(|x| remap(x, f)).collect::<Result<_, _>>()?),
        Val::Rec(n, fs) => Val::Rec(
            n.clone(),
            fs.iter()
                .map(|(k, x)| Ok((k.clone(), remap(x, f)?)))
                .collect::<Result<_, String>>()?,
        ),
        x => x.clone(),
    })
}

/// A pointer and a count as a slice; refuses NULL for a non-zero count.
fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Result<&'a [T], Fault> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(Fault::BadArg(format!("{what} is NULL")));
    }
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// The text of the nodes reachable from `roots`, one line per node.
fn dump(c: &Ctx, roots: &[u32]) -> Result<String, Fault> {
    let mut seen = vec![false; c.nodes.len()];
    let mut stack = vec![];
    for &r in roots {
        c.node(r)?;
        stack.push(r)
    }
    while let Some(h) = stack.pop() {
        if std::mem::replace(&mut seen[h as usize], true) {
            continue;
        }
        let n = &c.nodes[h as usize];
        stack.extend(&n.ops);
        n.fields.iter().for_each(|f| refs(f, &mut stack));
    }
    let mut pos = vec![0u32; seen.len()];
    let mut k = 0;
    for (h, s) in seen.iter().enumerate() {
        if *s {
            pos[h] = k;
            k += 1
        }
    }
    let mut out = format!("{HEADER}\n");
    let mut at = |h: u32| -> Result<u32, String> { Ok(pos[h as usize]) };
    for h in (0..seen.len()).filter(|&h| seen[h]) {
        let n = &c.nodes[h];
        let cl = &c.table.classes[n.class];
        let shape = Val::Tuple(n.shape.iter().map(|&d| Val::Int(d as i128)).collect());
        let ops = Val::Tuple(n.ops.iter().map(|&o| Val::Node(pos[o as usize])).collect());
        let mut fs = vec![];
        for (name, v) in cl.slots.iter().zip(&n.fields) {
            fs.push(Val::Tuple(vec![
                Val::Str(name.clone()),
                remap(v, &mut at).unwrap(),
            ]))
        }
        let tag = n.tag.clone().map_or(Val::None, Val::Str);
        out += &format!(
            "n|{}|{}|{}|{}|{}|{}\n",
            cl.name,
            n.dtype,
            text_val(&shape),
            text_val(&tag),
            text_val(&ops),
            text_val(&Val::Tuple(fs))
        );
    }
    for &r in roots {
        out += &format!("root|#{}\n", pos[r as usize])
    }
    Ok(out)
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_dump(
    ctx: *mut Ctx,
    roots: *const u32,
    n: usize,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let s = dump(c, slice(roots, n, "roots")?)?;
        put(buf, cap, len, &s)
    })
}

/// The part of `s` up to the first `|`, and the rest after it.
fn part(s: &str) -> Option<(&str, &str)> {
    s.split_once('|')
}

/// The tag part: a quoted string may hold `|`, so it is scanned to its closing quote.
fn tag_part(s: &str) -> Option<(&str, &str)> {
    let Some(q) = s.chars().next().filter(|c| *c == '\'' || *c == '"') else {
        return part(s);
    };
    let (mut esc, mut end) = (false, None);
    for (i, ch) in s.char_indices().skip(1) {
        if esc {
            esc = false
        } else if ch == '\\' {
            esc = true
        } else if ch == q {
            end = Some(i + 1);
            break;
        }
    }
    let end = end?;
    s[end..].strip_prefix('|').map(|r| (&s[..end], r))
}

/// One parsed line of a dump.
struct Line {
    /// The class name.
    class: String,
    /// The element type name.
    dtype: String,
    /// The extents.
    shape: Vec<i64>,
    /// The tag, if any.
    tag: Option<String>,
    /// The operand lines, by position.
    ops: Vec<u32>,
    /// The field values.
    fields: Val,
}

/// Parses one node line of a dump.
fn node_line(s: &str) -> Result<Line, String> {
    let bad = || "malformed node line".to_string();
    let (class, s) = part(s).ok_or_else(bad)?;
    let (dtype, s) = part(s).ok_or_else(bad)?;
    let (shape, s) = part(s).ok_or_else(bad)?;
    let (tag, s) = tag_part(s).ok_or_else(bad)?;
    let (ops, fields) = part(s).ok_or_else(bad)?;
    let shape = match parse(shape)? {
        Val::Tuple(t) => t
            .into_iter()
            .map(|d| {
                if let Val::Int(d) = d {
                    i64::try_from(d).map_err(|_| "shape extent out of range".to_string())
                } else {
                    Err("shape holds a non-int".into())
                }
            })
            .collect::<Result<_, _>>()?,
        _ => return Err("shape is not a tuple".into()),
    };
    let tag = match parse(tag)? {
        Val::None => None,
        Val::Str(t) => Some(t),
        _ => return Err("tag is neither None nor a string".into()),
    };
    let ops = match parse(ops)? {
        Val::Tuple(t) => t
            .into_iter()
            .map(|o| {
                if let Val::Node(p) = o {
                    Ok(p)
                } else {
                    Err("operands hold a non-reference".to_string())
                }
            })
            .collect::<Result<_, _>>()?,
        _ => return Err("operands are not a tuple".into()),
    };
    Ok(Line {
        class: class.into(),
        dtype: dtype.into(),
        shape,
        tag,
        ops,
        fields: parse(fields).map_err(|e| format!("unparsable field: {e}"))?,
    })
}

/// Loads the nodes of a dump into the context; returns their handles in order.
fn load(c: &mut Ctx, text: &str) -> Result<Vec<u32>, Fault> {
    if text.is_empty() {
        return refuse("line 1: missing header");
    }
    let body = text.strip_suffix('\n');
    let lines: Vec<&str> = body.unwrap_or(text).split('\n').collect();
    let at = |i: usize, m: &str| -> Fault { Fault::Refused(format!("line {}: {m}", i + 1)) };
    if lines[0] != HEADER {
        return Err(at(
            0,
            &if lines[0].starts_with("raptor.ir.dump/") {
                format!("unknown version {:?}", lines[0])
            } else {
                "wrong or missing header".into()
            },
        ));
    }
    if body.is_none() {
        return Err(at(lines.len() - 1, "text after the last newline"));
    }
    let mut map: Vec<u32> = vec![];
    let mut roots = vec![];
    for (i, l) in lines.iter().enumerate().skip(1) {
        if let Some(r) = l.strip_prefix("root|") {
            match parse(r) {
                Ok(Val::Node(p)) if (p as usize) < map.len() => roots.push(map[p as usize]),
                Ok(Val::Node(p)) => return Err(at(i, &format!("root #{p} is out of range"))),
                _ => return Err(at(i, "a root is #<position>")),
            }
        } else if let Some(s) = l.strip_prefix("n|") {
            if !roots.is_empty() {
                return Err(at(i, "node line after a root line"));
            }
            let ln = node_line(s).map_err(|m| at(i, &m))?;
            let mut f = |p: u32| {
                map.get(p as usize)
                    .copied()
                    .ok_or(format!("reference #{p} is not an earlier position"))
            };
            let ops: Vec<u32> = ln
                .ops
                .iter()
                .map(|&p| f(p))
                .collect::<Result<_, _>>()
                .map_err(|m| at(i, &m))?;
            let fields = remap(&ln.fields, &mut f).map_err(|m| at(i, &m))?;
            let h = c
                .make(
                    &ln.class,
                    &ln.dtype,
                    &ln.shape,
                    ln.tag.as_deref(),
                    &ops,
                    &text_val(&fields),
                )
                .map_err(|e| match e {
                    Fault::Refused(m) | Fault::BadArg(m) => at(i, &m),
                    e => e,
                })?;
            map.push(h);
        } else {
            return Err(at(i, "a line is neither n| nor root|"));
        }
    }
    Ok(roots)
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_load(
    ctx: *mut Ctx,
    text: *const c_char,
    n: usize,
    roots: *mut u32,
    cap: usize,
    nroots: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        if nroots.is_null() {
            return Err(Fault::BadArg("nroots is NULL".into()));
        }
        let text = std::str::from_utf8(slice(text as *const u8, n, "text")?)
            .map_err(|_| Fault::BadArg("text is not UTF-8".into()))?;
        let before = c.nodes.len();
        let r = load(c, text).and_then(|r| {
            unsafe { *nroots = r.len() }
            if cap < r.len() || (roots.is_null() && !r.is_empty()) {
                return Err(Fault::Truncated);
            }
            unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), roots, r.len()) }
            Ok(())
        });
        if r.is_err() {
            c.nodes.truncate(before)
        }
        r
    })
}
