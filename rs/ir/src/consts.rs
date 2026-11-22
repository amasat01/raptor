//! The constants raptor owns, by name, as JSON text: the vocabularies and policy names of the manifest the crate embeds.
use crate::door::{cstr, guard, put};
use crate::record::{refuse, Ctx, Fault};
use rcore::{Fact, CONSTS, FACT_ROWS, STR_CONSTS};
use std::ffi::{c_char, c_int};

/// Writes `s` as a JSON string into `o`.
fn json_str(s: &str, o: &mut String) {
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
}

/// A list of texts as a JSON array.
fn list(v: &[&str]) -> String {
    let mut o = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            o.push(',')
        }
        json_str(x, &mut o)
    }
    o.push(']');
    o
}

/// The value of a constant as JSON text: a list of strings, a string or an integer.
pub fn constant(name: &str) -> Option<String> {
    if let Some(c) = CONSTS.iter().find(|c| c.0 == name) {
        return Some(list(c.1));
    }
    if let Some(c) = STR_CONSTS.iter().find(|c| c.0 == name) {
        let mut o = String::new();
        json_str(c.1, &mut o);
        return Some(o);
    }
    FACT_ROWS.iter().find(|f| f.0 == name).map(|f| match &f.1 {
        Fact::I(i) => i.to_string(),
        Fact::S(s) => {
            let mut o = String::new();
            json_str(s, &mut o);
            o
        }
        Fact::L(l) => list(l),
    })
}

/// How many names `constant` answers.
pub fn count() -> usize {
    CONSTS.len() + STR_CONSTS.len() + FACT_ROWS.len()
}

/// The value of a constant as JSON text, through the two-call text protocol. Not exported: only the
/// replay tools and the door tests read a constant as JSON; a backend asks `rx_const_names`.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` bytes and `len` for one write.
pub unsafe extern "C" fn rx_const(
    ctx: *mut Ctx,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let n = cstr(name, "name")?;
        match constant(n) {
            Some(t) => put(buf, cap, len, &t),
            None => {
                let _ = c;
                refuse(format!("raptor owns no constant named {n:?}"))
            }
        }
    })
}

/// The texts of a constant that is a list of strings (a string constant answers its one text), in
/// order, each NUL-terminated, back to back
/// in `buf`; `*count` is the number of texts and `*len` the bytes they take with their NULs (the
/// text protocol: `cap` is at least `*len + 1`, RX_TRUNC when smaller). Refuses a name raptor owns
/// no list under.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` bytes; `len` and `count` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn rx_const_names(
    ctx: *mut Ctx,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |_| {
        let n = cstr(name, "name")?;
        let Some(items) = names(n) else {
            return refuse(format!("raptor owns no list of names under {n:?}"));
        };
        put_names(&items, buf, cap, len, count)
    })
}

/// Writes `items` as the names protocol: each text NUL-terminated, back to back in `buf`, `*count` the number of texts and
/// `*len` the bytes they take with their NULs. The doors of other crates call it with their caller's pointers, so it stays callable
/// without `unsafe`, as `door::put` does.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn put_names<S: AsRef<str>>(
    items: &[S],
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> Result<(), Fault> {
    let mut packed = String::new();
    for item in items {
        packed.push_str(item.as_ref());
        packed.push('\0');
    }
    if count.is_null() {
        return Err(Fault::BadArg("count is NULL".into()));
    }
    unsafe { *count = items.len() }
    put(buf, cap, len, &packed)
}

/// Whether the update `node` commits through a companion plane (`*out` is 0 or 1): an add under the compensated summation.
///
/// # Safety
/// `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_compensates(ctx: *mut Ctx, node: u32, out: *mut c_int) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(crate::record::Fault::BadArg("out is NULL".into()));
        }
        unsafe { *out = c.compensates(node)? as c_int };
        Ok(())
    })
}

/// The lists the core composes from the manifest's lists: every linear-algebra kind, the cast kind, and every op kind a body can
/// mint (the order is the one `all_kinds` gives).
fn composed_names(name: &str) -> Option<Vec<&'static str>> {
    match name {
        "LINALG_OPS" => Some(
            [
                "LINALG_FACTORISATIONS",
                "LINALG_CONSUMERS",
                "LINALG_PROJECTIONS",
            ]
            .into_iter()
            .flat_map(|n| CONSTS.iter().find(|c| c.0 == n).map_or(&[][..], |c| c.1))
            .copied()
            .collect(),
        ),
        "CAST" => Some(vec!["cast"]),
        "OP_KINDS" => Some(crate::typing::all_kinds()),
        _ => None,
    }
}

/// The texts of a constant that is a list of strings, or the one text of a constant that is a
/// string; `None` for any other name or kind (an integer).
pub fn names(name: &str) -> Option<Vec<&'static str>> {
    if let Some(composed) = composed_names(name) {
        return Some(composed);
    }
    if let Some(c) = CONSTS.iter().find(|c| c.0 == name) {
        return Some(c.1.to_vec());
    }
    if let Some(c) = STR_CONSTS.iter().find(|c| c.0 == name) {
        return Some(vec![c.1]);
    }
    match FACT_ROWS.iter().find(|f| f.0 == name).map(|f| &f.1) {
        Some(Fact::L(l)) => Some(l.to_vec()),
        Some(Fact::S(s)) => Some(vec![*s]),
        _ => None,
    }
}
