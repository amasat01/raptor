//! The typed answers of the query door: a node's type as a C struct, a field read as text, integer,
//! real, flag, node
//! references or names (absence is the status RX_ABSENT), and whether a node lies inside a loop
//! body.
use crate::door::{cstr, guard, put, Node};
use crate::loops::give;
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::value::{lift_refs, Val};
use std::ffi::{c_char, c_int};

/// The largest rank `rx_type` carries; a node of a higher rank is refused.
pub const MAX_RANK: usize = 8;
/// The size of the tag buffer of `rx_type`, with its NUL; a longer tag is refused.
pub const TAG_CAP: usize = 64;

/// A node's type as the C struct `rx_type` of raptor_ir_query.h.
#[repr(C)]
pub struct RxType {
    /// The number of extents.
    pub rank: u32,
    /// The extents; those past `rank` are zero.
    pub dims: [i64; MAX_RANK],
    /// The dtype code: the position in the core's dtype list.
    pub dtype: c_int,
    /// The tag, NUL-terminated; empty for none.
    pub tag: [c_char; TAG_CAP],
}

/// The dtype code of a dtype name: its position in the core's list, the numbering of the header's
/// enum.
fn dtype_code(name: &str) -> Result<c_int, Fault> {
    match rcore::facts::DTYPES.iter().position(|d| *d == name) {
        Some(code) => Ok(code as c_int),
        None => refuse(format!("dtype {name:?} has no code")),
    }
}

/// A type's parts as the C struct: refuses a rank over `MAX_RANK` or a tag over `TAG_CAP - 1` bytes.
pub fn rx_type_of_parts(dtype: &str, shape: &[i64], tag: Option<&str>) -> Result<RxType, Fault> {
    if shape.len() > MAX_RANK {
        return refuse(format!("rank {} is over {MAX_RANK}", shape.len()));
    }
    let tag = tag.unwrap_or("");
    if tag.len() >= TAG_CAP {
        return refuse(format!(
            "tag of {} bytes is over {}",
            tag.len(),
            TAG_CAP - 1
        ));
    }
    let mut out = RxType {
        rank: shape.len() as u32,
        dims: [0; MAX_RANK],
        dtype: dtype_code(dtype)?,
        tag: [0; TAG_CAP],
    };
    out.dims[..shape.len()].copy_from_slice(shape);
    for (slot, byte) in out.tag.iter_mut().zip(tag.bytes()) {
        *slot = byte as c_char;
    }
    Ok(out)
}

impl Ctx {
    /// The type of `node` as the C struct: refuses a rank over `MAX_RANK` or a tag over `TAG_CAP -
    /// 1` bytes.
    pub fn type_of(&self, node: Node) -> Result<RxType, Fault> {
        let n = self.node(node)?;
        rx_type_of_parts(&n.dtype, &n.shape, n.tag.as_deref())
    }

    /// The value of a field that holds something: a field whose value is None is `Fault::Absent`.
    fn present(&self, node: Node, name: &str) -> Result<Val, Fault> {
        match self.field(node, name)? {
            Val::None => Err(Fault::Absent),
            value => Ok(value),
        }
    }
}

/// Writes a scalar through an out pointer that must not be NULL.
fn store<T>(out: *mut T, value: T) -> Result<(), Fault> {
    if out.is_null() {
        return Err(Fault::BadArg("out is NULL".into()));
    }
    unsafe { *out = value };
    Ok(())
}

/// `rx_field_kind` answer: the field holds None.
const KIND_ABSENT: c_int = 0;
/// `rx_field_kind` answer: the field holds text.
const KIND_TEXT: c_int = 1;
/// `rx_field_kind` answer: the field holds an integer.
const KIND_INTEGER: c_int = 2;
/// `rx_field_kind` answer: the field holds a real number.
const KIND_REAL: c_int = 3;
/// `rx_field_kind` answer: the field holds a flag.
const KIND_FLAG: c_int = 4;
/// `rx_field_kind` answer: the field holds node references.
const KIND_NODES: c_int = 5;
/// `rx_field_kind` answer: the field holds a list of texts.
const KIND_NAMES: c_int = 6;
/// `rx_field_kind` answer: the field holds a list of integers.
const KIND_INTS: c_int = 7;
/// `rx_field_kind` answer: the field holds something else.
const KIND_OTHER: c_int = 8;

/// The type of a node: rank, extents, dtype code (the enum of raptor_ir_query.h) and tag (empty for
/// none).
/// Refuses a node of a rank over RX_MAX_RANK or a tag of RX_TAG_CAP bytes or more.
///
/// # Safety
/// `out` must be valid for one write of `rx_type`.
#[no_mangle]
pub unsafe extern "C" fn rx_ttype_of(ctx: *mut Ctx, node: Node, out: *mut RxType) -> c_int {
    guard(ctx, |c| {
        let ty = c.type_of(node)?;
        store(out, ty)
    })
}

/// A field that holds text, copied without quotes; RX_ABSENT when the field holds None, a refusal
/// when it holds
/// anything but text or the node has no such field.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` bytes and `len` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_text(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |c| match c.present(node, cstr(name, "name")?)? {
        Val::Str(text) => put(buf, cap, len, &text),
        _ => refuse("the field is not text"),
    })
}

/// A field that holds an integer; RX_ABSENT when it holds None, a refusal for any other kind of
/// value or one that
/// does not fit 64 bits.
///
/// # Safety
/// `name` is a NUL-terminated string; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_int(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    out: *mut i64,
) -> c_int {
    guard(ctx, |c| match c.present(node, cstr(name, "name")?)? {
        Val::Int(v) => match i64::try_from(v) {
            Ok(v) => store(out, v),
            Err(_) => refuse("the integer does not fit 64 bits"),
        },
        _ => refuse("the field is not an integer"),
    })
}

/// A field that holds a real number (an integer is read as its real value); RX_ABSENT when it holds
/// None, a
/// refusal for any other kind of value.
///
/// # Safety
/// `name` is a NUL-terminated string; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_real(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    out: *mut f64,
) -> c_int {
    guard(ctx, |c| match c.present(node, cstr(name, "name")?)? {
        Val::Float(v) => store(out, v),
        Val::Int(v) => store(out, v as f64),
        _ => refuse("the field is not a real number"),
    })
}

/// A field that holds a flag (*out is 0 or 1); RX_ABSENT when it holds None, a refusal for any
/// other kind of value.
///
/// # Safety
/// `name` is a NUL-terminated string; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_flag(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| match c.present(node, cstr(name, "name")?)? {
        Val::Bool(v) => store(out, v as c_int),
        _ => refuse("the field is not a flag"),
    })
}

/// The node references a field holds, each distinct node once, in order of first appearance (a
/// reference may sit
/// inside a tuple, a list or a record); `*n` is the count, with the status RX_TRUNC when `cap` is
/// smaller.
/// RX_ABSENT when the field holds None; a field holding no reference at all answers an empty array.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` nodes and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_nodes(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    buf: *mut Node,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let mut refs = vec![];
        lift_refs(c.present(node, cstr(name, "name")?)?, &mut refs);
        give(buf, cap, n, &refs)
    })
}

/// Appends every node reference of `value` in the order it appears, repeats kept.
fn every_ref(value: &Val, out: &mut Vec<u32>) {
    match value {
        Val::Node(handle) => out.push(*handle),
        Val::Tuple(items) | Val::List(items) => items.iter().for_each(|item| every_ref(item, out)),
        Val::Rec(_, fields) => fields.iter().for_each(|(_, item)| every_ref(item, out)),
        _ => {}
    }
}

/// The node references a field holds in the order they appear, a node listed as often as the
/// field names it (a positional list: the carried values of a loop may name one node twice);
/// `*n` is the count, with the status RX_TRUNC when `cap` is smaller. RX_ABSENT when the field
/// holds None; a field holding no reference at all answers an empty array.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` nodes and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_node_list(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    buf: *mut Node,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let mut refs = vec![];
        every_ref(&c.present(node, cstr(name, "name")?)?, &mut refs);
        give(buf, cap, n, &refs)
    })
}

/// The names a field holds as a tuple or list of texts: `*count` names, each NUL-terminated, back
/// to back in `buf`;
/// `*len` is the bytes needed with the NULs (RX_TRUNC when `cap` is smaller). RX_ABSENT when the
/// field holds None;
/// a refusal when it holds anything else or an element that is not text.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` bytes; `len` and `count` for one
/// write each.
#[no_mangle]
pub unsafe extern "C" fn rx_field_names(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let (Val::Tuple(items) | Val::List(items)) = c.present(node, cstr(name, "name")?)? else {
            return refuse("the field is not a list of names");
        };
        let mut packed = String::new();
        for item in &items {
            let Val::Str(text) = item else {
                return refuse("the field holds an element that is not text");
            };
            packed.push_str(text);
            packed.push('\0');
        }
        store(count, items.len())?;
        put(buf, cap, len, &packed)
    })
}

/// The kind of value a field holds, as one of the `RX_KIND_*` codes of raptor_ir_query.h: absent
/// (None), text, integer, real, flag, nodes (a node or a tuple or list of references), names (a
/// non-empty tuple or list of texts), ints (a tuple or list of integers, empty included) or
/// other. The caller picks the typed reader from the answer. Refuses a node without the field.
///
/// # Safety
/// `name` is a NUL-terminated string; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_kind(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        let kind = match c.field(node, cstr(name, "name")?)? {
            Val::None => KIND_ABSENT,
            Val::Str(_) => KIND_TEXT,
            Val::Int(_) => KIND_INTEGER,
            Val::Float(_) => KIND_REAL,
            Val::Bool(_) => KIND_FLAG,
            Val::Node(_) => KIND_NODES,
            Val::Tuple(items) | Val::List(items) => list_kind(&items),
            Val::Rec(..) => KIND_OTHER,
        };
        store(out, kind)
    })
}

/// The kind code of a list value: ints when every element is an integer (or none), names when
/// every element is text, nodes when every element is a reference, else other.
fn list_kind(items: &[Val]) -> c_int {
    if items.iter().all(|item| matches!(item, Val::Int(_))) {
        KIND_INTS
    } else if items.iter().all(|item| matches!(item, Val::Str(_))) {
        KIND_NAMES
    } else if items.iter().all(|item| matches!(item, Val::Node(_))) {
        KIND_NODES
    } else {
        KIND_OTHER
    }
}

/// A field that holds a tuple or list of integers: `*n` is the count, with the status RX_TRUNC
/// when `cap` is smaller. RX_ABSENT when the field holds None; a refusal when it holds anything
/// else, an element that is not an integer or one that does not fit 64 bits.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` integers and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_ints(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    buf: *mut i64,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let (Val::Tuple(items) | Val::List(items)) = c.present(node, cstr(name, "name")?)? else {
            return refuse("the field is not a list of integers");
        };
        let mut ints = vec![];
        for item in &items {
            let Val::Int(v) = item else {
                return refuse("the field holds an element that is not an integer");
            };
            let Ok(v) = i64::try_from(*v) else {
                return refuse("the integer does not fit 64 bits");
            };
            ints.push(v);
        }
        if n.is_null() {
            return badarg("n is NULL");
        }
        unsafe { *n = ints.len() }
        if cap < ints.len() || (buf.is_null() && !ints.is_empty()) {
            return Err(Fault::Truncated);
        }
        unsafe { std::ptr::copy_nonoverlapping(ints.as_ptr(), buf, ints.len()) }
        Ok(())
    })
}

/// Whether `node` lies inside a loop body of the walk: 1 when the walk's flat traversal has no
/// position for it (a
/// loop body's nodes are the ones it skips), 0 when it has one. Refuses a node or a walk the
/// context does not hold.
///
/// # Safety
/// `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_in_loop_body(
    ctx: *mut Ctx,
    walk: u32,
    node: Node,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        c.node(node)?;
        let inside = !c.walk(walk)?.positions.contains_key(&node);
        store(out, inside as c_int)
    })
}

/// Whether the walk binds the slot (`role`, `name`): 1 when its argument list holds that pair, 0
/// when not. Refuses a walk the context does not hold.
///
/// # Safety
/// `role` and `name` are NUL-terminated strings; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_binds(
    ctx: *mut Ctx,
    walk: u32,
    role: *const c_char,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        let (role, name) = (cstr(role, "role")?, cstr(name, "name")?);
        let bound = c
            .walk(walk)?
            .leaves
            .iter()
            .any(|s| s.0 .0 == role && s.0 .1 == name);
        store(out, bound as c_int)
    })
}

/// The walk's argument list: the slots it binds as (role, name) pairs, in argument order. `buf`
/// holds the texts back to back, each NUL-terminated, role then name for each pair; `*count` is the
/// number of pairs and `*len` the bytes they take with their NULs (the text protocol: `cap` is at
/// least `*len + 1`, RX_TRUNC when smaller). Refuses a walk the context does not hold.
///
/// # Safety
/// `buf` is valid for `cap` bytes; `len` and `count` for one write each.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_args(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let slots = &c.walk(walk)?.leaves;
        let mut packed = String::new();
        for ((role, name), _) in slots {
            packed.push_str(role);
            packed.push('\0');
            packed.push_str(name);
            packed.push('\0');
        }
        store(count, slots.len())?;
        put(buf, cap, len, &packed)
    })
}

/// One part of a pair: absent, a whole number, or a text. `kind` tells which of `number` and `text` holds the value.
#[repr(C)]
pub struct RxPairPart {
    /// 0: absent; 1: a whole number in `number`; 2: a text in `text`.
    pub kind: c_int,
    /// The whole number of a part of kind 1.
    pub number: i64,
    /// The text of a part of kind 2, NUL-terminated.
    pub text: [c_char; TAG_CAP],
}

/// A pair, each part possibly absent: what a tuple of pairs such as the kept lane axes (label, extent) holds.
#[repr(C)]
pub struct RxPair {
    /// The first part.
    pub first: RxPairPart,
    /// The second part.
    pub second: RxPairPart,
}

/// A value as one part of a pair: None, a whole number or a text shorter than `TAG_CAP`.
pub fn pair_part(v: &Val) -> Result<RxPairPart, Fault> {
    let mut part = RxPairPart {
        kind: 0,
        number: 0,
        text: [0; TAG_CAP],
    };
    match v {
        Val::None => {}
        Val::Int(i) => {
            part.kind = 1;
            part.number =
                i64::try_from(*i).or_else(|_| refuse("the integer does not fit 64 bits"))?;
        }
        Val::Str(t) if t.len() < TAG_CAP => {
            part.kind = 2;
            for (slot, byte) in part.text.iter_mut().zip(t.bytes()) {
                *slot = byte as c_char;
            }
        }
        _ => return refuse("a part of a pair is None, a whole number or a short text"),
    }
    Ok(part)
}

/// A pair of two texts, the second possibly absent.
pub fn text_pair(first: &str, second: Option<&str>) -> Result<RxPair, Fault> {
    let second = second.map_or(Val::None, |t| Val::Str(t.into()));
    Ok(RxPair {
        first: pair_part(&Val::Str(first.into()))?,
        second: pair_part(&second)?,
    })
}

/// Copies `pairs` into the caller's buffer by the two-call protocol: `*n` is the count, and the status is RX_TRUNC when `cap` is smaller.
/// The doors of other crates call it with their caller's pointers, so it stays callable without `unsafe`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn put_pairs(
    pairs: &[RxPair],
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> Result<(), Fault> {
    if n.is_null() {
        return badarg("n is NULL");
    }
    unsafe { *n = pairs.len() }
    if cap < pairs.len() || (buf.is_null() && !pairs.is_empty()) {
        return Err(Fault::Truncated);
    }
    unsafe { std::ptr::copy_nonoverlapping(pairs.as_ptr(), buf, pairs.len()) }
    Ok(())
}

/// A field that holds a tuple or list of pairs: `*n` is the count, with the status RX_TRUNC when `cap` is smaller. RX_ABSENT when
/// the field holds None; a refusal when it holds anything else or an element that is no pair of None, whole numbers and short texts.
///
/// # Safety
/// `name` is a NUL-terminated string; `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_field_pairs(
    ctx: *mut Ctx,
    node: Node,
    name: *const c_char,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let (Val::Tuple(items) | Val::List(items)) = c.present(node, cstr(name, "name")?)? else {
            return refuse("the field is not a list of pairs");
        };
        let mut pairs = vec![];
        for item in &items {
            let (Val::Tuple(two) | Val::List(two)) = item else {
                return refuse("the field holds an element that is not a pair");
            };
            let [a, b] = two.as_slice() else {
                return refuse("the field holds an element that is not a pair");
            };
            pairs.push(RxPair {
                first: pair_part(a)?,
                second: pair_part(b)?,
            });
        }
        put_pairs(&pairs, buf, cap, n)
    })
}
