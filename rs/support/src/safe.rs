//! Safe forms of the door's pointer functions for the test tools: each calls the exported function with the same arguments.
//! The caller owns the pointers it passes, as in any test of the C door.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use raptor_ir::door::{guard, put};
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::{lift_refs, ttype_val};
use rcore::keytext::text_val;
use std::ffi::{c_char, c_int};

pub use raptor_ir::door::{rx_ctx_new, Node, RX_BADARG, RX_OK, RX_PANIC, RX_REFUSED, RX_TRUNC};

/// `raptor_ir::build::rx_new_loop` as a safe function.
#[allow(clippy::too_many_arguments)] // mirrors the C signature it wraps
pub fn rx_new_loop(
    ctx: *mut Ctx,
    index: Node,
    carries: *const Node,
    ncarries: usize,
    inits: *const Node,
    nexts: *const Node,
    start: i64,
    stop: i64,
    step: i64,
    body_sinks: *const Node,
    nsinks: usize,
    exit_cond: *const Node,
    exit_values: *const Node,
    nexit: usize,
    count: *const Node,
    count_tail: c_int,
    dynamic: *const c_char,
    out: *mut Node,
) -> c_int {
    unsafe {
        raptor_ir::build::rx_new_loop(
            ctx,
            index,
            carries,
            ncarries,
            inits,
            nexts,
            start,
            stop,
            step,
            body_sinks,
            nsinks,
            exit_cond,
            exit_values,
            nexit,
            count,
            count_tail,
            dynamic,
            out,
        )
    }
}

/// `raptor_ir::build::rx_substitute` as a safe function.
#[allow(clippy::too_many_arguments)] // mirrors the C signature it wraps
pub fn rx_substitute(
    ctx: *mut Ctx,
    roots: *const Node,
    nroots: usize,
    from: *const Node,
    to: *const Node,
    nmap: usize,
    buf: *mut Node,
    cap: usize,
    n: *mut usize,
) -> c_int {
    unsafe { raptor_ir::build::rx_substitute(ctx, roots, nroots, from, to, nmap, buf, cap, n) }
}

/// `raptor_ir::build::rx_rebuild` as a safe function.
pub fn rx_rebuild(
    ctx: *mut Ctx,
    node: Node,
    operands: *const Node,
    nops: usize,
    out: *mut Node,
) -> c_int {
    unsafe { raptor_ir::build::rx_rebuild(ctx, node, operands, nops, out) }
}

/// `raptor_ir::canon::rx_canonical` as a safe function.
pub fn rx_canonical(
    ctx: *mut Ctx,
    sinks: *const u32,
    n: usize,
    declared_kt: *const c_char,
    force_access: *const c_char,
    exempt_kt: *const c_char,
    out: *mut u32,
) -> c_int {
    unsafe {
        raptor_ir::canon::rx_canonical(ctx, sinks, n, declared_kt, force_access, exempt_kt, out)
    }
}

/// `raptor_ir::canon::rx_walk_order` as a safe function.
pub fn rx_walk_order(ctx: *mut Ctx, w: u32, buf: *mut u32, cap: usize, n: *mut usize) -> c_int {
    unsafe { raptor_ir::canon::rx_walk_order(ctx, w, buf, cap, n) }
}

/// `raptor_ir::canon::rx_walk_field` as a safe function.
pub fn rx_walk_field(
    ctx: *mut Ctx,
    w: u32,
    field: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    unsafe { raptor_ir::canon::rx_walk_field(ctx, w, field, buf, cap, len) }
}

/// `raptor_ir::typed::rx_walk_binds` as a safe function.
pub fn rx_walk_binds(
    ctx: *mut Ctx,
    w: u32,
    role: *const c_char,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    unsafe { raptor_ir::typed::rx_walk_binds(ctx, w, role, name, out) }
}

/// `raptor_ir::typed::rx_walk_zeroed` as a safe function.
pub fn rx_walk_zeroed(
    ctx: *mut Ctx,
    w: u32,
    role: *const c_char,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    unsafe { raptor_ir::typed::rx_walk_zeroed(ctx, w, role, name, out) }
}

/// `raptor_ir::typed::rx_walk_args` as a safe function.
pub fn rx_walk_args(
    ctx: *mut Ctx,
    w: u32,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    unsafe { raptor_ir::typed::rx_walk_args(ctx, w, buf, cap, len, count) }
}

/// `raptor_ir::canon::rx_canonical_like` as a safe function.
pub fn rx_canonical_like(
    ctx: *mut Ctx,
    sinks: *const u32,
    n: usize,
    like: u32,
    out: *mut u32,
) -> c_int {
    unsafe { raptor_ir::canon::rx_canonical_like(ctx, sinks, n, like, out) }
}

/// `raptor_ir::canon::rx_digest` as a safe function.
pub fn rx_digest(ctx: *mut Ctx, w: u32, out65: *mut c_char) -> c_int {
    unsafe { raptor_ir::canon::rx_digest(ctx, w, out65) }
}

/// `raptor_ir::canon::rx_walk_position` as a safe function.
pub fn rx_walk_position(ctx: *mut Ctx, w: u32, node: u32, pos: *mut u32) -> c_int {
    unsafe { raptor_ir::canon::rx_walk_position(ctx, w, node, pos) }
}

/// `raptor_ir::consts::rx_const_names` as a safe function.
pub fn rx_const_names(
    ctx: *mut Ctx,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    unsafe { raptor_ir::consts::rx_const_names(ctx, name, buf, cap, len, count) }
}

/// `raptor_ir::consts::rx_const` as a safe function.
pub fn rx_const(
    ctx: *mut Ctx,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    unsafe { raptor_ir::consts::rx_const(ctx, name, buf, cap, len) }
}

/// `raptor_ir::dialect::rx_declare_dialect` as a safe function.
pub fn rx_declare_dialect(
    ctx: *mut Ctx,
    manifest_json: *const c_char,
    n: usize,
    first_party_kt: *const c_char,
) -> c_int {
    unsafe { raptor_ir::dialect::rx_declare_dialect(ctx, manifest_json, n, first_party_kt) }
}

/// `raptor_ir::door::rx_ctx_free` as a safe function.
pub fn rx_ctx_free(ctx: *mut Ctx) {
    unsafe { raptor_ir::door::rx_ctx_free(ctx) }
}

/// `raptor_ir::door::rx_last_error` as a safe function.
pub fn rx_last_error(ctx: *const Ctx) -> *const c_char {
    unsafe { raptor_ir::door::rx_last_error(ctx) }
}

/// `raptor_ir::door::rx_make` as a safe function.
#[allow(clippy::too_many_arguments)] // mirrors the C signature it wraps
pub fn rx_make(
    ctx: *mut Ctx,
    cls: *const c_char,
    dtype: *const c_char,
    shape: *const i64,
    rank: usize,
    tag: *const c_char,
    ops: *const Node,
    nops: usize,
    fields_kt: *const c_char,
    n: usize,
    out: *mut Node,
) -> c_int {
    unsafe {
        raptor_ir::door::rx_make(
            ctx, cls, dtype, shape, rank, tag, ops, nops, fields_kt, n, out,
        )
    }
}

/// `raptor_ir::door::rx_class` as a safe function.
pub fn rx_class(ctx: *mut Ctx, h: Node, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    unsafe { raptor_ir::door::rx_class(ctx, h, buf, cap, len) }
}

/// The field name a caller passed as a C string.
fn field_name<'a>(name: *const c_char) -> Result<&'a str, Fault> {
    if name.is_null() {
        return Err(Fault::BadArg("name is NULL".into()));
    }
    unsafe { std::ffi::CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| Fault::BadArg("name is not UTF-8".into()))
}

/// The text form of a node's type (`TensorType(shape=(), dtype='f64', tag=None)`), through the two-call text protocol.
pub fn rx_ttype(ctx: *mut Ctx, h: Node, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    guard(ctx, |c| {
        let n = c.node(h)?;
        put(
            buf,
            cap,
            len,
            &text_val(&ttype_val(&n.dtype, &n.shape, n.tag.as_deref())),
        )
    })
}

/// `raptor_ir::door::rx_operands` as a safe function.
pub fn rx_operands(ctx: *mut Ctx, h: Node, buf: *mut Node, cap: usize, n: *mut usize) -> c_int {
    unsafe { raptor_ir::door::rx_operands(ctx, h, buf, cap, n) }
}

/// The text form of a node's field (quotes and reprs as the front end writes them), through the two-call text protocol.
pub fn rx_field(
    ctx: *mut Ctx,
    h: Node,
    name: *const c_char,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let v = c.field(h, field_name(name)?)?;
        put(buf, cap, len, &text_val(&v))
    })
}

/// `raptor_ir::door::rx_new` as a safe function.
pub fn rx_new(
    ctx: *mut Ctx,
    cls: *const c_char,
    params_kt: *const c_char,
    n: usize,
    refs: *const Node,
    nrefs: usize,
    out: *mut Node,
) -> c_int {
    unsafe { raptor_ir::door::rx_new(ctx, cls, params_kt, n, refs, nrefs, out) }
}

/// `raptor_ir::door::rx_make_op` as a safe function.
pub fn rx_make_op(
    ctx: *mut Ctx,
    kind: *const c_char,
    ops: *const Node,
    nops: usize,
    literal_kt: *const c_char,
    n: usize,
    out: *mut Node,
) -> c_int {
    unsafe { raptor_ir::door::rx_make_op(ctx, kind, ops, nops, literal_kt, n, out) }
}

/// `raptor_ir::door::rx_make_op_like` as a safe function.
pub fn rx_make_op_like(
    ctx: *mut Ctx,
    kind: *const c_char,
    ops: *const Node,
    nops: usize,
    like: *const Node,
    out: *mut Node,
) -> c_int {
    unsafe { raptor_ir::door::rx_make_op_like(ctx, kind, ops, nops, like, out) }
}

/// The text form of a node's field with each node reference written `#i`, the i-th entry of the array returned beside it.
#[allow(clippy::too_many_arguments)] // the argument list of the two-call protocol the tools use
pub fn rx_field2(
    ctx: *mut Ctx,
    h: Node,
    name: *const c_char,
    kt: *mut c_char,
    cap: usize,
    len: *mut usize,
    refs: *mut Node,
    rcap: usize,
    nrefs: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        if len.is_null() || nrefs.is_null() {
            return Err(Fault::BadArg("len or nrefs is NULL".into()));
        }
        let v = c.field(h, field_name(name)?)?;
        let mut r: Vec<u32> = vec![];
        let t = text_val(&lift_refs(v, &mut r));
        unsafe {
            *len = t.len();
            *nrefs = r.len()
        }
        if cap < t.len() + 1 || kt.is_null() || rcap < r.len() || (refs.is_null() && !r.is_empty())
        {
            return Err(Fault::Truncated);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(t.as_ptr(), kt as *mut u8, t.len());
            *kt.add(t.len()) = 0;
            std::ptr::copy_nonoverlapping(r.as_ptr(), refs, r.len());
        }
        Ok(())
    })
}

/// `raptor_ir::door::rx_set_error` as a safe function.
pub fn rx_set_error(ctx: *mut Ctx, msg: *const c_char, n: usize) -> c_int {
    unsafe { raptor_ir::door::rx_set_error(ctx, msg, n) }
}

/// `raptor_ir::dump::rx_dump` as a safe function.
pub fn rx_dump(
    ctx: *mut Ctx,
    roots: *const u32,
    n: usize,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    unsafe { raptor_ir::dump::rx_dump(ctx, roots, n, buf, cap, len) }
}

/// `raptor_ir::dump::rx_load` as a safe function.
pub fn rx_load(
    ctx: *mut Ctx,
    text: *const c_char,
    n: usize,
    roots: *mut u32,
    cap: usize,
    nroots: *mut usize,
) -> c_int {
    unsafe { raptor_ir::dump::rx_load(ctx, text, n, roots, cap, nroots) }
}

/// `raptor_ir::loops::rx_body_nodes` as a safe function.
pub fn rx_body_nodes(ctx: *mut Ctx, lp: u32, buf: *mut u32, cap: usize, n: *mut usize) -> c_int {
    unsafe { raptor_ir::loops::rx_body_nodes(ctx, lp, buf, cap, n) }
}

/// `raptor_ir::loops::rx_body_roots` as a safe function.
pub fn rx_body_roots(ctx: *mut Ctx, lp: u32, buf: *mut u32, cap: usize, n: *mut usize) -> c_int {
    unsafe { raptor_ir::loops::rx_body_roots(ctx, lp, buf, cap, n) }
}

/// `raptor_ir::query::rx_through` as a safe function.
pub fn rx_through(ctx: *mut Ctx, node: Node, out: *mut Node) -> c_int {
    unsafe { raptor_ir::query::rx_through(ctx, node, out) }
}

/// `raptor_ir::query::rx_same_index` as a safe function.
pub fn rx_same_index(ctx: *mut Ctx, a: Node, b: Node, out: *mut c_int) -> c_int {
    unsafe { raptor_ir::query::rx_same_index(ctx, a, b, out) }
}

/// `raptor_ir::query::rx_is_a` as a safe function.
pub fn rx_is_a(ctx: *mut Ctx, node: Node, name: *const c_char, out: *mut c_int) -> c_int {
    unsafe { raptor_ir::query::rx_is_a(ctx, node, name, out) }
}

/// `raptor_ir::query::rx_zeroes_plane` as a safe function.
pub fn rx_zeroes_plane(ctx: *mut Ctx, node: Node, out: *mut c_int) -> c_int {
    unsafe { raptor_ir::query::rx_zeroes_plane(ctx, node, out) }
}

/// `raptor_ir::query::rx_dispatch_nodes` as a safe function.
pub fn rx_dispatch_nodes(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut Node,
    cap: usize,
    n: *mut usize,
) -> c_int {
    unsafe { raptor_ir::query::rx_dispatch_nodes(ctx, walk, buf, cap, n) }
}
