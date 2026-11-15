//! The C door: every export takes and returns C types only, never unwinds, and keeps all state in the context.
use crate::record::{Ctx, Fault};
use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The call succeeded.
pub const RX_OK: c_int = 0;
/// The request was refused; the context holds the reason.
pub const RX_REFUSED: c_int = 1;
/// An argument was not a valid value.
pub const RX_BADARG: c_int = 2;
/// The caller's buffer was too small; the needed size is returned.
pub const RX_TRUNC: c_int = 3;
/// The call panicked inside the core.
pub const RX_PANIC: c_int = 4;
/// A valid question whose answer is that there is none.
pub const RX_ABSENT: c_int = 5;
/// The version of the C interface.
const ABI_VERSION: u32 = 1;

/// A node handle: the position of the node in its context.
pub type Node = u32;

/// Runs `f` on the context: a refusal or a bad argument sets the last error, a panic becomes RX_PANIC.
///
/// The wings' own doors call this with their caller's pointer, so it stays callable without `unsafe`: a NULL context is
/// answered, any other must be a live context made by `rx_ctx_new` (the contract of every door function).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn guard(ctx: *mut Ctx, f: impl FnOnce(&mut Ctx) -> Result<(), Fault>) -> c_int {
    if ctx.is_null() {
        return RX_BADARG;
    }
    let c = unsafe { &mut *ctx };
    match catch_unwind(AssertUnwindSafe(|| f(c))) {
        Ok(Ok(())) => RX_OK,
        Ok(Err(Fault::Refused(m))) => {
            c.set_err(&m);
            RX_REFUSED
        }
        Ok(Err(Fault::BadArg(m))) => {
            c.set_err(&m);
            RX_BADARG
        }
        Ok(Err(Fault::Absent)) => {
            c.set_err("the field holds no value");
            RX_ABSENT
        }
        Ok(Err(Fault::Truncated)) => {
            c.set_err("output buffer too small");
            RX_TRUNC
        }
        Err(p) => {
            let m = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            c.set_err(&format!("panic: {m}"));
            RX_PANIC
        }
    }
}

/// Text out through the caller's buffer: `*len` is the length needed, without the terminating NUL; the text is copied with a NUL
/// when `cap` is at least `*len + 1`, else nothing is copied and the status is RX_TRUNC.
///
/// `buf` is valid for `cap` bytes and `len` for one write (the contract of the caller-buffer protocol); the wings call this
/// from their own doors, so it stays callable without `unsafe`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn put(buf: *mut c_char, cap: usize, len: *mut usize, s: &str) -> Result<(), Fault> {
    if len.is_null() {
        return Err(Fault::BadArg("len is NULL".into()));
    }
    unsafe { *len = s.len() }
    if cap < s.len() + 1 || buf.is_null() {
        return Err(Fault::Truncated);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, s.len());
        *buf.add(s.len()) = 0
    }
    Ok(())
}

/// A NUL-terminated C string argument.
pub(crate) fn cstr<'a>(p: *const c_char, what: &str) -> Result<&'a str, Fault> {
    if p.is_null() {
        return Err(Fault::BadArg(format!("{what} is NULL")));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| Fault::BadArg(format!("{what} is not UTF-8")))
}

/// A pointer and a count as a slice (a NULL pointer is allowed only for a count of 0).
pub(crate) fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Result<&'a [T], Fault> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(Fault::BadArg(format!("{what} is NULL")));
    }
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// The version of the C interface.
#[no_mangle]
pub extern "C" fn rx_abi_version() -> u32 {
    ABI_VERSION
}

/// A door function over raw pointers.
#[no_mangle]
pub extern "C" fn rx_ctx_new() -> *mut Ctx {
    match catch_unwind(Ctx::new) {
        Ok(Ok(c)) => Box::into_raw(Box::new(c)),
        _ => std::ptr::null_mut(),
    }
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_ctx_free(ctx: *mut Ctx) {
    if !ctx.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(ctx) })));
    }
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_last_error(ctx: *const Ctx) -> *const c_char {
    if ctx.is_null() {
        return c"".as_ptr();
    }
    unsafe { &*ctx }.err.as_ptr()
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_make(
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
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        let fields = std::str::from_utf8(slice(fields_kt as *const u8, n, "fields_kt")?)
            .map_err(|_| Fault::BadArg("fields_kt is not UTF-8".into()))?;
        let tag = if tag.is_null() {
            None
        } else {
            Some(cstr(tag, "tag")?)
        };
        let h = c.make(
            cstr(cls, "cls")?,
            cstr(dtype, "dtype")?,
            slice(shape, rank, "shape")?,
            tag,
            slice(ops, nops, "ops")?,
            fields,
        )?;
        unsafe { *out = h }
        Ok(())
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_class(
    ctx: *mut Ctx,
    h: Node,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let n = c.node(h)?;
        put(buf, cap, len, &c.table.classes[n.class].name)
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_operands(
    ctx: *mut Ctx,
    h: Node,
    buf: *mut Node,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        if n.is_null() {
            return Err(Fault::BadArg("n is NULL".into()));
        }
        let ops = &c.node(h)?.ops;
        unsafe { *n = ops.len() }
        if cap < ops.len() || (buf.is_null() && !ops.is_empty()) {
            return Err(Fault::Truncated);
        }
        unsafe { std::ptr::copy_nonoverlapping(ops.as_ptr(), buf, ops.len()) }
        Ok(())
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_new(
    ctx: *mut Ctx,
    cls: *const c_char,
    params_kt: *const c_char,
    n: usize,
    refs: *const Node,
    nrefs: usize,
    out: *mut Node,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        let params = std::str::from_utf8(slice(params_kt as *const u8, n, "params_kt")?)
            .map_err(|_| Fault::BadArg("params_kt is not UTF-8".into()))?;
        let h = c.new_node(cstr(cls, "cls")?, params, slice(refs, nrefs, "refs")?)?;
        unsafe { *out = h }
        Ok(())
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_make_op(
    ctx: *mut Ctx,
    kind: *const c_char,
    ops: *const Node,
    nops: usize,
    literal_kt: *const c_char,
    n: usize,
    out: *mut Node,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        let lit = std::str::from_utf8(slice(literal_kt as *const u8, n, "literal_kt")?)
            .map_err(|_| Fault::BadArg("literal_kt is not UTF-8".into()))?;
        let lit = if lit.trim().is_empty() {
            crate::value::Val::None
        } else {
            crate::value::parse(lit).map_err(Fault::Refused)?
        };
        let h = c.make_op(cstr(kind, "kind")?, slice(ops, nops, "ops")?, &lit)?;
        unsafe { *out = h }
        Ok(())
    })
}

/// Builds an Op of a kind over the operands, carrying the `literal` field of the node `like` (an
/// element of a vector built from the vector's own operation keeps its literal), or no literal
/// when `like` is NULL. The result type is inferred as `rx_make_op` does. Refuses a node or an
/// operand the context does not hold.
///
/// # Safety
/// `kind` is a NUL-terminated string; `ops` is valid for `nops` nodes; `like` is NULL or valid for
/// one read; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_make_op_like(
    ctx: *mut Ctx,
    kind: *const c_char,
    ops: *const Node,
    nops: usize,
    like: *const Node,
    out: *mut Node,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        let lit = if like.is_null() {
            crate::value::Val::None
        } else {
            c.field(unsafe { *like }, "literal")?
        };
        let h = c.make_op(cstr(kind, "kind")?, slice(ops, nops, "ops")?, &lit)?;
        unsafe { *out = h }
        Ok(())
    })
}

/// A door function over raw pointers.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_set_error(ctx: *mut Ctx, msg: *const c_char, n: usize) -> c_int {
    guard(ctx, |c| {
        let m = std::str::from_utf8(slice(msg as *const u8, n, "msg")?)
            .map_err(|_| Fault::BadArg("msg is not UTF-8".into()))?;
        c.set_err(m);
        Ok(())
    })
}
