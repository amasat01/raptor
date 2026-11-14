//! The C door: every export takes and returns C types only, never unwinds, and keeps all state in the context.
use crate::record::{refuse, Ctx, Fault};
use crate::value::{text_val, ttype_val};
use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub const RX_OK: c_int = 0;
pub const RX_REFUSED: c_int = 1;
pub const RX_BADARG: c_int = 2;
pub const RX_TRUNC: c_int = 3;
pub const RX_PANIC: c_int = 4;
const ABI_VERSION: u32 = 1;

pub type Node = u32;

/// Runs `f` on the context: a refusal or a bad argument sets the last error, a panic becomes RX_PANIC.
pub fn guard(ctx: *mut Ctx, f: impl FnOnce(&mut Ctx) -> Result<(), Fault>) -> c_int {
    if ctx.is_null() { return RX_BADARG }
    let c = unsafe { &mut *ctx };
    match catch_unwind(AssertUnwindSafe(|| f(c))) {
        Ok(Ok(())) => RX_OK,
        Ok(Err(Fault::Refused(m))) => { c.set_err(&m); RX_REFUSED }
        Ok(Err(Fault::BadArg(m))) => { c.set_err(&m); RX_BADARG }
        Ok(Err(Fault::Truncated)) => { c.set_err("output buffer too small"); RX_TRUNC }
        Err(p) => {
            let m = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
            c.set_err(&format!("panic: {m}"));
            RX_PANIC
        }
    }
}

/// The answer of a call that is declared but not built yet.
pub fn not_built(ctx: *mut Ctx, name: &str) -> c_int { guard(ctx, |_| refuse(format!("not in this build: {name}"))) }

/// Text out through the caller's buffer: `*len` is the length needed, without the terminating NUL; the text is copied with a NUL
/// when `cap` is at least `*len + 1`, else nothing is copied and the status is RX_TRUNC.
pub fn put(buf: *mut c_char, cap: usize, len: *mut usize, s: &str) -> Result<(), Fault> {
    if len.is_null() { return Err(Fault::BadArg("len is NULL".into())) }
    unsafe { *len = s.len() }
    if cap < s.len() + 1 || buf.is_null() { return Err(Fault::Truncated) }
    unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, s.len()); *buf.add(s.len()) = 0 }
    Ok(())
}

/// A NUL-terminated C string argument.
pub(crate) fn cstr<'a>(p: *const c_char, what: &str) -> Result<&'a str, Fault> {
    if p.is_null() { return Err(Fault::BadArg(format!("{what} is NULL"))) }
    unsafe { CStr::from_ptr(p) }.to_str().map_err(|_| Fault::BadArg(format!("{what} is not UTF-8")))
}

/// A pointer and a count as a slice (a NULL pointer is allowed only for a count of 0).
pub(crate) fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Result<&'a [T], Fault> {
    if n == 0 { return Ok(&[]) }
    if p.is_null() { return Err(Fault::BadArg(format!("{what} is NULL"))) }
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

#[no_mangle]
pub extern "C" fn rx_abi_version() -> u32 { ABI_VERSION }

#[no_mangle]
pub extern "C" fn rx_ctx_new() -> *mut Ctx {
    match catch_unwind(Ctx::new) { Ok(Ok(c)) => Box::into_raw(Box::new(c)), _ => std::ptr::null_mut() }
}

#[no_mangle]
pub extern "C" fn rx_ctx_free(ctx: *mut Ctx) {
    if !ctx.is_null() { let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(ctx) }))); }
}

#[no_mangle]
pub extern "C" fn rx_last_error(ctx: *const Ctx) -> *const c_char {
    if ctx.is_null() { return c"".as_ptr() }
    unsafe { &*ctx }.err.as_ptr()
}

#[no_mangle]
pub extern "C" fn rx_make(ctx: *mut Ctx, cls: *const c_char, dtype: *const c_char, shape: *const i64, rank: usize, tag: *const c_char,
                          ops: *const Node, nops: usize, fields_kt: *const c_char, n: usize, out: *mut Node) -> c_int {
    guard(ctx, |c| {
        if out.is_null() { return Err(Fault::BadArg("out is NULL".into())) }
        let fields = std::str::from_utf8(slice(fields_kt as *const u8, n, "fields_kt")?).map_err(|_| Fault::BadArg("fields_kt is not UTF-8".into()))?;
        let tag = if tag.is_null() { None } else { Some(cstr(tag, "tag")?) };
        let h = c.make(cstr(cls, "cls")?, cstr(dtype, "dtype")?, slice(shape, rank, "shape")?, tag, slice(ops, nops, "ops")?, fields)?;
        unsafe { *out = h }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn rx_class(ctx: *mut Ctx, h: Node, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    guard(ctx, |c| { let n = c.node(h)?; put(buf, cap, len, &c.table.classes[n.class].name) })
}

#[no_mangle]
pub extern "C" fn rx_ttype(ctx: *mut Ctx, h: Node, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    guard(ctx, |c| { let n = c.node(h)?; put(buf, cap, len, &text_val(&ttype_val(&n.dtype, &n.shape, n.tag.as_deref()))) })
}

#[no_mangle]
pub extern "C" fn rx_operands(ctx: *mut Ctx, h: Node, buf: *mut Node, cap: usize, n: *mut usize) -> c_int {
    guard(ctx, |c| {
        if n.is_null() { return Err(Fault::BadArg("n is NULL".into())) }
        let ops = &c.node(h)?.ops;
        unsafe { *n = ops.len() }
        if cap < ops.len() || (buf.is_null() && !ops.is_empty()) { return Err(Fault::Truncated) }
        unsafe { std::ptr::copy_nonoverlapping(ops.as_ptr(), buf, ops.len()) }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn rx_field(ctx: *mut Ctx, h: Node, name: *const c_char, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    guard(ctx, |c| { let v = c.field(h, cstr(name, "name")?)?; put(buf, cap, len, &text_val(&v)) })
}
