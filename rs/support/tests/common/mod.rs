//! Helpers that drive the door's C functions.
#![allow(dead_code)]
use raptor_ir::record::Ctx;
use raptor_support::safe::*;
use std::ffi::{c_char, CStr, CString};

pub fn mk(c: *mut Ctx, cls: &str, tag: Option<&str>, ops: &[u32], fields: &str) -> u32 {
    let (cls, dtype, tag) = (
        CString::new(cls).unwrap(),
        CString::new("f64").unwrap(),
        tag.map(|t| CString::new(t).unwrap()),
    );
    let mut out = u32::MAX;
    let st = rx_make(
        c,
        cls.as_ptr(),
        dtype.as_ptr(),
        std::ptr::null(),
        0,
        tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
        ops.as_ptr(),
        ops.len(),
        fields.as_ptr() as *const _,
        fields.len(),
        &mut out,
    );
    assert_eq!(st, RX_OK, "{}", err(c));
    out
}
pub fn mk_shaped(c: *mut Ctx, shape: &[i64]) -> u32 {
    let mut out = u32::MAX;
    let f = "(('literal', 1),)";
    let st = rx_make(
        c,
        c"Const".as_ptr(),
        c"f64".as_ptr(),
        shape.as_ptr(),
        shape.len(),
        std::ptr::null(),
        std::ptr::null(),
        0,
        f.as_ptr() as *const _,
        f.len(),
        &mut out,
    );
    assert_eq!(st, RX_OK, "{}", err(c));
    out
}
pub fn err(c: *mut Ctx) -> String {
    unsafe { CStr::from_ptr(rx_last_error(c)) }
        .to_str()
        .unwrap()
        .to_string()
}
pub fn text(c: *mut Ctx, f: impl Fn(*mut Ctx, *mut c_char, usize, *mut usize) -> i32) -> String {
    let mut len = 0usize;
    assert_eq!(f(c, std::ptr::null_mut(), 0, &mut len), RX_TRUNC);
    let mut buf = vec![1 as c_char; len + 1];
    assert_eq!(f(c, buf.as_mut_ptr(), len + 1, &mut len), RX_OK);
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_str()
        .unwrap()
        .to_string()
}
pub fn dump(c: *mut Ctx, roots: &[u32]) -> String {
    text(c, |c, b, cap, l| {
        rx_dump(c, roots.as_ptr(), roots.len(), b, cap, l)
    })
}
pub fn load(c: *mut Ctx, t: &str) -> (i32, Vec<u32>) {
    let mut roots = vec![0u32; 16];
    let mut n = 0;
    let st = rx_load(
        c,
        t.as_ptr() as *const _,
        t.len(),
        roots.as_mut_ptr(),
        16,
        &mut n,
    );
    roots.truncate(if st == RX_OK { n } else { 0 });
    (st, roots)
}
pub fn count(c: *mut Ctx) -> usize {
    unsafe { &*c }.nodes.len()
}

pub const LEAF: &str = "(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '{}'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
pub const ADD: &str = "(('kind', 'add'), ('literal', None))";
pub fn leaf(c: *mut Ctx, n: &str) -> u32 {
    mk(c, "Leaf", None, &[], &LEAF.replace("{}", n))
}

/// a, b, a + b, 2.0, (a + b) + 2.0, y = that
pub fn expression(c: *mut Ctx) -> [u32; 6] {
    let (a, b) = (leaf(c, "a"), leaf(c, "b"));
    let s = mk(c, "Op", None, &[a, b], ADD);
    let k = mk(c, "Const", None, &[], "(('literal', 2.0),)");
    let e = mk(c, "Op", None, &[s, k], ADD);
    let y = mk(c, "Assign", None, &[e], "(('name', 'y'),)");
    [a, b, s, k, e, y]
}

/// index, carry, a constant, a body sink and a loop over them; returns (loop, sink).
pub fn a_loop(c: *mut Ctx) -> (u32, u32) {
    let (l, i) = (leaf(c, "p"), mk(c, "Const", None, &[], "(('literal', 0),)"));
    let ix = mk(
        c,
        "LoopIndex",
        None,
        &[],
        "(('depth', 0), ('name', 'i'), ('bounds', None))",
    );
    let cr = mk(
        c,
        "LoopCarry",
        None,
        &[],
        "(('depth', 0), ('slot', 0), ('name', 'c'))",
    );
    let sink = mk(c, "Assign", None, &[cr], "(('name', 'o'),)");
    let f = format!("(('index', #{ix}), ('carries', (#{cr},)), ('nexts', (#{l},)), ('start', 0), ('stop', 2), ('step', 1), ('trip', 2), ('carry_count', 1), ('body_sinks', (#{sink},)), \
                     ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), ('count_tail', False), ('dynamic', None), ('_free_end', 1))");
    (mk(c, "Loop", None, &[i], &f), sink)
}
