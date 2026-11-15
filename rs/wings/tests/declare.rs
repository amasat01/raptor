//! A fresh context with only the wings' own declaration runs the rewrites that make wing kinds, and
//! what they make survives a dump and a load in another such context.
use raptor_ir::record::Ctx;
use raptor_support::safe::{
    rx_ctx_free, rx_ctx_new, rx_dump, rx_last_error, rx_load, rx_make, rx_make_op,
};
use raptor_wings::boxing::rw_box_boxed_sinks;
use raptor_wings::declare::{rw_declare_kinds, MANIFESTS};
use raptor_wings::lstsq::rw_lstsq_lowered;
use raptor_wings::place::rw_wide_place;
use std::ffi::{c_char, CStr, CString};

/// A context with the wings' kinds declared and nothing else.
fn fresh() -> *mut Ctx {
    let c = rx_ctx_new();
    assert_eq!(unsafe { rw_declare_kinds(c) }, 0, "{}", message(c));
    c
}

fn message(c: *mut Ctx) -> String {
    unsafe { CStr::from_ptr(rx_last_error(c)) }
        .to_string_lossy()
        .into_owned()
}

fn make(c: *mut Ctx, class: &str, dtype: &str, shape: &[i64], ops: &[u32], fields: &str) -> u32 {
    let (class, dtype, f) = (
        CString::new(class).unwrap(),
        CString::new(dtype).unwrap(),
        CString::new(fields).unwrap(),
    );
    let mut out = 0;
    let st = rx_make(
        c,
        class.as_ptr(),
        dtype.as_ptr(),
        shape.as_ptr(),
        shape.len(),
        std::ptr::null(),
        ops.as_ptr(),
        ops.len(),
        f.as_ptr(),
        f.as_bytes().len(),
        &mut out,
    );
    assert_eq!(st, 0, "{}", message(c));
    out
}

fn leaf(c: *mut Ctx, role: &str, name: &str, shape: &[i64]) -> u32 {
    let fields = format!("((\"kind\", \"vocab_read\"), (\"role\", {role:?}), (\"name\", {name:?}), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))");
    make(c, "Leaf", "f64", shape, &[], &fields)
}

fn assign(c: *mut Ctx, name: &str, value: u32, shape: &[i64]) -> u32 {
    let fields = format!("((\"kind\", \"assign\"), (\"name\", {name:?}), (\"role\", \"mutable\"))");
    make(c, "Assign", "f64", shape, &[value], &fields)
}

/// The text of the roots, dumped in `c`, loaded into another fresh context and dumped again.
fn round_trip(c: *mut Ctx, roots: &[u32]) {
    let dump = |c: *mut Ctx, roots: &[u32]| {
        let (mut buf, mut len) = (vec![0 as c_char; 1 << 16], 0usize);
        assert_eq!(
            rx_dump(
                c,
                roots.as_ptr(),
                roots.len(),
                buf.as_mut_ptr(),
                buf.len(),
                &mut len
            ),
            0,
            "{}",
            message(c)
        );
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    let text = dump(c, roots);
    let other = fresh();
    let (mut loaded, mut n) = (vec![0u32; roots.len()], 0usize);
    let st = rx_load(
        other,
        text.as_ptr() as *const c_char,
        text.len(),
        loaded.as_mut_ptr(),
        loaded.len(),
        &mut n,
    );
    assert_eq!(st, 0, "{}", message(other));
    assert_eq!(dump(other, &loaded[..n]), text);
    rx_ctx_free(other);
}

#[test]
fn boxed_sinks_in_a_context_with_only_the_wings_kinds() {
    let c = fresh();
    let x = leaf(c, "per_sample", "x", &[]);
    let op = |kind: &str, ops: &[u32]| {
        let mut out = 0;
        let kind = CString::new(kind).unwrap();
        assert_eq!(
            rx_make_op(
                c,
                kind.as_ptr(),
                ops.as_ptr(),
                ops.len(),
                c"".as_ptr(),
                0,
                &mut out
            ),
            0,
            "{}",
            message(c)
        );
        out
    };
    let sink = assign(c, "o", op("exp", &[x]), &[]);
    let (mut made, mut count) = ([0u32; 2], 0usize);
    assert_eq!(
        unsafe { rw_box_boxed_sinks(c, &sink, 1, 0, 2, 1, made.as_mut_ptr(), 2, &mut count) },
        0,
        "{}",
        message(c)
    );
    assert_eq!(count, 2);
    round_trip(c, &made);
    rx_ctx_free(c);
}

#[test]
fn lowered_sinks_in_a_context_with_only_the_wings_kinds() {
    let c = fresh();
    let (a, b) = (
        leaf(c, "mat_in", "a", &[4, 2]),
        leaf(c, "vec_in", "b", &[4]),
    );
    let mut solve = 0;
    assert_eq!(
        rx_make_op(
            c,
            c"lstsq".as_ptr(),
            [a, b].as_ptr(),
            2,
            c"".as_ptr(),
            0,
            &mut solve
        ),
        0,
        "{}",
        message(c)
    );
    let sink = assign(c, "x", solve, &[2]);
    let (mut made, mut count, mut changed, max_dim) = ([0u32; 1], 0usize, 0, 32i64);
    assert_eq!(
        unsafe {
            rw_lstsq_lowered(
                c,
                &sink,
                1,
                &max_dim,
                made.as_mut_ptr(),
                1,
                &mut count,
                &mut changed,
            )
        },
        0,
        "{}",
        message(c)
    );
    assert_eq!(changed, 1);
    round_trip(c, &made);
    rx_ctx_free(c);
}

#[test]
fn placed_sinks_in_a_context_with_only_the_wings_kinds() {
    let c = fresh();
    let (a, b) = (
        leaf(c, "vec_in", "a", &[100]),
        leaf(c, "vec_in", "b", &[100]),
    );
    let sum = make(
        c,
        "Op",
        "f64",
        &[100],
        &[a, b],
        "((\"kind\", \"add\"), (\"literal\", None))",
    );
    let sink = assign(c, "y", sum, &[100]);
    let (mut made, mut count, mut changed, rolled) = ([0u32; 1], 0usize, 0, 16i64);
    let runtime = [2i64, 4, 0, 8, 4096];
    let st = unsafe {
        rw_wide_place(
            c,
            &sink,
            1,
            0,
            runtime.as_ptr(),
            64,
            &rolled,
            c"auto".as_ptr(),
            c"".as_ptr(),
            made.as_mut_ptr(),
            1,
            &mut count,
            &mut changed,
        )
    };
    assert_eq!((st, changed), (0, 1), "{}", message(c));
    round_trip(c, &made);
    rx_ctx_free(c);
}

#[test]
fn declaring_twice_is_accepted() {
    let c = fresh();
    assert_eq!(unsafe { rw_declare_kinds(c) }, 0, "{}", message(c));
    rx_ctx_free(c);
}

/// The embedded manifests are the files the build's files: when `WING_MANIFEST_SOURCE` names the
/// directory that holds them as `<wing>/wing.json`, every copy must be equal to its source.
#[test]
fn embedded_manifests_equal_their_source() {
    let Ok(dir) = std::env::var("WING_MANIFEST_SOURCE") else {
        return;
    };
    for (wing, copy) in MANIFESTS {
        let source = std::fs::read_to_string(format!("{dir}/{wing}/wing.json"))
            .expect("the source manifest");
        assert!(
            source == copy,
            "the copy of the {wing} manifest differs from its source"
        );
    }
}
