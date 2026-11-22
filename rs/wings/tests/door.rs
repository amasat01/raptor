//! One door-level test per function on a small hand-made graph.
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_canonical, rx_ctx_free, rx_ctx_new, rx_last_error, rx_make};
use raptor_wings::place::rw_wide_place;
use raptor_wings::slotwise::{rw_slotwise_chain, rw_slotwise_is_structural_zero};
use raptor_wings::*;
use std::ffi::{CStr, CString};

fn make(c: *mut Ctx, cls: &str, dtype: &str, shape: &[i64], ops: &[u32], fields: &str) -> u32 {
    let (cls, dtype, f) = (
        CString::new(cls).unwrap(),
        CString::new(dtype).unwrap(),
        CString::new(fields).unwrap(),
    );
    let mut o = 0;
    let st = rx_make(
        c,
        cls.as_ptr(),
        dtype.as_ptr(),
        shape.as_ptr(),
        shape.len(),
        std::ptr::null(),
        ops.as_ptr(),
        ops.len(),
        f.as_ptr(),
        f.as_bytes().len(),
        &mut o,
    );
    assert_eq!(
        st,
        0,
        "{}",
        unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy()
    );
    o
}

struct G {
    c: *mut Ctx,
    exp: u32,
    lu: u32,
}

/// x -> exp -> assign o; and a compensated accumulate into plane acc.
fn graph() -> G {
    let c = rx_ctx_new();
    let x = make(c, "Leaf", "f64", &[], &[], "((\"kind\", \"vocab_read\"), (\"role\", \"plane_in\"), (\"name\", \"x\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))");
    let a = make(c, "Leaf", "f64", &[2, 2], &[], "((\"kind\", \"vocab_read\"), (\"role\", \"plane_in\"), (\"name\", \"A\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))");
    let exp = make(
        c,
        "Op",
        "f64",
        &[],
        &[x],
        "((\"kind\", \"exp\"), (\"literal\", None))",
    );
    let lu = make(
        c,
        "Op",
        "f64",
        &[2, 2],
        &[a],
        "((\"kind\", \"lu\"), (\"literal\", None))",
    );
    let o = make(
        c,
        "Assign",
        "f64",
        &[],
        &[exp],
        "((\"kind\", \"assign\"), (\"name\", \"o\"), (\"role\", \"mutable\"))",
    );
    let acc = make(c, "AccumWrite", "f64", &[], &[exp], "((\"kind\", \"accum_write\"), (\"name\", \"acc\"), (\"role\", \"accum_out\"), (\"contract\", \"compensated\"), (\"reduced\", None))");
    let sinks = [o, acc];
    let mut walk = 0;
    let st = rx_canonical(
        c,
        sinks.as_ptr(),
        2,
        c"[]".as_ptr(),
        std::ptr::null(),
        c"[]".as_ptr(),
        &mut walk,
    );
    assert_eq!(
        st,
        0,
        "{}",
        unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy()
    );
    G { c, exp, lu }
}

fn flag(st: i32, v: i32) -> bool {
    assert_eq!(st, 0);
    v != 0
}

#[test]
fn node_questions() {
    let g = graph();
    let mut o = 0;
    assert!(flag(
        unsafe { rw_cost_is_library_call(g.c, g.exp, &mut o) },
        o
    ));
    assert!(!flag(
        unsafe { rw_cost_is_library_call(g.c, g.lu, &mut o) },
        o
    ));
    assert!(flag(unsafe { rw_linalg_is_linalg(g.c, g.lu, &mut o) }, o));
    assert!(!flag(unsafe { rw_linalg_is_linalg(g.c, g.exp, &mut o) }, o));
    assert!(flag(
        unsafe { rw_dispatch_switch_form(g.c, g.exp, &mut o) },
        o
    ));
    assert_eq!(unsafe { rw_cost_is_library_call(g.c, 1000, &mut o) }, 1);
    assert_eq!(
        unsafe { rw_dispatch_switch_form(std::ptr::null_mut(), 0, &mut o) },
        2
    );
    rx_ctx_free(g.c);
}

/// A wide elementwise value stored whole to a mutable is placed in that mutable's plane; the thread form leaves it.
#[test]
fn wide_place_wraps_a_stored_value_in_its_plane() {
    let c = rx_ctx_new();
    let leaf = |name: &str| {
        make(c, "Leaf", "f64", &[100], &[], &format!("((\"kind\", \"vocab_read\"), (\"role\", \"plane_in\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"))
    };
    let (a, b) = (leaf("a"), leaf("b"));
    let sum = make(
        c,
        "Op",
        "f64",
        &[100],
        &[a, b],
        "((\"kind\", \"add\"), (\"literal\", None))",
    );
    let sink = make(
        c,
        "Assign",
        "f64",
        &[100],
        &[sum],
        "((\"kind\", \"assign\"), (\"name\", \"y\"), (\"role\", \"mutable\"))",
    );
    let (mut made, mut count, mut changed, rolled) = ([0u32; 1], 0usize, 0, 16i64);
    let place = |thread_form: i32, made: &mut [u32; 1], count: &mut usize, changed: &mut i32| unsafe {
        rw_wide_place(
            c,
            &sink,
            1,
            thread_form,
            [2i64, 4, 0, 8, 4096].as_ptr(),
            5,
            64,
            &rolled,
            c"auto".as_ptr(),
            c"".as_ptr(),
            made.as_mut_ptr(),
            1,
            count,
            changed,
        )
    };
    assert_eq!(place(1, &mut made, &mut count, &mut changed), 0);
    assert_eq!((made[0], changed), (sink, 0));
    assert_eq!(place(0, &mut made, &mut count, &mut changed), 0);
    assert_eq!((count, changed), (1, 1));
    let value = unsafe { &*c }.nodes[made[0] as usize].ops[0];
    let mut resident = 0;
    assert_eq!(unsafe { rw_wide_is_resident(c, value, &mut resident) }, 0);
    assert_eq!(resident, 1);
    rx_ctx_free(c);
}

/// A lane index is not uniform; a constant, and a lane divided by a multiple of the warp, are.
#[test]
fn warp_uniform_answers_per_node() {
    let c = rx_ctx_new();
    let lane = make(c, "SampleIndex", "i32", &[], &[], "()");
    let literal = |n: i64| {
        make(
            c,
            "Const",
            "i32",
            &[],
            &[],
            &format!("((\"literal\", {n}),)"),
        )
    };
    let (sixty_four, one) = (literal(64), literal(1));
    let divide = |a: u32, b: u32| {
        make(
            c,
            "Op",
            "i32",
            &[],
            &[a, b],
            "((\"kind\", \"div\"), (\"literal\", None))",
        )
    };
    let (block, same) = (divide(lane, sixty_four), divide(lane, one));
    let sink = make(
        c,
        "Assign",
        "i32",
        &[],
        &[block],
        "((\"kind\", \"assign\"), (\"name\", \"o\"), (\"role\", \"mutable\"))",
    );
    let (nodes, warp, mut flags) = ([lane, one, block, same], 32i64, [9; 4]);
    let st = unsafe {
        raptor_wings::lanes::warp::rw_lanes_warp_uniform(
            c,
            &sink,
            1,
            &warp,
            nodes.as_ptr(),
            4,
            flags.as_mut_ptr(),
        )
    };
    assert_eq!(
        st,
        0,
        "{}",
        unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy()
    );
    assert_eq!(flags, [0, 1, 1, 0]);
    rx_ctx_free(c);
}

/// A scalar constant of the literal text.
fn constant(c: *mut Ctx, literal: &str) -> u32 {
    make(
        c,
        "Const",
        "f64",
        &[],
        &[],
        &format!("((\"literal\", {literal}),)"),
    )
}

/// An op of the kind over the operands with the given result shape.
fn op(c: *mut Ctx, kind: &str, shape: &[i64], ops: &[u32]) -> u32 {
    let fields = format!("((\"kind\", \"{kind}\"), (\"literal\", None))");
    make(c, "Op", "f64", shape, ops, &fields)
}

#[test]
fn slotwise_chain_lists_its_links_in_order() {
    let c = rx_ctx_new();
    let base = make(c, "Leaf", "f64", &[2], &[], "((\"kind\", \"vocab_read\"), (\"role\", \"plane_in\"), (\"name\", \"v\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))");
    let (i0, v0, i1, v1) = (
        constant(c, "0"),
        constant(c, "1.5"),
        constant(c, "1"),
        constant(c, "2.5"),
    );
    let first = op(c, "add_at", &[2], &[base, i0, v0]);
    let second = op(c, "add_at", &[2], &[first, i1, v1]);
    let (mut found, mut buf, mut n) = (u32::MAX, [0u32; 8], 0usize);
    let st = unsafe { rw_slotwise_chain(c, second, &mut found, buf.as_mut_ptr(), 8, &mut n) };
    assert_eq!((st, found, n), (0, base, 4));
    assert_eq!(&buf[..n], &[i0, v0, i1, v1]);
    // a node that is no add_at is its own base with no links
    let st = unsafe { rw_slotwise_chain(c, base, &mut found, buf.as_mut_ptr(), 8, &mut n) };
    assert_eq!((st, found, n), (0, base, 0));
    // a buffer that is too small is reported with the size needed
    let st = unsafe { rw_slotwise_chain(c, second, &mut found, buf.as_mut_ptr(), 2, &mut n) };
    assert_eq!((st, n), (3, 4));
    rx_ctx_free(c);
}

#[test]
fn slotwise_zero_is_a_vec_or_splat_of_zero_literals() {
    let c = rx_ctx_new();
    let (zero, one) = (constant(c, "0.0"), constant(c, "1.0"));
    let zero_int = constant(c, "0");
    let ask = |node: u32| {
        let mut out = -1;
        assert_eq!(
            unsafe { rw_slotwise_is_structural_zero(c, node, &mut out) },
            0
        );
        out
    };
    assert_eq!(ask(op(c, "vec", &[2], &[zero, zero_int])), 1);
    assert_eq!(ask(op(c, "splat", &[2], &[zero])), 1);
    assert_eq!(ask(op(c, "vec", &[2], &[])), 1);
    assert_eq!(ask(op(c, "vec", &[2], &[zero, one])), 0);
    assert_eq!(ask(op(c, "splat", &[2], &[one])), 0);
    assert_eq!(ask(op(c, "add", &[2], &[zero, zero])), 0);
    assert_eq!(ask(zero), 0);
    let mut out = 0;
    assert_eq!(
        unsafe { rw_slotwise_is_structural_zero(c, 999, &mut out) },
        1
    );
    rx_ctx_free(c);
}

/// The class name of a node.
fn class_of(c: *mut Ctx, node: u32) -> String {
    let ctx = unsafe { &*c };
    ctx.table.classes[ctx.nodes[node as usize].class]
        .name
        .clone()
}

/// A wide input held sliced: its product with itself is a vector of the slice's width stored by a warp store, and a sum of it is a
/// sum over the lanes of the warp.
#[test]
fn slice_sinks_holds_a_wide_value_across_the_warp() {
    let c = rx_ctx_new();
    assert_eq!(unsafe { raptor_wings::declare::rw_declare_kinds(c) }, 0);
    let x = make(c, "Leaf", "f64", &[128], &[], "((\"kind\", \"vocab_read\"), (\"role\", \"plane_in\"), (\"name\", \"x\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))");
    let op = |kind: &str, shape: &[i64], ops: &[u32]| {
        make(
            c,
            "Op",
            "f64",
            shape,
            ops,
            &format!("((\"kind\", {kind:?}), (\"literal\", None))"),
        )
    };
    let (square, total) = (op("mul", &[128], &[x, x]), op("sum", &[], &[x]));
    let store = |name: &str, shape: &[i64], value: u32| {
        make(
            c,
            "Assign",
            "f64",
            shape,
            &[value],
            &format!("((\"kind\", \"assign\"), (\"name\", {name:?}), (\"role\", \"mutable\"))"),
        )
    };
    let sinks = [store("y", &[128], square), store("t", &[], total)];
    let (mut out, mut count, mut found) = ([0u32; 2], 0usize, 0);
    let ask = |explicit: i32, out: &mut [u32; 2], count: &mut usize, found: &mut i32| unsafe {
        raptor_wings::lanes::slice::rw_lanes_slice_sinks(
            c,
            sinks.as_ptr(),
            2,
            32,
            128,
            128,
            explicit,
            out.as_mut_ptr(),
            2,
            count,
            found,
        )
    };
    assert_eq!(ask(1, &mut out, &mut count, &mut found), 0);
    assert_eq!((found, count), (1, 2));
    assert_eq!(class_of(c, out[0]), "WarpStore");
    let node = |h: u32| unsafe { &(&*c).nodes[h as usize] };
    assert_eq!(node(node(out[0]).ops[0]).shape, [4]);
    let fold = node(node(out[1]).ops[0]);
    assert_eq!(
        fold.fields[0],
        raptor_ir::value::Val::Str("warp_sum".into())
    );
    // a plain kernel under the automatic choice is kept when its widest sliced value reaches the team minimum
    assert_eq!(ask(0, &mut out, &mut count, &mut found), 0);
    assert_eq!(found, 1);
    rx_ctx_free(c);
}
