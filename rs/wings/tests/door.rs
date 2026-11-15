//! One door-level test per function on a small hand-made graph.
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_canonical, rx_ctx_free, rx_ctx_new, rx_last_error, rx_make};
use raptor_wings::place::rw_wide_place;
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
    let x = make(c, "Leaf", "f64", &[], &[], "((\"kind\", \"vocab_read\"), (\"role\", \"per_sample\"), (\"name\", \"x\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))");
    let a = make(c, "Leaf", "f64", &[2, 2], &[], "((\"kind\", \"vocab_read\"), (\"role\", \"mat_in\"), (\"name\", \"A\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))");
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
        make(c, "Leaf", "f64", &[100], &[], &format!("((\"kind\", \"vocab_read\"), (\"role\", \"vec_in\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"))
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
