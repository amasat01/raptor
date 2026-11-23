//! Group scopes: the three reference kernels as stored documents, the phase numbering the door states, and every refusal
//! the typing gives by name. Oracle for the phases: written by hand from the rule that a barrier ends the phase it closes, a
//! group-wide reduce or scan ends the phase its operand is read in, and phases never go backwards.
mod common;
use common::*;
use raptor_ir::record::Ctx;
use raptor_ir::typed::rx_field_ints;
use raptor_ir::value::Val;
use raptor_support::groups::{self as g, Fixture, Membership};
use raptor_support::safe::*;
use std::ffi::{c_char, CString};

/// A constructor of one reference kernel.
type Make = fn(&mut Ctx) -> Result<Fixture, String>;

/// The reference kernels with their stored documents.
const FIXTURES: [(&str, Make, &str); 3] = [
    ("tile", g::tile, include_str!("data/groups/tile.dump")),
    ("rows", g::rows, include_str!("data/groups/rows.dump")),
    ("ring", g::ring, include_str!("data/groups/ring.dump")),
];

/// The refusal text of a build that must refuse.
fn refused<T>(built: Result<T, String>) -> String {
    match built {
        Ok(_) => panic!("the build was expected to refuse"),
        Err(message) => message,
    }
}

/// A fresh context.
fn ctx() -> Ctx {
    Ctx::new().unwrap()
}

/// `tile` with the body and the scratch the caller gives: a group of 8 over the scratch `t`.
fn tile_with(
    ctx: &mut Ctx,
    body: impl FnOnce(&mut Ctx, u32, u32) -> Vec<u32>,
) -> Result<u32, String> {
    let x = g::plane_in(ctx, "x", "f64")?;
    let t = g::scratch(ctx, "t", "f64", 8)?;
    let body = body(ctx, x, t);
    g::group(ctx, &Membership::count(8), &[t], &body)
}

#[test]
fn the_three_fixtures_equal_their_stored_documents() {
    for (name, make, stored) in FIXTURES {
        let c = rx_ctx_new();
        let fixture = make(unsafe { &mut *c }).unwrap();
        assert_eq!(dump(c, &[fixture.group]), stored, "{name}");
        rx_ctx_free(c);
    }
}

#[test]
fn a_stored_document_loads_and_dumps_again_unchanged() {
    for (name, _, stored) in FIXTURES {
        let c = rx_ctx_new();
        let (status, roots) = load(c, stored);
        assert_eq!(status, RX_OK, "{name}: {}", err(c));
        assert_eq!(roots.len(), 1, "{name}");
        assert_eq!(dump(c, &roots), stored, "{name}");
        rx_ctx_free(c);
    }
}

/// The phases of the body and their count, as the door states them for the group `root`.
fn phases_at_door(c: *mut Ctx, root: u32) -> (Vec<i64>, String) {
    let mut buf = [0i64; 8];
    let mut n = 0usize;
    let st = unsafe { rx_field_ints(c, root, c"phases".as_ptr(), buf.as_mut_ptr(), 8, &mut n) };
    assert_eq!(st, RX_OK, "{}", err(c));
    let field = |name: &CString| {
        text(c, |c, b, cap, l| {
            rx_field(c, root, name.as_ptr(), b, cap, l)
        })
    };
    (
        buf[..n].to_vec(),
        field(&CString::new("phase_count").unwrap()),
    )
}

#[test]
fn the_door_states_the_membership_the_scratch_and_the_phase_of_each_sink() {
    // (document, membership, extra fields, phases of the body sinks, number of phases)
    let want = [
        ("tile", "'count'", "8", "(#1,)", vec![0, 1], "2"),
        ("rows", "'axes'", "None", "()", vec![1, 1, 1], "2"),
        ("ring", "'count'", "8", "(#1,)", vec![0, 0, 1], "2"),
    ];
    for ((name, _, stored), (want_name, membership, count, scratch, phases, phase_count)) in
        FIXTURES.into_iter().zip(want)
    {
        assert_eq!(name, want_name);
        let c = rx_ctx_new();
        let root = load(c, stored).1[0];
        let field = |f: &str| {
            let f = CString::new(f).unwrap();
            text(c, |c, b, cap, l| rx_field(c, root, f.as_ptr(), b, cap, l))
        };
        assert_eq!(field("membership"), membership, "{name}");
        assert_eq!(field("count"), count, "{name}");
        assert_eq!(field("scratch"), scratch, "{name}");
        assert_eq!(
            phases_at_door(c, root),
            (phases, phase_count.to_string()),
            "{name}"
        );
        rx_ctx_free(c);
    }
}

#[test]
fn a_membership_by_axes_states_the_lane_axes_and_the_ones_that_span_a_group() {
    let c = rx_ctx_new();
    let root = load(c, FIXTURES[1].2).1[0];
    let field = |f: &str| {
        let f = CString::new(f).unwrap();
        text(c, |c, b, cap, l| rx_field(c, root, f.as_ptr(), b, cap, l))
    };
    assert_eq!(field("grid"), "(('row', None), ('col', 4))");
    assert_eq!(field("axes"), "('col',)");
    rx_ctx_free(c);
}

#[test]
fn a_scratch_declares_its_name_its_element_type_and_its_one_axis() {
    let c = rx_ctx_new();
    let tile = load(c, FIXTURES[0].2).1[0];
    let mut one = [0u32; 4];
    let mut n = 0usize;
    let st = unsafe {
        raptor_ir::typed::rx_field_node_list(
            c,
            tile,
            c"scratch".as_ptr(),
            one.as_mut_ptr(),
            4,
            &mut n,
        )
    };
    assert_eq!((st, n), (RX_OK, 1), "{}", err(c));
    let field = |f: &str| {
        let f = CString::new(f).unwrap();
        text(c, |c, b, cap, l| rx_field(c, one[0], f.as_ptr(), b, cap, l))
    };
    assert_eq!(field("name"), "'t'");
    assert_eq!(field("role"), "'scratch'");
    assert_eq!(field("axes"), "(('rank', 8),)");
    let ttype = text(c, |c, b, cap, l| rx_ttype(c, one[0], b, cap, l));
    assert!(ttype.contains("f64"), "{ttype}");
    rx_ctx_free(c);
}

// phases

#[test]
fn a_scan_ends_the_phase_like_a_reduce_and_a_trailing_barrier_adds_none() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let running = g::scan(&mut c, "sum", true, x).unwrap();
    let out = g::commit(&mut c, "y", running).unwrap();
    let stop = g::barrier(&mut c).unwrap();
    for (body, phases, count) in [(vec![out], vec![1], 2), (vec![out, stop], vec![1, 1], 2)] {
        let plan = c.phase_plan(&body);
        assert_eq!((plan.count(), plan.sinks), (count, phases));
    }
}

#[test]
fn two_sinks_using_one_reduce_share_its_phase() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let total = g::reduce(&mut c, "sum", "tree", x).unwrap();
    let one = g::apply(&mut c, "add", &[total, x]).unwrap();
    let (first, second) = (
        g::commit(&mut c, "a", total).unwrap(),
        g::commit(&mut c, "b", one).unwrap(),
    );
    let plan = c.phase_plan(&[first, second]);
    assert_eq!(plan.sinks, vec![1, 1]);
}

// rule (a): the phase rule

#[test]
fn a_neighbour_read_in_the_phase_that_writes_the_scratch_is_refused() {
    let m = refused(g::ring_with(&mut ctx(), false));
    assert!(
        m.contains("is read at another lane's element in the phase that writes it; put a barrier between the write and the read"),
        "{m}"
    );
}

#[test]
fn a_barrier_between_the_write_and_the_neighbour_read_makes_it_legal() {
    g::ring_with(&mut ctx(), true).unwrap();
}

#[test]
fn a_neighbour_read_inside_a_reduce_shares_the_phase_of_the_write_unless_a_barrier_separates_them()
{
    let build = |barrier: bool| {
        tile_with(&mut ctx(), |c, x, t| {
            let place = g::rank(c).unwrap();
            let fill = g::store(c, "t", 8, x, place).unwrap();
            let one = g::whole(c, "i32", 1).unwrap();
            let next = g::apply(c, "add", &[place, one]).unwrap();
            let around = g::read(c, t, next, "wrap").unwrap();
            let total = g::reduce(c, "sum", "tree", around).unwrap();
            let out = g::commit(c, "y", total).unwrap();
            if barrier {
                vec![fill, g::barrier(c).unwrap(), out]
            } else {
                vec![fill, out]
            }
        })
    };
    assert!(refused(build(false)).contains("in the phase that writes it"));
    build(true).unwrap();
}

#[test]
fn a_read_at_the_lanes_own_rank_in_the_writing_phase_is_legal() {
    g::tile(&mut ctx()).unwrap();
}

#[test]
fn a_stored_document_without_its_barrier_is_refused_on_load() {
    let broken = FIXTURES[2].2.replace("(#3, #10, #9)", "(#3, #9)");
    assert_ne!(broken, FIXTURES[2].2);
    let c = rx_ctx_new();
    let (status, _) = load(c, &broken);
    assert_eq!(status, RX_REFUSED);
    assert!(err(c).contains("in the phase that writes it"), "{}", err(c));
    rx_ctx_free(c);
}

// rule (b): group elements and scratch outside a group

/// The status and the message of the canonical walk over `sinks`.
fn walk(c: &mut Ctx, sinks: &[u32]) -> (i32, String) {
    let (p, mut w) = (c as *mut Ctx, 0u32);
    let none: *const c_char = std::ptr::null();
    let status = rx_canonical(p, sinks.as_ptr(), sinks.len(), none, none, none, &mut w);
    (status, err(p))
}

#[test]
fn a_group_element_or_a_scratch_outside_a_group_is_refused_by_name() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let t = g::scratch(&mut c, "t", "f64", 8).unwrap();
    let place = g::rank(&mut c).unwrap();
    let place_out = g::commit(&mut c, "p", place).unwrap();
    let size_out = {
        let size = g::size(&mut c).unwrap();
        g::commit(&mut c, "s", size).unwrap()
    };
    let total = g::reduce(&mut c, "sum", "tree", x).unwrap();
    let total_out = g::commit(&mut c, "r", total).unwrap();
    let running = g::scan(&mut c, "sum", false, x).unwrap();
    let running_out = g::commit(&mut c, "q", running).unwrap();
    let first = g::whole(&mut c, "i32", 0).unwrap();
    let read_out = {
        let read = g::read(&mut c, t, first, "clamp").unwrap();
        g::commit(&mut c, "u", read).unwrap()
    };
    let store_out = g::store(&mut c, "t", 8, x, first).unwrap();
    let stop = g::barrier(&mut c).unwrap();
    for (root, what) in [
        (place_out, "GroupRank"),
        (size_out, "GroupSize"),
        (total_out, "GroupReduce"),
        (running_out, "GroupScan"),
        (read_out, "scratch 't'"),
        (store_out, "scratch 't'"),
        (stop, "Barrier"),
    ] {
        let (status, message) = walk(&mut c, &[root]);
        assert_eq!(status, RX_REFUSED, "{what}");
        assert!(
            message.contains(&format!("{what} is reached outside a Group")),
            "{what}: {message}"
        );
    }
}

// rule (c) and (d): groups in groups and in loops, loops in phases

#[test]
fn a_group_inside_a_group_is_refused() {
    let mut c = ctx();
    let inner = g::tile(&mut c).unwrap().group;
    let m = refused(g::group(&mut c, &Membership::count(4), &[], &[inner]));
    assert!(m.contains("a group inside a group is not written"), "{m}");
}

/// A loop over a carry whose next value is `next`, holding `body` as its one body sink.
fn a_loop(c: &mut Ctx, next: u32, body: u32) -> Result<u32, String> {
    let init = c
        .make("Const", "f64", &[], None, &[], "(('literal', 0),)")
        .unwrap();
    let index = c
        .make(
            "LoopIndex",
            "f64",
            &[],
            None,
            &[],
            "(('depth', 0), ('name', 'i'), ('bounds', None))",
        )
        .unwrap();
    let carry = c
        .make(
            "LoopCarry",
            "f64",
            &[],
            None,
            &[],
            "(('depth', 0), ('slot', 0), ('name', 'c'))",
        )
        .unwrap();
    let fields = format!(
        "(('index', #{index}), ('carries', (#{carry},)), ('nexts', (#{next},)), ('start', 0), ('stop', 2), ('step', 1), ('trip', 2), \
         ('carry_count', 1), ('body_sinks', (#{body},)), ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), \
         ('count_tail', False), ('dynamic', None), ('_free_end', 1))"
    );
    c.make("Loop", "f64", &[], None, &[init], &fields)
        .map_err(|f| format!("{f:?}"))
}

#[test]
fn a_loop_inside_one_phase_of_a_group_is_legal() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let carry = c
        .make(
            "LoopCarry",
            "f64",
            &[],
            None,
            &[],
            "(('depth', 0), ('slot', 0), ('name', 'k'))",
        )
        .unwrap();
    let body = g::commit(&mut c, "o", carry).unwrap();
    let looped = a_loop(&mut c, x, body).unwrap();
    let out = g::commit(&mut c, "y", looped).unwrap();
    g::group(&mut c, &Membership::count(4), &[], &[out]).unwrap();
}

#[test]
fn a_barrier_a_reduce_a_scan_or_a_group_inside_a_loop_is_refused() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let stop = g::barrier(&mut c).unwrap();
    let total = g::reduce(&mut c, "sum", "tree", x).unwrap();
    let running = g::scan(&mut c, "sum", true, x).unwrap();
    let out = g::commit(&mut c, "o", x).unwrap();
    let inner = g::group(&mut c, &Membership::count(4), &[], &[out]).unwrap();
    // each in the body sinks, in the next values, and held by a sink
    let held = g::commit(&mut c, "h", total).unwrap();
    for (what, next, body) in [
        ("Barrier", x, stop),
        ("Group", x, inner),
        ("GroupReduce", total, out),
        ("GroupScan", running, out),
        ("GroupReduce", x, held),
    ] {
        let m = refused(a_loop(&mut c, next, body));
        assert!(
            m.contains(&format!("a {what} inside a loop body is not written")),
            "{what}: {m}"
        );
    }
}

#[test]
fn a_store_into_scratch_inside_a_loop_inside_a_group_is_refused() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let t = g::scratch(&mut c, "t", "f64", 8).unwrap();
    let place = g::rank(&mut c).unwrap();
    let carry = c
        .make(
            "LoopCarry",
            "f64",
            &[],
            None,
            &[],
            "(('depth', 0), ('slot', 0), ('name', 'k'))",
        )
        .unwrap();
    let body = g::store(&mut c, "t", 8, carry, place).unwrap();
    let looped = a_loop(&mut c, x, body).unwrap();
    let out = g::commit(&mut c, "y", looped).unwrap();
    let m = refused(g::group(&mut c, &Membership::count(8), &[t], &[out]));
    assert!(m.contains("is stored inside a loop"), "{m}");
}

// rule (e): the membership and the scratch size

#[test]
fn a_count_is_a_positive_whole_number_of_lanes() {
    for lanes in [0, -3] {
        let mut c = ctx();
        let x = g::plane_in(&mut c, "x", "f64").unwrap();
        let out = g::commit(&mut c, "y", x).unwrap();
        let m = refused(g::group(&mut c, &Membership::count(lanes), &[], &[out]));
        assert!(
            m.contains("membership 'count' takes a positive whole number of lanes"),
            "{m}"
        );
    }
}

/// A one-sink group over `m` declaring a scratch of `extent` elements when it is given.
fn grouped(m: &Membership, extent: Option<i128>) -> Result<u32, String> {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64")?;
    let out = g::commit(&mut c, "y", x)?;
    let scratch: Vec<u32> = match extent {
        Some(n) => vec![g::scratch(&mut c, "t", "f64", n)?],
        None => vec![],
    };
    g::group(&mut c, m, &scratch, &[out])
}

#[test]
fn an_axes_membership_names_lane_axes_the_kernel_has() {
    let grid = vec![("row", None), ("col", Some(4))];
    let by = |axes: Vec<&'static str>| Membership {
        kind: "axes",
        grid: grid.clone(),
        axes,
        ..Default::default()
    };
    assert!(refused(grouped(&by(vec!["depth"]), None))
        .contains("names the axis 'depth', which the lanes do not have (lane axes:"));
    assert!(refused(grouped(&by(vec![]), None)).contains("names no lane axis"));
    assert!(refused(grouped(&by(vec!["col", "col"]), None)).contains("is named twice"));
    grouped(&by(vec!["col"]), None).unwrap();
    grouped(&by(vec!["row", "col"]), None).unwrap();
}

#[test]
fn a_key_is_one_integer_scalar_per_lane() {
    let mut c = ctx();
    let real = g::plane_in(&mut c, "k", "f64").unwrap();
    let key = g::plane_in(&mut c, "j", "i32").unwrap();
    let out = g::commit(&mut c, "y", real).unwrap();
    let by = |key| Membership {
        kind: "key",
        key: Some(key),
        ..Default::default()
    };
    let m = refused(g::group(&mut c, &by(real), &[], &[out]));
    assert!(
        m.contains("membership 'key' takes one integer scalar per lane as the key, got f64"),
        "{m}"
    );
    g::group(&mut c, &by(key), &[], &[out]).unwrap();
}

#[test]
fn a_key_group_with_scratch_needs_max_size_and_its_extent() {
    let mut c = ctx();
    let key = g::plane_in(&mut c, "j", "i32").unwrap();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let t = g::scratch(&mut c, "t", "f64", 16).unwrap();
    let out = g::commit(&mut c, "y", x).unwrap();
    let by = |max_size| Membership {
        kind: "key",
        key: Some(key),
        max_size,
        ..Default::default()
    };
    let m = refused(g::group(&mut c, &by(None), &[t], &[out]));
    assert!(
        m.contains("membership 'key' with scratch needs max_size"),
        "{m}"
    );
    let m = refused(g::group(&mut c, &by(Some(8)), &[t], &[out]));
    assert!(
        m.contains("has 16 elements on its rank axis but a group holds at most 8 lanes"),
        "{m}"
    );
    g::group(&mut c, &by(Some(16)), &[t], &[out]).unwrap();
    g::group(&mut c, &by(None), &[], &[out]).unwrap();
}

#[test]
fn a_scratch_over_a_run_time_axis_is_refused_and_over_static_axes_has_their_product() {
    let runtime = Membership {
        kind: "axes",
        grid: vec![("row", None), ("col", Some(4))],
        axes: vec!["row"],
        ..Default::default()
    };
    let m = refused(grouped(&runtime, Some(4)));
    assert!(
        m.contains("known only at run time; a scratch needs a size known at build"),
        "{m}"
    );
    let fixed = Membership {
        kind: "axes",
        grid: vec![("r", Some(2)), ("c", Some(3))],
        axes: vec!["r", "c"],
        ..Default::default()
    };
    let m = refused(grouped(&fixed, Some(5)));
    assert!(
        m.contains("has 5 elements on its rank axis but a group holds at most 6 lanes"),
        "{m}"
    );
    grouped(&fixed, Some(6)).unwrap();
}

#[test]
fn a_field_of_another_membership_is_refused() {
    let with_max = Membership {
        max_size: Some(4),
        ..Membership::count(4)
    };
    assert!(refused(grouped(&with_max, None))
        .contains("max_size belongs to membership 'key', not to 'count'"));
    let with_axes = Membership {
        axes: vec!["c"],
        ..Membership::count(4)
    };
    assert!(
        refused(grouped(&with_axes, None)).contains("axes and grid belongs to membership 'axes'")
    );
}

#[test]
fn a_scratch_has_one_axis_named_rank_and_a_group_holds_a_body() {
    let mut c = ctx();
    let m = refused(c_scratch(&mut c, &[("rank", Some(4)), ("k", Some(2))]));
    assert!(m.contains("has one axis, 'rank'"), "{m}");
    let m = refused(c_scratch(&mut c, &[("lane", Some(4))]));
    assert!(m.contains("has one axis, 'rank'"), "{m}");
    let m = refused(g::group(&mut c, &Membership::count(4), &[], &[]));
    assert!(m.contains("a group commits nothing"), "{m}");
}

/// A group of 4 whose scratch `t` is declared over `axes`.
fn c_scratch(c: &mut Ctx, axes: &[(&str, Option<i128>)]) -> Result<u32, String> {
    let given = vec![
        ("name".to_string(), Val::Str("t".into())),
        (
            "ttype".to_string(),
            raptor_ir::value::ttype_val("f64", &[], None),
        ),
        ("axes".to_string(), g::axes(axes)),
    ];
    let index = c.table.find("Scratch").unwrap();
    let t = c.new_from(index, &given).map_err(|f| format!("{f:?}"))?;
    let x = g::plane_in(c, "x", "f64")?;
    let out = g::commit(c, "y", x)?;
    g::group(c, &Membership::count(4), &[t], &[out])
}

// rule (f): stores

#[test]
fn a_group_commits_through_ordinary_updates_and_a_store_into_another_groups_scratch_is_refused() {
    let mut c = ctx();
    let owner = g::tile(&mut c).unwrap();
    assert_eq!(owner.outputs, ["y"]);
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let place = g::rank(&mut c).unwrap();
    let foreign = g::store(&mut c, "t", 8, x, place).unwrap();
    let m = refused(g::group(&mut c, &Membership::count(8), &[], &[foreign]));
    assert!(
        m.contains("names no scratch of this group; a group writes only the scratch it declares"),
        "{m}"
    );
    let own = g::scratch(&mut c, "u", "f64", 8).unwrap();
    let m = refused(g::group(&mut c, &Membership::count(8), &[own], &[foreign]));
    assert!(m.contains("names no scratch of this group"), "{m}");
}

#[test]
fn a_read_of_another_groups_scratch_is_refused() {
    let mut c = ctx();
    let other = g::scratch(&mut c, "t", "f64", 8).unwrap();
    let own = g::scratch(&mut c, "u", "f64", 8).unwrap();
    let place = g::rank(&mut c).unwrap();
    let read = g::read(&mut c, other, place, "clamp").unwrap();
    let out = g::commit(&mut c, "y", read).unwrap();
    let m = refused(g::group(&mut c, &Membership::count(8), &[own], &[out]));
    assert!(
        m.contains("scratch 't' is not declared by this group"),
        "{m}"
    );
}

#[test]
fn a_scratch_is_written_by_a_store_and_read_by_at_alone() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let t = g::scratch(&mut c, "t", "f64", 8).unwrap();
    let place = g::rank(&mut c).unwrap();
    let index = c.table.find("Update").unwrap();
    let add = [
        ("name", Val::Str("t".into())),
        ("value", Val::Node(x)),
        ("indices", Val::Tuple(vec![Val::Node(place)])),
        ("op", Val::Str("add".into())),
        ("role", Val::Str("scratch".into())),
        ("axes", g::axes(&[("rank", Some(8))])),
    ];
    let add: Vec<(String, Val)> = add.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let accumulate = c.new_from(index, &add).unwrap();
    let m = refused(g::group(&mut c, &Membership::count(8), &[t], &[accumulate]));
    assert!(
        m.contains("scratch 't' is written by an Update of op 'store'"),
        "{m}"
    );
}

#[test]
fn an_update_takes_the_role_plane_out_or_scratch_only() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let index = c.table.find("Update").unwrap();
    let given = [
        ("name", Val::Str("y".into())),
        ("value", Val::Node(x)),
        ("role", Val::Str("mutable".into())),
    ];
    let given: Vec<(String, Val)> = given.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let m = refused(c.new_from(index, &given).map_err(|f| format!("{f:?}")));
    assert!(
        m.contains("update 'y': the role 'mutable' is not one of ('plane_out', 'scratch')"),
        "{m}"
    );
}

// rule (g): the group-wide reduce and scan

#[test]
fn a_floating_sum_may_be_added_by_the_tree_in_rank_order_or_in_any_order() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    for order in ["tree", "lane", "any"] {
        let node = g::reduce(&mut c, "sum", order, x).unwrap();
        assert_eq!(c.node(node).unwrap().dtype, "f64");
    }
}

#[test]
fn an_order_other_than_the_tree_is_refused_where_the_operation_is_order_free() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    for order in ["lane", "any"] {
        let m = refused(g::reduce(&mut c, "min", order, x));
        assert!(m.contains(&format!("GroupReduce: order '{order}' states how a sum is added; 'min' is order-free, leave it at 'tree'")), "{m}");
    }
    let flag = g::apply(&mut c, "gt", &[x, x]).unwrap();
    assert!(refused(g::reduce(&mut c, "any", "lane", flag)).contains("is order-free"));
}

#[test]
fn a_reduce_takes_one_number_and_any_and_all_take_a_bool_and_give_one() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let flag = g::apply(&mut c, "gt", &[x, x]).unwrap();
    let all = g::reduce(&mut c, "all", "tree", flag).unwrap();
    assert_eq!(c.node(all).unwrap().dtype, "bool");
    assert!(refused(g::reduce(&mut c, "any", "tree", x)).contains("'any' takes a bool, got f64"));
    assert!(refused(g::reduce(&mut c, "sum", "tree", flag))
        .contains("a bool is combined by 'any' or 'all'"));
    assert!(refused(g::reduce(&mut c, "median", "tree", x)).contains("op 'median' is not one of"));
    let vector = c.make("Leaf", "f64", &[3], None, &[], "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'v'), ('quantity', None), ('wire_index', 1), ('sorted', False))").unwrap();
    assert!(refused(g::reduce(&mut c, "sum", "tree", vector))
        .contains("a value of shape [3] is not combined in this step"));
}

#[test]
fn a_scan_is_a_sum_and_inclusive_or_exclusive() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let inclusive = g::scan(&mut c, "sum", true, x).unwrap();
    let exclusive = g::scan(&mut c, "sum", false, x).unwrap();
    assert_ne!(inclusive, exclusive);
    assert!(refused(g::scan(&mut c, "min", true, x))
        .contains("GroupScan: op 'min' is not one of ('sum',)"));
}

// the phase of a reduce depends on its operand, not on the order of the sinks

#[test]
fn independent_reductions_share_a_phase() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let (sum, min) = (
        g::reduce(&mut c, "sum", "tree", x).unwrap(),
        g::reduce(&mut c, "min", "tree", x).unwrap(),
    );
    let body: Vec<u32> = [("a", sum), ("b", min), ("c", x)]
        .iter()
        .map(|&(name, value)| g::commit(&mut c, name, value).unwrap())
        .collect();
    let plan = c.phase_plan(&body);
    assert_eq!((plan.count(), plan.sinks), (2, vec![1, 1, 1]));
}

#[test]
fn a_reduce_of_a_reduces_result_waits_for_it() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let total = g::reduce(&mut c, "sum", "tree", x).unwrap();
    let centred = g::apply(&mut c, "sub", &[x, total]).unwrap();
    let spread = g::reduce(&mut c, "max", "tree", centred).unwrap();
    let out = g::commit(&mut c, "y", spread).unwrap();
    let plan = c.phase_plan(&[out]);
    assert_eq!((plan.count(), plan.sinks), (3, vec![2]));
    assert_eq!(plan.evaluated[&total], 0);
    assert_eq!(plan.evaluated[&spread], 1);
}

#[test]
fn a_reduce_sees_the_stores_before_its_sink_and_never_one_after() {
    // store t (phase 0); y = reduce(t[rank]) (the reduce reads in phase 0); store t again (phase 1, after the reduce's phase)
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let t = g::scratch(&mut c, "t", "f64", 8).unwrap();
    let place = g::rank(&mut c).unwrap();
    let first = g::store(&mut c, "t", 8, x, place).unwrap();
    let own = g::read(&mut c, t, place, "clamp").unwrap();
    let total = g::reduce(&mut c, "sum", "tree", own).unwrap();
    let out = g::commit(&mut c, "y", total).unwrap();
    let twice = g::apply(&mut c, "add", &[x, x]).unwrap();
    let second = g::store(&mut c, "t", 8, twice, place).unwrap();
    let body = [first, out, second];
    let plan = c.phase_plan(&body);
    assert_eq!(plan.sinks, vec![0, 1, 1]);
    assert_eq!(plan.evaluated[&total], 0);
    g::group(&mut c, &Membership::count(8), &[t], &body).unwrap();
}

#[test]
fn a_barrier_after_a_store_that_waited_for_a_reduce_puts_the_neighbour_read_after_it() {
    // y0 = reduce(x); store t = y0 (phase 1); a reduce over t's neighbour needs the barrier, which lifts it to phase 2
    let build = |barrier: bool| {
        tile_with(&mut ctx(), |c, x, t| {
            let place = g::rank(c).unwrap();
            let total = g::reduce(c, "sum", "tree", x).unwrap();
            let fill = g::store(c, "t", 8, total, place).unwrap();
            let one = g::whole(c, "i32", 1).unwrap();
            let next = g::apply(c, "add", &[place, one]).unwrap();
            let around = g::read(c, t, next, "wrap").unwrap();
            let spread = g::reduce(c, "max", "tree", around).unwrap();
            let out = g::commit(c, "y", spread).unwrap();
            if barrier {
                vec![fill, g::barrier(c).unwrap(), out]
            } else {
                vec![fill, out]
            }
        })
    };
    assert!(refused(build(false)).contains("in the phase that writes it"));
    build(true).unwrap();
}

// a Group is a root of a walk

/// The walk of `sinks` and its status.
fn canon(c: &mut Ctx, sinks: &[u32]) -> (i32, u32) {
    let (p, mut w) = (c as *mut Ctx, 0u32);
    let none: *const c_char = std::ptr::null();
    (
        rx_canonical(p, sinks.as_ptr(), sinks.len(), none, none, none, &mut w),
        w,
    )
}

/// The (role, name) of each slot of walk `w`, in the walk's order.
fn slots(c: &mut Ctx, w: u32) -> Vec<(String, String)> {
    let p = c as *mut Ctx;
    let (mut len, mut count) = (0usize, 0usize);
    assert_eq!(
        rx_walk_args(p, w, std::ptr::null_mut(), 0, &mut len, &mut count),
        RX_TRUNC
    );
    let mut buf = vec![0 as c_char; len + 1];
    assert_eq!(
        rx_walk_args(p, w, buf.as_mut_ptr(), len + 1, &mut len, &mut count),
        RX_OK
    );
    let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
    let texts: Vec<String> = bytes
        .split(|&b| b == 0)
        .take(count * 2)
        .map(|t| String::from_utf8(t.to_vec()).unwrap())
        .collect();
    texts
        .chunks(2)
        .map(|p| (p[0].clone(), p[1].clone()))
        .collect()
}

/// The digest of walk `w`.
fn digest_of(c: &mut Ctx, w: u32) -> String {
    let mut d = [0 as c_char; 65];
    assert_eq!(rx_digest(c as *mut Ctx, w, d.as_mut_ptr()), RX_OK);
    unsafe { std::ffi::CStr::from_ptr(d.as_ptr()) }
        .to_str()
        .unwrap()
        .to_string()
}

fn pair(role: &str, name: &str) -> (String, String) {
    (role.to_string(), name.to_string())
}

#[test]
fn the_fixtures_canonicalise_with_the_planes_they_read_and_write_as_slots_and_no_scratch() {
    // oracle: the manifest's role order (outputs, then inputs) and, inside a run, names in order
    let want = [
        ("tile", vec![pair("plane_out", "y"), pair("plane_in", "x")]),
        (
            "rows",
            vec![
                pair("plane_out", "least"),
                pair("plane_out", "place"),
                pair("plane_out", "total"),
                pair("plane_in", "x"),
            ],
        ),
        ("ring", vec![pair("plane_out", "y"), pair("plane_in", "x")]),
    ];
    for ((name, make, _), (want_name, expected)) in FIXTURES.into_iter().zip(want) {
        assert_eq!(name, want_name);
        let mut c = ctx();
        let group = make(&mut c).unwrap().group;
        let (status, w) = canon(&mut c, &[group]);
        assert_eq!(status, RX_OK, "{name}: {}", err(&mut c as *mut Ctx));
        assert_eq!(slots(&mut c, w), expected, "{name}");
    }
}

#[test]
fn a_key_operand_is_bound_like_any_plane() {
    let mut c = ctx();
    let key = g::plane_in(&mut c, "j", "i32").unwrap();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let out = g::commit(&mut c, "y", x).unwrap();
    let by_key = Membership {
        kind: "key",
        key: Some(key),
        ..Default::default()
    };
    let group = g::group(&mut c, &by_key, &[], &[out]).unwrap();
    let (status, w) = canon(&mut c, &[group]);
    assert_eq!(status, RX_OK);
    assert!(slots(&mut c, w).contains(&pair("plane_in", "j")));
}

#[test]
fn ordinary_sinks_sit_beside_a_group_and_two_groups_are_refused() {
    let mut c = ctx();
    let tile = g::tile(&mut c).unwrap().group;
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let beside = g::commit(&mut c, "z", x).unwrap();
    let (status, w) = canon(&mut c, &[tile, beside]);
    assert_eq!(status, RX_OK);
    assert!(slots(&mut c, w).contains(&pair("plane_out", "z")));
    let other = g::ring(&mut c).unwrap().group;
    let (status, _) = canon(&mut c, &[tile, other]);
    assert_eq!(status, RX_REFUSED);
    assert!(err(&mut c as *mut Ctx).contains("a kernel holds one Group"));
}

/// A tile whose last step is the operation `kind` of the share and the sum, over a group of `lanes` lanes with that many scratch cells.
fn tile_of(c: &mut Ctx, kind: &str, lanes: i128) -> u32 {
    let x = g::plane_in(c, "x", "f64").unwrap();
    let t = g::scratch(c, "t", "f64", lanes).unwrap();
    let place = g::rank(c).unwrap();
    let fill = g::store(c, "t", lanes, x, place).unwrap();
    let own = g::read(c, t, place, "clamp").unwrap();
    let total = g::reduce(c, "sum", "tree", own).unwrap();
    let share = g::apply(c, kind, &[own, total]).unwrap();
    let out = g::commit(c, "y", share).unwrap();
    g::group(c, &Membership::count(lanes), &[t], &[fill, out]).unwrap()
}

#[test]
fn two_groups_that_differ_in_one_body_node_or_in_their_membership_never_share_a_digest() {
    let mut c = ctx();
    let divided = tile_of(&mut c, "div", 8);
    let multiplied = tile_of(&mut c, "mul", 8);
    let narrower = tile_of(&mut c, "div", 4);
    let again = tile_of(&mut c, "div", 8);
    let digests: Vec<String> = [divided, multiplied, narrower, again]
        .iter()
        .map(|&group| {
            let (status, w) = canon(&mut c, &[group]);
            assert_eq!(status, RX_OK);
            digest_of(&mut c, w)
        })
        .collect();
    assert_eq!(digests[0], digests[3]);
    assert_ne!(digests[0], digests[1]);
    assert_ne!(digests[0], digests[2]);
    assert_ne!(digests[1], digests[2]);
}

#[test]
fn the_canonical_form_of_a_stored_document_equals_that_of_the_built_kernel() {
    for (name, make, stored) in FIXTURES {
        let mut built = ctx();
        let group = make(&mut built).unwrap().group;
        let (status, w) = canon(&mut built, &[group]);
        assert_eq!(status, RX_OK, "{name}");
        let want = (digest_of(&mut built, w), slots(&mut built, w));
        let mut loaded = ctx();
        let root = loaded_root(&mut loaded, stored);
        let (status, w) = canon(&mut loaded, &[root]);
        assert_eq!(status, RX_OK, "{name}");
        assert_eq!(
            (digest_of(&mut loaded, w), slots(&mut loaded, w)),
            want,
            "{name}"
        );
    }
}

/// The root of the document `text` loaded into `c`.
fn loaded_root(c: &mut Ctx, text: &str) -> u32 {
    let (status, roots) = load(c as *mut Ctx, text);
    assert_eq!(status, RX_OK);
    roots[0]
}

// an Update without a role is the one on the plane_out role it always was

#[test]
fn an_update_without_a_role_is_the_update_on_plane_out_in_the_dump_and_the_digest() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let bare = g::commit(&mut c, "y", x).unwrap();
    let given = [
        ("name", Val::Str("y".into())),
        ("value", Val::Node(x)),
        ("role", Val::Str("plane_out".into())),
    ];
    let given: Vec<(String, Val)> = given.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let stated = c.new_from(c.table.find("Update").unwrap(), &given).unwrap();
    let digests: Vec<String> = [bare, stated]
        .iter()
        .map(|&sink| {
            let (status, w) = canon(&mut c, &[sink]);
            assert_eq!(status, RX_OK);
            digest_of(&mut c, w)
        })
        .collect();
    assert_eq!(digests[0], digests[1]);
    assert!(dump(&mut c as *mut Ctx, &[bare]).contains("('role', 'plane_out')"));
}
