//! The log of an ordered gradient: each ordered accumulate of a table's gradient is stored into slots of two log planes and one team
//! stage folds each target's log, each on a small hand-made kernel.
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_ctx_new, rx_last_error, rx_make, rx_new_loop};
use raptor_wings::blocks::LinalgForm;
use raptor_wings::declare::rw_declare_kinds;
use raptor_wings::stages::{stages, StageFacts, Stages};
use raptor_wings::team::{Grid, TeamLimits, Unit};
use std::ffi::{CStr, CString};

/// A context with the wings' kinds declared.
/// The facts of a build whose team limits do not matter to the kernels of these tests.
fn facts() -> StageFacts {
    let team = TeamLimits {
        form: LinalgForm::Auto,
        max_dim: 32,
        team_max_dim: 4096,
        copy_threads: 256,
        team_threads: 128,
        team_threads_large: 256,
        forced: 0,
    };
    StageFacts {
        routed: false,
        team,
        in_place: true,
        scatter: (256, 32),
    }
}

fn fresh() -> *mut Ctx {
    let c = rx_ctx_new();
    assert_eq!(unsafe { rw_declare_kinds(c) }, 0);
    c
}

/// A node of the class, its fields given as key text.
fn make(c: *mut Ctx, class: &str, dtype: &str, shape: &[i64], ops: &[u32], fields: &str) -> u32 {
    let (class, dtype, f) = (
        CString::new(class).unwrap(),
        CString::new(dtype).unwrap(),
        CString::new(fields).unwrap(),
    );
    let mut out = 0;
    let status = rx_make(
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
    assert_eq!(status, 0, "{}", unsafe {
        CStr::from_ptr(rx_last_error(c)).to_string_lossy()
    });
    out
}

/// A per-sample read of the plane `name`.
fn read(c: *mut Ctx, name: &str) -> u32 {
    let fields = format!(
        "((\"kind\", \"vocab_read\"), (\"role\", \"per_sample\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"
    );
    make(c, "Leaf", "f64", &[], &[], &fields)
}

/// A per-sample read of the integer plane `name`.
fn read_int(c: *mut Ctx, name: &str) -> u32 {
    let fields = format!(
        "((\"kind\", \"vocab_read\"), (\"role\", \"per_sample\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"
    );
    make(c, "Leaf", "i64", &[], &[], &fields)
}

/// The ordered accumulate of `value` into cell `index` of the plane `name`, the gradient of a sum kept over the lane axis `net`.
fn accumulate(c: *mut Ctx, name: &str, (value, index, key): (u32, u32, u32)) -> u32 {
    let fields = format!(
        "((\"name\", \"{name}\"), (\"role\", \"accum_out\"), (\"kept\", ((\"net\", None),)))"
    );
    make(
        c,
        "OrderedAccumWrite",
        "f64",
        &[],
        &[value, index, key],
        &fields,
    )
}

/// Three accumulates, two into `g` and one into `h`: `g` holds two slots per lane and `h` one, each target folds in a stage of its
/// own after the stage that fills the logs, and the stores take the slots in program order.
#[test]
fn each_target_is_folded_by_its_own_keyed_stage_after_the_stage_that_fills_its_log() {
    let c = fresh();
    let (x, key) = (read(c, "x"), read_int(c, "k"));
    let (zero, one) = (read_int(c, "i0"), read_int(c, "i1"));
    let sinks = [
        accumulate(c, "g", (x, zero, key)),
        accumulate(c, "g", (x, one, key)),
        accumulate(c, "h", (x, zero, key)),
    ];
    let Stages::Cut(cut) = stages(unsafe { &mut *c }, &sinks, &facts()).unwrap() else {
        panic!("a kernel with ordered accumulates is cut into stages");
    };
    assert_eq!(
        cut.stages.len(),
        3,
        "the stage that fills the logs, then one fold per target"
    );
    let fill = &cut.stages[0];
    assert_eq!(fill.sinks.len(), 6, "two log stores per accumulate");
    assert!(fill.team.is_none());
    for (stage, target, width) in [(&cut.stages[1], "g", 2), (&cut.stages[2], "h", 1)] {
        let team = stage.team.as_ref().expect("a fold is a team stage");
        let [values, cells] = raptor_wings::logs::names(target);
        assert_eq!(team.steps.len(), 1);
        assert_eq!((team.steps[0].op, team.steps[0].n), ("ordered_fold", width));
        let roles: Vec<(&str, &str)> = team.steps[0]
            .planes
            .iter()
            .map(|(r, n)| (r.as_str(), n.as_str()))
            .collect();
        assert_eq!(
            roles,
            [
                ("v", values.as_str()),
                ("i", cells.as_str()),
                ("out", target)
            ]
        );
        assert_eq!(team.out.as_deref(), Some(target));
        let units: Vec<_> = team
            .planes
            .iter()
            .map(|p| (p.name.as_str(), p.per_lane, p.unit))
            .collect();
        assert_eq!(
            units,
            [
                (values.as_str(), width, Unit::Lane),
                (cells.as_str(), width, Unit::Key)
            ]
        );
        let Grid::Keys { keyed, .. } = &team.grid else {
            panic!("a fold runs one block per key")
        };
        assert_eq!(
            (keyed.name.as_str(), keyed.width, keyed.kept.clone()),
            (target, 1, vec![(Some("net".to_string()), None)])
        );
        assert_eq!(
            team.threads, 128,
            "a fold of a few slots takes the small team"
        );
    }
}

/// The accumulate of a vector is refused by name: the log holds one scalar per slot.
#[test]
fn an_ordered_accumulate_of_a_vector_is_refused_by_name() {
    let c = fresh();
    let key = read_int(c, "k");
    let fields = "((\"kind\", \"vocab_read\"), (\"role\", \"per_sample\"), (\"name\", \"v\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))";
    let vector = make(c, "Leaf", "f64", &[3], &[], fields);
    let fields = "((\"name\", \"g\"), (\"role\", \"accum_out\"), (\"kept\", ()))";
    let sink = make(
        c,
        "OrderedAccumWrite",
        "f64",
        &[3],
        &[vector, key, key],
        fields,
    );
    let Stages::Refused(why) = stages(unsafe { &mut *c }, &[sink], &facts()).unwrap() else {
        panic!("a vector accumulate is refused");
    };
    assert!(why.contains("the one into g is not a scalar"), "{why}");
}

/// A loop at nesting `depth` over `start..stop` by `step` holding the sinks `body`. It carries one value, which it leaves as it is:
/// a lowered loop carries at least one.
fn lp(c: *mut Ctx, depth: i64, (start, stop, step): (i64, i64, i64), body: &[u32]) -> u32 {
    let (index, start_value) = (loop_index(c, depth), read(c, "x"));
    let fields = format!("((\"depth\", {depth}), (\"slot\", 0), (\"name\", \"s\"))");
    let carry = make(c, "LoopCarry", "f64", &[], &[], &fields);
    let none: [u32; 0] = [];
    let mut out = 0;
    let status = rx_new_loop(
        c,
        index,
        &carry,
        1,
        &start_value,
        &carry,
        start,
        stop,
        step,
        body.as_ptr(),
        body.len(),
        std::ptr::null(),
        none.as_ptr(),
        0,
        std::ptr::null(),
        0,
        std::ptr::null(),
        &mut out,
    );
    assert_eq!(status, 0, "{}", unsafe {
        CStr::from_ptr(rx_last_error(c)).to_string_lossy()
    });
    out
}

/// The index of a loop at nesting depth `depth`.
fn loop_index(c: *mut Ctx, depth: i64) -> u32 {
    let fields = format!("((\"depth\", {depth}), (\"name\", \"h{depth}\"), (\"bounds\", None))");
    make(c, "LoopIndex", "i32", &[], &[], &fields)
}

/// Three accumulates of the net kernel's shape in a loop of eight trips take 3 x 8 slots: one slot per site and iteration, one
/// fold of that width, and the loop keeps its stores of the log in its body.
#[test]
fn the_accumulates_of_a_loop_take_one_slot_per_site_and_iteration() {
    let c = fresh();
    let (x, key, zero) = (read(c, "x"), read_int(c, "k"), read_int(c, "i0"));
    let body: Vec<u32> = (0..3).map(|_| accumulate(c, "g", (x, zero, key))).collect();
    let looped = lp(c, 0, (0, 8, 1), &body);
    let Stages::Cut(cut) = stages(unsafe { &mut *c }, &[looped], &facts()).unwrap() else {
        panic!("a kernel with ordered accumulates in a loop is cut into stages");
    };
    assert_eq!(
        cut.stages.len(),
        2,
        "the stage that fills the logs, then the fold"
    );
    let fold = cut.stages[1].team.as_ref().expect("a fold is a team stage");
    assert_eq!((fold.steps[0].op, fold.steps[0].n), ("ordered_fold", 24));
    assert!(fold.planes.iter().all(|p| p.per_lane == 24));
    let kept = &cut.stages[0].sinks;
    assert_eq!(kept.len(), 1, "the loop is the one sink of the fill stage");
    let ctx = unsafe { &*c };
    let mut inner = vec![];
    raptor_ir::dump::refs(&ctx.field(kept[0], "body_sinks").unwrap(), &mut inner);
    assert_eq!(inner.len(), 6, "two log stores per accumulate");
    assert!(inner
        .iter()
        .all(|&s| ctx.table.is_a(ctx.nodes[s as usize].class, "LogStore")));
}

/// A loop that steps by two, a loop inside a loop and a vector accumulate in a loop are each refused with their reason.
#[test]
fn a_loop_the_log_cannot_hold_is_refused_with_its_reason() {
    let c = fresh();
    let (x, key, zero) = (read(c, "x"), read_int(c, "k"), read_int(c, "i0"));
    let stepped = lp(c, 0, (0, 8, 2), &[accumulate(c, "g", (x, zero, key))]);
    let inner = lp(c, 1, (0, 4, 1), &[accumulate(c, "g", (x, zero, key))]);
    let nested = lp(c, 0, (0, 8, 1), &[accumulate(c, "g", (inner, zero, key))]);
    for (sink, reason) in [(stepped, "a loop of step 2"), (nested, "nested loops")] {
        let Stages::Refused(why) = stages(unsafe { &mut *c }, &[sink], &facts()).unwrap() else {
            panic!("{reason} is refused");
        };
        assert!(why.contains(reason), "{why}");
    }
}
