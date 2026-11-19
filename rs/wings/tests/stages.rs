//! The cut of a kernel at its reductions over samples and the plain form of its block sums, each on a small hand-made kernel.
use raptor_ir::record::Ctx;
use raptor_support::safe::{rx_ctx_free, rx_ctx_new, rx_last_error, rx_make};
use raptor_wings::blocks::{plain_blocks, BlockLimits, LinalgForm};
use raptor_wings::declare::rw_declare_kinds;
use raptor_wings::stages::{stages, StageFacts, Stages};
use raptor_wings::team::{TeamLimits, WarpEntry};
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
        warp_threads: None,
        warp_eigh: vec![],
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

/// `out = x / sum(x)`: the sum is read after the samples have all been summed.
#[test]
fn a_value_read_after_its_sum_is_cut_into_two_stages() {
    let c = fresh();
    let x = read(c, "x");
    let sum = make(
        c,
        "ReduceSamples",
        "f64",
        &[],
        &[x],
        "((\"op\", \"sum\"), (\"key\", None), (\"count\", None), (\"contract\", \"auto\"), (\"kept\", ()))",
    );
    let div = make(
        c,
        "Op",
        "f64",
        &[],
        &[x, sum],
        "((\"kind\", \"div\"), (\"literal\", None))",
    );
    let out = make(
        c,
        "Assign",
        "f64",
        &[],
        &[div],
        "((\"kind\", \"assign\"), (\"role\", \"mutable\"), (\"name\", \"out\"))",
    );
    let found = stages(unsafe { &mut *c }, &[out], &facts()).unwrap();
    let Stages::Cut(cut) = found else {
        panic!("not cut: {found:?}")
    };
    assert_eq!(cut.stages.len(), 2);
    assert_eq!(cut.stages[0].cells_out, ["cut0"]);
    assert_eq!(cut.stages[1].cells_in, ["cut0"]);
    assert_eq!(cut.stages[1].recomputed, Vec::<String>::new());
    assert!(cut.cells[0].internal);
    let plain = make(
        c,
        "Assign",
        "f64",
        &[],
        &[x],
        "((\"kind\", \"assign\"), (\"role\", \"mutable\"), (\"name\", \"y\"))",
    );
    assert_eq!(
        stages(unsafe { &mut *c }, &[plain], &facts()).unwrap(),
        Stages::Whole
    );
    rx_ctx_free(c);
}

/// A block sum the thread form does not keep is written as the keyed accumulates of its plain sum.
#[test]
fn a_block_sum_is_written_per_entry_under_the_thread_form() {
    let c = fresh();
    let (u, r) = (read(c, "u"), read(c, "r"));
    let payload = make(
        c,
        "Op",
        "f64",
        &[],
        &[u, r],
        "((\"kind\", \"mul\"), (\"literal\", None))",
    );
    let key = make(c, "Const", "i32", &[], &[], "((\"literal\", 0),)");
    let count = make(c, "Const", "i32", &[], &[], "((\"literal\", 1),)");
    let block = make(
        c,
        "BlockSum",
        "f64",
        &[],
        &[payload, key, count],
        "((\"contract\", \"auto\"), (\"kept\", ((\"key\", None),)))",
    );
    let write = make(
        c,
        "BlockSumWrite",
        "f64",
        &[],
        &[block],
        "((\"name\", \"out\"),)",
    );
    let limits = BlockLimits {
        rolled_fold_min: 16,
        team_max_dim: 4096,
    };
    let thread = plain_blocks(unsafe { &mut *c }, &[write], LinalgForm::Thread, &limits).unwrap();
    assert_eq!(thread.map(|sinks| sinks.len()), Some(1));
    let team = plain_blocks(unsafe { &mut *c }, &[write], LinalgForm::Team, &limits).unwrap();
    assert_eq!(team, None);
    rx_ctx_free(c);
}

/// A team-form solve in the body of a loop is a launch of its own, so the split refuses it by name (it once indexed the depths
/// with a sink that is not in the flat order).
#[test]
fn a_team_op_inside_a_loop_is_refused_by_name() {
    let c = fresh();
    let leaf = |role: &str, name: &str, shape: &[i64]| {
        let fields = format!(
            "((\"kind\", \"vocab_read\"), (\"role\", \"{role}\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"
        );
        make(c, "Leaf", "f64", shape, &[], &fields)
    };
    let (a, b) = (leaf("mat_in", "A", &[40, 40]), leaf("vec_in", "b", &[40]));
    let index = make(
        c,
        "LoopIndex",
        "int32",
        &[],
        &[],
        "((\"depth\", 0), (\"name\", \"_\"), (\"bounds\", (0, 3)))",
    );
    let carry = make(
        c,
        "LoopCarry",
        "f64",
        &[40],
        &[],
        "((\"depth\", 0), (\"slot\", 0), (\"name\", \"s\"))",
    );
    let factor = make(
        c,
        "Op",
        "f64",
        &[40, 40],
        &[a],
        "((\"kind\", \"chol\"), (\"literal\", None))",
    );
    let next = make(
        c,
        "Op",
        "f64",
        &[40],
        &[factor, carry],
        "((\"kind\", \"cho_solve\"), (\"literal\", None))",
    );
    let store = make(
        c,
        "Assign",
        "f64",
        &[40],
        &[next],
        "((\"kind\", \"assign\"), (\"role\", \"mutable\"), (\"name\", \"y\"))",
    );
    let parts = raptor_ir::build::LoopParts {
        index,
        carries: &[carry],
        inits: &[b],
        nexts: &[next],
        range: (0, 3, 1),
        body_sinks: &[store],
        exit_cond: None,
        exit_values: &[],
        count: None,
        count_tail: false,
        dynamic: None,
    };
    let looped = unsafe { &mut *c }.build_loop(&parts).unwrap();
    let found = stages(unsafe { &mut *c }, &[looped], &facts()).unwrap();
    let Stages::Refused(why) = found else {
        panic!("not refused: {found:?}")
    };
    assert!(
        why.contains("op \"cho_solve\" at n=40 is worked by a team of threads")
            && why.contains("cannot sit inside a loop of the body"),
        "{why}"
    );
    rx_ctx_free(c);
}

/// The warp row of the device card for the values of an order-32 `eigh`.
fn warp_row() -> WarpEntry {
    WarpEntry {
        kind: "eigh".into(),
        n: 32,
        scalar: "float32".into(),
        outputs: vec![],
    }
}

/// The facts of a build with the given linear-algebra form and warp rows.
fn warp_facts(form: LinalgForm, rows: Vec<WarpEntry>) -> StageFacts {
    let mut facts = facts();
    facts.team.form = form;
    facts.team.warp_eigh = rows;
    facts.team.warp_threads = Some(32);
    facts
}

/// The sinks of `y = eigenvalues of the order-n matrix A` (and `v = its vectors` when `vectors`), with the factor read inside an
/// arm of a dispatch when `arm` is set, and the `(threads, op)` of every team step of the stages the split cuts them into.
fn eigh_steps(n: i64, vectors: bool, arm: bool, facts: &StageFacts) -> Vec<(i64, String)> {
    let c = fresh();
    let leaf = |role: &str, name: &str, dtype: &str, shape: &[i64]| {
        let fields = format!(
            "((\"kind\", \"vocab_read\"), (\"role\", \"{role}\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"
        );
        make(c, "Leaf", dtype, shape, &[], &fields)
    };
    let op = |kind: &str, shape: &[i64], ops: &[u32]| {
        let fields = format!("((\"kind\", \"{kind}\"), (\"literal\", None))");
        make(c, "Op", "f64", shape, ops, &fields)
    };
    let store = |name: &str, shape: &[i64], value: u32| {
        let fields =
            format!("((\"kind\", \"assign\"), (\"role\", \"mutable\"), (\"name\", \"{name}\"))");
        make(c, "Assign", "f64", shape, &[value], &fields)
    };
    let matrix = leaf("mat_in", "A", "f64", &[n, n]);
    let factor = op("eigh", &[n, n], &[matrix]);
    let values = op("eigh_w", &[n], &[factor]);
    let mut sinks = vec![];
    if arm {
        // the factor is read only by the arms of a dispatch on the plane `k`
        let key = leaf("per_sample", "k", "i32", &[]);
        let other = op("exp", &[n], &[values]);
        let dispatch = make(c, "Dispatch", "f64", &[n], &[key, values, other], "()");
        sinks.push(store("y", &[n], dispatch));
    } else {
        sinks.push(store("y", &[n], values));
    }
    if vectors {
        let v = op("eigh_v", &[n, n], &[factor]);
        sinks.push(store("v", &[n, n], v));
    }
    let found = stages(unsafe { &mut *c }, &sinks, facts).unwrap();
    let steps = match found {
        Stages::Cut(cut) => cut
            .stages
            .iter()
            .filter_map(|stage| stage.team.as_ref())
            .flat_map(|team| team.steps.iter().map(|s| (team.threads, s.op.to_string())))
            .collect(),
        _ => vec![],
    };
    rx_ctx_free(c);
    steps
}

/// Oracle: the old rule of the warp form (the removed hawk test of the same names): the values of an order-32 `eigh` under the auto
/// form, on a build that has the warp row, are one `eigh_warp` stage of 32 threads.
#[test]
fn the_values_of_an_order_32_eigh_are_one_warp_stage() {
    let facts = warp_facts(LinalgForm::Auto, vec![warp_row()]);
    assert_eq!(
        eigh_steps(32, false, false, &facts),
        [(32, "eigh_warp".to_string())]
    );
}

/// Oracle: the same removed test's neighbours: every other build keeps the form it had (no stage in the per-lane form, an `eigh` team
/// stage when the build asks for teams or the order is above the lane's limit).
#[test]
fn every_other_build_keeps_the_form_it_had() {
    let row = || vec![warp_row()];
    let none = || warp_facts(LinalgForm::Auto, vec![]);
    let team_of =
        |steps: Vec<(i64, String)>| steps.iter().all(|(_, op)| op == "eigh") && !steps.is_empty();
    // no warp row (a host target, a double build on a slow device, another scalar type)
    assert_eq!(eigh_steps(32, false, false, &none()), []);
    // thread form: no team stage at all; team form: a team stage that is not the warp one
    assert_eq!(
        eigh_steps(32, false, false, &warp_facts(LinalgForm::Thread, row())),
        []
    );
    assert!(team_of(eigh_steps(
        32,
        false,
        false,
        &warp_facts(LinalgForm::Team, row())
    )));
    // the vectors are read: the row is for the values only, so the per-lane form stays
    assert_eq!(
        eigh_steps(32, true, false, &warp_facts(LinalgForm::Auto, row())),
        []
    );
    // order 31 has no row and stays per lane; order 33 is above the lane's limit and is a team stage
    assert_eq!(
        eigh_steps(31, false, false, &warp_facts(LinalgForm::Auto, row())),
        []
    );
    assert!(team_of(eigh_steps(
        33,
        false,
        false,
        &warp_facts(LinalgForm::Auto, row())
    )));
    // a factor only an arm of a dispatch reads has no stage of its own to be a warp one
    assert_eq!(
        eigh_steps(32, false, true, &warp_facts(LinalgForm::Auto, row())),
        []
    );
}

/// An order-32 `eigh` whose values a loop stores each iteration is in the loop body, which a launch of its own cannot be: the
/// warp row does not apply and the per-lane form stays.
#[test]
fn an_eigh_in_a_loop_body_keeps_the_per_lane_form() {
    let c = fresh();
    let leaf = |name: &str| {
        let fields = format!(
            "((\"kind\", \"vocab_read\"), (\"role\", \"mat_in\"), (\"name\", \"{name}\"), (\"quantity\", None), (\"wire_index\", 0), (\"sorted\", False))"
        );
        make(c, "Leaf", "f64", &[32, 32], &[], &fields)
    };
    let op = |kind: &str, shape: &[i64], ops: &[u32]| {
        let fields = format!("((\"kind\", \"{kind}\"), (\"literal\", None))");
        make(c, "Op", "f64", shape, ops, &fields)
    };
    let a = leaf("A");
    let index = make(
        c,
        "LoopIndex",
        "int32",
        &[],
        &[],
        "((\"depth\", 0), (\"name\", \"_\"), (\"bounds\", (0, 3)))",
    );
    let carry = make(
        c,
        "LoopCarry",
        "f64",
        &[32, 32],
        &[],
        "((\"depth\", 0), (\"slot\", 0), (\"name\", \"s\"))",
    );
    // the matrix read by the factorisation changes with the carry, so the factorisation is part of the body
    let sum = op("add", &[32, 32], &[carry, a]);
    let values = op("eigh_w", &[32], &[op("eigh", &[32, 32], &[sum])]);
    let store = make(
        c,
        "Assign",
        "f64",
        &[32],
        &[values],
        "((\"kind\", \"assign\"), (\"role\", \"mutable\"), (\"name\", \"y\"))",
    );
    let parts = raptor_ir::build::LoopParts {
        index,
        carries: &[carry],
        inits: &[a],
        nexts: &[sum],
        range: (0, 3, 1),
        body_sinks: &[store],
        exit_cond: None,
        exit_values: &[],
        count: None,
        count_tail: false,
        dynamic: None,
    };
    let looped = unsafe { &mut *c }.build_loop(&parts).unwrap();
    let facts = warp_facts(LinalgForm::Auto, vec![warp_row()]);
    assert_eq!(
        stages(unsafe { &mut *c }, &[looped], &facts).unwrap(),
        Stages::Whole
    );
    rx_ctx_free(c);
}
