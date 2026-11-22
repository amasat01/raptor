//! The lane space of a kernel: its axes, the key run of a keyed accumulate and the thread box.
//!
//! Each kernel is a hand-made graph over `lane = SampleIndex`. A split of the lane by `n` is `j = lane / n` and
//! `s = lane - j * n`, so the divisor `n` makes two axes of strides `[n, 1]`. The expected answers are derived in each test
//! from the key's form along those axes; the number of forms in a key run is the number of lane axes.
use raptor_ir::record::Ctx;
use raptor_wings::boxing::choose_box;
use raptor_wings::lanes::{key_run, varying_nodes, KeyRun};

/// A context and the node builders the kernels below share.
struct Kernel {
    c: Ctx,
    lane: u32,
}

impl Kernel {
    fn new() -> Kernel {
        let mut c = Ctx::new().unwrap();
        let lane = c.make("SampleIndex", "i32", &[], None, &[], "()").unwrap();
        Kernel { c, lane }
    }

    fn int(&mut self, n: i64) -> u32 {
        let fields = format!("(('literal', {n}),)");
        self.c
            .make("Const", "i32", &[], None, &[], &fields)
            .unwrap()
    }

    fn op(&mut self, kind: &str, ops: &[u32]) -> u32 {
        let fields = format!("(('kind', '{kind}'), ('literal', None))");
        self.c.make("Op", "i32", &[], None, ops, &fields).unwrap()
    }

    fn leaf(&mut self, name: &str, role: &str, sorted: bool) -> u32 {
        let sorted = if sorted { "True" } else { "False" };
        let fields = format!(
            "(('kind', 'vocab_read'), ('role', '{role}'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', {sorted}))"
        );
        self.c.make("Leaf", "i32", &[], None, &[], &fields).unwrap()
    }

    /// `lane / n`.
    fn quotient(&mut self, n: i64) -> u32 {
        let n = self.int(n);
        self.op("div", &[self.lane, n])
    }

    /// `lane - (lane / n) * n`: the position inside a row of `n` lanes.
    fn remainder(&mut self, n: i64) -> u32 {
        let (q, k) = (self.quotient(n), self.int(n));
        let rows = self.op("mul", &[q, k]);
        self.op("sub", &[self.lane, rows])
    }

    /// An accumulate into plane `acc` at `key` of a value that also divides the lane by `divisors`, as the real kernels'
    /// bodies do; it keeps those divisors in the kernel without touching the key.
    fn accumulate(&mut self, key: u32, divisors: &[i64]) -> u32 {
        let mut value = self
            .c
            .make("Const", "f64", &[], None, &[], "(('literal', 1.0),)");
        for &d in divisors {
            let q = self.quotient(d);
            let as_real = self.c.make(
                "Op",
                "f64",
                &[],
                None,
                &[q],
                "(('kind', 'cast'), ('literal', None))",
            );
            let sum = self.c.make(
                "Op",
                "f64",
                &[],
                None,
                &[value.unwrap(), as_real.unwrap()],
                "(('kind', 'add'), ('literal', None))",
            );
            value = sum;
        }
        let fields = "(('kind', 'update'), ('name', 'acc'), ('role', 'plane_out'), ('op', 'add'), ('summation', 'plain'), ('reduced', None))";
        self.c
            .make("Update", "f64", &[], None, &[value.unwrap(), key], fields)
            .unwrap()
    }

    /// The key run of `sink` with a run of up to 32 lanes, the width of a warp.
    fn run(&self, sink: u32) -> Option<KeyRun> {
        key_run(&self.c, &[sink], sink, 32, 8).unwrap()
    }
}

/// A key `lane / n` of a lane split by `n`: the key is constant along the fastest axis.
fn quotient_key_run(n: i64) -> Option<KeyRun> {
    let mut k = Kernel::new();
    let key = k.quotient(n);
    let sink = k.accumulate(key, &[]);
    k.run(sink)
}

#[test]
fn a_key_constant_along_the_fastest_axis_has_a_proven_run_of_the_warp_width() {
    // 8192 is a multiple of 32, so a warp never leaves one row of the split: the key is one value across it
    let run = quotient_key_run(8192).unwrap();
    assert_eq!((run.form, run.width), ("proven", 32));
}

#[test]
fn a_run_shortens_to_the_largest_power_of_two_that_divides_the_row_length() {
    // rows of 48 lanes: a warp of 32 straddles two keys; 16 divides 48, so a run of 16 never does
    let run = quotient_key_run(48).unwrap();
    assert_eq!((run.form, run.width), ("proven", 16));
}

#[test]
fn a_key_that_changes_with_the_lane_inside_a_row_has_no_run() {
    // the key is the position inside a row: it moves along the fastest axis, so no two lanes share it
    let mut k = Kernel::new();
    let key = k.remainder(64);
    let sink = k.accumulate(key, &[]);
    assert_eq!(k.run(sink), None);
}

#[test]
fn a_key_read_from_a_plane_has_no_run() {
    let mut k = Kernel::new();
    let key = k.leaf("key", "plane_in", false);
    let sink = k.accumulate(key, &[8192]);
    assert_eq!(k.run(sink), None);
}

#[test]
fn a_key_scaled_by_the_launch_sample_count_still_holds_one_value_along_the_fastest_axis() {
    // the sample count is the same for every lane, so `j * n_samples + 3` varies only with the slow axis
    let mut k = Kernel::new();
    let (j, count, three) = (k.quotient(8192), k.leaf("n", "nsamples", false), k.int(3));
    let scaled = k.op("mul", &[j, count]);
    let key = k.op("add", &[scaled, three]);
    let sink = k.accumulate(key, &[]);
    let run = k.run(sink).unwrap();
    assert_eq!((run.form, run.width), ("proven", 32));
}

#[test]
fn a_quotient_key_over_divisors_that_are_not_a_chain_is_tested_at_run_time() {
    // 64 and 6 do not divide each other: one flat axis, on which `lane / 64` is a quotient, not an affine step
    let mut k = Kernel::new();
    let key = k.quotient(64);
    let sink = k.accumulate(key, &[6]);
    let run = k.run(sink).unwrap();
    assert_eq!((run.form, run.width, run.forms.len()), ("tested", 32, 1));
}

#[test]
fn a_key_plane_declared_sorted_has_a_run_found_at_run_time() {
    let mut k = Kernel::new();
    let key = k.leaf("key", "plane_in", true);
    let sink = k.accumulate(key, &[8192]);
    let run = k.run(sink).unwrap();
    assert_eq!((run.form, run.width), ("sorted", 32));
}

#[test]
fn a_sorted_plane_plus_a_constant_is_still_sorted() {
    let mut k = Kernel::new();
    let (plane, three) = (k.leaf("key", "plane_in", true), k.int(3));
    let key = k.op("add", &[plane, three]);
    let sink = k.accumulate(key, &[8192]);
    assert_eq!(k.run(sink).unwrap().form, "sorted");
}

#[test]
fn a_sorted_plane_minus_a_constant_is_still_sorted() {
    let mut k = Kernel::new();
    let (plane, three) = (k.leaf("key", "plane_in", true), k.int(3));
    let key = k.op("sub", &[plane, three]);
    let sink = k.accumulate(key, &[8192]);
    assert_eq!(k.run(sink).unwrap().form, "sorted");
}

#[test]
fn a_sorted_plane_scaled_or_reversed_is_not_known_to_be_sorted() {
    // `2 * key` and `3 - key` are not the plane's own order (a product is opaque, a negation reverses it)
    let mut k = Kernel::new();
    let (plane, two, three) = (k.leaf("key", "plane_in", true), k.int(2), k.int(3));
    let doubled = k.op("mul", &[plane, two]);
    let negated = k.op("sub", &[three, plane]);
    for key in [doubled, negated] {
        let sink = k.accumulate(key, &[8192]);
        assert_eq!(k.run(sink), None);
    }
}

#[test]
fn a_sorted_plane_plus_a_lane_term_is_not_known_to_be_sorted() {
    // the added term moves with a lane axis, so the key is no longer the plane's order
    let mut k = Kernel::new();
    let plane = k.leaf("key", "plane_in", true);
    let s = k.remainder(8192);
    let key = k.op("add", &[plane, s]);
    let sink = k.accumulate(key, &[]);
    assert_eq!(k.run(sink), None);
}

#[test]
fn a_sink_with_no_key_has_no_run() {
    let mut k = Kernel::new();
    let value = k.int(1);
    let fields = "(('kind', 'assign'), ('name', 'o'), ('role', 'mutable'))";
    let sink =
        k.c.make("Assign", "i32", &[], None, &[value], fields)
            .unwrap();
    assert_eq!(k.run(sink), None);
}

#[test]
fn a_lone_divisor_is_a_two_axis_lane_space() {
    let run = quotient_key_run(4).unwrap();
    assert_eq!(run.forms.len(), 2);
}

#[test]
fn divisors_that_are_not_a_chain_fall_back_to_one_flat_axis() {
    let mut k = Kernel::new();
    let key = k.quotient(4);
    let sink = k.accumulate(key, &[6]);
    assert_eq!(k.run(sink).unwrap().forms.len(), 1);
}

#[test]
fn a_chain_of_divisors_gives_one_axis_each() {
    // 32 and 8 divide each other: rows of 32 lanes made of groups of 8, so three axes
    let mut k = Kernel::new();
    let key = k.quotient(32);
    let sink = k.accumulate(key, &[8]);
    assert_eq!(k.run(sink).unwrap().forms.len(), 3);
}

#[test]
fn a_node_varies_along_an_axis_only_when_its_value_depends_on_it() {
    // the key `lane / 8192` moves with the slow axis (0) and is the same along the fast one (1); the lane moves with both
    let mut k = Kernel::new();
    let key = k.quotient(8192);
    let sink = k.accumulate(key, &[]);
    let (along_slow, along_fast) = (
        varying_nodes(&k.c, &[sink], 0).unwrap(),
        varying_nodes(&k.c, &[sink], 1).unwrap(),
    );
    assert!(along_slow.contains(&key) && !along_fast.contains(&key));
    assert!(along_slow.contains(&k.lane) && along_fast.contains(&k.lane));
}

/// `out = feat[position inside a row of 8]`: every lane of a row reads a different cell, and every row reads the same cells.
fn row_reader() -> (Kernel, u32) {
    let mut k = Kernel::new();
    let feat = k.leaf("feat", "lookup", false);
    let index = k.remainder(8);
    let read =
        k.c.make("At", "i32", &[], None, &[feat, index], "()")
            .unwrap();
    let fields = "(('kind', 'assign'), ('name', 'o'), ('role', 'mutable'))";
    let out =
        k.c.make("Assign", "i32", &[], None, &[read], fields)
            .unwrap();
    (k, out)
}

#[test]
fn a_plane_read_the_same_in_every_row_is_boxed_along_the_row_axis() {
    // the read ignores the slow axis (rows), whose extent is a run-time number: the box takes the card's 4 lanes, one row apart
    let (k, out) = row_reader();
    let chosen = choose_box(&k.c, &[out], 1, 4).unwrap().unwrap();
    assert_eq!((chosen.axis, chosen.lanes, chosen.stride), (0, 4, 8));
}

#[test]
fn a_kernel_with_one_lane_axis_has_no_box() {
    // `feat[lane + 1]` has no axis but the fastest
    let mut k = Kernel::new();
    let feat = k.leaf("feat", "lookup", false);
    let one = k.int(1);
    let index = k.op("add", &[k.lane, one]);
    let read =
        k.c.make("At", "i32", &[], None, &[feat, index], "()")
            .unwrap();
    let fields = "(('kind', 'assign'), ('name', 'o'), ('role', 'mutable'))";
    let out =
        k.c.make("Assign", "i32", &[], None, &[read], fields)
            .unwrap();
    assert!(choose_box(&k.c, &[out], 1, 4).unwrap().is_none());
}
