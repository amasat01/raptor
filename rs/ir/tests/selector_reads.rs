//! The reads of a dispatch selector: the plane leaves its closure reaches, a loop carry standing for the leaves of its start value
//! and its update and never for a plane of its own name. Oracle: the planes written into each hand-made IR dump below.
use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};
use raptor_ir::dump::rx_load;
use raptor_ir::record::Ctx;

/// A plane read leaf of the given role and name.
fn leaf(role: &str, name: &str) -> String {
    format!("n|Leaf|f64|()|None|()|(('kind', 'vocab_read'), ('role', '{role}'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))")
}

/// A loop carry placeholder of slot `slot`.
fn carry(slot: usize, name: &str) -> String {
    format!("n|LoopCarry|f64|()|None|()|(('depth', 0), ('slot', {slot}), ('name', '{name}'))")
}

/// A binary node `kind` over two nodes.
fn op(kind: &str, a: usize, b: usize) -> String {
    format!("n|Op|f64|()|None|(#{a}, #{b})|(('kind', '{kind}'), ('literal', None))")
}

/// A loop over the starts, carries and updates (node numbers), with the starts as its operands.
fn lp(starts: &[usize], carries: &[usize], nexts: &[usize]) -> String {
    let list = |v: &[usize]| v.iter().map(|n| format!("#{n}, ")).collect::<String>();
    format!(
        "n|Loop|f64|()|None|({})|(('index', #0), ('carries', ({})), ('nexts', ({})), ('start', 0), ('stop', 4), ('step', 1), ('trip', 4), ('carry_count', {}), ('body_sinks', ()), ('_body_key', None), ('exit_cond', None), ('exit_values', ()), ('count', None), ('count_tail', False), ('dynamic', None), ('_free_end', {}))",
        list(starts),
        list(carries),
        list(nexts),
        carries.len(),
        starts.len()
    )
}

/// Loads the nodes (node 0 must be the loop index) and returns the reads of the dispatch whose selector is the last node.
fn reads_of(nodes: &[String]) -> Vec<(String, String)> {
    let mut text = String::from("raptor.ir.dump/1\n");
    // the selector is the last node cast to the integer a dispatch chooses by; its branches are that node too
    let last = nodes.len() - 1;
    let all = nodes.iter().cloned().chain([
        format!("n|Op|i32|()|None|(#{last},)|(('kind', 'cast'), ('literal', 'i32'))"),
        format!(
            "n|Dispatch|f64|()|None|(#{}, #{last}, #{last})|()",
            last + 1
        ),
    ]);
    for line in all.clone() {
        text += &(line + "\n");
    }
    for at in 0..all.count() {
        text += &format!("root|#{at}\n");
    }
    let c = rx_ctx_new();
    let mut roots = vec![0u32; 64];
    let mut n = 0;
    // SAFETY: `c` is the live context just made; the text and the buffers outlive the call.
    let status = unsafe {
        rx_load(
            c,
            text.as_ptr().cast(),
            text.len(),
            roots.as_mut_ptr(),
            roots.len(),
            &mut n,
        )
    };
    // SAFETY: `c` is live and `rx_last_error` returns its NUL-terminated message.
    let why = unsafe { std::ffi::CStr::from_ptr(raptor_ir::door::rx_last_error(c)) }
        .to_string_lossy()
        .into_owned();
    assert_eq!(status, RX_OK, "the dump loads: {why}");
    // SAFETY: `c` is the live context; nothing else holds it.
    let ctx: &Ctx = unsafe { &*c };
    let found = ctx
        .selector_reads(roots[n - 1])
        .expect("the reads")
        .into_iter()
        .collect();
    // SAFETY: `c` is the live context made above and is not used after.
    unsafe { rx_ctx_free(c) };
    found
}

/// The index node every loop of these dumps names.
const INDEX: &str = "n|LoopIndex|i32|()|None|()|(('depth', 0), ('name', '_'), ('bounds', (0, 3)))";

/// The pairs as the tests state them.
fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
    list.iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

#[test]
fn a_selector_over_a_carry_started_from_a_plane_reads_that_plane_and_not_the_carry() {
    // the shape of the jump kernel: xs starts from the plane x, is updated by a constant-free add, and the selector mixes xs with t
    let nodes = [
        INDEX.to_string(),
        leaf("per_sample", "x"),
        leaf("uniform", "t"),
        carry(0, "xs"),
        op("add", 3, 3),
        lp(&[1], &[3], &[4]),
        op("add", 3, 2),
    ];
    assert_eq!(
        reads_of(&nodes),
        pairs(&[("per_sample", "x"), ("uniform", "t")])
    );
}

#[test]
fn a_carry_whose_update_reads_another_plane_includes_that_plane() {
    let nodes = [
        INDEX.to_string(),
        leaf("per_sample", "x"),
        leaf("per_sample", "y"),
        carry(0, "acc"),
        op("add", 3, 2),
        lp(&[1], &[3], &[4]),
        op("add", 3, 3),
    ];
    assert_eq!(
        reads_of(&nodes),
        pairs(&[("per_sample", "x"), ("per_sample", "y")])
    );
}

#[test]
fn two_carries_that_feed_each_other_terminate_and_read_both_starts() {
    let nodes = [
        INDEX.to_string(),
        leaf("per_sample", "x"),
        leaf("uniform", "t"),
        carry(0, "a"),
        carry(1, "b"),
        lp(&[1, 2], &[3, 4], &[4, 3]),
        op("add", 3, 3),
    ];
    assert_eq!(
        reads_of(&nodes),
        pairs(&[("per_sample", "x"), ("uniform", "t")])
    );
}
