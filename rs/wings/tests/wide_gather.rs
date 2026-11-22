//! A run of reads of one plane as a gather: a run over the last axis of a plane with several axes is one gather with its lead
//! indices, and reads that differ in a lead index or in their edge rule are not a run.
use raptor_ir::record::Ctx;
use raptor_wings::wide::{rw_wide_gather, GatherAnswer};

/// An integer constant of the type.
fn int(c: &mut Ctx, n: i64) -> u32 {
    let fields = format!("(('literal', {n}),)");
    c.make("Const", "i32", &[], None, &[], &fields).unwrap()
}

/// The plane `t` of two axes (4 rows of 8), and the lane's row index `r` (a per-sample read).
fn plane_and_row(c: &mut Ctx) -> (u32, u32) {
    let plane = "(('kind', 'table_read'), ('role', 'lookup'), ('name', 't'), ('axes', (('a', 4), ('b', 8))))";
    let plane = c.make("Leaf", "f64", &[], None, &[], plane).unwrap();
    let row = "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'r'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
    (plane, c.make("Leaf", "i32", &[], None, &[], row).unwrap())
}

/// The vector of the reads of `plane` at `(rows[k], columns[k])` under the edge rule `edges[k]`.
fn vector(c: &mut Ctx, plane: u32, rows: &[u32], columns: &[i64], edges: &[&str]) -> u32 {
    let mut reads = vec![];
    for k in 0..columns.len() {
        let column = int(c, columns[k]);
        let fields = format!(
            "((\"kind\", \"at\"), (\"literal\", None), (\"edge\", \"{}\"))",
            edges[k]
        );
        reads.push(
            c.make("At", "f64", &[], None, &[plane, rows[k], column], &fields)
                .unwrap(),
        );
    }
    c.make(
        "Op",
        "f64",
        &[reads.len() as i64],
        None,
        &reads,
        "(('kind', 'vec'), ('literal', None))",
    )
    .unwrap()
}

/// What the door answers for `v`: None, or (plane, lead, base, start, step, clamped).
fn gather(c: &Ctx, v: u32) -> Option<(u32, Vec<u32>, u32, i64, i64, i32)> {
    let mut answer = GatherAnswer::default();
    let status = unsafe { rw_wide_gather(c as *const Ctx as *mut Ctx, v, &mut answer) };
    assert_eq!(status, 0);
    let GatherAnswer {
        found,
        plane,
        base,
        start,
        step,
        clamped,
        nlead,
        lead,
    } = answer;
    (found == 1).then(|| (plane, lead[..nlead].to_vec(), base, start, step, clamped))
}

#[test]
fn a_run_over_the_last_axis_of_a_two_axis_plane_is_one_gather_with_its_lead() {
    let mut c = Ctx::new().unwrap();
    let (plane, row) = plane_and_row(&mut c);
    let v = vector(&mut c, plane, &[row; 4], &[2, 3, 4, 5], &["clamp"; 4]);
    // the base is UINT32_MAX when the running index has no run-time part
    assert_eq!(gather(&c, v), Some((plane, vec![row], u32::MAX, 2, 1, 1)));
}

#[test]
fn reads_at_different_lead_indices_are_not_a_run() {
    let mut c = Ctx::new().unwrap();
    let (plane, row) = plane_and_row(&mut c);
    let other = int(&mut c, 1);
    let v = vector(
        &mut c,
        plane,
        &[row, row, other, other],
        &[0, 1, 2, 3],
        &["clamp"; 4],
    );
    assert!(gather(&c, v).is_none());
}

#[test]
fn reads_under_different_edge_rules_are_not_a_run() {
    let mut c = Ctx::new().unwrap();
    let (plane, row) = plane_and_row(&mut c);
    let v = vector(
        &mut c,
        plane,
        &[row; 3],
        &[0, 1, 2],
        &["clamp", "unchecked", "clamp"],
    );
    assert!(gather(&c, v).is_none());
}
