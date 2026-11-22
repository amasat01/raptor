//! The team stages of a block sum: a factor that is a bare read of an input plane is read in place, with no copy of it in scratch.
use raptor_ir::record::Ctx;
use raptor_wings::blockteam::blocks;
use raptor_wings::declare::rw_declare_kinds;

/// A stored block sum of a scalar read of the plane `x` whose role is `role` and whose value has the given shape.
fn blocks_of(role: &str, shape: &[i64]) -> Vec<(&'static str, u32)> {
    let mut c = Ctx::new().unwrap();
    assert_eq!(unsafe { rw_declare_kinds(&mut c) }, 0);
    let fields = format!(
        "(('kind', 'vocab_read'), ('role', '{role}'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))"
    );
    let payload = c.make("Leaf", "f64", shape, None, &[], &fields).unwrap();
    let key = fields.replace("'x'", "'k'");
    let key = c.make("Leaf", "i32", &[], None, &[], &key).unwrap();
    let count = c
        .make("Const", "i32", &[], None, &[], "(('literal', 4),)")
        .unwrap();
    let block = c
        .make(
            "BlockSum",
            "f64",
            shape,
            None,
            &[payload, key, count],
            "(('contract', 'ordered'), ('kept', ()))",
        )
        .unwrap();
    let write = c
        .make(
            "BlockSumWrite",
            "f64",
            &[],
            None,
            &[block],
            "(('kind', 'update'), ('name', 's'), ('role', 'plane_out'), ('op', 'add'), ('summation', 'plain'), ('order', 'any'), ('reduced', None), ('axes', ()))",
        )
        .unwrap();
    let found = blocks(&c, &[write], &[block], true).unwrap();
    assert_eq!(found.len(), 1);
    found[0].in_place.clone()
}

#[test]
fn a_block_sum_of_a_bare_scalar_read_of_an_input_plane_has_no_copy_stage() {
    let held = blocks_of("plane_in", &[]);
    assert_eq!(held.iter().map(|h| h.0).collect::<Vec<_>>(), ["u"]);
}

#[test]
fn a_block_sum_of_a_vector_read_keeps_its_copy_stage() {
    assert!(blocks_of("plane_in", &[3]).is_empty());
}
