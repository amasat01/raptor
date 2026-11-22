//! The key of an ordered scatter: an accumulate at several indices adds into the one cell of the target that the whole index
//! list names, the position in storage with the last axis fastest, as the ordered gradient's log states it.
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;
use raptor_wings::declare::rw_declare_kinds;
use raptor_wings::scatter::{logged, names};

/// The value of the integer expression `node` when the leaves named in `leaves` hold the given numbers.
fn eval(c: &Ctx, node: u32, leaves: &[(&str, i128)]) -> i128 {
    let ops = &c.nodes[node as usize].ops;
    match c.str_field(node, "kind") {
        Ok(kind) if kind == "vocab_read" => {
            let name = c.str_field(node, "name").unwrap();
            leaves.iter().find(|l| l.0 == name).unwrap().1
        }
        Ok(kind) if kind == "constant" => match c.field(node, "literal").unwrap() {
            Val::Int(n) => n,
            other => panic!("a constant of {other:?}"),
        },
        Ok(kind) => {
            let args: Vec<i128> = ops.iter().map(|&o| eval(c, o, leaves)).collect();
            match kind.as_str() {
                "cast" => args[0],
                "add" => args[0] + args[1],
                "mul" => args[0] * args[1],
                other => panic!("an index expression holds only cast, add and mul, not {other}"),
            }
        }
        Err(fault) => panic!("a node without a kind: {fault:?}"),
    }
}

/// An integer per-sample read of the plane `name`.
fn read(c: &mut Ctx, name: &str) -> u32 {
    let fields = format!(
        "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))"
    );
    c.make("Leaf", "i32", &[], None, &[], &fields).unwrap()
}

/// The cell each ordered add of `value` at `indices` over `axes` stores its key at, for the index values `at`.
fn key_of(axes: &str, indices: &[&str], at: &[(&str, i128)]) -> i128 {
    let mut c = Ctx::new().unwrap();
    assert_eq!(unsafe { rw_declare_kinds(&mut c) }, 0);
    let value = c
        .make("Const", "f64", &[], None, &[], "(('literal', 1.0),)")
        .unwrap();
    let mut ops = vec![value];
    ops.extend(indices.iter().map(|name| read(&mut c, name)));
    let fields = format!(
        "(('kind', 'update'), ('name', 'acc'), ('role', 'plane_out'), ('op', 'add'), ('summation', 'plain'), ('order', 'lane'), ('reduced', None), ('axes', {axes}))"
    );
    let sink = c.make("Update", "f64", &[], None, &ops, &fields).unwrap();
    let (stores, slots) = logged(&mut c, &[sink], (false, 32)).unwrap().unwrap();
    assert_eq!(slots, [("acc".to_string(), 1)]);
    let keys = &names("acc")[0];
    let store = stores
        .into_iter()
        .find(|&s| c.str_field(s, "name").is_ok_and(|n| n == *keys))
        .expect("a store of the key");
    eval(&c, c.nodes[store as usize].ops[0], at)
}

#[test]
fn the_key_of_a_two_index_accumulate_is_the_storage_cell() {
    let axes = "(('row', 4), ('col', 5))";
    for (row, col) in [(0, 0), (0, 4), (1, 0), (2, 3), (3, 4)] {
        let key = key_of(axes, &["row", "col"], &[("row", row), ("col", col)]);
        assert_eq!(key, row * 5 + col, "cell ({row}, {col}) of a 4 x 5 plane");
    }
}

#[test]
fn the_key_of_a_one_index_accumulate_is_the_index() {
    assert_eq!(key_of("(('bin', 9),)", &["row"], &[("row", 7)]), 7);
}
