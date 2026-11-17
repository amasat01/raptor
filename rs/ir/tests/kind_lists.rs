//! The kind lists the core composes (`LINALG_OPS`, `CAST`, `OP_KINDS`) through the constant-name door. Oracle:
//! tests/data/kind_lists.json, the lists of the previous Python package.
use raptor_ir::consts::names;
use rcore::json;

/// Each composed list is the previous package's list, in the same order.
#[test]
fn the_composed_kind_lists_are_the_recorded_ones() {
    let all = json::parse(include_str!("data/kind_lists.json")).unwrap();
    assert_eq!(all.obj().len(), 3);
    for (name, want) in all.obj() {
        let got: Vec<String> = names(name).unwrap().into_iter().map(String::from).collect();
        assert_eq!(got, want.strs(), "{name}");
    }
}
