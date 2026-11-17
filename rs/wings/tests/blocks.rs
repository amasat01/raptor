//! The block predicates of a keyed-sum payload: its factors, whether the ordered contract keeps it as one block, whether it can be
//! one block by structure. Oracle: tests/data/blocks.json, recorded from the previous Python functions on the same payloads.
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;
use raptor_wings::blocks::{block_shaped, factors, ordered_case};
use rcore::json::{self, J};
use std::collections::HashMap;

/// The graph a spec describes; a leaf name is one node however often it appears.
fn build(c: &mut Ctx, spec: &J, leaves: &mut HashMap<String, u32>) -> u32 {
    if let Some(name) = spec.get("leaf").and_then(J::str) {
        if let Some(&known) = leaves.get(name) {
            return known;
        }
        let shape: Vec<i64> = spec
            .get("shape")
            .unwrap()
            .arr()
            .iter()
            .map(J::int)
            .collect();
        let dtype = spec.get("dtype").and_then(J::str).unwrap();
        let fields = format!("(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))");
        let made = c.make("Leaf", dtype, &shape, None, &[], &fields).unwrap();
        leaves.insert(name.to_string(), made);
        return made;
    }
    let args: Vec<u32> = spec
        .get("args")
        .unwrap()
        .arr()
        .iter()
        .map(|a| build(c, a, leaves))
        .collect();
    c.make_op(
        spec.get("kind").and_then(J::str).unwrap(),
        &args,
        &Val::None,
    )
    .unwrap()
}

/// The node a path of operand positions leads to from `root`.
fn follow(c: &Ctx, root: u32, path: &J) -> u32 {
    path.arr()
        .iter()
        .fold(root, |n, i| c.node(n).unwrap().ops[i.int() as usize])
}

/// Every recorded payload answers as the previous functions did.
#[test]
fn the_block_predicates_answer_as_the_recorded_ones() {
    let all = json::parse(include_str!("data/blocks.json")).unwrap();
    let cases = all.get("cases").unwrap().arr();
    assert_eq!(cases.len(), 272);
    for case in cases {
        let id = case.get("id").and_then(J::str).unwrap();
        let mut c = Ctx::new().unwrap();
        let payload = build(&mut c, case.get("spec").unwrap(), &mut HashMap::new());
        let want = case.get("expect").unwrap();
        let contract = case.get("contract").and_then(J::str).unwrap_or("auto");
        let limit = case
            .get("max_dim")
            .filter(|m| !matches!(m, J::Null))
            .map(J::int);
        match case.get("fn").and_then(J::str).unwrap() {
            "factors" => match (factors(&c, payload), want) {
                (None, J::Null) => {}
                (Some((u, v, rows)), J::Obj(_)) => {
                    assert_eq!(u, follow(&c, payload, want.get("u").unwrap()), "{id}");
                    assert_eq!(v, follow(&c, payload, want.get("v").unwrap()), "{id}");
                    assert_eq!(rows, want.get("rows").unwrap().int(), "{id}");
                }
                (got, _) => panic!("{id}: factors found: {}", got.is_some()),
            },
            "ordered_case" => {
                let wanted = matches!(want, J::Bool(true));
                assert_eq!(ordered_case(&c, payload, contract, limit), wanted, "{id}");
            }
            _ => {
                let wanted = matches!(want, J::Bool(true));
                assert_eq!(block_shaped(&c, payload, contract, limit), wanted, "{id}");
            }
        }
    }
}
