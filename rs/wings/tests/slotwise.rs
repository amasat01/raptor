//! The slotwise rewrite over a context's nodes. Oracle: tests/data/slotwise.json, recorded by running the previous Python pass on
//! the same root graphs. A case gives its root specs, the structural text of the rewritten roots (one line per distinct node in
//! post-order, so creation order does not matter) and which roots come back as the same node.
use raptor_ir::record::Ctx;
use raptor_ir::value::{parse, text_val, ttype_val, Val};
use raptor_wings::declare::declare_kinds;
use raptor_wings::slotwise::slotwise;
use rcore::json::{self, J};
use std::collections::HashMap;

/// A node of the class from its parameters.
fn made(c: &mut Ctx, class: &str, given: Vec<(&str, Val)>) -> u32 {
    let index = c.table.find(class).unwrap();
    let given: Vec<(String, Val)> = given.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    c.new_from(index, &given).unwrap()
}

/// The type value of a node.
fn type_of(c: &Ctx, n: u32) -> Val {
    let node = c.node(n).unwrap();
    ttype_val(&node.dtype, &node.shape, node.tag.as_deref())
}

/// The nodes a spec lists, built left to right.
fn all(c: &mut Ctx, specs: &[J], leaves: &mut HashMap<String, u32>) -> Vec<u32> {
    specs.iter().map(|s| build(c, s, leaves)).collect()
}

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
        let n = c.make("Leaf", dtype, &shape, None, &[], &fields).unwrap();
        leaves.insert(name.to_string(), n);
        return n;
    }
    if let Some(value) = spec.get("const") {
        let dtype = spec.get("dtype").and_then(J::str).unwrap();
        let literal = match value {
            J::Num(v) if dtype == "i32" => Val::Int(*v as i128),
            J::Num(v) => Val::Float(*v),
            _ => panic!("a constant is a number"),
        };
        return made(
            c,
            "Const",
            vec![("literal", literal), ("ttype", ttype_val(dtype, &[], None))],
        );
    }
    if let Some(arms) = spec.get("select") {
        let n = all(c, arms.arr(), leaves);
        let ty = type_of(c, n[1]);
        let given = vec![
            ("cond", Val::Node(n[0])),
            ("on_true", Val::Node(n[1])),
            ("on_false", Val::Node(n[2])),
            ("ttype", ty),
        ];
        return made(c, "Select", given);
    }
    if let Some(d) = spec.get("dispatch") {
        let selector = build(c, d.get("selector").unwrap(), leaves);
        let branches = all(c, d.get("branches").unwrap().arr(), leaves);
        let ty = type_of(c, branches[0]);
        let given = vec![
            ("selector", Val::Node(selector)),
            (
                "branches",
                Val::Tuple(branches.iter().map(|&b| Val::Node(b)).collect()),
            ),
            ("ttype", ty),
        ];
        let core = made(c, "Dispatch", given);
        return match d.get("strategy").and_then(J::str).unwrap() {
            "switch" => core,
            word => {
                let class = format!("{}Dispatch", word[..1].to_uppercase() + &word[1..]);
                let (ops, node) = (c.node(core).unwrap().ops.clone(), c.node(core).unwrap());
                let (dtype, shape) = (node.dtype.clone(), node.shape.clone());
                c.make(&class, &dtype, &shape, None, &ops, "()").unwrap()
            }
        };
    }
    if let Some(name) = spec.get("assign").and_then(J::str) {
        let value = build(c, spec.get("value").unwrap(), leaves);
        let params = format!("(('name', '{name}'), ('value', #0))");
        return c.new_node("Assign", &params, &[value]).unwrap();
    }
    if spec.get("loop").is_some() {
        return counting_loop(c, leaves);
    }
    let args = all(c, spec.get("args").unwrap().arr(), leaves);
    let lit = parse(spec.get("lit").and_then(J::str).unwrap()).unwrap();
    c.make_op(spec.get("kind").and_then(J::str).unwrap(), &args, &lit)
        .unwrap()
}

/// A loop `c = c + x` over three trips.
fn counting_loop(c: &mut Ctx, leaves: &mut HashMap<String, u32>) -> u32 {
    let x = build(
        c,
        &json::parse(r#"{"leaf":"x","shape":[],"dtype":"f64"}"#).unwrap(),
        leaves,
    );
    let index = c
        .make(
            "LoopIndex",
            "int32",
            &[],
            None,
            &[],
            "(('depth', 0), ('name', '_'), ('bounds', (0, 3)))",
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
    let next = c.make_op("add", &[carry, x], &Val::None).unwrap();
    let parts = raptor_ir::build::LoopParts {
        index,
        carries: &[carry],
        inits: &[x],
        nexts: &[next],
        range: (0, 3, 1),
        body_sinks: &[],
        exit_cond: None,
        exit_values: &[],
        count: None,
        count_tail: false,
        dynamic: None,
    };
    c.build_loop(&parts).unwrap()
}

/// The structural post-order text of the graphs under `roots`.
fn canon(c: &Ctx, roots: &[u32]) -> Vec<String> {
    let (mut seen, mut lines, mut memo) = (HashMap::new(), vec![], HashMap::new());
    let ids: Vec<usize> = roots
        .iter()
        .map(|&r| visit(c, r, &mut seen, &mut lines, &mut memo))
        .collect();
    lines.extend(ids.iter().map(|i| format!("root|{i}")));
    lines
}

/// The index of the line of node `h`, adding the lines of what it reads first.
fn visit(
    c: &Ctx,
    h: u32,
    seen: &mut HashMap<String, usize>,
    lines: &mut Vec<String>,
    memo: &mut HashMap<u32, usize>,
) -> usize {
    if let Some(&i) = memo.get(&h) {
        return i;
    }
    let node = c.node(h).unwrap();
    let ops: Vec<Val> = node
        .ops
        .clone()
        .into_iter()
        .map(|o| Val::Int(visit(c, o, seen, lines, memo) as i128))
        .collect();
    let class = c.table.classes[node.class].name.clone();
    let text = |f: &str| text_val(&c.field(h, f).unwrap());
    let key = match class.as_str() {
        "Op" => format!("{}|{}", c.str_field(h, "kind").unwrap(), text("literal")),
        "Const" => text("literal"),
        "Leaf" | "Assign" => c.str_field(h, "name").unwrap(),
        _ => String::new(),
    };
    let shape = Val::Tuple(node.shape.iter().map(|&d| Val::Int(d as i128)).collect());
    let tag = node.tag.clone().map_or(Val::None, Val::Str);
    let line = format!(
        "{class}|{}|{}|{}|{}|{key}",
        node.dtype,
        text_val(&shape),
        text_val(&tag),
        text_val(&Val::Tuple(ops))
    );
    let next = seen.len();
    let i = *seen.entry(line.clone()).or_insert(next);
    if i == lines.len() {
        lines.push(line);
    }
    memo.insert(h, i);
    i
}

/// Every recorded case is rewritten to the text the previous pass gave, and the same roots come back unchanged.
#[test]
fn the_rewrite_gives_what_the_recorded_pass_gave() {
    let data = json::parse(include_str!("data/slotwise.json")).unwrap();
    let cases = data.get("cases").unwrap().arr();
    assert_eq!(cases.len(), 55);
    for case in cases {
        let id = case.get("id").and_then(J::str).unwrap();
        let mut c = Ctx::new().unwrap();
        declare_kinds(&mut c).unwrap();
        let roots = all(
            &mut c,
            case.get("roots").unwrap().arr(),
            &mut HashMap::new(),
        );
        let done = slotwise(&mut c, &roots).unwrap_or_else(|e| {
            panic!(
                "{id}: refused {}",
                matches!(e, raptor_ir::record::Fault::Refused(_))
            )
        });
        let want: Vec<String> = case
            .get("expect")
            .unwrap()
            .arr()
            .iter()
            .map(|l| l.str().unwrap().to_string())
            .collect();
        assert_eq!(canon(&c, &done), want, "{id}");
        let same: Vec<bool> = case
            .get("same")
            .unwrap()
            .arr()
            .iter()
            .map(|b| matches!(b, J::Bool(true)))
            .collect();
        let got: Vec<bool> = done.iter().zip(&roots).map(|(a, b)| a == b).collect();
        assert_eq!(got, same, "{id}: unchanged roots");
    }
}
