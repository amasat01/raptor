//! The canonical walk of a kernel's sinks with the companion, finished-count, fused-steps and branch-mix rules, and the typed reads of
//! the walk. Oracle: tests/data/walk.json, recorded from the previous Python `cells.walk` on the same sinks.
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::typed::RxType;
use raptor_ir::value::{ttype_val, Val};
use raptor_ir::walkreads::{rx_walk_prior, rx_walk_slot_type, rx_walk_static_selector};
use raptor_wings::cells::kernel_walk;
use raptor_wings::declare::declare_kinds;
use raptor_wings::segment::{rw_walk_strategies, strategies, Strategy};
use rcore::json::{self, J};
use std::ffi::{c_char, CStr, CString};

/// A node of the class from its parameters.
fn made(c: &mut Ctx, class: &str, given: Vec<(&str, Val)>) -> u32 {
    let index = c.table.find(class).unwrap();
    let given: Vec<(String, Val)> = given.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    c.new_from(index, &given).unwrap()
}

/// A leaf of the role and kind.
fn leaf(c: &mut Ctx, (role, kind): (&str, &str), name: &str, dtype: &str) -> u32 {
    let fields = format!("(('kind', '{kind}'), ('role', '{role}'), ('name', '{name}'), ('quantity', None), ('wire_index', 0), ('sorted', False))");
    c.make("Leaf", dtype, &[], None, &[], &fields).unwrap()
}

/// A sink of the class over a value.
fn sink(c: &mut Ctx, class: &str, name: &str, value: u32) -> u32 {
    let params = format!("(('name', '{name}'), ('value', #0))");
    c.new_node(class, &params, &[value]).unwrap()
}

/// A summing or other sample sum of `value`.
fn sample_sum(c: &mut Ctx, name: &str, value: u32, op: &str) -> u32 {
    made(
        c,
        "SampleSum",
        vec![
            ("name", Val::Str(name.into())),
            ("value", Val::Node(value)),
            ("op", Val::Str(op.into())),
        ],
    )
}

/// A dispatch of two branches pinned to the strategy word.
fn dispatch(c: &mut Ctx, (x, y): (u32, u32), word: &str) -> u32 {
    let selector = leaf(c, ("per_sample", "vocab_read"), "k", "i32");
    let (a, b) = (
        c.make_op("add", &[x, y], &Val::None).unwrap(),
        c.make_op("mul", &[x, y], &Val::None).unwrap(),
    );
    let ty = ttype_val("f64", &[], None);
    let given = vec![
        ("selector", Val::Node(selector)),
        ("branches", Val::Tuple(vec![Val::Node(a), Val::Node(b)])),
        ("ttype", ty),
    ];
    let core = made(c, "Dispatch", given);
    if word == "switch" {
        return core;
    }
    let class = format!("{}Dispatch", word[..1].to_uppercase() + &word[1..]);
    let ops = c.node(core).unwrap().ops.clone();
    c.make(&class, "f64", &[], None, &ops, "()").unwrap()
}

/// The sinks and declared slots of a recorded case.
fn case_sinks(c: &mut Ctx, name: &str) -> (Vec<u32>, String) {
    let per = ("per_sample", "vocab_read");
    let (x, y) = (leaf(c, per, "x", "f64"), leaf(c, per, "y", "f64"));
    let sum = |c: &mut Ctx, a, b| c.make_op("add", &[a, b], &Val::None).unwrap();
    let mut declared = String::from("()");
    let sinks = match name {
        "no_companion" => vec![{
            let v = sum(c, x, y);
            sink(c, "Assign", "o", v)
        }],
        "one_summing_sample_sum" => vec![{
            let v = c.make_op("mul", &[x, y], &Val::None).unwrap();
            sample_sum(c, "s", v, "sum")
        }],
        "max_sample_sum_has_no_companion" => vec![sample_sum(c, "s", x, "max")],
        "collision" => vec![
            sample_sum(c, "s", x, "sum"),
            sample_sum(c, "s_comp", y, "sum"),
        ],
        "finishing" => {
            let done = c.make_op("gt", &[x, y], &Val::None).unwrap();
            vec![sink(c, "Assign", "o", x), sink(c, "Finish", "done", done)]
        }
        "auto_dispatch" | "predicated_dispatch" | "segmented_dispatch" | "switch_dispatch" => {
            let d = dispatch(c, (x, y), name.trim_end_matches("_dispatch"));
            vec![sink(c, "Assign", "o", d)]
        }
        "fused_steps_read" => {
            let steps = leaf(c, ("lookup", "table_read"), "fused_steps", "f64");
            vec![{
                let v = sum(c, x, steps);
                sink(c, "Assign", "o", v)
            }]
        }
        "declared_slot" => {
            declared = "(Slot(role='lookup', name='extra', ttype=TensorType(shape=(), dtype='i32', tag=None)),)".into();
            vec![sink(c, "Assign", "o", x)]
        }
        "prior_read" => {
            let p = leaf(c, ("per_sample", "prior_read"), "p", "f64");
            vec![{
                let v = sum(c, x, p);
                sink(c, "Assign", "o", v)
            }]
        }
        "sum_finish_auto" => {
            let done = c.make_op("gt", &[x, y], &Val::None).unwrap();
            let d = dispatch(c, (x, y), "auto");
            vec![
                sample_sum(c, "s", x, "sum"),
                sink(c, "Finish", "done", done),
                sink(c, "Assign", "o", d),
            ]
        }
        other => panic!("no builder for {other}"),
    };
    (sinks, declared)
}

/// Every recorded kernel walks to the digest the previous function gave, with the same finish, prior reads and dispatch groups, or
/// is refused with its text.
#[test]
fn the_kernel_walk_is_the_recorded_one() {
    let data = json::parse(include_str!("data/walk.json")).unwrap();
    assert_eq!(data.arr().len(), 13);
    for case in data.arr() {
        let id = case.get("id").and_then(J::str).unwrap();
        let mut c = Ctx::new().unwrap();
        declare_kinds(&mut c).unwrap();
        let (sinks, declared) = case_sinks(&mut c, id);
        let declared = raptor_ir::canon::declared_slots(Some(&declared)).unwrap();
        let got = kernel_walk(&mut c, &sinks, &declared, None);
        if let Some(want) = case.get("error").and_then(J::str) {
            assert!(
                matches!(&got, Err(Fault::Refused(m)) if m == want),
                "{id}: wanted the refusal {want:?}"
            );
            continue;
        }
        let w = got.unwrap_or_else(|e| panic!("{id}: {}", matches!(e, Fault::Refused(_))));
        let walk = c.walk(w).unwrap();
        assert_eq!(
            walk.digest,
            case.get("digest").and_then(J::str).unwrap(),
            "{id}: digest"
        );
        assert_eq!(
            walk.finish.as_deref(),
            case.get("finish").and_then(J::str),
            "{id}: finish"
        );
        assert_eq!(
            walk.prior,
            case.get("prior").unwrap().strs(),
            "{id}: prior reads"
        );
        let groups: Vec<(String, Strategy)> = strategies(&c, w).unwrap();
        let want = case.get("dispatches").unwrap().arr();
        assert_eq!(groups.len(), want.len(), "{id}: dispatch groups");
        for ((name, strategy), row) in groups.iter().zip(want) {
            let word = match strategy {
                Strategy::Auto => "auto",
                Strategy::Predicated => "predicated",
                Strategy::Segmented => "segmented",
                Strategy::Switch => "switch",
            };
            assert_eq!(
                (name.as_str(), word),
                (row.arr()[0].str().unwrap(), row.arr()[3].str().unwrap()),
                "{id}"
            );
        }
    }
}

/// The names a names-protocol door writes.
fn names_of(call: impl Fn(*mut c_char, usize, *mut usize, *mut usize) -> i32) -> Vec<String> {
    let (mut buf, mut len, mut count) = ([0 as c_char; 512], 0, 0);
    assert_eq!(call(buf.as_mut_ptr(), buf.len(), &mut len, &mut count), 0);
    let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
    bytes
        .split(|&b| b == 0)
        .take(count)
        .map(|n| String::from_utf8(n.to_vec()).unwrap())
        .collect()
}

/// The typed reads of a walk give what the previous walk's members held: slot types, prior reads, and the group strategies and
/// static selectors.
#[test]
fn the_typed_reads_of_a_walk_are_the_recorded_members() {
    let data = json::parse(include_str!("data/walk.json")).unwrap();
    for case in data.arr().iter().filter(|c| c.get("error").is_none()) {
        let id = case.get("id").and_then(J::str).unwrap();
        let mut c = Ctx::new().unwrap();
        declare_kinds(&mut c).unwrap();
        let (sinks, declared) = case_sinks(&mut c, id);
        let declared = raptor_ir::canon::declared_slots(Some(&declared)).unwrap();
        let w = kernel_walk(&mut c, &sinks, &declared, None).unwrap();
        let ctx: *mut Ctx = &mut c;
        for row in case.get("slot_types").unwrap().arr() {
            let (key, ty) = (row.arr()[0].strs(), row.arr()[1].arr());
            let (role, name) = (
                CString::new(key[0].clone()).unwrap(),
                CString::new(key[1].clone()).unwrap(),
            );
            let mut got = RxType {
                rank: 0,
                dims: [0; 8],
                dtype: 0,
                tag: [0; 64],
            };
            assert_eq!(
                unsafe { rx_walk_slot_type(ctx, w, role.as_ptr(), name.as_ptr(), &mut got) },
                0,
                "{id} {key:?}"
            );
            let shape: Vec<i64> = ty[0].arr().iter().map(J::int).collect();
            assert_eq!(&got.dims[..got.rank as usize], &shape[..], "{id} {key:?}");
            let tag = unsafe { CStr::from_ptr(got.tag.as_ptr()) }
                .to_str()
                .unwrap();
            assert_eq!(
                Some(tag).filter(|t| !t.is_empty()),
                ty[2].str(),
                "{id} {key:?}"
            );
        }
        let prior =
            names_of(|b, cap, len, count| unsafe { rx_walk_prior(ctx, w, b, cap, len, count) });
        assert_eq!(prior, case.get("prior").unwrap().strs(), "{id}");
        let words = names_of(|b, cap, len, count| unsafe {
            rw_walk_strategies(ctx, w, b, cap, len, count)
        });
        let rows = case.get("dispatches").unwrap().arr();
        assert_eq!(
            words,
            rows.iter()
                .map(|r| r.arr()[3].str().unwrap().to_string())
                .collect::<Vec<_>>(),
            "{id}"
        );
        for (i, row) in rows.iter().enumerate() {
            let mut flag = 9;
            assert_eq!(
                unsafe { rx_walk_static_selector(ctx, w, i as i64, &mut flag) },
                0
            );
            assert_eq!(
                flag == 1,
                matches!(row.arr()[2], J::Bool(true)),
                "{id}: static selector {i}"
            );
        }
    }
}

/// The dispatch strategies are the previous package's four, derived from the wing manifests (`switch` first, then the manifests'
/// own order; the previous tuple listed `switch` second).
#[test]
fn the_dispatch_strategies_are_the_recorded_four() {
    let mut got = raptor_wings::segment::dispatch_strategies();
    assert_eq!(got, ["switch", "predicated", "segmented", "auto"]);
    got.sort();
    assert_eq!(got, ["auto", "predicated", "segmented", "switch"]); // the recorded tuple ("predicated", "switch", "segmented", "auto")
}
