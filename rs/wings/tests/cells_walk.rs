//! The canonical walk of a kernel's sinks with the companion, finished-count, fused-steps and branch-mix rules, and the typed reads of
//! the walk. Oracle: tests/data/walk.json, recorded from the previous Python `cells.walk` on the same sinks.
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::typed::{RxPair, RxPairPart, RxType};
use raptor_ir::value::{text_val, ttype_val, Val};
use raptor_ir::walkreads::{
    rx_walk_access_class, rx_walk_accum_companions, rx_walk_prior, rx_walk_selector_reads,
    rx_walk_slot_type, rx_walk_static_selector,
};
use raptor_wings::cells::{kernel_walk, rw_walk_companions};
use raptor_wings::declare::declare_kinds;
use raptor_wings::segment::{
    auto_dispatch, rw_group_plane_names, rw_walk_mix_slots, rw_walk_strategies, strategies,
    Strategy,
};
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
    let selector = leaf(c, ("plane_in", "vocab_read"), "k", "i32");
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
    let per = ("plane_in", "vocab_read");
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
        "finishing" | "finishing_declared" | "finishing_declared_mistyped" => {
            let dtype = if name.ends_with("mistyped") {
                "f64"
            } else {
                "uint32"
            };
            if name != "finishing" {
                declared = format!("(Slot(role='lookup', name='finished_count', ttype=TensorType(shape=(), dtype='{dtype}', tag=None)),)");
            }
            let done = c.make_op("gt", &[x, y], &Val::None).unwrap();
            vec![sink(c, "Assign", "o", x), sink(c, "Finish", "done", done)]
        }
        "auto_dispatch" | "predicated_dispatch" | "segmented_dispatch" | "switch_dispatch" => {
            let d = dispatch(c, (x, y), name.trim_end_matches("_dispatch"));
            vec![sink(c, "Assign", "o", d)]
        }
        "accum_compensated_and_plain" => {
            let write = |c: &mut Ctx, name: &str, value: u32, summation: &str| {
                let fields = format!("(('kind', 'update'), ('name', '{name}'), ('role', 'plane_out'), ('op', 'add'), ('summation', '{summation}'), ('reduced', None))");
                c.make("Update", "f64", &[], None, &[value], &fields)
                    .unwrap()
            };
            vec![
                write(c, "a", x, "compensated"),
                write(c, "b", y, "auto"),
                write(c, "a", y, "compensated"),
                write(c, "c", x, "compensated"),
            ]
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
            let p = leaf(c, ("plane_in", "prior_read"), "p", "f64");
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
    assert_eq!(data.arr().len(), 16);
    for case in data.arr() {
        let id = case.get("id").and_then(J::str).unwrap();
        let mut c = Ctx::new().unwrap();
        declare_kinds(&mut c).unwrap();
        // the recorded digests name the read planes by the rank of their value
        c.role_form = Some(raptor_support::migrate::old_role_name);
        c.key_form = Some(raptor_support::migrate::old_read_role);
        raptor_support::migrate::note_recorded_contract("c", "keyed_auto");
        let (sinks, declared) = case_sinks(&mut c, id);
        let declared = raptor_ir::canon::declared_slots(Some(&declared)).unwrap();
        let got = kernel_walk(&mut c, &sinks, &declared, None, &[]);
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

/// A pair of absent parts, to be written by a door.
fn empty_pair() -> RxPair {
    let part = || RxPairPart {
        kind: 0,
        number: 0,
        text: [0; 64],
    };
    RxPair {
        first: part(),
        second: part(),
    }
}

/// The text of a pair part, None for an absent one.
fn part_text(part: &RxPairPart) -> Option<String> {
    match part.kind {
        0 => None,
        2 => Some(
            unsafe { CStr::from_ptr(part.text.as_ptr()) }
                .to_str()
                .unwrap()
                .to_string(),
        ),
        kind => panic!("a part of kind {kind}"),
    }
}

/// The pairs of texts a pairs-protocol door writes, each part None when absent.
fn pairs_of(
    call: impl Fn(*mut RxPair, usize, *mut usize) -> i32,
) -> Vec<(Option<String>, Option<String>)> {
    let mut buf: Vec<RxPair> = (0..8).map(|_| empty_pair()).collect();
    let mut count = 0;
    assert_eq!(call(buf.as_mut_ptr(), buf.len(), &mut count), 0);
    buf[..count]
        .iter()
        .map(|p| (part_text(&p.first), part_text(&p.second)))
        .collect()
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
        raptor_support::migrate::note_recorded_contract("c", "keyed_auto");
        let (sinks, declared) = case_sinks(&mut c, id);
        let declared = raptor_ir::canon::declared_slots(Some(&declared)).unwrap();
        let w = kernel_walk(&mut c, &sinks, &declared, None, &[]).unwrap();
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
        let slots = case.get("slot_types").unwrap().arr();
        let binds = |slot: &str| slots.iter().any(|r| r.arr()[0].strs() == ["lookup", slot]);
        let mut mixes: Vec<RxPair> = (0..rows.len()).map(|_| empty_pair()).collect();
        let mut count = 0;
        let status =
            unsafe { rw_walk_mix_slots(ctx, w, mixes.as_mut_ptr(), mixes.len(), &mut count) };
        assert_eq!((status, count), (0, rows.len()), "{id}: mix slots");
        for (pair, row) in mixes.iter().zip(rows) {
            let group = row.arr()[0].str().unwrap();
            let mix = format!("{group}_mix");
            assert_eq!(part_text(&pair.first), Some(group.to_string()), "{id}");
            // the previous walk bound the mix slot of a group exactly when it had one
            assert_eq!(part_text(&pair.second), binds(&mix).then_some(mix), "{id}");
        }
        // what the previous walk answered: its companion pairs, the selector reads of each group and its access class
        let companions = pairs_of(|b, cap, n| unsafe { rw_walk_companions(ctx, w, b, cap, n) });
        let want: Vec<_> = case
            .get("companions")
            .unwrap()
            .arr()
            .iter()
            .map(|r| r.strs())
            .collect();
        assert_eq!(
            companions,
            want.iter()
                .map(|r| (Some(r[0].clone()), Some(r[1].clone())))
                .collect::<Vec<_>>(),
            "{id}: companions"
        );
        let accum = pairs_of(|b, cap, n| unsafe { rx_walk_accum_companions(ctx, w, b, cap, n) });
        let want: Vec<_> = case
            .get("accum_companions")
            .unwrap()
            .arr()
            .iter()
            .map(|r| r.strs())
            .collect();
        assert_eq!(
            accum,
            want.iter()
                .map(|r| (Some(r[0].clone()), Some(r[1].clone())))
                .collect::<Vec<_>>(),
            "{id}: accumulate companions"
        );
        // the segmented form of an automatic group is made of the walk's one automatic group, and of no other kernel
        let found = auto_dispatch(&c, w).unwrap();
        let autos = rows
            .iter()
            .filter(|r| r.arr()[3].str() == Some("auto"))
            .count();
        assert_eq!(
            found.is_some(),
            autos == 1 && rows.len() == 1,
            "{id}: automatic dispatch"
        );
        if let Some(split) = found {
            assert_eq!((split.index, split.branches), (0, 2), "{id}");
        }
        let reads = case.get("selector_reads").unwrap().arr();
        for (i, group) in reads.iter().enumerate() {
            let got = pairs_of(|b, cap, n| unsafe {
                rx_walk_selector_reads(ctx, w, i as i64, b, cap, n)
            });
            let want: Vec<_> = group.arr().iter().map(|r| r.strs()).collect();
            assert_eq!(
                got,
                want.iter()
                    .map(|r| (Some(r[0].clone()), Some(r[1].clone())))
                    .collect::<Vec<_>>(),
                "{id}: selector reads {i}"
            );
        }
        let (mut text, mut len) = ([0 as c_char; 32], 0);
        assert_eq!(
            unsafe { rx_walk_access_class(ctx, w, text.as_mut_ptr(), 32, &mut len) },
            0
        );
        let class = String::from_utf8(text[..len].iter().map(|&b| b as u8).collect()).unwrap();
        assert_eq!(
            Some(class.as_str()),
            case.get("access").unwrap().arr()[0].str(),
            "{id}: access"
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

/// The names the wings own are the previous package's: the special kinds (as a set), and the two reserved planes.
#[test]
fn the_wing_names_are_the_recorded_ones() {
    let data = json::parse(include_str!("data/wing_names.json")).unwrap();
    let mut special: Vec<String> = raptor_wings::special_kinds()
        .into_iter()
        .map(String::from)
        .collect();
    special.sort();
    assert_eq!(special, data.get("SPECIAL_KINDS").unwrap().strs());
    assert_eq!(
        [raptor_wings::cells::FUSED_STEPS_PLANE],
        data.get("FUSED_STEPS_PLANE").unwrap().strs().as_slice()
    );
    assert_eq!(
        [raptor_wings::cells::FINISHED_PLANE],
        data.get("FINISHED_PLANE").unwrap().strs().as_slice()
    );
}

/// The names a dispatch group owns are the previous wing's `offsets_name`, `class_name`, `stale_name` and `mix_name` of its name.
#[test]
fn the_plane_names_of_a_group_are_the_recorded_ones() {
    let mut c = Ctx::new().unwrap();
    let group = CString::new("dispatch3").unwrap();
    let ctx: *mut Ctx = &mut c;
    let names = names_of(|b, cap, len, count| unsafe {
        rw_group_plane_names(ctx, group.as_ptr(), b, cap, len, count)
    });
    assert_eq!(
        names,
        [
            "dispatch3_offsets",
            "dispatch3_class",
            "dispatch3_stale",
            "dispatch3_mix"
        ]
    );
}

/// A key line of a walk as it was written before a plane stated its axes and a read and a store stated their edge rule: the fields
/// those changes added are removed, and the extent of a flat plane is named as it was.
fn key_line_before_axes(line: &str) -> String {
    if line.contains("('accum_write'") || line.contains("('wide_write'") {
        if let Some(head) = line.strip_suffix(", 'unchecked', ()))") {
            return format!("{head}))");
        }
    }
    if line.contains("('assign'") {
        if let Some(head) = line.strip_suffix(", ()))") {
            return format!("{head}))");
        }
    }
    if line.contains("('at'") {
        if let Some(head) = line.strip_suffix(", ('None', 'clamp'))") {
            return format!("{head}, 'None')");
        }
    }
    if line.contains("('axis_extent'") {
        if let Some(head) = line.strip_suffix(", 0)") {
            return format!(
                "{}, None)",
                head.replace("('axis_extent'", "('plane_extent'")
            );
        }
    }
    line.to_string()
}

/// The digests recorded before the change are the walks of the same sinks with the added fields removed: the key lines of today's
/// walk, restated as they were written, hash with the walk's own tail to the digest the previous function gave. This is what lets
/// `walk.json` hold today's digests.
#[test]
fn the_digests_recorded_before_the_axes_are_the_walks_with_the_added_fields_removed() {
    let before = json::parse(include_str!("data/walk_digests_before_axes.json")).unwrap();
    let data = json::parse(include_str!("data/walk.json")).unwrap();
    let mut proved = 0;
    for case in data.arr() {
        let id = case.get("id").and_then(J::str).unwrap();
        let mut c = Ctx::new().unwrap();
        declare_kinds(&mut c).unwrap();
        raptor_support::migrate::note_recorded_contract("c", "keyed_auto");
        let (sinks, declared) = case_sinks(&mut c, id);
        let declared = raptor_ir::canon::declared_slots(Some(&declared)).unwrap();
        let Ok(w) = kernel_walk(&mut c, &sinks, &declared, None, &[]) else {
            continue; // a refused kernel has no digest
        };
        let walk = c.walk(w).unwrap();
        assert!(
            walk.quantities.is_empty(),
            "{id}: the proof states no quantity tail"
        );
        let field = |name: &str| walk.fields.iter().find(|f| f.0 == name).unwrap().1.clone();
        let (mut tail, mut old_tail) = (String::new(), String::new());
        for ((role, name), tt) in &walk.leaves {
            let ttype = text_val(&ttype_val(&tt.dtype, &tt.shape, tt.tag.as_deref()));
            let old = raptor_support::migrate::old_role_name(
                role,
                name,
                tt.shape.len(),
                walk.zeroed.contains(&(role.clone(), name.clone())),
            );
            tail.push_str(&format!("leaf|{role}|{name}|{ttype}\n"));
            old_tail.push_str(&format!("leaf|{old}|{name}|{ttype}\n"));
        }
        tail.push_str(&format!(
            "arg_spec|{}\naccess|{}|None\n",
            field("arg_spec"),
            walk.access
        ));
        // the recordings name the read planes by the rank of their value
        let old_arg_spec = field("arg_spec")
            .replace("'plane_in'", "'per_sample'")
            .replace("'plane_out'", "'accum_out'");
        old_tail.push_str(&format!(
            "arg_spec|{old_arg_spec}\naccess|{}|None\n",
            walk.access
        ));
        let pre = field("preimage");
        assert_eq!(
            rcore::digest(&pre, tail.as_bytes()),
            walk.digest,
            "{id}: the tail is the walk's own"
        );
        let old: String = pre
            .lines()
            .map(|l| key_line_before_axes(&raptor_support::migrate::old_read_role(l)) + "\n")
            .collect();
        let recorded = before
            .get(id)
            .and_then(J::str)
            .unwrap_or_else(|| panic!("{id}: no recorded digest"));
        assert_eq!(
            rcore::digest(&old, old_tail.as_bytes()),
            recorded,
            "{id}: the walk before the axes"
        );
        proved += 1;
    }
    assert!(proved >= 10, "{proved} walks proved");
}
