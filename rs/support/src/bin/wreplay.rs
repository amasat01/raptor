//! wreplay <texts.jsonl> [--target host] [--limit n] [--only s] [--show k]
//! Rebuilds each recorded unit in a fresh context through the C door, asks the wings every question the recorded render asked
//! (the functions this crate implements), and compares the answers with the recorded ones.
use raptor_ir::record::Ctx;
use raptor_support::corpus::{declare, err, make_nodes, manifests, tagged_str, text_out, val};
use raptor_support::corpus::{walk_options, with_analysis_nodes, Vals};
use raptor_support::safe::*;
// SAFETY (every `unsafe` call of an `rw_*` export below): `c` is a live context made in this tool, and
// every pointer passed is to a local buffer or value that outlives the call, sized as the call states.
use raptor_wings::boxing::*;
use raptor_wings::declare::rw_declare_kinds;
use raptor_wings::lanes::warp::rw_lanes_warp_uniform;
use raptor_wings::loops::*;
use raptor_wings::lstsq::*;
use raptor_wings::place::*;
use raptor_wings::split::{rw_split_compensated_target, CompensatedTarget};
use raptor_wings::wide::*;
use raptor_wings::*;
use raptor_wings::{KeyRunAnswer, PlaneRow};
use rcore::json::{self, J};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::{c_char, CStr, CString};
use std::io::{BufRead, BufReader};

/// Whether an accumulate under the contract commits through a companion plane (the recorded
/// function's rule).
fn compensates(contract: &str) -> bool {
    contract == "compensated"
        || raptor_ir::consts::constant("KEYED_AUTO").is_some_and(|k| k == format!("{contract:?}"))
}

/// The recorded functions this tool answers itself.
const MINE: [(&str, &str); 35] = [
    (
        "raptor.wings.base.lanes:warp_uniform()",
        "rw_lanes_warp_uniform",
    ),
    ("raptor.wings.base.wide:place", "rw_wide_place"),
    ("raptor.ir.nodes:through", "rx_through"),
    ("raptor.ir.nodes:same_index", "rx_same_index"),
    ("raptor.ir.nodes:compensates", "rx_compensates"),
    ("raptor.ir.nodes:is_param_cell", "Ctx::is_param_cell"),
    ("raptor.ir.nodes:has_feather", "Ctx::has_feather"),
    ("raptor.ir.nodes:sample_sums", "Ctx::sample_sums"),
    ("raptor.ir.kinds:is_a", "rx_is_a"),
    (
        "raptor.ir.walk:canonical_nodes",
        "rx_canonical + rx_walk_order + rx_walk_position",
    ),
    (
        "raptor.ir.walk:dispatch_groups",
        "rx_dispatch_nodes + rx_walk_field",
    ),
    ("raptor.ir.walk:substitute", "rx_substitute"),
    ("raptor.ir.cells:with_value", "Ctx::with_value"),
    ("raptor.ir.ops:make", "rx_make_op"),
    ("raptor.wings.base.lstsq:lowered", "rw_lstsq_lowered"),
    ("raptor.wings.base.box:boxed_sinks", "rw_box_boxed_sinks"),
    ("raptor.wings.base.wide:is_scratch", "rw_wide_is_scratch"),
    ("raptor.wings.base.wide:is_prior", "rw_wide_is_prior"),
    (
        "raptor.wings.base.wide:element_kind",
        "rw_wide_element_kind",
    ),
    ("raptor.wings.base.wide:gather", "rw_wide_gather"),
    ("raptor.wings.base.loops:flat", "rw_loops_flat"),
    ("raptor.ir.walk:companions", "rw_ir_companions"),
    ("raptor.ir.walk:param_planes", "rw_ir_param_plane"),
    (
        "raptor.wings.base.loops:speculative_step",
        "rw_loops_speculative_step",
    ),
    (
        "raptor.wings.base.split:compensated_target",
        "rw_split_compensated_target",
    ),
    ("raptor.wings.base.wide:record", "rw_wide_record"),
    ("raptor.ir.fold:rolled", "rw_fold_rolled"),
    (
        "raptor.wings.base.lanes:key_run",
        "rw_lanes_key_run_in_kernel",
    ),
    ("raptor.wings.base.wide:is_resident", "rw_wide_is_resident"),
    ("raptor.wings.base.wide:is_tiled", "rw_wide_is_tiled"),
    (
        "raptor.wings.cpu.vectorise:lane_form_access",
        "rw_cpu_lane_form_access",
    ),
    (
        "raptor.wings.cpu.vectorise:interchanged_loops",
        "rw_cpu_interchanged_loops",
    ),
    (
        "raptor.wings.base.cost:is_library_call",
        "rw_cost_is_library_call",
    ),
    ("raptor.ir.linalg:is_linalg", "rw_linalg_is_linalg"),
    (
        "raptor.wings.base.dispatch:switch_form",
        "rw_dispatch_switch_form",
    ),
];

/// The most trips of a loop the host compiler unrolls, as the writers state it.
const UNROLL_LIMIT: i64 = 16;

/// Why a recorded question was not answered equal: it differs, it is skipped, or it is equal.
enum Out {
    Equal,
    Diff(String),
    Skip(String),
}

#[derive(Default)]
/// Counters of one recorded function family.
struct Stat {
    eq: usize,
    calls: usize,
    rec_eq: usize,
    recs: usize,
    units: HashSet<usize>,
}

/// A skipped question, with the reason.
fn skip<T>(m: &str) -> Result<T, Out> {
    Err(Out::Skip(m.into()))
}

/// The nodes of the unit and what an analysis added to them, as one table.
fn joined(rec: &J, what: &str) -> Vec<J> {
    let (a, b) = (
        rec.get("inputs").and_then(|i| i.get(what)),
        rec.get("analysis_extra").and_then(|i| i.get(what)),
    );
    a.map_or(&[][..], |x| x.arr())
        .iter()
        .chain(b.map_or(&[][..], |x| x.arr()))
        .cloned()
        .collect()
}

/// The handle of the recorded node reference `a`, looked up in `h`.
fn node_ref(a: Option<&J>, h: &[u32]) -> Result<u32, Out> {
    match a.and_then(|a| a.get("n")) {
        Some(n) => h
            .get(n.int() as usize)
            .copied()
            .map_or_else(|| skip("node outside the node table"), Ok),
        None => skip("argument is not a node reference"),
    }
}

/// The value text of a recorded argument.
fn texts(v: &J, h: &[u32]) -> String {
    val(v, h)
}

/// The walk a recorded Walk record stands for: canonical over its sink nodes, with the options recorded for its digest, else its leaves as declared slots.
fn walk_of(
    c: *mut Ctx,
    w: &J,
    rec: &J,
    h: &[u32],
    cache: &mut HashMap<String, u32>,
) -> Result<u32, Out> {
    let v = w
        .get("v")
        .ok_or(Out::Skip("walk record has no fields".into()))?;
    let want = v
        .get("digest")
        .and_then(tagged_str)
        .unwrap_or("")
        .to_string();
    if let Some(&x) = cache.get(&want) {
        return Ok(x);
    }
    let order: Vec<u32> = match v.get("order").and_then(|o| o.get("t")) {
        Some(o) => o
            .arr()
            .iter()
            .filter_map(|n| n.get("n").and_then(|n| h.get(n.int() as usize).copied()))
            .collect(),
        None => return skip("walk has no order"),
    };
    let cx = unsafe { &*c };
    let mut sinks: Vec<u32> = order
        .iter()
        .copied()
        .filter(|&n| cx.table.is_a(cx.nodes[n as usize].class, "Sink"))
        .collect();
    if sinks.is_empty() {
        // a walk over bare values: its roots are the nodes no other node of the order reads
        let used: HashSet<u32> = order
            .iter()
            .flat_map(|&n| cx.nodes[n as usize].ops.iter().copied())
            .collect();
        sinks = order
            .iter()
            .copied()
            .filter(|n| !used.contains(n))
            .collect();
    }
    let mut sets: Vec<(String, Option<CString>, String)> = walk_options(rec, &want, h)
        .into_iter()
        .map(|o| {
            (
                o.declared,
                o.force.and_then(|f| CString::new(f).ok()),
                o.exempt,
            )
        })
        .collect();
    let recorded_options = !sets.is_empty();
    let leaves = v
        .get("leaves")
        .and_then(|l| l.get("t"))
        .cloned()
        .unwrap_or(J::Arr(vec![]));
    sets.push((
        texts(&J::Obj(vec![("l".into(), leaves)]), h),
        None,
        "[]".into(),
    ));
    let mut why = String::new();
    for (d, f, e) in sets {
        let (d, e) = (CString::new(d).unwrap(), CString::new(e).unwrap());
        let mut wh = 0u32;
        let st = rx_canonical(
            c,
            sinks.as_ptr(),
            sinks.len(),
            d.as_ptr(),
            f.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
            e.as_ptr(),
            &mut wh,
        );
        if st != 0 {
            why = err(c);
            continue;
        }
        let mut dg = [0 as c_char; 65];
        rx_digest(c, wh, dg.as_mut_ptr());
        let got = unsafe { CStr::from_ptr(dg.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if got == want {
            cache.insert(want, wh);
            return Ok(wh);
        }
        why = "digest differs".into();
    }
    let classes: Vec<String> = order
        .iter()
        .take(6)
        .map(|&n| cx.table.classes[cx.nodes[n as usize].class].name.clone())
        .collect();
    let basis = if recorded_options {
        "recorded options"
    } else {
        "no options recorded for the digest"
    };
    skip(&format!(
        "walk rebuild ({basis}): {} {classes:?}",
        why.chars().take(40).collect::<String>()
    ))
}

/// Reads a flag answer from a door call: a refusal status is a difference; the flag is compared with `want` when one is given.
fn flag_of(st: i32, out: i32, want: Option<bool>, what: &str) -> Result<bool, Out> {
    if st != 0 {
        return Err(Out::Diff(format!("{what} refused ({st})")));
    }
    let _ = want;
    Ok(out != 0)
}

/// Reads a list of handles from a door call that takes a buffer: asks for the length, then for the list.
fn list_of(mut f: impl FnMut(*mut u32, usize, *mut usize) -> i32) -> Result<Vec<u32>, Out> {
    let mut n = 0usize;
    let st = f(std::ptr::null_mut(), 0, &mut n);
    if st == 0 {
        return Ok(vec![]);
    }
    if st != 3 {
        return Err(Out::Diff(format!("refused ({st})")));
    }
    let mut b = vec![0u32; n];
    let st = f(b.as_mut_ptr(), n, &mut n);
    if st != 0 {
        return Err(Out::Diff(format!("refused ({st})")));
    }
    b.truncate(n);
    Ok(b)
}

/// The recorded result as a boolean; any other kind is skipped.
fn bool_want(r: &J) -> Result<bool, Out> {
    match r {
        J::Bool(b) => Ok(*b),
        _ => skip("recorded result is not a bool"),
    }
}

/// Checks the recorded call has `n` arguments, else the call is skipped.
fn walk_call(args: &[J], n: usize) -> Result<(), Out> {
    if args.len() == n {
        Ok(())
    } else {
        skip("unexpected argument count")
    }
}

/// Compares the computed answer with the recorded one; a difference names the arguments, the recorded and the computed text.
fn same(args: &[J], h: &[u32], ok: bool, got: String, want: String) -> Result<(), Out> {
    if ok {
        Ok(())
    } else {
        Err(Out::Diff(format!(
            "args {} | recorded {want} | computed {got}",
            args.iter()
                .map(|a| texts(a, h))
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

/// Answers the recorded call of the lane form access family function family: runs the matching library function and compares.
fn lane_form_access_family(
    c: *mut Ctx,
    fname: &str,
    args: &[J],
    recorded: Result<&J, &J>,
    h: &[u32],
    rec: &J,
    cache: &mut HashMap<String, u32>,
) -> Result<(), Out> {
    walk_call(args, 1)?;
    let Some(w) = args[0].get("r").and(Some(&args[0])) else {
        return skip("argument is not a walk");
    };
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let wh = walk_of(c, w, rec, h, cache)?;
    if fname.ends_with("lane_form_access") {
        let want = bool_want(r)?;
        let mut o = 0;
        let st = unsafe { rw_cpu_lane_form_access(c, wh, &mut o) };
        let g = flag_of(st, o, None, "lane_form_access")?;
        same(args, h, g == want, g.to_string(), want.to_string())
    } else {
        let items = match r.get("t").or(r.get("l")) {
            Some(i) => i.arr(),
            None => return skip("recorded result is not a sequence"),
        };
        let mut want = vec![];
        for x in items {
            want.push(node_ref(Some(x), h)?)
        }
        let got = list_of(|b, cap, n| unsafe {
            rw_cpu_interchanged_loops(c, wh, UNROLL_LIMIT, b, cap, n)
        })?;
        same(
            args,
            h,
            got == want,
            format!("{got:?}"),
            format!("{want:?}"),
        )
    }
}

/// Answers the recorded call of the is library call family function family: runs the matching library function and compares.
fn is_library_call_family(
    c: *mut Ctx,
    fname: &str,
    args: &[J],
    recorded: Result<&J, &J>,
    h: &[u32],
) -> Result<(), Out> {
    walk_call(args, 1)?;
    let n = node_ref(args.first(), h)?;
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let want = bool_want(r)?;
    let mut o = 0;
    let st = match fname {
        "raptor.wings.base.cost:is_library_call" => unsafe {
            rw_cost_is_library_call(c, n, &mut o)
        },
        "raptor.ir.linalg:is_linalg" => unsafe { rw_linalg_is_linalg(c, n, &mut o) },
        "raptor.wings.base.wide:is_resident" => unsafe { rw_wide_is_resident(c, n, &mut o) },
        "raptor.wings.base.wide:is_tiled" => unsafe { rw_wide_is_tiled(c, n, &mut o) },
        "raptor.wings.base.loops:speculative_step" => unsafe {
            rw_loops_speculative_step(c, n, &mut o)
        },
        _ => unsafe { rw_dispatch_switch_form(c, n, &mut o) },
    };
    let g = flag_of(st, o, None, fname)?;
    same(args, h, g == want, g.to_string(), want.to_string())
}

/// Answers the recorded call of the record function family: runs the matching library function and compares.
fn record(c: *mut Ctx, args: &[J], recorded: Result<&J, &J>, h: &[u32]) -> Result<(), Out> {
    walk_call(args, 1)?;
    let mut order = vec![];
    for x in args[0]
        .get("t")
        .or(args[0].get("l"))
        .map_or(&[][..], |t| t.arr())
    {
        order.push(node_ref(Some(x), h)?)
    }
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let got = match record_text(c, &order) {
        Ok(rows) => rows,
        Err(st) => format!("status {st}"),
    };
    let mut want = String::new();
    for p in r.get("d").map_or(&[][..], |d| d.arr()) {
        let (k, inner) = (
            p.arr()[0].get("s").and_then(J::str).unwrap_or("?"),
            p.arr()[1].get("d").map_or(&[][..], |d| d.arr()),
        );
        let g = |name: &str| {
            inner
                .iter()
                .find(|q| q.arr()[0].get("s").and_then(J::str) == Some(name))
                .map(|q| {
                    let v = &q.arr()[1];
                    v.get("s")
                        .and_then(J::str)
                        .or(v.get("i").and_then(J::str))
                        .unwrap_or("?")
                        .to_string()
                })
                .unwrap_or_default()
        };
        want.push_str(&format!(
            "{k} {} {} {}\n",
            g("form"),
            g("layout"),
            g("width")
        ));
    }
    same(args, h, got == want, got.clone(), want.clone())
}

/// Answers the recorded call of the rolled function family: runs the matching library function and compares.
fn rolled(c: *mut Ctx, args: &[J], recorded: Result<&J, &J>, h: &[u32]) -> Result<(), Out> {
    if args.len() < 2 || args.len() > 3 {
        return skip("unexpected argument count");
    }
    let n = node_ref(args.first(), h)?;
    let fm: Option<i64> = args[1]
        .get("i")
        .and_then(J::str)
        .and_then(|x| x.parse().ok());
    let form = CString::new(
        args.get(2)
            .and_then(|f| f.get("s"))
            .and_then(J::str)
            .unwrap_or("auto"),
    )
    .unwrap();
    let mut o = 0;
    let st = unsafe {
        rw_fold_rolled(
            c,
            n,
            fm.as_ref().map_or(std::ptr::null(), |f| f as *const i64),
            form.as_ptr(),
            &mut o,
        )
    };
    match recorded {
        Ok(r) => {
            let want = bool_want(r)?;
            if st != 0 {
                same(args, h, false, format!("status {st}"), want.to_string())
            } else {
                same(
                    args,
                    h,
                    (o != 0) == want,
                    (o != 0).to_string(),
                    want.to_string(),
                )
            }
        }
        Err(_) => same(args, h, st == 1, format!("status {st}"), "an error".into()),
    }
}

/// Answers the recorded call of the key run function family: runs the matching library function and compares.
fn key_run(
    c: *mut Ctx,
    args: &[J],
    recorded: Result<&J, &J>,
    h: &[u32],
    objs: &[J],
    target: &str,
) -> Result<(), Out> {
    if args.len() != 3 {
        return skip("unexpected argument count");
    }
    let mut sinks = vec![];
    for x in args[0]
        .get("t")
        .or(args[0].get("l"))
        .map_or(&[][..], |t| t.arr())
    {
        sinks.push(node_ref(Some(x), h)?)
    }
    let sink = node_ref(Some(&args[1]), h)?;
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let (mut run_max, mut block_run) = (None, None);
    if let Some(by) = args[2]
        .get("v")
        .and_then(|v| v.get("by_target"))
        .and_then(|b| b.get("d"))
    {
        let pick = by
            .arr()
            .iter()
            .find(|p| p.arr()[0].get("s").and_then(J::str) == Some(target))
            .or(by.arr().first());
        let lanes = pick
            .and_then(|p| p.arr()[1].get("of"))
            .and_then(|o| o.get("d"))
            .and_then(|d| {
                d.arr()
                    .iter()
                    .find(|q| q.arr()[0].get("s").and_then(J::str) == Some("lanes"))
            });
        for q in lanes
            .and_then(|l| l.arr()[1].get("of"))
            .and_then(|o| o.get("d"))
            .map_or(&[][..], |d| d.arr())
        {
            let (k, v) = (
                q.arr()[0].get("s").and_then(J::str),
                q.arr()[1]
                    .get("i")
                    .and_then(J::str)
                    .and_then(|x| x.parse::<i64>().ok()),
            );
            if k == Some("run_max") {
                run_max = v
            } else if k == Some("block_run") {
                block_run = v
            }
        }
    }
    let (Some(rm), Some(br)) = (run_max, block_run) else {
        return skip("the spec names no lane numbers for the target");
    };
    let got = match key_run_text(c, &sinks, sink, rm, br) {
        Ok(text) => text,
        Err(st) => return same(args, h, false, format!("status {st}"), texts(r, h)),
    };
    let mut want = match r.get("v") {
        None => "none".to_string(),
        Some(v) => format!(
            "{} {} {}",
            v.get("form").and_then(tagged_str).unwrap_or("?"),
            v.get("width")
                .and_then(|w| w.get("i"))
                .and_then(J::str)
                .unwrap_or("?"),
            matches!(v.get("block"), Some(J::Bool(true))) as u8
        ),
    };
    if got == want && r.get("v").is_some() {
        // the per-axis forms: the class of each axis and its coefficient
        let cx = unsafe { &*c };
        let k = raptor_wings::lanes::key_run(cx, &sinks, sink, rm as i128, br as i128)
            .ok()
            .flatten();
        let rec: Vec<(String, String)> = r
            .get("v")
            .and_then(|v| v.get("forms"))
            .and_then(|f| f.get("t"))
            .map_or(&[][..], |f| f.arr())
            .iter()
            .map(|f| {
                let v = f.get("v");
                let cls = v
                    .and_then(|v| v.get("cls"))
                    .and_then(tagged_str)
                    .unwrap_or("?")
                    .to_string();
                let co = match v.and_then(|v| v.get("coefficient")) {
                    Some(J::Null) | None => "-".to_string(),
                    Some(o) => o
                        .get("o")
                        .and_then(|k| objs.get(k.int() as usize))
                        .and_then(|x| x.get("a"))
                        .map(|a| {
                            let g = |n: &str| {
                                a.get(n)
                                    .and_then(|x| x.get("i"))
                                    .and_then(J::str)
                                    .unwrap_or("?")
                                    .to_string()
                            };
                            format!("{}/{}", g("_numerator"), g("_denominator"))
                        })
                        .unwrap_or("?".into()),
                };
                (cls, co)
            })
            .collect();
        let mine: Vec<(String, String)> = k.map_or(vec![], |k| {
            k.forms
                .iter()
                .map(|f| {
                    (
                        f.0.to_string(),
                        f.1.map_or("-".to_string(), |c| format!("{}/{}", c.n, c.d)),
                    )
                })
                .collect()
        });
        if mine != rec {
            want = format!("{want} forms {rec:?}");
            return same(args, h, false, format!("{got} forms {mine:?}"), want);
        }
    }
    same(args, h, got == want, got, want)
}

/// Answers the recorded call of the compensated target function family: runs the matching library function and compares.
fn compensated_target(
    c: *mut Ctx,
    args: &[J],
    recorded: Result<&J, &J>,
    h: &[u32],
) -> Result<(), Out> {
    if args.len() != 2 {
        return skip("unexpected argument count");
    }
    let mut sinks = vec![];
    for x in args[1]
        .get("t")
        .or(args[1].get("l"))
        .map_or(&[][..], |t| t.arr())
    {
        sinks.push(node_ref(Some(x), h)?)
    }
    let Some(kind) = args[0].get("v") else {
        return skip("argument is not a kind record");
    };
    let text = |j: Option<&J>| j.and_then(tagged_str).and_then(|t| CString::new(t).ok());
    let (policy, output) = (
        kind.get("sink").and_then(|s| s.get("v")),
        kind.get("output").and_then(|o| o.get("v")),
    );
    let sink_id = text(policy.and_then(|p| p.get("id"))).unwrap_or_default();
    let compensation = text(policy.and_then(|p| p.get("compensation")));
    let returned = output
        .and_then(|o| o.get("returned_ttype"))
        .is_some_and(|t| !matches!(t, J::Null));
    let slot = if returned {
        text(output.and_then(|o| o.get("slot")))
    } else {
        None
    };
    let names: Vec<&str> = output
        .and_then(|o| o.get("names"))
        .and_then(|n| n.get("t"))
        .map_or(&[][..], |n| n.arr())
        .iter()
        .filter_map(tagged_str)
        .collect();
    let names = CString::new(names.join("\n")).unwrap();
    let (mut len, mut buf) = (0usize, vec![0 as c_char; 256]);
    let ptr = |s: &Option<CString>| s.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let mut target = CompensatedTarget {
        found: 0,
        own_column: 0,
    };
    let st = unsafe {
        {
            rw_split_compensated_target(
                c,
                sink_id.as_ptr(),
                ptr(&compensation),
                ptr(&slot),
                names.as_ptr(),
                sinks.as_ptr(),
                sinks.len(),
                &mut target,
                buf.as_mut_ptr(),
                buf.len(),
                &mut len,
            )
        }
    };
    let got = if st != 0 {
        format!("status {st}")
    } else if target.found == 0 {
        "none".to_string()
    } else {
        let name = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy();
        format!("{name} {}", target.own_column)
    };
    match recorded {
        Err(_) => same(args, h, st == 1, got, "an error".into()),
        Ok(r) => {
            let want = match r.get("t") {
                Some(t) => format!(
                    "{} {}",
                    t.arr()[0].get("s").and_then(J::str).unwrap_or("?"),
                    matches!(t.arr()[1], J::Bool(true)) as u8
                ),
                None => "none".into(),
            };
            same(args, h, got == want, got.clone(), want.clone())
        }
    }
}

/// Answers the recorded call of the is scratch family function family: runs the matching library function and compares.
fn is_scratch_family(
    c: *mut Ctx,
    fname: &str,
    args: &[J],
    recorded: Result<&J, &J>,
    h: &[u32],
) -> Result<(), Out> {
    walk_call(args, 1)?;
    let n = node_ref(args.first(), h)?;
    let want = bool_want(match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    })?;
    let mut o = 0;
    let st = match fname {
        "raptor.wings.base.wide:is_scratch" => unsafe { rw_wide_is_scratch(c, n, &mut o) },
        "raptor.wings.base.wide:is_prior" => unsafe { rw_wide_is_prior(c, n, &mut o) },
        _ => return skip("not implemented here"),
    };
    let g = flag_of(st, o, None, fname)?;
    same(args, h, g == want, g.to_string(), want.to_string())
}

/// Answers the recorded call of the element kind function family: runs the matching library function and compares.
fn element_kind(c: *mut Ctx, args: &[J], recorded: Result<&J, &J>, h: &[u32]) -> Result<(), Out> {
    walk_call(args, 1)?;
    let n = node_ref(args.first(), h)?;
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let want = r.get("s").and_then(J::str).unwrap_or("none").to_string();
    let got = match element_kind_word(c, n) {
        Ok(word) => word,
        Err(st) => format!("status {st}"),
    };
    same(args, h, got == want, got.clone(), want.clone())
}

/// Answers the recorded call of the gather function family: runs the matching library function and compares.
fn gather(c: *mut Ctx, args: &[J], recorded: Result<&J, &J>, h: &[u32]) -> Result<(), Out> {
    walk_call(args, 1)?;
    let n = node_ref(args.first(), h)?;
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let (mut found, mut plane, mut base, mut start, mut step, mut clamped) =
        (0, 0u32, 0u32, 0i64, 0i64, 0);
    let st = unsafe {
        rw_wide_gather(
            c,
            n,
            &mut found,
            &mut plane,
            &mut base,
            &mut start,
            &mut step,
            &mut clamped,
        )
    };
    if st != 0 {
        return same(args, h, false, format!("status {st}"), texts(r, h));
    }
    let got = if found == 0 {
        "none".to_string()
    } else {
        format!(
            "plane {plane} base {} start {start} step {step} clamped {clamped}",
            if base == u32::MAX {
                "-".into()
            } else {
                base.to_string()
            }
        )
    };
    let want = match r.get("v") {
        None => "none".to_string(),
        Some(v) => {
            let num = |k: &str| {
                v.get(k)
                    .and_then(|x| x.get("i"))
                    .and_then(J::str)
                    .unwrap_or("?")
                    .to_string()
            };
            let node = |k: &str| {
                v.get(k)
                    .and_then(|x| x.get("n"))
                    .and_then(|x| h.get(x.int() as usize))
                    .map_or("-".to_string(), |x| x.to_string())
            };
            format!(
                "plane {} base {} start {} step {} clamped {}",
                node("plane"),
                node("base"),
                num("start"),
                num("step"),
                matches!(v.get("clamped"), Some(J::Bool(true))) as i32
            )
        }
    };
    same(args, h, got == want, got.clone(), want.clone())
}

/// Answers the recorded call of the flat function family: runs the matching library function and compares.
fn flat(c: *mut Ctx, args: &[J], recorded: Result<&J, &J>, h: &[u32]) -> Result<(), Out> {
    walk_call(args, 1)?;
    let mut given = vec![];
    for x in args[0]
        .get("t")
        .or(args[0].get("l"))
        .map_or(&[][..], |t| t.arr())
    {
        given.push(node_ref(Some(x), h)?)
    }
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let mut want = vec![];
    for x in r.get("l").or(r.get("t")).map_or(&[][..], |t| t.arr()) {
        want.push(node_ref(Some(x), h)?)
    }
    let got =
        list_of(|b, cap, k| unsafe { rw_loops_flat(c, given.as_ptr(), given.len(), b, cap, k) })?;
    same(
        args,
        h,
        got == want,
        format!("{got:?}"),
        format!("{want:?}"),
    )
}

/// Answers the recorded call of the boxed sinks function family: runs the matching library function and compares.
fn boxed_sinks(c: *mut Ctx, args: &[J], recorded: Result<&J, &J>, h: &[u32]) -> Result<(), Out> {
    walk_call(args, 2)?;
    let mut sinks = vec![];
    for x in args[0]
        .get("t")
        .or(args[0].get("l"))
        .or(args[0].get("of").and_then(|o| o.get("l")))
        .map_or(&[][..], |t| t.arr())
    {
        sinks.push(node_ref(Some(x), h)?)
    }
    let Some(spec) = args[1].get("v") else {
        return skip("argument is not a box record");
    };
    let num = |k: &str| {
        spec.get(k)
            .and_then(|x| x.get("i"))
            .and_then(J::str)
            .and_then(|x| x.parse::<i64>().ok())
    };
    let (Some(axis), Some(lanes), Some(stride)) = (num("axis"), num("T"), num("stride")) else {
        return skip("the box record lacks axis, T or stride");
    };
    let r = match recorded {
        Ok(r) => r,
        Err(_) => return skip("recorded error"),
    };
    let mut want = vec![];
    for x in r.get("t").or(r.get("l")).map_or(&[][..], |t| t.arr()) {
        want.push(node_ref(Some(x), h)?)
    }
    let mut made = vec![0u32; sinks.len() * lanes as usize];
    let mut count = 0usize;
    let st = unsafe {
        rw_box_boxed_sinks(
            c,
            sinks.as_ptr(),
            sinks.len(),
            axis as usize,
            lanes as usize,
            stride,
            made.as_mut_ptr(),
            made.len(),
            &mut count,
        )
    };
    if st != 0 {
        return if st == 1 {
            skip(&format!(
                "refused by the wing: {}",
                err(c).chars().take(60).collect::<String>()
            ))
        } else {
            same(args, h, false, format!("status {st}"), "a list".into())
        };
    }
    let digest = |nodes: &[u32]| -> String {
        let mut walk = 0u32;
        if rx_canonical(
            c,
            nodes.as_ptr(),
            nodes.len(),
            c"()".as_ptr(),
            std::ptr::null(),
            c"()".as_ptr(),
            &mut walk,
        ) != 0
        {
            return format!("canonical refused: {}", err(c));
        }
        let mut dg = [0 as c_char; 65];
        rx_digest(c, walk, dg.as_mut_ptr());
        unsafe { CStr::from_ptr(dg.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    let (mine, recorded_digest) = (digest(&made[..count]), digest(&want));
    if recorded_digest.starts_with("canonical refused") {
        return skip("the recorded answer has no canonical walk");
    }
    same(
        args,
        h,
        mine == recorded_digest,
        mine.clone(),
        recorded_digest.clone(),
    )
}

/// Answers one recorded call: picks the function family from its name and compares the computed answer with the recorded one.
fn question(
    c: *mut Ctx,
    call: &J,
    h: &[u32],
    rec: &J,
    cache: &mut HashMap<String, u32>,
    objs: &[J],
    target: &str,
) -> Result<(), Out> {
    let fname = call.get("fn").and_then(J::str).unwrap_or("");
    let args = call
        .get("args")
        .and_then(|a| a.get("l"))
        .map_or(&[][..], |a| a.arr());
    if !fname.ends_with("refuse_sample_sums")
        && !fname.ends_with("warp_uniform()")
        && !call
            .get("kwargs")
            .and_then(|k| k.get("d"))
            .is_none_or(|d| d.arr().is_empty())
    {
        return skip("keyword arguments");
    }
    let recorded = if let Some(e) = call.get("error") {
        Err(e)
    } else {
        match call.get("result") {
            Some(r) => Ok(r),
            None => return skip("the record lacks the answer"),
        }
    };
    match fname {
        "raptor.wings.cpu.vectorise:lane_form_access"
        | "raptor.wings.cpu.vectorise:interchanged_loops" => {
            lane_form_access_family(c, fname, args, recorded, h, rec, cache)
        }
        "raptor.wings.base.cost:is_library_call"
        | "raptor.ir.linalg:is_linalg"
        | "raptor.wings.base.dispatch:switch_form"
        | "raptor.wings.base.wide:is_resident"
        | "raptor.wings.base.wide:is_tiled"
        | "raptor.wings.base.loops:speculative_step" => {
            is_library_call_family(c, fname, args, recorded, h)
        }
        "raptor.wings.base.wide:record" => record(c, args, recorded, h),
        "raptor.ir.fold:rolled" => rolled(c, args, recorded, h),
        "raptor.wings.base.lanes:key_run" => key_run(c, args, recorded, h, objs, target),
        "raptor.wings.base.split:compensated_target" => compensated_target(c, args, recorded, h),
        "raptor.wings.base.wide:is_scratch" | "raptor.wings.base.wide:is_prior" => {
            is_scratch_family(c, fname, args, recorded, h)
        }
        "raptor.wings.base.wide:element_kind" => element_kind(c, args, recorded, h),
        "raptor.wings.base.wide:gather" => gather(c, args, recorded, h),
        "raptor.wings.base.loops:flat" => flat(c, args, recorded, h),
        "raptor.wings.base.box:boxed_sinks" => boxed_sinks(c, args, recorded, h),
        _ if fname.starts_with("raptor.ir.") => {
            core_callee(c, fname, args, recorded, h, rec, cache)
        }
        "raptor.wings.base.lanes:warp_uniform()" => Err(Out::Equal),
        _ => skip("not implemented here"),
    }
}

/// The canonical digest over some nodes with the default options, or why there is none.
fn digest_of(c: *mut Ctx, nodes: &[u32]) -> Result<String, String> {
    let mut walk = 0u32;
    if rx_canonical(
        c,
        nodes.as_ptr(),
        nodes.len(),
        c"()".as_ptr(),
        std::ptr::null(),
        c"()".as_ptr(),
        &mut walk,
    ) != 0
    {
        return Err(err(c));
    }
    let mut digest = [0 as c_char; 65];
    rx_digest(c, walk, digest.as_mut_ptr());
    Ok(unsafe { CStr::from_ptr(digest.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

/// Whether two nodes are the same value: one handle, or equal canonical digests.
fn same_node(c: *mut Ctx, mine: u32, recorded: u32) -> Result<bool, Out> {
    if mine == recorded {
        return Ok(true);
    }
    match (digest_of(c, &[mine]), digest_of(c, &[recorded])) {
        (Ok(a), Ok(b)) => Ok(a == b),
        _ => skip("no canonical walk over the node"),
    }
}

/// The nodes a recorded sequence names, in order.
fn node_list(seq: &J, h: &[u32]) -> Result<Vec<u32>, Out> {
    let held = seq
        .get("t")
        .or(seq.get("l"))
        .or(seq.get("of").and_then(|o| o.get("l").or(o.get("t"))));
    let items = held.map_or(&[][..], |t| t.arr());
    items.iter().map(|x| node_ref(Some(x), h)).collect()
}

/// A text argument the recorded call passed.
fn text_arg(arg: Option<&J>) -> Result<CString, Out> {
    match arg
        .and_then(|a| a.get("s"))
        .and_then(J::str)
        .and_then(|t| CString::new(t).ok())
    {
        Some(t) => Ok(t),
        None => skip("argument is not a string"),
    }
}

/// The `raptor.ir` functions the writers call, answered by the core's door and compared with
/// the record.
fn core_callee(
    c: *mut Ctx,
    fname: &str,
    args: &[J],
    recorded: Result<&J, &J>,
    h: &[u32],
    rec: &J,
    cache: &mut HashMap<String, u32>,
) -> Result<(), Out> {
    let Ok(answer) = recorded else {
        return skip("recorded error");
    };
    let differs = |what: String| Err(Out::Diff(format!("{what} | recorded {}", texts(answer, h))));
    let flag = |status: i32, out: i32| {
        if status == 0 {
            Ok(out != 0)
        } else {
            Err(Out::Diff(format!("status {status}")))
        }
    };
    let node = |k: usize| node_ref(args.get(k), h);
    let want_flag = || match answer {
        J::Bool(b) => Ok(*b),
        _ => skip("recorded result is not a bool"),
    };
    let mut out = 0i32;
    let got = match fname.rsplit(':').next().unwrap_or("") {
        "through" => {
            let (n, want, mut made) = (node(0)?, node_ref(Some(answer), h)?, 0u32);
            let status = rx_through(c, n, &mut made);
            return if status == 0 && made == want {
                Ok(())
            } else {
                differs(format!("through {n} gave {made} (status {status})"))
            };
        }
        "same_index" if args.iter().any(|a| matches!(a, J::Null)) => {
            // None is no node: it equals only None
            args.iter().all(|a| matches!(a, J::Null))
        }
        "same_index" => flag(rx_same_index(c, node(0)?, node(1)?, &mut out), out)?,
        "compensates" => compensates(text_arg(args.first())?.to_str().unwrap_or("")),
        "is_param_cell" => core_of(c).is_param_cell(node(0)?).map_err(refused)?,
        "has_feather" => {
            let name = text_arg(args.get(1))?;
            core_of(c)
                .has_feather(node(0)?, name.to_str().unwrap_or(""))
                .map_err(refused)?
        }
        "is_a" => flag(
            rx_is_a(c, node(0)?, text_arg(args.get(1))?.as_ptr(), &mut out),
            out,
        )?,
        "sample_sums" => return check_sample_sums(c, args, answer, h),
        "canonical_nodes" => return check_canonical_nodes(c, args, answer, h),
        "dispatch_groups" => return check_dispatch_groups(c, args, answer, h, rec, cache),
        "substitute" => return check_substitute(c, args, answer, h),
        "with_value" => {
            let made = unsafe { &mut *c }
                .with_value(node(0)?, node(1)?)
                .map_err(refused)?;
            return if same_node(c, made, node_ref(Some(answer), h)?)? {
                Ok(())
            } else {
                differs("the sink over the new value differs".to_string())
            };
        }
        "make" => return check_make(c, args, answer, h),
        _ => return skip("not implemented here"),
    };
    let want = want_flag()?;
    if got == want {
        Ok(())
    } else {
        differs(format!("computed {got}"))
    }
}

/// The context behind a handle the replay owns.
fn core_of<'a>(c: *mut Ctx) -> &'a Ctx {
    unsafe { &*c }
}

/// A refusal of a core method as a difference.
fn refused(fault: raptor_ir::record::Fault) -> Out {
    Out::Diff(format!("refused: {fault:?}"))
}

/// `sample_sums`: the sample-sum sinks among the given ones.
fn check_sample_sums(c: *mut Ctx, args: &[J], answer: &J, h: &[u32]) -> Result<(), Out> {
    let sinks = node_list(args.first().ok_or(Out::Skip("no argument".into()))?, h)?;
    let got = core_of(c).sample_sums(&sinks).map_err(refused)?;
    let want = node_list(answer, h)?;
    if got == want {
        Ok(())
    } else {
        Err(Out::Diff(format!(
            "sample sums {got:?} | recorded {want:?}"
        )))
    }
}

/// `canonical_nodes`: the walk's order and the position of every node the record maps.
fn check_canonical_nodes(c: *mut Ctx, args: &[J], answer: &J, h: &[u32]) -> Result<(), Out> {
    let sinks = node_list(args.first().ok_or(Out::Skip("no argument".into()))?, h)?;
    let Some(pair) = answer.get("t").map(|t| t.arr()).filter(|t| t.len() == 2) else {
        return skip("result is not (order, map)");
    };
    let mut walk = 0u32;
    let (declared, exempt) = (c"()".as_ptr(), c"()".as_ptr());
    if rx_canonical(
        c,
        sinks.as_ptr(),
        sinks.len(),
        declared,
        std::ptr::null(),
        exempt,
        &mut walk,
    ) != 0
    {
        return skip("no canonical walk with the default options");
    }
    let order = list_of(|b, cap, n| rx_walk_order(c, walk, b, cap, n))?;
    let want = node_list(&pair[0], h)?;
    if order != want {
        return Err(Out::Diff(format!("order {order:?} | recorded {want:?}")));
    }
    for entry in pair[1].get("idmap").map_or(&[][..], |m| m.arr()) {
        let (key, position) = (
            entry.arr()[0].int() as usize,
            entry.arr()[1]
                .get("i")
                .and_then(J::str)
                .and_then(|p| p.parse::<u32>().ok()),
        );
        let mut got = 0u32;
        let status = rx_walk_position(c, walk, h[key], &mut got);
        if status != 0 || Some(got) != position {
            return Err(Out::Diff(format!(
                "position of node {key}: {got} (status {status}) | recorded {position:?}"
            )));
        }
    }
    Ok(())
}

/// `dispatch_groups`: the dispatch nodes of a recorded walk, named dispatch0, dispatch1, ... in
/// order.
fn check_dispatch_groups(
    c: *mut Ctx,
    args: &[J],
    answer: &J,
    h: &[u32],
    rec: &J,
    cache: &mut HashMap<String, u32>,
) -> Result<(), Out> {
    let Some(walk) = args.first().filter(|w| w.get("r").is_some()) else {
        return skip("argument is not a walk");
    };
    let walk = walk_of(c, walk, rec, h, cache)?;
    let got = list_of(|b, cap, n| rx_dispatch_nodes(c, walk, b, cap, n))?;
    let name = CString::new("dispatches").unwrap();
    let listed = text_out(c, |c, b, cap, l| {
        rx_walk_field(c, walk, name.as_ptr(), b, cap, l)
    });
    let mut want = vec![];
    for (i, pair) in answer
        .get("d")
        .map_or(&[][..], |d| d.arr())
        .iter()
        .enumerate()
    {
        if pair.arr()[0].get("s").and_then(J::str) != Some(&format!("dispatch{i}")) {
            return skip("a group is not named by its position");
        }
        want.push(node_ref(Some(&pair.arr()[1]), h)?);
    }
    let named = listed.matches("'dispatch").count();
    if got == want && named == want.len() {
        Ok(())
    } else {
        Err(Out::Diff(format!(
            "dispatch nodes {got:?} ({named} listed) | recorded {want:?}"
        )))
    }
}

/// `substitute`: the roots over the mapping; a node the call made is compared by canonical digest.
fn check_substitute(c: *mut Ctx, args: &[J], answer: &J, h: &[u32]) -> Result<(), Out> {
    if args.len() != 2 {
        return skip("unexpected argument count");
    }
    let roots = node_list(&args[0], h)?;
    let (mut from, mut to) = (vec![], vec![]);
    for entry in args[1].get("idmap").map_or(&[][..], |m| m.arr()) {
        from.push(
            *h.get(entry.arr()[0].int() as usize)
                .ok_or(Out::Skip("key outside the node table".into()))?,
        );
        to.push(node_ref(Some(&entry.arr()[1]), h)?);
    }
    let got = list_of(|b, cap, n| {
        rx_substitute(
            c,
            roots.as_ptr(),
            roots.len(),
            from.as_ptr(),
            to.as_ptr(),
            from.len(),
            b,
            cap,
            n,
        )
    })?;
    let want = node_list(answer, h)?;
    if got.len() != want.len() {
        return Err(Out::Diff(format!(
            "{} roots | recorded {}",
            got.len(),
            want.len()
        )));
    }
    for (mine, recorded) in got.into_iter().zip(want) {
        if !same_node(c, mine, recorded)? {
            return Err(Out::Diff(format!(
                "root {mine} differs from recorded {recorded}"
            )));
        }
    }
    Ok(())
}

/// `ops.make`: an Op of a kind over operands, with its inferred type.
fn check_make(c: *mut Ctx, args: &[J], answer: &J, h: &[u32]) -> Result<(), Out> {
    let Some(kind) = args
        .first()
        .and_then(|a| a.get("s"))
        .and_then(J::str)
        .and_then(|k| CString::new(k).ok())
    else {
        return skip("kind is not a string");
    };
    let operands = node_list(args.get(1).ok_or(Out::Skip("no operands".into()))?, h)?;
    let literal = match args.get(2) {
        None | Some(J::Null) => String::new(),
        Some(l) => val(l, h),
    };
    let mut made = 0u32;
    let status = rx_make_op(
        c,
        kind.as_ptr(),
        operands.as_ptr(),
        operands.len(),
        literal.as_ptr() as *const c_char,
        literal.len(),
        &mut made,
    );
    if status != 0 {
        return Err(Out::Diff(format!("status {status}: {}", err(c))));
    }
    let want = node_ref(Some(answer), h)?;
    let part = |node: u32, what: &str| match what {
        "ttype" => text_out(c, |c, b, cap, l| rx_ttype(c, node, b, cap, l)),
        "operands" => format!(
            "{:?}",
            list_of(|b, cap, n| rx_operands(c, node, b, cap, n)).unwrap_or_default()
        ),
        field => {
            let f = CString::new(field).unwrap();
            text_out(c, |c, b, cap, l| rx_field(c, node, f.as_ptr(), b, cap, l))
        }
    };
    let same = ["ttype", "operands", "kind", "literal"]
        .iter()
        .all(|what| part(made, what) == part(want, what));
    if same {
        Ok(())
    } else {
        Err(Out::Diff("the op differs from the recorded one".into()))
    }
}
/// The recorded `companions` and `param_planes` of each walk the unit made, against the wings' answers.
fn check_walk_props<'a>(
    c: *mut Ctx,
    rec: &J,
    cache: &HashMap<String, u32>,
    stats: &mut BTreeMap<&'a str, Stat>,
    skips: &mut BTreeMap<(String, String), usize>,
    unit: usize,
) {
    let cx = unsafe { &*c };
    let mut digests: Vec<&String> = cache.keys().collect();
    digests.sort();
    for digest in digests {
        let walk = cache[digest];
        let Some(props) = rec.get("walk_props").and_then(|p| p.get(digest)) else {
            continue;
        };
        let record = |name: &str, ok: Result<(), String>, stats: &mut BTreeMap<&'a str, Stat>| {
            let key = MINE.iter().find(|m| m.0 == name).unwrap().0;
            let s = stats.entry(key).or_default();
            s.calls += 1;
            s.recs += 1;
            s.units.insert(unit);
            match ok {
                Ok(()) => {
                    s.eq += 1;
                    s.rec_eq += 1
                }
                Err(d) => println!("DIFF walk {digest} | {name} | {d}"),
            }
        };
        match props
            .get("companions")
            .and_then(|p| p.get("result"))
            .and_then(|r| r.get("d"))
        {
            Some(d) => {
                let want: String = d
                    .arr()
                    .iter()
                    .map(|p| {
                        format!(
                            "{}={}\n",
                            p.arr()[0].get("s").and_then(J::str).unwrap_or("?"),
                            p.arr()[1].get("s").and_then(J::str).unwrap_or("?")
                        )
                    })
                    .collect();
                let got = match companions_text(c, walk) {
                    Ok(lines) => lines,
                    Err(st) => format!("status {st}"),
                };
                record(
                    "raptor.ir.walk:companions",
                    if got == want {
                        Ok(())
                    } else {
                        Err(format!("recorded {want:?} computed {got:?}"))
                    },
                    stats,
                );
            }
            None => {
                *skips
                    .entry((
                        "raptor.ir.walk:companions".into(),
                        "no recorded companions".into(),
                    ))
                    .or_default() += 1
            }
        }
        match props
            .get("param_planes")
            .and_then(|p| p.get("result"))
            .and_then(|r| r.get("t"))
        {
            Some(t) => {
                let want: Vec<&str> = t
                    .arr()
                    .iter()
                    .filter_map(|x| x.get("s").and_then(J::str))
                    .collect();
                let order = cx.walk(walk).map(|w| w.order.clone()).unwrap_or_default();
                let mut names: Vec<String> = want.iter().map(|s| s.to_string()).collect();
                for n in order {
                    if cx.table.is_a(cx.nodes[n as usize].class, "Leaf")
                        && cx.str_field(n, "role").is_ok_and(|r| r == "lookup")
                    {
                        names.push(cx.str_field(n, "name").unwrap_or_default())
                    }
                }
                names.sort();
                names.dedup();
                let mut bad = vec![];
                for name in names {
                    let cname = CString::new(name.as_str()).unwrap();
                    let mut o = 0;
                    let st = unsafe { rw_ir_param_plane(c, walk, cname.as_ptr(), &mut o) };
                    if st != 0 || (o != 0) != want.contains(&name.as_str()) {
                        bad.push(name)
                    }
                }
                record(
                    "raptor.ir.walk:param_planes",
                    if bad.is_empty() {
                        Ok(())
                    } else {
                        Err(format!("planes {bad:?} (recorded {want:?})"))
                    },
                    stats,
                );
            }
            None => {
                *skips
                    .entry((
                        "raptor.ir.walk:param_planes".into(),
                        "no recorded param_planes".into(),
                    ))
                    .or_default() += 1
            }
        }
    }
}

/// The `linalg.max_dim` the unit's build spec states for its target, if any.
fn max_dim_of(j: &J, target: &str) -> Option<i64> {
    let pairs = |d: Option<&J>| {
        d.and_then(|d| d.get("d"))
            .map_or(&[][..], |d| d.arr())
            .to_vec()
    };
    let key =
        |p: &J, name: &str| p.arr().first().and_then(|k| k.get("s")).and_then(J::str) == Some(name);
    if j.get("r").and_then(J::str) == Some("BuildSpec") {
        let by_target = pairs(j.get("v").and_then(|v| v.get("by_target")));
        let per_target = by_target
            .iter()
            .find(|p| key(p, target))?
            .arr()
            .get(1)?
            .clone();
        let linalg = pairs(per_target.get("of"))
            .into_iter()
            .find(|p| key(p, "linalg"))?;
        let held = pairs(linalg.arr().get(1)?.get("of"))
            .into_iter()
            .find(|p| key(p, "max_dim"))?;
        return held
            .arr()
            .get(1)?
            .get("i")
            .and_then(J::str)
            .and_then(|x| x.parse().ok());
    }
    match j {
        J::Obj(fields) => fields.iter().find_map(|(_, v)| max_dim_of(v, target)),
        J::Arr(items) => items.iter().find_map(|v| max_dim_of(v, target)),
        _ => None,
    }
}

/// The units whose table holds least-squares nodes: the lowering of the sinks that reach them, against the
/// sinks the recorded render made (its new `Assign` nodes), by the digest of the canonical walk over each.
fn check_lowered(
    c: *mut Ctx,
    rec: &J,
    h: &[u32],
    stats: &mut BTreeMap<&str, Stat>,
    skips: &mut BTreeMap<(String, String), usize>,
    unit: usize,
    name: &str,
) {
    let cx = unsafe { &*c };
    let key = MINE.iter().find(|m| m.1 == "rw_lstsq_lowered").unwrap().0;
    let table = rec
        .get("inputs")
        .and_then(|i| i.get("nodes"))
        .map_or(0, |n| n.arr().len());
    let is_lstsq = |n: u32| {
        cx.str_field(n, "kind")
            .is_ok_and(|k| k.starts_with("lstsq"))
            && cx.table.is_a(cx.nodes[n as usize].class, "Op")
    };
    let reaches = |root: u32| {
        let (mut stack, mut seen) = (vec![root], HashSet::new());
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if is_lstsq(n) {
                return true;
            }
            stack.extend(cx.nodes[n as usize].ops.iter().copied())
        }
        false
    };
    let sink = |n: u32| cx.table.is_a(cx.nodes[n as usize].class, "Assign");
    let originals: Vec<u32> = h[..table.min(h.len())]
        .iter()
        .copied()
        .filter(|&n| sink(n) && reaches(n))
        .collect();
    if originals.is_empty() {
        return;
    }
    let made: Vec<u32> = h[table.min(h.len())..]
        .iter()
        .copied()
        .filter(|&n| sink(n))
        .collect();
    let mut skip_with =
        |why: &str| *skips.entry((key.to_string(), why.to_string())).or_default() += 1;
    if made.len() != originals.len() {
        return skip_with("the record does not hold one new sink per lowered sink");
    }
    let Some(spec_dim) = max_dim_of(
        rec.get("inputs").unwrap(),
        rec.get("target").and_then(J::str).unwrap_or(""),
    ) else {
        return skip_with("the unit's spec names no linalg.max_dim");
    };
    let mut lowered = vec![0u32; originals.len()];
    let (mut count, mut changed) = (0usize, 0);
    let st = unsafe {
        rw_lstsq_lowered(
            c,
            originals.as_ptr(),
            originals.len(),
            &spec_dim,
            lowered.as_mut_ptr(),
            lowered.len(),
            &mut count,
            &mut changed,
        )
    };
    if st == 1 {
        return skip_with(&format!(
            "refused by the wing: {}",
            err(c).chars().take(60).collect::<String>()
        ));
    }
    let digest = |nodes: &[u32]| -> String {
        let mut walk = 0u32;
        if rx_canonical(
            c,
            nodes.as_ptr(),
            nodes.len(),
            c"()".as_ptr(),
            std::ptr::null(),
            c"()".as_ptr(),
            &mut walk,
        ) != 0
        {
            return format!("canonical refused: {}", err(c));
        }
        let mut dg = [0 as c_char; 65];
        rx_digest(c, walk, dg.as_mut_ptr());
        unsafe { CStr::from_ptr(dg.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    let (mine, recorded) = if st == 0 {
        (digest(&lowered[..count]), digest(&made))
    } else {
        (format!("status {st}"), String::new())
    };
    let s = stats.entry(key).or_default();
    s.calls += 1;
    s.recs += 1;
    s.units.insert(unit);
    if mine == recorded {
        s.eq += 1;
        s.rec_eq += 1
    } else {
        println!("DIFF {name} | {key} | computed {mine} | recorded {recorded}")
    }
}

/// The digest of the canonical walk over the sinks, or the reason there is none.
fn sinks_digest(c: *mut Ctx, nodes: &[u32]) -> String {
    let mut walk = 0u32;
    if rx_canonical(
        c,
        nodes.as_ptr(),
        nodes.len(),
        c"()".as_ptr(),
        std::ptr::null(),
        c"()".as_ptr(),
        &mut walk,
    ) != 0
    {
        return format!("canonical refused: {}", err(c));
    }
    let mut dg = [0 as c_char; 65];
    rx_digest(c, walk, dg.as_mut_ptr());
    unsafe { CStr::from_ptr(dg.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// A field of a recorded frozen record or dictionary: through its `v` and `of` wrappers, by a text or integer key.
fn dget<'a>(j: &'a J, key: &str) -> Option<&'a J> {
    let (mut j, wrapped) = (j, |j: &'a J| j.get("v").or(j.get("of")));
    while let Some(inner) = wrapped(j) {
        j = inner
    }
    let Some(pairs) = j.get("d").map(J::arr) else {
        return j.get(key);
    };
    pairs
        .iter()
        .find(|p| {
            let k = &p.arr()[0];
            k.get("s").or(k.get("i")).and_then(J::str) == Some(key)
        })
        .map(|p| &p.arr()[1])
}

/// The integer of a recorded tagged integer, if any.
fn dint(j: Option<&J>) -> Option<i64> {
    j?.get("i").and_then(J::str).and_then(|x| x.parse().ok())
}

/// The numbers `rw_wide_place` takes, read from a recorded build spec as the old pass reads them.
struct PlaceFacts {
    thread_form: bool,
    runtime_min: [i64; 5],
    wide_min: i64,
    rolled_min: Option<i64>,
    form: String,
}

/// The placement facts that a recorded spec holds, with `real_bytes` as the real size.
fn place_facts(spec: &J, real_bytes: i64) -> Result<PlaceFacts, String> {
    let v = spec.get("v").ok_or("spec has no fields")?;
    let settings = v.get("settings").ok_or("spec has no settings")?;
    let text = |name: &str| {
        dget(settings, name)
            .and_then(|x| x.get("s"))
            .and_then(J::str)
            .map(String::from)
    };
    let device = v.get("device").filter(|d| !matches!(d, J::Null));
    let primary = v
        .get("focus")
        .filter(|d| !matches!(d, J::Null))
        .or(device)
        .or(v.get("host"))
        .ok_or("spec has no wing")?;
    let rolled_min = dint(dget(primary, "form").and_then(|f| dget(f, "rolled_fold_min")));
    let held = match device {
        None => dget(primary, "host_wide").ok_or("the host wing has no host_wide")?,
        Some(device) => {
            let classes = dget(device, "wide_classes").ok_or("the device has no wide classes")?;
            let named = v
                .get("wide_class")
                .and_then(|c| c.get("s"))
                .and_then(J::str)
                .unwrap_or("");
            let pairs = classes
                .get("of")
                .unwrap_or(classes)
                .get("d")
                .map_or(&[][..], |d| d.arr());
            let chosen = pairs
                .iter()
                .find(|p| p.arr()[0].get("s").and_then(J::str) == Some(named))
                .or(pairs.first())
                .ok_or("no wide class")?;
            &chosen.arr()[1]
        }
    };
    let per_bytes = |table: Option<&J>| -> Result<i64, String> {
        let table = table.ok_or("a table is missing")?;
        dint(dget(table, &real_bytes.to_string()))
            .or(dint(dget(table, "8")))
            .ok_or("table lacks the scalar width".into())
    };
    let runtime = dget(held, "runtime_min").ok_or("no runtime_min")?;
    let mut runtime_min = [0i64; 5];
    for (k, name) in ["read", "slot_read", "rotate", "write", "accum"]
        .iter()
        .enumerate()
    {
        runtime_min[k] = per_bytes(dget(runtime, name))?
    }
    Ok(PlaceFacts {
        thread_form: text("wide_form").as_deref() == Some("thread"),
        runtime_min,
        wide_min: per_bytes(dget(held, "wide_min"))?,
        rolled_min,
        form: text("form").unwrap_or("auto".into()),
    })
}

/// The planes the old pass returns, `(name, per_lane)` in order, from a recorded result.
fn recorded_planes(result: &J) -> Vec<(String, i64)> {
    result
        .get("t")
        .and_then(|t| t.arr().get(1))
        .and_then(|p| p.get("t"))
        .map_or(&[][..], |p| p.arr())
        .iter()
        .map(|p| {
            (
                dget(p, "name")
                    .and_then(|x| x.get("s"))
                    .and_then(J::str)
                    .unwrap_or("")
                    .to_string(),
                dint(dget(p, "per_lane")).unwrap_or(-1),
            )
        })
        .collect()
}

/// The rows `rw_wide_record` gives over the canonical walk of `sinks`, and from them the scratch planes (the warp-tiled rows).
fn placed_rows(c: *mut Ctx, sinks: &[u32]) -> Option<(String, Vec<(String, i64)>)> {
    let mut walk = 0u32;
    if rx_canonical(
        c,
        sinks.as_ptr(),
        sinks.len(),
        c"()".as_ptr(),
        std::ptr::null(),
        c"()".as_ptr(),
        &mut walk,
    ) != 0
    {
        return None;
    }
    let order = list_of(|b, cap, n| rx_walk_order(c, walk, b, cap, n)).ok()?;
    let rows = record_text(c, &order).ok()?;
    let planes = rows
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            (f.get(2) == Some(&"warp_tiled"))
                .then(|| (f[0].to_string(), f[3].parse().unwrap_or(-1)))
        })
        .collect();
    Some((rows, planes))
}

/// Every recorded call of `wide:place`: the export on the recorded sinks, `taken`, `real_bytes` and the thresholds of the recorded
/// spec, against the recorded result: the canonical digest of the placed sinks, whether the sinks changed, and the planes.
fn check_placed(
    c: *mut Ctx,
    rec: &J,
    h: &[u32],
    stats: &mut BTreeMap<&str, Stat>,
    skips: &mut BTreeMap<(String, String), usize>,
    unit: usize,
    name: &str,
) {
    let key = MINE.iter().find(|m| m.1 == "rw_wide_place").unwrap().0;
    for call in rec.get("placement").map_or(&[][..], |p| p.arr()) {
        let mut skip_with = |why: &str| {
            *skips
                .entry((key.to_string(), why.chars().take(90).collect()))
                .or_default() += 1
        };
        let args = call
            .get("args")
            .and_then(|a| a.get("l"))
            .map_or(&[][..], |a| a.arr());
        let refs = |t: Option<&J>| -> Vec<u32> {
            t.and_then(|t| t.get("t"))
                .map_or(&[][..], |t| t.arr())
                .iter()
                .filter_map(|x| node_ref(Some(x), h).ok())
                .collect()
        };
        let (given, want) = (
            refs(args.first()),
            refs(
                call.get("result")
                    .and_then(|r| r.get("t"))
                    .and_then(|t| t.arr().first()),
            ),
        );
        let kwargs = call.get("kwargs");
        let real_bytes = dint(kwargs.and_then(|k| dget(k, "real_bytes"))).unwrap_or(8);
        let taken: Vec<String> = kwargs
            .and_then(|k| dget(k, "taken"))
            .and_then(|t| t.get("of"))
            .and_then(|o| o.get("l"))
            .map_or(vec![], |l| {
                l.arr()
                    .iter()
                    .filter_map(|x| x.get("s").and_then(J::str).map(String::from))
                    .collect()
            });
        let facts = match args.get(1).map(|spec| place_facts(spec, real_bytes)) {
            Some(Ok(f)) => f,
            Some(Err(e)) => {
                skip_with(&format!("spec: {e}"));
                continue;
            }
            None => {
                skip_with("no spec");
                continue;
            }
        };
        let taken_text = CString::new(taken.join("\n")).unwrap();
        let (form, mut made) = (
            CString::new(facts.form.as_str()).unwrap(),
            vec![0u32; given.len()],
        );
        let (mut count, mut changed) = (0usize, 0);
        let st = unsafe {
            rw_wide_place(
                c,
                given.as_ptr(),
                given.len(),
                facts.thread_form as i32,
                facts.runtime_min.as_ptr(),
                facts.wide_min,
                facts
                    .rolled_min
                    .as_ref()
                    .map_or(std::ptr::null(), |r| r as *const i64),
                form.as_ptr(),
                taken_text.as_ptr(),
                made.as_mut_ptr(),
                made.len(),
                &mut count,
                &mut changed,
            )
        };
        let s = stats.entry(key).or_default();
        s.calls += 1;
        s.recs += 1;
        s.units.insert(unit);
        let (mine, recorded) = if st == 0 {
            (sinks_digest(c, &made[..count]), sinks_digest(c, &want))
        } else {
            (format!("status {st}: {}", err(c)), String::new())
        };
        let rows_mine = if st == 0 {
            placed_rows(c, &made[..count])
        } else {
            None
        };
        let rows_want = placed_rows(c, &want);
        let planes_ok = rows_mine.as_ref().map(|r| &r.1) == rows_want.as_ref().map(|r| &r.1)
            && rows_mine
                .as_ref()
                .map(|r| r.1.clone())
                .unwrap_or_default()
                .iter()
                .map(|p| (p.0.clone(), p.1))
                .eq(recorded_planes(call.get("result").unwrap_or(&J::Null)));
        let changed_ok = (changed != 0) == (want != given);
        if mine == recorded
            && changed_ok
            && planes_ok
            && rows_mine.map(|r| r.0) == rows_want.map(|r| r.0)
        {
            s.eq += 1;
            s.rec_eq += 1
        } else {
            println!("DIFF {name} | {key} | changed {changed} (recorded {}) planes_ok {planes_ok} | computed {mine} | recorded {recorded}", want != given)
        }
    }
}

/// The closure `warp_uniform(sinks, warp)` returned answers one node at a time; the recording names the call that made it by a hash
/// (`made_by`) the program cannot invert, so the closures of a unit are matched to its outer calls in the order they first appear.
/// Each recorded answer is compared with the export asked about the same node over the matching outer call's sinks and warp.
fn check_warp_uniform(
    c: *mut Ctx,
    rec: &J,
    h: &[u32],
    stats: &mut BTreeMap<&str, Stat>,
    skips: &mut BTreeMap<(String, String), usize>,
    unit: usize,
    name: &str,
) {
    let key = MINE
        .iter()
        .find(|m| m.1 == "rw_lanes_warp_uniform")
        .unwrap()
        .0;
    let analyses = rec.get("analyses").map_or(&[][..], |x| x.arr());
    let outers: Vec<&J> = analyses
        .iter()
        .filter(|a| {
            a.get("fn")
                .and_then(J::str)
                .is_some_and(|f| f.ends_with("lanes:warp_uniform"))
        })
        .collect();
    let mut hashes: Vec<String> = vec![];
    for call in analyses
        .iter()
        .filter(|a| a.get("fn").and_then(J::str) == Some(key))
    {
        let n = call.get("n").map_or(1, |n| n.int() as usize);
        let hash = call
            .get("kwargs")
            .and_then(|k| k.get("d"))
            .and_then(|d| d.arr().first())
            .and_then(|p| p.arr().get(1))
            .and_then(|v| v.get("s"))
            .and_then(J::str)
            .unwrap_or("")
            .to_string();
        if !hashes.contains(&hash) {
            hashes.push(hash.clone())
        }
        let at = hashes.iter().position(|x| *x == hash).unwrap();
        let mut skip_with =
            |why: &str| *skips.entry((key.to_string(), why.to_string())).or_default() += n;
        let Some(outer) = outers.get(at) else {
            skip_with("no outer call for the closure");
            continue;
        };
        let oargs = outer
            .get("args")
            .and_then(|a| a.get("l"))
            .map_or(&[][..], |a| a.arr());
        let sinks: Vec<u32> = oargs
            .first()
            .and_then(|t| t.get("t"))
            .map_or(&[][..], |t| t.arr())
            .iter()
            .filter_map(|x| node_ref(Some(x), h).ok())
            .collect();
        let warp: Option<i64> = oargs
            .get(1)
            .and_then(|w| w.get("i"))
            .and_then(J::str)
            .and_then(|x| x.parse().ok());
        let Some(want) =
            call.get("result")
                .and_then(|r| if let J::Bool(b) = r { Some(*b) } else { None })
        else {
            skip_with("recorded answer is not a bool");
            continue;
        };
        let Ok(node) = node_ref(
            call.get("args")
                .and_then(|a| a.get("l"))
                .and_then(|l| l.arr().first()),
            h,
        ) else {
            skip_with("argument is not a node reference");
            continue;
        };
        let mut flag = 0;
        let st = unsafe {
            rw_lanes_warp_uniform(
                c,
                sinks.as_ptr(),
                sinks.len(),
                warp.as_ref().map_or(std::ptr::null(), |w| w as *const i64),
                &node,
                1,
                &mut flag,
            )
        };
        let s = stats.entry(key).or_default();
        s.calls += n;
        s.recs += 1;
        s.units.insert(unit);
        if st == 0 && (flag != 0) == want {
            s.eq += n;
            s.rec_eq += 1
        } else {
            println!("DIFF {name} | {key} | status {st} | computed {flag} | recorded {want}")
        }
    }
}

/// The element kind of a node as the word the recording holds.
fn element_kind_word(c: *mut Ctx, node: u32) -> Result<String, i32> {
    let mut kind = 0;
    let st = unsafe { rw_wide_element_kind(c, node, &mut kind) };
    if st != 0 {
        return Err(st);
    }
    let words = ["none", "proxy", "input", "parts", "splat", "elementwise"];
    Ok(words.get(kind as usize).unwrap_or(&"none").to_string())
}

/// The placements over the nodes as the lines `plane resident layout width` the recording holds.
fn record_text(c: *mut Ctx, order: &[u32]) -> Result<String, i32> {
    let mut rows: Vec<PlaneRow> = Vec::new();
    let mut count = 0usize;
    let st = unsafe {
        rw_wide_record(
            c,
            order.as_ptr(),
            order.len(),
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    if st != 0 && st != 3 {
        return Err(st);
    }
    rows.reserve(count);
    let st = unsafe {
        rw_wide_record(
            c,
            order.as_ptr(),
            order.len(),
            rows.as_mut_ptr(),
            count,
            &mut count,
        )
    };
    if st != 0 {
        return Err(st);
    }
    unsafe { rows.set_len(count) };
    let cx = unsafe { &*c };
    let mut text = String::new();
    for row in rows {
        let layout = if row.layout == 1 {
            "warp_tiled"
        } else {
            "plane_major"
        };
        let name = cx.str_field(row.plane, "name").unwrap_or_default();
        text += &format!("{name} resident {layout} {}\n", row.width);
    }
    Ok(text)
}

/// The companions of a walk as the lines `companion=target` the recording holds.
fn companions_text(c: *mut Ctx, walk: u32) -> Result<String, i32> {
    let nodes =
        list_of(|b, cap, n| unsafe { rw_ir_companions(c, walk, b, cap, n) }).map_err(|_| 1)?;
    let cx = unsafe { &*c };
    Ok(nodes
        .iter()
        .map(|&n| {
            let name = cx.str_field(n, "name").unwrap_or_default();
            format!("{name}_c={name}\n")
        })
        .collect())
}

/// The key run of a sink as the text `none` or `<form> <width> <block>`.
fn key_run_text(
    c: *mut Ctx,
    sinks: &[u32],
    sink: u32,
    run_max: i64,
    block_run: i64,
) -> Result<String, i32> {
    let mut run = KeyRunAnswer {
        found: 0,
        form: 0,
        width: 0,
        block: 0,
    };
    let st = unsafe {
        rw_lanes_key_run_in_kernel(
            c,
            sinks.as_ptr(),
            sinks.len(),
            sink,
            run_max,
            block_run,
            &mut run,
        )
    };
    if st != 0 {
        return Err(st);
    }
    let form = ["none", "proven", "sorted", "tested"][run.form as usize];
    Ok(if run.found == 0 {
        "none".to_string()
    } else {
        format!("{form} {} {}", run.width, run.block)
    })
}

/// True for a recorded call this tool answers.
fn is_mine(call: &J) -> bool {
    let f = call.get("fn").and_then(J::str);
    MINE.iter().any(|m| Some(m.0) == f)
}

/// True for the recorded calls of the warp-uniform check: the closure's own calls and the outer calls that carry its arguments.
fn warp_uniform_call(call: &J) -> bool {
    let key = MINE
        .iter()
        .find(|m| m.1 == "rw_lanes_warp_uniform")
        .unwrap()
        .0;
    let f = call.get("fn").and_then(J::str);
    f == Some(key) || f.is_some_and(|f| f.ends_with("lanes:warp_uniform"))
}

/// Reads the arguments, answers every recorded call of every unit and prints the counts per family.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let opt = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let (path, target, only) = (
        a.get(1)
            .cloned()
            .expect("usage: wreplay <texts.jsonl> [--target t] [--limit n] [--only s] [--show k]"),
        opt("--target"),
        opt("--only"),
    );
    let (limit, show) = (
        opt("--limit")
            .and_then(|x| x.parse::<usize>().ok())
            .unwrap_or(usize::MAX),
        opt("--show")
            .and_then(|x| x.parse::<usize>().ok())
            .unwrap_or(0),
    );
    let mut stats: BTreeMap<&str, Stat> = BTreeMap::new();
    let mut skips: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut shown = 0usize;
    let (mut lines, mut bad, mut units, mut build_fail) = (0usize, 0usize, 0usize, 0usize);
    for raw in BufReader::new(std::fs::File::open(&path).expect("open")).split(b'\n') {
        let Ok(raw) = raw else {
            bad += 1;
            continue;
        };
        lines += 1;
        let text = String::from_utf8_lossy(&raw);
        // the recorded calls are read again, one at a time, where they are answered; only the closure calls the warp check needs stay
        let Ok(rec) = json::parse_keeping(&text, "analyses", &warp_uniform_call) else {
            bad += 1;
            continue;
        };
        if target
            .as_deref()
            .is_some_and(|t| rec.get("target").and_then(J::str) != Some(t))
        {
            continue;
        }
        let unit = format!(
            "{} {}",
            rec.get("unit").and_then(J::str).unwrap_or("?"),
            rec.get("test").and_then(J::str).unwrap_or("?")
        );
        if only.as_deref().is_some_and(|o| !unit.contains(o)) {
            continue;
        }
        let mut any_call = rec.get("placement").is_some();
        if !any_call {
            let _ = json::each_in(&text, "analyses", &mut |c| {
                any_call = is_mine(&c);
                !any_call
            });
        }
        if !any_call {
            continue;
        }
        if units >= limit {
            break;
        }
        units += 1;
        let table = with_analysis_nodes(rec.get("inputs").unwrap(), &rec);
        let c = rx_ctx_new();
        let merged = a.iter().any(|x| x == "--own-kinds-merge");
        let own = merged || a.iter().any(|x| x == "--own-kinds");
        let mut declared = manifests(&table);
        if own && !merged {
            declared.retain(|m| {
                !m.starts_with("{\"name\":\"base\"") && !m.starts_with("{\"name\":\"simt\"")
            })
        }
        let built = declare(c, &declared)
            .and_then(|_| {
                if own && unsafe { rw_declare_kinds(c) } != 0 {
                    Err(err(c))
                } else {
                    Ok(())
                }
            })
            .and_then(|_| make_nodes(c, &table, &mut Vals::default()));
        let h = match built {
            Ok(h) => h,
            Err(e) => {
                build_fail += 1;
                let _ = json::each_in(&text, "analyses", &mut |call| {
                    if is_mine(&call) {
                        let f = call.get("fn").and_then(J::str).unwrap_or("");
                        *skips
                            .entry((
                                f.into(),
                                format!(
                                    "unit nodes not rebuilt: {}",
                                    e.chars().take(70).collect::<String>()
                                ),
                            ))
                            .or_default() += call.get("n").map_or(1, |n| n.int() as usize)
                    }
                    true
                });
                rx_ctx_free(c);
                continue;
            }
        };
        let (objs, target) = (
            joined(&rec, "objs"),
            rec.get("target").and_then(J::str).unwrap_or("").to_string(),
        );
        let mut cache = HashMap::new();
        let _ = json::each_in(&text, "analyses", &mut |call| {
            if !is_mine(&call) {
                return true;
            }
            let call = &call;
            let f = MINE
                .iter()
                .find(|m| Some(m.0) == call.get("fn").and_then(J::str))
                .unwrap()
                .0;
            let n = call.get("n").map_or(1, |n| n.int() as usize);
            let r = question(c, call, &h, &rec, &mut cache, &objs, &target);
            let s = stats.entry(f).or_default();
            match r {
                Ok(()) => {
                    s.eq += n;
                    s.calls += n;
                    s.rec_eq += 1;
                    s.recs += 1;
                    s.units.insert(units)
                }
                Err(Out::Diff(d)) => {
                    s.calls += n;
                    s.recs += 1;
                    s.units.insert(units);
                    if shown < show {
                        shown += 1;
                        println!(
                            "DIFF {unit} | {f} | n={n} varies={} | {d}",
                            call.get("varies").map_or(0, |v| v.int())
                        )
                    }
                    false
                }
                Err(Out::Skip(m)) => {
                    *skips.entry((f.into(), m)).or_default() += n;
                    false
                }
                Err(Out::Equal) => false,
            };

            true
        });
        check_walk_props(c, &rec, &cache, &mut stats, &mut skips, units);
        check_lowered(c, &rec, &h, &mut stats, &mut skips, units, &unit);
        check_placed(c, &rec, &h, &mut stats, &mut skips, units, &unit);
        check_warp_uniform(c, &rec, &h, &mut stats, &mut skips, units, &unit);
        rx_ctx_free(c);
    }
    let mut report = format!("lines {lines}, unreadable {bad}, units with questions {units}, units whose nodes were not rebuilt {build_fail}\n");
    for m in MINE {
        let s = stats
            .get(m.0)
            .map(|s| (s.eq, s.calls, s.rec_eq, s.recs, s.units.len()))
            .unwrap_or_default();
        report += &format!(
            "analysis-equal {} / {}   {} ({}: {} / {} distinct, {} units)\n",
            s.0, s.1, m.0, m.1, s.2, s.3, s.4
        );
    }
    for ((f, m), n) in &skips {
        report += &format!("skipped {n}   {f}   {m}\n")
    }
    print!("{report}");
    if let Some(table) = opt("--table") {
        write_table(&table, target.as_deref().unwrap_or("all"), &path, &report)
    }
}

/// Writes the section of `target` in the table file, keeping the sections of other targets: the table is made by this program.
fn write_table(file: &str, target: &str, corpus: &str, report: &str) {
    let corpus = corpus
        .split("corpus/")
        .last()
        .map(|c| format!("corpus/{c}"))
        .unwrap_or_default();
    let old = std::fs::read_to_string(file).unwrap_or_default();
    let mut sections: Vec<(String, String)> = old
        .split("\n== ")
        .skip(1)
        .filter_map(|part| {
            part.split_once(':')
                .map(|(t, rest)| (t.to_string(), rest.to_string()))
        })
        .collect();
    sections.retain(|(t, _)| t != target);
    sections.push((
        target.to_string(),
        format!(" wreplay {corpus} --target {target}\n{report}"),
    ));
    sections.sort();
    let mut out = format!("WINGS_T3: rs/wings on {corpus} (written by wreplay; K / calls weighted by recorded count n)\n");
    for (t, body) in sections {
        out += &format!("\n== {t}:{}", body.trim_end().to_string() + "\n")
    }
    std::fs::write(file, out).expect("write the table");
}
