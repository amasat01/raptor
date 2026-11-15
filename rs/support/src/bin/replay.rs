//! replay <walks.jsonl> [--limit n] [--only <substring of a test id>] [--only-exact <test id>] [--show <k>] [--roundtrip] [--values] [--cost <k>] [--params] [--typed] [--positions <positions.jsonl>]
//! Rebuilds each recorded walk in a fresh context through the C door and compares the digest, arg_spec and access with the recorded ones.
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;
use raptor_support::corpus::*;
use raptor_support::safe::*;
use rcore::json::{self, J};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io::BufRead;
use std::time::Instant;

/// The text of a field of a walk, read through the door.
fn field(c: *mut Ctx, w: u32, f: &str) -> String {
    let f = CString::new(f).unwrap();
    let mut len = 0usize;
    rx_walk_field(c, w, f.as_ptr(), std::ptr::null_mut(), 0, &mut len);
    let mut buf = vec![0 as std::ffi::c_char; len + 1];
    rx_walk_field(c, w, f.as_ptr(), buf.as_mut_ptr(), len + 1, &mut len);
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// The features a recorded walk holds, whatever the door supports.
fn holds(inp: &J) -> Vec<&'static str> {
    let mut h = vec![];
    let nodes = inp.get("nodes").map_or(&[][..], |n| n.arr());
    let class = |n: &J| n.get("c").and_then(J::str).unwrap_or("").to_string();
    if nodes.iter().any(|n| {
        let c = class(n);
        c.starts_with("Loop") || c == "TapeRead" || c == "TapeWrite"
    }) {
        h.push("loops")
    }
    if nodes.iter().any(|n| {
        n.get("v")
            .and_then(|v| v.get("quantity"))
            .is_some_and(|q| !matches!(q, J::Null))
    }) {
        h.push("quantities")
    }
    if inp
        .get("declared")
        .and_then(|d| d.get("l"))
        .is_some_and(|l| !l.arr().is_empty())
    {
        h.push("declared")
    }
    if !inp.get("kinds").map_or(&[][..], |k| k.arr()).is_empty() {
        h.push("dialect")
    }
    if nodes
        .iter()
        .any(|n| n.get("v").and_then(|v| v.get("kind")).and_then(tagged_str) == Some("raw_device"))
    {
        h.push("raw blocks")
    }
    h
}

/// What replaying one recorded unit came to.
enum Outcome {
    Equal,
    Mismatch(String),
    Skip(String),
    MakeRefused(String),
    Raised(bool, String),
}

/// Replays one recorded unit in a fresh context and compares the results with the recorded ones.
fn replay(
    rec: &J,
    counts: &mut [f64; 3],
    roundtrip: bool,
    rt: &mut [usize; 2],
    vals: &mut Vals,
) -> Outcome {
    let inp = rec.get("inputs").unwrap();
    let ms = manifests(inp);
    let t0 = Instant::now();
    let c = rx_ctx_new();
    counts[0] += t0.elapsed().as_secs_f64();
    let mut c2 = std::ptr::null_mut();
    let out = (|| {
        if let Err(e) = declare(c, &ms) {
            return Outcome::MakeRefused(e);
        }
        let h = match make_nodes(c, inp, vals) {
            Ok(h) => h,
            Err(e) => return Outcome::Raised(true, e),
        };
        let sinks: Vec<u32> = inp
            .get("sinks")
            .unwrap()
            .arr()
            .iter()
            .map(|s| h[s.get("n").unwrap().int() as usize])
            .collect();
        let declared = CString::new(val(inp.get("declared").unwrap(), &h)).unwrap();
        let exempt = CString::new(val(inp.get("exempt").unwrap(), &h)).unwrap();
        let force = inp
            .get("force")
            .and_then(tagged_str)
            .map(|f| CString::new(f).unwrap());
        let mut w = 0u32;
        let st = rx_canonical(
            c,
            sinks.as_ptr(),
            sinks.len(),
            declared.as_ptr(),
            force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
            exempt.as_ptr(),
            &mut w,
        );
        if st != 0 {
            let e = err(c);
            return match e.strip_prefix("not in this build: ") {
                Some(f) => Outcome::Skip(f.into()),
                None => Outcome::Raised(false, e),
            };
        }
        let mut dg = [0 as std::ffi::c_char; 65];
        rx_digest(c, w, dg.as_mut_ptr());
        let dg = unsafe { CStr::from_ptr(dg.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let res = rec.get("result").unwrap();
        let (want_dg, want_spec, want_acc) = (
            res.get("digest").and_then(J::str).unwrap_or(""),
            val(
                &J::Obj(vec![(
                    "t".into(),
                    res.get("arg_spec")
                        .and_then(|a| a.get("l"))
                        .cloned()
                        .unwrap_or(J::Arr(vec![])),
                )]),
                &h,
            ),
            val(res.get("access").unwrap(), &h),
        );
        let mut bad = vec![];
        if dg != want_dg {
            bad.push("digest".to_string());
            let (got, want) = (
                field(c, w, "preimage"),
                rec.get("preimage")
                    .and_then(J::str)
                    .unwrap_or("")
                    .to_string(),
            );
            let (g, wl): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
            let i = (0..g.len().max(wl.len()))
                .find(|&i| g.get(i) != wl.get(i))
                .unwrap_or(0);
            bad.push(format!(
                "first differing line {i}: recorded {:?} computed {:?}",
                wl.get(i),
                g.get(i)
            ));
        }
        // the recording holds the sequence as a list; it is compared as the tuple the door writes
        if field(c, w, "arg_spec") != want_spec {
            bad.push("arg_spec".into())
        }
        if field(c, w, "access") != want_acc {
            bad.push("access".into())
        }
        if bad.is_empty() && roundtrip {
            rt[1] += 1;
            let mut len = 0usize;
            rx_dump(
                c,
                sinks.as_ptr(),
                sinks.len(),
                std::ptr::null_mut(),
                0,
                &mut len,
            );
            let mut buf = vec![0 as std::ffi::c_char; len + 1];
            if rx_dump(
                c,
                sinks.as_ptr(),
                sinks.len(),
                buf.as_mut_ptr(),
                len + 1,
                &mut len,
            ) != 0
            {
                return Outcome::Mismatch(format!("roundtrip: dump: {}", err(c)));
            }
            c2 = rx_ctx_new();
            if let Err(e) = declare(c2, &ms) {
                return Outcome::Mismatch(format!("roundtrip: {e}"));
            }
            let mut roots = vec![0u32; sinks.len() + 8];
            let mut nr = 0usize;
            if rx_load(
                c2,
                buf.as_ptr(),
                len,
                roots.as_mut_ptr(),
                roots.len(),
                &mut nr,
            ) != 0
            {
                return Outcome::Mismatch(format!("roundtrip: load: {}", err(c2)));
            }
            let mut w2 = 0u32;
            if rx_canonical(
                c2,
                roots.as_ptr(),
                nr,
                declared.as_ptr(),
                force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
                exempt.as_ptr(),
                &mut w2,
            ) != 0
            {
                return Outcome::Mismatch(format!("roundtrip: canonical: {}", err(c2)));
            }
            let mut d2 = [0 as std::ffi::c_char; 65];
            rx_digest(c2, w2, d2.as_mut_ptr());
            if unsafe { CStr::from_ptr(d2.as_ptr()) }.to_string_lossy() != dg {
                return Outcome::Mismatch("roundtrip: digest".into());
            }
            rt[0] += 1;
        }
        if bad.is_empty() {
            Outcome::Equal
        } else {
            Outcome::Mismatch(bad.join("; "))
        }
    })();
    rx_ctx_free(c);
    if !c2.is_null() {
        rx_ctx_free(c2)
    }
    counts[1] += t0.elapsed().as_secs_f64();
    out
}

/// Reads the arguments, replays every recorded unit and prints the counts; the exit status is 0 when nothing differs.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a
        .get(1)
        .expect("usage: replay <walks.jsonl> [--limit n] [--only s] [--show k]");
    let arg = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1));
    let limit: usize = arg("--limit")
        .and_then(|x| x.parse().ok())
        .unwrap_or(usize::MAX);
    let show: usize = arg("--show").and_then(|x| x.parse().ok()).unwrap_or(3);
    let only = arg("--only");
    let roundtrip = a.iter().any(|x| x == "--roundtrip");
    let mut rt = [0usize; 2];
    let mut vals = Vals {
        on: a.iter().any(|x| x == "--values"),
        ..Default::default()
    };
    if let Some(k) = arg("--cost").and_then(|x| x.parse().ok()) {
        cost(path, k);
        return;
    }
    let exact = arg("--only-exact");
    let (do_params, do_typed) = (
        a.iter().any(|x| x == "--params"),
        a.iter().any(|x| x == "--typed"),
    );
    let (mut pst, mut tst) = (PStats::default(), TStats::default());
    let mut sst = SStats::default();
    let (do_loops, mut lst) = (arg("--loops").is_some(), LoopStats::default());
    let loops_oracle = arg("--loops")
        .map(|f| read_loops_oracle(f))
        .unwrap_or_default();
    let oracle: HashMap<String, J> = match arg("--positions") {
        Some(f) => std::fs::read_to_string(f)
            .expect("positions")
            .lines()
            .map(|l| {
                let j = json::parse(l).expect("json");
                (j.get("key").and_then(J::str).unwrap_or("").to_string(), j)
            })
            .collect(),
        None => HashMap::new(),
    };
    let mut holding: HashMap<&str, usize> = HashMap::new();
    let start = Instant::now();
    let (mut u, mut equal, mut raised, mut made_bad, mut bad) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut spec_eq, mut acc_eq, mut dg_eq) = (0usize, 0usize, 0usize);
    let mut skips: HashMap<String, usize> = HashMap::new();
    let mut refused: HashMap<String, usize> = HashMap::new();
    let mut shown = 0;
    let (mut ref_eq, mut diffs) = (0usize, Vec::<String>::new());
    let (mut make_eq, mut make_n, mut canon_n) = (0usize, 0usize, 0usize);
    let mut counts = [0f64; 3];
    for line in std::io::BufReader::new(std::fs::File::open(path).expect("open")).lines() {
        if u >= limit {
            break;
        }
        let line = line.unwrap();
        let rec = json::parse(&line).expect("json");
        let tests: Vec<String> = rec
            .get("tests")
            .map_or(&[][..], |t| t.arr())
            .iter()
            .filter_map(|t| t.str().map(String::from))
            .collect();
        if let Some(o) = exact {
            if !tests.iter().any(|t| t == o) {
                continue;
            }
        }
        if let Some(o) = only {
            if !tests.iter().any(|t| t.contains(o.as_str())) {
                continue;
            }
        }
        u += 1;
        let want_err = match rec.get("error") {
            None | Some(J::Null) => None,
            Some(e) => Some(e.get("msg").and_then(J::str).unwrap_or("").to_string()),
        };
        for f in holds(rec.get("inputs").unwrap()) {
            *holding.entry(f).or_default() += 1
        }
        let want_err_now = !matches!(rec.get("error"), None | Some(J::Null));
        if !want_err_now {
            if do_params {
                let t = Instant::now();
                params_walk(&rec, &mut pst);
                pst.secs += t.elapsed().as_secs_f64()
            }
            if !oracle.is_empty() {
                let t = Instant::now();
                if let Some(o) = rec.get("key").and_then(J::str).and_then(|k| oracle.get(k)) {
                    positions_walk(&rec, o, &mut sst)
                }
                sst.secs += t.elapsed().as_secs_f64()
            }
            if do_loops {
                loops_walk(&rec, &loops_oracle, &mut lst)
            }
            if do_typed {
                let t = Instant::now();
                typed_walk(&rec, &mut tst);
                tst.secs += t.elapsed().as_secs_f64()
            }
        }
        let test = tests.first().map_or("?", |t| t.as_str());
        match replay(&rec, &mut counts, roundtrip, &mut rt, &mut vals) {
            Outcome::Raised(at_make, m) => match &want_err {
                Some(w) => {
                    raised += 1;
                    if at_make {
                        make_n += 1
                    } else {
                        canon_n += 1
                    }
                    // a refusal at make is "<kind>: <text>"; the oracle's walk verifier wrote "node <i> (<kind>): <text>"
                    let rewritten = w
                        .strip_prefix("node ")
                        .and_then(|r| r.split_once(" ("))
                        .and_then(|(_, r)| r.split_once("): "))
                        .map(|(k, t)| format!("{k}: {t}"));
                    if at_make && (*w == m || rewritten.as_deref() == Some(m.as_str())) {
                        make_eq += 1
                    } else if !at_make && *w == m {
                        ref_eq += 1
                    } else {
                        diffs.push(format!("{test}\n  recorded: {w:?}\n  computed: {m:?}"))
                    }
                }
                None if at_make => {
                    made_bad += 1;
                    *refused.entry(m).or_default() += 1
                }
                None => {
                    bad += 1;
                    if shown < show {
                        shown += 1;
                        println!("MISMATCH {test} : canonical refused: {m}")
                    }
                }
            },
            _ if want_err.is_some() => {
                raised += 1;
                diffs.push(format!(
                    "{test}\n  recorded: {:?}\n  computed: no refusal",
                    want_err.unwrap()
                ))
            }
            Outcome::Equal => {
                equal += 1;
                dg_eq += 1;
                spec_eq += 1;
                acc_eq += 1
            }
            Outcome::Skip(f) => *skips.entry(f).or_default() += 1,
            Outcome::MakeRefused(m) => {
                made_bad += 1;
                *refused.entry(m).or_default() += 1
            }
            Outcome::Mismatch(m) => {
                bad += 1;
                if !m.contains("digest") {
                    dg_eq += 1
                }
                if !m.contains("arg_spec") {
                    spec_eq += 1
                }
                if !m.contains("access") {
                    acc_eq += 1
                }
                if shown < show {
                    shown += 1;
                    println!(
                        "MISMATCH {} : {m}",
                        tests.first().map_or("?", |t| t.as_str())
                    )
                }
            }
        }
    }
    let ran = equal + bad;
    let sk = |k: &str| skips.get(k).copied().unwrap_or(0);
    println!("digest-equal {dg_eq} / {u}");
    println!("arg_spec-equal {spec_eq} / {ran}");
    println!("access-equal {acc_eq} / {ran}");
    let mut other: Vec<_> = skips
        .iter()
        .filter(|(k, _)| {
            !["loops", "quantities", "declared slots", "dialect"].contains(&k.as_str())
        })
        .collect();
    other.sort();
    println!("skipped by feature: loops {}, quantities {}, declared {}, dialect {}, raised-in-oracle {raised}{}", sk("loops"), sk("quantities"), sk("declared slots"), sk("dialect"),
             other.iter().map(|(k, n)| format!(", {k} {n}")).collect::<String>());
    println!(
        "walks holding: {}",
        ["loops", "quantities", "declared", "dialect", "raw blocks"]
            .iter()
            .map(|k| format!("{k} {}", holding.get(k).copied().unwrap_or(0)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if roundtrip {
        println!("roundtrip-equal {} / {}", rt[0], rt[1])
    }
    println!("refusal-equal {ref_eq} / {canon_n}");
    println!("refused-at-make-equal {make_eq} / {make_n}");
    for d in &diffs {
        println!("REFUSAL DIFFERS {d}")
    }
    println!("make-refused {made_bad}");
    let mut top: Vec<_> = refused.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (m, n) in top.iter().take(5) {
        println!("  {n} x {m}")
    }
    println!("mismatched {bad}");
    if vals.on {
        report_values(&vals);
    }
    if !oracle.is_empty() {
        sst.report()
    }
    if do_params {
        pst.report()
    }
    if do_typed {
        tst.report()
    }
    if do_loops {
        report_loops(&lst);
    }
    println!(
        "timing: total {:.2} s, rx_ctx_new {:.1} us mean, {:.1} us per walk",
        start.elapsed().as_secs_f64(),
        counts[0] / ran.max(1) as f64 * 1e6,
        counts[1] / ran.max(1) as f64 * 1e6
    );
    std::process::exit(if bad == 0 && made_bad == 0 { 0 } else { 1 });
}

/// Prints the value read-back counts, the histogram of value kinds and every difference.
fn report_values(vals: &Vals) {
    println!("values read back equal {} / {}", vals.equal, vals.total);
    println!("fixed-or-dropped {}", vals.fixed);
    println!("field2-equal {} / {}", vals.f2_equal, vals.f2_total);
    let mut hist: Vec<_> = vals.hist.iter().collect();
    hist.sort();
    for (k, [n, e]) in hist {
        println!("  {k}: {n} (in equal slots {e})")
    }
    for d in &vals.diffs {
        println!("DIFFERENCE {d}")
    }
}

/// Prints how many loop rebuilds equal the old build, and every difference.
fn report_loops(lst: &LoopStats) {
    println!(
        "loop-rebuild-equal {} / {}  (compared with the old build_loop, line by line)",
        lst.equal, lst.loops
    );
    for d in &lst.diffs {
        println!("LOOP DIFFERS {d}")
    }
}

/// The median of the values.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The cost of crossing the door for the k largest walks: dump, load into a new context, canonical there (median of 5).
fn cost(path: &str, k: usize) {
    let mut top: Vec<(usize, String)> = vec![];
    for line in std::io::BufReader::new(std::fs::File::open(path).expect("open")).lines() {
        let line = line.unwrap();
        let rec = json::parse(&line).expect("json");
        if !matches!(rec.get("error"), None | Some(J::Null)) {
            continue;
        }
        let n = rec
            .get("inputs")
            .unwrap()
            .get("nodes")
            .map_or(0, |n| n.arr().len());
        if top.len() < k || n > top.last().unwrap().0 {
            top.push((n, line));
            top.sort_by_key(|row| std::cmp::Reverse(row.0));
            top.truncate(k)
        }
    }
    println!(
        "{:>8} {:>11} {:>9} {:>9} {:>9} {:>8}  test",
        "nodes", "dump bytes", "dump ms", "load ms", "canon ms", "ctx ms"
    );
    let mut cols: [Vec<f64>; 6] = Default::default();
    for (n, line) in &top {
        let rec = json::parse(line).unwrap();
        let inp = rec.get("inputs").unwrap();
        let ms = manifests(inp);
        let c = rx_ctx_new();
        declare(c, &ms).unwrap();
        let h = make_nodes(c, inp, &mut Vals::default()).expect("make");
        let sinks: Vec<u32> = inp
            .get("sinks")
            .unwrap()
            .arr()
            .iter()
            .map(|s| h[s.get("n").unwrap().int() as usize])
            .collect();
        let declared = CString::new(val(inp.get("declared").unwrap(), &h)).unwrap();
        let exempt = CString::new(val(inp.get("exempt").unwrap(), &h)).unwrap();
        let force = inp
            .get("force")
            .and_then(tagged_str)
            .map(|f| CString::new(f).unwrap());
        let (mut dump_t, mut load_t, mut canon_t, mut ctx_t) = (vec![], vec![], vec![], vec![]);
        let mut bytes = 0usize;
        for _ in 0..5 {
            let mut len = 0usize;
            rx_dump(
                c,
                sinks.as_ptr(),
                sinks.len(),
                std::ptr::null_mut(),
                0,
                &mut len,
            );
            let mut buf = vec![0 as std::ffi::c_char; len + 1];
            let t = Instant::now();
            assert_eq!(
                rx_dump(
                    c,
                    sinks.as_ptr(),
                    sinks.len(),
                    buf.as_mut_ptr(),
                    len + 1,
                    &mut len
                ),
                0,
                "{}",
                err(c)
            );
            dump_t.push(t.elapsed().as_secs_f64() * 1e3);
            bytes = len;
            let t = Instant::now();
            let c2 = rx_ctx_new();
            ctx_t.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            declare(c2, &ms).unwrap();
            let mut roots = vec![0u32; sinks.len() + 8];
            let mut nr = 0usize;
            assert_eq!(
                rx_load(
                    c2,
                    buf.as_ptr(),
                    len,
                    roots.as_mut_ptr(),
                    roots.len(),
                    &mut nr
                ),
                0,
                "{}",
                err(c2)
            );
            load_t.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let mut w = 0u32;
            assert_eq!(
                rx_canonical(
                    c2,
                    roots.as_ptr(),
                    nr,
                    declared.as_ptr(),
                    force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
                    exempt.as_ptr(),
                    &mut w
                ),
                0,
                "{}",
                err(c2)
            );
            canon_t.push(t.elapsed().as_secs_f64() * 1e3);
            rx_ctx_free(c2);
        }
        rx_ctx_free(c);
        let id = rec
            .get("tests")
            .and_then(|t| t.arr().first())
            .and_then(J::str)
            .unwrap_or("?");
        let id: String = id.chars().take(60).collect();
        let row = [
            *n as f64,
            bytes as f64,
            median(dump_t),
            median(load_t),
            median(canon_t),
            median(ctx_t),
        ];
        println!(
            "{:>8} {:>11} {:>9.3} {:>9.3} {:>9.3} {:>8.3}  {id}",
            n, bytes, row[2], row[3], row[4], row[5]
        );
        for (i, v) in row.iter().enumerate() {
            cols[i].push(*v)
        }
    }
    let m: Vec<f64> = cols.iter().map(|c| median(c.clone())).collect();
    println!(
        "{:>8} {:>11} {:>9.3} {:>9.3} {:>9.3} {:>8.3}  median over {} walks",
        m[0],
        m[1],
        m[2],
        m[3],
        m[4],
        m[5],
        top.len()
    );
}

#[derive(Default)]
/// Counters of the argument-spec comparison.
struct PStats {
    by_class: std::collections::BTreeMap<String, [usize; 2]>,
    walks: usize,
    digest_eq: usize,
    diffs: Vec<String>,
    secs: f64,
    nodes: usize,
}
impl PStats {
    /// Prints the counters.
    fn report(&self) {
        let (p, n): (usize, usize) = self
            .by_class
            .values()
            .fold((0, 0), |a, v| (a.0 + v[1], a.1 + v[0]));
        println!("params-equal {p} / {n}");
        for (k, v) in &self.by_class {
            println!("  {k}: {} / {}", v[1], v[0])
        }
        println!("digest-equal-by-params {} / {}", self.digest_eq, self.walks);
        for d in &self.diffs {
            println!("PARAMS DIFFERENCE {d}")
        }
        println!(
            "params mode: {:.2} s, {:.1} us per node",
            self.secs,
            self.secs / self.nodes.max(1) as f64 * 1e6
        );
    }
}

/// The type text of a node, read through the door.
fn read_ttype(c: *mut Ctx, h: u32) -> String {
    text_out(c, |c, b, cap, l| rx_ttype(c, h, b, cap, l))
}
/// The operand handles of a node, read through the door.
fn read_ops(c: *mut Ctx, h: u32) -> Vec<u32> {
    let mut got = vec![0u32; 64];
    let mut n = 0usize;
    if rx_operands(c, h, got.as_mut_ptr(), got.len(), &mut n) != 0 {
        got = vec![0u32; n];
        rx_operands(c, h, got.as_mut_ptr(), got.len(), &mut n);
    }
    got.truncate(n);
    got
}
/// The text of field `k` of a node, read through the door.
fn read_field(c: *mut Ctx, h: u32, k: &str) -> String {
    let f = CString::new(k).unwrap();
    text_out(c, |c, b, cap, l| rx_field(c, h, f.as_ptr(), b, cap, l))
}

/// Every node again from its constructor parameters in a second context; the records must equal, and so must the walk's digest.
fn params_walk(rec: &J, st: &mut PStats) {
    let inp = rec.get("inputs").unwrap();
    let ms = manifests(inp);
    let c1 = rx_ctx_new();
    let c2 = rx_ctx_new();
    st.walks += 1;
    let done = (|| -> Option<()> {
        declare(c1, &ms).ok()?;
        declare(c2, &ms).ok()?;
        let h = make_nodes(c1, inp, &mut Vals::default()).ok()?;
        let nodes = inp.get("nodes").unwrap().arr();
        for (i, &hi) in h.iter().enumerate() {
            st.nodes += 1;
            let cname = text_out(c1, |c, b, cap, l| rx_class(c, hi, b, cap, l));
            let e = st.by_class.entry(cname.clone()).or_default();
            e[0] += 1;
            let mut refs: Vec<u32> = vec![];
            let kt = match raptor_ir::ctor::params_kt(unsafe { &*c1 }, hi, &mut refs) {
                Ok(k) => k,
                Err(m) => {
                    if st.diffs.len() < 5 {
                        st.diffs.push(format!(
                            "{cname} (node {i}): parameters from the state: {m}"
                        ))
                    }
                    return None;
                }
            };
            let cn = CString::new(cname.as_str()).unwrap();
            let mut out = 0u32;
            if rx_new(
                c2,
                cn.as_ptr(),
                kt.as_ptr() as *const _,
                kt.len(),
                refs.as_ptr(),
                refs.len(),
                &mut out,
            ) != 0
            {
                if st.diffs.len() < 5 {
                    st.diffs.push(format!(
                        "{cname} (node {i}): rx_new refused: {}\n  parameters: {kt}",
                        err(c2)
                    ))
                }
                return None;
            }
            let mut bad = vec![];
            if out != hi {
                bad.push("handle".to_string())
            }
            if read_ttype(c1, hi) != read_ttype(c2, out) {
                bad.push(format!(
                    "ttype {} vs {}",
                    read_ttype(c1, hi),
                    read_ttype(c2, out)
                ))
            }
            if read_ops(c1, hi) != read_ops(c2, out) {
                bad.push("operands".into())
            }
            for (k, _) in nodes[i]
                .get("v")
                .unwrap()
                .obj()
                .iter()
                .filter(|(k, _)| k != "ttype" && k != "operands")
            {
                let (a, b) = (read_field(c1, hi, k), read_field(c2, out, k));
                if a != b {
                    bad.push(format!("{k}: {a} vs {b}"))
                }
            }
            if bad.is_empty() {
                st.by_class.get_mut(&cname).unwrap()[1] += 1
            } else if st.diffs.len() < 5 {
                st.diffs.push(format!(
                    "{cname} (node {i}): {}\n  parameters: {kt}",
                    bad.join("; ")
                ))
            }
        }
        let sinks: Vec<u32> = inp
            .get("sinks")
            .unwrap()
            .arr()
            .iter()
            .map(|s| h[s.get("n").unwrap().int() as usize])
            .collect();
        let declared = CString::new(val(inp.get("declared").unwrap(), &h)).unwrap();
        let exempt = CString::new(val(inp.get("exempt").unwrap(), &h)).unwrap();
        let force = inp
            .get("force")
            .and_then(tagged_str)
            .map(|f| CString::new(f).unwrap());
        let mut w = 0u32;
        if rx_canonical(
            c2,
            sinks.as_ptr(),
            sinks.len(),
            declared.as_ptr(),
            force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
            exempt.as_ptr(),
            &mut w,
        ) != 0
        {
            return None;
        }
        let mut d = [0 as std::ffi::c_char; 65];
        raptor_support::safe::rx_digest(c2, w, d.as_mut_ptr());
        let got = unsafe { CStr::from_ptr(d.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if Some(got.as_str())
            == rec
                .get("result")
                .and_then(|r| r.get("digest"))
                .and_then(J::str)
        {
            st.digest_eq += 1
        }
        Some(())
    })();
    let _ = done;
    rx_ctx_free(c1);
    rx_ctx_free(c2);
}

/// What the loop constructor gave for every recorded loop, against the recorded node.
#[derive(Default)]
struct LoopStats {
    loops: usize,
    equal: usize,
    diffs: Vec<String>,
}

/// The handles a recorded loop field holds, in order.
fn loop_refs(cx: &raptor_ir::record::Ctx, lp: u32, field: &str) -> Vec<u32> {
    match cx.field(lp, field) {
        Ok(Val::Node(n)) => vec![n],
        Ok(Val::Tuple(v)) | Ok(Val::List(v)) => v
            .iter()
            .filter_map(|x| if let Val::Node(n) = x { Some(*n) } else { None })
            .collect(),
        _ => vec![],
    }
}

/// The old function's answers: one line per loop, by the walk's key and the loop's entry.
fn read_loops_oracle(file: &str) -> HashMap<(String, usize), J> {
    let text = std::fs::read_to_string(file).expect("loops oracle");
    text.lines()
        .map(|line| {
            let row = json::parse(line).expect("json");
            let key = row.get("key").and_then(J::str).unwrap_or("").to_string();
            ((key, row.get("pos").map_or(0, |p| p.int() as usize)), row)
        })
        .collect()
}

/// A node as the oracle writes it: `#entry` for a node of the walk, else its class, kind, literal,
/// type and operands.
fn node_form(cx: &raptor_ir::record::Ctx, node: u32, entries: usize) -> String {
    if (node as usize) < entries {
        return format!("#{node}");
    }
    let n = &cx.nodes[node as usize];
    let text = |f: &str, none: &str| match cx.field(node, f) {
        Ok(v) => raptor_ir::value::text_val(&v),
        Err(_) => none.to_string(),
    };
    let dims: Vec<String> = n.shape.iter().map(|d| d.to_string()).collect();
    let shape = match dims.len() {
        1 => format!("({},)", dims[0]),
        _ => format!("({})", dims.join(", ")),
    };
    let kind = match cx.field(node, "kind") {
        Ok(Val::Str(k)) => k,
        _ => String::new(),
    };
    let operands: Vec<String> = n.ops.iter().map(|&o| node_form(cx, o, entries)).collect();
    let class = &cx.table.classes[n.class].name;
    let literal = text("literal", "None");
    format!(
        "({class}|{kind}|{literal}|{}|{shape}|{})",
        n.dtype,
        operands.join(",")
    )
}

/// The form of a built loop: operands, nexts, break, count and free count, as one line.
fn loop_form(cx: &raptor_ir::record::Ctx, lp: u32, entries: usize) -> String {
    let list = |f: &str| {
        let items: Vec<String> = loop_refs(cx, lp, f)
            .iter()
            .map(|&n| node_form(cx, n, entries))
            .collect();
        format!("[{}]", items.join(","))
    };
    let one = |f: &str| {
        loop_refs(cx, lp, f)
            .first()
            .map_or("None".to_string(), |&n| node_form(cx, n, entries))
    };
    let ops: Vec<String> = cx.nodes[lp as usize]
        .ops
        .iter()
        .map(|&n| node_form(cx, n, entries))
        .collect();
    let free_end = match cx.field(lp, "_free_end") {
        Ok(Val::Int(i)) => i.to_string(),
        _ => "?".into(),
    };
    format!(
        "ops[{}] nexts{} exit {} {} count {} free_end {free_end}",
        ops.join(","),
        list("nexts"),
        one("exit_cond"),
        list("exit_values"),
        one("count")
    )
}

/// The same line from an oracle row.
fn oracle_form(row: &J) -> String {
    let form = row.get("form").unwrap();
    let list = |f: &str| {
        form.get(f)
            .unwrap()
            .arr()
            .iter()
            .map(|x| x.str().unwrap().to_string())
    };
    let one = |f: &str| form.get(f).and_then(J::str).unwrap_or("None").to_string();
    let join = |f: &str| list(f).collect::<Vec<_>>().join(",");
    let free_end = form.get("free_end").map_or(0, |f| f.int());
    format!(
        "ops[{}] nexts[{}] exit {} [{}] count {} free_end {free_end}",
        join("operands"),
        join("nexts"),
        one("exit_cond"),
        join("exit_values"),
        one("count")
    )
}

/// Every Loop of a recorded walk built again by `rx_new_loop` from its own index, carries,
/// initial values, results, break and count;
/// the new node must equal the recorded one in its type, operands and every field.
fn loops_walk(rec: &J, oracle: &HashMap<(String, usize), J>, st: &mut LoopStats) {
    let inp = rec.get("inputs").unwrap();
    let c = rx_ctx_new();
    if declare(c, &manifests(inp)).is_err() {
        rx_ctx_free(c);
        return;
    }
    let Ok(h) = make_nodes(c, inp, &mut Vals::default()) else {
        rx_ctx_free(c);
        return;
    };
    let cx = unsafe { &*c };
    let nodes = inp.get("nodes").unwrap().arr();
    for (i, &lp) in h.iter().enumerate() {
        if nodes[i].get("c").and_then(J::str) != Some("Loop") {
            continue;
        }
        st.loops += 1;
        let (carries, inits, nexts) = (
            loop_refs(cx, lp, "carries"),
            loop_refs(cx, lp, "inits"),
            loop_refs(cx, lp, "nexts"),
        );
        let (sinks, exit_values) = (
            loop_refs(cx, lp, "body_sinks"),
            loop_refs(cx, lp, "exit_values"),
        );
        let int = |f: &str| match cx.field(lp, f) {
            Ok(Val::Int(v)) => v as i64,
            _ => 0,
        };
        let one = |f: &str| loop_refs(cx, lp, f).first().copied();
        let (cond, count) = (one("exit_cond"), one("count"));
        let dynamic = match cx.field(lp, "dynamic") {
            Ok(Val::Str(d)) => CString::new(d).ok(),
            _ => None,
        };
        let ptr = |n: &Option<u32>| n.as_ref().map_or(std::ptr::null(), |n| n as *const u32);
        let mut out = 0u32;
        let status = raptor_support::safe::rx_new_loop(
            c,
            loop_refs(cx, lp, "index")[0],
            carries.as_ptr(),
            carries.len(),
            inits.as_ptr(),
            nexts.as_ptr(),
            int("start"),
            int("stop"),
            int("step"),
            sinks.as_ptr(),
            sinks.len(),
            ptr(&cond),
            exit_values.as_ptr(),
            exit_values.len(),
            ptr(&count),
            (cx.field(lp, "count_tail").ok() == Some(Val::Bool(true))) as i32,
            dynamic.as_ref().map_or(std::ptr::null(), |d| d.as_ptr()),
            &mut out,
        );
        let mut bad = vec![];
        let key = rec.get("key").and_then(J::str).unwrap_or("");
        match (oracle.get(&(key.to_string(), i)), status) {
            (None, _) => bad.push("the oracle has no line for this loop".to_string()),
            (Some(want), _) if want.get("refused").is_some() => {
                if status == 0 {
                    bad.push("the old function refused, the new one built".to_string())
                }
            }
            (Some(_), s) if s != 0 => bad.push(format!("rx_new_loop refused: {}", err(c))),
            (Some(want), _) => {
                let (mine, theirs) = (loop_form(cx, out, h.len()), oracle_form(want));
                if mine != theirs {
                    bad.push(format!("form {mine} vs old {theirs}"))
                }
            }
        }
        if bad.is_empty() {
            st.equal += 1
        } else if st.diffs.len() < 12 {
            st.diffs.push(format!(
                "loop node {i} of {}: {}",
                rec.get("tests")
                    .and_then(|t| t.arr().first())
                    .and_then(J::str)
                    .unwrap_or("?"),
                bad.join("; ")
            ))
        }
    }
    rx_ctx_free(c);
}
#[derive(Default)]
/// Counters of the type comparison.
struct TStats {
    by_kind: std::collections::BTreeMap<String, [usize; 2]>,
    explicit: std::collections::BTreeMap<String, usize>,
    diffs: Vec<String>,
    secs: f64,
    ops: usize,
}
impl TStats {
    /// Prints the counters.
    fn report(&self) {
        let (t, n): (usize, usize) = self
            .by_kind
            .values()
            .fold((0, 0), |a, v| (a.0 + v[1], a.1 + v[0]));
        println!("type-equal {t} / {n}");
        for (k, v) in &self.by_kind {
            println!("  {k}: {} / {}", v[1], v[0])
        }
        println!("explicit {}", self.explicit.values().sum::<usize>());
        for (k, v) in &self.explicit {
            println!("  {k}: {v}")
        }
        for d in &self.diffs {
            println!("TYPED DIFFERENCE {d}")
        }
        println!(
            "typed mode: {:.2} s, {:.1} us per op",
            self.secs,
            self.secs / self.ops.max(1) as f64 * 1e6
        );
    }
}

/// Every Op again through the typed constructor: the inferred type must be the recorded one.
fn typed_walk(rec: &J, st: &mut TStats) {
    let inp = rec.get("inputs").unwrap();
    let ms = manifests(inp);
    let (c1, c2) = (rx_ctx_new(), rx_ctx_new());
    let _ = (|| -> Option<()> {
        declare(c1, &ms).ok()?;
        declare(c2, &ms).ok()?;
        let h = make_nodes(c1, inp, &mut Vals::default()).ok()?;
        make_nodes(c2, inp, &mut Vals::default()).ok()?;
        for (i, n) in inp.get("nodes").unwrap().arr().iter().enumerate() {
            if n.get("c").and_then(J::str) != Some("Op") {
                continue;
            }
            st.ops += 1;
            let hi = h[i];
            let kind = match raptor_ir::value::parse(&read_field(c1, hi, "kind")) {
                Ok(Val::Str(k)) => k,
                _ => continue,
            };
            let lit = read_field(c1, hi, "literal");
            let ops = read_ops(c1, hi);
            let (ck, cl) = (
                CString::new(kind.as_str()).unwrap(),
                CString::new(lit.as_str()).unwrap(),
            );
            let mut out = 0u32;
            let e = st.by_kind.entry(kind.clone()).or_default();
            if rx_make_op(
                c2,
                ck.as_ptr(),
                ops.as_ptr(),
                ops.len(),
                cl.as_ptr() as *const _,
                lit.len(),
                &mut out,
            ) != 0
            {
                let m = err(c2);
                st.by_kind.get_mut(&kind).unwrap()[0] = st.by_kind[&kind][0].saturating_sub(0);
                let why = if m.starts_with("unknown op kind") {
                    "outside the vocabulary"
                } else {
                    "refused by the typing rules"
                };
                *st.explicit.entry(format!("{kind} ({why})")).or_default() += 1;
                if st.by_kind[&kind] == [0, 0] {
                    st.by_kind.remove(&kind);
                }
                continue;
            }
            e[0] += 1;
            let (a, b) = (read_ttype(c1, hi), read_ttype(c2, out));
            if a == b {
                st.by_kind.get_mut(&kind).unwrap()[1] += 1
            } else if st.diffs.len() < 5 {
                st.diffs.push(format!(
                    "{kind}: recorded {a}, inferred {b}\n  operands {ops:?}, literal {lit}"
                ))
            }
        }
        Some(())
    })();
    rx_ctx_free(c1);
    rx_ctx_free(c2);
}

#[derive(Default)]
/// Counters of the position comparison.
struct SStats {
    pos: [usize; 2],
    bn: [usize; 2],
    br: [usize; 2],
    diffs: Vec<String>,
    secs: f64,
    walks: usize,
}
impl SStats {
    /// Prints the counters.
    fn report(&self) {
        println!("positions-equal {} / {}", self.pos[1], self.pos[0]);
        println!("body-nodes-equal {} / {}", self.bn[1], self.bn[0]);
        println!("body-roots-equal {} / {}", self.br[1], self.br[0]);
        for d in &self.diffs {
            println!("POSITIONS DIFFERENCE {d}")
        }
        println!(
            "positions mode: {:.2} s over {} walks",
            self.secs, self.walks
        );
    }
}

/// A list of handles that a door call writes into a buffer, read for walk or loop `h`.
fn handles(
    c: *mut Ctx,
    f: fn(*mut Ctx, u32, *mut u32, usize, *mut usize) -> i32,
    h: u32,
) -> Option<Vec<u32>> {
    let mut n = 0usize;
    let mut buf = vec![0u32; 64];
    let mut st = f(c, h, buf.as_mut_ptr(), buf.len(), &mut n);
    if st == 3 {
        buf = vec![0u32; n];
        st = f(c, h, buf.as_mut_ptr(), buf.len(), &mut n)
    }
    if st != 0 {
        return None;
    }
    buf.truncate(n);
    Some(buf)
}

/// The door's positions and loop bodies against the ones the frozen constructor gave (one oracle line per walk).
fn positions_walk(rec: &J, oracle: &J, st: &mut SStats) {
    let inp = rec.get("inputs").unwrap();
    let ms = manifests(inp);
    let c = rx_ctx_new();
    st.walks += 1;
    let _ = (|| -> Option<()> {
        declare(c, &ms).ok()?;
        let h = make_nodes(c, inp, &mut Vals::default()).ok()?;
        let sinks: Vec<u32> = inp
            .get("sinks")
            .unwrap()
            .arr()
            .iter()
            .map(|s| h[s.get("n").unwrap().int() as usize])
            .collect();
        let declared = CString::new(val(inp.get("declared").unwrap(), &h)).unwrap();
        let exempt = CString::new(val(inp.get("exempt").unwrap(), &h)).unwrap();
        let force = inp
            .get("force")
            .and_then(tagged_str)
            .map(|f| CString::new(f).unwrap());
        let mut w = 0u32;
        if rx_canonical(
            c,
            sinks.as_ptr(),
            sinks.len(),
            declared.as_ptr(),
            force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
            exempt.as_ptr(),
            &mut w,
        ) != 0
        {
            return None;
        }
        let want = oracle.get("positions").unwrap();
        let key = rec.get("key").and_then(J::str).unwrap_or("?");
        for (i, &hi) in h.iter().enumerate() {
            st.pos[0] += 1;
            let mut p = 0u32;
            let got = if rx_walk_position(c, w, hi, &mut p) == 0 {
                Some(p as i64)
            } else {
                None
            };
            let exp = want.get(&i.to_string()).map(|x| x.int());
            if got == exp {
                st.pos[1] += 1
            } else if st.diffs.len() < 5 {
                st.diffs.push(format!(
                    "walk {} entry {i}: oracle {exp:?}, door {got:?}",
                    &key[..12.min(key.len())]
                ))
            }
        }
        for (k, l) in oracle.get("loops").map_or(&[][..], |l| l.obj()) {
            let i: usize = k.parse().ok()?;
            let list = |name: &str| -> Vec<u32> {
                l.get(name)
                    .map_or(&[][..], |a| a.arr())
                    .iter()
                    .map(|x| h[x.int() as usize])
                    .collect()
            };
            for (what, f, exp, tot) in [
                (
                    "nodes",
                    rx_body_nodes as fn(*mut Ctx, u32, *mut u32, usize, *mut usize) -> i32,
                    list("nodes"),
                    0,
                ),
                ("roots", rx_body_roots, list("roots"), 1),
            ] {
                let got = handles(c, f, h[i]);
                let slot = if tot == 0 { &mut st.bn } else { &mut st.br };
                slot[0] += 1;
                if got.as_ref() == Some(&exp) {
                    slot[1] += 1
                } else if st.diffs.len() < 5 {
                    st.diffs.push(format!(
                        "walk {} loop entry {i} body {what}: oracle {exp:?}, door {got:?}",
                        &key[..12.min(key.len())]
                    ))
                }
            }
        }
        Some(())
    })();
    rx_ctx_free(c);
}
