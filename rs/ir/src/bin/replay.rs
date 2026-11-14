//! replay <walks.jsonl> [--limit n] [--only <substring of a test id>] [--only-exact <test id>] [--show <k>] [--roundtrip] [--values] [--cost <k>]
//! Rebuilds each recorded walk in a fresh context through the C door and compares the digest, arg_spec and access with the recorded ones.
use raptor_ir::canon::{rx_canonical, rx_digest, rx_walk_field};
use raptor_ir::door::{rx_ctx_free, rx_ctx_new, rx_field, rx_last_error, rx_make, rx_operands, rx_ttype};
use raptor_ir::value::{text_val, ttype_val, Val};
use raptor_ir::dialect::rx_declare_dialect;
use raptor_ir::dump::{rx_dump, rx_load};
use raptor_ir::record::Ctx;
use rcore::json::{self, J};
use rcore::keytext::py_str;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io::BufRead;
use std::time::Instant;

fn seq(v: &[J], h: &[u32], open: &str, close: &str, one: bool) -> String {
    let items: Vec<String> = v.iter().map(|x| val(x, h)).collect();
    format!("{open}{}{}{close}", items.join(", "), if one && items.len() == 1 { "," } else { "" })
}

/// The corpus's tagged value as value text; `{"n": i}` becomes the handle made for node i.
fn val(j: &J, h: &[u32]) -> String {
    match j {
        J::Null => "None".into(),
        J::Bool(b) => if *b { "True" } else { "False" }.into(),
        J::Obj(_) => {
            if let Some(x) = j.get("i").or(j.get("f")) { return x.str().unwrap_or("?").into() }
            if let Some(x) = j.get("s") { return py_str(x.str().unwrap_or("")) }
            if let Some(x) = j.get("t") { return seq(x.arr(), h, "(", ")", true) }
            if let Some(x) = j.get("l").or(j.get("fs")) { return seq(x.arr(), h, "[", "]", false) }
            if let Some(x) = j.get("d") { let pairs: Vec<J> = x.arr().iter().map(|p| J::Arr(p.arr().to_vec())).collect(); return seq(&pairs.iter().map(|p| J::Obj(vec![("t".into(), p.clone())])).collect::<Vec<_>>(), h, "[", "]", false) }
            if let Some(x) = j.get("n") { return h.get(x.int() as usize).map_or("#?".into(), |n| format!("#{n}")) }
            if let Some(r) = j.get("r") {
                let fs: Vec<String> = j.get("v").map_or(&[][..], |v| v.obj()).iter().map(|(k, x)| format!("{k}={}", val(x, h))).collect();
                return format!("{}({})", r.str().unwrap_or("?"), fs.join(", "));
            }
            "Unknown(?)".into()
        }
        _ => "Unknown(?)".into(),
    }
}

fn err(c: *mut Ctx) -> String { unsafe { CStr::from_ptr(rx_last_error(c)) }.to_string_lossy().into_owned() }
fn field(c: *mut Ctx, w: u32, f: &str) -> String {
    let f = CString::new(f).unwrap();
    let mut len = 0usize;
    rx_walk_field(c, w, f.as_ptr(), std::ptr::null_mut(), 0, &mut len);
    let mut buf = vec![0 as std::ffi::c_char; len + 1];
    rx_walk_field(c, w, f.as_ptr(), buf.as_mut_ptr(), len + 1, &mut len);
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

fn tagged_str(j: &J) -> Option<&str> { j.get("s").and_then(J::str) }

/// A tagged value as plain JSON text.
fn json_of(j: &J) -> String {
    match j {
        J::Null => "null".into(), J::Bool(b) => b.to_string(),
        J::Obj(_) => {
            if let Some(x) = j.get("i").or(j.get("f")) { return x.str().unwrap_or("0").into() }
            if let Some(x) = j.get("s") { return format!("{:?}", x.str().unwrap_or("")) }
            if let Some(x) = j.get("t").or(j.get("l")) { return format!("[{}]", x.arr().iter().map(json_of).collect::<Vec<_>>().join(",")) }
            "null".into()
        }
        _ => "null".into(),
    }
}

/// One manifest per extension from the resolved declarations a walk carries (parents first), as JSON text.
fn manifests(inp: &J) -> Vec<String> {
    let mut by_ext: Vec<(String, String, Vec<String>)> = vec![];
    let mut seen: Vec<String> = vec![];
    for chain in inp.get("kinds").map_or(&[][..], |k| k.arr()) {
        for item in chain.arr().iter().rev() {
            let d = item.get("decl").and_then(|d| d.get("v")).unwrap();
            let g = |k: &str| d.get(k).map_or("null".to_string(), json_of);
            let name = d.get("name").and_then(tagged_str).unwrap_or("").to_string();
            if seen.contains(&name) { continue }
            seen.push(name.clone());
            let ops = d.get("operands").unwrap().get("t").unwrap().arr();
            let hi = if matches!(ops[1], J::Null) { "\"many\"".to_string() } else { json_of(&ops[1]) };
            let fields: Vec<String> = d.get("fields").and_then(|f| f.get("t")).map_or(&[][..], |f| f.arr()).iter().map(|f| {
                let t = f.get("t").unwrap().arr();
                let req = matches!(t[3], J::Bool(true));
                format!("{{\"name\":{},\"type\":{}{}}}", json_of(&t[0]), json_of(&t[1]), if req { String::new() } else { format!(",\"default\":{}", json_of(&t[2])) })
            }).collect();
            let sp = d.get("spelling").and_then(|x| x.get("t")).map(|t| format!("{{\"dialect\":{},\"name\":{},\"args\":{}}}", json_of(&t.arr()[0]), json_of(&t.arr()[1]), json_of(&t.arr()[2]))).unwrap_or("null".into());
            let row = format!("{{\"name\":{},\"family\":{},\"refines\":{},\"operands\":{{\"min\":{},\"max\":{hi},\"families\":{}}},\"fields\":[{}],\"result\":{},\"identity\":{},\"feathers\":{},\"spelling\":{sp},\"wire\":{},\"no_writer\":{},\"diff\":{}}}",
                g("name"), g("family"), g("refines"), json_of(&ops[0]), json_of(&ops[2]), fields.join(","), g("result"), g("identity"), g("feathers"), g("wire"), g("no_writer"), g("diff"));
            let ext = d.get("ext").and_then(tagged_str).unwrap_or("").to_string();
            let dialect = d.get("dialect").and_then(tagged_str).unwrap_or("").to_string();
            match by_ext.iter_mut().find(|e| e.0 == ext) { Some(e) => e.2.push(row), None => by_ext.push((ext, dialect, vec![row])) }
        }
    }
    by_ext.iter().map(|(e, d, rows)| format!("{{\"name\":{e:?},\"dialect\":{d:?},\"kinds\":[{}]}}", rows.join(","))).collect()
}

fn declare(c: *mut Ctx, ms: &[String]) -> Result<(), String> {
    for m in ms {
        let ext = m.split('"').nth(3).unwrap_or("");
        let fp = CString::new(format!("({:?},)", ext)).unwrap();
        if rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), fp.as_ptr()) != 0 { return Err(format!("declare: {}", err(c))) }
    }
    Ok(())
}

/// The features a recorded walk holds, whatever the door supports.
fn holds(inp: &J) -> Vec<&'static str> {
    let mut h = vec![];
    let nodes = inp.get("nodes").map_or(&[][..], |n| n.arr());
    let class = |n: &J| n.get("c").and_then(J::str).unwrap_or("").to_string();
    if nodes.iter().any(|n| { let c = class(n); c.starts_with("Loop") || c == "TapeRead" || c == "TapeWrite" }) { h.push("loops") }
    if nodes.iter().any(|n| n.get("v").and_then(|v| v.get("quantity")).is_some_and(|q| !matches!(q, J::Null))) { h.push("quantities") }
    if inp.get("declared").and_then(|d| d.get("l")).is_some_and(|l| !l.arr().is_empty()) { h.push("declared") }
    if !inp.get("kinds").map_or(&[][..], |k| k.arr()).is_empty() { h.push("dialect") }
    if nodes.iter().any(|n| n.get("v").and_then(|v| v.get("kind")).and_then(tagged_str) == Some("raw_device")) { h.push("raw blocks") }
    h
}

/// What the door gave back for the values that went in.
#[derive(Default)]
struct Vals { on: bool, total: usize, equal: usize, fixed: usize, hist: HashMap<String, [usize; 2]>, diffs: Vec<String> }

fn kinds_of(j: &J, out: &mut Vec<String>) {
    match j {
        J::Null => out.push("None".into()), J::Bool(_) => out.push("bool".into()),
        J::Obj(_) => {
            if j.get("i").is_some() { out.push("int".into()) } else if j.get("f").is_some() { out.push("float".into()) } else if j.get("s").is_some() { out.push("str".into()) }
            else if j.get("n").is_some() { out.push("node reference".into()) }
            else if let Some(r) = j.get("r") { out.push(format!("record:{}", r.str().unwrap_or("?"))); for (_, x) in j.get("v").map_or(&[][..], |v| v.obj()) { kinds_of(x, out) } }
            else if let Some(t) = j.get("t") { out.push("tuple".into()); for x in t.arr() { kinds_of(x, out) } }
            else if let Some(t) = j.get("l").or(j.get("fs")) { out.push("list".into()); for x in t.arr() { kinds_of(x, out) } }
            else if let Some(t) = j.get("d") { out.push("list".into()); for x in t.arr() { kinds_of(x, out) } }
            else { out.push("other".into()) }
        }
        _ => out.push("other".into()),
    }
}

fn text_out(c: *mut Ctx, f: impl Fn(*mut Ctx, *mut std::ffi::c_char, usize, *mut usize) -> i32) -> String {
    let mut len = 0usize;
    f(c, std::ptr::null_mut(), 0, &mut len);
    let mut buf = vec![0 as std::ffi::c_char; len + 1];
    f(c, buf.as_mut_ptr(), len + 1, &mut len);
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

/// Makes every recorded node in `c`; with `vals.on`, reads each value back and compares it with the text that went in.
fn make_nodes(c: *mut Ctx, inp: &J, vals: &mut Vals) -> Result<Vec<u32>, String> {
    let mut h: Vec<u32> = vec![];
    for n in inp.get("nodes").unwrap().arr() {
        let (cname, v) = (n.get("c").unwrap().str().unwrap(), n.get("v").unwrap());
        let cls = CString::new(cname).unwrap();
        let tt = v.get("ttype").and_then(|t| t.get("v")).unwrap();
        let dtype = CString::new(tt.get("dtype").and_then(tagged_str).unwrap_or("")).unwrap();
        let shape: Vec<i64> = tt.get("shape").and_then(|s| s.get("t")).map_or(&[][..], |s| s.arr()).iter().map(|d| d.get("i").and_then(J::str).and_then(|x| x.parse().ok()).unwrap_or(-1)).collect();
        let tag = tt.get("tag").and_then(tagged_str).map(|t| CString::new(t).unwrap());
        let ops: Vec<u32> = v.get("operands").and_then(|o| o.get("t")).map_or(&[][..], |o| o.arr()).iter().map(|o| h[o.get("n").unwrap().int() as usize]).collect();
        let slots: Vec<(&String, &J, String)> = v.obj().iter().filter(|(k, _)| k != "ttype" && k != "operands").map(|(k, x)| (k, x, val(x, &h))).collect();
        let fields: Vec<String> = slots.iter().map(|(k, _, t)| format!("({}, {})", py_str(k), t)).collect();
        let ft = format!("({}{})", fields.join(", "), if fields.len() == 1 { "," } else { "" });
        let mut o = 0u32;
        let st = rx_make(c, cls.as_ptr(), dtype.as_ptr(), shape.as_ptr(), shape.len(), tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
                         ops.as_ptr(), ops.len(), ft.as_ptr() as *const _, ft.len(), &mut o);
        if st != 0 { return Err(err(c)) }
        h.push(o);
        if !vals.on { continue }
        let mut check = |what: &str, kinds: Vec<String>, want: String, got: String, vals: &mut Vals| {
            vals.total += 1;
            let eq = want == got;
            if eq { vals.equal += 1 } else if vals.diffs.len() < 5 { vals.diffs.push(format!("{cname}.{what}\n  passed: {want}\n  read:   {got}")) }
            for k in kinds { let e = vals.hist.entry(k).or_default(); e[0] += 1; if eq { e[1] += 1 } }
        };
        for (k, x, want) in &slots {
            let f = CString::new(k.as_str()).unwrap();
            let got = text_out(c, |c, b, cap, l| rx_field(c, o, f.as_ptr(), b, cap, l));
            let mut kinds = vec![];
            kinds_of(x, &mut kinds);
            if *want != got && k.starts_with('_') { vals.fixed += 1; continue }
            check(k, kinds, want.clone(), got, vals);
        }
        let want_tt = text_val(&ttype_val(dtype.to_str().unwrap(), &shape, tag.as_ref().map(|t| t.to_str().unwrap())));
        check("ttype", vec!["ttype".into()], want_tt, text_out(c, |c, b, cap, l| rx_ttype(c, o, b, cap, l)), vals);
        let mut got = vec![0u32; ops.len() + 1];
        let mut cnt = 0usize;
        rx_operands(c, o, got.as_mut_ptr(), got.len(), &mut cnt);
        check("operands", vec!["operands".into()], format!("{ops:?}"), format!("{:?}", &got[..cnt.min(got.len())]), vals);
    }
    Ok(h)
}

enum Outcome { Equal, Mismatch(String), Skip(String), MakeRefused(String), Raised(bool, String) }

fn replay(rec: &J, counts: &mut [f64; 3], roundtrip: bool, rt: &mut [usize; 2], vals: &mut Vals) -> Outcome {
    let inp = rec.get("inputs").unwrap();
    let ms = manifests(inp);
    let t0 = Instant::now();
    let c = rx_ctx_new();
    counts[0] += t0.elapsed().as_secs_f64();
    let mut c2 = std::ptr::null_mut();
    let out = (|| {
        if let Err(e) = declare(c, &ms) { return Outcome::MakeRefused(e) }
        let h = match make_nodes(c, inp, vals) { Ok(h) => h, Err(e) => return Outcome::Raised(true, e) };
        let sinks: Vec<u32> = inp.get("sinks").unwrap().arr().iter().map(|s| h[s.get("n").unwrap().int() as usize]).collect();
        let declared = CString::new(val(inp.get("declared").unwrap(), &h)).unwrap();
        let exempt = CString::new(val(inp.get("exempt").unwrap(), &h)).unwrap();
        let force = inp.get("force").and_then(tagged_str).map(|f| CString::new(f).unwrap());
        let mut w = 0u32;
        let st = rx_canonical(c, sinks.as_ptr(), sinks.len(), declared.as_ptr(), force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()), exempt.as_ptr(), &mut w);
        if st != 0 {
            let e = err(c);
            return match e.strip_prefix("not in this build: ") { Some(f) => Outcome::Skip(f.into()), None => Outcome::Raised(false, e) };
        }
        let mut dg = [0 as std::ffi::c_char; 65];
        rx_digest(c, w, dg.as_mut_ptr());
        let dg = unsafe { CStr::from_ptr(dg.as_ptr()) }.to_string_lossy().into_owned();
        let res = rec.get("result").unwrap();
        let (want_dg, want_spec, want_acc) = (res.get("digest").and_then(J::str).unwrap_or(""), val(&J::Obj(vec![("t".into(), res.get("arg_spec").and_then(|a| a.get("l")).cloned().unwrap_or(J::Arr(vec![])))]), &h), val(res.get("access").unwrap(), &h));
        let mut bad = vec![];
        if dg != want_dg {
            bad.push("digest".to_string());
            let (got, want) = (field(c, w, "preimage"), rec.get("preimage").and_then(J::str).unwrap_or("").to_string());
            let (g, wl): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
            let i = (0..g.len().max(wl.len())).find(|&i| g.get(i) != wl.get(i)).unwrap_or(0);
            bad.push(format!("first differing line {i}: recorded {:?} computed {:?}", wl.get(i), g.get(i)));
        }
        // the recording holds the sequence as a list; it is compared as the tuple the door writes
        if field(c, w, "arg_spec") != want_spec { bad.push("arg_spec".into()) }
        if field(c, w, "access") != want_acc { bad.push("access".into()) }
        if bad.is_empty() && roundtrip {
            rt[1] += 1;
            let mut len = 0usize;
            rx_dump(c, sinks.as_ptr(), sinks.len(), std::ptr::null_mut(), 0, &mut len);
            let mut buf = vec![0 as std::ffi::c_char; len + 1];
            if rx_dump(c, sinks.as_ptr(), sinks.len(), buf.as_mut_ptr(), len + 1, &mut len) != 0 { return Outcome::Mismatch(format!("roundtrip: dump: {}", err(c))) }
            c2 = rx_ctx_new();
            if let Err(e) = declare(c2, &ms) { return Outcome::Mismatch(format!("roundtrip: {e}")) }
            let mut roots = vec![0u32; sinks.len() + 8];
            let mut nr = 0usize;
            if rx_load(c2, buf.as_ptr(), len, roots.as_mut_ptr(), roots.len(), &mut nr) != 0 { return Outcome::Mismatch(format!("roundtrip: load: {}", err(c2))) }
            let mut w2 = 0u32;
            if rx_canonical(c2, roots.as_ptr(), nr, declared.as_ptr(), force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()), exempt.as_ptr(), &mut w2) != 0 { return Outcome::Mismatch(format!("roundtrip: canonical: {}", err(c2))) }
            let mut d2 = [0 as std::ffi::c_char; 65];
            rx_digest(c2, w2, d2.as_mut_ptr());
            if unsafe { CStr::from_ptr(d2.as_ptr()) }.to_string_lossy() != dg { return Outcome::Mismatch("roundtrip: digest".into()) }
            rt[0] += 1;
        }
        if bad.is_empty() { Outcome::Equal } else { Outcome::Mismatch(bad.join("; ")) }
    })();
    rx_ctx_free(c);
    if !c2.is_null() { rx_ctx_free(c2) }
    counts[1] += t0.elapsed().as_secs_f64();
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: replay <walks.jsonl> [--limit n] [--only s] [--show k]");
    let arg = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1));
    let limit: usize = arg("--limit").and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let show: usize = arg("--show").and_then(|x| x.parse().ok()).unwrap_or(3);
    let only = arg("--only");
    let roundtrip = a.iter().any(|x| x == "--roundtrip");
    let mut rt = [0usize; 2];
    let mut vals = Vals { on: a.iter().any(|x| x == "--values"), ..Default::default() };
    if let Some(k) = arg("--cost").and_then(|x| x.parse().ok()) { cost(path, k); return }
    let exact = arg("--only-exact");
    let mut holding: HashMap<&str, usize> = HashMap::new();
    let start = Instant::now();
    let (mut u, mut equal, mut raised, mut made_bad, mut bad) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut spec_eq, mut acc_eq, mut dg_eq) = (0usize, 0usize, 0usize);
    let mut skips: HashMap<String, usize> = HashMap::new();
    let mut refused: HashMap<String, usize> = HashMap::new();
    let mut shown = 0;
    let (mut ref_eq, mut diffs) = (0usize, Vec::<String>::new());
    let (mut make_eq, mut make_n, mut canon_n) = (0usize, 0usize, 0usize);
    let mut counts = [0f64; 3];
    for line in std::io::BufReader::new(std::fs::File::open(path).expect("open")).lines() {
        if u >= limit { break }
        let line = line.unwrap();
        let rec = json::parse(&line).expect("json");
        let tests: Vec<String> = rec.get("tests").map_or(&[][..], |t| t.arr()).iter().filter_map(|t| t.str().map(String::from)).collect();
        if let Some(o) = exact { if !tests.iter().any(|t| t == o) { continue } }
        if let Some(o) = only { if !tests.iter().any(|t| t.contains(o.as_str())) { continue } }
        u += 1;
        let want_err = match rec.get("error") { None | Some(J::Null) => None, Some(e) => Some(e.get("msg").and_then(J::str).unwrap_or("").to_string()) };
        for f in holds(rec.get("inputs").unwrap()) { *holding.entry(f).or_default() += 1 }
        let test = tests.first().map_or("?", |t| t.as_str());
        match replay(&rec, &mut counts, roundtrip, &mut rt, &mut vals) {
            Outcome::Raised(at_make, m) => match &want_err {
                Some(w) => { raised += 1; if at_make { make_n += 1 } else { canon_n += 1 }
                    // a refusal at make is "<kind>: <text>"; the oracle's walk verifier wrote "node <i> (<kind>): <text>"
                    let rewritten = w.strip_prefix("node ").and_then(|r| r.split_once(" (")).and_then(|(_, r)| r.split_once("): ")).map(|(k, t)| format!("{k}: {t}"));
                    if at_make && (*w == m || rewritten.as_deref() == Some(m.as_str())) { make_eq += 1 } else if !at_make && *w == m { ref_eq += 1 } else { diffs.push(format!("{test}\n  recorded: {w:?}\n  computed: {m:?}")) } }
                None if at_make => { made_bad += 1; *refused.entry(m).or_default() += 1 }
                None => { bad += 1; if shown < show { shown += 1; println!("MISMATCH {test} : canonical refused: {m}") } }
            },
            _ if want_err.is_some() => { raised += 1; diffs.push(format!("{test}\n  recorded: {:?}\n  computed: no refusal", want_err.unwrap())) }
            Outcome::Equal => { equal += 1; dg_eq += 1; spec_eq += 1; acc_eq += 1 }
            Outcome::Skip(f) => *skips.entry(f).or_default() += 1,
            Outcome::Raised(..) => {}
            Outcome::MakeRefused(m) => { made_bad += 1; *refused.entry(m).or_default() += 1 }
            Outcome::Mismatch(m) => {
                bad += 1;
                if !m.contains("digest") { dg_eq += 1 }
                if !m.contains("arg_spec") { spec_eq += 1 }
                if !m.contains("access") { acc_eq += 1 }
                if shown < show { shown += 1; println!("MISMATCH {} : {m}", tests.first().map_or("?", |t| t.as_str())) }
            }
        }
    }
    let ran = equal + bad;
    let sk = |k: &str| skips.get(k).copied().unwrap_or(0);
    println!("digest-equal {dg_eq} / {u}");
    println!("arg_spec-equal {spec_eq} / {ran}");
    println!("access-equal {acc_eq} / {ran}");
    let mut other: Vec<_> = skips.iter().filter(|(k, _)| !["loops", "quantities", "declared slots", "dialect"].contains(&k.as_str())).collect();
    other.sort();
    println!("skipped by feature: loops {}, quantities {}, declared {}, dialect {}, raised-in-oracle {raised}{}", sk("loops"), sk("quantities"), sk("declared slots"), sk("dialect"),
             other.iter().map(|(k, n)| format!(", {k} {n}")).collect::<String>());
    println!("walks holding: {}", ["loops", "quantities", "declared", "dialect", "raw blocks"].iter().map(|k| format!("{k} {}", holding.get(k).copied().unwrap_or(0))).collect::<Vec<_>>().join(", "));
    if roundtrip { println!("roundtrip-equal {} / {}", rt[0], rt[1]) }
    println!("refusal-equal {ref_eq} / {canon_n}");
    println!("refused-at-make-equal {make_eq} / {make_n}");
    for d in &diffs { println!("REFUSAL DIFFERS {d}") }
    println!("make-refused {made_bad}");
    let mut top: Vec<_> = refused.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (m, n) in top.iter().take(5) { println!("  {n} x {m}") }
    println!("mismatched {bad}");
    if vals.on {
        println!("values read back equal {} / {}", vals.equal, vals.total);
        println!("fixed-or-dropped {}", vals.fixed);
        let mut hist: Vec<_> = vals.hist.iter().collect();
        hist.sort();
        for (k, [n, e]) in hist { println!("  {k}: {n} (in equal slots {e})") }
        for d in &vals.diffs { println!("DIFFERENCE {d}") }
    }
    println!("timing: total {:.2} s, rx_ctx_new {:.1} us mean, {:.1} us per walk", start.elapsed().as_secs_f64(), counts[0] / ran.max(1) as f64 * 1e6, counts[1] / ran.max(1) as f64 * 1e6);
    std::process::exit(if bad == 0 && made_bad == 0 { 0 } else { 1 });
}

fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

/// The cost of crossing the door for the k largest walks: dump, load into a new context, canonical there (median of 5).
fn cost(path: &str, k: usize) {
    let mut top: Vec<(usize, String)> = vec![];
    for line in std::io::BufReader::new(std::fs::File::open(path).expect("open")).lines() {
        let line = line.unwrap();
        let rec = json::parse(&line).expect("json");
        if !matches!(rec.get("error"), None | Some(J::Null)) { continue }
        let n = rec.get("inputs").unwrap().get("nodes").map_or(0, |n| n.arr().len());
        if top.len() < k || n > top.last().unwrap().0 { top.push((n, line)); top.sort_by(|a, b| b.0.cmp(&a.0)); top.truncate(k) }
    }
    println!("{:>8} {:>11} {:>9} {:>9} {:>9} {:>8}  test", "nodes", "dump bytes", "dump ms", "load ms", "canon ms", "ctx ms");
    let mut cols: [Vec<f64>; 6] = Default::default();
    for (n, line) in &top {
        let rec = json::parse(line).unwrap();
        let inp = rec.get("inputs").unwrap();
        let ms = manifests(inp);
        let c = rx_ctx_new();
        declare(c, &ms).unwrap();
        let h = make_nodes(c, inp, &mut Vals::default()).expect("make");
        let sinks: Vec<u32> = inp.get("sinks").unwrap().arr().iter().map(|s| h[s.get("n").unwrap().int() as usize]).collect();
        let declared = CString::new(val(inp.get("declared").unwrap(), &h)).unwrap();
        let exempt = CString::new(val(inp.get("exempt").unwrap(), &h)).unwrap();
        let force = inp.get("force").and_then(tagged_str).map(|f| CString::new(f).unwrap());
        let (mut dump_t, mut load_t, mut canon_t, mut ctx_t) = (vec![], vec![], vec![], vec![]);
        let mut bytes = 0usize;
        for _ in 0..5 {
            let mut len = 0usize;
            rx_dump(c, sinks.as_ptr(), sinks.len(), std::ptr::null_mut(), 0, &mut len);
            let mut buf = vec![0 as std::ffi::c_char; len + 1];
            let t = Instant::now();
            assert_eq!(rx_dump(c, sinks.as_ptr(), sinks.len(), buf.as_mut_ptr(), len + 1, &mut len), 0, "{}", err(c));
            dump_t.push(t.elapsed().as_secs_f64() * 1e3);
            bytes = len;
            let t = Instant::now();
            let c2 = rx_ctx_new();
            ctx_t.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            declare(c2, &ms).unwrap();
            let mut roots = vec![0u32; sinks.len() + 8];
            let mut nr = 0usize;
            assert_eq!(rx_load(c2, buf.as_ptr(), len, roots.as_mut_ptr(), roots.len(), &mut nr), 0, "{}", err(c2));
            load_t.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let mut w = 0u32;
            assert_eq!(rx_canonical(c2, roots.as_ptr(), nr, declared.as_ptr(), force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()), exempt.as_ptr(), &mut w), 0, "{}", err(c2));
            canon_t.push(t.elapsed().as_secs_f64() * 1e3);
            rx_ctx_free(c2);
        }
        rx_ctx_free(c);
        let id = rec.get("tests").and_then(|t| t.arr().first()).and_then(J::str).unwrap_or("?");
        let id: String = id.chars().take(60).collect();
        let row = [*n as f64, bytes as f64, median(dump_t), median(load_t), median(canon_t), median(ctx_t)];
        println!("{:>8} {:>11} {:>9.3} {:>9.3} {:>9.3} {:>8.3}  {id}", n, bytes, row[2], row[3], row[4], row[5]);
        for (i, v) in row.iter().enumerate() { cols[i].push(*v) }
    }
    let m: Vec<f64> = cols.iter().map(|c| median(c.clone())).collect();
    println!("{:>8} {:>11} {:>9.3} {:>9.3} {:>9.3} {:>8.3}  median over {} walks", m[0], m[1], m[2], m[3], m[4], m[5], top.len());
}
