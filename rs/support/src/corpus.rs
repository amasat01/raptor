//! The corpus-to-door helpers: a recorded walk (tagged JSON values) becomes value text, a dialect manifest and one `rx_make` per node.
//! A replay program is built on these; they call the door's C functions only.
use crate::safe::*;
use raptor_ir::record::Ctx;
use raptor_ir::value::{text_val, ttype_val};
use rcore::json::J;
use rcore::keytext::py_str;
use std::collections::HashMap;
use std::ffi::{CStr, CString};

/// A context for replaying recorded walks. Every recording predates the shared read-plane role, so the digests take the role of
/// a slot as the recording named it. For a walk recorded before the planes carried axes (`before_axes`), they also take the keys
/// without the fields the recording predates; otherwise the keys are the current ones.
pub fn replay_ctx(before_axes: bool) -> *mut Ctx {
    let c = rx_ctx_new();
    // SAFETY: `rx_ctx_new` returned a live context that nothing else holds yet.
    unsafe {
        (*c).role_form = Some(crate::migrate::old_role_name);
        (*c).key_form = Some(if before_axes {
            crate::migrate::remove_added_fields
        } else {
            crate::migrate::old_read_role
        });
    }
    c
}

/// Writes the recorded elements `v` between `open` and `close` as value text; `one` adds the comma of a one-element tuple.
pub fn seq(v: &[J], h: &[u32], open: &str, close: &str, one: bool) -> String {
    let items: Vec<String> = v.iter().map(|x| val(x, h)).collect();
    format!(
        "{open}{}{}{close}",
        items.join(", "),
        if one && items.len() == 1 { "," } else { "" }
    )
}

/// The corpus's tagged value as value text; `{"n": i}` becomes the handle made for node i.
pub fn val(j: &J, h: &[u32]) -> String {
    match j {
        J::Null => "None".into(),
        J::Bool(b) => if *b { "True" } else { "False" }.into(),
        J::Obj(_) => {
            if let Some(x) = j.get("i").or(j.get("f")) {
                return x.str().unwrap_or("?").into();
            }
            if let Some(x) = j.get("s") {
                return py_str(x.str().unwrap_or(""));
            }
            if let Some(x) = j.get("t") {
                return seq(x.arr(), h, "(", ")", true);
            }
            if let Some(x) = j.get("l").or(j.get("fs")) {
                return seq(x.arr(), h, "[", "]", false);
            }
            if let Some(x) = j.get("d") {
                let pairs: Vec<J> = x.arr().iter().map(|p| J::Arr(p.arr().to_vec())).collect();
                return seq(
                    &pairs
                        .iter()
                        .map(|p| J::Obj(vec![("t".into(), p.clone())]))
                        .collect::<Vec<_>>(),
                    h,
                    "[",
                    "]",
                    false,
                );
            }
            if let Some(x) = j.get("n") {
                return h
                    .get(x.int() as usize)
                    .map_or("#?".into(), |n| format!("#{n}"));
            }
            if let Some(r) = j.get("r") {
                let fs: Vec<String> = j
                    .get("v")
                    .map_or(&[][..], |v| v.obj())
                    .iter()
                    .map(|(k, x)| format!("{k}={}", val(x, h)))
                    .collect();
                return format!("{}({})", r.str().unwrap_or("?"), fs.join(", "));
            }
            "Unknown(?)".into()
        }
        _ => "Unknown(?)".into(),
    }
}

/// The door's last error text of the context.
pub fn err(c: *mut Ctx) -> String {
    unsafe { CStr::from_ptr(rx_last_error(c)) }
        .to_string_lossy()
        .into_owned()
}
/// The text of a recorded string value, or none when `j` is another kind.
pub fn tagged_str(j: &J) -> Option<&str> {
    j.get("s").and_then(J::str)
}

/// A tagged value as plain JSON text.
pub fn json_of(j: &J) -> String {
    match j {
        J::Null => "null".into(),
        J::Bool(b) => b.to_string(),
        J::Obj(_) => {
            if let Some(x) = j.get("i").or(j.get("f")) {
                return x.str().unwrap_or("0").into();
            }
            if let Some(x) = j.get("s") {
                return format!("{:?}", x.str().unwrap_or(""));
            }
            if let Some(x) = j.get("t").or(j.get("l")) {
                return format!(
                    "[{}]",
                    x.arr().iter().map(json_of).collect::<Vec<_>>().join(",")
                );
            }
            "null".into()
        }
        _ => "null".into(),
    }
}

/// One manifest per extension from the resolved declarations a walk carries (parents first), as JSON text.
pub fn manifests(inp: &J) -> Vec<String> {
    let mut by_ext: Vec<(String, String, Vec<String>)> = vec![];
    let mut seen: Vec<String> = vec![];
    for chain in inp.get("kinds").map_or(&[][..], |k| k.arr()) {
        for item in chain.arr().iter().rev() {
            let d = item.get("decl").and_then(|d| d.get("v")).unwrap();
            let g = |k: &str| d.get(k).map_or("null".to_string(), json_of);
            let name = d.get("name").and_then(tagged_str).unwrap_or("").to_string();
            if seen.contains(&name) {
                continue;
            }
            seen.push(name.clone());
            let ops = d.get("operands").unwrap().get("t").unwrap().arr();
            let hi = if matches!(ops[1], J::Null) {
                "\"many\"".to_string()
            } else {
                json_of(&ops[1])
            };
            let fields: Vec<String> = d
                .get("fields")
                .and_then(|f| f.get("t"))
                .map_or(&[][..], |f| f.arr())
                .iter()
                .map(|f| {
                    let t = f.get("t").unwrap().arr();
                    let req = matches!(t[3], J::Bool(true));
                    format!(
                        "{{\"name\":{},\"type\":{}{}}}",
                        json_of(&t[0]),
                        json_of(&t[1]),
                        if req {
                            String::new()
                        } else {
                            format!(",\"default\":{}", json_of(&t[2]))
                        }
                    )
                })
                .collect();
            let sp = d
                .get("spelling")
                .and_then(|x| x.get("t"))
                .map(|t| {
                    format!(
                        "{{\"dialect\":{},\"name\":{},\"args\":{}}}",
                        json_of(&t.arr()[0]),
                        json_of(&t.arr()[1]),
                        json_of(&t.arr()[2])
                    )
                })
                .unwrap_or("null".into());
            let row = format!("{{\"name\":{},\"family\":{},\"refines\":{},\"operands\":{{\"min\":{},\"max\":{hi},\"families\":{}}},\"fields\":[{}],\"result\":{},\"identity\":{},\"feathers\":{},\"spelling\":{sp},\"wire\":{},\"no_writer\":{},\"diff\":{}}}",
                g("name"), g("family"), g("refines"), json_of(&ops[0]), json_of(&ops[2]), fields.join(","), g("result"), g("identity"), g("feathers"), g("wire"), g("no_writer"), g("diff"));
            let ext = d.get("ext").and_then(tagged_str).unwrap_or("").to_string();
            let dialect = d
                .get("dialect")
                .and_then(tagged_str)
                .unwrap_or("")
                .to_string();
            match by_ext.iter_mut().find(|e| e.0 == ext) {
                Some(e) => e.2.push(row),
                None => by_ext.push((ext, dialect, vec![row])),
            }
        }
    }
    by_ext
        .iter()
        .map(|(e, d, rows)| {
            format!(
                "{{\"name\":{e:?},\"dialect\":{d:?},\"kinds\":[{}]}}",
                rows.join(",")
            )
        })
        .collect()
}

/// Declares each manifest in the context; the first refusal is returned with the door's error text.
pub fn declare(c: *mut Ctx, ms: &[String]) -> Result<(), String> {
    for m in ms {
        let ext = m.split('"').nth(3).unwrap_or("");
        let fp = CString::new(format!("({:?},)", ext)).unwrap();
        if rx_declare_dialect(c, m.as_ptr() as *const _, m.len(), fp.as_ptr()) != 0 {
            return Err(format!("declare: {}", err(c)));
        }
    }
    Ok(())
}

/// What the door gave back for the values that went in.
#[derive(Default)]
pub struct Vals {
    pub on: bool,
    pub total: usize,
    pub equal: usize,
    pub fixed: usize,
    pub hist: HashMap<String, [usize; 2]>,
    pub diffs: Vec<String>,
    pub f2_total: usize,
    pub f2_equal: usize,
}

/// Collects into `out` the kind names that the recorded value `j` mentions.
pub fn kinds_of(j: &J, out: &mut Vec<String>) {
    match j {
        J::Null => out.push("None".into()),
        J::Bool(_) => out.push("bool".into()),
        J::Obj(_) => {
            if j.get("i").is_some() {
                out.push("int".into())
            } else if j.get("f").is_some() {
                out.push("float".into())
            } else if j.get("s").is_some() {
                out.push("str".into())
            } else if j.get("n").is_some() {
                out.push("node reference".into())
            } else if let Some(r) = j.get("r") {
                out.push(format!("record:{}", r.str().unwrap_or("?")));
                for (_, x) in j.get("v").map_or(&[][..], |v| v.obj()) {
                    kinds_of(x, out)
                }
            } else if let Some(t) = j.get("t") {
                out.push("tuple".into());
                for x in t.arr() {
                    kinds_of(x, out)
                }
            } else if let Some(t) = j.get("l").or(j.get("fs")) {
                out.push("list".into());
                for x in t.arr() {
                    kinds_of(x, out)
                }
            } else if let Some(t) = j.get("d") {
                out.push("list".into());
                for x in t.arr() {
                    kinds_of(x, out)
                }
            } else {
                out.push("other".into())
            }
        }
        _ => out.push("other".into()),
    }
}

/// Reads a text from a door call that takes a buffer: asks for the length, then for the text.
pub fn text_out(
    c: *mut Ctx,
    f: impl Fn(*mut Ctx, *mut std::ffi::c_char, usize, *mut usize) -> i32,
) -> String {
    let mut len = 0usize;
    f(c, std::ptr::null_mut(), 0, &mut len);
    let mut buf = vec![0 as std::ffi::c_char; len + 1];
    f(c, buf.as_mut_ptr(), len + 1, &mut len);
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Makes every recorded node in `c`; with `vals.on`, reads each value back and compares it with the text that went in.
pub fn make_nodes(c: *mut Ctx, inp: &J, vals: &mut Vals) -> Result<Vec<u32>, String> {
    let mut h: Vec<u32> = vec![];
    for n in inp.get("nodes").unwrap().arr() {
        let (cname, v) = (n.get("c").unwrap().str().unwrap(), n.get("v").unwrap());
        let cls = CString::new(cname).unwrap();
        let tt = v.get("ttype").and_then(|t| t.get("v")).unwrap();
        let dtype = CString::new(tt.get("dtype").and_then(tagged_str).unwrap_or("")).unwrap();
        let shape: Vec<i64> = tt
            .get("shape")
            .and_then(|s| s.get("t"))
            .map_or(&[][..], |s| s.arr())
            .iter()
            .map(|d| {
                d.get("i")
                    .and_then(J::str)
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(-1)
            })
            .collect();
        let tag = tt
            .get("tag")
            .and_then(tagged_str)
            .map(|t| CString::new(t).unwrap());
        let ops: Vec<u32> = v
            .get("operands")
            .and_then(|o| o.get("t"))
            .map_or(&[][..], |o| o.arr())
            .iter()
            .map(|o| h[o.get("n").unwrap().int() as usize])
            .collect();
        let slots: Vec<(&String, &J, String)> = v
            .obj()
            .iter()
            .filter(|(k, _)| k != "ttype" && k != "operands")
            .map(|(k, x)| (k, x, val(x, &h)))
            .collect();
        let fields: Vec<String> = slots
            .iter()
            .map(|(k, _, t)| format!("({}, {})", py_str(k), t))
            .collect();
        let ft = format!(
            "({}{})",
            fields.join(", "),
            if fields.len() == 1 { "," } else { "" }
        );
        let mut o = 0u32;
        let st = rx_make(
            c,
            cls.as_ptr(),
            dtype.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            tag.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()),
            ops.as_ptr(),
            ops.len(),
            ft.as_ptr() as *const _,
            ft.len(),
            &mut o,
        );
        if st != 0 {
            return Err(err(c));
        }
        h.push(o);
        if !vals.on {
            continue;
        }
        let check = |what: &str, kinds: Vec<String>, want: String, got: String, vals: &mut Vals| {
            vals.total += 1;
            let eq = want == got;
            if eq {
                vals.equal += 1
            } else if vals.diffs.len() < 5 {
                vals.diffs
                    .push(format!("{cname}.{what}\n  passed: {want}\n  read:   {got}"))
            }
            for k in kinds {
                let e = vals.hist.entry(k).or_default();
                e[0] += 1;
                if eq {
                    e[1] += 1
                }
            }
        };
        for (k, x, want) in &slots {
            let f = CString::new(k.as_str()).unwrap();
            let got = text_out(c, |c, b, cap, l| rx_field(c, o, f.as_ptr(), b, cap, l));
            let mut kinds = vec![];
            kinds_of(x, &mut kinds);
            if *want != got && k.starts_with('_') {
                vals.fixed += 1;
                continue;
            }
            if got.contains('#') {
                // the same field with its references as indices: written back with the handles it names, it must equal the plain text
                let (mut buf, mut refs, mut len, mut nr) = (
                    vec![0 as std::ffi::c_char; got.len() + 64],
                    vec![0u32; 256],
                    0usize,
                    0usize,
                );
                let st = rx_field2(
                    c,
                    o,
                    f.as_ptr(),
                    buf.as_mut_ptr(),
                    buf.len(),
                    &mut len,
                    refs.as_mut_ptr(),
                    refs.len(),
                    &mut nr,
                );
                if st != 0 {
                    buf = vec![0; len + 1];
                    refs = vec![0; nr];
                    rx_field2(
                        c,
                        o,
                        f.as_ptr(),
                        buf.as_mut_ptr(),
                        buf.len(),
                        &mut len,
                        refs.as_mut_ptr(),
                        refs.len(),
                        &mut nr,
                    );
                }
                let t2 = unsafe { CStr::from_ptr(buf.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                vals.f2_total += 1;
                if unlift(&t2, &refs[..nr]) == got {
                    vals.f2_equal += 1
                } else if vals.diffs.len() < 5 {
                    vals.diffs.push(format!(
                        "{cname}.{k} (field2)\n  field:  {got}\n  field2: {t2} refs {:?}",
                        &refs[..nr]
                    ))
                }
            }
            check(k, kinds, want.clone(), got, vals);
        }
        let want_tt = text_val(&ttype_val(
            dtype.to_str().unwrap(),
            &shape,
            tag.as_ref().map(|t| t.to_str().unwrap()),
        ));
        check(
            "ttype",
            vec!["ttype".into()],
            want_tt,
            text_out(c, |c, b, cap, l| rx_ttype(c, o, b, cap, l)),
            vals,
        );
        let mut got = vec![0u32; ops.len() + 1];
        let mut cnt = 0usize;
        rx_operands(c, o, got.as_mut_ptr(), got.len(), &mut cnt);
        check(
            "operands",
            vec!["operands".into()],
            format!("{ops:?}"),
            format!("{:?}", &got[..cnt.min(got.len())]),
            vals,
        );
    }
    Ok(h)
}

/// Value text in which `#i` stands for `refs[i]`, written with the handles (a `#` inside a string is left alone).
pub fn unlift(t: &str, refs: &[u32]) -> String {
    let (mut o, mut q, mut esc, cs) = (
        String::new(),
        None::<char>,
        false,
        t.chars().collect::<Vec<_>>(),
    );
    let mut i = 0;
    while i < cs.len() {
        let ch = cs[i];
        if let Some(qc) = q {
            o.push(ch);
            if esc {
                esc = false
            } else if ch == '\\' {
                esc = true
            } else if ch == qc {
                q = None
            }
            i += 1;
            continue;
        }
        if ch == '\'' || ch == '"' {
            q = Some(ch);
            o.push(ch);
            i += 1;
            continue;
        }
        if ch == '#' {
            let d: String = cs[i + 1..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(k) = d.parse::<usize>() {
                o.push('#');
                o.push_str(&refs.get(k).map_or("?".to_string(), |r| r.to_string()));
                i += 1 + d.len();
                continue;
            }
        }
        o.push(ch);
        i += 1;
    }
    o
}

/// The options one recorded walk call was made with, as value text (`declared`, `exempt`) and the forced access class.
pub struct WalkOpts {
    pub declared: String,
    pub force: Option<String>,
    pub exempt: String,
}

/// The option sets a unit's walk was recorded under, by the walk's digest, in recorded order. "default" is the call with none.
/// Several sets can stand for one digest (the same walk was asked for twice with different options); the caller tries them in order.
pub fn walk_options(rec: &J, digest: &str, h: &[u32]) -> Vec<WalkOpts> {
    rec.get("walk_opts")
        .and_then(|w| w.get(digest))
        .map_or(&[][..], |l| l.arr())
        .iter()
        .filter_map(|e| {
            let o = e.get("opts")?;
            Some(match o {
                J::Str(_) => WalkOpts {
                    declared: "()".into(),
                    force: None,
                    exempt: "()".into(),
                },
                _ => WalkOpts {
                    declared: val(o.get("declared")?, h),
                    force: o.get("force").and_then(tagged_str).map(String::from),
                    exempt: val(o.get("exempt")?, h),
                },
            })
        })
        .collect()
}

/// A unit's inputs with the nodes (and declarations) its recorded analyses name added after its own: the analyses refer to them by
/// the numbers that continue the unit's table.
pub fn with_analysis_nodes(inp: &J, rec: &J) -> J {
    let inp = &unwrap_unit_proxy(inp);
    let extra = match rec.get("analysis_extra") {
        Some(e) => e,
        None => return inp.clone(),
    };
    let (nodes, kinds) = (
        extra.get("nodes").map_or(&[][..], |n| n.arr()),
        extra.get("kinds").map_or(&[][..], |k| k.arr()),
    );
    if nodes.is_empty() && kinds.is_empty() {
        return inp.clone();
    }
    J::Obj(
        inp.obj()
            .iter()
            .map(|(k, v)| match k.as_str() {
                "nodes" => (
                    k.clone(),
                    J::Arr(v.arr().iter().chain(nodes).cloned().collect()),
                ),
                "kinds" => (
                    k.clone(),
                    J::Arr(v.arr().iter().chain(kinds).cloned().collect()),
                ),
                _ => (k.clone(), v.clone()),
            })
            .collect(),
    )
}

/// A lowered unit the recorder met as a wrapper (`{"proxy": ..., "kernel": <stage record>,
/// "walk": ...}`) is its kernel with the wrapper's walk: the inputs come back with
/// `lowered.kernel` written as that kernel record.
fn unwrap_unit_proxy(inp: &J) -> J {
    let wrapped = |lowered: &J| {
        lowered
            .get("v")
            .and_then(|v| v.get("kernel"))
            .filter(|k| k.get("proxy").is_some())
            .cloned()
    };
    let Some(wrapper) = inp.get("lowered").and_then(wrapped) else {
        return inp.clone();
    };
    let (Some(stage), walk) = (
        wrapper.get("kernel").and_then(|k| k.get("v")),
        wrapper.get("walk"),
    ) else {
        return inp.clone();
    };
    let fields: Vec<(String, J)> = stage
        .obj()
        .iter()
        .filter(|(n, _)| n != "walk")
        .cloned()
        .chain(walk.map(|w| ("walk".to_string(), w.clone())))
        .collect();
    let kernel = J::Obj(vec![
        ("r".into(), J::Str("Kernel".into())),
        ("v".into(), J::Obj(fields)),
    ]);
    let replace_kernel = |lowered: &J| match lowered {
        J::Obj(top) => J::Obj(
            top.iter()
                .map(|(k, v)| match (k.as_str(), v) {
                    ("v", J::Obj(inner)) => (
                        k.clone(),
                        J::Obj(
                            inner
                                .iter()
                                .map(|(n, x)| {
                                    if n == "kernel" {
                                        (n.clone(), kernel.clone())
                                    } else {
                                        (n.clone(), x.clone())
                                    }
                                })
                                .collect(),
                        ),
                    ),
                    _ => (k.clone(), v.clone()),
                })
                .collect(),
        ),
        other => other.clone(),
    };
    J::Obj(
        inp.obj()
            .iter()
            .map(|(k, v)| {
                if k == "lowered" {
                    (k.clone(), replace_kernel(v))
                } else {
                    (k.clone(), v.clone())
                }
            })
            .collect(),
    )
}
