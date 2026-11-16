//! The analyses of the IR the writers ask about a lowered walk, behind a C door (include/raptor_wings.h). Each answer is a pure function of the
//! context's nodes and the walk; vocabularies come from the core manifest's constants.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
pub mod blocks;
pub mod blockteam;
pub mod boxing;
pub mod cells;
pub mod chain;
pub mod construct;
pub mod declare;
pub mod lanes;
pub mod loops;
pub mod lstsq;
pub mod place;
pub mod plan;
pub mod planes;
pub mod rewrite;
pub mod scatter;
pub mod segment;
pub mod slots;
pub mod slotwise;
pub mod split;
pub mod stages;
pub mod team;
pub mod wide;
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::ffi::{c_char, c_int, CStr};

/// A list of the core manifest's constants.
fn konst(name: &str) -> &'static [&'static str] {
    rcore::CONSTS
        .iter()
        .find(|c| c.0 == name)
        .map_or(&[][..], |c| c.1)
}
/// The text constant of the core manifest with that name, or an empty text.
fn kstr(name: &str) -> &'static str {
    rcore::STR_CONSTS
        .iter()
        .find(|c| c.0 == name)
        .map_or("", |c| c.1)
}

/// Elementwise unary ops that are one hardware instruction.
const CHEAP_UNARY: [&str; 11] = [
    "neg", "abs", "sqrt", "rsqrt", "floor", "ceil", "trunc", "round", "rint", "sign", "spacing",
];
/// Elementwise binaries that call into the math library.
const LIBRARY_BINARY: [&str; 5] = ["pow", "atan2", "hypot", "fmod", "remainder"];
/// Reductions over one vector.
const REDUCTIONS: [&str; 3] = ["sum", "dot", "norm"];
/// The kind of an op that places a value in a plane.
const RESIDENT_KIND: &str = "resident";
/// The name prefix of a plane the runtime made (a scratch plane), by which such planes are told
/// from a mutable's own. Marking them with a field of the leaf would change the leaf and so the
/// digests of every recorded walk and placement comparison; the prefix stays until a recording
/// made with such a field exists.
pub(crate) const PLANE_PREFIX: &str = "hwd";

/// The result of an analysis: its answer, or the refusal that stopped it.
type R<T> = Result<T, Fault>;

/// Whether the node is of the class or a refinement of it.
pub(crate) fn is_a(c: &Ctx, h: u32, class: &str) -> bool {
    c.table.is_a(c.nodes[h as usize].class, class)
}

/// The kind of an Op node (any refinement of Op), else None.
pub(crate) fn op_kind(c: &Ctx, h: u32) -> R<Option<String>> {
    c.node(h)?;
    if is_a(c, h, "Op") {
        c.str_field(h, "kind").map(Some)
    } else {
        Ok(None)
    }
}

/// Whether the node is an op whose kind passes the test.
pub(crate) fn in_kinds(c: &Ctx, h: u32, kinds: impl Fn(&str) -> bool) -> R<bool> {
    Ok(op_kind(c, h)?.is_some_and(|k| kinds(&k)))
}

/// The nodes and, after each loop, everything inside its body.
pub(crate) fn flat(c: &Ctx, nodes: &[u32], out: &mut Vec<u32>) -> R<()> {
    for &n in nodes {
        c.node(n)?;
        out.push(n);
        if is_a(c, n, "Loop") {
            flat(c, &c.body_nodes(n)?, out)?
        }
    }
    Ok(())
}

/// The index of a read before its clamp.
pub(crate) fn raw_index(ctx: &Ctx, read: u32) -> u32 {
    let index = ctx.nodes[read as usize].ops[1];
    let clamped = in_kinds(ctx, index, |kind| kind == "clamp_index").unwrap_or(false);
    if clamped {
        ctx.nodes[index as usize].ops[0]
    } else {
        index
    }
}

/// The nodes of the walk in its order.
pub(crate) fn walk_order(c: &Ctx, w: u32) -> R<Vec<u32>> {
    Ok(c.walk(w)?.order.clone())
}

/// Whether the lanes of a tile may run side by side over the walk.
fn lane_form(c: &Ctx, w: u32) -> R<bool> {
    match c.walk(w)?.access.as_str() {
        "sample_local" => Ok(true),
        "cross_sample_read" => {
            let mut all = vec![];
            flat(c, &walk_order(c, w)?, &mut all)?;
            for n in all {
                if in_kinds(c, n, |k| k == kstr("RAW_KIND"))? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Whether the node's list field is not empty.
fn nonempty(c: &Ctx, h: u32, f: &str) -> R<bool> {
    Ok(matches!(c.field(h, f)?, Val::Tuple(v) | Val::List(v) if !v.is_empty()))
}

/// Whether a node renders as a `switch`: neither every branch evaluated nor the branches split apart.
pub(crate) fn switch_form(c: &Ctx, h: u32) -> R<bool> {
    c.node(h)?;
    let Val::Tuple(fe) = c.field(h, "feathers")? else {
        return refuse("feathers is not a tuple");
    };
    let has = |name: &str| fe.iter().any(|f| matches!(f, Val::Str(s) if s == name));
    Ok(!(has("evaluates_all_branches") || has("splits_by_branch")))
}

/// Whether a trip of the loop folds a vector longer than the compiler unrolls.
fn folds_wide_vector(c: &Ctx, lp: u32, unroll_limit: i64) -> R<bool> {
    let mut all = vec![];
    flat(c, &c.body_nodes(lp)?, &mut all)?;
    for n in all {
        if in_kinds(c, n, |k| REDUCTIONS.contains(&k))? {
            if let Some(&o) = c.nodes[n as usize].ops.first() {
                let s = &c.node(o)?.shape;
                if s.len() == 1 && s[0] > unroll_limit {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// The top-level loops the host runs interchanged, in walk order.
fn interchanged(c: &Ctx, w: u32, unroll_limit: i64) -> R<Vec<u32>> {
    if !lane_form(c, w)? {
        return Ok(vec![]);
    }
    let order = walk_order(c, w)?;
    let loops: Vec<u32> = order
        .iter()
        .copied()
        .filter(|&n| is_a(c, n, "Loop"))
        .collect();
    if loops.is_empty() {
        return Ok(vec![]);
    }
    let mut all = vec![];
    flat(c, &order, &mut all)?;
    for &n in &all {
        if is_a(c, n, "TapeRead")
            || is_a(c, n, "LoopCount")
            || in_kinds(c, n, |k| {
                konst("RANDOM_OPS").contains(&k) || is_linalg_kind(k)
            })?
        {
            return Ok(vec![]);
        }
        if is_a(c, n, "Loop") && nonempty(c, n, "body_sinks")? {
            return Ok(vec![]);
        }
    }
    // a loop only a switch case reaches renders inside that case
    let used: std::collections::HashSet<u32> = order
        .iter()
        .flat_map(|&n| c.nodes[n as usize].ops.iter().copied())
        .collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack: Vec<u32> = order
        .iter()
        .copied()
        .filter(|n| !used.contains(n))
        .collect();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if is_a(c, n, "Dispatch") && switch_form(c, n)? {
            stack.extend(c.nodes[n as usize].ops.first())
        } else {
            stack.extend(c.nodes[n as usize].ops.iter())
        }
    }
    if loops.iter().any(|l| !seen.contains(l)) {
        return Ok(vec![]);
    }
    let mut out = vec![];
    for l in loops {
        if !folds_wide_vector(c, l, unroll_limit)? {
            out.push(l)
        }
    }
    Ok(out)
}

/// Whether the op kind is a linear-algebra factorisation, consumer or projection.
fn is_linalg_kind(k: &str) -> bool {
    [
        "LINALG_FACTORISATIONS",
        "LINALG_CONSUMERS",
        "LINALG_PROJECTIONS",
    ]
    .iter()
    .any(|g| konst(g).contains(&k))
}

/// Whether the op kind is a call of a library function.
fn library_kind(k: &str) -> bool {
    (konst("UNARY_MATH").contains(&k) && !CHEAP_UNARY.contains(&k))
        || LIBRARY_BINARY.contains(&k)
        || konst("RANDOM_OPS").contains(&k)
}

/// The lookup planes read as parameter cells, in first-seen order.
pub fn param_cells(c: &Ctx, w: u32) -> R<Vec<String>> {
    let mut out: Vec<String> = vec![];
    for n in walk_order(c, w)? {
        if !is_a(c, n, "At") {
            continue;
        }
        let ops = &c.nodes[n as usize].ops;
        let (Some(&plane), Some(&index)) = (ops.first(), ops.get(1)) else {
            continue;
        };
        if is_a(c, index, "RoleConst") && c.str_field(index, "label")? == kstr("PARAM_CELL") {
            let name = c.str_field(plane, "name")?;
            if !out.contains(&name) {
                out.push(name)
            }
        }
    }
    Ok(out)
}

/// The matrix input planes the walk reads only through `at`, sorted by name: their column count is the caller's, not the lane
/// count.
pub fn gather_mats(c: &Ctx, w: u32) -> R<Vec<String>> {
    let (mut direct, mut crossed) = (vec![], vec![]);
    for n in walk_order(c, w)? {
        let at = is_a(c, n, "At");
        for (place, &o) in c.nodes[n as usize].ops.iter().enumerate() {
            let matrix = is_a(c, o, "Leaf") && c.str_field(o, "role")? == "mat_in";
            if !matrix {
                continue;
            }
            let name = c.str_field(o, "name")?;
            if at && place == 0 {
                crossed.push(name);
            } else {
                direct.push(name);
            }
        }
    }
    crossed.retain(|name| !direct.contains(name));
    crossed.sort();
    crossed.dedup();
    Ok(crossed)
}

/// The accumulates of the walk that keep a companion plane, once per name.
fn companions(c: &Ctx, w: u32) -> R<Vec<u32>> {
    let mut out: Vec<u32> = vec![];
    let mut names: Vec<String> = vec![];
    for n in walk_order(c, w)? {
        if !is_a(c, n, "AccumWrite") {
            continue;
        }
        let contract = c.str_field(n, "contract")?;
        if contract == "compensated" || contract == kstr("KEYED_AUTO") {
            let name = c.str_field(n, "name")?;
            if !names.contains(&name) {
                names.push(name);
                out.push(n)
            }
        }
    }
    Ok(out)
}

// ---- the door

/// Writes a text to the caller's buffer by the two-call length protocol.
pub(crate) fn text_out(buf: *mut c_char, cap: usize, len: *mut usize, s: &str) -> R<()> {
    if len.is_null() {
        return Err(Fault::BadArg("len is NULL".into()));
    }
    unsafe { *len = s.len() }
    if cap < s.len() + 1 {
        // nothing is needed from an empty text: a caller that sized with cap 0 has its answer
        return if s.is_empty() && cap == 0 {
            Ok(())
        } else {
            Err(Fault::Truncated)
        };
    }
    if buf.is_null() {
        return Err(Fault::BadArg("buf is NULL".into()));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, s.len());
        *buf.add(s.len()) = 0
    }
    Ok(())
}

/// Writes a list to the caller's buffer by the two-call length protocol.
pub(crate) fn list_out(buf: *mut u32, cap: usize, n: *mut usize, v: &[u32]) -> R<()> {
    if n.is_null() {
        return Err(Fault::BadArg("n is NULL".into()));
    }
    unsafe { *n = v.len() }
    if cap < v.len() || (buf.is_null() && !v.is_empty()) {
        return Err(Fault::Truncated);
    }
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), buf, v.len()) }
    Ok(())
}

/// Writes a flag as 0 or 1 to the caller's cell.
pub(crate) fn flag_out(out: *mut c_int, v: bool) -> R<()> {
    if out.is_null() {
        return Err(Fault::BadArg("out is NULL".into()));
    }
    unsafe { *out = v as c_int }
    Ok(())
}

/// The text a caller's name pointer holds; refuses when it is null or not UTF-8.
pub(crate) fn name_of<'a>(p: *const c_char, what: &str) -> R<&'a str> {
    if p.is_null() {
        return Err(Fault::BadArg(format!("{what} is NULL")));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| Fault::BadArg(format!("{what} is not UTF-8")))
}

/// Whether the lanes of a tile may run side by side over the walk; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_cpu_lane_form_access(
    ctx: *mut Ctx,
    walk: u32,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| flag_out(out, lane_form(c, walk)?))
}

/// The top-level loops the host runs interchanged, in walk order; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_cpu_interchanged_loops(
    ctx: *mut Ctx,
    walk: u32,
    unroll_limit: i64,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        list_out(buf, cap, n, &interchanged(c, walk, unroll_limit)?)
    })
}

/// Whether the node is a call of a library function; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_cost_is_library_call(
    ctx: *mut Ctx,
    node: u32,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| flag_out(out, in_kinds(c, node, library_kind)?))
}

/// Whether the node is a linear-algebra node that branches per lane; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_linalg_is_linalg(ctx: *mut Ctx, node: u32, out: *mut c_int) -> c_int {
    guard(ctx, |c| flag_out(out, in_kinds(c, node, is_linalg_kind)?))
}

/// Whether a dispatch node renders as a switch; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_dispatch_switch_form(
    ctx: *mut Ctx,
    node: u32,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| flag_out(out, switch_form(c, node)?))
}

/// Whether the walk promotes the named lookup plane to a parameter cell; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_ir_param_plane(
    ctx: *mut Ctx,
    walk: u32,
    name: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        let n = name_of(name, "name")?;
        flag_out(out, param_cells(c, walk)?.iter().any(|p| p == n))
    })
}

/// The accumulates of the walk that keep a companion plane; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_ir_companions(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut u32,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| list_out(buf, cap, n, &companions(c, walk)?))
}

/// Whether the node places a value in a plane; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_is_resident(ctx: *mut Ctx, node: u32, out: *mut c_int) -> c_int {
    guard(ctx, |c| {
        flag_out(out, in_kinds(c, node, |k| k == RESIDENT_KIND)?)
    })
}

/// Whether the node names a scratch plane the runtime made; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_is_tiled(ctx: *mut Ctx, node: u32, out: *mut c_int) -> c_int {
    guard(ctx, |c| flag_out(out, wide::is_tiled(c, node)?))
}

/// The placements of the nodes and the bodies of their loops, one row per plane.
pub fn wide_record(c: &Ctx, order: &[u32]) -> R<Vec<PlaneRow>> {
    let mut out: std::collections::BTreeMap<String, PlaneRow> = Default::default();
    let mut nodes: Vec<u32> = order.to_vec();
    while let Some(n) = nodes.pop() {
        c.node(n)?;
        if is_a(c, n, "Loop") {
            nodes.extend(c.body_nodes(n)?)
        } else if in_kinds(c, n, |k| k == RESIDENT_KIND)? {
            let plane = *c.nodes[n as usize]
                .ops
                .get(1)
                .ok_or(Fault::Refused("a placement has no plane".into()))?;
            let shape = &c.nodes[n as usize].shape;
            let layout = if wide::is_tiled(c, plane)? { 1 } else { 0 };
            let width = if shape.len() == 1 { shape[0] } else { 0 };
            out.insert(
                c.str_field(plane, "name")?,
                PlaneRow {
                    plane,
                    layout,
                    width,
                },
            );
        }
    }
    Ok(out.into_values().collect())
}

/// One placed plane: the leaf that names it, its layout (0 plane-major, 1 warp-tiled) and width.
#[repr(C)]
pub struct PlaneRow {
    /// The leaf that names the plane.
    pub plane: u32,
    /// How the plane is laid out (`RW_LAYOUT_*` in the header).
    pub layout: c_int,
    /// The plane's width.
    pub width: i64,
}

/// Whether the node is spelled as its runtime twin: a product or a long reduction passing the fold threshold.
fn fold_rolled(c: &Ctx, h: u32, fold_min: Option<i64>, form: &str) -> R<bool> {
    let Some(k) = op_kind(c, h)? else {
        return Ok(false);
    };
    let shape_of = |i: usize| -> R<Vec<i64>> {
        Ok(c.node(
            *c.nodes[h as usize]
                .ops
                .get(i)
                .ok_or(Fault::Refused("missing operand".into()))?,
        )?
        .shape
        .clone())
    };
    let (fold, size) = if k == "mv" || k == "mm" {
        (
            *shape_of(0)?
                .get(1)
                .ok_or(Fault::Refused("operand has no second extent".into()))?,
            c.nodes[h as usize].shape.iter().product::<i64>(),
        )
    } else if REDUCTIONS.contains(&k.as_str()) && shape_of(0)?.len() == 1 {
        (shape_of(0)?[0], 1)
    } else {
        return Ok(false);
    };
    if form != "auto" {
        return Ok(form == "rolled");
    }
    let Some(limit) = fold_min else {
        return refuse("the facts of this build name no fold threshold");
    };
    Ok(fold > limit || size > limit)
}

/// The placements of the given nodes, one row per plane; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_wide_record(
    ctx: *mut Ctx,
    order: *const u32,
    n: usize,
    rows: *mut PlaneRow,
    cap: usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        if (n > 0 && order.is_null()) || count.is_null() {
            return Err(Fault::BadArg("order or count is NULL".into()));
        }
        let o = if n == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(order, n) }
        };
        let found = wide_record(c, o)?;
        unsafe { *count = found.len() }
        if cap < found.len() || (rows.is_null() && !found.is_empty()) {
            return Err(Fault::Truncated);
        }
        for (k, row) in found.into_iter().enumerate() {
            unsafe { rows.add(k).write(row) }
        }
        Ok(())
    })
}

/// Whether the node is spelled as its runtime twin; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_fold_rolled(
    ctx: *mut Ctx,
    node: u32,
    fold_min: *const i64,
    form: *const c_char,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        let f = name_of(form, "form")?;
        flag_out(
            out,
            fold_rolled(
                c,
                node,
                if fold_min.is_null() {
                    None
                } else {
                    Some(unsafe { *fold_min })
                },
                f,
            )?,
        )
    })
}

/// The key run of one keyed sink of a kernel; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_lanes_key_run_in_kernel(
    ctx: *mut Ctx,
    sinks: *const u32,
    nsinks: usize,
    sink: u32,
    run_max: i64,
    block_run: i64,
    out: *mut KeyRunAnswer,
) -> c_int {
    guard(ctx, |c| {
        if (nsinks > 0 && sinks.is_null()) || out.is_null() {
            return Err(Fault::BadArg("sinks or out is NULL".into()));
        }
        let s = if nsinks == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        let answer = match lanes::key_run(c, s, sink, run_max as i128, block_run as i128)? {
            None => KeyRunAnswer {
                found: 0,
                form: 0,
                width: 0,
                block: 0,
            },
            Some(k) => {
                let form = match k.form {
                    "proven" => 1,
                    "sorted" => 2,
                    _ => 3,
                };
                KeyRunAnswer {
                    found: 1,
                    form,
                    width: k.width as i64,
                    block: k.block as c_int,
                }
            }
        };
        unsafe { out.write(answer) }
        Ok(())
    })
}

/// The key run of a sink: `found` 0 for none, else the form (1 proven, 2 sorted, 3 tested), the
/// width and whether a block of lanes shares it.
#[repr(C)]
pub struct KeyRunAnswer {
    /// 1 when the sink has a key run, else 0.
    pub found: c_int,
    /// How the key moves (`RW_KEY_*` in the header).
    pub form: c_int,
    /// How many consecutive lanes the run covers.
    pub width: i64,
    /// 1 when the key is constant over a block of lanes, else 0.
    pub block: c_int,
}
