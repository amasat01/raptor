//! The split wing's question about a kernel whose sink commits through a compensated sum: which
//! output the correction belongs to, and whether that output is the lane's own column.
use crate::{is_a, name_of, text_out};
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx, Fault};
use raptor_ir::value::Val;
use std::ffi::{c_char, c_int};

/// The result of an analysis: its answer, or the refusal that stopped it.
type Outcome<T> = Result<T, Fault>;

/// The kernel's output description: where the pool of candidate target names comes from.
struct OutputSpec<'a> {
    /// The slot name when the kernel returns a typed value, which then is the only candidate.
    returned_slot: Option<&'a str>,
    /// The names the kernel declared for its outputs (empty when it declared none).
    names: Vec<&'a str>,
}

/// The distinct names of the sinks, in first-seen order.
fn sink_names(ctx: &Ctx, sinks: &[u32]) -> Outcome<Vec<String>> {
    let mut names: Vec<String> = vec![];
    for &sink in sinks {
        let name = ctx.str_field(sink, "name")?;
        if !names.contains(&name) {
            names.push(name);
        }
    }
    Ok(names)
}

/// Whether a sink commits straight into the lane's own column: an assignment, or an
/// accumulate without an index.
fn writes_own_column(ctx: &Ctx, sink: u32) -> Outcome<bool> {
    if is_a(ctx, sink, "Assign") {
        return Ok(true);
    }
    Ok(is_a(ctx, sink, "AccumWrite") && ctx.field(sink, "index")? == Val::None)
}

/// The compensated sink's target `(name, own_column)`: the one output other than the companion.
/// Refuses when the outputs do not name exactly one target, or the target is a wide output.
fn compensated_target(
    ctx: &Ctx,
    compensation: &str,
    output: &OutputSpec,
    sinks: &[u32],
) -> Outcome<(String, bool)> {
    let all_names = sink_names(ctx, sinks)?;
    let pool: Vec<String> = match (output.returned_slot, output.names.is_empty()) {
        (Some(slot), _) => vec![slot.to_string()],
        (None, false) => output.names.iter().map(|n| n.to_string()).collect(),
        (None, true) => all_names,
    };
    let candidates: Vec<&String> = pool
        .iter()
        .filter(|name| name.as_str() != compensation)
        .collect();
    let [name] = candidates[..] else {
        return refuse(format!(
            "a compensated sink needs exactly one output target besides the companion \
             {compensation:?}; resolved {candidates:?} from {pool:?}"
        ));
    };
    let mut matching = vec![];
    for &sink in sinks {
        if ctx.str_field(sink, "name")? == *name {
            matching.push(sink);
        }
    }
    if matching.iter().any(|&sink| is_a(ctx, sink, "WideWrite")) {
        return refuse(format!(
            "compensated target {name:?} is a wide output; it must be a mutable"
        ));
    }
    let mut own_column = false;
    for &sink in &matching {
        own_column |= writes_own_column(ctx, sink)?;
    }
    Ok((name.clone(), own_column))
}

/// The answer of `rw_split_compensated_target`: the layout of `rw_compensated_target` in
/// raptor_wings.h.
#[repr(C)]
pub struct CompensatedTarget {
    /// 1 when the sink is compensated, else 0.
    pub found: c_int,
    /// 1 when the target has a column of its own, else 0.
    pub own_column: c_int,
}

/// The compensated target of a kernel's sinks as a struct and a name; see `raptor_wings.h`. When
/// `sink_id` is not
/// "compensated" nothing is found. Refuses when the outputs do not name exactly one target, or it
/// is a wide output.
///
/// # Safety
/// `out` is valid for one write; the other pointers as the header states.
#[no_mangle]
pub unsafe extern "C" fn rw_split_compensated_target(
    ctx: *mut Ctx,
    sink_id: *const c_char,
    compensation: *const c_char,
    returned_slot: *const c_char,
    output_names: *const c_char,
    sinks: *const u32,
    nsinks: usize,
    out: *mut CompensatedTarget,
    name_buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |ctx| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        if nsinks > 0 && sinks.is_null() {
            return Err(Fault::BadArg("sinks is NULL".into()));
        }
        let sink_list = if nsinks == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(sinks, nsinks) }
        };
        let names_text = name_of(output_names, "output_names")?;
        let output = OutputSpec {
            returned_slot: if returned_slot.is_null() {
                None
            } else {
                Some(name_of(returned_slot, "returned_slot")?)
            },
            names: names_text.lines().collect(),
        };
        let compensation = if compensation.is_null() {
            ""
        } else {
            name_of(compensation, "compensation")?
        };
        if name_of(sink_id, "sink_id")? != "compensated" {
            unsafe {
                *out = CompensatedTarget {
                    found: 0,
                    own_column: 0,
                }
            };
            return text_out(name_buf, cap, len, "");
        }
        let (name, own_column) = compensated_target(ctx, compensation, &output, sink_list)?;
        text_out(name_buf, cap, len, &name)?;
        unsafe {
            *out = CompensatedTarget {
                found: 1,
                own_column: own_column as c_int,
            }
        };
        Ok(())
    })
}
