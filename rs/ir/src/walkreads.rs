//! Typed reads of a walk a front end asks beyond the arguments: the type of a bound slot, the planes read at launch start, and
//! whether a dispatch's selector is static. Owns the C doors `rx_walk_slot_type`, `rx_walk_prior`, `rx_walk_static_selector`, `rx_walk_selector_reads`, `rx_walk_access_class`, `rx_walk_accum_companions`, `rx_walk_quantity_wires` and `rx_walk_quantity_slots`.
use crate::canon::str_const;
use crate::consts::put_names;
use crate::door::{cstr, guard, put};
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::typed::{pair_part, put_pairs, rx_type_of_parts, text_pair, RxPair, RxType};
use crate::value::Val;
use std::ffi::{c_char, c_int};

/// The type of the slot (`role`, `name`) the walk binds, as the C struct.
///
/// # Safety
/// The texts are NUL-terminated; `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_slot_type(
    ctx: *mut Ctx,
    walk: u32,
    role: *const c_char,
    name: *const c_char,
    out: *mut RxType,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return badarg("out is NULL");
        }
        let (role, name) = (cstr(role, "role")?, cstr(name, "name")?);
        let found = c
            .walk(walk)?
            .leaves
            .iter()
            .find(|l| l.0 .0 == role && l.0 .1 == name);
        let Some((_, t)) = found else {
            return refuse(format!("the walk binds no slot {role} {name}"));
        };
        unsafe { *out = rx_type_of_parts(&t.dtype, &t.shape, t.tag.as_deref())? };
        Ok(())
    })
}

/// The names of the planes whose launch-start value the walk reads, sorted, as the names protocol.
///
/// # Safety
/// `buf` is valid for `cap` bytes; `len` and `count` for one write each.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_prior(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        put_names(&c.walk(walk)?.prior, buf, cap, len, count)
    })
}

/// Whether the selector of the dispatch at `index` (in the order of `rx_dispatch_nodes`) is static: no sink of the walk writes a
/// plane it reads (`*out` is 0 or 1).
///
/// # Safety
/// `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_static_selector(
    ctx: *mut Ctx,
    walk: u32,
    index: i64,
    out: *mut c_int,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return badarg("out is NULL");
        }
        let flags = &c.walk(walk)?.static_selector;
        let Some(&flag) = usize::try_from(index).ok().and_then(|i| flags.get(i)) else {
            return Err(Fault::Refused(format!("the walk has no dispatch {index}")));
        };
        unsafe { *out = c_int::from(flag) };
        Ok(())
    })
}

/// The `(role, name)` slots the selector of the dispatch at `index` (in the order of `rx_dispatch_nodes`) reads, sorted, as pairs.
///
/// # Safety
/// `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_selector_reads(
    ctx: *mut Ctx,
    walk: u32,
    index: i64,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let all = &c.walk(walk)?.selector_reads;
        let Some(reads) = usize::try_from(index).ok().and_then(|i| all.get(i)) else {
            return Err(Fault::Refused(format!("the walk has no dispatch {index}")));
        };
        let pairs = reads
            .iter()
            .map(|(role, name)| text_pair(role, Some(name)))
            .collect::<Result<Vec<_>, _>>()?;
        put_pairs(&pairs, buf, cap, n)
    })
}

/// The access class of the walk (`sample_local`, `cross_sample_read`, ...), as text.
///
/// # Safety
/// `buf` is valid for `cap` bytes and `len` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_access_class(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> c_int {
    guard(ctx, |c| put(buf, cap, len, &c.walk(walk)?.access))
}

/// The companion planes of the compensated keyed accumulates of the walk, each as the pair (companion plane `<plane>_c`, the plane it
/// compensates), in the walk's order, once per companion.
///
/// # Safety
/// `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_accum_companions(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let mut pairs: Vec<(String, String)> = vec![];
        for &node in &c.walk(walk)?.order {
            if c.compensates(node)? {
                let plane = c.str_field(node, "name")?;
                let comp = format!("{plane}{}", str_const("COMPANION_SUFFIX"));
                if !pairs.iter().any(|(held, _)| *held == comp) {
                    pairs.push((comp, plane));
                }
            }
        }
        let pairs = pairs
            .iter()
            .map(|(comp, plane)| text_pair(comp, Some(plane)))
            .collect::<Result<Vec<_>, _>>()?;
        put_pairs(&pairs, buf, cap, n)
    })
}

/// The counting planes of the walk, each as the pair (count plane, the plane it counts), in the order the caller declared them.
///
/// # Safety
/// `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_counts(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let pairs = c
            .walk(walk)?
            .counts
            .iter()
            .map(|(count, of)| text_pair(count, Some(of)))
            .collect::<Result<Vec<_>, _>>()?;
        put_pairs(&pairs, buf, cap, n)
    })
}

/// The winner planes of the walk, each as the pair (winner plane, the plane it ranks), in the order the caller declared them.
///
/// # Safety
/// `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_winners(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let pairs = c
            .walk(walk)?
            .winners
            .iter()
            .map(|(winner, of)| text_pair(winner, Some(of)))
            .collect::<Result<Vec<_>, _>>()?;
        put_pairs(&pairs, buf, cap, n)
    })
}

/// The wires of the quantities the walk reaches, in walk order, one pair (namespace, slot name of the wire) per wire; the rows of
/// `rx_walk_quantity_slots` follow the same order.
///
/// # Safety
/// `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_quantity_wires(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let mut pairs = vec![];
        for (name, wires, _) in &c.walk(walk)?.quantities {
            for wire in wires {
                pairs.push(text_pair(name, Some(wire))?);
            }
        }
        put_pairs(&pairs, buf, cap, n)
    })
}

/// The slot numbers of the wires of the quantities the walk reaches, one pair (namespace, slot number) per wire, in the order of
/// `rx_walk_quantity_wires`.
///
/// # Safety
/// `buf` is valid for `cap` pairs and `n` for one write.
#[no_mangle]
pub unsafe extern "C" fn rx_walk_quantity_slots(
    ctx: *mut Ctx,
    walk: u32,
    buf: *mut RxPair,
    cap: usize,
    n: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let mut pairs = vec![];
        for (name, _, slots) in &c.walk(walk)?.quantities {
            for &slot in slots {
                pairs.push(RxPair {
                    first: pair_part(&Val::Str(name.clone()))?,
                    second: pair_part(&Val::Int(slot as i128))?,
                });
            }
        }
        put_pairs(&pairs, buf, cap, n)
    })
}
