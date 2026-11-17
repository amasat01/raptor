//! Typed reads of a walk a front end asks beyond the arguments: the type of a bound slot, the planes read at launch start, and
//! whether a dispatch's selector is static. Owns the C doors `rx_walk_slot_type`, `rx_walk_prior` and `rx_walk_static_selector`.
use crate::consts::put_names;
use crate::door::{cstr, guard};
use crate::record::{badarg, refuse, Ctx, Fault};
use crate::typed::{rx_type_of_parts, RxType};
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
