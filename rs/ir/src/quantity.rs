//! The construction rules of a compound quantity declaration: its slug, its pack and its wires. A quantity is a
//! namespace (its slug) owning wires, each a read role and a name; the leaf kind a role mints is the manifest's
//! `QUANTITY_ROLE_PAIRS`. Owns the C door `rx_quantity_check`.
use crate::consts::names;
use crate::door::{cstr, guard};
use crate::record::{refuse, Ctx, Fault};
use crate::value::{parse, text_val, Val};
use rcore::keytext::py_str;
use std::ffi::{c_char, c_int};

/// The wires of a quantity as the declaration states them: (read role, name), in declaration order.
fn wires_of(text: &str) -> Result<Vec<(String, String)>, Fault> {
    let Val::Tuple(items) = parse(text).map_err(Fault::Refused)? else {
        return refuse("wires: expected a tuple of (role, name) pairs");
    };
    items
        .into_iter()
        .map(|item| match item {
            Val::Tuple(pair) => match <[Val; 2]>::try_from(pair) {
                Ok([Val::Str(role), Val::Str(name)]) => Ok((role, name)),
                _ => refuse("wires: a wire is a (role, name) pair of strings"),
            },
            _ => refuse("wires: a wire is a (role, name) pair of strings"),
        })
        .collect()
}

/// Refuses, by name, a quantity declaration the library cannot expand: a pack outside the closed set, an empty slug, no
/// wire, a wire role no leaf kind mints, or two wires of one name.
pub fn quantity_check(slug: &str, pack: &str, wires: &[(String, String)]) -> Result<(), Fault> {
    let sl = py_str(slug);
    let packs = names("QUANTITY_PACKS").unwrap_or_default();
    if !packs.contains(&pack) {
        let listed = Val::Tuple(packs.iter().map(|p| Val::Str((*p).into())).collect());
        return refuse(format!(
            "Quantity {sl}: pack {} is not one of {}",
            py_str(pack),
            text_val(&listed)
        ));
    }
    if slug.is_empty() {
        return refuse("Quantity: slug is required — it is the namespace's first segment");
    }
    if wires.is_empty() {
        return refuse(format!("Quantity {sl}: declares no wires"));
    }
    // the manifest lists the role pairs as role, leaf kind, role, leaf kind, ...
    let pairs = names("QUANTITY_ROLE_PAIRS").unwrap_or_default();
    let mut roles: Vec<&str> = pairs.chunks(2).map(|pair| pair[0]).collect();
    let mut taken: Vec<&str> = vec![];
    for (role, name) in wires {
        if !roles.contains(&role.as_str()) {
            roles.sort_unstable();
            let listed: Vec<String> = roles.iter().map(|r| py_str(r)).collect();
            return refuse(format!(
                "Quantity {sl}: wire {} has role {}, which no leaf kind mints; declared read roles are [{}]",
                py_str(name),
                py_str(role),
                listed.join(", ")
            ));
        }
        if taken.contains(&name.as_str()) {
            return refuse(format!(
                "Quantity {sl}: duplicate wire name {}",
                py_str(name)
            ));
        }
        taken.push(name);
    }
    Ok(())
}

/// Refuses a quantity declaration the library cannot expand (see `quantity_check`); `wires_kt` is the key text of a tuple of
/// (role, name) pairs.
///
/// # Safety
/// Every pointer must be a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn rx_quantity_check(
    ctx: *mut Ctx,
    slug: *const c_char,
    pack: *const c_char,
    wires_kt: *const c_char,
) -> c_int {
    guard(ctx, |_| {
        quantity_check(
            cstr(slug, "slug")?,
            cstr(pack, "pack")?,
            &wires_of(cstr(wires_kt, "wires_kt")?)?,
        )
    })
}
