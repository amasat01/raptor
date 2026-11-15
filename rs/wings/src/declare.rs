//! The kinds the wings declare, embedded from the build's wing manifests, and the one
//! export that declares them in a context. A rewrite of this crate makes nodes of these kinds (a
//! box store, for one), and a context must have them declared before such a node can be made.
use raptor_ir::door::guard;
use raptor_ir::record::{Ctx, Fault};
use std::ffi::c_int;

/// The wing manifests, one per extension: the build's files, copied here unchanged.
pub const MANIFESTS: [(&str, &str); 2] = [
    ("base", include_str!("../manifests/base.json")),
    ("simt", include_str!("../manifests/simt.json")),
];

/// The extensions whose kinds may be unqualified: the wings are first party.
const FIRST_PARTY: &str = "(\"base\", \"simt\")";

/// Declares every kind of the embedded wing manifests in the context. A kind already declared
/// equally (by an earlier call, or by a unit that declares the same kinds) is accepted as it is.
pub fn declare_kinds(ctx: &mut Ctx) -> Result<(), Fault> {
    for (_, manifest) in MANIFESTS {
        ctx.declare_dialect(manifest, FIRST_PARTY)?;
    }
    Ok(())
}

/// Declares the wings' kinds in the context; see `raptor_wings.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); every other pointer is valid for the reads and writes `raptor_wings.h` gives it, and a list or text buffer holds `cap` elements.
#[no_mangle]
pub unsafe extern "C" fn rw_declare_kinds(ctx: *mut Ctx) -> c_int {
    guard(ctx, declare_kinds)
}
