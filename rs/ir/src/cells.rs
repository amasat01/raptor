//! The parameter cell: the read of one named parameter out of a one-cell plane, or out of the cell a sample's own
//! `param_cell` plane names. One owner of the layout that `Ctx::is_param_cell` reads back. Owns the C door `rx_param_cell`.
use crate::consts::names;
use crate::door::{cstr, guard, slice};
use crate::record::{Ctx, Fault};
use std::ffi::{c_char, c_int};

/// The written form of a tensor type, as the constructors of the nodes take it.
fn type_text(dtype: &str, shape: &[i64], tag: Option<&str>) -> String {
    let dims: Vec<String> = shape.iter().map(i64::to_string).collect();
    let shape = match dims.len() {
        1 => format!("({},)", dims[0]),
        _ => format!("({})", dims.join(", ")),
    };
    let tag = tag.map_or("None".into(), rcore::keytext::py_str);
    format!(
        "TensorType(shape={shape}, dtype={}, tag={tag})",
        rcore::keytext::py_str(dtype)
    )
}

/// The one text a string constant of the manifest holds.
fn constant(name: &str) -> Result<&'static str, Fault> {
    names(name)
        .and_then(|texts| texts.first().copied())
        .ok_or_else(|| Fault::Refused(format!("raptor owns no constant named {name:?}")))
}

impl Ctx {
    /// The read of parameter `name` of type (`dtype`, `shape`, `tag`): cell 0 of a one-cell `lookup` plane, or, with
    /// `per_cell`, the cell the sample's `param_cell` plane names.
    pub fn param_cell(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[i64],
        tag: Option<&str>,
        per_cell: bool,
    ) -> Result<u32, Fault> {
        let ttype = type_text(dtype, shape, tag);
        let index_type = type_text("i32", &[], None);
        let plane = format!(
            "(('kind', 'table_read'), ('role', 'lookup'), ('name', {}), ('ttype', {ttype}))",
            rcore::keytext::py_str(name)
        );
        let plane = self.new_node("Leaf", &plane, &[])?;
        let index = if per_cell {
            let cell = constant("CELL_PLANE")?;
            let params = format!(
                "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', {}), ('ttype', {index_type}))",
                rcore::keytext::py_str(cell)
            );
            self.new_node("Leaf", &params, &[])?
        } else {
            let label = rcore::keytext::py_str(constant("PARAM_CELL")?);
            let params = format!("(('value', 0), ('label', {label}), ('ttype', {index_type}))");
            self.new_node("RoleConst", &params, &[])?
        };
        let params = format!(
            "(('plane', #0), ('indices', (#1,)), ('edge', 'unchecked'), ('ttype', {ttype}))"
        );
        self.new_node("At", &params, &[plane, index])
    }
}

/// The read of a named parameter (see `Ctx::param_cell`).
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_param_cell(
    ctx: *mut Ctx,
    name: *const c_char,
    dtype: *const c_char,
    shape: *const i64,
    rank: usize,
    tag: *const c_char,
    per_cell: c_int,
    out: *mut u32,
) -> c_int {
    guard(ctx, |c| {
        if out.is_null() {
            return Err(Fault::BadArg("out is NULL".into()));
        }
        let tag = if tag.is_null() {
            None
        } else {
            Some(cstr(tag, "tag")?)
        };
        let node = c.param_cell(
            cstr(name, "name")?,
            cstr(dtype, "dtype")?,
            slice(shape, rank, "shape")?,
            tag,
            per_cell != 0,
        )?;
        unsafe { *out = node };
        Ok(())
    })
}
