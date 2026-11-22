//! The sampled read: a `Sample` of a table at real coordinates is written as `At` reads at whole indices, so no later layer
//! (differentiation, lowering, writers, runtime) knows the element. Coordinates are in index units: a whole number names the stored
//! element exactly.
//!
//! - `nearest` reads the element at `floor(x + 0.5)` on each axis, so a half rounds up.
//! - `linear` reads, per axis, the elements at `i = floor(x)` and `i + 1` and weighs them by `1 - w` and `w` with `w = x - i`; over
//!   several axes each of the 2, 4 or 8 corners is weighed by the product of its per-axis weights.
//!
//! The edge rule is the `At` rule of each read, so a corner outside the table is clamped, wrapped or zero by itself.
use crate::cells::build;
use crate::planes::type_parts;
use crate::scatter::constant;
use crate::{is_a, R};
use raptor_ir::door::guard;
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::value::{ttype_val, Val};

/// A real constant of the type `dtype`.
fn real(ctx: &mut Ctx, value: f64, dtype: &str) -> R<u32> {
    build(
        ctx,
        "Const",
        vec![
            ("literal", Val::Float(value)),
            ("ttype", ttype_val(dtype, &[], None)),
        ],
    )
}

/// The whole index of a real coordinate: `real` cast to a 32-bit integer.
fn whole(ctx: &mut Ctx, real: u32) -> R<u32> {
    ctx.make_op("cast", &[real], &Val::Str("i32".into()))
}

/// The read of the plane at the whole `indices` under `edge`, of the value type `ttype`.
fn read(ctx: &mut Ctx, (plane, edge): (u32, &str), indices: Vec<u32>, ttype: Val) -> R<u32> {
    build(
        ctx,
        "At",
        vec![
            ("plane", Val::Node(plane)),
            (
                "indices",
                Val::Tuple(indices.into_iter().map(Val::Node).collect()),
            ),
            ("edge", Val::Str(edge.into())),
            ("ttype", ttype),
        ],
    )
}

/// The `Sample` `node` written as reads of its plane: the node that takes its place.
///
/// Refuses a node that is not a `Sample`.
pub fn expand(ctx: &mut Ctx, node: u32) -> R<u32> {
    if !is_a(ctx, node, "Sample") {
        return refuse("sample expansion: the node is not a Sample");
    }
    let ops = ctx.nodes[node as usize].ops.clone();
    let (plane, coordinates) = (ops[0], &ops[1..]);
    let (filter, edge) = (ctx.str_field(node, "filter")?, ctx.str_field(node, "edge")?);
    let value = type_parts(ctx, node);
    let ttype = ttype_val(&value.dtype, &value.shape, value.tag.as_deref());
    let site = (plane, edge.as_str());
    if filter == "nearest" {
        let mut indices = vec![];
        for &x in coordinates {
            let dtype = ctx.nodes[x as usize].dtype.clone();
            let half = real(ctx, 0.5, &dtype)?;
            let shifted = ctx.make_op("add", &[x, half], &Val::None)?;
            let floor = ctx.make_op("floor", &[shifted], &Val::None)?;
            indices.push(whole(ctx, floor)?);
        }
        return read(ctx, site, indices, ttype);
    }
    linear(ctx, site, coordinates, (ttype, &value.dtype))
}

/// The two whole indices an axis reads and the weights of its two elements: `(i, i + 1)` and `(1 - w, w)` for `x = i + w`.
fn axis_pair(ctx: &mut Ctx, x: u32, weight_dtype: &str) -> R<([u32; 2], [u32; 2])> {
    let dtype = ctx.nodes[x as usize].dtype.clone();
    let floor = ctx.make_op("floor", &[x], &Val::None)?;
    let low = whole(ctx, floor)?;
    let one = constant(ctx, 1, "i32")?;
    let high = ctx.make_op("add", &[low, one], &Val::None)?;
    let w = ctx.make_op("sub", &[x, floor], &Val::None)?;
    let one_real = real(ctx, 1.0, &dtype)?;
    let complement = ctx.make_op("sub", &[one_real, w], &Val::None)?;
    let mut weights = [complement, w];
    if dtype != weight_dtype {
        for weight in &mut weights {
            *weight = ctx.make_op("cast", &[*weight], &Val::Str(weight_dtype.into()))?;
        }
    }
    Ok(([low, high], weights))
}

/// The weighted sum of the 2, 4 or 8 corners of the cell the coordinates are in, the first axis varying slowest.
fn linear(
    ctx: &mut Ctx,
    site: (u32, &str),
    coordinates: &[u32],
    (ttype, dtype): (Val, &str),
) -> R<u32> {
    let mut axes = vec![];
    for &x in coordinates {
        axes.push(axis_pair(ctx, x, dtype)?);
    }
    let mut total: Option<u32> = None;
    for corner in 0..1usize << axes.len() {
        let (mut indices, mut weight) = (vec![], None);
        for (k, (at, weights)) in axes.iter().enumerate() {
            let side = (corner >> (axes.len() - 1 - k)) & 1;
            indices.push(at[side]);
            weight = Some(match weight {
                None => weights[side],
                Some(product) => ctx.make_op("mul", &[product, weights[side]], &Val::None)?,
            });
        }
        let value = read(ctx, site, indices, ttype.clone())?;
        let term = ctx.make_op(
            "mul",
            &[weight.expect("one axis at least"), value],
            &Val::None,
        )?;
        total = Some(match total {
            None => term,
            Some(sum) => ctx.make_op("add", &[sum, term], &Val::None)?,
        });
    }
    Ok(total.expect("one corner at least"))
}

/// The `Sample` `node` written as reads of its plane, in `*out`; see `raptor_wings_front.h`.
///
/// # Safety
/// `ctx` is a live context from the IR door, or null (then the call refuses); `out` is valid for one write.
#[no_mangle]
pub unsafe extern "C" fn rw_sample_expand(
    ctx: *mut Ctx,
    node: u32,
    out: *mut u32,
) -> std::ffi::c_int {
    guard(ctx, |c| {
        c.node(node)?;
        let written = expand(c, node)?;
        if out.is_null() {
            return refuse("out is NULL");
        }
        unsafe { *out = written };
        Ok(())
    })
}
