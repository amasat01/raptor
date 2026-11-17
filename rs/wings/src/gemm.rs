//! The matrix product a recognised contraction is, as the door's `eg_unit_gemm` states it: which operand is A and which B, the
//! orientation of each, the stored shapes of the three planes, and the lanes that share one slot of the operand counted by the
//! launch. It is the old route's planning read off the recogniser's strides and
//! the lane split, with no declaration of the planes: a shape the strides do not state is not derived, and the unit has no product.
use crate::contraction::{Contraction, Operand, OperandRole, Stride, TermForm};

/// An extent of a stored plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    /// A number known when the unit is lowered.
    Fixed(i64),
    /// The run's lane count divided by the product's `lane_fanin`.
    PerLane,
}

/// One matrix product `C = op(A) * op(B)` over three planes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmCall {
    /// The name of the plane that is A.
    pub a: String,
    /// The name of the plane that is B.
    pub b: String,
    /// The name of the plane that is C.
    pub c: String,
    /// A is stored (k, m): op(A) is its transpose.
    pub trans_a: bool,
    /// B is stored (n, k): op(B) is its transpose.
    pub trans_b: bool,
    /// C is stored through a transposed view: (n, m).
    pub trans_c: bool,
    /// C accumulates the product (beta 1) instead of being set to it.
    pub accumulate: bool,
    /// The stored shapes (rows, cols) of A, B and C.
    pub shapes: [(Extent, Extent); 3],
}

/// The product of a unit: its call and the lanes that share one slot of the plane counted by the launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmPlan {
    /// The forward product.
    pub forward: GemmCall,
    /// The adjoint's two products, in order: the first operand's gradient, then the second's; each accumulates into a gradient
    /// plane named `<operand>_bar` from the cotangent `<output>_bar`.
    pub adjoint: [GemmCall; 2],
    /// The other operand's free extent: a `PerLane` extent is the lane count over it; 0 when no extent is `PerLane`.
    pub lane_fanin: i64,
}

/// An operand resolved to a matrix: which axis the loop contracts and the extent of the other.
struct Matrix {
    /// 0 when the contracted axis is the major one (stored (k, free)), 1 when it is the minor one (stored (free, k)).
    axis: usize,
    /// The extent of the free axis, none when the strides do not state it.
    free: Option<Extent>,
}

/// An operand as a matrix, from how one step of the loop moves its index: a stride of one walks the minor axis, any other
/// stride the major axis and is the free extent, and a stride of the sample count walks the major axis of a plane counted by
/// the launch. A plane whose minor axis is contracted does not state its free extent.
fn matrix(op: &Operand) -> Option<Matrix> {
    match op.reduce_stride.as_ref()? {
        Stride::Symbol(_) => Some(Matrix {
            axis: 0,
            free: Some(Extent::PerLane),
        }),
        Stride::Int(1) => Some(Matrix {
            axis: 1,
            free: None,
        }),
        Stride::Int(s) if *s > 1 => Some(Matrix {
            axis: 0,
            free: Some(Extent::Fixed(*s)),
        }),
        Stride::Int(_) => None,
    }
}

/// Whether the operand is read by the major half of the lane split: only through a division of the lane index.
fn is_major(op: &Operand) -> bool {
    op.lane_ops == ["div"]
}

/// Whether the operand is read by the minor half: the lane minus its division times the divisor.
fn is_minor(op: &Operand) -> bool {
    op.lane_ops.len() > 1 && op.lane_ops.iter().any(|k| k == "div")
}

/// The forward product of a contraction and its adjoint, or none when the old route did not take it or the strides do not state a shape.
#[must_use]
pub fn gemm_plan(found: &Contraction) -> Option<GemmPlan> {
    let [a, b] = &found.operands[..] else {
        return None;
    };
    let plain = |op: &Operand| matches!(op.role, OperandRole::Plane | OperandRole::Staged);
    if !(plain(a) && plain(b) && found.dense && found.term_form == TermForm::ProductOfReads)
        || found.outputs.len() != 1
        || (found.reduce_start, found.reduce_step) != (0, 1)
    {
        return None;
    }
    let (ma, mb) = (matrix(a)?, matrix(b)?);
    // the lanes are the product in row-major order: one operand is read by the major half of the split, the other by the minor
    // half, and both halves divide by the same number
    let a_major = is_major(a) && is_minor(b);
    if !(a_major || (is_major(b) && is_minor(a))) || a.fanin.is_none() || a.fanin != b.fanin {
        return None;
    }
    let (minor, minor_matrix, major_matrix) = if a_major {
        (b, &mb, &ma)
    } else {
        (a, &ma, &mb)
    };
    let m = minor.fanin?;
    // the minor operand's free extent is the divisor; its strides must agree when they state it
    let minor_free = match minor_matrix.free {
        Some(Extent::Fixed(n)) if n != m => return None,
        Some(Extent::PerLane) => return None,
        _ => Extent::Fixed(m),
    };
    let major_free = major_matrix.free?;
    let (free_a, free_b) = if a_major {
        (major_free, minor_free)
    } else {
        (minor_free, major_free)
    };
    let k = Extent::Fixed(found.reduce_extent);
    let stored = |matrix: &Matrix, free: Extent| {
        if matrix.axis == 0 {
            (k, free)
        } else {
            (free, k)
        }
    };
    let lane_flat = !a_major;
    let ((first, first_matrix, first_free), (second, second_matrix, second_free)) = if lane_flat {
        ((b, &mb, free_b), (a, &ma, free_a))
    } else {
        ((a, &ma, free_a), (b, &mb, free_b))
    };
    let per_lane = [free_a, free_b].contains(&Extent::PerLane);
    let (shape_a, shape_b) = (stored(&ma, free_a), stored(&mb, free_b));
    let (bar, out) = (
        format!("{}_bar", found.outputs[0].0),
        found.outputs[0].0.clone(),
    );
    let c_shape = (first_free, second_free);
    let forward = GemmCall {
        a: first.name.clone(),
        b: second.name.clone(),
        c: out,
        trans_a: first_matrix.axis == 0,
        trans_b: second_matrix.axis == 1,
        trans_c: false,
        accumulate: false,
        shapes: [
            stored(first_matrix, first_free),
            stored(second_matrix, second_free),
            c_shape,
        ],
    };
    // the gradient of each operand is the cotangent times the other operand, written back in the operand's own layout
    let gradient = |into: &Operand,
                    other: &Operand,
                    trans_a: bool,
                    other_axis: usize,
                    into_axis: usize,
                    shapes: [(Extent, Extent); 2]| {
        GemmCall {
            a: bar.clone(),
            b: other.name.clone(),
            c: format!("{}_bar", into.name),
            trans_a,
            trans_b: other_axis == 0,
            trans_c: into_axis == 0,
            accumulate: true,
            shapes: [c_shape, shapes[0], shapes[1]],
        }
    };
    Some(GemmPlan {
        forward,
        adjoint: [
            gradient(a, b, lane_flat, mb.axis, ma.axis, [shape_b, shape_a]),
            gradient(b, a, !lane_flat, ma.axis, mb.axis, [shape_a, shape_b]),
        ],
        lane_fanin: if per_lane { m } else { 0 },
    })
}
