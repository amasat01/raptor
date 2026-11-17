//! The per-lane storage a kernel declares for its library results: the figure a device backend reads a compiled kernel's local
//! memory against. It is decided here, with the same rolled-site test the lowering uses (`fold_rolled`), so the figure cannot
//! disagree with the form the lowering chose.
use crate::plan::exceeds;
use crate::{fold_rolled, is_linalg_kind, op_kind};
use raptor_ir::record::{refuse, Ctx, Fault};

/// The result of an analysis: its answer, or the refusal that stopped it.
type R<T> = Result<T, Fault>;

/// Bytes of one index word of a factorisation: a pivot or scale per row, and a status.
const WORD_BYTES: i64 = 4;

/// Bytes one scalar of the element type takes in a library result.
fn scalar_bytes(dtype: &str) -> R<i64> {
    match dtype {
        "f64" => Ok(8),
        "f32" => Ok(4),
        other => refuse(format!(
            "a library result of type {other} has no stored width"
        )),
    }
}

/// Bytes a linear-algebra node holds: a factor is its struct (the matrix, the pivots or scales and a status word; `eigh` also the
/// working copy its rotations turn), anything else the result it copies out.
fn linalg_bytes(c: &Ctx, h: u32, kind: &str) -> R<i64> {
    let node = c.node(h)?;
    let first = c.node(node.ops[0])?;
    let width = scalar_bytes(&first.dtype)?;
    let n = *first
        .shape
        .last()
        .ok_or(Fault::Refused("a factored operand has no extent".into()))?;
    Ok(match kind {
        "lu" => n * n * width + WORD_BYTES * n + WORD_BYTES,
        "chol" | "chol_from" => n * n * width + WORD_BYTES,
        "eigh" => (2 * n * n + n) * width + WORD_BYTES,
        "qr" => (first.shape[0] * n + n) * width + WORD_BYTES,
        _ => node.shape.iter().product::<i64>() * scalar_bytes(&node.dtype)?,
    })
}

/// Bytes of per-lane storage the kernel whose canonical walk is `order` declares for its library results: every linear-algebra node with a shape
/// at its own size, and each product or reduction site rolled under `fold_min` and `form` (see `fold_rolled`), and each built
/// matrix with more entries than `fold_min`, as the item it is written into. Values the kernel keeps in registers are not counted, and neither are the nodes inside a loop's body (they are not in
/// `order`, which lists a loop once).
///
/// # Errors
/// Refuses a handle the context does not hold, a library result of another element type than `f64` or `f32`, and, under form
/// "auto", a build whose facts name no fold threshold.
pub fn working_set_bytes(c: &Ctx, order: &[u32], fold_min: Option<i64>, form: &str) -> R<i64> {
    let mut total = 0;
    for &h in order {
        let node = c.node(h)?;
        let Some(kind) = op_kind(c, h)? else { continue };
        if node.shape.is_empty() {
            continue;
        }
        let entries: i64 = node.shape.iter().product();
        let built_matrix = kind == "vec" && node.shape.len() == 2 && exceeds(entries, fold_min)?;
        if is_linalg_kind(&kind) {
            total += linalg_bytes(c, h, &kind)?;
        } else if fold_rolled(c, h, fold_min, form)? || built_matrix {
            total += entries * if node.dtype == "f32" { 4 } else { 8 };
        }
    }
    Ok(total)
}
