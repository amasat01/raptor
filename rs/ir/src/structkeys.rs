//! Structural keys: which nodes of a walk's order are one value, so that a value traced twice counts once. Two nodes share a
//! key when their class, kind, type, literal, identity state, name and operands' keys are all equal; a literal enters as the
//! core's key text, so `nan` equals `nan`, `-0.0` is not `0.0` and `True` is not `1`. Owns the C door `rx_structural_keys`.
use crate::door::{guard, slice};
use crate::record::{badarg, Ctx, Fault};
use crate::value::{text_val, Val};
use std::collections::HashMap;
use std::ffi::c_int;

impl Ctx {
    /// The key text of one node, given the key numbers of the nodes seen before it (an operand outside the order has none).
    fn structural_key(&self, h: u32, seen: &HashMap<u32, i64>) -> Result<String, Fault> {
        let node = self.node(h)?;
        let literal = match self.field(h, "literal") {
            Ok(Val::None) | Err(_) => "-".to_string(),
            Ok(v) => text_val(&v),
        };
        let name = match self.field(h, "name") {
            Ok(v) => text_val(&v),
            Err(_) => "-".into(),
        };
        let operands: Vec<String> = node
            .ops
            .iter()
            .map(|o| seen.get(o).map_or("-".into(), i64::to_string))
            .collect();
        Ok([
            self.table.classes[node.class].name.clone(),
            self.kind_of(h)?,
            format!("{}{:?}{:?}", node.dtype, node.shape, node.tag),
            literal,
            text_val(&self.dedup_val(h)?),
            name,
            operands.join(","),
        ]
        .join("\u{1}"))
    }

    /// A number per node of `order` (children before parents): equal numbers mean equal structural keys. The numbers are
    /// dense, from 0 in order of first appearance, and mean something only within one call.
    pub fn structural_keys(&self, order: &[u32]) -> Result<Vec<i64>, Fault> {
        let (mut numbers, mut seen) = (HashMap::<String, i64>::new(), HashMap::new());
        let mut out = vec![];
        for &h in order {
            let key = self.structural_key(h, &seen)?;
            let next = numbers.len() as i64;
            let number = *numbers.entry(key).or_insert(next);
            seen.insert(h, number);
            out.push(number);
        }
        Ok(out)
    }
}

/// Copies the numbers to the caller's array (the two-call array protocol); refuses a short buffer.
fn give_numbers(buf: *mut i64, cap: usize, count: *mut usize, v: &[i64]) -> Result<(), Fault> {
    if count.is_null() {
        return badarg("count is NULL");
    }
    unsafe { *count = v.len() }
    if cap < v.len() || (buf.is_null() && !v.is_empty()) {
        return Err(Fault::Truncated);
    }
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), buf, v.len()) }
    Ok(())
}

/// The structural key numbers of the nodes of an order.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_structural_keys(
    ctx: *mut Ctx,
    order: *const u32,
    n: usize,
    buf: *mut i64,
    cap: usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        let numbers = c.structural_keys(slice(order, n, "order")?)?;
        give_numbers(buf, cap, count, &numbers)
    })
}
