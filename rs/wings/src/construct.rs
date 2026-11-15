//! Reading the node references a node holds in a field.
use raptor_ir::record::{Ctx, Fault};
use raptor_ir::value::Val;

/// The nodes a field holds: the one node of a single reference, or the nodes of a list or tuple of
/// them in order. Other values hold none.
pub(crate) fn refs(ctx: &Ctx, node: u32, field: &str) -> Result<Vec<u32>, Fault> {
    Ok(match ctx.field(node, field)? {
        Val::Node(one) => vec![one],
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .filter_map(|v| if let Val::Node(n) = v { Some(*n) } else { None })
            .collect(),
        _ => vec![],
    })
}
