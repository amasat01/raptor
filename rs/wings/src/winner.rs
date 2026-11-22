//! The winners of a minimum or maximum taken in lane order: for each element, the lowest lane whose value equals the element's
//! finished value. The finished value is known only after every lane has folded, so the winner is a second launch of the kernel:
//! each lane that finds its own value in the element does an unsigned minimum of its lane number into the element of the winner
//! plane. The winner plane starts as all ones, so an element no lane reached keeps that. The minimum of lane numbers is exact and
//! does not depend on the order lanes arrive in.
use crate::cells::{build, int_like, plane_read};
use crate::logs::cell_of;
use crate::planes::type_parts;
use crate::team::sample_index;
use crate::{is_update_of, R};
use raptor_ir::record::{refuse, Ctx};
use raptor_ir::value::{ttype_val, Val};

/// The sinks that record the winners of the keyed minimums and maximums among `sinks` that take the lane order, for the planes
/// the caller states a winner plane for (`ranked` holds the pairs (winner plane, the plane it ranks)). Each reads the finished
/// element of the ranked plane, so it belongs to a launch after the one that folds.
pub(crate) fn sinks(ctx: &mut Ctx, sinks: &[u32], ranked: &[(String, String)]) -> R<Vec<u32>> {
    let mut out = vec![];
    for &sink in sinks {
        let folds = is_update_of(ctx, sink, "min")? || is_update_of(ctx, sink, "max")?;
        if !folds
            || ctx.str_field(sink, "order")? != "lane"
            || ctx.nodes[sink as usize].ops.len() < 2
        {
            continue;
        }
        let plane = ctx.str_field(sink, "name")?;
        let Some((winner, _)) = ranked.iter().find(|(_, of)| *of == plane) else {
            continue;
        };
        out.push(winner_sink(ctx, sink, winner)?);
    }
    Ok(out)
}

/// The sink that records into `winner` the lane of each element of the plane that the fold `sink` writes.
fn winner_sink(ctx: &mut Ctx, sink: u32, winner: &str) -> R<u32> {
    let value = ctx.nodes[sink as usize].ops[0];
    let kind = type_parts(ctx, value);
    if !kind.shape.is_empty() {
        return refuse(format!(
            "the winner of {} is recorded for a scalar value, not a value of shape {:?}",
            ctx.str_field(sink, "name")?,
            kind.shape
        ));
    }
    let indices = ctx.nodes[sink as usize].ops[1..].to_vec();
    let element = cell_of(ctx, sink, &indices)?;
    let finished = plane_read(ctx, &ctx.str_field(sink, "name")?, element, &kind.dtype)?;
    let own = ctx.make_op("eq", &[value, finished], &Val::None)?;
    // an element the lane did not win is addressed outside the plane, which the zero edge rule drops
    let mut places = vec![];
    for &index in &indices {
        let none = int_like(ctx, -1, index)?;
        let dtype = ctx.nodes[index as usize].dtype.clone();
        let given = vec![
            ("cond", Val::Node(own)),
            ("on_true", Val::Node(index)),
            ("on_false", Val::Node(none)),
            ("ttype", ttype_val(&dtype, &[], None)),
        ];
        places.push(Val::Node(build(ctx, "Select", given)?));
    }
    let lane = sample_index(ctx)?;
    let lane = ctx.make_op("cast", &[lane], &Val::Str("uint32".into()))?;
    let given = vec![
        ("name", Val::Str(winner.into())),
        ("value", Val::Node(lane)),
        ("indices", Val::Tuple(places)),
        ("axes", ctx.field(sink, "axes")?),
        ("edge", Val::Str("zero".into())),
        ("ttype", ttype_val("uint32", &[], None)),
        ("op", Val::Str("min".into())),
    ];
    build(ctx, "Update", given)
}
