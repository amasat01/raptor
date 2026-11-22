//! The checks of the access nodes (`At`, `Sample`, `AxisExtent`, `Update`, and the plane leaves) over the node slots: the axis list of
//! a plane, the index and coordinate types against it, and the split of an `Update`'s trailing operands. Each function states the
//! refusal text of its same-named function in the constructor checks of the IR crate (`rs/ir/src/ctor.rs`).
use super::*;
use pyo3::types::{PyInt, PyString};

/// The axes of a plane as (label, runtime-sized) pairs; a plane that declares none has the one runtime axis of a flat plane.
fn plane_axes(declared: &Bound<'_, PyAny>) -> PyResult<Vec<(String, bool)>> {
    let mut axes = vec![];
    for axis in declared.try_iter()? {
        let axis = axis?;
        let label = axis.get_item(0)?.extract::<String>().unwrap_or_default();
        axes.push((label, axis.get_item(1)?.is_none()));
    }
    if axes.is_empty() {
        axes.push((String::new(), true));
    }
    Ok(axes)
}

/// The name a refusal gives the axis at `position`: its label, else its position.
fn axis_name(axes: &[(String, bool)], position: usize) -> String {
    match axes.get(position) {
        Some((label, _)) if !label.is_empty() => format!("axis {label:?}"),
        _ => format!("axis {position}"),
    }
}

/// The dtype and rank of a node's type.
fn dtype_and_rank(node: &Bound<'_, PyAny>) -> PyResult<(String, usize)> {
    let ttype = node.getattr(intern!(node.py(), "ttype"))?;
    Ok((
        ttype.getattr("dtype")?.extract()?,
        ttype.getattr("shape")?.len()?,
    ))
}

/// The name of a plane leaf.
fn plane_name(plane: &Bound<'_, PyAny>) -> PyResult<String> {
    plane.getattr("name")?.extract()
}

/// Why `axes` is not an axis list: a tuple of at most eight (label, extent) pairs, each label a distinct non-empty text and each
/// extent a positive whole number or none (an extent known at run time).
pub fn axes_fault(axes: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let Ok(list) = axes.cast_exact::<PyTuple>() else {
        return Ok(Some("axes is a tuple of (label, extent) pairs".into()));
    };
    if list.len() > 8 {
        return Ok(Some(format!(
            "a plane has at most 8 axes, got {}",
            list.len()
        )));
    }
    let mut seen: Vec<String> = vec![];
    for axis in list.iter() {
        let Ok(pair) = axis.cast_exact::<PyTuple>() else {
            return Ok(Some("an axis is a (label, extent) pair".into()));
        };
        let label = match (pair.len(), pair.get_item(0)) {
            (2, Ok(l)) if l.is_exact_instance_of::<PyString>() => l.extract::<String>()?,
            _ => String::new(),
        };
        if label.is_empty() {
            return Ok(Some(
                "an axis is a (label, extent) pair with a non-empty label".into(),
            ));
        }
        if seen.contains(&label) {
            return Ok(Some(format!("the axis label {label:?} is given twice")));
        }
        let extent = pair.get_item(1)?;
        let positive = extent.is_exact_instance_of::<PyInt>()
            && extent.extract::<i128>().is_ok_and(|n| n >= 1);
        if !extent.is_none() && !positive {
            return Ok(Some(format!(
                "axis {label:?}: the extent is a positive whole number or none, got {}",
                rep(&extent)
            )));
        }
        seen.push(label);
    }
    Ok(None)
}

/// The count and type faults of `indices` against the axes of the named plane or sink: one rank-0 integer (i32 or i64) per axis.
fn address_fault(
    what: &str,
    name: &str,
    axes: &[(String, bool)],
    indices: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let indices: Vec<_> = indices.try_iter()?.collect::<PyResult<_>>()?;
    if indices.len() != axes.len() {
        let labels: Vec<String> = (0..axes.len()).map(|k| axis_name(axes, k)).collect();
        return Ok(Some(format!(
            "{what} {name:?} has {} axes ({}), so an access takes one index per axis, got {}",
            axes.len(),
            labels.join(", "),
            indices.len()
        )));
    }
    for (k, index) in indices.iter().enumerate() {
        let (dtype, rank) = dtype_and_rank(index)?;
        if rank != 0 || !(dtype == "i32" || dtype == "i64") {
            let shape = index.getattr("ttype")?.getattr("shape")?;
            return Ok(Some(format!(
                "{what} {name:?} {}: the index must be a rank-0 integer (i32 or i64), got {dtype} of shape {}",
                axis_name(axes, k),
                rep(&shape)
            )));
        }
    }
    Ok(None)
}

/// Why `indices` do not address `plane`: the count against the plane's axes, then each index being a rank-0 integer.
pub fn index_fault(
    plane: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let axes = plane_axes(&plane.getattr("axes")?)?;
    address_fault("plane", &plane_name(plane)?, &axes, indices)
}

/// Why a store at `indices` into the plane `name` that declares `axes` is refused; a store with no index commits at the lane's own
/// cell and states no address.
pub fn store_index_fault(
    name: &str,
    axes: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    if indices.len()? == 0 {
        return Ok(None);
    }
    let declared = plane_axes(axes)?;
    address_fault("sink", name, &declared, indices)
}

/// Why `coordinates` cannot sample `plane` with `filter`: one to three of them, one per axis of the plane, each a rank-0 real, and a
/// real table for a linear sample.
pub fn sample_fault(
    plane: &Bound<'_, PyAny>,
    coordinates: &Bound<'_, PyAny>,
    filter: &Bound<'_, PyAny>,
) -> PyResult<Option<String>> {
    let axes = plane_axes(&plane.getattr("axes")?)?;
    let name = plane_name(plane)?;
    let coordinates: Vec<_> = coordinates.try_iter()?.collect::<PyResult<_>>()?;
    if !(1..=3).contains(&coordinates.len()) {
        return Ok(Some(format!(
            "a sample takes one to three coordinates, got {}",
            coordinates.len()
        )));
    }
    if coordinates.len() != axes.len() {
        return Ok(Some(format!(
            "plane {name:?} has {} axes, so a sample takes one coordinate per axis, got {}",
            axes.len(),
            coordinates.len()
        )));
    }
    let (value, _) = dtype_and_rank(plane)?;
    let real = |dtype: &str| dtype == "f32" || dtype == "f64";
    if filter.extract::<String>().is_ok_and(|f| f == "linear") && !real(&value) {
        return Ok(Some(format!(
            "plane {name:?} holds {value}: a linear sample weighs values, so it takes a real table (use filter 'nearest')"
        )));
    }
    for (k, coordinate) in coordinates.iter().enumerate() {
        let (dtype, rank) = dtype_and_rank(coordinate)?;
        if rank != 0 || !real(&dtype) {
            let shape = coordinate.getattr("ttype")?.getattr("shape")?;
            return Ok(Some(format!(
                "plane {name:?} {}: the coordinate must be a rank-0 real (f32 or f64), got {dtype} of shape {}",
                axis_name(&axes, k),
                rep(&shape)
            )));
        }
    }
    Ok(None)
}

/// Why `axis` is not an axis of `plane`.
pub fn axis_fault(plane: &Bound<'_, PyAny>, axis: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let count = plane_axes(&plane.getattr("axes")?)?.len();
    let axis = axis.extract::<i128>()?;
    let known = usize::try_from(axis).is_ok_and(|a| a < count);
    Ok((!known).then(|| {
        format!(
            "plane {:?} has {count} axes, there is no axis {axis}",
            plane_name(plane).unwrap_or_default()
        )
    }))
}

/// The operands of an `Update` after its value, split into the expected value (a `compare_exchange` takes one right after the value)
/// and the index list.
fn update_operands<'py>(
    py: Python<'py>,
    op: &Bound<'py, PyAny>,
    rest: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
    let rest: Vec<_> = rest.try_iter()?.collect::<PyResult<_>>()?;
    let compare_exchange = op
        .extract::<String>()
        .is_ok_and(|o| o == "compare_exchange");
    let taken = usize::from(compare_exchange && !rest.is_empty());
    let (expected, address) = rest.split_at(taken);
    Ok((PyTuple::new(py, expected)?, PyTuple::new(py, address)?))
}

/// The expected value of an `Update`: one operand for a `compare_exchange`, none for any other operation.
pub fn expected_of<'py>(
    py: Python<'py>,
    op: &Bound<'py, PyAny>,
    rest: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    Ok(update_operands(py, op, rest)?.0)
}

/// The index operands of an `Update`.
pub fn address_of<'py>(
    py: Python<'py>,
    op: &Bound<'py, PyAny>,
    rest: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    Ok(update_operands(py, op, rest)?.1)
}
