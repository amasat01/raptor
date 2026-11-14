//! Field values: the key-text value of rcore, and the reading of a fields text.
pub use rcore::keytext::{parse, text_val, write_val, Val};

/// A fields text is a tuple of (name, value) pairs.
pub fn fields_of(text: &str) -> Result<Vec<(String, Val)>, String> {
    let Val::Tuple(items) = parse(text)? else { return Err("fields text: expected a tuple of (name, value) pairs".into()) };
    items.into_iter().map(|it| match it {
        Val::Tuple(mut p) if p.len() == 2 => match (p.remove(0), p.remove(0)) {
            (Val::Str(k), v) => Ok((k, v)),
            _ => Err("fields text: a field name is a string".into()),
        },
        _ => Err("fields text: expected a (name, value) pair".into()),
    }).collect()
}

/// The text of a TensorType record.
pub fn ttype_val(dtype: &str, shape: &[i64], tag: Option<&str>) -> Val {
    Val::Rec("TensorType".into(), vec![
        ("shape".into(), Val::Tuple(shape.iter().map(|&d| Val::Int(d as i128)).collect())),
        ("dtype".into(), Val::Str(dtype.into())),
        ("tag".into(), tag.map_or(Val::None, |t| Val::Str(t.into()))),
    ])
}
