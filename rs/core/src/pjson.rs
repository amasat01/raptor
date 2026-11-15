//! A manifest value as the validator checks it: an object keeps its key order, an integer, a float and a boolean are kept apart,
//! and a tuple or any other kind of value is a kind of its own. `repr` writes a value in the key text layout (see `keytext`).

use crate::keytext::{float_repr as repr_float, py_str as repr_str};

#[derive(Debug, Clone, PartialEq)]
/// A manifest value as the validator checks it.
pub enum Json {
    /// The null value.
    Null,
    /// A boolean.
    Bool(bool),
    /// An integer; an integer, a float and a boolean are kept apart.
    Int(i128),
    /// A floating-point number.
    Float(f64),
    /// A string.
    Str(String),
    /// A list.
    Arr(Vec<Json>),
    /// A tuple, which is a different kind from a list.
    Tup(Vec<Json>),
    /// Any other kind of value: its kind name and its text.
    Other(String, String),
    /// An object as key and value pairs in the order written.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// The value of key `k` when this is an object that holds it.
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(a, _)| a == k).map(|(_, b)| b),
            _ => None,
        }
    }
    /// Whether this is an object that holds key `k`.
    pub fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    /// The keys of an object in the order written; empty for any other kind.
    pub fn keys(&self) -> Vec<&str> {
        match self {
            Json::Obj(v) => v.iter().map(|(a, _)| a.as_str()).collect(),
            _ => vec![],
        }
    }
    /// The name of this value's kind as the refusal texts spell it.
    pub fn type_name(&self) -> String {
        match self {
            Json::Null => "NoneType".into(),
            Json::Bool(_) => "bool".into(),
            Json::Int(_) => "int".into(),
            Json::Float(_) => "float".into(),
            Json::Str(_) => "str".into(),
            Json::Arr(_) => "list".into(),
            Json::Tup(_) => "tuple".into(),
            Json::Other(t, _) => t.clone(),
            Json::Obj(_) => "dict".into(),
        }
    }
    /// The value written in the key text layout: strings quoted and escaped, floats in the shortest form, `None`, `True`, `False`.
    pub fn repr(&self) -> String {
        match self {
            Json::Null => "None".into(),
            Json::Bool(true) => "True".into(),
            Json::Bool(false) => "False".into(),
            Json::Int(i) => i.to_string(),
            Json::Float(f) => repr_float(*f),
            Json::Str(s) => repr_str(s),
            Json::Tup(v) => {
                if v.len() == 1 {
                    format!("({},)", v[0].repr())
                } else {
                    format!(
                        "({})",
                        v.iter().map(Json::repr).collect::<Vec<_>>().join(", ")
                    )
                }
            }
            Json::Other(_, r) => r.clone(),
            Json::Arr(v) => format!(
                "[{}]",
                v.iter().map(Json::repr).collect::<Vec<_>>().join(", ")
            ),
            Json::Obj(v) => format!(
                "{{{}}}",
                v.iter()
                    .map(|(k, x)| format!("{}: {}", repr_str(k), x.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    /// The value as plain text: a string as it is, any other value as `repr` writes it.
    pub fn pystr(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            o => o.repr(),
        }
    }
}
