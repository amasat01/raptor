//! A small JSON reader (no dependency): the manifest is read by the build script and by the binding with the same code.
#[derive(Debug, Clone)]
/// A parsed JSON value; an object keeps its keys in the order they were written.
pub enum J {
    /// The JSON `null`.
    Null,
    // read by neither copy of this file that compiles in a build script; kept as part of the JSON value set
    #[allow(dead_code)]
    /// A JSON `true` or `false`.
    Bool(bool),
    /// A JSON number.
    Num(f64),
    /// A JSON string.
    Str(String),
    /// A JSON array.
    Arr(Vec<J>),
    /// A JSON object as key and value pairs in the order written.
    Obj(Vec<(String, J)>),
}
impl J {
    /// The value of key `k` when this is an object that holds it.
    pub fn get(&self, k: &str) -> Option<&J> {
        if let J::Obj(v) = self {
            v.iter().find(|x| x.0 == k).map(|x| &x.1)
        } else {
            None
        }
    }
    /// The text when this is a string.
    pub fn str(&self) -> Option<&str> {
        if let J::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    /// The elements when this is an array, else an empty slice.
    pub fn arr(&self) -> &[J] {
        if let J::Arr(a) = self {
            a
        } else {
            &[]
        }
    }
    /// The key and value pairs when this is an object, else an empty slice.
    pub fn obj(&self) -> &[(String, J)] {
        if let J::Obj(a) = self {
            a
        } else {
            &[]
        }
    }
    /// The number with its fraction dropped when this is a number, else 0.
    pub fn int(&self) -> i64 {
        if let J::Num(n) = self {
            *n as i64
        } else {
            0
        }
    }
    /// The string elements of an array as owned strings; other elements are skipped.
    pub fn strs(&self) -> Vec<String> {
        self.arr()
            .iter()
            .filter_map(|x| x.str().map(|s| s.to_string()))
            .collect()
    }
}
/// Reads one JSON document; text after the value is refused.
pub fn parse(s: &str) -> Result<J, String> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let v = val(&c, &mut i)?;
    ws(&c, &mut i);
    if i != c.len() {
        return Err(format!("json: trailing text at {i}"));
    }
    Ok(v)
}
/// Moves `i` past white space.
fn ws(c: &[char], i: &mut usize) {
    while *i < c.len() && c[*i].is_whitespace() {
        *i += 1
    }
}
/// Reads the JSON value that starts at `i` and moves `i` past it.
fn val(c: &[char], i: &mut usize) -> Result<J, String> {
    ws(c, i);
    match c.get(*i) {
        None => Err("json: end".into()),
        Some('{') => object(c, i),
        Some('[') => array(c, i),
        Some('"') => string(c, i),
        Some('t') => {
            *i += 4;
            Ok(J::Bool(true))
        }
        Some('f') => {
            *i += 5;
            Ok(J::Bool(false))
        }
        Some('n') => {
            *i += 4;
            Ok(J::Null)
        }
        Some(_) => number(c, i),
    }
}
/// Reads an object whose `{` is at `i`.
fn object(c: &[char], i: &mut usize) -> Result<J, String> {
    *i += 1;
    let mut o = vec![];
    loop {
        ws(c, i);
        if c[*i] == '}' {
            *i += 1;
            return Ok(J::Obj(o));
        }
        let k = match val(c, i)? {
            J::Str(s) => s,
            _ => return Err("json: key".into()),
        };
        ws(c, i);
        if c[*i] != ':' {
            return Err("json: ':'".into());
        }
        *i += 1;
        o.push((k, val(c, i)?));
        ws(c, i);
        if c[*i] == ',' {
            *i += 1
        }
    }
}
/// Reads an array whose `[` is at `i`.
fn array(c: &[char], i: &mut usize) -> Result<J, String> {
    *i += 1;
    let mut a = vec![];
    loop {
        ws(c, i);
        if c[*i] == ']' {
            *i += 1;
            return Ok(J::Arr(a));
        }
        a.push(val(c, i)?);
        ws(c, i);
        if c[*i] == ',' {
            *i += 1
        }
    }
}
/// Reads a string whose opening quote is at `i`, resolving its escapes.
fn string(c: &[char], i: &mut usize) -> Result<J, String> {
    *i += 1;
    let mut s = String::new();
    while c[*i] != '"' {
        if c[*i] == '\\' {
            *i += 1;
            match c[*i] {
                'n' => s.push('\n'),
                't' => s.push('\t'),
                'r' => s.push('\r'),
                '/' => s.push('/'),
                '\\' => s.push('\\'),
                '"' => s.push('"'),
                'u' => {
                    let h: String = c[*i + 1..*i + 5].iter().collect();
                    s.push(
                        char::from_u32(u32::from_str_radix(&h, 16).map_err(|e| e.to_string())?)
                            .unwrap_or('?'),
                    );
                    *i += 4
                }
                x => return Err(format!("json: escape {x}")),
            }
        } else {
            s.push(c[*i])
        }
        *i += 1;
    }
    *i += 1;
    Ok(J::Str(s))
}
/// Reads a number that starts at `i`.
fn number(c: &[char], i: &mut usize) -> Result<J, String> {
    let st = *i;
    while *i < c.len() && (c[*i].is_ascii_digit() || "+-.eE".contains(c[*i])) {
        *i += 1
    }
    c[st..*i]
        .iter()
        .collect::<String>()
        .parse()
        .map(J::Num)
        .map_err(|_| "json: number".to_string())
}
