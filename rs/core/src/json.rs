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
/// Reads one JSON document; text after the value is refused. The text is read in place, never copied.
pub fn parse(s: &str) -> Result<J, String> {
    read(s, &Top::default())
}
/// A test that says whether an element of an array stays.
type Keep<'a> = &'a dyn Fn(&J) -> bool;
/// What a caller reads of the top-level object only: the keys to scan past, and one array whose elements are kept when a test says so.
#[derive(Default)]
struct Top<'a> {
    /// The keys whose values are scanned past.
    skip: &'a [&'a str],
    /// The key of the array to filter and the test an element must pass.
    keep: Option<(&'a str, Keep<'a>)>,
}
/// Reads one JSON document like `parse`; in a top-level object the values under the keys in `skip` are scanned past, not
/// built (the key is left out of the result), so a field the caller never reads costs no memory. A skipped value is
/// checked for its quotes and brackets only.
#[allow(dead_code)] // the build script that shares this file does not use it
pub fn parse_without(s: &str, skip: &[&str]) -> Result<J, String> {
    read(s, &Top { skip, keep: None })
}
/// Reads one JSON document like `parse`; the array under `key` in a top-level object is read one element at a time and an element
/// stays in the result only when `keep` accepts it, so the elements the caller never reads cost no memory.
#[allow(dead_code)] // the build script that shares this file does not use it
pub fn parse_keeping(s: &str, key: &str, keep: &dyn Fn(&J) -> bool) -> Result<J, String> {
    read(
        s,
        &Top {
            skip: &[],
            keep: Some((key, keep)),
        },
    )
}
/// Hands the elements of the array under `key` in a top-level object to `visit`, one at a time, each dropped before the next is read;
/// `visit` returns false to stop. Other keys are scanned past; a line without the key hands over nothing.
#[allow(dead_code)] // the build script that shares this file does not use it
pub fn each_in(s: &str, key: &str, visit: &mut dyn FnMut(J) -> bool) -> Result<(), String> {
    let (c, mut i) = (s.as_bytes(), 0);
    ws(s, &mut i);
    if c.get(i) != Some(&b'{') {
        return Err("json: object".into());
    }
    i += 1;
    loop {
        ws(s, &mut i);
        if c[i] == b'}' {
            return Ok(());
        }
        let k = match val(s, &mut i)? {
            J::Str(k) => k,
            _ => return Err("json: key".into()),
        };
        ws(s, &mut i);
        if c[i] != b':' {
            return Err("json: ':'".into());
        }
        i += 1;
        ws(s, &mut i);
        if k != key || c.get(i) != Some(&b'[') {
            skip_value(s, &mut i)?
        } else {
            i += 1;
            loop {
                ws(s, &mut i);
                if c[i] == b']' {
                    i += 1;
                    break;
                }
                if !visit(val(s, &mut i)?) {
                    return Ok(());
                }
                ws(s, &mut i);
                if c[i] == b',' {
                    i += 1
                }
            }
        }
        ws(s, &mut i);
        if c[i] == b',' {
            i += 1
        }
    }
}
/// Reads one document with the top-level choices in `top`.
fn read(s: &str, top: &Top) -> Result<J, String> {
    let mut i = 0;
    ws(s, &mut i);
    let v = if s.as_bytes().get(i) == Some(&b'{') {
        object(s, &mut i, top)?
    } else {
        val(s, &mut i)?
    };
    ws(s, &mut i);
    if i != s.len() {
        let at = s[..i.min(s.len())].chars().count() + i.saturating_sub(s.len());
        return Err(format!("json: trailing text at {at}"));
    }
    Ok(v)
}
/// Moves `i` past white space.
fn ws(s: &str, i: &mut usize) {
    let c = s.as_bytes();
    while *i < c.len() {
        if c[*i] < 0x80 {
            if !matches!(c[*i], b' ' | 9..=13) {
                return;
            }
            *i += 1
        } else {
            match s.get(*i..).and_then(|r| r.chars().next()) {
                Some(ch) if ch.is_whitespace() => *i += ch.len_utf8(),
                _ => return,
            }
        }
    }
}
/// Moves `i` past the value that starts at `i` without building it: a string, an object or array (to its closing
/// bracket), or a bare word up to the next `,`, `}` or `]`.
fn skip_value(s: &str, i: &mut usize) -> Result<(), String> {
    let c = s.as_bytes();
    ws(s, i);
    let (st, mut depth, mut quoted) = (*i, 0usize, false);
    while let Some(&b) = c.get(*i) {
        *i += 1;
        if quoted {
            match b {
                b'\\' => *i += 1,
                b'"' => quoted = false,
                _ => {}
            }
        } else {
            match b {
                b'"' => quoted = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' | b',' if depth == 0 => {
                    *i -= 1;
                    return Ok(());
                }
                b'}' | b']' => depth -= 1,
                _ => {}
            }
        }
        if depth == 0 && !quoted && matches!(b, b'"' | b'}' | b']') {
            return Ok(());
        }
    }
    if *i > st && depth == 0 && !quoted {
        Ok(())
    } else {
        Err("json: end".into())
    }
}
/// Moves `i` past `n` characters (a bare word is taken by its length, not checked); past the end, one byte stands for each missing one.
fn skip_chars(s: &str, i: &mut usize, n: usize) {
    let (mut bytes, mut seen) = (0, 0);
    for ch in s[*i..].chars().take(n) {
        (bytes, seen) = (bytes + ch.len_utf8(), seen + 1)
    }
    *i += bytes + (n - seen)
}
/// Reads the JSON value that starts at `i` and moves `i` past it.
fn val(s: &str, i: &mut usize) -> Result<J, String> {
    ws(s, i);
    match s.as_bytes().get(*i) {
        None => Err("json: end".into()),
        Some(b'{') => object(s, i, &Top::default()),
        Some(b'[') => array(s, i, None),
        Some(b'"') => string(s, i),
        Some(b't') => {
            skip_chars(s, i, 4);
            Ok(J::Bool(true))
        }
        Some(b'f') => {
            skip_chars(s, i, 5);
            Ok(J::Bool(false))
        }
        Some(b'n') => {
            skip_chars(s, i, 4);
            Ok(J::Null)
        }
        Some(_) => number(s, i),
    }
}
/// Reads an object whose `{` is at `i`; the values under the keys `top` skips are not built and its array is filtered.
fn object(s: &str, i: &mut usize, top: &Top) -> Result<J, String> {
    let c = s.as_bytes();
    *i += 1;
    let mut o = vec![];
    loop {
        ws(s, i);
        if c[*i] == b'}' {
            *i += 1;
            return Ok(J::Obj(o));
        }
        let k = match val(s, i)? {
            J::Str(s) => s,
            _ => return Err("json: key".into()),
        };
        ws(s, i);
        if c[*i] != b':' {
            return Err("json: ':'".into());
        }
        *i += 1;
        ws(s, i);
        if top.skip.contains(&k.as_str()) {
            skip_value(s, i)?
        } else if let (Some((name, keep)), Some(b'[')) = (top.keep, c.get(*i)) {
            let kept = if name == k { Some(keep) } else { None };
            o.push((k, array(s, i, kept)?));
        } else {
            o.push((k, val(s, i)?));
        }
        ws(s, i);
        if c[*i] == b',' {
            *i += 1
        }
    }
}
/// Reads an array whose `[` is at `i`; with `keep`, an element that it refuses is dropped as soon as it is read.
fn array(s: &str, i: &mut usize, keep: Option<Keep>) -> Result<J, String> {
    let c = s.as_bytes();
    *i += 1;
    let mut a = vec![];
    loop {
        ws(s, i);
        if c[*i] == b']' {
            *i += 1;
            return Ok(J::Arr(a));
        }
        let v = val(s, i)?;
        if keep.is_none_or(|k| k(&v)) {
            a.push(v)
        }
        ws(s, i);
        if c[*i] == b',' {
            *i += 1
        }
    }
}
/// Reads a string whose opening quote is at `i`, resolving its escapes.
fn string(s: &str, i: &mut usize) -> Result<J, String> {
    let c = s.as_bytes();
    *i += 1;
    let mut out = String::new();
    while c[*i] != b'"' {
        if c[*i] == b'\\' {
            *i += 1;
            match c[*i] {
                b'n' => out.push('\n'),
                b't' => out.push('\t'),
                b'r' => out.push('\r'),
                b'/' => out.push('/'),
                b'\\' => out.push('\\'),
                b'"' => out.push('"'),
                b'u' => {
                    let digits: Vec<usize> =
                        s[*i + 1..].chars().take(4).map(char::len_utf8).collect();
                    if digits.len() < 4 {
                        return Err("json: escape u".into());
                    }
                    let n = digits.iter().sum::<usize>();
                    let h = &s[*i + 1..*i + 1 + n];
                    out.push(
                        char::from_u32(u32::from_str_radix(h, 16).map_err(|e| e.to_string())?)
                            .unwrap_or('?'),
                    );
                    *i += n
                }
                _ => {
                    let x = s[*i..].chars().next().unwrap_or('?');
                    return Err(format!("json: escape {x}"));
                }
            }
            *i += 1;
        } else {
            let end = c[*i..]
                .iter()
                .position(|b| *b == b'"' || *b == b'\\')
                .map_or(c.len(), |n| *i + n);
            out.push_str(&s[*i..end]);
            *i = end;
        }
    }
    *i += 1;
    Ok(J::Str(out))
}
/// Reads a number that starts at `i`.
fn number(s: &str, i: &mut usize) -> Result<J, String> {
    let c = s.as_bytes();
    let st = *i;
    while *i < c.len() && (c[*i].is_ascii_digit() || b"+-.eE".contains(&c[*i])) {
        *i += 1
    }
    s[st..*i]
        .parse()
        .map(J::Num)
        .map_err(|_| "json: number".to_string())
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A skipped key is left out whatever it holds, and the other keys read as usual.
    #[test]
    fn skipped_keys_are_left_out() {
        let t = r#"{"a":1,"big":[{"x":"q\"]}"},[1,2],"z"],"b":{"c":"d"},"big2":12,"e":3}"#;
        let v = parse_without(t, &["big", "big2"]).unwrap();
        assert_eq!(v.obj().len(), 3);
        assert_eq!(
            v.get("b").and_then(|b| b.get("c")).and_then(J::str),
            Some("d")
        );
        assert!(v.get("big").is_none() && v.get("big2").is_none());
        assert!(parse_without(r#"{"big":"#, &["big"]).is_err());
    }

    /// Only the elements of the named array that the test accepts stay; other arrays are read whole.
    #[test]
    fn an_array_is_filtered_as_it_is_read() {
        let t = r#"{"b":[1,2,3],"a":[{"n":1},{"n":2},{"n":3}],"c":0}"#;
        let v = parse_keeping(t, "a", &|e| e.get("n").is_some_and(|n| n.int() != 2)).unwrap();
        assert_eq!(v.get("a").unwrap().arr().len(), 2);
        assert_eq!(v.get("b").unwrap().arr().len(), 3);
        assert_eq!(v.get("c").map(J::int), Some(0));
    }

    /// The elements of one array reach the caller one at a time, and the caller can stop the walk.
    #[test]
    fn an_array_is_visited_element_by_element() {
        let t = r#"{"b":[9],"a":[1,{"n":2},3],"c":0}"#;
        let mut seen = vec![];
        each_in(t, "a", &mut |e| {
            seen.push(format!("{e:?}"));
            seen.len() < 2
        })
        .unwrap();
        assert_eq!(seen.len(), 2);
        let mut all = 0;
        each_in(t, "a", &mut |_| {
            all += 1;
            true
        })
        .unwrap();
        assert_eq!(all, 3);
    }

    /// The reader takes text in place: escapes, wide characters and the position of trailing text count characters.
    #[test]
    fn text_is_read_in_place() {
        let v = parse(r#"{"k":"é\n日"}"#).unwrap();
        assert_eq!(v.get("k").and_then(J::str), Some("é\n日"));
        assert_eq!(parse("1 日").unwrap_err(), "json: trailing text at 2");
    }
}
