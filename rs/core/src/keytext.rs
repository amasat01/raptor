//! The key text: the exact text a value takes when it is part of a node's dedup identity. The walk digest is taken over these
//! texts, so the rules below are part of the digest's definition. A kind of value outside this list is refused by name.
//!
//! - `None` is `None`; a boolean is `True` or `False`; an integer is its decimal digits, with a leading `-` when negative.
//! - A float is the shortest digit string that reads back to the same value (the correctly rounded one when it reads back). `nan`,
//!   `inf` and `-inf` are written so; zero is `0.0` or `-0.0`. With `decpt` the position of the decimal point relative to the digits,
//!   when `-4 < decpt <= 16` the plain form is written (`0.001`, `1234.5`, `1000000000000000.0`; `1e+16` is past it), else `d.ddde+XX` or `de-XX`:
//!   one digit, the rest after a point when there are more, `e`, the sign always, the exponent in at least two digits.
//! - A string is in single quotes, or in double quotes when it holds a single quote and no double quote. Inside, a backslash is
//!   written as two backslashes, the quote in use as a backslash and the quote, and tab, line feed and carriage return as `\t`,
//!   `\n` and `\r`. Any other code point is written as it is when it is printable, else as `\xNN` below U+0100, `\uNNNN` below
//!   U+10000 and `\UNNNNNNNN` above, in lower-case hex. Printable means U+0020 to U+007E, and above U+007F every code point that is
//!   not in `nonprintable::NONPRINTABLE`.
//! - A tuple is `(a, b)` with `, ` between elements, `(a,)` for one element, `()` for none; a list is `[a, b]`, `[a]`, `[]`.
//! - The extended form `Val` adds a node reference `#n` (n decimal) and a record `Name(field=value, ...)`.
//!
//! `parse` reads back exactly what `write_val` writes.
use crate::nonprintable::NONPRINTABLE;

#[derive(Clone, Debug, PartialEq)]
/// A value of the key text without node references or records.
pub enum V {
    /// The null value, written `None`.
    None,
    /// A boolean, written `True` or `False`.
    Bool(bool),
    /// An integer, written in decimal.
    Int(i128),
    /// A float, written by `float_repr`.
    Float(f64),
    /// A string, quoted and escaped by `str_repr`.
    Str(String),
    /// Text that is already in the key text, copied as it is.
    Text(String),
    /// A tuple, written `(a, b)`, or `(a,)` for one element.
    Tuple(Vec<V>),
    /// A list, written `[a, b]`.
    List(Vec<V>),
}

/// A float as the shortest text that reads back to the same value, in the layout described above: `inf`, `-inf`, `nan`, `-0.0`, a plain
/// decimal when the decimal exponent lies in (-4, 16], else `d.ddde+XX` (exponent sign always, at least two digits).
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.into();
    }
    let a = x.abs();
    let mut e = format!("{:e}", a); // shortest round-trip digit COUNT: "d[.ddd]e<exp>"
                                    // among the strings of that length, the one closest to the value is the correctly rounded one (the shortest-digits routine may pick
                                    // another that also reads back, e.g. at 17 digits); use it when it reads back, so the text is the correctly rounded one
    let n = e
        .split_once('e')
        .unwrap()
        .0
        .chars()
        .filter(|c| *c != '.')
        .count();
    let exact = format!("{:.*e}", n - 1, a);
    if exact.parse::<f64>() == Ok(a) {
        e = exact
    }
    let (m, ex) = e.split_once('e').unwrap();
    let ex: i32 = ex.parse().unwrap();
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let decpt = ex + 1; // position of the decimal point relative to the digits
    let n = digits.len() as i32;
    let body = if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..decpt as usize],
                &digits[decpt as usize..]
            )
        }
    } else {
        let mant = if n == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let e = decpt - 1;
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    };
    format!("{sign}{body}")
}

/// Whether the key text writes code point `c` as it is: space and U+0021 to U+007E are printable, the controls U+0000 to
/// U+001F and U+007F are not, and above that a code point is printable unless it lies in the table of non-printable ranges.
fn printable(c: u32) -> bool {
    if c == 0x20 {
        return true;
    }
    if c < 0x20 || c == 0x7f {
        return false;
    }
    if c < 0x7f {
        return true;
    }
    NONPRINTABLE
        .binary_search_by(|&(a, b)| {
            if c < a {
                std::cmp::Ordering::Greater
            } else if c > b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_err()
}

/// Appends the key text of a string to `out`: in single quotes (double quotes when it holds a single quote and no double quote); `\\`, the quote, `\t \n \r` escaped;
/// any other non-printable code point as `\xNN`, `\uNNNN` or `\UNNNNNNNN` (lower-case hex), the rest as it is.
pub fn str_repr(s: &str, out: &mut String) {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(q);
    for c in s.chars() {
        let u = c as u32;
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c)
            }
            c if printable(u) => out.push(c),
            _ if u < 0x100 => out.push_str(&format!("\\x{u:02x}")),
            _ if u < 0x10000 => out.push_str(&format!("\\u{u:04x}")),
            _ => out.push_str(&format!("\\U{u:08x}")),
        }
    }
    out.push(q);
}

/// Writes the elements of `v` between `open` and `close`, separated by `, `; a one-element tuple gets a trailing comma
/// (`one_comma`).
fn seq(v: &[V], open: char, close: char, one_comma: bool, out: &mut String) {
    out.push(open);
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            out.push_str(", ")
        }
        write(x, out)
    }
    if one_comma && v.len() == 1 {
        out.push(',')
    }
    out.push(close);
}

/// Appends the key text of `v` to `out`.
pub fn write(v: &V, out: &mut String) {
    match v {
        V::None => out.push_str("None"),
        V::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        V::Int(i) => out.push_str(&i.to_string()),
        V::Float(f) => out.push_str(&float_repr(*f)),
        V::Str(s) => str_repr(s, out),
        V::Text(t) => out.push_str(t),
        V::Tuple(t) => seq(t, '(', ')', true, out),
        V::List(l) => seq(l, '[', ']', false, out),
    }
}

/// The key text of `v` as a new string.
pub fn text(v: &V) -> String {
    let mut o = String::new();
    write(v, &mut o);
    o
}

/// The key text of a string, as a new string (the same rule as `str_repr`).
pub fn py_str(s: &str) -> String {
    let mut o = String::new();
    str_repr(s, &mut o);
    o
}

/// A value of the key text in full: the kinds of `V` plus a node reference `#h` and a record `Name(f=v, ...)`. It is the value of a
/// node's field; `write_val` writes it and `parse` reads it back, byte for byte.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    /// The null value, written `None`.
    None,
    /// A boolean, written `True` or `False`.
    Bool(bool),
    /// An integer, written in decimal.
    Int(i128),
    /// A float, written by `float_repr`.
    Float(f64),
    /// A string, quoted and escaped by `str_repr`.
    Str(String),
    /// A tuple, written `(a, b)`, or `(a,)` for one element.
    Tuple(Vec<Val>),
    /// A list, written `[a, b]`.
    List(Vec<Val>),
    /// A reference to a node by its number, written `#n`.
    Node(u32),
    /// A record, written `Name(field=value, ...)`: its name and its fields in order.
    Rec(String, Vec<(String, Val)>),
}

/// Writes the elements of `v` between `open` and `close`, as `seq` does for a `V`.
fn seq_val(v: &[Val], open: char, close: char, one_comma: bool, out: &mut String) {
    out.push(open);
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            out.push_str(", ")
        }
        write_val(x, out)
    }
    if one_comma && v.len() == 1 {
        out.push(',')
    }
    out.push(close);
}

/// Appends the key text of `v` to `out`.
pub fn write_val(v: &Val, out: &mut String) {
    match v {
        Val::None => out.push_str("None"),
        Val::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Val::Int(i) => out.push_str(&i.to_string()),
        Val::Float(f) => out.push_str(&float_repr(*f)),
        Val::Str(s) => str_repr(s, out),
        Val::Tuple(t) => seq_val(t, '(', ')', true, out),
        Val::List(l) => seq_val(l, '[', ']', false, out),
        Val::Node(h) => {
            out.push('#');
            out.push_str(&h.to_string())
        }
        Val::Rec(n, fs) => {
            out.push_str(n);
            out.push('(');
            for (i, (k, x)) in fs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ")
                }
                out.push_str(k);
                out.push('=');
                write_val(x, out)
            }
            out.push(')');
        }
    }
}

/// The key text of `v` as a new string.
pub fn text_val(v: &Val) -> String {
    let mut o = String::new();
    write_val(v, &mut o);
    o
}

/// The deepest nesting of tuples, lists and records `parse` reads; deeper text is refused.
const MAX_DEPTH: usize = 200;

/// The reader state of `parse`: the text and the byte position in it.
struct Rd<'a> {
    /// The whole text being read.
    s: &'a str,
    /// The byte position of the next character.
    i: usize,
}

impl<'a> Rd<'a> {
    /// Moves past white space.
    fn ws(&mut self) {
        while self.s[self.i..].starts_with(|c: char| c.is_ascii_whitespace()) {
            self.i += 1
        }
    }
    /// The character at the position, if any.
    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }
    /// Moves past `c` and returns true when it is the next character.
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    /// An error that names what was expected and the byte position.
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("value text: {what} at {}", self.i))
    }
    /// The longest run of characters, from the position, that satisfy `ok`; moves past it.
    fn run(&mut self, ok: impl Fn(char) -> bool) -> &'a str {
        let st = self.i;
        while self.s[self.i..].starts_with(&ok) {
            self.i += 1
        }
        &self.s[st..self.i]
    }
    /// Reads the elements up to the `close` bracket; also reports whether the last element was followed by a comma.
    fn items(&mut self, close: char, d: usize) -> Result<(Vec<Val>, bool), String> {
        let mut v = vec![];
        let mut comma = false;
        loop {
            self.ws();
            if self.eat(close) {
                return Ok((v, comma));
            }
            v.push(self.val(d + 1)?);
            self.ws();
            comma = self.eat(',');
            if !comma {
                self.ws();
                if self.eat(close) {
                    return Ok((v, false));
                }
                return self.err("expected ',' or the closing bracket");
            }
        }
    }
    /// Reads one value at nesting depth `d`.
    fn val(&mut self, d: usize) -> Result<Val, String> {
        if d > MAX_DEPTH {
            return self.err("nesting too deep");
        }
        self.ws();
        let Some(c) = self.peek() else {
            return self.err("unexpected end");
        };
        match c {
            '\'' | '"' => self.string(c).map(Val::Str),
            '(' => {
                self.i += 1;
                let (v, comma) = self.items(')', d)?;
                if v.len() == 1 && !comma {
                    return self.err("a 1-tuple is written (x,)");
                }
                Ok(Val::Tuple(v))
            }
            '[' => {
                self.i += 1;
                Ok(Val::List(self.items(']', d)?.0))
            }
            '#' => {
                self.i += 1;
                let t = self.run(|c| c.is_ascii_digit());
                t.parse::<u32>()
                    .map(Val::Node)
                    .or_else(|_| self.err("a node reference is #<int>"))
            }
            '-' | '0'..='9' => {
                let t = self.run(|c| c.is_ascii_alphanumeric() || c == '.' || c == '+' || c == '-');
                if t == "-inf" {
                    return Ok(Val::Float(f64::NEG_INFINITY));
                }
                if t.contains(['.', 'e', 'E']) {
                    if !t.chars().all(|c| c.is_ascii_digit() || ".eE+-".contains(c)) {
                        return self.err("bad float");
                    }
                    t.parse::<f64>()
                        .map(Val::Float)
                        .or_else(|_| self.err("bad float"))
                } else {
                    t.parse::<i128>()
                        .map(Val::Int)
                        .or_else(|_| self.err("bad integer (up to 128 bits)"))
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let id = self.run(|c| c.is_ascii_alphanumeric() || c == '_');
                if self.eat('(') {
                    return self.record(id, d);
                }
                match id {
                    "None" => Ok(Val::None),
                    "True" => Ok(Val::Bool(true)),
                    "False" => Ok(Val::Bool(false)),
                    "inf" => Ok(Val::Float(f64::INFINITY)),
                    "nan" => Ok(Val::Float(f64::NAN)),
                    _ => self.err(&format!("unknown token {id}")),
                }
            }
            _ => self.err("unexpected character"),
        }
    }
    /// Reads the fields of a record named `name`, after its opening parenthesis.
    fn record(&mut self, name: &str, d: usize) -> Result<Val, String> {
        let mut fs = vec![];
        self.ws();
        if self.eat(')') {
            return Ok(Val::Rec(name.into(), fs));
        }
        loop {
            self.ws();
            let k = self.run(|c| c.is_ascii_alphanumeric() || c == '_');
            if k.is_empty() {
                return self.err("a field name");
            }
            self.ws();
            if !self.eat('=') {
                return self.err("expected '='");
            }
            fs.push((k.to_string(), self.val(d + 1)?));
            self.ws();
            if self.eat(')') {
                return Ok(Val::Rec(name.into(), fs));
            }
            if !self.eat(',') {
                return self.err("expected ',' or ')'");
            }
        }
    }
    /// Reads a string whose opening quote `q` is at the position, resolving its escapes.
    fn string(&mut self, q: char) -> Result<String, String> {
        self.i += 1;
        let mut o = String::new();
        loop {
            let Some(c) = self.peek() else {
                return self.err("unterminated string");
            };
            self.i += c.len_utf8();
            if c == q {
                return Ok(o);
            }
            if c != '\\' {
                o.push(c);
                continue;
            }
            let Some(e) = self.peek() else {
                return self.err("unterminated string");
            };
            self.i += e.len_utf8();
            let n = match e {
                'x' => 2,
                'u' => 4,
                'U' => 8,
                _ => 0,
            };
            if n == 0 {
                o.push(match e {
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return self.err("unknown escape"),
                });
                continue;
            }
            let h = self
                .s
                .get(self.i..self.i + n)
                .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()));
            let Some(h) = h else {
                return self.err("bad hex escape");
            };
            self.i += n;
            match char::from_u32(u32::from_str_radix(h, 16).unwrap()) {
                Some(c) => o.push(c),
                None => return self.err("escape is not a code point"),
            }
        }
    }
}

/// Reads one value of the key text (the grammar of `write_val`); text after the value is refused.
pub fn parse(s: &str) -> Result<Val, String> {
    let mut r = Rd { s, i: 0 };
    let v = r.val(0)?;
    r.ws();
    if r.i != s.len() {
        return r.err("trailing text");
    }
    Ok(v)
}
