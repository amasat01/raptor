//! The affine form of an integer index expression in an index variable, over exact rationals: how a lane's index moves with
//! the loop index or the sample index. Owns the C door `rx_index_form`.
use crate::door::{guard, put};
use crate::record::{Ctx, Fault};
use crate::value::Val;
use std::collections::HashSet;
use std::ffi::{c_char, c_int};

/// An exact rational, always reduced, with a positive denominator.
#[derive(Clone, Copy)]
struct Q {
    /// Numerator.
    n: i128,
    /// Denominator, positive.
    d: i128,
}

/// The greatest common divisor of two integers, never negative.
fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

impl Q {
    /// The reduced rational `n / d`.
    fn new(n: i128, d: i128) -> Q {
        let g = gcd(n, d).max(1);
        let sign = if d < 0 { -1 } else { 1 };
        Q {
            n: sign * n / g,
            d: sign * d / g,
        }
    }

    /// The integer `n` as a rational.
    fn int(n: i128) -> Q {
        Q { n, d: 1 }
    }

    /// The sum of two rationals.
    fn add(self, o: Q) -> Q {
        Q::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }

    /// The product of two rationals.
    fn mul(self, o: Q) -> Q {
        Q::new(self.n * o.n, self.d * o.d)
    }

    /// Whether the rational is an integer.
    fn exact(self) -> bool {
        self.d == 1
    }

    /// The written form `n/d`.
    fn text(self) -> String {
        format!("{}/{}", self.n, self.d)
    }
}

/// How an index expression moves with the index variable: `coef * s + off`, or, when `coef` is not an integer, only through
/// `floor(coef * s + off)`.
struct Form {
    /// The coefficient of the variable.
    coef: Q,
    /// The part free of the variable; none when it is not a compile-time integer.
    off: Option<Q>,
    /// The declaration-named constants the coefficient is scaled by.
    labels: Vec<String>,
    /// Whether a bare literal also scales it.
    anon: bool,
    /// `n_samples` when the coefficient is scaled by the launch's sample count, else empty.
    symbol: String,
}

impl Form {
    /// A form with no label, no literal scaling and no symbol.
    fn plain(coef: Q, off: Option<Q>) -> Form {
        Form {
            coef,
            off,
            labels: vec![],
            anon: false,
            symbol: String::new(),
        }
    }
}

/// The question: an expression, the variable it is read against (none: the sample index) and whether quotients count.
struct Asker<'a> {
    /// The context holding the nodes.
    ctx: &'a Ctx,
    /// The index variable, matched by identity; none means the sample index, matched by kind.
    var: Option<u32>,
}

impl Asker<'_> {
    /// Whether `n` is the index variable.
    fn is_var(&self, n: u32) -> bool {
        match self.var {
            None => self.ctx.is_a(n, "SampleIndex"),
            Some(v) => n == v,
        }
    }

    /// Whether the variable occurs under `n`.
    fn carries(&self, n: u32) -> Result<bool, Fault> {
        let (mut pending, mut seen) = (vec![n], HashSet::new());
        while let Some(cur) = pending.pop() {
            if !seen.insert(cur) {
                continue;
            }
            if self.is_var(cur) {
                return Ok(true);
            }
            pending.extend(self.ctx.node(cur)?.ops.iter());
        }
        Ok(false)
    }

    /// The value and label of a compile-time integer node: a role constant, or an `i32` / `i64` constant (no label).
    fn int_factor(&self, n: u32) -> Result<Option<(i128, Option<String>)>, Fault> {
        let literal = |h| match self.ctx.field(h, "literal") {
            Ok(Val::Int(v)) => Some(v),
            _ => None,
        };
        if self.ctx.is_a(n, "RoleConst") {
            let label = match self.ctx.field(n, "label")? {
                Val::Str(s) => Some(s),
                _ => None,
            };
            return Ok(literal(n).map(|v| (v, label)));
        }
        let dtype = &self.ctx.node(n)?.dtype;
        if self.ctx.is_a(n, "Const") && (dtype == "i32" || dtype == "i64") {
            return Ok(literal(n).map(|v| (v, None)));
        }
        Ok(None)
    }

    /// The form of `node`, or none when no form is claimed.
    fn form(&self, node: u32, quotients: bool) -> Result<Option<Form>, Fault> {
        if !self.carries(node)? {
            let off = self.int_factor(node)?.map(|v| Q::int(v.0));
            return Ok(Some(Form::plain(Q::int(0), off)));
        }
        if self.is_var(node) {
            return Ok(Some(Form::plain(Q::int(1), Some(Q::int(0)))));
        }
        if !self.ctx.is_a(node, "Op") {
            return Ok(None);
        }
        let kind = match self.ctx.field(node, "kind")? {
            Val::Str(k) => k,
            _ => return Ok(None),
        };
        let ops = self.ctx.node(node)?.ops.clone();
        match kind.as_str() {
            "add" | "sub" => self.sum_form(&kind, &ops, quotients),
            "mul" => self.product_form(&ops, quotients),
            "div" if quotients => self.quotient_form(node, &ops),
            _ => Ok(None),
        }
    }

    /// The form of `a + b` or `a - b`: exact forms add; a quotient shifted by a free term keeps its own.
    fn sum_form(&self, kind: &str, ops: &[u32], quotients: bool) -> Result<Option<Form>, Fault> {
        let (Some(l), Some(r)) = (self.form(ops[0], quotients)?, self.form(ops[1], quotients)?)
        else {
            return Ok(None);
        };
        let sign = if kind == "add" { 1 } else { -1 };
        if l.coef.exact() && r.coef.exact() {
            let off = match (l.off, r.off) {
                (Some(a), Some(b)) => Some(a.add(b.mul(Q::int(sign)))),
                _ => None,
            };
            let symbol = if l.symbol.is_empty() {
                r.symbol
            } else {
                l.symbol
            };
            let labels = l.labels.into_iter().chain(r.labels).collect();
            return Ok(Some(Form {
                coef: l.coef.add(r.coef.mul(Q::int(sign))),
                off,
                labels,
                anon: l.anon || r.anon,
                symbol,
            }));
        }
        let right_is_quotient = if l.coef.n == 0 {
            true
        } else if r.coef.n == 0 {
            false
        } else {
            return Ok(None);
        };
        if sign == -1 && right_is_quotient {
            return Ok(None);
        }
        Ok(Some(if right_is_quotient { r } else { l }))
    }

    /// The form of `a * k`: a compile-time factor scales an exact form; a quotient keeps its coefficient; the sample count
    /// scales by a run-time stride.
    fn product_form(&self, ops: &[u32], quotients: bool) -> Result<Option<Form>, Fault> {
        let (a_has, b_has) = (self.carries(ops[0])?, self.carries(ops[1])?);
        if a_has && b_has {
            return Ok(None);
        }
        let (term, factor_node) = if a_has {
            (ops[0], ops[1])
        } else {
            (ops[1], ops[0])
        };
        let is_sample_count = self.ctx.is_a(factor_node, "Leaf")
            && matches!(self.ctx.field(factor_node, "kind"), Ok(Val::Str(k)) if k == "nsamples");
        if is_sample_count {
            let Some(inner) = self.form(term, quotients)? else {
                return Ok(None);
            };
            if !inner.coef.exact() || !inner.symbol.is_empty() {
                return Ok(None);
            }
            return Ok(Some(Form {
                off: None,
                symbol: "n_samples".into(),
                ..inner
            }));
        }
        let Some((factor, label)) = self.int_factor(factor_node)? else {
            return Ok(None);
        };
        let Some(inner) = self.form(term, quotients)? else {
            return Ok(None);
        };
        let anon = inner.anon || label.is_none();
        let mut labels = inner.labels;
        labels.extend(label);
        let scale = Q::int(factor);
        let (coef, off) = if inner.coef.exact() {
            (inner.coef.mul(scale), inner.off.map(|o| o.mul(scale)))
        } else if factor == 0 {
            (Q::int(0), Some(Q::int(0)))
        } else {
            (inner.coef, inner.off)
        };
        Ok(Some(Form {
            coef,
            off,
            labels,
            anon,
            symbol: String::new(),
        }))
    }

    /// The form of `a / k` of a non-negative exact form by a positive compile-time `k`: truncating division then equals floor.
    fn quotient_form(&self, node: u32, ops: &[u32]) -> Result<Option<Form>, Fault> {
        let dtype = &self.ctx.node(node)?.dtype;
        if dtype != "i32" && dtype != "i64" {
            return Ok(None);
        }
        let (num, div) = (ops[0], ops[1]);
        if self.carries(div)? {
            return Ok(None);
        }
        let Some((k, label)) = self.int_factor(div)? else {
            return Ok(None);
        };
        if k <= 0 {
            return Ok(None);
        }
        let Some(inner) = self.form(num, true)? else {
            return Ok(None);
        };
        let Some(off) = inner.off else {
            return Ok(None);
        };
        if !inner.coef.exact() || inner.coef.n <= 0 || off.n < 0 {
            return Ok(None);
        }
        let anon = inner.anon || label.is_none();
        let mut labels = inner.labels;
        labels.extend(label);
        let coef = Q::new(inner.coef.n, inner.coef.d * k);
        let off = if coef.exact() {
            Q::int((off.n / off.d) / k)
        } else {
            Q::new(off.n, off.d * k)
        };
        Ok(Some(Form {
            coef,
            off: Some(off),
            labels,
            anon,
            symbol: String::new(),
        }))
    }
}

impl Ctx {
    /// The affine form of `node` in `var` (none: the sample index), or none when no form is claimed. The answer is the
    /// texts `form`, the coefficient `n/d`, the offset `n/d` or `-`, `1` when a bare literal scales it else `0`, the symbol
    /// or `-`, then the labels; or the one text `none`. Conservative: `none` claims nothing.
    pub fn index_form(
        &self,
        node: u32,
        var: Option<u32>,
        quotients: bool,
    ) -> Result<Vec<String>, Fault> {
        self.node(node)?;
        if let Some(v) = var {
            self.node(v)?;
        }
        let Some(f) = (Asker { ctx: self, var }).form(node, quotients)? else {
            return Ok(vec!["none".into()]);
        };
        let dash = |s: String| if s.is_empty() { "-".into() } else { s };
        let mut out = vec![
            "form".to_string(),
            f.coef.text(),
            f.off.map_or("-".into(), Q::text),
            u8::from(f.anon).to_string(),
            dash(f.symbol),
        ];
        out.extend(f.labels);
        Ok(out)
    }
}

/// The affine form of an index expression (see `Ctx::index_form`), as the packed texts of the names protocol.
///
/// # Safety
/// Every pointer must be valid for the length given beside it, as the header states.
#[no_mangle]
pub unsafe extern "C" fn rx_index_form(
    ctx: *mut Ctx,
    node: u32,
    var: *const u32,
    quotients: c_int,
    buf: *mut c_char,
    cap: usize,
    len: *mut usize,
    count: *mut usize,
) -> c_int {
    guard(ctx, |c| {
        if count.is_null() {
            return Err(Fault::BadArg("count is NULL".into()));
        }
        let var = if var.is_null() {
            None
        } else {
            Some(unsafe { *var })
        };
        let texts = c.index_form(node, var, quotients != 0)?;
        unsafe { *count = texts.len() };
        let packed: String = texts.iter().flat_map(|t| [t.as_str(), "\0"]).collect();
        put(buf, cap, len, &packed)
    })
}
