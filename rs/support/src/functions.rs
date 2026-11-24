//! Constructors of functions and calls for tests and for the writers that consume them: the pieces (a parameter, a declaration,
//! a call) and three reference documents, `twice`, `native_with_rules` and `nested`, whose stored forms are under
//! `rs/support/tests/data/functions/` in the dump format.
use crate::groups::{apply, commit, plane_in};
use crate::groups::{build, nodes, text};
use raptor_ir::record::Ctx;
use raptor_ir::value::{ttype_val, Val};

/// A parameter `name` of element type `dtype`, passed by value.
pub fn param(ctx: &mut Ctx, name: &str, dtype: &str) -> Result<u32, String> {
    let given = vec![("name", text(name)), ("ttype", ttype_val(dtype, &[], None))];
    build(ctx, "Param", given)
}

/// The facts of a function declaration; `Declaration::default()` states a lane function with no body and no text.
#[derive(Default, Clone)]
pub struct Declaration<'a> {
    /// The name the table and the calls know it by.
    pub name: &'a str,
    /// `lane` (the default when empty) or `group`.
    pub scope: &'a str,
    /// The `Param` nodes.
    pub params: &'a [u32],
    /// The element type of the result.
    pub returns: &'a str,
    /// The root of a traced body.
    pub body: Option<u32>,
    /// The native texts as (target name, text) pairs.
    pub natives: &'a [(&'a str, &'a str)],
    /// The forward rule.
    pub jvp: Option<u32>,
    /// The reverse rules, one per read.
    pub vjp: &'a [u32],
}

/// The function a declaration states, entered in the table of the context.
pub fn declare(ctx: &mut Ctx, d: &Declaration) -> Result<u32, String> {
    let natives = d
        .natives
        .iter()
        .map(|(target, body)| Val::Tuple(vec![text(target), text(body)]))
        .collect();
    let mut given = vec![
        ("name", text(d.name)),
        ("params", nodes(d.params)),
        ("returns", ttype_val(d.returns, &[], None)),
        ("natives", Val::Tuple(natives)),
        ("body", d.body.map_or(Val::None, Val::Node)),
        ("jvp", d.jvp.map_or(Val::None, Val::Node)),
        ("vjp", nodes(d.vjp)),
    ];
    if !d.scope.is_empty() {
        given.push(("scope", text(d.scope)));
    }
    build(ctx, "Function", given)
}

/// The name and digest by which a call names the function `function`.
pub fn reference(ctx: &Ctx, function: u32) -> Result<Val, String> {
    let name = ctx
        .str_field(function, "name")
        .map_err(|e| format!("{e:?}"))?;
    let digest = ctx
        .function_digest(function)
        .map_err(|e| format!("{e:?}"))?;
    Ok(Val::Tuple(vec![Val::Str(name), Val::Str(digest)]))
}

/// A call of the function `function` with `args` in parameter order.
pub fn call(ctx: &mut Ctx, function: u32, args: &[u32]) -> Result<u32, String> {
    let given = vec![
        ("function", reference(ctx, function)?),
        ("args", nodes(args)),
    ];
    build(ctx, "Call", given)
}

/// A call that names `function` as given, whatever the table holds.
pub fn call_named(ctx: &mut Ctx, function: Val, args: &[u32]) -> Result<u32, String> {
    build(
        ctx,
        "Call",
        vec![("function", function), ("args", nodes(args))],
    )
}

/// A reference document: the nodes to dump together, and the name of its function.
pub struct Fixture {
    /// The roots of the stored document.
    pub roots: Vec<u32>,
    /// The function the document is about.
    pub function: u32,
}

/// `twice`: the traced lane function `weighted(x, y) = x * x + y`, called twice by a kernel that assigns `out = weighted(a, b) +
/// weighted(b, a)`. The two calls name one function; the document holds it once.
pub fn twice(ctx: &mut Ctx) -> Result<Fixture, String> {
    let (x, y) = (param(ctx, "x", "f64")?, param(ctx, "y", "f64")?);
    let square = apply(ctx, "mul", &[x, x])?;
    let body = apply(ctx, "add", &[square, y])?;
    let weighted = declare(
        ctx,
        &Declaration {
            name: "weighted",
            params: &[x, y],
            returns: "f64",
            body: Some(body),
            ..Default::default()
        },
    )?;
    let (a, b) = (plane_in(ctx, "a", "f64")?, plane_in(ctx, "b", "f64")?);
    let first = call(ctx, weighted, &[a, b])?;
    let second = call(ctx, weighted, &[b, a])?;
    let sum = apply(ctx, "add", &[first, second])?;
    let out = commit(ctx, "out", sum)?;
    Ok(Fixture {
        roots: vec![out],
        function: weighted,
    })
}

/// `native_with_rules`: the native lane function `product(x, y)` with a text for each of the targets `t1` and `t2`, a forward
/// rule `y * dx + x * dy` and one reverse rule per read, `y * bar` for `x` and `x * bar` for `y`, all traced functions of the
/// same table.
pub fn native_with_rules(ctx: &mut Ctx) -> Result<Fixture, String> {
    let [x, y, dx, dy, bar] = ["x", "y", "dx", "dy", "bar"].map(|n| param(ctx, n, "f64"));
    let (x, y, dx, dy, bar) = (x?, y?, dx?, dy?, bar?);
    let along_x = apply(ctx, "mul", &[y, dx])?;
    let along_y = apply(ctx, "mul", &[x, dy])?;
    let tangent = apply(ctx, "add", &[along_x, along_y])?;
    let rule = |ctx: &mut Ctx, name, params: &[u32], body| {
        declare(
            ctx,
            &Declaration {
                name,
                params,
                returns: "f64",
                body: Some(body),
                ..Default::default()
            },
        )
    };
    let jvp = rule(ctx, "product_jvp", &[x, y, dx, dy], tangent)?;
    let for_x = apply(ctx, "mul", &[y, bar])?;
    let for_y = apply(ctx, "mul", &[x, bar])?;
    let vjp_x = rule(ctx, "product_vjp_x", &[x, y, bar], for_x)?;
    let vjp_y = rule(ctx, "product_vjp_y", &[x, y, bar], for_y)?;
    let function = declare(
        ctx,
        &Declaration {
            name: "product",
            params: &[x, y],
            returns: "f64",
            natives: &[("t1", "return x * y;"), ("t2", "return x * y")],
            jvp: Some(jvp),
            vjp: &[vjp_x, vjp_y],
            ..Default::default()
        },
    )?;
    Ok(Fixture {
        roots: vec![function],
        function,
    })
}

/// `nested`: the traced function `shifted(v) = scaled(v) + v`, whose body calls the traced function `scaled(x) = x * x`, called
/// by a kernel that assigns `out = shifted(a)`. The document holds both functions through the calls that name them.
pub fn nested(ctx: &mut Ctx) -> Result<Fixture, String> {
    let x = param(ctx, "x", "f64")?;
    let product = apply(ctx, "mul", &[x, x])?;
    let scaled = declare(
        ctx,
        &Declaration {
            name: "scaled",
            params: &[x],
            returns: "f64",
            body: Some(product),
            ..Default::default()
        },
    )?;
    let v = param(ctx, "v", "f64")?;
    let inner = call(ctx, scaled, &[v])?;
    let body = apply(ctx, "add", &[inner, v])?;
    let shifted = declare(
        ctx,
        &Declaration {
            name: "shifted",
            params: &[v],
            returns: "f64",
            body: Some(body),
            ..Default::default()
        },
    )?;
    let a = plane_in(ctx, "a", "f64")?;
    let result = call(ctx, shifted, &[a])?;
    let out = commit(ctx, "out", result)?;
    Ok(Fixture {
        roots: vec![out],
        function: shifted,
    })
}
