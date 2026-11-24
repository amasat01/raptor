//! Functions and calls: the three reference documents as stored forms, the typing of a declaration and of a call by name, and the
//! digest that names a function. Oracle of the digest: the property itself (equal declarations give equal digests, each single
//! change gives another one); oracle of the pseudo-code: written by hand from the form of the document.
mod common;
use common::*;
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;
use raptor_support::functions::{self as f, Declaration, Fixture};
use raptor_support::groups::{self as g, Membership};
use raptor_support::pseudo::pseudo;
use raptor_support::safe::*;
use std::ffi::c_char;

/// A constructor of one reference document.
type Make = fn(&mut Ctx) -> Result<Fixture, String>;

/// The reference documents with their stored forms.
const FIXTURES: [(&str, Make, &str); 3] = [
    ("twice", f::twice, include_str!("data/functions/twice.dump")),
    (
        "native_with_rules",
        f::native_with_rules,
        include_str!("data/functions/native_with_rules.dump"),
    ),
    (
        "nested",
        f::nested,
        include_str!("data/functions/nested.dump"),
    ),
];

/// A fresh context.
fn ctx() -> Ctx {
    Ctx::new().unwrap()
}

/// The refusal text of a build that must refuse.
fn refused<T>(built: Result<T, String>) -> String {
    match built {
        Ok(_) => panic!("the build was expected to refuse"),
        Err(message) => message,
    }
}

/// `weighted(x, y) = x * y` over `dtype`, a traced lane function.
fn product(c: &mut Ctx, name: &str, dtype: &str) -> Result<(u32, u32, u32), String> {
    let (x, y) = (f::param(c, "x", dtype)?, f::param(c, "y", dtype)?);
    let body = g::apply(c, "mul", &[x, y])?;
    let function = f::declare(
        c,
        &Declaration {
            name,
            params: &[x, y],
            returns: dtype,
            body: Some(body),
            ..Default::default()
        },
    )?;
    Ok((function, x, y))
}

#[test]
fn the_three_fixtures_equal_their_stored_documents() {
    for (name, make, stored) in FIXTURES {
        let c = rx_ctx_new();
        let fixture = make(unsafe { &mut *c }).unwrap();
        assert_eq!(dump(c, &fixture.roots), stored, "{name}");
        rx_ctx_free(c);
    }
}

#[test]
fn a_stored_document_loads_dumps_again_unchanged_and_keeps_the_digest_of_its_function() {
    for (name, make, stored) in FIXTURES {
        let mut original = ctx();
        let fixture = make(&mut original).unwrap();
        let digest = original.function_digest(fixture.function).unwrap();
        let c = rx_ctx_new();
        let (status, roots) = load(c, stored);
        assert_eq!(status, RX_OK, "{name}: {}", err(c));
        assert_eq!(roots.len(), 1, "{name}");
        assert_eq!(dump(c, &roots), stored, "{name}");
        let loaded = unsafe { &*c };
        let function = original.str_field(fixture.function, "name").unwrap();
        assert!(
            loaded.function_named(&function, &digest).is_some(),
            "{name}"
        );
        rx_ctx_free(c);
    }
}

#[test]
fn a_document_whose_call_names_a_function_the_document_lacks_is_refused_by_name_when_loaded() {
    let (_, _, stored) = FIXTURES[0];
    let renamed = stored.replace("('function', ('weighted',", "('function', ('ghost',");
    let c = rx_ctx_new();
    let (status, _) = load(c, &renamed);
    assert_eq!(status, RX_REFUSED);
    assert!(
        err(c).contains("call of unknown function 'ghost'"),
        "{}",
        err(c)
    );
    rx_ctx_free(c);
}

#[test]
fn the_door_states_the_digest_of_a_function() {
    let mut c = ctx();
    let fixture = f::twice(&mut c).unwrap();
    let want = c.function_digest(fixture.function).unwrap();
    let p = &mut c as *mut Ctx;
    let got = text(p, |p, b: *mut c_char, cap, l| {
        rx_field(p, fixture.function, c"digest".as_ptr(), b, cap, l)
    });
    assert_eq!(got, format!("'{want}'"));
}

// the typing of a call

#[test]
fn a_traced_function_is_called_twice_with_the_type_of_its_result() {
    let mut c = ctx();
    let fixture = f::twice(&mut c).unwrap();
    let calls: Vec<u32> = (0..c.nodes.len() as u32)
        .filter(|&h| c.table.is_a(c.nodes[h as usize].class, "Call"))
        .collect();
    assert_eq!(calls.len(), 2);
    for call in calls {
        let node = c.node(call).unwrap();
        assert_eq!((node.dtype.as_str(), node.shape.len()), ("f64", 0));
    }
    assert_eq!(c.str_field(fixture.function, "scope").unwrap(), "lane");
}

#[test]
fn a_call_with_another_number_of_arguments_is_refused_by_name() {
    let mut c = ctx();
    let (weighted, x, _) = product(&mut c, "weighted", "f64").unwrap();
    let m = refused(f::call(&mut c, weighted, &[x]));
    assert!(
        m.contains("call of function 'weighted' passes 1 argument(s), the function takes 2"),
        "{m}"
    );
    let m = refused(f::call(&mut c, weighted, &[x, x, x]));
    assert!(m.contains("passes 3 argument(s)"), "{m}");
}

#[test]
fn a_call_with_an_argument_of_another_type_is_refused_by_name() {
    let mut c = ctx();
    let (weighted, x, _) = product(&mut c, "weighted", "f64").unwrap();
    let count = g::whole(&mut c, "i32", 3).unwrap();
    let m = refused(f::call(&mut c, weighted, &[x, count]));
    assert!(
        m.contains("call of function 'weighted': argument 1 ('y') has the type"),
        "{m}"
    );
    assert!(m.contains("i32"), "{m}");
}

#[test]
fn a_call_of_a_function_the_table_does_not_hold_is_refused_by_name() {
    let mut c = ctx();
    let (weighted, x, y) = product(&mut c, "weighted", "f64").unwrap();
    let ghost = Val::Tuple(vec![Val::Str("ghost".into()), Val::Str("0".into())]);
    let m = refused(f::call_named(&mut c, ghost, &[x, y]));
    assert!(m.contains("call of unknown function 'ghost'"), "{m}");
    let stale = Val::Tuple(vec![Val::Str("weighted".into()), Val::Str("0".into())]);
    let m = refused(f::call_named(&mut c, stale, &[x, y]));
    assert!(
        m.contains("call of function 'weighted' names the digest 0"),
        "{m}"
    );
    assert!(f::call(&mut c, weighted, &[x, y]).is_ok());
}

#[test]
fn a_call_that_does_not_name_a_function_as_a_pair_is_refused() {
    let mut c = ctx();
    let x = g::plane_in(&mut c, "x", "f64").unwrap();
    let m = refused(f::call_named(&mut c, Val::Str("weighted".into()), &[x]));
    assert!(m.contains("(name, digest) pair"), "{m}");
}

#[test]
fn a_native_function_is_called_with_the_type_it_declares() {
    let mut c = ctx();
    let fixture = f::native_with_rules(&mut c).unwrap();
    let a = g::plane_in(&mut c, "a", "f64").unwrap();
    let call = f::call(&mut c, fixture.function, &[a, a]).unwrap();
    assert_eq!(c.node(call).unwrap().dtype, "f64");
}

#[test]
fn a_call_answers_the_function_it_names() {
    let mut c = ctx();
    let fixture = f::native_with_rules(&mut c).unwrap();
    let a = g::plane_in(&mut c, "a", "f64").unwrap();
    let call = f::call(&mut c, fixture.function, &[a, a]).unwrap();
    assert!(matches!(c.field(call, "callee"), Ok(Val::Node(h)) if h == fixture.function));
    assert!(c.field(a, "callee").is_err());
}

// the typing of a declaration

#[test]
fn a_function_calling_itself_is_refused_by_name() {
    let mut c = ctx();
    let (first, x, y) = product(&mut c, "loopy", "f64").unwrap();
    let again = f::call(&mut c, first, &[x, y]).unwrap();
    let m = refused(f::declare(
        &mut c,
        &Declaration {
            name: "loopy",
            params: &[x, y],
            returns: "f64",
            body: Some(again),
            ..Default::default()
        },
    ));
    assert!(
        m.contains("function 'loopy' calls itself; recursion is not written"),
        "{m}"
    );
}

#[test]
fn a_cycle_through_another_function_is_refused_by_name() {
    let mut c = ctx();
    let (ping, x, y) = product(&mut c, "ping", "f64").unwrap();
    let via = f::call(&mut c, ping, &[x, y]).unwrap();
    let pong = f::declare(
        &mut c,
        &Declaration {
            name: "pong",
            params: &[x, y],
            returns: "f64",
            body: Some(via),
            ..Default::default()
        },
    )
    .unwrap();
    let back = f::call(&mut c, pong, &[x, y]).unwrap();
    let m = refused(f::declare(
        &mut c,
        &Declaration {
            name: "ping",
            params: &[x, y],
            returns: "f64",
            body: Some(back),
            ..Default::default()
        },
    ));
    assert!(m.contains("function 'ping' calls itself"), "{m}");
}

#[test]
fn a_group_function_called_outside_a_group_is_refused_by_name_and_inside_one_is_taken() {
    let mut c = ctx();
    let x = f::param(&mut c, "x", "f64").unwrap();
    let body = g::apply(&mut c, "add", &[x, x]).unwrap();
    let bump = f::declare(
        &mut c,
        &Declaration {
            name: "bump",
            scope: "group",
            params: &[x],
            returns: "f64",
            body: Some(body),
            ..Default::default()
        },
    )
    .unwrap();
    let plane = g::plane_in(&mut c, "p", "f64").unwrap();
    let call = f::call(&mut c, bump, &[plane]).unwrap();
    let out = g::commit(&mut c, "y", call).unwrap();
    let (p, mut w) = (&mut c as *mut Ctx, 0u32);
    let none: *const c_char = std::ptr::null();
    let status = rx_canonical(p, [out].as_ptr(), 1, none, none, none, &mut w);
    assert_eq!(status, RX_REFUSED);
    assert!(
        err(p).contains("the call of the group function 'bump' is reached outside a Group"),
        "{}",
        err(p)
    );
    assert!(g::group(&mut c, &Membership::count(4), &[], &[out]).is_ok());
}

#[test]
fn a_lane_function_calling_a_group_function_is_refused_by_name() {
    let mut c = ctx();
    let x = f::param(&mut c, "x", "f64").unwrap();
    let body = g::apply(&mut c, "add", &[x, x]).unwrap();
    let bump = f::declare(
        &mut c,
        &Declaration {
            name: "bump",
            scope: "group",
            params: &[x],
            returns: "f64",
            body: Some(body),
            ..Default::default()
        },
    )
    .unwrap();
    let inner = f::call(&mut c, bump, &[x]).unwrap();
    let m = refused(f::declare(
        &mut c,
        &Declaration {
            name: "lane_user",
            params: &[x],
            returns: "f64",
            body: Some(inner),
            ..Default::default()
        },
    ));
    assert!(
        m.contains("function 'lane_user': a traced body is pure"),
        "{m}"
    );
    assert!(m.contains("a call of the group function 'bump'"), "{m}");
}

/// The refusal of a traced lane function `impure(x)` whose body is the node `body`.
fn impure(c: &mut Ctx, x: u32, body: u32, returns: &str) -> String {
    refused(f::declare(
        c,
        &Declaration {
            name: "impure",
            params: &[x],
            returns,
            body: Some(body),
            ..Default::default()
        },
    ))
}

#[test]
fn a_traced_body_that_reads_a_plane_holds_an_update_a_group_or_a_sample_reduction_is_refused_by_name(
) {
    let mut c = ctx();
    let x = f::param(&mut c, "x", "f64").unwrap();
    let plane = g::plane_in(&mut c, "p", "f64").unwrap();
    let read = g::apply(&mut c, "add", &[x, plane]).unwrap();
    let m = impure(&mut c, x, read, "f64");
    assert!(
        m.contains(
            "function 'impure': a traced body is pure and its body holds a read of the plane 'p'"
        ),
        "{m}"
    );
    let update = g::commit_wide(&mut c, "w", x).unwrap();
    let m = impure(&mut c, x, update, "f64");
    assert!(
        m.contains(
            "function 'impure': a traced body is pure and its body holds an Update, which writes"
        ),
        "{m}"
    );
    let place = g::rank(&mut c).unwrap();
    let m = impure(&mut c, x, place, "i32");
    assert!(m.contains("function 'impure': a traced body is pure and its body holds a GroupRank, a group element"), "{m}");
    let total = g::reduce(&mut c, "sum", "tree", x).unwrap();
    let m = impure(&mut c, x, total, "f64");
    assert!(m.contains("a GroupReduce, a group element"), "{m}");
    let index = c.table.find("ReduceSamples").unwrap();
    let given = vec![
        ("op".to_string(), Val::Str("sum".into())),
        ("value".to_string(), Val::Node(plane)),
    ];
    let samples = c.new_from(index, &given).unwrap();
    let m = impure(&mut c, x, samples, "f64");
    assert!(m.contains("a ReduceSamples, a sample reduction"), "{m}");
}

/// A loop placeholder of the class `kind` over `f64`, with the fields `fields`.
fn placeholder(c: &mut Ctx, kind: &str, fields: &str) -> u32 {
    c.make(kind, "f64", &[], None, &[], fields).unwrap()
}

#[test]
fn a_traced_body_that_holds_the_index_or_a_carry_of_a_loop_it_is_not_in_is_refused_by_name() {
    let mut c = ctx();
    let x = f::param(&mut c, "x", "f64").unwrap();
    let index = placeholder(
        &mut c,
        "LoopIndex",
        "(('depth', 0), ('name', 'i'), ('bounds', None))",
    );
    let carry = placeholder(
        &mut c,
        "LoopCarry",
        "(('depth', 0), ('slot', 0), ('name', 'c'))",
    );
    for held in [index, carry] {
        let body = g::apply(&mut c, "add", &[x, held]).unwrap();
        let m = impure(&mut c, x, body, "f64");
        assert!(
            m.contains(
                "function 'impure': a traced body is pure and its body holds the index or a carry of a loop it is not in"
            ),
            "{m}"
        );
    }
}

#[test]
fn a_traced_body_with_another_type_than_the_result_or_a_foreign_parameter_is_refused_by_name() {
    let mut c = ctx();
    let x = f::param(&mut c, "x", "f64").unwrap();
    let other = f::param(&mut c, "z", "f64").unwrap();
    let body = g::apply(&mut c, "add", &[x, x]).unwrap();
    let m = impure(&mut c, x, body, "f32");
    assert!(
        m.contains("function 'impure': the body has the type"),
        "{m}"
    );
    let foreign = g::apply(&mut c, "add", &[x, other]).unwrap();
    let m = impure(&mut c, x, foreign, "f64");
    assert!(
        m.contains("function 'impure': the body reaches the parameter 'z'"),
        "{m}"
    );
}

#[test]
fn a_native_function_with_an_empty_text_map_is_refused_by_name() {
    let mut c = ctx();
    let x = f::param(&mut c, "x", "f64").unwrap();
    let m = refused(f::declare(
        &mut c,
        &Declaration {
            name: "hollow",
            params: &[x],
            returns: "f64",
            ..Default::default()
        },
    ));
    assert!(m.contains("native function 'hollow' holds no text"), "{m}");
}

#[test]
fn a_declaration_that_breaks_the_form_of_a_function_is_refused_by_name() {
    let mut c = ctx();
    let x = f::param(&mut c, "x", "f64").unwrap();
    let body = g::apply(&mut c, "add", &[x, x]).unwrap();
    let base = Declaration {
        name: "odd",
        params: &[x],
        returns: "f64",
        body: Some(body),
        ..Default::default()
    };
    let both = Declaration {
        natives: &[("t1", "text")],
        ..base.clone()
    };
    assert!(refused(f::declare(&mut c, &both))
        .contains("function 'odd': the body is traced or native, not both"));
    let twice = Declaration {
        body: None,
        natives: &[("t1", "a"), ("t1", "b")],
        ..base.clone()
    };
    assert!(refused(f::declare(&mut c, &twice))
        .contains("function 'odd': the target 't1' has two texts"));
    let scoped = Declaration {
        scope: "block",
        ..base.clone()
    };
    assert!(refused(f::declare(&mut c, &scoped))
        .contains("Function 'odd': scope 'block' is not one of"));
    let nameless = Declaration {
        name: "",
        ..base.clone()
    };
    assert!(refused(f::declare(&mut c, &nameless)).contains("a function has a name"));
    let doubled = Declaration {
        params: &[x, x],
        ..base.clone()
    };
    assert!(refused(f::declare(&mut c, &doubled))
        .contains("function 'odd': parameter 'x' is declared twice"));
    assert!(refused(f::declare(
        &mut c,
        &Declaration {
            params: &[body],
            ..base.clone()
        }
    ))
    .contains("function 'odd': a parameter is a Param, got a Op"));
}

#[test]
fn a_name_holds_one_function_and_an_equal_declaration_may_be_entered_again() {
    let mut c = ctx();
    let (first, x, y) = product(&mut c, "weighted", "f64").unwrap();
    let (again, ..) = product(&mut c, "weighted", "f64").unwrap();
    assert_eq!(
        c.function_digest(first).unwrap(),
        c.function_digest(again).unwrap()
    );
    let other = g::apply(&mut c, "add", &[x, y]).unwrap();
    let m = refused(f::declare(
        &mut c,
        &Declaration {
            name: "weighted",
            params: &[x, y],
            returns: "f64",
            body: Some(other),
            ..Default::default()
        },
    ));
    assert!(
        m.contains("function 'weighted' is already declared with another body"),
        "{m}"
    );
}

#[test]
fn a_rule_of_the_wrong_scope_signature_or_class_is_refused_by_name() {
    let mut c = ctx();
    let (x, dx) = (
        f::param(&mut c, "x", "f64").unwrap(),
        f::param(&mut c, "dx", "f64").unwrap(),
    );
    let tangent = g::apply(&mut c, "mul", &[x, dx]).unwrap();
    let rule = |c: &mut Ctx, name: &str, params: &[u32], scope: &str, body: u32| {
        f::declare(
            c,
            &Declaration {
                name,
                scope,
                params,
                returns: "f64",
                body: Some(body),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let doubled = g::apply(&mut c, "add", &[x, x]).unwrap();
    let good = rule(&mut c, "good", &[x, dx], "", tangent);
    let short = rule(&mut c, "short", &[x], "", doubled);
    let grouped = rule(&mut c, "grouped", &[x, dx], "group", tangent);
    let native = |c: &mut Ctx, name: &str, jvp: u32, vjp: &[u32]| {
        f::declare(
            c,
            &Declaration {
                name,
                params: &[x],
                returns: "f64",
                natives: &[("t1", "text")],
                jvp: Some(jvp),
                vjp,
                ..Default::default()
            },
        )
    };
    assert!(native(&mut c, "a", good, &[]).is_ok());
    let m = refused(native(&mut c, "b", short, &[]));
    assert!(
        m.contains("function 'b': the jvp rule takes 1 parameter(s), it needs 2"),
        "{m}"
    );
    let m = refused(native(&mut c, "c", grouped, &[]));
    assert!(
        m.contains("the jvp rule is a group function; a rule has the scope of its function"),
        "{m}"
    );
    let m = refused(native(&mut c, "d", x, &[]));
    assert!(m.contains("the jvp rule is a Function, got a Param"), "{m}");
    assert!(
        native(&mut c, "e", good, &[good]).is_ok(),
        "a rule over one read returns the type of the read"
    );
    let m = refused(native(&mut c, "f", good, &[short]));
    assert!(
        m.contains("the vjp 0 rule takes 1 parameter(s), it needs 2"),
        "{m}"
    );
}

// the digest

/// A native function `name(x: dtype) -> returns` with the text of `t1` and an optional forward rule over `rule`.
fn native_digest(
    name: &str,
    dtype: &str,
    returns: &str,
    text: &[(&str, &str)],
    rule: Option<&str>,
) -> String {
    let mut c = ctx();
    let x = f::param(&mut c, "x", dtype).unwrap();
    let jvp = rule.map(|op| {
        let dx = f::param(&mut c, "dx", dtype).unwrap();
        let body = g::apply(&mut c, op, &[x, dx]).unwrap();
        f::declare(
            &mut c,
            &Declaration {
                name: "rule",
                params: &[x, dx],
                returns: dtype,
                body: Some(body),
                ..Default::default()
            },
        )
        .unwrap()
    });
    let function = f::declare(
        &mut c,
        &Declaration {
            name,
            params: &[x],
            returns,
            natives: text,
            jvp,
            ..Default::default()
        },
    )
    .unwrap();
    c.function_digest(function).unwrap()
}

/// A traced function `name(x: dtype, y: dtype)` whose body is `op` of its parameters.
fn traced_digest(name: &str, dtype: &str, op: &str) -> String {
    let mut c = ctx();
    let (x, y) = (
        f::param(&mut c, "x", dtype).unwrap(),
        f::param(&mut c, "y", dtype).unwrap(),
    );
    let body = g::apply(&mut c, op, &[x, y]).unwrap();
    let function = f::declare(
        &mut c,
        &Declaration {
            name,
            params: &[x, y],
            returns: dtype,
            body: Some(body),
            ..Default::default()
        },
    )
    .unwrap();
    c.function_digest(function).unwrap()
}

/// Asserts that every digest of `changed` differs from `base` and from each other.
fn all_differ(base: &str, changed: &[(&str, String)]) {
    for (what, digest) in changed {
        assert_ne!(digest, base, "{what}");
    }
    for (i, (a, x)) in changed.iter().enumerate() {
        for (b, y) in &changed[i + 1..] {
            assert_ne!(x, y, "{a} against {b}");
        }
    }
}

#[test]
fn the_digest_of_a_traced_function_is_stable_and_changes_with_each_part() {
    let base = traced_digest("weighted", "f64", "mul");
    assert_eq!(base, traced_digest("weighted", "f64", "mul"));
    all_differ(
        &base,
        &[
            ("name", traced_digest("scaled", "f64", "mul")),
            (
                "parameter and result type",
                traced_digest("weighted", "f32", "mul"),
            ),
            ("body", traced_digest("weighted", "f64", "add")),
        ],
    );
}

#[test]
fn the_digest_of_a_native_function_is_stable_and_changes_with_each_part() {
    let texts = [("t1", "return x;"), ("t2", "return x")];
    let base = native_digest("id", "f64", "f64", &texts, None);
    assert_eq!(base, native_digest("id", "f64", "f64", &texts, None));
    let reordered = [("t2", "return x"), ("t1", "return x;")];
    assert_eq!(
        base,
        native_digest("id", "f64", "f64", &reordered, None),
        "the texts are a map"
    );
    all_differ(
        &base,
        &[
            ("name", native_digest("same", "f64", "f64", &texts, None)),
            (
                "parameter type",
                native_digest("id", "f32", "f64", &texts, None),
            ),
            (
                "result type",
                native_digest("id", "f64", "f32", &texts, None),
            ),
            (
                "one text",
                native_digest(
                    "id",
                    "f64",
                    "f64",
                    &[("t1", "return x;"), ("t2", "return -x")],
                    None,
                ),
            ),
            (
                "a target name",
                native_digest(
                    "id",
                    "f64",
                    "f64",
                    &[("t1", "return x;"), ("t3", "return x")],
                    None,
                ),
            ),
            (
                "a rule",
                native_digest("id", "f64", "f64", &texts, Some("mul")),
            ),
            (
                "another rule",
                native_digest("id", "f64", "f64", &texts, Some("add")),
            ),
        ],
    );
}

#[test]
fn the_digest_of_a_caller_changes_with_the_function_it_calls() {
    let digest = |op: &str| {
        let mut c = ctx();
        let (inner, ..) = {
            let (x, y) = (
                f::param(&mut c, "x", "f64").unwrap(),
                f::param(&mut c, "y", "f64").unwrap(),
            );
            let body = g::apply(&mut c, op, &[x, y]).unwrap();
            let inner = f::declare(
                &mut c,
                &Declaration {
                    name: "inner",
                    params: &[x, y],
                    returns: "f64",
                    body: Some(body),
                    ..Default::default()
                },
            )
            .unwrap();
            (inner, x, y)
        };
        let (a, b) = (
            f::param(&mut c, "a", "f64").unwrap(),
            f::param(&mut c, "b", "f64").unwrap(),
        );
        let call = f::call(&mut c, inner, &[a, b]).unwrap();
        let outer = f::declare(
            &mut c,
            &Declaration {
                name: "outer",
                params: &[a, b],
                returns: "f64",
                body: Some(call),
                ..Default::default()
            },
        )
        .unwrap();
        c.function_digest(outer).unwrap()
    };
    assert_eq!(digest("mul"), digest("mul"));
    assert_ne!(digest("mul"), digest("add"));
}

// the pseudo-code

#[test]
fn the_pseudo_code_of_a_function_and_of_a_call_is_written_from_the_document() {
    let mut c = ctx();
    let fixture = f::twice(&mut c).unwrap();
    assert_eq!(
        pseudo(&c, fixture.function).unwrap(),
        "function weighted(x: f64, y: f64) -> f64\n  return ((x * x) + y)\n"
    );
    assert_eq!(
        pseudo(&c, fixture.roots[0]).unwrap(),
        "out = (weighted(a, b) + weighted(b, a))\n"
    );
    let mut c = ctx();
    let fixture = f::native_with_rules(&mut c).unwrap();
    assert_eq!(
        pseudo(&c, fixture.function).unwrap(),
        "function product(x: f64, y: f64) -> f64\n  native t1: 'return x * y;'\n  native t2: 'return x * y'\n  jvp product_jvp\n  vjp product_vjp_x\n  vjp product_vjp_y\n"
    );
    let mut c = ctx();
    let fixture = f::nested(&mut c).unwrap();
    assert_eq!(
        pseudo(&c, fixture.function).unwrap(),
        "function shifted(v: f64) -> f64\n  return (scaled(v) + v)\n"
    );
    assert_eq!(pseudo(&c, fixture.roots[0]).unwrap(), "out = shifted(a)\n");
}

// the walk of a kernel that calls

/// The digest of the walk of the kernel `out = f(a, b)` where `f(x, y) = x <op> y`.
fn walk_digest(op: &str) -> String {
    let mut c = ctx();
    let (x, y) = (
        f::param(&mut c, "x", "f64").unwrap(),
        f::param(&mut c, "y", "f64").unwrap(),
    );
    let body = g::apply(&mut c, op, &[x, y]).unwrap();
    let function = f::declare(
        &mut c,
        &Declaration {
            name: "f",
            params: &[x, y],
            returns: "f64",
            body: Some(body),
            ..Default::default()
        },
    )
    .unwrap();
    let (a, b) = (
        g::plane_in(&mut c, "a", "f64").unwrap(),
        g::plane_in(&mut c, "b", "f64").unwrap(),
    );
    let call = f::call(&mut c, function, &[a, b]).unwrap();
    let out = g::commit(&mut c, "out", call).unwrap();
    let (p, mut w) = (&mut c as *mut Ctx, 0u32);
    let none: *const c_char = std::ptr::null();
    assert_eq!(
        rx_canonical(p, [out].as_ptr(), 1, none, none, none, &mut w),
        RX_OK,
        "{}",
        err(p)
    );
    let mut d = [0 as c_char; 65];
    assert_eq!(rx_digest(p, w, d.as_mut_ptr()), RX_OK);
    unsafe { std::ffi::CStr::from_ptr(d.as_ptr()) }
        .to_str()
        .unwrap()
        .to_string()
}

#[test]
fn the_digest_of_a_walk_through_a_call_follows_the_function_it_names() {
    assert_eq!(walk_digest("mul"), walk_digest("mul"));
    assert_ne!(walk_digest("mul"), walk_digest("add"));
}

// the reverse rules, one per read

/// A native function `pair(x: f64, y: f32) -> f64` whose reverse rules are the functions built over `bodies`, one rule per body,
/// each over the parameters (x, y, bar). The rule `i` is `rule_i`.
fn pair_with_rules(c: &mut Ctx, bodies: &[(&str, [usize; 2])]) -> Result<u32, String> {
    let (x, y) = (f::param(c, "x", "f64")?, f::param(c, "y", "f32")?);
    let bar = f::param(c, "bar", "f64")?;
    let plane = [x, y, bar];
    let mut rules = vec![];
    for (i, (op, pick)) in bodies.iter().enumerate() {
        let body = g::apply(c, op, &[plane[pick[0]], plane[pick[1]]])?;
        let name = format!("rule_{i}");
        let returns = c.nodes[body as usize].dtype.clone();
        rules.push(f::declare(
            c,
            &Declaration {
                name: &name,
                params: &[x, y, bar],
                returns: &returns,
                body: Some(body),
                ..Default::default()
            },
        )?);
    }
    f::declare(
        c,
        &Declaration {
            name: "pair",
            params: &[x, y],
            returns: "f64",
            natives: &[("t1", "text")],
            vjp: &rules,
            ..Default::default()
        },
    )
}

#[test]
fn a_function_with_reverse_rules_states_one_rule_per_read_by_name() {
    let mut c = ctx();
    let m = refused(pair_with_rules(&mut c, &[("mul", [0, 2])]));
    assert!(
        m.contains(
            "function 'pair': the vjp rules are one per read: 1 given, the function has 2 read(s)"
        ),
        "{m}"
    );
    let m = refused(pair_with_rules(&mut c, &[("mul", [0, 2]); 3]));
    assert!(m.contains("3 given, the function has 2 read(s)"), "{m}");
}

#[test]
fn a_reverse_rule_whose_result_differs_from_its_read_is_refused_by_name() {
    let mut c = ctx();
    // rule 1 gives the gradient of `y`, an f32 read, and its body is an f64 product
    let m = refused(pair_with_rules(&mut c, &[("mul", [0, 2]), ("mul", [0, 2])]));
    assert!(
        m.contains("function 'pair': the vjp 1 rule returns") && m.contains("dtype='f32'"),
        "{m}"
    );
}

/// The digest of the function `pair(x: f64, y: f64)` with two reverse rules, the second being `op` of (y, bar).
fn digest_with_second_rule(op: &str) -> String {
    let mut c = ctx();
    let [x, y, bar] = ["x", "y", "bar"].map(|n| f::param(&mut c, n, "f64").unwrap());
    let rules: Vec<u32> = [("mul", x), ("rule_1", y)]
        .iter()
        .enumerate()
        .map(|(i, (_, read))| {
            let body = g::apply(&mut c, if i == 0 { "mul" } else { op }, &[*read, bar]).unwrap();
            f::declare(
                &mut c,
                &Declaration {
                    name: &format!("rule_{i}"),
                    params: &[x, y, bar],
                    returns: "f64",
                    body: Some(body),
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();
    let function = f::declare(
        &mut c,
        &Declaration {
            name: "pair",
            params: &[x, y],
            returns: "f64",
            natives: &[("t1", "text")],
            vjp: &rules,
            ..Default::default()
        },
    )
    .unwrap();
    c.function_digest(function).unwrap()
}

#[test]
fn the_digest_changes_when_the_second_reverse_rule_changes() {
    assert_eq!(
        digest_with_second_rule("mul"),
        digest_with_second_rule("mul")
    );
    assert_ne!(
        digest_with_second_rule("mul"),
        digest_with_second_rule("add")
    );
}
