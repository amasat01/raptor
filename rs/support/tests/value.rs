use raptor_ir::value::{parse, text_val, Val};

const GOOD: &[&str] = &[
    "None",
    "True",
    "False",
    "0",
    "7",
    "-7",
    "170141183460469231731687303715884105727",
    "-170141183460469231731687303715884105728",
    "0.0",
    "-0.0",
    "1.0",
    "-2.5",
    "0.1",
    "1e+16",
    "1e-05",
    "5e-324",
    "1.7976931348623157e+308",
    "123456789012345.0",
    "inf",
    "-inf",
    "nan",
    "''",
    "'abc'",
    "\"it's\"",
    "'say \"hi\"'",
    "'\\\\'",
    "'a\\nb\\rc\\td'",
    "'\\x00\\x1f'",
    "'\\u200b'",
    "'\\U000e0001'",
    "'caf\u{e9}'",
    "'it\\'s \"q\"'",
    "()",
    "(1,)",
    "(1, 2)",
    "(None, True, 'x')",
    "((1, 2), (3,))",
    "((),)",
    "[]",
    "[1]",
    "[1, 2.5, 'x']",
    "[[], ()]",
    "([1], [2, 3])",
    "Wire()",
    "TensorType(shape=(), dtype='f64', tag=None)",
    "TensorType(shape=(2, 3), dtype='i32', tag='lu_factor')",
    "Slot(role='plane_in', name='a', ttype=TensorType(shape=(), dtype='f64', tag=None))",
    "Quantity(slug='q', name='x', wires=('a', 'b'), pack='auto')",
    "#0",
    "#12",
    "(#1, #2)",
    "(#3,)",
    "[#1, None]",
    "Rec(a=#4, b=(1,), c=[#5])",
    "(('k', #1), ('m', 2.0))",
];

#[test]
fn round_trip() {
    assert!(GOOD.len() >= 40);
    for t in GOOD {
        let v = parse(t).unwrap_or_else(|e| panic!("{t}: {e}"));
        assert_eq!(text_val(&v), *t, "write(parse(text))");
    }
}

#[test]
fn shapes() {
    assert_eq!(parse("(1,)").unwrap(), Val::Tuple(vec![Val::Int(1)]));
    assert_eq!(parse("-0.0").unwrap(), Val::Float(-0.0));
    assert!(matches!(parse("-0.0").unwrap(), Val::Float(f) if f.is_sign_negative()));
    assert!(matches!(parse("nan").unwrap(), Val::Float(f) if f.is_nan()));
    assert_eq!(
        parse("  ( 1 , 2 )  ").unwrap(),
        Val::Tuple(vec![Val::Int(1), Val::Int(2)])
    );
    assert_eq!(parse("#7").unwrap(), Val::Node(7));
    assert_eq!(
        parse("R(a=1)").unwrap(),
        Val::Rec("R".into(), vec![("a".into(), Val::Int(1))])
    );
}

#[test]
fn refusals() {
    for t in [
        "",
        "(",
        ")",
        "(1",
        "(1)",
        "[1,",
        "'abc",
        "\"abc'",
        "'\\q'",
        "'\\x4'",
        "'\\ud800'",
        "1.2.3",
        "1e",
        "--1",
        "-",
        "+1",
        "Foo",
        "foo(",
        "R(a)",
        "R(a=1,)",
        "R(=1)",
        "#",
        "#x",
        "#-1",
        "1 2",
        "None None",
        "340282366920938463463374607431768211456",
        "(,)",
        "{}",
        "@",
        "'a' 'b'",
        "inf1",
    ] {
        assert!(parse(t).is_err(), "{t:?} should be refused");
    }
    assert!(parse(&"(".repeat(500)).is_err());
}
