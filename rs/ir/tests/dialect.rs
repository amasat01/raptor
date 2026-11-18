//! Who declares a kind: one declaring wing per kind, and a third party's names carry its own prefix.
use raptor_ir::record::Ctx;

/// A manifest of the wing `wing` declaring the kind `kind` (a one-operand expression that takes its operand's type).
fn manifest(wing: &str, kind: &str) -> String {
    format!(
        r#"{{"name":"{wing}","kinds":[{{"name":"{kind}","family":"expr","operands":{{"min":1,"max":1}},"result":["same_as",0]}}]}}"#
    )
}

/// The wings whose kinds may be unqualified.
const BOTH_FIRST_PARTY: &str = r#"("w1", "w2")"#;

#[test]
fn a_kind_declared_by_two_different_wings_is_refused_naming_both_even_when_the_bodies_are_equal() {
    let mut c = Ctx::new().unwrap();
    c.declare_dialect(&manifest("w1", "zz_kind"), BOTH_FIRST_PARTY)
        .unwrap();
    let why = c
        .declare_dialect(&manifest("w2", "zz_kind"), BOTH_FIRST_PARTY)
        .unwrap_err();
    assert_eq!(
        format!("{why:?}"),
        "Refused(\"kind 'zz_kind': declared by the wings 'w1' and 'w2'; a kind has one declaring wing: rename it in one of them or drop one declaration\")"
    );
}

#[test]
fn the_same_wing_declaring_a_kind_again_with_an_equal_declaration_is_accepted() {
    let mut c = Ctx::new().unwrap();
    for _ in 0..2 {
        c.declare_dialect(&manifest("w1", "zz_kind"), BOTH_FIRST_PARTY)
            .unwrap();
    }
}

#[test]
fn the_same_wing_declaring_a_kind_again_with_a_different_declaration_is_refused() {
    let mut c = Ctx::new().unwrap();
    c.declare_dialect(&manifest("w1", "zz_kind"), BOTH_FIRST_PARTY)
        .unwrap();
    let changed = manifest("w1", "zz_kind").replace(r#""max":1"#, r#""max":2"#);
    let why = c.declare_dialect(&changed, BOTH_FIRST_PARTY).unwrap_err();
    assert!(
        format!("{why:?}").contains("already declared with a different declaration"),
        "{why:?}"
    );
}

#[test]
fn a_third_party_kind_not_prefixed_with_its_declarer_is_refused() {
    let mut c = Ctx::new().unwrap();
    let why = c
        .declare_dialect(&manifest("w3", "zz_kind"), "()")
        .unwrap_err();
    assert!(
        format!("{why:?}").contains("a third-party kind is spelled 'w3.<name>'"),
        "{why:?}"
    );
}
