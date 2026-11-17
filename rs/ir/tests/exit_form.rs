//! Where a loop's break sits, read from the loop's `exit_form` property. Oracle: the previous Python property on the same four
//! loops (no break; the break yields the carries, the next values, or a third value).
use raptor_ir::build::LoopParts;
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;

/// A loop `c = c + x` over three trips, breaking on `c > x` with the values `values` names (`None`: no break).
fn exit_form_of(values: Option<&str>) -> Val {
    let mut c = Ctx::new().unwrap();
    let fields = "(('kind', 'uniform'), ('role', 'uniform'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
    let x = c.make("Leaf", "f64", &[], None, &[], fields).unwrap();
    let index = c
        .make(
            "LoopIndex",
            "int32",
            &[],
            None,
            &[],
            "(('depth', 0), ('name', '_'), ('bounds', (0, 3)))",
        )
        .unwrap();
    let carry = c
        .make(
            "LoopCarry",
            "f64",
            &[],
            None,
            &[],
            "(('depth', 0), ('slot', 0), ('name', 'c'))",
        )
        .unwrap();
    let next = c.make_op("add", &[carry, x], &Val::None).unwrap();
    let cond = c.make_op("gt", &[carry, x], &Val::None).unwrap();
    let held = match values {
        Some("carries") => vec![carry],
        Some("nexts") => vec![next],
        Some(_) => vec![x],
        None => vec![],
    };
    let parts = LoopParts {
        index,
        carries: &[carry],
        inits: &[x],
        nexts: &[next],
        range: (0, 3, 1),
        body_sinks: &[],
        exit_cond: values.map(|_| cond),
        exit_values: &held,
        count: None,
        count_tail: false,
        dynamic: None,
    };
    let lp = c.build_loop(&parts).unwrap();
    c.field(lp, "exit_form").unwrap()
}

/// The four forms a loop's break can have, as the previous property answered them.
#[test]
fn the_exit_form_of_a_loop_is_head_middle_tail_or_none() {
    let text = |s: &str| Val::Str(s.into());
    assert_eq!(exit_form_of(None), Val::None);
    assert_eq!(exit_form_of(Some("carries")), text("head"));
    assert_eq!(exit_form_of(Some("nexts")), text("tail"));
    assert_eq!(exit_form_of(Some("other")), text("middle"));
}
