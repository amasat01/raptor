//! Pseudo-code through the door's C functions.
mod common;
use common::*;
use raptor_ir::record::Ctx;
use raptor_support::safe::*;

/// The pseudo-code of a node in a context.
fn pseudo(c: *mut Ctx, h: u32) -> String {
    raptor_support::pseudo::pseudo(unsafe { &*c }, h).unwrap()
}

#[test]
fn probe_expression() {
    let c = rx_ctx_new();
    let h = expression(c);
    assert_eq!(pseudo(c, h[5]), "y = ((a + b) + 2.0)\n");
    assert_eq!(pseudo(c, h[5]), pseudo(c, h[5]));
    assert_eq!(pseudo(c, h[4]), "((a + b) + 2.0)\n");
    rx_ctx_free(c);
}

#[test]
fn shared_node_is_named_and_sink_kind_shows() {
    let c = rx_ctx_new();
    let x = leaf(c, "x");
    let s = mk(c, "Op", None, &[x], "(('kind', 'neg'), ('literal', None))");
    let e = mk(
        c,
        "Op",
        None,
        &[s, s],
        "(('kind', 'mul'), ('literal', None))",
    );
    let w = mk(c, "Update", None, &[e], "(('name', 'w'),)");
    assert_eq!(pseudo(c, w), "t0 = neg(x)\nupdate w = (t0 * t0)\n");
    rx_ctx_free(c);
}

#[test]
fn a_loop_renders() {
    let c = rx_ctx_new();
    let (lp, _) = a_loop(c);
    let t = pseudo(c, lp);
    assert_eq!(
        t,
        "loop i in 0..2 step 1 carry [c = 0]\n  c <- p\n  o = c\n"
    );
    assert_eq!(t, pseudo(c, lp));
    rx_ctx_free(c);
}
