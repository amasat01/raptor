use raptor_support::safe::*;
use std::ffi::{CStr, CString};

fn make(c: *mut raptor_ir::record::Ctx, cls: &str, ops: &[u32], fields: &str) -> u32 {
    let (cls, dt) = (CString::new(cls).unwrap(), CString::new("f64").unwrap());
    let mut o = 0;
    let st = rx_make(
        c,
        cls.as_ptr(),
        dt.as_ptr(),
        std::ptr::null(),
        0,
        std::ptr::null(),
        ops.as_ptr(),
        ops.len(),
        fields.as_ptr() as *const _,
        fields.len(),
        &mut o,
    );
    assert_eq!(st, RX_OK);
    o
}

#[test]
fn probe_digest() {
    let c = rx_ctx_new();
    let leaf = |n: &str| {
        format!("(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '{n}'), ('quantity', None), ('wire_index', 0), ('sorted', False))")
    };
    let (a, b) = (
        make(c, "Leaf", &[], &leaf("a")),
        make(c, "Leaf", &[], &leaf("b")),
    );
    let op = "(('kind', 'add'), ('literal', None))";
    let s = make(c, "Op", &[a, b], op);
    let k = make(c, "Const", &[], "(('literal', 2.0),)");
    let e = make(c, "Op", &[s, k], op);
    let y = make(c, "Assign", &[e], "(('name', 'y'),)");
    let mut w = 0;
    assert_eq!(
        rx_canonical(
            c,
            &y,
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &mut w
        ),
        RX_OK
    );
    let mut d = [0 as std::ffi::c_char; 65];
    assert_eq!(rx_digest(c, w, d.as_mut_ptr()), RX_OK);
    assert_eq!(
        unsafe { CStr::from_ptr(d.as_ptr()) }.to_str().unwrap(),
        "5822d803d90146cc81ffa03b8c357ee07e1b1faba2e520342aac9851898ed233"
    );
    // The digest recorded before an assignment stated its axes is this walk with that field removed from the key text.
    let walk = unsafe { &*c }.walk(w).unwrap();
    let field = |name: &str| walk.fields.iter().find(|f| f.0 == name).unwrap().1.clone();
    let mut tail = String::new();
    for ((role, name), tt) in &walk.leaves {
        let ttype = raptor_ir::value::text_val(&raptor_ir::value::ttype_val(
            &tt.dtype,
            &tt.shape,
            tt.tag.as_deref(),
        ));
        let old = raptor_support::migrate::old_role_name(
            role,
            name,
            tt.shape.len(),
            walk.zeroed.contains(&(role.clone(), name.clone())),
        );
        tail.push_str(&format!("leaf|{old}|{name}|{ttype}\n"));
    }
    // a recording of that time names the read planes by the rank of their value
    let arg_spec = field("arg_spec").replace("'plane_in'", "'per_sample'");
    tail.push_str(&format!(
        "arg_spec|{arg_spec}\naccess|{}|None\n",
        walk.access
    ));
    let before = field("preimage")
        .replace("('mutable', 'y', ()))", "('mutable', 'y'))")
        .replace("'plane_in', 'a', ())", "'per_sample', 'a')")
        .replace("'plane_in', 'b', ())", "'per_sample', 'b')");
    assert_eq!(
        rcore::digest(&before, tail.as_bytes()),
        "0db90d308394bc9e5c08cc87f761e9479027b3b2708e945db3864380f004398a"
    );
    let mut order = [0u32; 8];
    let mut n = 0;
    assert_eq!(rx_walk_order(c, w, order.as_mut_ptr(), 8, &mut n), RX_OK);
    assert_eq!(&order[..n], &[a, b, s, k, e, y]);
    let f = CString::new("arg_spec").unwrap();
    let mut buf = [0 as std::ffi::c_char; 256];
    assert_eq!(
        rx_walk_field(c, w, f.as_ptr(), buf.as_mut_ptr(), 256, &mut n),
        RX_OK
    );
    assert_eq!(
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap(),
        "(('mutable', 'y'), ('plane_in', 'a'), ('plane_in', 'b'))"
    );
    walk_slot_answers(c, w, y);
    rx_ctx_free(c);
}

/// The typed answers about a walk's slots: the walk of the test binds ('mutable', 'y') and the two
/// reads, and a canonical walk that keeps its slots binds them again.
fn walk_slot_answers(c: *mut raptor_ir::record::Ctx, w: u32, y: u32) {
    let binds = |w: u32, role: &str, name: &str| {
        let (role, name) = (CString::new(role).unwrap(), CString::new(name).unwrap());
        let mut out = -1;
        assert_eq!(
            rx_walk_binds(c, w, role.as_ptr(), name.as_ptr(), &mut out),
            RX_OK
        );
        out
    };
    assert_eq!(binds(w, "mutable", "y"), 1);
    assert_eq!(binds(w, "plane_in", "b"), 1);
    assert_eq!(binds(w, "plane_in", "y"), 0);
    assert_eq!(binds(w, "terminated", "b"), 0);
    let (mut len, mut count) = (0usize, 0usize);
    assert_eq!(
        rx_walk_args(c, w, std::ptr::null_mut(), 0, &mut len, &mut count),
        RX_TRUNC
    );
    let mut buf = vec![0 as std::ffi::c_char; len + 1];
    assert_eq!(
        rx_walk_args(c, w, buf.as_mut_ptr(), len + 1, &mut len, &mut count),
        RX_OK
    );
    let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
    let texts: Vec<&str> = bytes
        .split(|&b| b == 0)
        .take(count * 2)
        .map(|t| std::str::from_utf8(t).unwrap())
        .collect();
    assert_eq!(count, 3);
    assert_eq!(texts, ["mutable", "y", "plane_in", "a", "plane_in", "b"]);
    let mut again = 0;
    assert_eq!(rx_canonical_like(c, &y, 1, w, &mut again), RX_OK);
    assert_eq!(binds(again, "plane_in", "a"), 1);
    assert_eq!(rx_walk_binds_refuses(c), RX_REFUSED);
}

/// A walk the context does not hold is refused.
fn rx_walk_binds_refuses(c: *mut raptor_ir::record::Ctx) -> i32 {
    let (role, name) = (CString::new("a").unwrap(), CString::new("b").unwrap());
    let mut out = 0;
    rx_walk_binds(c, 999, role.as_ptr(), name.as_ptr(), &mut out)
}

#[test]
fn replay_small_walks() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_replay"))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/walks_small.jsonl"
        ))
        .arg("--roundtrip")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}");
    assert!(
        text.contains("digest-equal 24 / 24") && text.contains("roundtrip-equal 24 / 24"),
        "{text}"
    );
}
