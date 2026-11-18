//! The quantities a walk reaches. Oracle: derived by hand from the fixture below, not read from the code under test.
use raptor_ir::record::Ctx;
use raptor_ir::typed::{RxPair, RxPairPart};
use raptor_ir::walkreads::{rx_walk_quantity_slots, rx_walk_quantity_wires};

/// A read leaf of a wire of the quantity `slug`: the slot is named `<slug>__<wire>` and the leaf carries the whole declaration.
fn wire_leaf(c: &mut Ctx, slug: &str, wires: &[&str], at: usize) -> u32 {
    let declared: Vec<String> = wires
        .iter()
        .map(|w| format!("Wire(name='{w}', role='vec_in', shape=(), dtype='f64', tag=None)"))
        .collect();
    let quantity = format!(
        "Quantity(slug='{slug}', pack='auto', wires=({},))",
        declared.join(", ")
    );
    let fields = format!("(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '{slug}__{}'), ('quantity', {quantity}), ('wire_index', {at}), ('sorted', False))", wires[at]);
    c.make("Leaf", "f64", &[], None, &[], &fields).unwrap()
}

/// The text of a pair part.
fn text(p: &RxPairPart) -> String {
    unsafe { std::ffi::CStr::from_ptr(p.text.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn a_walk_names_its_quantities_with_their_wires_and_slots_in_walk_order() {
    let mut c = Ctx::new().unwrap();
    // The walk reaches `pos` (wires x, v) and then `acc` (wire w); `y` is a plain plane.
    // Slots are ordered by role (mutable, out, ..., vec_in, ...), and inside a role the wires of the reached quantities come first, in
    // the order the quantities are reached and then in declaration order. So the mutable plane `y` is slot 0 and the vec_in slots are
    // pos__x 1, pos__v 2, acc__w 3.
    let (px, pv) = (
        wire_leaf(&mut c, "pos", &["x", "v"], 0),
        wire_leaf(&mut c, "pos", &["x", "v"], 1),
    );
    let aw = wire_leaf(&mut c, "acc", &["w"], 0);
    let add = |c: &mut Ctx, a: u32, b: u32| {
        c.make(
            "Op",
            "f64",
            &[],
            None,
            &[a, b],
            "(('kind', 'add'), ('literal', None))",
        )
        .unwrap()
    };
    let s1 = add(&mut c, px, pv);
    let value = add(&mut c, s1, aw);
    let sink = c
        .make(
            "Assign",
            "f64",
            &[],
            None,
            &[value],
            "(('kind', 'assign'), ('name', 'y'), ('role', 'mutable'))",
        )
        .unwrap();
    let walk = c.canonical(&[sink], &[], None, &[]).unwrap() as u32;
    let ctx: *mut Ctx = &mut c;
    let rows =
        |read: unsafe extern "C" fn(*mut Ctx, u32, *mut RxPair, usize, *mut usize) -> i32| {
            let mut buf = Vec::new();
            buf.resize_with(8, || unsafe { std::mem::zeroed::<RxPair>() });
            let mut n = 0;
            assert_eq!(
                unsafe { read(ctx, walk, buf.as_mut_ptr(), buf.len(), &mut n) },
                0
            );
            buf.truncate(n);
            buf
        };
    let wires: Vec<(String, String)> = rows(rx_walk_quantity_wires)
        .iter()
        .map(|p| (text(&p.first), text(&p.second)))
        .collect();
    let want: Vec<(String, String)> = [("pos", "pos__x"), ("pos", "pos__v"), ("acc", "acc__w")]
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .into();
    assert_eq!(wires.len(), 3, "{wires:?}");
    let slots: Vec<(String, i64)> = rows(rx_walk_quantity_slots)
        .iter()
        .map(|p| (text(&p.first), p.second.number))
        .collect();
    assert_eq!(
        slots,
        [
            ("pos".to_string(), 1),
            ("pos".to_string(), 2),
            ("acc".to_string(), 3)
        ]
    );
    assert_eq!(wires, want);
}
