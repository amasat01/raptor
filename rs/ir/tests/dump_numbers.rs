//! The number a root has in a dump: the line of the `root|#p` entry the dump writes for it.
use raptor_ir::dump::{rx_dump, rx_dump_numbers};
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;
use std::ffi::{c_char, CStr};

/// The numbers the door gives agree with the `root|#p` lines of the dump of the same roots, in any root order.
#[test]
fn a_root_number_is_the_line_its_dump_gives_it() {
    let mut c = Ctx::new().unwrap();
    let fields = "(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))";
    let x = c.make("Leaf", "f64", &[], None, &[], fields).unwrap();
    let sum = c.make_op("add", &[x, x], &Val::None).unwrap();
    let product = c.make_op("mul", &[sum, x], &Val::None).unwrap();
    let ctx: *mut Ctx = &mut c;
    for roots in [vec![product, x], vec![x, sum, product], vec![]] {
        let mut numbers = [0_i64; 8];
        let mut count = 0;
        let status = unsafe {
            rx_dump_numbers(
                ctx,
                roots.as_ptr(),
                roots.len(),
                numbers.as_mut_ptr(),
                8,
                &mut count,
            )
        };
        assert_eq!((status, count), (0, roots.len()));
        let (mut text, mut len) = ([0 as c_char; 4096], 0);
        assert_eq!(
            unsafe {
                rx_dump(
                    ctx,
                    roots.as_ptr(),
                    roots.len(),
                    text.as_mut_ptr(),
                    4096,
                    &mut len,
                )
            },
            0
        );
        let dump = unsafe { CStr::from_ptr(text.as_ptr()) }.to_str().unwrap();
        let listed: Vec<i64> = dump
            .lines()
            .filter_map(|l| l.strip_prefix("root|#"))
            .map(|p| p.parse().unwrap())
            .collect();
        assert_eq!(&numbers[..count], &listed[..], "{roots:?}");
    }
}
