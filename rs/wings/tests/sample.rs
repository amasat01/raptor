//! The sampled read written as `At` reads: each `Sample` is built on a hand-made kernel, expanded, and the expansion is evaluated
//! by a small interpreter against the interpolation as its definition states it (written here per axis, not read off the nodes).
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;
use raptor_wings::declare::rw_declare_kinds;
use raptor_wings::sample::expand;

/// A table of up to three axes: its extents and its cells in storage order, the last axis fastest.
struct Table {
    /// The extent of each axis.
    extents: Vec<i64>,
    /// The cells.
    cells: Vec<f64>,
}

impl Table {
    /// The cell at `at` under the edge rule, the same rule for every axis.
    fn read(&self, at: &[i64], edge: &str) -> f64 {
        let mut cell = 0;
        for (&i, &n) in at.iter().zip(&self.extents) {
            let i = match edge {
                "clamp" => i.clamp(0, n - 1),
                "wrap" => i.rem_euclid(n),
                _ if (0..n).contains(&i) => i,
                _ => return 0.0,
            };
            cell = cell * n + i;
        }
        self.cells[cell as usize]
    }
}

/// The value of the expression `node` for the coordinates `x` (a leaf named `x0`, `x1`, ... reads `x[0]`, `x[1]`, ...).
fn eval(c: &Ctx, node: u32, table: &Table, x: &[f64]) -> f64 {
    let ops = c.nodes[node as usize].ops.clone();
    let kind = c.str_field(node, "kind").unwrap_or_default();
    if c.nodes[node as usize].ops.is_empty() || kind == "constant" {
        return match (kind.as_str(), c.field(node, "literal")) {
            ("vocab_read", _) => {
                let name = c.str_field(node, "name").unwrap();
                x[name[1..].parse::<usize>().unwrap()]
            }
            (_, Ok(Val::Float(f))) => f,
            (_, Ok(Val::Int(n))) => n as f64,
            other => panic!("a leaf of {other:?}"),
        };
    }
    if kind == "at" {
        let at: Vec<i64> = ops[1..]
            .iter()
            .map(|&o| eval(c, o, table, x) as i64)
            .collect();
        return table.read(&at, &c.str_field(node, "edge").unwrap());
    }
    let a: Vec<f64> = ops.iter().map(|&o| eval(c, o, table, x)).collect();
    match kind.as_str() {
        "add" => a[0] + a[1],
        "sub" => a[0] - a[1],
        "mul" => a[0] * a[1],
        "floor" => a[0].floor(),
        "cast" => a[0].trunc(),
        other => panic!("the expansion holds only add, sub, mul, floor, cast and At, not {other}"),
    }
}

/// Whether the expression `node` has a `Sample` anywhere below it.
fn holds_a_sample(c: &Ctx, node: u32) -> bool {
    c.str_field(node, "kind").is_ok_and(|kind| kind == "sample")
        || c.nodes[node as usize]
            .ops
            .iter()
            .any(|&o| holds_a_sample(c, o))
}

/// A context with a lookup plane `t` of the given extents and one real coordinate leaf per axis.
fn kernel(extents: &[i64]) -> (Ctx, u32, Vec<u32>) {
    let mut c = Ctx::new().unwrap();
    assert_eq!(unsafe { rw_declare_kinds(&mut c) }, 0);
    let axes: Vec<String> = extents
        .iter()
        .enumerate()
        .map(|(k, n)| format!("('a{k}', {n})"))
        .collect();
    let plane = format!(
        "(('kind', 'table_read'), ('role', 'lookup'), ('name', 't'), ('axes', ({},)))",
        axes.join(", ")
    );
    let plane = c.make("Leaf", "f64", &[], None, &[], &plane).unwrap();
    let coordinates = (0..extents.len())
        .map(|k| {
            let fields = format!(
                "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'x{k}'), ('quantity', None), ('wire_index', 0), ('sorted', False))"
            );
            c.make("Leaf", "f64", &[], None, &[], &fields).unwrap()
        })
        .collect();
    (c, plane, coordinates)
}

/// The `Sample` of the plane at the coordinates, or the refusal text.
fn sample(
    (c, plane, coordinates): &mut (Ctx, u32, Vec<u32>),
    (filter, edge): (&str, &str),
) -> Result<u32, String> {
    let mut ops = vec![*plane];
    ops.extend(coordinates.iter());
    let fields = format!(
        "(('kind', 'sample'), ('literal', None), ('filter', '{filter}'), ('edge', '{edge}'), ('hardware', False))"
    );
    c.make("Sample", "f64", &[], None, &ops, &fields)
        .map_err(|fault| format!("{fault:?}"))
}

/// The interpolation of `table` at `x` as the definition states it, axis by axis.
fn reference(table: &Table, x: &[f64], (filter, edge): (&str, &str)) -> f64 {
    let n = x.len();
    if filter == "nearest" {
        let at: Vec<i64> = x.iter().map(|v| (v + 0.5).floor() as i64).collect();
        return table.read(&at, edge);
    }
    let low: Vec<i64> = x.iter().map(|v| v.floor() as i64).collect();
    let mut total = 0.0;
    for corner in 0..1usize << n {
        let (mut at, mut weight) = (vec![], 1.0);
        for k in 0..n {
            let w = x[k] - x[k].floor();
            let high = corner >> (n - 1 - k) & 1 == 1;
            at.push(low[k] + i64::from(high));
            weight *= if high { w } else { 1.0 - w };
        }
        total += weight * table.read(&at, edge);
    }
    total
}

/// Coordinates that are whole, fractional, on the last cell, and beyond both ends.
const POINTS: [f64; 9] = [0.0, 0.25, 1.0, 1.5, 2.5, 3.0, -0.5, -2.75, 4.5];

#[test]
fn an_expanded_sample_equals_the_interpolation_for_every_dimension_filter_and_edge_rule() {
    for extents in [vec![4], vec![4, 3], vec![4, 3, 2]] {
        let cells = (0..extents.iter().product::<i64>())
            .map(|k| ((k * 7 + 3) % 11) as f64 - 4.5)
            .collect();
        let table = Table {
            extents: extents.clone(),
            cells,
        };
        for filter in ["nearest", "linear"] {
            for edge in ["clamp", "wrap", "zero"] {
                let mut k = kernel(&extents);
                let node = sample(&mut k, (filter, edge)).unwrap();
                let done = expand(&mut k.0, node).unwrap();
                assert!(!holds_a_sample(&k.0, done), "a Sample remains");
                for &a in &POINTS {
                    let x: Vec<f64> = (0..extents.len())
                        .map(|d| a + 0.5 * d as f64 - 0.125)
                        .collect();
                    let want = reference(&table, &x, (filter, edge));
                    let got = eval(&k.0, done, &table, &x);
                    assert!(
                        (got - want).abs() < 1e-12,
                        "{filter} {edge} {extents:?} at {x:?}: {got} vs {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_nearest_sample_rounds_a_half_up() {
    let table = Table {
        extents: vec![5],
        cells: vec![10.0, 11.0, 12.0, 13.0, 14.0],
    };
    let mut k = kernel(&[5]);
    let node = sample(&mut k, ("nearest", "clamp")).unwrap();
    let done = expand(&mut k.0, node).unwrap();
    for (x, cell) in [
        (0.49, 0),
        (0.5, 1),
        (1.5, 2),
        (2.5, 3),
        (-0.5, 0),
        (-0.51, 0),
        (3.4999, 3),
    ] {
        assert_eq!(eval(&k.0, done, &table, &[x]), table.cells[cell], "x = {x}");
    }
}

#[test]
fn a_sample_is_refused_by_name_when_it_cannot_be_read() {
    let mut two = kernel(&[4, 3]);
    two.2.pop();
    let one_short = sample(&mut two, ("linear", "clamp")).unwrap_err();
    assert!(
        one_short.contains("one coordinate per axis, got 1"),
        "{one_short}"
    );

    let mut four = kernel(&[2, 2, 2, 2]);
    let too_many = sample(&mut four, ("linear", "clamp")).unwrap_err();
    assert!(
        too_many.contains("one to three coordinates, got 4"),
        "{too_many}"
    );

    let mut one = kernel(&[4]);
    for (options, wanted) in [
        (("cubic", "clamp"), "filter 'cubic'"),
        (("linear", "unchecked"), "edge rule 'unchecked'"),
        (("linear", "mirror"), "edge rule 'mirror'"),
    ] {
        let fault = sample(&mut one, options).unwrap_err();
        assert!(fault.contains(wanted), "{fault}");
    }

    let whole = one
        .0
        .make("Const", "i32", &[], None, &[], "(('literal', 1),)")
        .unwrap();
    one.2 = vec![whole];
    let fault = sample(&mut one, ("linear", "clamp")).unwrap_err();
    assert!(fault.contains("a rank-0 real"), "{fault}");
}

#[test]
fn a_hardware_fetch_is_refused_as_not_written_yet() {
    let (mut c, plane, coordinates) = kernel(&[4]);
    let ops = [plane, coordinates[0]];
    let fields = "(('kind', 'sample'), ('literal', None), ('filter', 'linear'), ('edge', 'clamp'), ('hardware', True))";
    let fault = c
        .make("Sample", "f64", &[], None, &ops, fields)
        .unwrap_err();
    assert!(
        format!("{fault:?}").contains("hardware fetch is not written yet"),
        "{fault:?}"
    );
}

#[test]
fn a_linear_sample_of_an_integer_table_is_refused_and_a_nearest_one_is_not() {
    let (mut c, _, coordinates) = kernel(&[4]);
    let plane =
        "(('kind', 'table_read'), ('role', 'lookup'), ('name', 'n'), ('axes', (('a0', 4),)))";
    let plane = c.make("Leaf", "i32", &[], None, &[], plane).unwrap();
    let mut counts = (c, plane, coordinates);
    let fault = sample(&mut counts, ("linear", "clamp")).unwrap_err();
    assert!(fault.contains("takes a real table"), "{fault}");
    assert!(sample(&mut counts, ("nearest", "clamp")).is_ok());
}
