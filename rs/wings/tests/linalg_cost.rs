//! The cost of the linear-algebra ops: operation counts by the documented formulas, the class of the working width, and the
//! bytes of per-lane storage each kind holds.
//!
//! Formulas (the library's own): `lu` n^3/3, `chol` n^3/6, `eigh` S(n) n(n-1)/2 (6n+10) with S the sweep count, a solve n^2 k,
//! a triangular half n^2 k / 2; the determinant of an LU factor n multiplies; the other reads of a factor (status, fields) cost nothing. Bytes, for an element of `w` bytes: `lu` n^2 w + 4n + 4, `chol` n^2 w + 4,
//! `eigh` (2n^2 + n) w + 4, a solve its result.
use raptor_ir::record::Ctx;
use raptor_wings::cost::fp64_ops;
use raptor_wings::working_set::working_set_bytes;

/// A context with the node builders of the tests.
struct Graph {
    c: Ctx,
}

impl Graph {
    fn new() -> Graph {
        Graph {
            c: Ctx::new().unwrap(),
        }
    }

    /// A matrix or vector input plane of the shape.
    fn input(&mut self, dtype: &str, shape: &[i64], role: &str) -> u32 {
        let fields = format!(
            "(('kind', 'vocab_read'), ('role', '{role}'), ('name', 'A'), ('quantity', None), ('wire_index', 0), ('sorted', False))"
        );
        self.c
            .make("Leaf", dtype, shape, None, &[], &fields)
            .unwrap()
    }

    /// An op of the kind over the operands.
    fn op(&mut self, kind: &str, dtype: &str, shape: &[i64], ops: &[u32]) -> u32 {
        let fields = format!("(('kind', '{kind}'), ('literal', None))");
        self.c.make("Op", dtype, shape, None, ops, &fields).unwrap()
    }

    /// The factorisation `kind` of an `n` by `n` matrix of the element type.
    fn factor(&mut self, kind: &str, dtype: &str, n: i64) -> u32 {
        let a = self.input(dtype, &[n, n], "mat_in");
        self.op(kind, dtype, &[n, n], &[a])
    }

    fn ops_of(&self, nodes: &[u32]) -> i64 {
        fp64_ops(&self.c, nodes).unwrap()
    }

    fn bytes_of(&self, nodes: &[u32]) -> i64 {
        working_set_bytes(&self.c, nodes, None, "thread").unwrap()
    }
}

#[test]
fn factorisation_operation_counts_are_the_formulas() {
    // n is a multiple of 6 so that n^3/3 and n^3/6 are whole numbers
    let mut g = Graph::new();
    for n in [6_i64, 12, 18] {
        let (lu, chol) = (g.factor("lu", "f64", n), g.factor("chol", "f64", n));
        assert_eq!(g.ops_of(&[lu]), n * n * n / 3);
        assert_eq!(g.ops_of(&[chol]), n * n * n / 6);
    }
}

#[test]
fn the_sweep_count_of_an_eigen_decomposition_steps_with_the_dimension() {
    // sweeps by n: 1-2 -> 2, 3-4 -> 6, 5-8 -> 8, 9-16 -> 12, 17 and up -> 15; n = 1 has no off-diagonal pair, so it is not listed
    let table = [
        (2, 2),
        (3, 6),
        (4, 6),
        (5, 8),
        (8, 8),
        (9, 12),
        (16, 12),
        (17, 15),
        (32, 15),
    ];
    let mut g = Graph::new();
    for (n, sweeps) in table {
        let eigh = g.factor("eigh", "f64", n);
        assert_eq!(
            g.ops_of(&[eigh]),
            sweeps * n * (n - 1) / 2 * (6 * n + 10),
            "n = {n}"
        );
    }
}

#[test]
fn solve_operation_counts_are_n_squared_times_the_right_hand_columns() {
    let mut g = Graph::new();
    for (n, k) in [(4_i64, 3_i64), (32, 5)] {
        let (lu, chol) = (g.factor("lu", "f64", n), g.factor("chol", "f64", n));
        let rhs = g.input("f64", &[n, k], "mat_in");
        let by_lu = g.op("lu_solve", "f64", &[n, k], &[lu, rhs]);
        let by_chol = g.op("cho_solve", "f64", &[n, k], &[chol, rhs]);
        assert_eq!(g.ops_of(&[by_lu]), n * n * k);
        assert_eq!(g.ops_of(&[by_chol]), n * n * k);
    }
}

#[test]
fn the_derivative_solves_cost_their_triangular_halves() {
    let mut g = Graph::new();
    for n in [4_i64, 32] {
        let (lu, chol) = (g.factor("lu", "f64", n), g.factor("chol", "f64", n));
        let rhs = g.input("f64", &[n, n], "mat_in");
        let transposed = g.op("lu_solve_t", "f64", &[n, n], &[lu, rhs]);
        let lower = g.op("tri_lower", "f64", &[n, n], &[chol, rhs]);
        let upper = g.op("tri_upper_t", "f64", &[n, n], &[chol, rhs]);
        assert_eq!(g.ops_of(&[transposed]), n * n * n);
        assert_eq!(g.ops_of(&[lower]), n * n * n / 2);
        assert_eq!(g.ops_of(&[upper]), n * n * n / 2);
    }
}

#[test]
fn the_field_reads_of_a_factorisation_are_free_in_the_cost_model() {
    let mut g = Graph::new();
    let n = 6;
    let (lu, chol, eigh) = (
        g.factor("lu", "f64", n),
        g.factor("chol", "f64", n),
        g.factor("eigh", "f64", n),
    );
    let free = [
        g.op("lu_info", "i32", &[], &[lu]),
        g.op("chol_info", "i32", &[], &[chol]),
        g.op("chol_factor", "f64", &[n, n], &[chol]),
        g.op("eigh_w", "f64", &[n], &[eigh]),
        g.op("eigh_v", "f64", &[n, n], &[eigh]),
        g.op("eigh_info", "i32", &[], &[eigh]),
    ];
    for node in free {
        assert_eq!(g.ops_of(&[node]), 0);
    }
}

#[test]
fn the_determinant_of_an_lu_factor_costs_n_multiplies() {
    // the old figure: 6 for a 6 by 6 factor, in the class of its width; a 32-bit determinant counts no 64-bit work
    let mut g = Graph::new();
    let (wide, narrow) = (g.factor("lu", "f64", 6), g.factor("lu", "f32", 6));
    let det = g.op("lu_det", "f64", &[], &[wide]);
    let det32 = g.op("lu_det", "f32", &[], &[narrow]);
    assert_eq!(g.ops_of(&[det]), 6);
    assert_eq!(g.ops_of(&[det32]), 0);
}

#[test]
fn factorisation_cost_lands_in_the_class_of_the_working_width() {
    // the 64-bit total counts only wide work: n^3/3 for a 64-bit factor, nothing for the same factor in 32 bits
    let mut g = Graph::new();
    let (wide, narrow) = (g.factor("lu", "f64", 6), g.factor("lu", "f32", 6));
    assert_eq!(g.ops_of(&[wide]), 72);
    assert_eq!(g.ops_of(&[narrow]), 0);
}

#[test]
fn storage_bytes_per_kind_follow_the_documented_sizes() {
    let mut g = Graph::new();
    for (n, w, dtype) in [
        (1_i64, 8_i64, "f64"),
        (5, 8, "f64"),
        (32, 4, "f32"),
        (7, 4, "f32"),
    ] {
        let (lu, chol, eigh) = (
            g.factor("lu", dtype, n),
            g.factor("chol", dtype, n),
            g.factor("eigh", dtype, n),
        );
        assert_eq!(g.bytes_of(&[lu]), n * n * w + 4 * n + 4);
        assert_eq!(g.bytes_of(&[chol]), n * n * w + 4);
        assert_eq!(g.bytes_of(&[eigh]), (2 * n * n + n) * w + 4);
        let rhs = g.input(dtype, &[n, 2], "mat_in");
        let solve = g.op("lu_solve", dtype, &[n, 2], &[lu, rhs]);
        assert_eq!(g.bytes_of(&[solve]), n * 2 * w);
        let column = g.input(dtype, &[n], "vec_in");
        let by_chol = g.op("cho_solve", dtype, &[n], &[chol, column]);
        assert_eq!(g.bytes_of(&[by_chol]), n * w);
        let factor = g.op("chol_factor", dtype, &[n, n], &[chol]);
        assert_eq!(g.bytes_of(&[factor]), n * n * w);
        let values = g.op("eigh_w", dtype, &[n], &[eigh]);
        assert_eq!(g.bytes_of(&[values]), n * w);
    }
}

#[test]
fn the_stack_of_a_kernel_is_the_sum_over_its_nodes() {
    let mut g = Graph::new();
    let lu = g.factor("lu", "f64", 32);
    let column = g.input("f64", &[32], "vec_in");
    let solve = g.op("lu_solve", "f64", &[32], &[lu, column]);
    let plane = g.input("f64", &[32, 32], "mat_in");
    assert_eq!(
        g.bytes_of(&[lu, solve, plane]),
        32 * 32 * 8 + 4 * 32 + 4 + 32 * 8
    );
    assert_eq!(g.bytes_of(&[]), 0);
    // the 64-bit factor of dimension 32 alone is above a 4 KB stack
    let chol = g.factor("chol", "f64", 32);
    assert!(g.bytes_of(&[chol]) > 4096);
}

#[test]
fn a_scalar_result_of_a_factorisation_adds_nothing_to_the_working_set() {
    // the working set counts the nodes that hold storage per lane: a determinant and a status word are registers
    let mut g = Graph::new();
    let lu = g.factor("lu", "f64", 6);
    let det = g.op("lu_det", "f64", &[], &[lu]);
    let info = g.op("lu_info", "i32", &[], &[lu]);
    assert_eq!(g.bytes_of(&[det, info]), 0);
}
