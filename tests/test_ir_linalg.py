# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The linear-algebra kinds (raptor.ir.linalg): the vocabulary, the cost formulas of aether::linalg, the storage bytes, and the
matrix reading of a factor. The formulas: ``lu`` n^3/3, ``chol`` n^3/6, ``eigh`` S(n) n(n-1)/2 (6n+10), a solve n^2 k."""
import pytest

from raptor.errors import RaptorError
from raptor.ir.linalg import (CONSUMERS, FACTOR_TAG, FACTORISATIONS, LINALG_OPS, PROJECTIONS, as_matrix, dimension, factor_bytes,
                              is_linalg, op_bytes, op_count, sweeps)
from raptor.ir.nodes import Leaf
from raptor.ir.ops import OP_KINDS, make
from raptor.ir.types import TensorType


def _mat(n, dtype="f64", name="A"):
    return Leaf("vocab_read", "mat_in", name, TensorType((n, n), dtype))


def _vec(n, dtype="f64"):
    return Leaf("vocab_read", "vec_in", "b", TensorType((n,), dtype))


def _rhs(n, k, dtype="f64"):
    return Leaf("vocab_read", "mat_in", "B", TensorType((n, k), dtype))


def _three(n, dtype="f64"):
    return tuple(make(k, (_mat(n, dtype),)) for k in ("lu", "chol", "eigh"))


def test_the_kinds_are_in_the_vocabulary():
    assert set(LINALG_OPS) <= set(OP_KINDS)
    assert LINALG_OPS == FACTORISATIONS + CONSUMERS + PROJECTIONS
    assert FACTORISATIONS == ("lu", "chol", "eigh", "chol_from", "qr")
    assert {"lu_solve", "cho_solve"} <= set(CONSUMERS) and {"lu_det", "eigh_w", "chol_factor"} <= set(PROJECTIONS)


def test_a_factorisation_carries_the_tag_of_its_kind():
    assert FACTOR_TAG == {"lu": "lu_factor", "chol": "chol_factor", "eigh": "eigh_factor", "chol_from": "chol_factor", "qr": "qr_factor"}
    for kind, tag in FACTOR_TAG.items():
        if kind not in ("qr",):
            assert make(kind, (_mat(4),)).ttype.tag == tag


def test_is_linalg_and_dimension():
    lu = make("lu", (_mat(5),))
    assert is_linalg(lu) and is_linalg(make("lu_solve", (lu, _vec(5)))) and is_linalg(make("lu_det", (lu,)))
    assert not is_linalg(make("add", (_vec(5), _vec(5)))) and not is_linalg(_mat(5))
    assert dimension(lu) == 5 and dimension(make("lu_solve", (lu, _rhs(5, 2)))) == 5
    assert dimension(make("mv", (_mat(5), _vec(5)))) == 5


@pytest.mark.parametrize("n", [1, 2, 5, 16, 32])
def test_factorisation_operation_counts_are_the_formulas(n):
    assert op_count(make("lu", (_mat(n),))) == pytest.approx(n ** 3 / 3)
    assert op_count(make("chol", (_mat(n),))) == pytest.approx(n ** 3 / 6)
    assert op_count(make("eigh", (_mat(n),))) == sweeps(n) * n * (n - 1) / 2 * (6 * n + 10)


def test_the_sweep_table_is_aethers():
    assert [sweeps(n) for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)] == [2, 2, 6, 6, 8, 8, 12, 12, 15, 15]


@pytest.mark.parametrize("n,k", [(1, 1), (4, 1), (4, 3), (32, 5)])
def test_solve_operation_counts_are_n_squared_k_and_the_triangular_ones_half(n, k):
    lu, chol, _ = _three(n)
    rhs = _vec(n) if k == 1 else _rhs(n, k)
    assert op_count(make("lu_solve", (lu, rhs))) == n * n * k
    assert op_count(make("cho_solve", (chol, rhs))) == n * n * k
    if k > 1:
        assert op_count(make("tri_lower", (chol, rhs))) == n * n * k / 2


def test_a_determinant_costs_n_and_a_projection_nothing():
    lu, chol, eigh = _three(6)
    assert op_count(make("lu_det", (lu,))) == 6
    for node in (make("lu_info", (lu,)), make("chol_factor", (chol,)), make("eigh_w", (eigh,)), make("eigh_v", (eigh,))):
        assert op_count(node) == 0


@pytest.mark.parametrize("n,w,dtype", [(1, 8, "f64"), (5, 8, "f64"), (32, 4, "f32"), (7, 4, "f32")])
def test_storage_bytes_per_kind(n, w, dtype):
    lu, chol, eigh = _three(n, dtype)
    assert op_bytes(lu) == n * n * w + 4 * n + 4
    assert op_bytes(chol) == n * n * w + 4
    assert op_bytes(eigh) == (2 * n * n + n) * w + 4
    assert op_bytes(make("lu_solve", (lu, _rhs(n, 2, dtype)))) == n * 2 * w
    assert op_bytes(make("lu_det", (lu,))) == w and op_bytes(make("lu_info", (lu,))) == 4
    # the bytes a factor type holds are the bytes of the node that makes it
    for node in (lu, chol, eigh):
        assert factor_bytes(node.ttype, w) == op_bytes(node)


def test_a_cholesky_factor_used_as_a_matrix_is_its_lower_projection():
    chol = make("chol", (_mat(3),))
    as_m = as_matrix(chol)
    assert as_m.kind == "chol_factor" and as_m.operands == (chol,) and as_m.ttype == TensorType((3, 3), "f64")
    assert make("mv", (chol, _vec(3))).operands[0].kind == "chol_factor"
    plain = _mat(3)
    assert as_matrix(plain) is plain


@pytest.mark.parametrize("kind,what", [("lu", "an LU factor is not a matrix"), ("eigh", "an eigh factor is not a matrix")])
def test_an_lu_or_eigh_factor_used_as_a_matrix_is_refused_by_name(kind, what):
    with pytest.raises(RaptorError, match=what):
        as_matrix(make(kind, (_mat(3),)))
