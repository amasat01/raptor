# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The node helpers and constants of raptor.ir.nodes: literal round-trip, parameter cells, index equality, clamp peeling, sample sums."""
import pytest

from raptor.errors import RaptorError
from raptor.ir import cells, ops
from raptor.ir.nodes import (ADJOINT_PREFIX, COMPANION_SUFFIX, GATHER_ROLES, RAW_LANGUAGE, ROLE_ORDER, Assign, Const, Leaf, SampleSum,
                             compensates, element_count, is_param_cell, param_cell_names, peel_clamp, plain_in, plain_out,
                             sample_sums, same_index)
from raptor.ir.types import TensorType

F64, I32 = TensorType((), "f64"), TensorType((), "i32")


def test_the_naming_constants_are_the_documented_literals():
    assert (ADJOINT_PREFIX, COMPANION_SUFFIX, RAW_LANGUAGE) == ("bar", "_c", "aether")
    assert GATHER_ROLES == ("lookup", "wide_in") and set(GATHER_ROLES) <= set(ROLE_ORDER)


def test_a_compensated_or_keyed_auto_accumulate_commits_through_a_companion():
    assert compensates("compensated") and compensates("keyed_auto")
    assert not compensates("fast") and not compensates("")


def test_a_literal_round_trips_through_plain_data_with_tuples_tagged():
    value = (1, (2.5, "s"), True)
    data = plain_out(value)
    assert data == {"tuple": [1, {"tuple": [2.5, "s"]}, True]}
    assert plain_in(data) == value and plain_in(7) == 7 and plain_out(None) is None
    with pytest.raises(RaptorError, match="not covered"):
        plain_out([1])


def test_a_parameter_cell_is_recognised_and_listed_in_first_seen_order():
    first, second = cells.param_cell("g", F64), cells.param_cell("a", F64)
    plain = Leaf("table_read", "lookup", "t", F64)
    assert is_param_cell(first) and is_param_cell(second)
    assert not is_param_cell(plain) and not is_param_cell(Const(0, I32))
    assert param_cell_names((second, plain, first, second)) == ("a", "g")
    assert param_cell_names(()) == ()


def test_peel_clamp_strips_one_clamp_index_and_nothing_else():
    i, n = Leaf("vocab_read", "per_sample", "i", I32), Const(8, I32)
    clamped = ops.make("clamp_index", (i, n))
    assert peel_clamp(clamped) is i and peel_clamp(i) is i
    assert peel_clamp(ops.make("add", (i, n))).kind == "add"


def test_same_index_is_one_node_or_equal_kind_literal_and_operands():
    i = Leaf("vocab_read", "per_sample", "i", I32)
    a, b = ops.make("add", (i, Const(1, I32))), ops.make("add", (i, Const(1, I32)))
    assert same_index(a, a) and same_index(a, b)
    assert not same_index(a, ops.make("add", (i, Const(2, I32))))
    assert not same_index(a, ops.make("sub", (i, Const(1, I32))))
    assert not same_index(Const(1, I32), Const(1, F64))


def test_sample_sums_are_the_sinks_of_that_kind_in_order():
    one, two = SampleSum("a", Const(1.0, F64)), SampleSum("b", Const(2.0, F64))
    keep = Assign("y", Const(0.0, F64), F64)
    assert sample_sums((one, keep, two)) == (one, two) and sample_sums((keep,)) == ()


def test_the_element_count_is_the_product_of_the_shape():
    assert element_count(F64) == 1 and element_count(TensorType((2, 3, 4), "f64")) == 24 and element_count(TensorType((5,), "i32")) == 5
