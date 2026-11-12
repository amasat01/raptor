# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Node construction and identity: a node refuses a bad literal or type when it is BUILT (naming the kind and the field), cannot lose
a slot after it is built, and its structural key agrees with the canonical form on every literal."""

import math
import re

import pytest

from raptor import _core
from raptor.ir import nodes
from raptor.errors import RaptorError
from raptor.ir.types import TensorType
from raptor.ir.walk import canonical_nodes, structural_keys, post_order

F64 = TensorType((), "f64")


def test_a_slot_cannot_be_deleted():
    c = nodes.Const(1.0, F64)
    with pytest.raises(RaptorError, match=r"Const\.literal: a node is not changed after it is built"):
        del c.literal
    assert c.literal == 1.0


@pytest.mark.parametrize("bad, what", [
    (len, "builtin_function_or_method"), (lambda: 0, "function"), ((1, len), "builtin_function_or_method"), (object(), "object"),
])
def test_a_literal_without_a_written_rule_is_refused_at_construction(bad, what):
    with pytest.raises(RaptorError, match=rf"Const\.literal: key text: a value of type {what} has no written rule"):
        nodes.Const(bad, F64)
    with pytest.raises(RaptorError, match=rf"Op\.literal: key text: a value of type {what}"):
        nodes.Op("add", (), F64, bad)


def test_a_list_literal_is_refused_at_construction():
    with pytest.raises(RaptorError, match=r"Const\.literal: a list is not a literal"):
        nodes.Const([1, 2], F64)
    with pytest.raises(RaptorError, match=r"Op\.literal: a list is not a literal"):
        nodes.Op("add", (), F64, (1, [2]))


def test_a_huge_int_literal_is_refused_at_construction():
    with pytest.raises(RaptorError, match=r"Const\.literal: key text: an int beyond 128 bits"):
        nodes.Const(1 << 200, F64)


@pytest.mark.parametrize("lit", [None, True, 3, 2.5, "s", (1, (2.0, "x")), math.nan, -0.0])
def test_plain_literals_construct(lit):
    nodes.Const(lit, F64)


def test_a_type_with_an_unknown_dtype_is_refused_at_construction():
    with pytest.raises(RaptorError, match=r"Const: TensorType\.dtype 'f16' is not one of \('f64'"):
        nodes.Const(1.0, TensorType((), "f16"))


def test_a_type_with_a_bad_shape_is_refused_at_construction():
    with pytest.raises(RaptorError, match=r"Const: TensorType\.shape \(-1,\) has a negative extent"):
        nodes.Const(1.0, TensorType((-1,), "f64"))
    with pytest.raises(RaptorError, match=r"Const: TensorType\.shape \[2\] is not a tuple"):
        nodes.Const(1.0, TensorType([2], "f64"))
    with pytest.raises(RaptorError, match=r"Const: TensorType\.shape \(1\.5,\) holds 1\.5, which is not an int"):
        nodes.Const(1.0, TensorType((1.5,), "f64"))


_PAIRS = [(1, 1.0), (1, True), (0.0, -0.0), (math.nan, math.nan), (2.0, 2.0), ("a", "a"), (1 << 100, 1 << 100), (math.nan, -math.nan),
          (math.inf, math.inf), ((1, 2.0), (1, 2.0)), ((1, 2.0), (1.0, 2.0))]


@pytest.mark.parametrize("a, b", _PAIRS, ids=[f"{a!r}-{b!r}" for a, b in _PAIRS])
def test_structural_keys_agree_with_the_canonical_form(a, b):
    x, y = nodes.Const(a, F64), nodes.Const(b, F64)
    order, _ = canonical_nodes([x, y])
    merged = len(order) == 1
    memo: dict = {}
    structural_keys(post_order([x, y]), memo)
    assert (memo[id(x)] == memo[id(y)]) is merged, f"{a!r} vs {b!r}: structural key {memo[id(x)] == memo[id(y)]}, canonical merge {merged}"
