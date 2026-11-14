# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Loops (raptor.ir.loops): the two classes, the classifier, the add-chain flattening, the boundary and the rebuild keywords."""
import dataclasses

import pytest

from raptor.errors import RaptorError
from raptor.ir.loops import (ACCUMULATOR, LOOP_CLASSES, RECURRENCE, LoopClass, addends, boundary, build_loop, classify_loop,
                             exit_kwargs, refuse_unsupported_nesting)
from raptor.ir.nodes import At, Const, Leaf, LoopCarry, LoopIndex, Select, body_key_widths
from raptor.ir.ops import make
from raptor.ir.types import TensorType

F64, I32 = TensorType((), "f64"), TensorType((), "i32")
IDX = LoopIndex(0, "k")
CARRY = LoopCarry(0, 0, "c", F64)


def _loop(nxt, **kw):
    return build_loop(IDX, (CARRY,), (Const(0.0, F64),), (nxt,), start=0, stop=3, step=1, **kw)


def _step():
    return make("cast", (IDX,), "f64")


def test_the_two_classes_are_the_closed_set():
    assert (ACCUMULATOR, RECURRENCE) == ("accumulator", "recurrence") and LOOP_CLASSES == (ACCUMULATOR, RECURRENCE)


def test_an_additive_carry_is_an_accumulator_with_its_addend():
    step = _step()
    found = classify_loop(_loop(make("add", (CARRY, step))))
    assert found.name == ACCUMULATOR and found.addends == (step,)
    assert classify_loop(_loop(make("add", (make("add", (step, CARRY)), Const(1.0, F64))))).name == ACCUMULATOR


def test_any_other_body_is_a_recurrence_and_the_record_is_frozen():
    found = classify_loop(_loop(make("mul", (CARRY, Const(2.0, F64)))))
    assert found == LoopClass(RECURRENCE, ()) and found.addends == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.name = ACCUMULATOR
    assert classify_loop(_loop(make("add", (make("mul", (CARRY, CARRY)), _step())))).name == RECURRENCE   # carry twice


def test_an_add_chain_is_flattened_and_a_sub_is_one_term():
    a, b, c = Const(1.0, F64), Const(2.0, F64), Const(3.0, F64)
    assert addends(make("add", (make("add", (a, b)), c))) == [a, b, c]
    top = make("sub", (a, b))
    assert addends(top) == [top] and addends(a) == [a]


def test_the_boundary_is_the_highest_invariant_node_on_each_path():
    table = Leaf("table_read", "lookup", "w", TensorType((8,), "f64"))
    one = Const(1.0, F64)
    body = make("add", (At(table, IDX, F64), one))
    assert set(map(id, boundary((body,), frozenset({id(IDX)})))) == {id(table), id(one)}
    inv = make("neg", (one,))
    assert boundary((inv,), frozenset({id(IDX)})) == (inv,)                  # an invariant next is its own boundary value
    assert boundary((_step(),), frozenset({id(IDX)})) == ()                  # a body of the loop index alone has none


def test_the_boundary_of_a_built_loop_is_its_free_operands():
    table = Leaf("table_read", "lookup", "w", TensorType((8,), "f64"))
    loop = _loop(make("add", (CARRY, At(table, IDX, F64))))
    assert table in loop.free


def test_the_exit_keywords_rebuild_the_same_loop():
    loop = _loop(make("add", (CARRY, _step())))
    kw = exit_kwargs(loop)
    assert kw == {"exit_cond": None, "exit_values": (), "count": None, "count_tail": False, "dynamic": None}
    again = build_loop(loop.index, loop.carries, loop.operands[:1], loop.body_roots()[:1], start=0, stop=3, step=1, **kw)
    assert (again.start, again.stop, again.step, again.exit_cond) == (0, 3, 1, None)


def test_a_flat_loop_is_not_refused_as_nested():
    refuse_unsupported_nesting(_loop(make("add", (CARRY, _step()))))


def test_the_body_key_widths_are_bytes_per_structural_key():
    widths = body_key_widths(_loop(make("add", (CARRY, _step()))))
    assert len(widths) == 2 and set(widths) == {32}
