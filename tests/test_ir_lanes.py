# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Lane facts (raptor.ir.lanes) on hand-built kernels: the lane space from the divisors, the per-axis index forms of every read and
sink, the key facts of a keyed sink, and the predicate a kernel varies along an axis."""
import dataclasses
from fractions import Fraction
from types import SimpleNamespace

import pytest

from raptor.ir import lanes
from raptor.ir.lanes import AFFINE, INVARIANT, Eval, LaneFacts, axes_from, axis_forms, divisors, key_facts, nodes_of, varies_along
from raptor.ir.nodes import AccumWrite, Assign, At, Const, Leaf, SampleIndex
from raptor.ir.ops import make
from raptor.ir.types import TensorType

F64, I32 = TensorType((), "f64"), TensorType((), "i32")
S = SampleIndex()
TABLE = Leaf("table_read", "lookup", "t", TensorType((64,), "f64"))


def _c(value):
    return Const(value, I32)


ROW = make("div", (S, _c(6)))                                         # six lanes per row
READ_ROW = At(TABLE, ROW, F64)                                        # moves with the row only
READ_LANE = At(TABLE, make("add", (make("mul", (S, _c(2))), _c(1))), F64)   # 2 * lane + 1
KERNEL = SimpleNamespace(sinks=(Assign("y", make("add", (READ_ROW, READ_LANE)), F64),))


def test_nodes_of_reaches_every_node_under_the_sinks_once():
    found = nodes_of(KERNEL)
    assert len(found) == len({id(n) for n in found}) == 12
    assert READ_ROW in found and S in found and TABLE in found
    assert nodes_of(SimpleNamespace(order=(S,))) == [S] and nodes_of((S, S)) == [S]


def test_the_divisors_are_the_distinct_lane_quotients():
    assert divisors(nodes_of(KERNEL)) == [(6, None)]
    both = make("add", (make("div", (S, _c(12))), make("div", (S, _c(3)))))
    assert divisors(nodes_of((both,))) == [(12, None), (3, None)]
    assert divisors([make("div", (S, Const(2.0, F64)))]) == []                 # a real quotient is not a lane axis


def test_the_axes_follow_a_divisor_chain_and_a_non_chain_is_one_flat_axis():
    axes, notes = axes_from([(6, None)])
    assert [(a.label, a.extent, a.stride) for a in axes] == [(None, None, 6), (None, 6, 1)] and notes == ()
    axes, notes = axes_from([(12, "lane_tile_stride"), (3, None)])
    assert [(a.label, a.extent, a.stride) for a in axes] == [("tile", None, 12), (None, 4, 3), (None, 3, 1)]
    flat, notes = axes_from([(6, None), (4, None)])
    assert [(a.extent, a.stride) for a in flat] == [(None, 1)] and "not a chain" in notes[0]
    assert [(a.extent, a.stride) for a in axes_from([])[0]] == [(None, 1)]


def test_the_facts_hold_one_form_per_axis_for_every_read_and_sink():
    facts = lanes.facts(KERNEL)
    assert isinstance(facts, LaneFacts) and facts.group == 1 and facts.notes == ()
    assert [a.stride for a in facts.axes] == [6, 1]
    row, lane = facts.reads
    assert [(f.cls, f.coefficient) for f in row.forms] == [(AFFINE, 1), (INVARIANT, None)] and not row.gather
    assert [(f.cls, f.coefficient) for f in lane.forms] == [(AFFINE, 12), (AFFINE, 2)] and lane.offset == 1
    (sink,) = facts.sinks
    assert (sink.name, sink.keyed, sink.kind) == ("y", False, "assign")
    assert [f.coefficient for f in sink.forms] == [6, 1]                      # an own-column write moves with every axis
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.group = 2
    assert facts.axis_index(1) == 1


def test_an_evaluated_index_gives_its_per_axis_forms_exactly():
    facts = lanes.facts(KERNEL)
    value = Eval(facts.axes).value(ROW)
    forms = axis_forms(value, len(facts.axes))
    assert [(f.cls, f.coefficient) for f in forms] == [(AFFINE, Fraction(1)), (INVARIANT, None)]


def test_a_constant_key_is_an_invariant_axis_key_with_an_exclusive_bound():
    facts = lanes.facts(SimpleNamespace(sinks=(AccumWrite("e", Const(1.0, F64), _c(3), F64),)))
    sink = facts.sinks[0]
    assert (sink.keyed, sink.key_origin, sink.key_extent) == (True, "invariant_axis", 4)
    assert key_facts(Eval(facts.axes), _c(3), facts.axes) == ("invariant_axis", 4, None, ())


def test_a_kernel_varies_along_an_axis_where_a_node_reads_it():
    along_rows, along_lane = (varies_along(KERNEL, i) for i in (0, 1))
    assert along_rows(READ_ROW) and along_lane(READ_LANE)
    assert not along_lane(READ_ROW)                                           # the row read is the same for the six lanes of a row
    assert not along_rows(Const(1.0, F64))


def test_the_lane_records_are_frozen_and_come_out_of_the_facts():
    facts = lanes.facts(KERNEL)
    axis, read, sink = facts.axes[0], facts.reads[1], facts.sinks[0]
    form = read.forms[0]
    assert isinstance(axis, lanes.LaneAxis) and (axis.label, axis.extent, axis.stride) == (None, None, 6)
    assert isinstance(read, lanes.ReadFact) and (read.plane, read.role, read.gather, read.lane_offset) == ("t", "lookup", False, None)
    assert isinstance(sink, lanes.SinkFact) and (sink.keyed, sink.key_origin, sink.contract) == (False, None, "auto")
    assert isinstance(form, lanes.AxisForm) and (form.cls, form.coefficient) == (AFFINE, 12) and form == lanes.AxisForm(AFFINE, Fraction(12))
    assert sink.invariant_axes(facts) == ()
    for record, field in ((axis, "stride"), (read, "plane"), (sink, "name"), (form, "cls")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, field, None)
