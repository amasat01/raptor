# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The walk (raptor.ir.walk): the frozen product of canonical(), the dispatch listing, substitution and resinking."""
import dataclasses
from types import SimpleNamespace

import pytest

from raptor.ir import Assign, Dispatch, canonical
from raptor.ir.loops import build_loop
from raptor.ir.nodes import Const, Leaf, LoopCarry, LoopIndex
from raptor.ir.ops import make
from raptor.ir.types import TensorType
from raptor.ir.walk import (Walk, dispatch_groups, loop_body_groups, resink, static_selectors, substitute, written_names)

F64, I32 = TensorType((), "f64"), TensorType((), "i32")
SEL = Leaf("vocab_read", "per_sample", "k", I32)


def _dispatched(name="y", selector=SEL):
    return Assign(name, Dispatch(selector, (Const(1.0, F64), Const(2.0, F64)), F64), F64)


def test_a_walk_is_the_frozen_record_of_its_sinks():
    walk = canonical((_dispatched(),))
    assert isinstance(walk, Walk)
    assert walk.arg_spec == (("mutable", "y"), ("per_sample", "k"))
    assert all(walk.arg_spec[i] == key for key, i in walk.slot_of.items())
    assert walk.access == ("sample_local", None) and "dispatch" in walk.kinds
    assert walk.digest == canonical((_dispatched(),)).digest != canonical((_dispatched("z"),)).digest
    with pytest.raises(dataclasses.FrozenInstanceError):
        walk.digest = ""


def test_a_top_level_dispatch_is_listed_by_name_and_is_not_inside_a_loop():
    walk = canonical((_dispatched(),))
    groups = dispatch_groups(walk)
    assert list(groups) == ["dispatch0"] and isinstance(groups["dispatch0"], Dispatch)
    assert loop_body_groups(walk) == frozenset()
    assert walk.dispatches[0].selector_reads == (("per_sample", "k"),) and walk.dispatches[0].K == 2


def test_a_dispatch_in_a_loop_body_is_a_loop_body_group():
    idx, carry = LoopIndex(0, "r"), LoopCarry(0, 0, "c", F64)
    pick = Dispatch(make("cast", (idx,), "i32"), (carry, Const(2.0, F64)), F64)      # reads the carry: not hoisted
    loop = build_loop(idx, (carry,), (Const(0.0, F64),), (make("add", (carry, pick)),), start=0, stop=2, step=1)
    walk = canonical((Assign("y", loop, F64),))
    assert loop_body_groups(walk) == frozenset(info.name for info in walk.dispatches)
    assert len(loop_body_groups(walk)) == 1 and list(dispatch_groups(walk)) == list(loop_body_groups(walk))


def test_written_names_are_the_planes_the_sinks_commit():
    assert written_names(canonical((_dispatched("y"), _dispatched("z"))).order) == {"y", "z"}
    assert written_names(()) == frozenset()


def test_a_group_is_static_unless_a_kernel_of_the_step_writes_its_selector_plane():
    reader = SimpleNamespace(name="reader", walk=canonical((_dispatched("y"),)))
    writer = SimpleNamespace(name="writer", walk=canonical((Assign("k", Const(0, I32), I32),)))
    assert static_selectors([reader]) == {"reader": {"dispatch0": True}}
    assert static_selectors([reader, writer]) == {"reader": {"dispatch0": False}, "writer": {}}


def test_substitute_replaces_by_identity_and_reuses_what_did_not_change():
    a, b = Const(1.0, F64), Const(2.0, F64)
    kept = make("neg", (b,))
    root = make("add", (make("neg", (a,)), kept))
    (out,) = substitute((root,), {id(a): Const(9.0, F64)})
    assert out.operands[0].operands[0].literal == 9.0 and out.operands[1] is kept
    assert substitute((root,), {}) == (root,) and root.operands[0].operands[0] is a      # the original is untouched


def test_resink_rebuilds_a_sink_over_the_rewritten_operand_and_returns_an_unchanged_one():
    a, b = Const(1.0, F64), Const(2.0, F64)
    sink = Assign("y", a, F64)
    assert resink(sink, {id(a): a}) is sink
    new = resink(sink, {id(a): b})
    assert new.name == "y" and new.operands[0] is b and sink.operands[0] is a


def test_a_reached_quantity_is_a_frozen_span_of_its_wires_and_slots():
    from raptor.ir.compound import Quantity, QuantitySpan
    from raptor.ir.types import Wire
    q = Quantity("body", "a", wires=(Wire("pos", "vec_in", (3,), "f64", None), Wire("mass", "per_sample", (), "f64", None)))
    pos, _mass = q.read()
    walk = canonical((Assign("y", make("sum", (pos,)), F64),))
    span = walk.quantities["body__a"]
    assert isinstance(span, QuantitySpan) and span.prefix == "body__a" and span.wires == ("body__a__pos", "body__a__mass")
    assert [walk.arg_spec[i][1] for i in span.slots] == list(span.wires)          # the unread wire is bound too
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.prefix = "x"
