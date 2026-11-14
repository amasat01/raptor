# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Parameter cells (raptor.ir.cells), the fold thresholds (raptor.ir.fold) and the compound quantity (raptor.ir.compound)."""
import dataclasses

import pytest

from raptor.errors import RaptorError
from raptor.ir import cells
from raptor.ir.compound import Quantity
from raptor.ir.fold import exceeds, rolled
from raptor.ir.nodes import Assign, At, Const, Leaf, Op, RoleConst
from raptor.ir.types import TensorType, Wire

F64 = TensorType((), "f64")
V3 = TensorType((3,), "f64")


def test_the_planes_of_a_cell_are_its_name_or_its_numbered_columns():
    assert cells.planes("w", 1) == ("w",)
    assert cells.planes("w", 3) == ("w_0", "w_1", "w_2")


def test_a_parameter_read_is_cell_zero_or_the_cell_the_sample_names():
    own = cells.param_cell("w", V3)
    assert isinstance(own, At) and own.ttype == V3
    assert (own.operands[0].kind, own.operands[0].role, own.operands[0].name) == ("table_read", "lookup", "w")
    assert isinstance(own.operands[1], RoleConst) and own.operands[1].literal == 0
    per = cells.param_cell("w", V3, per_cell=True)
    idx = per.operands[1]
    assert (idx.name, idx.role, idx.ttype) == (cells.CELL_PLANE, "per_sample", TensorType((), "i32"))
    assert cells.CELL_PLANE == "param_cell"


def test_with_value_rebuilds_a_commit_and_refuses_a_sink_that_is_not_one():
    old = Assign("acc", Const(1.0, F64), F64)
    new = cells.with_value(old, Const(2.0, F64))
    assert new is not old and new.operands[0].literal == 2.0 and new.name == "acc"
    assert old.operands[0].literal == 1.0
    with pytest.raises(RaptorError, match="is none of them"):
        cells.with_value(Const(1.0, F64), Const(2.0, F64))


def test_swap_replaces_the_mapped_node_by_identity_and_leaves_the_rest():
    a, b = Const(1.0, F64), Const(5.0, F64)
    s1, s2 = Assign("p", Op("neg", (a,), F64), F64), Assign("q", b, F64)
    out = cells.swap((s1, s2), {id(a): Const(9.0, F64)})
    assert out[0].operands[0].operands[0].literal == 9.0
    assert out[1].operands[0] is b
    same = cells.swap((s1, s2), {})
    assert [x.name for x in same] == ["p", "q"] and same[1].operands[0].literal == 5.0


def test_a_fold_threshold_is_compared_strictly_and_a_missing_one_is_refused():
    with pytest.raises(RaptorError, match="form.rolled_fold_min"):
        exceeds(10, None)
    assert exceeds(10, 4) and not exceeds(4, 4)


def test_a_dot_is_rolled_by_form_and_a_threshold_is_read_only_for_auto():
    v = Leaf("vocab_read", "per_sample", "v", TensorType((8,), "f64"))
    node = Op("dot", (v, v), TensorType((), "f64"))
    assert rolled(node, None, "rolled") and not rolled(node, None, "static")
    with pytest.raises(RaptorError, match="fold threshold"):
        rolled(node, None, "auto")
    assert rolled(node, 4) and not rolled(node, 8)          # 8 > 8 is false
    assert not rolled(v, 4)                                  # a leaf is never rolled


def test_a_quantity_names_its_wires_and_reads_them_as_leaves():
    wires = (Wire("pos", "vec_in", (3,), "f64", None), Wire("mass", "per_sample", (), "f64", None))
    q = Quantity("body", "a", wires=wires)
    assert q.prefix == "body__a" and Quantity("body", wires=wires).prefix == "body"
    assert q.wire_name(wires[0]) == "body__a__pos"
    pos, mass = q.read()
    assert (pos.name, pos.ttype.shape, mass.name, mass.ttype.shape) == ("body__a__pos", (3,), "body__a__mass", ())
    assert pos.quantity is q and (pos.wire_index, mass.wire_index) == (0, 1)
    assert [s.name for s in q.expand()] == ["body__a__pos", "body__a__mass"]
    assert Quantity("one", wires=wires[:1]).read().name == "one__pos"
    with pytest.raises(RaptorError):
        Quantity("bad", wires=wires, pack="nope")
