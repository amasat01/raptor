"""Feathers: named properties of a kind or an operation. The list is closed and owned by the core table; a row states its own; a reader asks one question."""
import pytest
from raptor.errors import RaptorError
from raptor.ir import kinds as K, make
from raptor.ir import nodes
from raptor.ir.nodes import FEATHERS, OP_FEATHERS, Leaf, ReduceSamples, extent_of, has_feather
from raptor.ir.types import TensorType

F64 = TensorType((), "f64")
X = Leaf("vocab_read", "per_sample", "x", F64)


def row(name="acme.thing", **extra):
    return {"name": name, "dialect": "d", "ext": "acme", "family": "expr", "refines": None, "operands": {"min": 1, "max": 1}, "result": ["same_as", 0], **extra}


def test_the_closed_list_names_each_feather_with_one_line_of_meaning():
    assert "reduces_samples" in FEATHERS and all(FEATHERS.values())


def test_a_core_row_states_a_feather_and_the_answer_is_by_kind():
    assert has_feather(ReduceSamples("sum", X), "reduces_samples") and not has_feather(X, "reduces_samples")
    assert extent_of(ReduceSamples("sum", X)) == "all_samples" and extent_of(X) == "per_sample"


def test_an_unknown_feather_is_refused_by_name_when_asked_and_when_declared():
    with pytest.raises(RaptorError, match="unknown feather 'no_such'"):
        has_feather(X, "no_such")
    with pytest.raises(RaptorError, match="unknown feather 'no_such'"):
        K.declare(K.Registry(), row(feathers=["no_such"]))


def test_a_declared_kind_states_feathers_and_the_old_trait_rows_are_feathers_now():
    reg = K.Registry()
    d = K.declare(reg, row(feathers=["reduces_samples"]))
    plain = K.declare(K.Registry(), row())
    assert d.feathers == ("reduces_samples",) and plain.feathers == ()
    node = K.make_kind(reg, "acme.thing", F64, (X,))
    assert has_feather(node, "reduces_samples")


def test_an_operation_can_state_a_feather(monkeypatch):
    assert OP_FEATHERS == {}
    monkeypatch.setattr(nodes, "OP_FEATHERS", {"neg": frozenset({"reduces_samples"})})
    assert has_feather(make("neg", (X,)), "reduces_samples") and not has_feather(make("abs", (X,)), "reduces_samples")


@pytest.mark.parametrize("name", ["elementwise_sink", "reads_by_element", "fresh_read", "static_index"])
def test_the_former_traits_are_members_of_the_closed_list(name):
    assert FEATHERS[name]
    with pytest.raises(RaptorError, match="unknown feather"):
        K.declare(K.Registry(), row(feathers=[name + "_x"]))
    assert K.declare(K.Registry(), row(feathers=[name])).feathers == (name,)


def test_a_declared_sink_keeps_operands_beyond_the_value_and_the_index():
    reg = K.Registry()
    K.declare(reg, {"name": "acme.put", "dialect": "d", "ext": "acme", "family": "sink", "refines": "wide_write", "operands": {"min": 3, "max": 3},
                    "result": ["same_as", 0], "feathers": ["commits_value"]})
    key = Leaf("vocab_read", "per_sample", "k", TensorType((), "i32"))
    node = K.make_kind(reg, "acme.put", F64, (X, key, key), name="p", role="accum_out")
    assert len(node.operands) == 3 and node.operands[2] is key and node.rebuilt(node.operands, {}).operands[2] is key
    assert has_feather(node, "commits_value")
