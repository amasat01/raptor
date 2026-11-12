"""The kinds builder (declare, make), the rule registry (register, instantiate, refusals) and the block predicates, in the Rust core, against
recorded verdicts and refusal texts: one case per distinct (kind of case, outcome). The recorded outcomes were compared equal to the Python
they replaced, over a larger grid, when the port landed."""
import json
from pathlib import Path

import pytest
import _cases as C

DATA = Path(__file__).parent / "data"


def _load(name):
    return json.loads((DATA / f"{name}_cases.json").read_text())


KINDS, PRIM, BLOCK = _load("kinds"), _load("primitive"), _load("block")


def test_declare_cases():
    assert len(KINDS["declare"]) > 30
    bad = [(c["name"], C.run_declare(c["data"], c["fp"]), c["want"]) for c in KINDS["declare"] if C.run_declare(c["data"], c["fp"]) != c["want"]]
    assert not bad, bad[:3]


def test_make_cases():
    assert len(KINDS["make"]) > 30
    bad = [(c["kind"], got, c["want"]) for c in KINDS["make"] if (got := C.run_make(c["kind"], c["ttype"], c["ops"], c["fields"])) != c["want"]]
    assert not bad, bad[:3]


def test_rule_registry_cases():
    assert len(PRIM) > 8
    bad = [(c["name"], got, c["want"]) for c in PRIM if (got := C.run_primitive(c)) != c["want"]]
    assert not bad, bad[:3]


def test_block_predicates():
    assert len(BLOCK) >= 2
    bad = [(c["payload"], got, c["want"]) for c in BLOCK if (got := C.run_block(c["payload"], c["policy"])) != c["want"]]
    assert not bad, bad[:3]


def test_reduce_rule_refuses_all_but_sum():
    from raptor.ir.nodes import Leaf, ReduceSamples
    from raptor.ir.types import TensorType
    from raptor.rules.core import rule_for
    x = Leaf("uniform", "uniform", "x", TensorType((), "f64"))
    rule = rule_for("reduce", 0)
    assert rule.vjp(ReduceSamples("sum", x), x)[0][0] is x
    assert rule.jvp(ReduceSamples("sum", x), lambda c: None) is None
    with pytest.raises(Exception, match="no derivative for the max over samples"):
        rule.vjp(ReduceSamples("max", x), x)


def test_expand_stores_and_modes():
    from raptor import _core
    from raptor.rules.core import RULES
    assert _core.expand_stores(iter(())) == ()
    modes = dict(_core.rule_modes())
    assert modes["at"] == modes["dispatch"] == "walker" and modes["lt"] == "zero"
    assert RULES[("at", None)].mode == "walker" and RULES[("floor", None)].mode == "zero"
