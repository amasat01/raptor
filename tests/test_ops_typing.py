"""ops.result_type / make (Rust, rs/binding/src/typing.rs) against what the Python rules gave: one recorded case per distinct (kind, outcome)
over a grid of 127k operand-type combinations (the full grid was compared equal when the port landed)."""
import json
from pathlib import Path
import pytest
from raptor.ir import ops
from raptor.ir.nodes import Leaf
from raptor.errors import RaptorError
from raptor.ir.types import TensorType

CASES = json.loads((Path(__file__).parent / "data" / "ops_typing_cases.json").read_text())


def _node(t):
    return Leaf("uniform", "uniform", "x", TensorType(tuple(t[0]), t[1], t[2]))


def test_recorded_cases():
    assert len(CASES) > 1500
    bad = []
    for c in CASES:
        lit = tuple(c["lit"]["t"]) if isinstance(c["lit"], dict) else c["lit"]
        try:
            got = "ok " + repr(ops.result_type(c["kind"], [_node(t) for t in c["ts"]], lit))
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        if got != c["want"]:
            bad.append((c["kind"], c["ts"], got, c["want"]))
    assert not bad, bad[:3]


def test_make_refuses_a_factor_used_as_a_matrix_and_promote():
    a = _node([[3, 3], "f64", "lu_factor"])
    with pytest.raises(RaptorError, match="an LU factor is not a matrix"):
        ops.make("mm", (_node([[3, 3], "f64", None]), a))
    assert ops.promote("i32", "f64") == "f64"
    with pytest.raises(RaptorError, match="does not mix"):
        ops.promote("int32", "f64")
