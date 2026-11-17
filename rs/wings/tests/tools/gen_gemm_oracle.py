"""Records the matrix products the old route made of a contraction: for each kernel the IR dump, its sinks, its staged planes and,
for the sample counts below, what ``eagle.gemm._route._Route`` planned (the forward product and the adjoint's two).

    python gen_gemm_oracle.py <output json>

It needs the hawk tree (``hawk.passes.contraction``), eagle's python package and the library's Python binding (``_osprey``).
A kernel the route does not take (``attach``'s own gates: a two-operand dense lowerable contraction with one output and a lane
split whose major operand is found) is recorded with ``plans`` null.
"""

from __future__ import annotations

import json
import sys

import hawk
import hawk.math as m
from eagle.gemm._route import _free_size, _Route
from eagle.gemm.plan import PlanUnavailable, _check_lowerable
from hawk import Mutable, Scalar, Table
from hawk.artifact.bridge import dump
from hawk.passes import contraction as hc
from raptor.ir.nodes import Op, SampleIndex

R, U, S = 8, 6, 5


@hawk.kernel
def contract(w: Table["row":R, "neuron":U], f: Table["row":R, "sample":S], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * f.at(row=r, sample=sample)
    y = acc


@hawk.kernel
def contract_staged(w: Table["row":R, "neuron":U], feat: hawk.Staged[hawk.Vector[R], m.n_samples], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * feat.at(plane=r, sample=sample)
    y = acc


@hawk.kernel
def swapped(w: Table["row":R, "neuron":U], f: Table["row":R, "sample":S], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + f.at(row=r, sample=sample) * w.at(row=r, neuron=unit)
    y = acc


@hawk.kernel
def swapped_staged(w: Table["row":R, "neuron":U], feat: hawk.Staged[hawk.Vector[R], m.n_samples], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + feat.at(plane=r, sample=sample) * w.at(row=r, neuron=unit)
    y = acc


@hawk.kernel
def w_minor(w: Table["neuron":U, "row":R], f: Table["row":R, "sample":S], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(neuron=unit, row=r) * f.at(row=r, sample=sample)
    y = acc


@hawk.kernel
def f_minor(w: Table["row":R, "neuron":U], f: Table["sample":S, "row":R], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * f.at(sample=sample, row=r)
    y = acc


@hawk.kernel
def both_minor(w: Table["neuron":U, "row":R], f: Table["sample":S, "row":R], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(neuron=unit, row=r) * f.at(sample=sample, row=r)
    y = acc


@hawk.kernel
def unit_major(w: Table["row":R, "neuron":U], f: Table["row":R, "sample":S], y: Mutable[Scalar]):
    unit, sample = m.split_index(S)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * f.at(row=r, sample=sample)
    y = acc


@hawk.kernel
def two_outputs(w: Table["row":R, "neuron":U], f: Table["row":R, "sample":S], y: Mutable[Scalar], z: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * f.at(row=r, sample=sample)
    y = acc
    z = acc


CASES = (contract, contract_staged, swapped, swapped_staged, w_minor, f_minor, both_minor, unit_major, two_outputs)


def major_operand(kernel, record):
    """``eagle.gemm._route._major_operand`` over ``hawk.passes.contraction``: the operand the lane split's major half reads."""
    per = {}
    for sink in hc._sinks_of(kernel):
        _loop, _slot, terms = hc._fold(sink)
        for term in terms:
            for read in hc._reads_of(term):
                per.setdefault(hc._read_name(read, {}), []).append(hc._read_index(read))

    def split(name):
        ops = {node.kind for ix in per.get(name, ()) for node in hc._walk(ix)
               if isinstance(node, Op) and any(isinstance(o, SampleIndex) for o in node.operands)}
        return hc._fanin(per.get(name, ())), ops

    a, b = record.operands
    for major, minor in ((a, b), (b, a)):
        (m_major, ops), (m_minor, ops_minor) = split(major.name), split(minor.name)
        if ops == {"div"} and ops_minor > {"div"} and m_major == m_minor == _free_size(minor):
            return major.name
    return None


def step(s):
    """The fields of one planned matrix product."""
    s = s.matmul
    return {"a": s.a, "b": s.b, "out": s.out, "transpose_a": s.transpose_a, "transpose_b": s.transpose_b,
            "transpose_out": s.transpose_out}


def planned(record, major, dtype, n):
    """The route's plans for ``n`` lanes: the forward step and the adjoint's two, with the stored shapes of every plane."""
    route = _Route(record, major, dtype)
    forward, adjoint = route.plans(n)
    shapes = {spec.name: list(spec.shape) for p in (forward, adjoint) for spec in tuple(p.inputs) + tuple(p.outputs)}
    return {"n": n, "lane_flat": route.lane_flat, "shapes": shapes,
            "forward": [step(s) for s in forward.steps], "adjoint": [step(s) for s in adjoint.steps],
            "free_sizes": {op.name: _free_size(op) for op in record.operands}}


def main(path):
    """Writes the oracle file."""
    cases = {}
    for kernel in CASES:
        text, _dialects, position = dump(kernel.sinks)
        staged = sorted(n for n, p in kernel.planes.items() if getattr(p, "staged", False))
        record = hc.recognize(kernel)
        plans = None
        if record is not None and record.operand_class == "gemm" and len(record.outputs) == 1:
            try:
                _check_lowerable(record, "forward")
                major = major_operand(kernel, record)
            except PlanUnavailable:
                major = None
            if major is not None:
                counts = [S * U, 2 * S * U] if staged else [S * U]
                plans = [planned(record, major, "float64", n) for n in counts]
        cases[kernel.name] = {"dump": text, "sinks": [position[id(s)] for s in kernel.sinks], "staged": staged,
                              "operands": [o.name for o in record.operands] if record else None, "plans": plans}
    with open(path, "w") as fh:
        json.dump({"cases": cases}, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
