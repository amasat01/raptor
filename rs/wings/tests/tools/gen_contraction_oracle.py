"""Records what the old recogniser (``hawk.passes.contraction.recognize``) answers over a set of kernels.

For each kernel it writes the IR dump text, the sink positions in the dump, the names of the staged planes the kernel
declares and the recogniser's record (or null): the independent oracle of ``tests/contraction.rs``.

    python gen_contraction_oracle.py <output json>

It needs the hawk tree and its tests on the path (the kernels are the ones of ``test_ir_contraction.py`` and
``test_contraction_multi.py``) and the library's Python binding (``_osprey``, for ``hawk.artifact.bridge.dump``).
"""

from __future__ import annotations

import json
import sys

import test_contraction_multi as multi
import test_ir_contraction as single
from hawk.artifact.bridge import dump
from hawk.passes.contraction import recognize


def operand(op):
    """The fields of one operand record the Rust port decides on."""
    return {
        "name": op.name,
        "role": op.role,
        "reduce_stride": op.reduce_stride,
        "reduce_dim": op.reduce_dim,
        "dense": op.dense,
        "refusals": list(op.refusals),
    }


def record(rec):
    """The recognised record as plain data, or null."""
    if rec is None:
        return None
    return {
        "carry": rec.carry,
        "reduce_var": rec.reduce_var,
        "reduce_extent": rec.reduce_extent,
        "reduce_start": rec.reduce_start,
        "reduce_step": rec.reduce_step,
        "operand_class": rec.operand_class,
        "term_form": rec.term_form,
        "outputs": [list(o) for o in rec.outputs],
        "dense": rec.dense,
        "refusals": list(rec.refusals),
        "staged_fanin": [list(f) for f in rec.staged_fanin],
        "operands": [operand(o) for o in rec.operands],
    }


def subjects():
    """Every kernel recorded, by case name."""
    out = {name: getattr(single, name) for name in (
        "spline_contract", "moe_contract", "row_sum_exp", "gathered", "hand_folded", "diagonal", "one_term",
        "three_planes", "offset_sum", "max_fold", "spline_contract_n", "vec_committed", "vec_committed_twice")}
    out["dot_staged_7"] = single._dot_staged(7)
    out["dot_staged_64"] = single._dot_staged(64)
    for name in ("matvec_input", "matvec_computed", "masked_sum", "two_sinks", "strided_by_literal", "two_loops",
                 "fold_and_plain", "three_operands"):
        out[name] = getattr(multi, name)
    return out


def main(path):
    """Writes the oracle file."""
    cases = {}
    for name, kernel in sorted(subjects().items()):
        text, _dialects, position = dump(kernel.sinks)
        staged = sorted(n for n, p in kernel.planes.items() if getattr(p, "staged", False))
        cases[name] = {
            "dump": text,
            "sinks": [position[id(s)] if id(s) in position else None for s in kernel.sinks],
            "staged": staged,
            "record": record(recognize(kernel)),
        }
    with open(path, "w") as fh:
        json.dump({"cases": cases}, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
