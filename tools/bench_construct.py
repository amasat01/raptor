# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Construction cost of IR nodes: ns per construction of an Op, a Const and an At, the median of several rounds.

    python tools/bench_construct.py [--n 20000] [--rounds 9]

Run it from the repository root with the built module in place (``make build``), on an otherwise idle machine; compare two builds
by running it on each. Node construction runs the kind's checks and the structural identity, so this is the number to watch when a
check or a manifest row is added."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raptor.ir import nodes
from raptor.ir.types import TensorType


def _cases():
    f64 = TensorType((), "f64")
    i32 = TensorType((), "i32")
    leaf = nodes.Leaf("vocab_read", "per_sample", "x", f64)
    table = nodes.Leaf("table_read", "lookup", "t", TensorType((8,), "f64"))
    one = nodes.Const(1.0, f64)
    idx = nodes.Const(3, i32)
    return {
        "Const": lambda i: nodes.Const(float(i), f64),
        "Op": lambda i: nodes.Op("add", (leaf, one), f64),
        "At": lambda i: nodes.At(table, idx, f64),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20000, help="constructions per round")
    ap.add_argument("--rounds", type=int, default=9)
    a = ap.parse_args(argv)
    cases = _cases()
    runs = {k: [] for k in cases}
    for _ in range(a.rounds):               # interleaved: every round times every case
        for k, fn in cases.items():
            t0 = time.perf_counter_ns()
            for i in range(a.n):
                fn(i)
            runs[k].append((time.perf_counter_ns() - t0) / a.n)
    print(f"python {sys.version.split()[0]}  n={a.n}  rounds={a.rounds}")
    for k, v in runs.items():
        q = statistics.quantiles(v, n=4)
        print(f"{k:6s} median {statistics.median(v):8.0f} ns   iqr {q[2] - q[0]:6.0f} ns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
