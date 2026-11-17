"""Records the per-branch costs of dispatch groups as the Python model counts them.

For each kernel below it writes the IR dump text, the walk order as positions in that dump, and what
``raptor.wings.base.cost.dispatch_costs`` answers: the independent oracle of ``tests/cost.rs``.

    python gen_cost_oracle.py <output json>

It needs the hawk tree on the path (``hawk.artifact.bridge.dump`` writes the dump) and the library's Python binding.
"""

from __future__ import annotations

import json
import sys

import hawk
import hawk.math as M
from hawk.artifact.bridge import dump
from hawk.trace.value import TableRef, node_of
from raptor.ir import Assign, Leaf, canonical
from raptor.ir.types import TensorType
from raptor.wings.base.cost import dispatch_costs, fp64_ops
from raptor.wings.base.dispatch import dispatch

S = TensorType((), "f64")
F32 = TensorType((), "f32")
I32 = TensorType((), "i32")
V3 = TensorType((3,), "f64")
Q = TensorType((4,), "f64", "quaternion")
M33 = TensorType((3, 3), "f64")
BOOL = TensorType((), "bool")
FIELDS = ("arith32", "arith64", "special32", "special64", "loads", "gathers", "chain32", "chain64",
          "chainSpecial32", "chainSpecial64", "stores")


def leaf(name, ttype):
    """A traced value over a fresh plane read of type ``ttype``."""
    if ttype.dtype == "bool":
        return hawk.Value(Leaf("terminated", "terminated", name, ttype))
    role = {0: "per_sample", 1: "vec_in", 2: "mat_in"}[len(ttype.shape)]
    return hawk.Value(Leaf("vocab_read", role, name, ttype))


def selector():
    """The integer a dispatch group chooses by."""
    return Leaf("vocab_read", "per_sample", "k", I32)


def group(branches, ttype, sel=None):
    """A dispatch node over traced values ``branches``."""
    return dispatch(sel or selector(), [node_of(b) for b in branches], "switch", ttype)


def sink(value, name="y"):
    """A write of ``value`` (a node) to the plane ``name``."""
    return (Assign(name, value, value.ttype),)


def group_kernel(branches, ttype, **kw):
    """The sinks of a kernel that writes one dispatch group straight to a plane."""
    return sink(group(branches, ttype, **kw))


def cases():
    """``{name: sinks}``: the dispatch groups whose costs are recorded."""
    x, y, z, b = leaf("x", S), leaf("y", S), leaf("z", S), leaf("p", BOOL)
    a3, b3, q, q2 = leaf("a", V3), leaf("b", V3), leaf("q", Q), leaf("r", Q)
    m, n = leaf("m", M33), leaf("n", M33)
    f, g = leaf("f", F32), leaf("g", F32)
    table = TableRef("tab", S)
    where = leaf("w", TensorType((), "int32", "index"))
    out = {
        "scalar_mixed": group_kernel([x * y, M.exp(x), M.log(x * x + 1.0) * z], S),
        "scalar_library": group_kernel([M.pow(x, y), M.sqrt(x) + M.atan2(x, y), M.fmod(x, y), M.sin(x) * M.cos(y)], S),
        "scalar_select": group_kernel([M.select(b, x, y), M.select(b, M.exp(x), y * z)], S),
        "vector_totals": group_kernel([M.norm(a3), M.dot(a3, b3) * x, M.sum(a3), M.sum(M.cross(a3, b3))], S),
        "vector_values": group_kernel([M.cross(a3, b3), a3 * x, M.quat_rotate(q, a3), M.as_vec3(M.quat_mul(q, q2))], V3),
        "quaternion_free_and_priced": group_kernel([M.as_vec3(M.quat_conj(q)), M.as_vec3(M.quat_recip(q))], V3),
        "matrix_products": group_kernel([M.outer(a3, b3), m @ n, M.transpose(m)], M33),
        "matrix_vector": group_kernel([m @ a3, a3 + b3], V3),
        "single_precision": group_kernel([f * g, M.sin(f), f + M.exp(g) * g], F32),
        "shared_value": group_kernel([x * x + 1.0, x * x + 2.0], S),
        "selector_reused": group_kernel([x * leaf("k", I32), x], S),
        "gather_branch": group_kernel([table.at(where), x * 2.0], S),
        "selector_gathers": group_kernel([x * y, x], S, sel=node_of(TableRef("tabk", I32).at(where))),
        "value_not_stored_directly": sink(node_of(hawk.Value(group([x * y, M.exp(x)], S)) * 2.0)),
        "two_groups_one_sink": sink(node_of(hawk.Value(group([x * y, M.exp(x)], S)) + hawk.Value(group([x, M.log(x)], S)))),
        "two_sinks": sink(group([x * y, M.exp(x)], S)) + sink(node_of(x * y), "w2"),
    }
    out.update(loop_cases())
    out.update(heavy_cases(x, a3, b3, m))
    return out


def heavy_cases(x, a3, b3, m):
    """Groups over linear algebra, ray ops and random draws."""
    seed, counter = leaf("seed", I32), leaf("counter", I32)
    cols = leaf("B", TensorType((3, 2), "f64"))
    return {
        "linalg_solve": group_kernel([M.solve(m, b3)[0], b3 * x], V3),
        "linalg_lu_and_cholesky": group_kernel([M.lu_solve(M.lu(m)[0], b3), M.cho_solve(M.cholesky(m)[0], b3)], V3),
        "linalg_multiple_rhs": group_kernel([M.lu_solve(M.lu(m)[0], cols), cols], TensorType((3, 2), "f64")),
        "ray_ops": group_kernel([M.ray_tri(a3, b3, a3, b3, a3, x), M.ray_aabb(a3, b3, a3, b3), M.ray_sphere(a3, b3, a3, x)],
                                TensorType((4,), "f64")),
        "random_draws": group_kernel([M.random_uniform(seed, counter), M.random_normal(seed, counter) * x,
                                      M.random_poisson1(seed, counter)], S),
    }


def loop_cases():
    """The groups whose branch holds a loop, as hawk kernels."""

    @hawk.kernel
    def looped(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for _ in range(4):
            acc = acc * x + 1.0
        y = M.dispatch(k, [acc, x], strategy="switch")

    @hawk.kernel
    def two_chains(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        a = 0.0
        b = 0.0
        for _ in range(8):
            a = a * x + 1.0
            b = b * x + 2.0
        y = M.dispatch(k, [a + b, x], strategy="switch")

    @hawk.kernel
    def gathered(k: hawk.Index, table: hawk.Table[hawk.Scalar], where: hawk.Index, x: hawk.Scalar,
                 y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for i in range(8):
            acc = acc + table.at(where + i)
        y = M.dispatch(k, [acc, x], strategy="switch")

    @hawk.kernel
    def special_loop(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for _ in range(3):
            acc = M.sin(acc) + x
        y = M.dispatch(k, [acc, M.exp(x)], strategy="switch")

    @hawk.kernel
    def nested_loops(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for _ in range(2):
            for _ in range(3):
                acc = acc * x + 1.0
        y = M.dispatch(k, [acc, x], strategy="switch")

    @hawk.kernel
    def counted_after_loop(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for _ in range(5):
            acc = acc * acc
        y = M.dispatch(k, [M.sqrt(acc) * x, acc], strategy="switch") + 1.0

    @hawk.kernel
    def open_bound(k: hawk.Index, n: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for _ in range(n):
            acc = acc * x + 0.5
        y = M.dispatch(k, [acc, x], strategy="switch")

    @hawk.kernel
    def bounded_by_runtime_value(k: hawk.Index, n: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for _ in range(n, max=6):
            acc = acc * x + 0.5
        y = M.dispatch(k, [acc, x], strategy="switch")

    kernels = {"loop_open_bound": open_bound, "loop_runtime_bound_with_max": bounded_by_runtime_value,
               "loop_one_carry": looped, "loop_two_chains": two_chains, "loop_with_gather": gathered,
               "loop_with_library_call": special_loop, "loop_nested": nested_loops,
               "loop_then_work_after": counted_after_loop}
    return {name: kernel.sinks for name, kernel in kernels.items()}


def branch_row(cost):
    """One branch as the record: the six counts, the four chains and the stores (null when not counted)."""
    return {name: getattr(cost, name) for name in FIELDS}


def record(name, sinks):
    """The recorded case: dump text, walk order as dump positions, per-group answers."""
    text, dialects, position = dump(sinks)
    if dialects:
        raise SystemExit(f"{name}: the dump needs dialect declarations the test does not make")
    walk = canonical(sinks)
    groups = [{"selector_reads": [list(r) for r in c.selector_reads], "branches": [branch_row(b) for b in c.branches]}
              for c in dispatch_costs(walk).values()]
    return {"name": name, "dump": text, "order": [position[id(n)] for n in walk.order], "fp64_ops": fp64_ops(walk.order),
            "groups": groups}


def main():
    """Writes the recorded cases to the file named by the argument."""
    recorded = []
    for name, sinks in cases().items():
        try:
            recorded.append(record(name, sinks))
        except Exception as why:  # a kernel the front end or the dump cannot build is reported, not recorded
            print(f"{name}: not recorded: {type(why).__name__}: {why}")
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump({"cases": recorded}, fh, indent=1, sort_keys=True)
        fh.write("\n")
    branches = sum(len(g["branches"]) for c in recorded for g in c["groups"])
    print(f"{len(recorded)} kernels, {sum(len(c['groups']) for c in recorded)} groups, {branches} branches")


if __name__ == "__main__":
    main()
