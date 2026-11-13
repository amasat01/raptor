# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tensor-typed IR nodes: leaf / op / sink / select.

ONE typed node family, not a class hierarchy per value shape: every node
carries a :class:`~hawk.types.TensorType` ``(shape, dtype, tag)`` as FIELDS.
Nodes are identity objects — structural equality is a KEY the canonical
walk computes (:mod:`hawk.ir.walk`), so a DAG may share a subexpression
freely. Leaves and sinks BIND a slot, keyed on ``(role, name)`` and merged
totally by the walk (:attr:`Node.binding`). Access is structural:
``own(i)`` (default), ``at(expr)`` (:class:`At`), ``scatter(expr)`` (a
wide/accum sink with an index operand).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Union

from .types import Slot, TensorType

#: What a literal of a node is: plain data.
Plain = Union[int, float, bool, str, tuple, None]

from .. import _core
from .._facts import facts as _facts
from . import _gen
from ..errors import RaptorError

_TABLE = _gen.parse(_core.manifest())
globals().update(_TABLE["const"])
globals().update(_TABLE["str"])


def refuse_factor_arms(what: str, ttype: TensorType) -> None:
    """Refuse a ``select`` or ``dispatch`` whose arms are factorisations (the check lives in the core)."""
    _core.run_check("refuse_factor_arms", what, ttype)


def plain_out(x: Any) -> Any:
    """A literal as JSON-safe data: scalars as they are, tuples tagged; anything else refused."""
    if x is None or type(x) in (int, float, str, bool):
        return x
    if type(x) is tuple:
        return {"tuple": [plain_out(i) for i in x]}
    raise RaptorError(f"literal of type {type(x).__name__} not covered")


def plain_in(x: Any) -> Any:
    return tuple(plain_in(i) for i in x["tuple"]) if isinstance(x, dict) else x


_CLASSES = _gen.classes(_TABLE, globals())
globals().update(_CLASSES)
_core.setup(TensorType, Node, Leaf, _CLASSES)
_core.set_loop(Loop)
_core.set_loop_carry(LoopCarry)

#: The closed list of feathers (named properties of a kind or an operation) with the meaning of each: ``KIND_FEATHERS`` of the core table.
FEATHERS = MappingProxyType(dict(line.split(": ", 1) for line in KIND_FEATHERS))
#: The feathers an operation of the generic op kind states, by operation name (the core table's ``op_feathers``; a name list per op).
OP_FEATHERS = MappingProxyType({op: frozenset(names) for op, names in _TABLE["man"].get("op_feathers", {}).items()})
Node.feathers = frozenset()
for _row in _TABLE["man"]["kinds"]:
    if "class" in _row and "feathers" in _row:
        _CLASSES[_row["class"]].feathers = frozenset(_row["feathers"])


def has_feather(node: Any, name: str) -> bool:
    """Whether the kind of ``node`` (or, for the generic op kind, its operation) states the feather ``name``; a feather is no part of identity."""
    if name not in FEATHERS:
        raise RaptorError(f"unknown feather {name!r}")
    return name in node.feathers or (isinstance(node, Op) and name in OP_FEATHERS.get(node.kind, ()))

#: the width of a structural body key (hex digits of the hash kept), fixed in the core
BODY_KEY_HEX = _facts("ir")["BODY_KEY_HEX"]


def body_key_widths(loop: "Loop") -> tuple[int, ...]:
    """The BYTE width of every structural key of ``loop``'s body, in body order (the diagnostic that sees an O(2**depth) key)."""
    return tuple(_core.body_key_widths(loop))


#: The language a raw block is written in unless it says otherwise; a writer
#: refuses a block in a language it cannot write.
RAW_LANGUAGE = "aether"

#: The leaf ROLES an ``at(expr)`` read may target: ``lookup`` (table plane)
#: and ``wide_in`` (pinned to the SAME 32-byte ``ScalarHandle``, same access),
#: and the per-sample input planes, read at another lane through ``u.at(i)``.
#: ``GATHER_ROLES`` are the ones read ONLY through ``at()``.
GATHER_ROLES = ("lookup", "wide_in")


def extent_of(node: Any) -> str:
    """The extent of ``node``: ``all_samples`` for a reduction over samples,
    ``shared`` when nothing under it depends on the sample (constants, uniforms,
    the sample count), else ``per_sample``."""
    if has_feather(node, "reduces_samples"):
        return "all_samples"
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, SampleIndex) or (
                isinstance(n, Leaf) and n.role not in ("uniform", "nsamples", "lookup",
                                                       "wide_in")):
            return "per_sample"
        if isinstance(n, At) and not is_param_cell(n):
            return "per_sample"
        stack.extend(n.operands)
    return "shared"


def peel_clamp(node: Node) -> Node:
    """``node`` without the ``clamp_index`` a table read holds its index in, else ``node`` itself."""
    return node.operands[0] if isinstance(node, Op) and node.kind == "clamp_index" else node


#: The label of the index a PARAMETER CELL carries: a ``uniform`` promoted to
#: a one-cell ``lookup`` plane (:func:`hawk.diff.param_planes`) is read as
#: ``At(plane, RoleConst(0, PARAM_CELL))``. The label tells the access
#: classifier it is one broadcast value, not a cross-sample read.


def is_param_cell(node: Any) -> bool:
    """Whether ``node`` is the read of a parameter cell."""
    return (isinstance(node, At) and isinstance(node.index, RoleConst)
            and node.index.label == PARAM_CELL)


def param_cell_names(nodes: Any) -> tuple:
    """The names of the parameter cells read in ``nodes``, in first-seen order."""
    out: dict = {}
    for node in nodes:
        if is_param_cell(node):
            out.setdefault(node.plane.name, None)
    return tuple(out)

def compensates(contract: str) -> bool:
    """Whether an accumulate under ``contract`` commits through a companion plane."""
    return contract in ("compensated", KEYED_AUTO)

#: The suffix of a compensated accumulate plane's companion.
COMPANION_SUFFIX = "_c"

#: The prefix of a vjp's adjoint plane (``bar_<plane>``).
ADJOINT_PREFIX = "bar"



def sample_sums(sinks: Sequence[Node]) -> tuple:
    """The :class:`SampleSum` sinks of ``sinks``, in order: column ``k`` of
    the kernel's row of terms is the ``k``-th."""
    return tuple(s for s in sinks if isinstance(s, SampleSum))


def element_count(ttype: TensorType) -> int:
    """How many scalars one value of ``ttype`` holds."""
    out = 1
    for extent in ttype.shape:
        out *= int(extent)
    return out



def through(node: Node) -> Node:
    """The node a reference to ``node`` is a reference TO: a Primitive
    stands for its forward and is never rendered itself."""
    while isinstance(node, Primitive):
        node = node.forward
    return node


def same_index(a: Node, b: Node, seen: dict | None = None) -> bool:
    """Whether two index expressions are the same value: one node, or equal
    kind, literal and operands (a key written twice is traced twice). ``seen``
    keeps a shared subexpression from being compared once per path."""
    if a is b:
        return True
    seen = {} if seen is None else seen
    pair = (id(a), id(b))
    if pair in seen:
        return seen[pair]
    out = (type(a) is type(b) and a.ttype == b.ttype
           and len(a.operands) == len(b.operands))
    if out and isinstance(a, Const):
        out = a.literal == b.literal
    elif out:
        out = (isinstance(a, (Op, At)) and getattr(a, "kind", None) == getattr(b, "kind", None)
               and getattr(a, "literal", None) == getattr(b, "literal", None)
               and all(same_index(through(x), through(y), seen)
                       for x, y in zip(a.operands, b.operands)))
    seen[pair] = out
    return out


def replace(node: Node, **changes: Any) -> Node:
    """A copy of ``node`` with some slots replaced: a node is not changed after it is built, a pass that needs another
    operand tuple (or any other field) makes a new node with this and hands it on."""
    new = object.__new__(type(node))
    for cls in type(node).__mro__:
        for slot in cls.__dict__.get("__slots__", ()):
            try:
                object.__setattr__(new, slot, object.__getattribute__(node, slot))
            except AttributeError:
                pass
    for key, value in changes.items():
        object.__setattr__(new, key, tuple(value) if key == "operands" else value)
    return new


# ---- IR-building methods that stay Python ----

def _At_raw_index(self) -> Node:
    """The index before its clamp (:func:`peel_clamp`): what a pass reads for the index's SHAPE. A pass that
    rewrites the read keeps :attr:`index`, so the read stays clamped."""
    return peel_clamp(self.operands[1])
At.raw_index = property(_At_raw_index)

def _ReduceSamples_recount(self, count: Node) -> ReduceSamples:
    """The same keyed reduction over ``count`` kept coordinates."""
    return self.rebuilt(self.operands, {**self.edges(), "count": count})
ReduceSamples.recount = _ReduceSamples_recount
