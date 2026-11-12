# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Access-class inference over a :class:`~hawk.ir.walk.Walk` (…).

HAWK DECLARES an access class; eagle's ``eagle.exec.check_placement`` refuses
an illegal placement. The classifier reads the walk's access FORMS, never
node classes: an :class:`~hawk.ir.nodes.At` read of a ``lookup`` plane is
``cross_sample_read``, a wide/accum sink carrying a scatter index is
``cross_sample_write``, everything else is ``sample_local``. The MOST
RESTRICTIVE of the three wins — over-restriction costs only a placement,
under-restriction is a wrong answer under ``RankPartition``. ``mapreduce``
is DISJOINT (an un-folded partial is a wrong answer, not a placement cost).
No annotation surface exists except for injected raw text (``@raw_device``):
a traced body can never carry a hand-written class.

The same module answers a finer question: how ONE index expression moves
with the sample index (:func:`index_form`) — a contraction's reduce-axis
stride, a scatter whose lanes share a key — so the descent over index
arithmetic exists once.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

from .. import _core
from .._facts import facts as _facts
from .nodes import Loop
from .nodes import (
    peel_clamp,
    FUSED_STEPS_PLANE,
    RAW_KIND,
    AccumWrite,
    At,
    Const,
    Leaf,
    Node,
    Op,
    RoleConst,
    is_param_cell,
    SampleIndex,
    WideWrite,
)
from ..errors import RaptorError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .walk import Walk


#: Access classes, LEAST to MOST restrictive (``mapreduce`` is disjoint).
ACCESS_CLASSES = _facts("ir")["ACCESS_RANK"] + ("mapreduce",)

_RANK = {c: i for i, c in enumerate(_facts("ir")["ACCESS_RANK"])}


from . import _gen
from .. import _core

# pure-data records of the manifest (raptor/ir/core.json, arm `records`, home 'access')
globals().update(_gen.records(_gen.parse(_core.manifest()), __name__, 'access'))


def infer_over(order: Iterable[Node]) -> Access:
    """The classifier proper, over a canonical node ORDER.

    A :class:`~hawk.ir.nodes.Loop`'s body is NOT in that order (it's a
    scope the emitter renders into), so the classifier descends into it
    here: a table read at the loop index is ``cross_sample_read`` whether
    the ``for`` was unrolled or lowered, and missing that would let eagle
    place a kernel under a ``RankPartition`` that splits the plane it
    reads — a wrong answer, not a placement cost.

    One read is NOT a form of the body: a ``steps="auto"`` loop's trip
    count, the reserved ``fused_steps`` word the runner sets (one cell, the
    same for every sample). It is the loop's ``count`` operand, which no
    traced body can mint, so a kernel's class is its single step's."""
    return Access(_core.infer_access(list(order)), None)


def infer(walk: Walk) -> Access:
    """The entry point: the access class + op a manifest declares."""
    return infer_over(walk.order)


class IndexForm(NamedTuple):
    """How one integer index expression moves with an index variable ``s``.

    ``coefficient``/``offset`` are exact rationals: integer coefficient
    means ``coefficient * s + offset`` for every ``s`` (``0`` = no
    dependence); non-integer means it depends on ``s`` only through
    ``floor(coefficient * s + offset)`` — lanes share a value in runs of
    about ``1 / coefficient`` (a quotient form).

    ``offset`` is ``None`` when the ``s``-free part isn't a compile-time
    integer. ``labels`` are the declaration-named constants scaled by;
    ``anonymous`` marks a bare-literal scaling factor instead. ``symbol`` is ``"n_samples"`` when the
    coefficient is scaled by the launch's sample count (a run-time stride)."""

    coefficient: Fraction
    offset: Fraction | None
    labels: tuple = ()
    anonymous: bool = False
    symbol: str = ""


def index_form(node: Node, var: Node | None = None, *,
               quotients: bool = True) -> IndexForm | None:
    """Classify ``node`` as affine in the index ``var``, or return ``None``.

    ``var`` is matched by IDENTITY (a loop's
    :class:`~hawk.ir.nodes.LoopIndex`); ``None`` means the sample
    index, matched by kind. A compile-time integer factor is a
    :class:`~hawk.ir.nodes.RoleConst` or an ``i32``/``i64``
    :class:`~hawk.ir.nodes.Const`.

    * free of ``var`` — coefficient ``0``, offset its value if compile-time,
      else ``None``; ``var`` itself — ``(1, 0)``;
    * ``a + b``/``a - b`` of two exact forms add; a quotient shifted by a
      ``var``-free term keeps its coefficient/offset;
    * ``a * k``/``k * a`` scales an exact form by compile-time ``k``; a
      quotient keeps its coefficient (``k == 0`` -> ``(0, 0)``);
    * with ``quotients`` (default), ``a / k`` of a non-negative exact form
      by positive compile-time ``k`` is ``(c/k, d/k)`` (truncating ``/``
      then equals ``floor``) — the major half of
      :func:`hawk.math.split_index`.

    Anything else mentioning ``var`` is ``None`` (a product of two
    ``var``-carrying terms, a runtime factor/divisor, the periodic minor
    half, a nested quotient, a bad division, any other op). Conservative:
    ``None`` claims nothing."""
    return _core.index_form(node, var, quotients)
