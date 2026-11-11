# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Building and classifying a lowered ``for``: the boundary, the classes.

Two questions live here, belonging in neither :mod:`hawk.ir.nodes` (node
SHAPES) nor :mod:`hawk.ir.walk` (the ONE traversal).

WHICH VALUES CROSS INTO THE SCOPE: a :class:`~hawk.ir.nodes.Loop`'s
``operands`` are the carried initial values plus the body's loop-INVARIANT
boundary subexpressions, which :func:`build_loop` computes as the MAXIMAL
subexpressions depending on neither the index nor any carry. This keeps
every leaf the body reads reachable from those operands, lets the emitter
name each boundary value once instead of per iteration, and lets the
digest key them by POSITION, not object identity.

WHAT KIND OF LOOP IT IS: :func:`classify_loop` answers, from the body's
STRUCTURE rather than an author-supplied name, the question reverse mode
must ask before differentiating a loop:

* ``accumulator`` — every carry updates as ``c = c + a`` with ``a`` free of
  EVERY carry, read nowhere else; the reverse pass is a plain forward loop
  storing NOTHING;
* ``recurrence`` — anything else the rule table covers: the forward loop
  keeps one bounded array per carry and the reverse loop runs backwards;
* refused, NAMED — a nested loop that leaves early or commits per iteration
  (:func:`refuse_unsupported_nesting`); any other nested loop differentiates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .. import _core
from .nodes import Loop, LoopCarry, LoopIndex, Node

#: The class whose adjoint stores nothing.
ACCUMULATOR = "accumulator"
#: The class whose adjoint runs in reverse over the forward loop's own
#: carried values.
RECURRENCE = "recurrence"
#: The two classes — a loop is one of these or REFUSED by name. No third
#: "assume it works" class.
LOOP_CLASSES = (ACCUMULATOR, RECURRENCE)


@dataclass(frozen=True)
class LoopClass:
    """What :func:`classify_loop` found: the class, plus an accumulator's
    per-carry ADDEND ``a`` in ``c = c + a`` (empty for a recurrence)."""

    name: str
    addends: tuple = ()


# --------------------------------------------------------------------------
# the boundary


def boundary(nexts: Sequence[Node], own: frozenset, gates: frozenset = frozenset()) -> tuple:
    """The body's loop-invariant BOUNDARY values, in deterministic order. Maximal: the walk stops at the highest invariant
    node on each path, so its children ride along inside that one definition. A ``next`` that is invariant in its entirety is
    its own boundary value. A plane read under a ``dispatch`` / ``select`` arm that gathers stays in the body unless it was
    gated (``gates``)."""
    return _core.loop_boundary(nexts, own, gates)


def build_loop(index: LoopIndex, carries: Sequence[LoopCarry],
               inits: Sequence[Node], nexts: Sequence[Node], *, start: int,
               stop: int, step: int, body_sinks: Sequence[Node] = (),
               exit_cond: Node | None = None, exit_values: Sequence[Node] = (),
               count: Node | None = None, count_tail: bool = False,
               dynamic: str | None = None) -> Loop:
    """Mint a :class:`~hawk.ir.nodes.Loop`, computing its boundary for it.

    THE constructor every producer uses (the tracer, :mod:`hawk.diff.loops`
    for a derived loop), so a derived loop's operands are computed by
    exactly the code that computed the primal's."""
    own = frozenset({id(index), *(id(c) for c in carries)})
    nexts, gates = _core.loop_gated_reads(nexts, own)
    body_sinks = tuple(body_sinks)
    exit_values = tuple(exit_values)
    roots = nexts + tuple(o for sink in body_sinks for o in sink.operands)
    if exit_cond is not None:
        roots += (exit_cond, *exit_values)
    held = {id(n) for n in inits}
    if count is not None:
        held.add(id(count))
    free = tuple(n for n in boundary(roots, own, gates) if id(n) not in held)
    loop = Loop(index, carries, inits, nexts, start=start, stop=stop, step=step,
                free=free, body_sinks=body_sinks, exit_cond=exit_cond,
                exit_values=exit_values, count=count, count_tail=count_tail,
                dynamic=dynamic)
    _core.loop_check_placeholders(loop)
    return loop


def exit_kwargs(loop: Loop) -> dict:
    """``loop``'s ``break``/count as :func:`build_loop` keywords, for a
    REBUILD that must leave early exactly where the original did."""
    return {"exit_cond": loop.exit_cond, "exit_values": loop.exit_values,
            "count": loop.count, "count_tail": loop.count_tail,
            "dynamic": loop.dynamic}


def addends(node: Node) -> list:
    """Flatten an ``add`` chain into its terms (``add`` only: a ``sub`` at the top alternates the sign of what follows)."""
    return _core.loop_addends(node)


def refuse_unsupported_nesting(loop: Loop) -> None:
    """Refuse, BY NAME, a body whose nested ``for`` neither direction can carry: one that leaves early, one that commits a
    sink per iteration or runs a recorded trip count (a derived loop)."""
    _core.loop_refuse_nesting(loop)


def classify_loop(loop: Loop) -> LoopClass:
    """Which of :data:`LOOP_CLASSES` ``loop`` is, or a refusal NAMING why. Narrow and structural: every carry's body result
    must be an ``add`` chain containing that carry EXACTLY once as a bare term, read nowhere else."""
    return _core.loop_classify(loop, LoopClass)
