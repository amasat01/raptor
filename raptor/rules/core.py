# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative rule table: ONE :class:`DiffRule` per node KIND.

Keyed on ``(kind, rank)`` with a rank-generic entry under ``(kind, None)``,
so a tensor-typed IR needs one rule where a per-class table needed several.
A VJP rule returns the reverse contributions ``((child, adjoint), ...)``
for one node; a JVP rule returns that node's tangent given ``t(child)``;
both build derivative IR with the same node constructors as a primal, so
the result goes through the same canonical walk. The table is
closed-world: a kind with no derivative carries an explicit zero rule
(``mode="zero"`` — comparisons, boolean connectives) or delegates to the
transform's cross-sample special case (``mode="walker"`` — ``at``, whose
reverse is a scatter-add), never absent, so :func:`rule_for` raising
names a kind that genuinely has no rule instead of silently contributing
zero.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from raptor import _core
from raptor.ir import Const, HawkError, Node
from raptor.ir import make as _mk
from raptor.ir.nodes import REDUCE_KIND
from raptor.ir.types import TensorType
from .linalg import LINALG_RULES
from .rays import RAY_RULES


@dataclass(frozen=True)
class DiffRule:
    """One node kind's two derivative directions."""

    vjp: Callable[..., Any] | None = None
    jvp: Callable[..., Any] | None = None
    mode: str = "rule"


def _o(kind: str, *args: Node, literal: Any = None) -> Node:
    return _mk(kind, args, literal)


def zero_like(t: TensorType) -> Node:
    """A structural zero of type ``t`` (rank-0 literal, else a splat)."""
    z = Const(0.0, TensorType((), t.dtype))
    return z if t.shape == () else _o("splat", z, literal=t.shape)


def _mul(a: Node, b: Node) -> Node:
    return _o("mul", a, b)


def _sum(terms: tuple[Node | None, ...]) -> Node | None:
    live = [x for x in terms if x is not None]
    if not live:
        return None
    out = live[0]
    for x in live[1:]:
        out = _o("add", out, x)
    return out


def _zero_vjp(n: Node, g: Node) -> tuple:
    return ()


def _zero_jvp(n: Node, t: Callable[[Node], Node | None]) -> None:
    return None


# --------------------------------------------------------------------------
# elementwise binaries (rank-generic; the transform un-broadcasts)

#: The rules of every binary are templates in the manifest (``rules.bin_vjp``, ``rules.bin_jvp``). Subgradients follow JAX
#: (``fmod``/``remainder`` are piecewise linear in both operands; ``copysign`` is ``|a|`` times a sign-of-``b`` constant, with
#: ``abs``'s ``+1`` at zero), except where noted in :mod:`hawk.math`.


def _bin_vjp(n: Node, g: Node) -> tuple:
    a, b = n.operands
    return _core.rule_factor("bin_vjp", n.kind, {"n": n, "g": g, "a": a, "b": b})


def _bin_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    a, b = n.operands
    return _sum(_core.rule_factor("bin_jvp", n.kind, {"n": n, "ta": t(a), "tb": t(b), "a": a, "b": b}))


# --------------------------------------------------------------------------
# elementwise unaries (rank-generic)

#: ``d/dx op(x)`` as a factor multiplying the incoming adjoint/tangent are templates in the manifest (``rules.unary_d``);
#: ``n`` is the primal node, reused so the emitter's CSE keeps one copy.
def _unary_d(n: Node, x: Node) -> Node:
    return _core.rule_factor("unary_d", n.kind, {"n": n, "x": x})


def _unary_vjp(n: Node, g: Node) -> tuple:
    x = n.operands[0]
    return ((x, _mul(g, _unary_d(n, x))),)


def _unary_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    x = n.operands[0]
    return _mul(t(x), _unary_d(n, x))


# --------------------------------------------------------------------------
# vector / matrix algebra, ``fma``, ``clip`` and the quaternion ops: programs in the manifest (``rules.core``), run in the core


def _program(kind: str) -> DiffRule:
    return DiffRule(vjp=lambda n, g: _core.rule_run("core", kind, "vjp", n, g),
                    jvp=lambda n, t: _core.rule_run("core", kind, "jvp", n, t))


_SHAPED: dict[str, DiffRule] = {k: _program(k) for k in _core.rule_kinds("core")}


def _reduce_vjp(n: Node, g: Node):
    return _core.rule_reduce("vjp", n, g)


def _reduce_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    return _core.rule_reduce("jvp", n, t)


#: The table proper: ``(kind, rank)`` -> rule, ``rank=None`` generic. The zero kinds (piecewise constant, integer-valued or random:
#: the table stays closed-world) and the walker kinds (``at``, ``dispatch``: the transform's own rules) are listed in the manifest
#: (``rules.modes``).
_MODES = _core.rule_modes()
RULES: dict[tuple[str, int | None], DiffRule] = {
    **{(k, None): DiffRule(vjp=_bin_vjp, jvp=_bin_jvp) for k in _core.rule_kinds("bin_vjp")},
    **{(k, None): DiffRule(vjp=_unary_vjp, jvp=_unary_jvp) for k in _core.rule_kinds("unary_d")},
    **{(k, None): DiffRule(vjp=_zero_vjp, jvp=_zero_jvp, mode="zero") for k, m in _MODES if m == "zero"},
    **{(k, None): DiffRule(vjp=v, jvp=j) for k, (v, j) in LINALG_RULES.items()},
    **{(k, None): DiffRule(vjp=v, jvp=j) for k, (v, j) in RAY_RULES.items()},
    **{(k, None): rule for k, rule in _SHAPED.items()},
    (REDUCE_KIND, None): DiffRule(vjp=_reduce_vjp, jvp=_reduce_jvp),
    **{(k, None): DiffRule(mode="walker") for k, m in _MODES if m == "walker"},
}


def rule_for(kind: str, rank: int) -> DiffRule:
    """The rule for one node kind: rank-specific first, then generic."""
    rule = RULES.get((kind, rank)) or RULES.get((kind, None))
    if rule is None:
        raise HawkError(
            f"no derivative rule for node kind {kind!r} (rank {rank}): HAWK's rule "
            f"table is closed-world — a kind with no derivative is registered "
            "with an explicit zero rule, never omitted, so this is a genuinely "
            "undifferentiable node, not a silent zero"
        )
    return rule

