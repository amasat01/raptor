# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivative rules of the ray kinds (:mod:`raptor.ir.rays`).

Closed forms built from hawk ops over the node's own operands, with the SAME
guarded denominators as the primitive (``aether/geometry/Ray.h``): every division
and square root sees a non-zero operand before the select, so a degenerate input
(a ray parallel to the triangle, a zero-area triangle, a zero direction, a
negative discriminant) gives a derivative of exactly zero, never ``0 * inf``. The
primitive's body is never differentiated.

``hit`` (entry 0) has zero derivative and the cotangent ``g[0]`` is ignored. The
values are the computed ones also on a miss, and the derivative follows them
(through the live arm of every ``min``/``max`` of the slab and the live root of the
sphere); there is no silhouette term, so the jump of ``t`` at a triangle edge or a
sphere's rim is not modelled. A slab with a zero direction component has a constant
interval end, so its derivative with respect to the origin and the box is zero.
``tmax`` only enters the hit test, so it has no derivative either.
"""

from __future__ import annotations

from collections.abc import Callable

from raptor import _core


def _rule(kind: str) -> tuple[Callable, Callable]:
    """``(vjp, jvp)`` of ``kind``: the program the manifest states (``rules.rays``), run in the core."""
    return (lambda n, g: _core.rule_run("rays", kind, "vjp", n, g),
            lambda n, t: _core.rule_run("rays", kind, "jvp", n, t))


#: ``kind -> (vjp, jvp)``.
RAY_RULES: dict[str, tuple[Callable, Callable]] = {k: _rule(k) for k in _core.rule_kinds("rays")}
