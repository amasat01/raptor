# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Ray intersection: the op kinds, their type rule and their cost.

Three kinds, each ONE node whose value is a rank-1 ``(4,)`` result read with the
``component`` op, like the quaternion kinds:

* ``ray_tri``    ``(o, d, a, b, c, tmax)`` -> ``[hit, t, u, v]``
* ``ray_aabb``   ``(o, d, lo, hi, tmax)``  -> ``[hit, t_enter, t_exit, 0]``
* ``ray_sphere`` ``(o, d, c, r, tmax)``    -> ``[hit, t0, t1, 0]``

``o``, ``d``, ``a``, ``b``, ``c``, ``lo``, ``hi`` are length-3 vectors; ``r`` and
``tmax`` are rank-0 reals. The arithmetic and the degenerate-input rules are
``aether::geometry``'s (``aether/geometry/Ray.h``): values are the computed ones also
on a miss, and a degenerate input gives finite values.
"""

from __future__ import annotations

from .nodes import RAY_OPS, Op  # noqa: F401

#: The ray kinds (the manifest's ``RAY_OPS``; its ``RAY_ARITY_PAIRS`` and ``RAY_SCALAR_PAIRS`` give the operand counts and the rank-0 positions).

#: Floating-point operations one lane runs (multiplies, adds, compares, selects).
_COUNT = {"ray_tri": 70.0, "ray_aabb": 60.0, "ray_sphere": 45.0}




def op_count(node: Op) -> float:
    """Operations ``node`` runs."""
    return _COUNT[node.kind]
