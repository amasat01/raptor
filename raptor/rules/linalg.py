# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivative rules of the per-lane linear-algebra kinds (:mod:`raptor.ir.linalg`).

A factorisation node (``lu``, ``chol``, ``eigh``) is an identity carrier: its
tangent IS its input's tangent and the adjoint it collects IS its input's
adjoint. The consumers and projections hang the rules off it, read the same
factor node for the primal and for the derivative, and never factor again, so
a derivative holds one factorisation per call site.

``lu_solve``/``lu_det`` differentiate ``A`` as given. ``cho_solve``,
``chol_factor`` and the ``eigh`` projections differentiate the symmetric part
``S = (A + A^T) / 2`` they read, so a non-symmetric perturbation agrees with
a finite difference of the primal. The Cholesky factor follows
``dL = L Phi(L^-1 dS L^-T)`` with ``Phi`` the lower triangle at half the
diagonal, reversed by the matching rule; both are two triangular
multi-right-hand-side solves and one matrix product on the existing factor.

A ``chol_from`` factor is the one carrier of a different input: its consumers reduce to the
adjoint (tangent) of ``A = L L^T``, and it maps that to the lower triangle of the ``L`` the
program built (``dA = dL L^T + L dL^T``).

A failed lane (``info > 0``) has defined outputs of zero, so its derivatives
are zero: the Cholesky rules mask on ``info`` (their triangular solves would
divide by the partial factor), and an eigenvector derivative on a lane whose
neighbouring eigenvalues are too close (``info`` bit 2) is zero.
"""

from __future__ import annotations

from collections.abc import Callable

from raptor import _core


def _rule(kind: str) -> tuple[Callable, Callable]:
    """``(vjp, jvp)`` of ``kind``: the program the manifest states (``rules.linalg``, with its ``defs``), run in the core."""
    return (lambda n, g: _core.rule_run("linalg", kind, "vjp", n, g),
            lambda n, t: _core.rule_run("linalg", kind, "jvp", n, t))


#: ``kind -> (vjp, jvp)``; the ``*_info`` kinds are integer-typed and sit with the explicit zero rules in :mod:`raptor.rules.core`.
LINALG_RULES: dict[str, tuple[Callable, Callable]] = {k: _rule(k) for k in _core.rule_kinds("linalg")}
