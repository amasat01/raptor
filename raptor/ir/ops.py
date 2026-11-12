# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The op-KIND vocabulary and its result-type rules.

One tensor-typed node family needs one place that says which op kinds
exist and what type each returns: the tracer mints nodes through
:func:`make`, the derivative rule table keys on the same :data:`OP_KINDS`,
and neither can drift from the other. Typing is rank-generic: an
elementwise op broadcasts a rank-0 operand against a rank-1/rank-2 one,
dtypes promote along ``bool < i32 < i64 < f32 < f64``, and a quaternion
``tag`` survives only ``add``/``sub``/``neg`` and a scalar scaling — so a
tangent or adjoint carries the SAME type as its primal and the walk's
total merge never sees a spurious collision.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import INT_PLANE_DTYPES, TensorType
from .. import _core
from . import linalg, rays
from . import nodes as _N
from .nodes import QUAT_OPS, Node, Op
from ..errors import RaptorError

#: Rank-generic elementwise binaries (a rank-0 operand broadcasts).
ELEMENTWISE_BINARY = _N.ELEMENTWISE_BINARY
#: Rank-generic elementwise unaries.
UNARY_MATH = _N.UNARY_MATH
#: Rank-generic elementwise ternaries (every non-rank-0 operand shares one shape).
ELEMENTWISE_TERNARY = _N.ELEMENTWISE_TERNARY
#: Elementwise floating-point class tests: bool-typed, zero derivative.
PREDICATE = _N.PREDICATE
#: Elementwise comparisons: bool-typed, zero derivative.
COMPARE = _N.COMPARE
#: Boolean connectives (``lnot`` is unary): bool-typed, zero derivative.
LOGICAL = _N.LOGICAL
#: Rank-changing vector algebra. ``component`` carries a literal index,
#: ``splat`` a literal shape; ``component_at`` is ``component``'s
#: RUNTIME-index twin (a loop's symbolic ``k``), a separate KIND since its
#: arity, type rule and emission differ. ``set_component_at`` is its WRITE
#: twin: a VALUE, not a statement, so ``v[k] = e`` rebinds the name to it.
VECTOR_OPS = _N.VECTOR_OPS
#: Matrix algebra: matrix-vector, matrix-matrix, transpose, outer product.
MATRIX_OPS = _N.MATRIX_OPS
#: Per-lane linear algebra (:mod:`hawk.ir.linalg`): a factorisation node, the
#: solves against it and the reads of its fields.
LINALG_OPS = linalg.LINALG_OPS
#: The least-squares kinds (:data:`raptor.ir.linalg.LSTSQ_OPS`): math in the shared IR, lowered by each backend.
LSTSQ_OPS = linalg.LSTSQ_OPS
#: Ops that exist only in a minted form: a derivative rule, a lowering or a read mints them, and no authoring name spells them.
#: A declared fact about the op, read by the authoring-surface check; an op that has an authoring spelling is not in it.
MINTED_OPS = _N.MINTED_OPS
#: Ray intersection (:mod:`hawk.ir.rays`): one node per primitive, a ``(4,)`` result.
RAY_OPS = rays.RAY_OPS
#: The language's stateless, counter-based draws: arity 2, ``(seed, counter)``,
#: both rank-0 integer-typed, lowered to its uniform and normal generators
#: with the kernel's
#: flattened sample index as the stream's third coordinate. A re-drawn
#: value re-evaluates the SAME two integers, never RNG state, so a
#: captured graph replays by writing the wire, never recapturing.
RANDOM_OPS = _N.RANDOM_OPS
#: Integer-only binaries: ``floordiv``/``mod`` follow Python's sign rules
#: (floor division; the remainder takes the divisor's sign) and give 0 for a
#: zero divisor, as numpy does; shifts are numpy's (a count outside
#: ``[0, 32)`` gives 0, or -1 for a negative ``int32`` shifted right).
INT_BINARY = _N.INT_BINARY
#: ``clamp_index(i, n)``: the index ``i`` held to ``[0, n)`` (a table read's bound, :meth:`hawk.trace.value.TableRef.at`).
#: Integer-only unaries.
INT_UNARY = _N.INT_UNARY
#: The explicit conversion; its ``literal`` is the target dtype. float ->
#: integer truncates toward zero, saturates at the type's range and maps NaN
#: to 0; integer -> integer wraps modulo 2^32 (numpy's ``astype``).
CAST = ("cast",)
#: What an integer plane value supports beside the explicit cast.
#: The dtypes a ``cast`` reaches: a 32-bit integer plane, an index
#: integer, or the kernel's ``Real`` (``f64``; the scalar mode narrows it).
CAST_TARGETS = INT_PLANE_DTYPES + ("i32", "f64")

#: Every op kind a traced body can mint (``at``/``select`` mint their own
#: node subclasses but are KINDS here, since the rule table keys on kind).
OP_KINDS = (ELEMENTWISE_BINARY + UNARY_MATH + COMPARE + LOGICAL + VECTOR_OPS
            + MATRIX_OPS + LINALG_OPS + LSTSQ_OPS + RAY_OPS + QUAT_OPS + RANDOM_OPS + ("select", "dispatch", "at")
            + ELEMENTWISE_TERNARY + PREDICATE + INT_BINARY + INT_UNARY + CAST)

#: Kinds under which a semantic tag survives (a quaternion sum is a quaternion).


def promote(a: str, b: str) -> str:
    """The dtype of an elementwise combination of ``a`` and ``b``.

    An integer plane dtype (``int32``/``uint32``) combines only with itself:
    any other partner is refused, naming the cast that makes it explicit."""
    return _core.promote_dtype(a, b)


def make(kind: str, operands: Sequence[Node], literal: object = None) -> Op:
    """Mint an :class:`~hawk.ir.nodes.Op` of ``kind`` with its inferred type."""
    return _core.make(kind, operands, literal)




def result_type(kind: str, operands: Sequence[Node],
                literal: object = None) -> TensorType:
    """The :class:`~hawk.types.TensorType` ``kind`` returns over ``operands``."""
    return _core.result_type(kind, operands, literal)


























