# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The block-shaped keyed sum, a kernel-model node: the sum over the lanes of a key of a product of per-row rank-1
values. A frontend emits it (the most structured form); whether a build keeps it as one block or expands it per entry
is a lowering decision and lives with the lowering passes."""

from __future__ import annotations

from .. import _core
from .nodes import BlockSum, BlockSumWrite, Node  # noqa: F401  (re-exported)


def factors(payload: Node) -> tuple | None:
    """``(u, v, rows)`` when ``payload`` is a sum over per-lane rows of rank-1 products, whose sum
    over the lanes of a key is ``sum_r u_r v_r^T``: ``outer(u, v)`` (a ``p x q`` block), ``u * v`` of
    a rank-1 ``u`` and a rank-0 ``v`` (a ``p`` vector: the ``q = 1`` block), and, for a per-lane
    ``rows x p`` matrix ``M``, ``transpose(M) @ M`` (``u = v = M``, a ``p x p`` block) and
    ``transpose(M) @ r`` (``v = r``, a ``p`` vector). ``rows`` is how many rows a lane holds (1 for
    the first two spellings); else ``None``."""
    return _core.block_factors(payload)


def ordered_case(payload: Node, policy: str, max_dim: int | None = None) -> bool:
    """Whether the ordered policy keeps ``payload`` (a floating scalar or vector of at most ``max_dim`` elements; ``None``:
    any length) as one block."""
    return _core.ordered_case(payload, policy, max_dim)


def block_shaped(payload: Node, policy: str = "auto", max_dim: int | None = None) -> bool:
    """Whether the keyed sum of ``payload`` CAN be one :class:`BlockSum`, by structure alone: a product of per-row
    rank-1 values of a float type of at most ``max_dim`` elements per extent (``None``: any); under ``policy="ordered"`` any floating scalar or vector
    payload is (rows of a key added in lane order: no atomics, the same bits on every launch)."""
    return _core.block_shaped(payload, policy, max_dim)


def expand_stores(sinks) -> tuple:
    """``sinks`` with every :class:`BlockSumWrite` replaced by its per-entry accumulates."""
    return _core.expand_stores(sinks)
