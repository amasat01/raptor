# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor.conformance — the interop certification declaration and its
reusable matrix-hygiene scanners. See :mod:`raptor.conformance.interop`.
"""

from __future__ import annotations

from .interop import (
    CERTIFIED,
    ROADMAP,
    ROW_KINDS,
    ROWS,
    UNIVERSE,
    RowDecl,
    assert_doc_rows_match,
    assert_no_skip_tokens,
    assert_rows_complete,
    certified_framework,
    register_row,
    roadmap_framework,
)

__all__ = [
    "CERTIFIED",
    "ROADMAP",
    "ROW_KINDS",
    "ROWS",
    "UNIVERSE",
    "RowDecl",
    "assert_doc_rows_match",
    "assert_no_skip_tokens",
    "assert_rows_complete",
    "certified_framework",
    "register_row",
    "roadmap_framework",
]
