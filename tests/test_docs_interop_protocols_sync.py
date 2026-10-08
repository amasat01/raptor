# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The foundational-protocols document is kept in sync with the
declaration, mechanically.

``docs/content/interop_protocols.md`` publishes the interop certification
matrix in prose. The declaration in :mod:`raptor.conformance.interop` is the
authority; this module asserts the two agree as SETS, in BOTH directions — a
declared row absent from the document is a broken promise (something is
certified and unpublished), and a row id in the document that is not declared
is worse (something is published and uncertified).

**Why the pattern is passed explicitly.**
:func:`~raptor.conformance.interop.assert_doc_rows_match` defaults
``row_id_pattern`` to *any* backtick-quoted identifier. Against a document
written as prose rather than a bare table, that default matches every
``numpy``, ``cupy``, ``_from_xp`` and ``CapabilityError`` the text mentions and
reports them as undeclared rows — measured on the shipped document, the default
pattern yields 16 spurious ids. :data:`ROW_ID_PATTERN` below instead matches
only the row-id LEXICAL SHAPE: uppercase alphanumeric segments joined by
hyphens, at least two of them (``T-IN-CUDA-ALIAS``, ``NP-ABI-PTR``,
``STREAM-TORCH-PRODUCER-ORDER``). Single-word backticked tokens (``ALIAS``,
``COPY``, ``ROADMAP``) and lowercase/underscored ones never match, so the
document stays readable without weakening the check.

The four ``execution_*`` rows use eagle's own snake_case row-id spelling,
not this file's UPPER-HYPHEN convention (kept as given rather than renamed —
see ``raptor.conformance.interop.ROWS``'s comment).
:data:`ROW_ID_PATTERN` therefore adds a SECOND, narrowly-scoped alternative:
identifiers starting with the literal ``execution_`` prefix. A blanket widen to
"any lowercase, underscore-joined identifier" was measured to spuriously match
this very document's own ``certified_framework``/``roadmap_framework`` backticked
function-name mentions (neither is a declared row) — the ``execution_`` prefix
anchor avoids that without reducing protection against the original spurious
matches (``ALIAS``, ``numpy``, ``_from_xp``, etc.) the module docstring above
already documents.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from raptor.conformance.interop import ROWS, assert_doc_rows_match

#: A guardrail: every test in this file reads the repo's own docs/ tree
#: (never shipped in the wheel) -- deselected only in the isolated-venv leg,
#: always run repo-rooted (CI, dev runs). See pyproject.toml's marker docstring.
pytestmark = pytest.mark.repo_local

#: the row-id lexical shape, and ONLY that: >=2 uppercase alphanumeric segments
#: joined by hyphens, backtick-quoted -- OR an identifier starting with
#: ``execution_`` (the execution rows' own snake_case spelling).
ROW_ID_PATTERN = r"`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+|execution_[a-z_]+)`"

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
PROTOCOLS_DOC = DOCS_ROOT / "content" / "interop_protocols.md"


def test_protocols_doc_exists():
    assert PROTOCOLS_DOC.is_file(), f"missing protocols document: {PROTOCOLS_DOC}"


def test_protocols_doc_rows_match_declaration():
    """Both directions: every declared row is published, and every published
    row id is declared."""
    assert_doc_rows_match(PROTOCOLS_DOC, ROWS, row_id_pattern=ROW_ID_PATTERN)


def test_protocols_doc_is_wired_into_the_toctree():
    """A document nobody can navigate to is not published. Sphinx would emit a
    warning for an orphan page, but warnings are not errors in the default
    build, so the wiring is asserted here instead."""
    index = (DOCS_ROOT / "index.md").read_text()
    assert re.search(r"^\s*content/interop_protocols\s*$", index, re.MULTILINE), (
        "docs/index.md's toctree does not list content/interop_protocols"
    )
