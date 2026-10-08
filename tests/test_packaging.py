# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Package metadata sanity — dist name, zero hard deps.

Reads the two flat scalars this test needs (``project.name``,
``project.dependencies``) directly from ``pyproject.toml`` TEXT with a
targeted regex, rather than a TOML parser: ``tomllib`` is 3.11+ only, and
raptor is zero-hard-runtime-deps BY DESIGN, so pulling in a ``tomli``
backport — even as a test-only extra — would be new machinery bought for one
test where a text read suffices. This test used to vanish outright below
3.11 via ``pytest.importorskip("tomllib", ...)`` — an absence-tolerant rc=0
path on exactly the packaging invariant the family cares most about: no such
path may exist anywhere in the family; a pytest SKIP is RC=0 and RED.
``[project]`` is a small, hand-written, flat TOML table, so a
regex scoped to that table's own text (never some other table's same-named
key) is not fragile the way it would be against arbitrary TOML.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

_NAME_RE = re.compile(r'^name\s*=\s*"([^"]+)"', re.MULTILINE)
_DEPENDENCIES_RE = re.compile(r"^dependencies\s*=\s*(\[[^\]]*\])", re.MULTILINE)


def _project_table_text(pyproject_text: str) -> str:
    """The raw text of the ``[project]`` table only — up to the next
    top-level ``[section]`` header or EOF — so the ``name``/``dependencies``
    regexes below can never match a same-named key in some other table
    (e.g. ``[tool.setuptools.package-data]``)."""
    m = re.search(r"^\[project\]\s*\n(.*?)(?=^\[|\Z)", pyproject_text, re.MULTILINE | re.DOTALL)
    assert m, "no [project] table found in pyproject.toml"
    return m.group(1)


@pytest.mark.repo_local
def test_pyproject_declares_zero_hard_deps_and_dist_name():
    """The DIST name is `raptor-core`; the IMPORT name stays flat (`raptor`).

    Bare `raptor` on PyPI is an unrelated py2 deploy tool, so the
    distribution carries the family's `raptor-` namespace while every import
    site is untouched. Pinning the dist name here is what makes an
    accidental revert to the colliding name a RED.
    """
    project_text = _project_table_text(PYPROJECT.read_text())

    name_match = _NAME_RE.search(project_text)
    assert name_match, 'no `name = "..."` found under [project] in pyproject.toml'
    assert name_match.group(1) == "raptor-core"

    deps_match = _DEPENDENCIES_RE.search(project_text)
    assert deps_match, "no `dependencies = [...]` found under [project] in pyproject.toml"
    assert deps_match.group(1) == "[]", (
        f"raptor must declare zero hard runtime deps; found "
        f"dependencies = {deps_match.group(1)}"
    )
