# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Values copied from eagle are VERBATIM — diff-against-source checks
against the LIVE eagle tree, not retyped from memory. ``cross_repo``-marked:
HARD FAILS (via ``import_eagle_module``, ``FileNotFoundError``) if eagle is
not found (``RAPTOR_EAGLE_ROOT``) — deselect explicitly with
``-m "not cross_repo"`` for a repo-isolated run; this file never skips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eagle_helpers import import_eagle_module  # noqa: E402

from raptor.schema import blocks, dtypes, manifest  # noqa: E402

pytestmark = pytest.mark.cross_repo


def test_recognized_patterns_verbatim():
    roles = import_eagle_module("roles")
    assert manifest.BASE_PATTERNS | {blocks.GRAPH_BLOCK_PATTERN} == roles.RECOGNIZED_PATTERNS


def test_scalar_types_verbatim():
    eagle_dtypes = import_eagle_module("dtypes")
    assert dtypes.SCALAR_TYPES == eagle_dtypes.SCALAR_TYPES


def test_user_dtype_names_match_eagle_for_every_tag():
    # eagle names the dtype of a scalar type (eagle.DType); raptor's plain-data table names the user-facing one. They agree
    # except for banded, whose eagle dtype is the 8-byte cell (named "banded") while user arrays are float64.
    eagle_dtypes = import_eagle_module("dtypes")
    for tag in dtypes.SCALAR_TYPES:
        eagle_name = eagle_dtypes.dtype_name(eagle_dtypes.dtype_of(tag))
        if tag == "banded":
            assert eagle_name == "banded" and dtypes.USER_DTYPE[tag]["name"] == "float64"
        else:
            assert dtypes.USER_DTYPE[tag]["name"] == eagle_name


def test_kernel_name_verbatim():
    launch_mod = import_eagle_module("launch")
    assert manifest.KERNEL_NAME == launch_mod.KERNEL_NAME
