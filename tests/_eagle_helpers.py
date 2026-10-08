# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Shared test helper: locate the eagle checkout and import one of its
submodules by dotted path, without needing eagle pip-installed or cupy
present (eagle-core's own import is cupy-free at module load).

NOT a test module itself (no ``test_`` prefix) — imported by the tests that
need to diff raptor's copied values against the live eagle source.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def eagle_root() -> Path | None:
    """Return the eagle checkout root: ``RAPTOR_EAGLE_ROOT`` or the sibling
    directory next to this repo (``tests/_eagle_helpers.py`` -> repo root ->
    its sibling ``eagle``); ``None`` if not found."""
    env = os.environ.get("RAPTOR_EAGLE_ROOT")
    here = Path(__file__).resolve()
    # The eagle checkout is the sibling whose directory name carries the same
    # suffix as this one (`raptor` -> `eagle`, `raptor_x` -> `eagle_x`).
    root = Path(env) if env else here.parents[2] / ("eagle" + here.parents[1].name.removeprefix("raptor"))
    return root if root.exists() else None


def import_eagle_module(dotted: str):
    """Import ``eagle.<dotted>`` from the checkout :func:`eagle_root` finds,
    returning the actual submodule object.

    Uses ``importlib.import_module`` rather than attribute access on the
    ``eagle`` package: ``eagle/__init__.py`` does
    ``from .launch import (..., launch, ...)``, which REBINDS the ``launch``
    attribute on the ``eagle`` package object to the ``launch`` FUNCTION,
    shadowing the submodule — ``eagle.launch.KERNEL_NAME`` would then raise
    ``AttributeError``. ``importlib.import_module`` looks the submodule up via
    ``sys.modules`` instead, side-stepping the shadow.

    Raises ``FileNotFoundError`` if no eagle checkout is found — this helper
    backs the ``cross_repo``-marked battery, which must fail loud
    on a missing sibling rather than hand the caller an absence sentinel to
    (silently or not) skip on."""
    root = eagle_root()
    if root is None:
        raise FileNotFoundError(
            "eagle checkout not found (set RAPTOR_EAGLE_ROOT to point at it); "
            "this is a cross_repo test — deselect with -m 'not cross_repo' for "
            "a repo-isolated run"
        )
    python_dir = str(root / "python")
    added = python_dir not in sys.path
    if added:
        sys.path.insert(0, python_dir)
    try:
        return importlib.import_module(f"eagle.{dotted}")
    finally:
        if added:
            sys.path.remove(python_dir)
