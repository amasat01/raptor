# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor's own distribution version.

Distinct from :data:`raptor.schema.manifest.SCHEMA_VERSION` (the plugin-schema
version this package's schema documents describe) — this is raptor-the
-package's own version number.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _checkout_version() -> str:
    """The ``version`` string of the checkout's ``pyproject.toml`` (its one home), else ``"unknown"``."""
    try:
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return found.group(1) if found else "unknown"


try:
    __version__ = version("raptor-core")
except PackageNotFoundError:  # run from a checkout without an install
    __version__ = _checkout_version()
