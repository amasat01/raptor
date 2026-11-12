# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The facts both languages hold, read from ONE home: the ``consts`` arm of raptor/ir/core.json, rows named ``group.NAME``.

The Rust core reads the same rows at build time (rs/core/build.rs, module ``facts``); a fact is never spelled again in either
language. A list is served as a tuple, an integer or a string as itself."""

from __future__ import annotations

import json

from . import _core

_ALL = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(_core.manifest())["consts"].items() if "." in k}


def facts(group: str) -> dict:
    """Every fact of ``group`` ("schema", "ir"), by name."""
    head = group + "."
    return {k[len(head):]: v for k, v in _ALL.items() if k.startswith(head)}
