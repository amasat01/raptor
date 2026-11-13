# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor's surface is frozen. The surface is read statically (ast and json, no import) and listed in ci/surface.txt:

(a) every Python module under raptor/; (b) every public top-level name of each module (``__all__`` when it has one);
(c) the names the core table (raptor/ir/core.json) declares: kinds, classes, feathers, records, constants, rules, checks,
key-text records and op feathers.

Section 1 of the file is the base; section 2 holds the additions a ruling allowed, each line ending ``# U-<number>``."""

import ast
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_local         # reads the package's own source as text, from the repo

_PKG = Path(__file__).resolve().parents[1]
_FILE = _PKG / "ci" / "surface.txt"
_BASE_HEAD = "# base, generated 2026-11-13"
_ADDED_HEAD = "# added by ruling"

#: The count of section-1 lines. It may only be lowered.
BASE_COUNT = 428


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_PKG).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _public_names(tree: ast.Module) -> set:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                return set(ast.literal_eval(node.value))
            except ValueError:
                break
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):      # a re-export: ``from .x import A as A``
            out.update(a.asname for a in node.names if a.asname and a.asname == a.name)
    return {n for n in out if not n.startswith("_")}


def computed() -> set:
    items = set()
    for path in sorted((_PKG / "raptor").rglob("*.py")):
        mod = _module_name(path)
        items.add(f"module {mod}")
        items.update(f"{mod} {n}" for n in _public_names(ast.parse(path.read_text())))
    man = json.loads((_PKG / "raptor" / "ir" / "core.json").read_text())
    for row in man["kinds"]:
        items.add(f"core.json kind {row['name']}")
        if "class" in row:
            items.add(f"core.json class {row['class']}")
        items.update(f"core.json feather {f}" for f in row.get("feathers", ()))
    for section, label in (("records", "record"), ("consts", "const"), ("rules", "rule"), ("checks", "check"),
                           ("op_feathers", "op_feather")):
        value = man.get(section, ())
        names = (r["name"] for r in value) if isinstance(value, list) else value
        items.update(f"core.json {label} {n}" for n in names)
    items.update(f"core.json key_text {n}" for n in man.get("key_text", ()))
    return items


def read_file(text: str) -> tuple:
    """The lines of section 1 and of section 2 (comments and blanks left out)."""
    base, added, into = [], [], None
    for line in text.splitlines():
        if line.startswith(_BASE_HEAD):
            into = base
        elif line.startswith(_ADDED_HEAD):
            into = added
        elif line.strip() and not line.startswith("#") and into is not None:
            into.append(line.rstrip())
    return base, added


def problems(items: set, base: list, added: list) -> list:
    """What is wrong when ``items`` is checked against the two sections."""
    listed = set(base) | {re.sub(r"\s+# U-\d+$", "", a) for a in added}
    out = []
    new = sorted(items - listed)
    if new:
        out.append("raptor's surface is frozen: an addition needs the owner's ruling id in ci/surface.txt. Not listed:\n  "
                   + "\n  ".join(new))
    stale = sorted(listed - items)
    if stale:
        out.append("stale lines, remove them:\n  " + "\n  ".join(stale))
    bare = [a for a in added if not re.search(r"\s# U-\d+$", a)]
    if bare:
        out.append("section 2 lines without a ruling id (end the line with ' # U-<number>'):\n  " + "\n  ".join(bare))
    if len(base) > BASE_COUNT:
        out.append(f"section 1 has {len(base)} lines, above BASE_COUNT={BASE_COUNT}: the base may only shrink")
    return out


def test_the_surface_is_the_listed_one():
    base, added = read_file(_FILE.read_text())
    bad = problems(computed(), base, added)
    assert not bad, "\n".join(bad)


def test_the_guard_bites():
    items = {"a x", "a y"}
    assert problems(items, ["a x", "a y"], []) == []
    assert any("frozen" in p for p in problems(items | {"a z"}, ["a x", "a y"], []))
    assert any("stale" in p for p in problems(items, ["a x", "a y", "a gone"], []))
    assert any("ruling id" in p for p in problems(items, ["a x"], ["a y"]))
    assert not any("ruling id" in p for p in problems(items, ["a x"], ["a y # U-1"]))
