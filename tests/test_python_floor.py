# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The Python floor is 3.9, for raptor and for the family.

(a) every .py under raptor/ parses with the 3.9 grammar; (b) none uses, at run time, what 3.9 lacks (``match``, ``except*``, a union
spelled ``A | B`` outside an annotation, ``zip(strict=)``, ``dataclass(slots=/kw_only=)``, a 3.10+ stdlib name, a parenthesised
``with``); (c) the four packages of the family state the same floor and the stable-ABI feature of this package names it.

(c) reads the sibling pyprojects ONLY from environment variables (a test does not depend on a path outside its repository):

    RAPTOR_FLOOR_HAWK    hawk's pyproject.toml
    RAPTOR_FLOOR_EAGLE   eagle's pyproject.toml
    RAPTOR_FLOOR_AETHER  aether's python pyproject

A variable that is unset skips its test, naming the variable; one that names a missing file fails."""

import ast
import os
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_local

_PKG = Path(__file__).resolve().parents[1]
_PY = sorted((_PKG / "raptor").rglob("*.py"))
_FLOOR = (3, 9)
_MATCH = getattr(ast, "Match", ())          # the node classes exist only on the interpreters whose grammar has the construct
_TRYSTAR = getattr(ast, "TryStar", ())

_TYPING_310 = {"ParamSpec", "TypeAlias", "TypeGuard", "Concatenate", "is_typeddict", "ParamSpecArgs", "ParamSpecKwargs"}
_TYPING_311 = {"Self", "LiteralString", "Never", "assert_never", "assert_type", "Required", "NotRequired", "Unpack", "TypeVarTuple",
               "reveal_type", "dataclass_transform", "get_overloads", "clear_overloads"}
_STDLIB = {("itertools", "pairwise"), ("types", "UnionType"), ("types", "NoneType"), ("types", "EllipsisType"), ("sys", "stdlib_module_names"),
           ("enum", "StrEnum"), ("datetime", "UTC"), ("math", "cbrt"), ("math", "exp2"), ("asyncio", "TaskGroup"),
           ("contextlib", "aclosing"), ("contextlib", "chdir"), ("inspect", "get_annotations"), ("operator", "call"), ("builtins", "aiter"),
           ("builtins", "anext"), ("itertools", "batched"), ("statistics", "correlation")}
_MODULES_311 = {"tomllib"}
_BAD_METHODS = {"bit_count"}                       # int.bit_count (3.10)
_TYPEISH = {"int", "float", "str", "bool", "bytes", "list", "dict", "tuple", "set", "frozenset", "complex", "object", "type"}


def _typeish(n: ast.AST) -> bool:
    if isinstance(n, ast.Constant):
        return n.value is None
    if isinstance(n, ast.Name):
        return n.id in _TYPEISH or n.id[:1].isupper()
    if isinstance(n, ast.Attribute):
        return n.attr[:1].isupper()
    if isinstance(n, ast.Subscript):
        return _typeish(n.value)
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
        return _typeish(n.left) and _typeish(n.right)
    return False


def _annotation_nodes(tree: ast.AST) -> set:
    """The ids of every node inside an annotation (not evaluated under ``from __future__ import annotations``)."""
    out: set = set()
    for n in ast.walk(tree):
        anns = []
        if isinstance(n, ast.AnnAssign):
            anns.append(n.annotation)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            anns += [a.annotation for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs + [n.args.vararg, n.args.kwarg] if a and a.annotation]
            if n.returns:
                anns.append(n.returns)
        for a in anns:
            out.update(id(x) for x in ast.walk(a))
    return out


def _hits(path: Path, grammar=_FLOOR) -> list:
    src = path.read_text()
    tree = ast.parse(src, feature_version=grammar)
    lazy = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names) for n in tree.body)
    in_ann = _annotation_nodes(tree) if lazy else set()
    out = []
    def hit(n, what):
        out.append(f"{path.relative_to(_PKG)}:{n.lineno}: {what}")
    for n in ast.walk(tree):
        if isinstance(n, _MATCH):
            hit(n, "match statement")
        elif isinstance(n, _TRYSTAR):
            hit(n, "except*")
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) and id(n) not in in_ann and _typeish(n.left) and _typeish(n.right):
            hit(n, "a type union spelled with | outside an annotation (evaluated at run time; 3.10+)")
        elif isinstance(n, ast.Call):
            f = n.func
            kws = {k.arg for k in n.keywords}
            if isinstance(f, ast.Name) and f.id == "zip" and "strict" in kws:
                hit(n, "zip(strict=)")
            if (isinstance(f, ast.Name) and f.id == "dataclass" or isinstance(f, ast.Attribute) and f.attr == "dataclass") and kws & {"slots", "kw_only", "match_args"}:
                hit(n, "dataclass(slots=/kw_only=/match_args=)")
            if isinstance(f, ast.Name) and f.id == "make_dataclass" or isinstance(f, ast.Attribute) and f.attr == "make_dataclass":
                if kws & {"slots", "kw_only", "match_args", "module", "weakref_slot"}:
                    hit(n, "make_dataclass(slots=/kw_only=/module=; 3.10+)")
            if isinstance(f, ast.Name) and f.id in ("aiter", "anext"):
                hit(n, f"{f.id}()")
            if isinstance(f, ast.Attribute) and f.attr in _BAD_METHODS:
                hit(n, f".{f.attr}() (3.10+)")
            if isinstance(f, ast.Name) and f.id == "field" and "kw_only" in kws:
                hit(n, "field(kw_only=)")
        elif isinstance(n, ast.ImportFrom):
            names = {a.name for a in n.names}
            if n.module == "typing" and names & (_TYPING_310 | _TYPING_311):
                hit(n, f"typing names of 3.10+: {sorted(names & (_TYPING_310 | _TYPING_311))}")
            for a in n.names:
                if (n.module, a.name) in _STDLIB:
                    hit(n, f"{n.module}.{a.name} (3.10+)")
            if n.module in _MODULES_311:
                hit(n, f"module {n.module} (3.11+)")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name in _MODULES_311:
                    hit(n, f"module {a.name} (3.11+)")
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and (n.value.id, n.attr) in _STDLIB:
            hit(n, f"{n.value.id}.{n.attr} (3.10+)")
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "typing" and n.attr in _TYPING_310 | _TYPING_311:
            hit(n, f"typing.{n.attr} (3.10+)")
    for i, line in enumerate(src.splitlines(), 1):
        if re.match(r"\s*(async\s+)?with\s*\(\s*(#.*)?$", line):
            out.append(f"{path.relative_to(_PKG)}:{i}: parenthesised with (3.10+ grammar)")
    return out


@pytest.mark.parametrize("path", _PY, ids=[str(p.relative_to(_PKG)) for p in _PY])
def test_parses_with_the_3_9_grammar(path):
    ast.parse(path.read_text(), feature_version=_FLOOR)


@pytest.mark.parametrize("path", _PY, ids=[str(p.relative_to(_PKG)) for p in _PY])
def test_no_construct_newer_than_3_9(path):
    assert not _hits(path)


def test_the_scan_sees_each_construct(tmp_path):
    """The scan is live: every construct it forbids, planted in a scratch package, is reported."""
    global _PKG
    import sys
    match_src = "    match x:\n        case 1: pass\n" if sys.version_info >= (3, 10) else ""     # a 3.9 parser cannot read it
    src = ("import itertools\nfrom typing import ParamSpec\nfrom dataclasses import dataclass\nA = int | None\n"
           "def f(x):\n" + match_src + "    isinstance(x, int | str)\n    zip(x, x, strict=True)\n    return x.bit_count(), itertools.pairwise(x)\n"
           "@dataclass(slots=True)\nclass C: pass\nmake_dataclass('N', [], module='m')\n")
    (tmp_path / "raptor").mkdir()
    p = tmp_path / "raptor" / "m.py"
    p.write_text(src)
    old, _PKG = _PKG, tmp_path
    try:
        got = _hits(p, grammar=None)
    finally:
        _PKG = old
    for what in (("match statement",) if match_src else ()) + ("outside an annotation", "zip(strict=)", "dataclass(slots", "make_dataclass(slots", ".bit_count()", "itertools.pairwise", "typing names of 3.10+"):
        assert any(what in g for g in got), (what, got)


def _requires(path: Path) -> str:
    m = re.search(r'^requires-python\s*=\s*"([^"]+)"', path.read_text(), re.MULTILINE)
    assert m, f"{path}: no requires-python"
    return m.group(1).replace(" ", "")


def _sibling(env: str) -> Path:
    got = os.environ.get(env)
    if not got:
        pytest.skip(f"{env} is not set: it names the sibling repository's pyproject.toml")
    if not Path(got).is_file():
        pytest.fail(f"{env}={got}: no such file")
    return Path(got)


def test_this_package_says_3_9():
    assert _requires(_PKG / "pyproject.toml") == ">=3.9"


def test_the_stable_abi_feature_names_py39():
    t = (_PKG / "rs" / "binding" / "Cargo.toml").read_text()
    assert 'abi3 = ["pyo3/abi3-py39"]' in t, "rs/binding/Cargo.toml: the abi3 feature must be pyo3/abi3-py39"
    assert "abi3-py310" not in t


@pytest.mark.parametrize("env", ["RAPTOR_FLOOR_HAWK", "RAPTOR_FLOOR_EAGLE", "RAPTOR_FLOOR_AETHER"])
def test_the_family_states_the_same_floor(env):
    p = _sibling(env)
    assert _requires(p) == ">=3.9", f"{p}: requires-python is not >=3.9"
