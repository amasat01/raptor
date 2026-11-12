# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor is the pure math contract, whole package: every Python file under raptor/, every Rust file under rs/ and the manifest
(raptor/ir/core.json) hold no hardware name and no machine number, and every Python module imports only the standard library
and raptor. There is no allow-list: a word that collides with math is narrowed in the pattern, not excused."""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

import _stdlib

pytestmark = pytest.mark.repo_local         # reads the package's own source as text, from the repo

_PKG = Path(__file__).resolve().parents[1]
_PY = sorted((_PKG / "raptor").rglob("*.py"))
_RS = sorted((_PKG / "rs").rglob("*.rs"))
_MANIFEST = _PKG / "raptor" / "ir" / "core.json"

#: Hardware as a word of a name, a comment or a string: a vendor, an ISA family, a device class, a cache level or size word.
_NAMES = re.compile(
    r"(?<![A-Za-z0-9])(?:cuda|nvidia|nvrtc|warps?|smem|avx[0-9]*|neon|sse[0-9]*|simd|xeon|gpus?|p2000|l1d|l[123][ _-]cache|cachelines?|"
    r"sm_[0-9]+|shared[ _-]memory|cache[ _-](?:line|size|bytes))(?![A-Za-z0-9])", re.IGNORECASE)
#: Words that make an UPPER_CASE integer constant a machine fact.
_MACHINE = {"warp", "bank", "banks", "lane", "lanes", "l1", "l2", "l3", "tile", "tiles", "width", "chunk", "chunks", "cache",
            "smem", "avx", "simd", "sse", "neon"}
_RS_CONST = re.compile(r"^\s*(?:pub(?:\([a-z]+\))?\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*[a-z0-9]+\s*=\s*-?\d+\s*;", re.MULTILINE)


def _machine_name(name: str) -> bool:
    return any(w in _MACHINE or w.rstrip("0123456789") in _MACHINE for w in name.lower().split("_"))


def _name_hits(path: Path) -> list:
    return [f"{path.relative_to(_PKG)}:{n}: {line.strip()[:70]}"
            for n, line in enumerate(path.read_text().splitlines(), 1) if _NAMES.search(line)]


def _py_constants(path: Path) -> list:
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        out += [f"{path.relative_to(_PKG)}:{node.lineno}: {t.id} = {v.value}" for t, v in _pairs(node)
                if isinstance(t, ast.Name) and t.id.isupper() and _machine_name(t.id)
                and isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool)]
    return out


def _pairs(node):
    """``(target, value)`` of one statement; an unpacked ``A, B = 1, 2`` gives a pair per name."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    value = getattr(node, "value", None)
    for t in targets:
        if isinstance(t, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) and len(t.elts) == len(value.elts):
            yield from zip(t.elts, value.elts)
        else:
            yield t, value


def _ids(paths):
    return [pytest.param(p, id=str(p.relative_to(_PKG))) for p in paths]


def test_the_scanned_set_is_not_empty():
    assert len(_PY) > 20 and len(_RS) > 5 and _MANIFEST.exists()


@pytest.mark.parametrize("path", _ids((*_PY, *_RS, _MANIFEST)))
def test_no_hardware_name(path):
    assert not _name_hits(path)


@pytest.mark.parametrize("path", _ids((*_PY, *_RS, _MANIFEST)))
def test_no_machine_number_constant(path):
    text = path.read_text()
    if path.suffix == ".py":
        hits = _py_constants(path)
    elif path.suffix == ".rs":
        hits = [m for m in _RS_CONST.findall(text) if _machine_name(m)]
    else:
        consts = json.loads(text).get("consts", {})
        hits = [f"{k} = {v}" for k, v in consts.items() if re.fullmatch(r"-?\d+", str(v)) and k.isupper() and _machine_name(k)]
    assert not hits, hits


def test_no_measured_table_or_machine_default_in_raptor():
    tables = ("_ROWS", "CLASSES", "ARCH_FEATURES", "DEFAULT_ARCH")
    hits = [f"{p.relative_to(_PKG)}: {name}" for p in _PY for name in tables
            if re.search(rf"^{name}\b", p.read_text(), re.MULTILINE)]
    assert not hits, hits


def _imports(path: Path) -> list:
    """Every module a file imports, at module or function level, and every string handed to ``importlib.import_module`` /
    ``__import__``."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                out.append((node.lineno, "raptor"))       # relative: inside raptor
            else:
                out.append((node.lineno, node.module or ""))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in ("import_module", "__import__"):
                arg = node.args[0] if node.args else None
                out.append((node.lineno, arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                            else "<dynamic>"))
    return out


@pytest.mark.parametrize("path", _ids(_PY))
def test_imports_only_the_standard_library_and_raptor(path):
    stdlib = _stdlib.names()
    hits = [f"{n}: {mod}" for n, mod in _imports(path)
            if mod == "<dynamic>" or (mod.split(".")[0] not in stdlib and mod.split(".")[0] != "raptor")]
    assert not hits, hits


# ---- the pieces above raptor are not named in raptor's code --------------------------------------------------------------
# Scope: CODE tokens only. A token is an identifier (name, attribute, import, parameter, keyword), or a string constant that
# is itself a dotted identifier ("pkg.mod.Name"; a bare word such as a row owner in a table is data); docstrings, comments and prose messages may describe the family.

_PIECES = {"hawk", "eagle"}


def _names_a_piece(token: str) -> bool:
    return any(p.lower() in _PIECES for p in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", token))


_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?:/[0-9]+(?:\\n)?)?")      # a digest domain "x.y/1" too


def _py_tokens(path: Path) -> list:
    tree = ast.parse(path.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.append((n.lineno, n.id))
        elif isinstance(n, ast.Attribute):
            out.append((n.lineno, n.attr))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((n.lineno, n.name))
        elif isinstance(n, ast.arg):
            out.append((n.lineno, n.arg))
        elif isinstance(n, ast.keyword) and n.arg:
            out.append((n.value.lineno, n.arg))
        elif isinstance(n, ast.alias):
            out.append((getattr(n, "lineno", 0), n.name))
        elif isinstance(n, ast.ImportFrom):
            out.append((n.lineno, n.module or ""))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and _DOTTED.fullmatch(n.value):
            out.append((n.lineno, n.value))
    return out


def _rs_tokens(path: Path) -> list:
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), path.read_text(), flags=re.S)
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("//"):
            continue
        strings = re.findall(r'"((?:[^"\\]|\\.)*)"', line)
        rest = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line.split("//")[0] if '"' not in line else line)
        out += [(n, t) for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rest)]
        out += [(n, s) for s in strings if _DOTTED.fullmatch(s)]
    return out


def _json_tokens(path: Path) -> list:
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k), walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and _DOTTED.fullmatch(x):
            out.append((0, x))
    walk(json.loads(path.read_text()))
    return out


@pytest.mark.parametrize("path", _ids((*_PY, *_RS, _MANIFEST)))
def test_no_piece_name_in_code(path):
    toks = _py_tokens(path) if path.suffix == ".py" else _rs_tokens(path) if path.suffix == ".rs" else _json_tokens(path)
    hits = [f"{path.relative_to(_PKG)}:{n}: {t}" for n, t in toks if _names_a_piece(t)]
    assert not hits, hits
