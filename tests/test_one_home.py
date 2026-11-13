# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One home per fact: a vocabulary, a version or a width that both languages use is a row of raptor/ir/core.json (``consts``, named
``group.NAME``) that the Python side reads (raptor/_facts.py) and the Rust core reads at build time (rs/core/build.rs, ``facts``).

Three guards: (1) the Python spelling of every fact equals the value the COMPILED module reports (``_core.facts()``), so an edit of
either side, or a manifest edited without a rebuild, is red; (2) the manifest on disk is the text compiled in; (3) no fact is spelled
again as a literal, in Python (a module-level assignment of the fact's name, or of a literal collection with the fact's members)
or in Rust (a ``const`` of the fact's name, or of a literal array with the fact's members)."""

import ast
import json
import re
from pathlib import Path

import pytest

from raptor import _core
from raptor.ir import access, nodes, types
from raptor.schema import blocks, manifest

pytestmark = pytest.mark.repo_local

_PKG = Path(__file__).resolve().parents[1]
_MANIFEST = _PKG / "raptor" / "ir" / "core.json"
_FACTS = {k: (tuple(v) if isinstance(v, list) else v) for k, v in json.loads(_MANIFEST.read_text())["consts"].items() if "." in k}

#: where the Python side spells each fact (module, attribute); a set-valued attribute is compared as a set
_PY = {
    "schema.SCHEMA_VERSION": (manifest, "SCHEMA_VERSION"), "schema.MAX_SCHEMA_VERSION": (manifest, "MAX_SCHEMA_VERSION"),
    "schema.AETHER_ABI_V1": (manifest, "AETHER_ABI_V1"), "schema.AETHER_ABI_V2": (manifest, "AETHER_ABI_V2"),
    "schema.EXEC_KEYS": (manifest, "EXEC_KEYS"), "schema.EXEC_TARGETS": (manifest, "EXEC_TARGETS"),
    "schema.EXEC_ACCESS_CLASSES": (manifest, "EXEC_ACCESS_CLASSES"), "schema.EXEC_OPS": (manifest, "EXEC_OPS"),
    "schema.BASE_PATTERNS": (manifest, "BASE_PATTERNS"), "schema.GRAPH_BLOCK_PATTERN": (blocks, "GRAPH_BLOCK_PATTERN"),
    "schema.ALL_PATTERNS": (blocks, "ALL_PATTERNS"),
    "schema.TOP_LEVEL_KEY_ORDER": (manifest, "TOP_LEVEL_KEY_ORDER"),
    "schema.TOP_LEVEL_KEY_ORDER_V2": (manifest, "TOP_LEVEL_KEY_ORDER_V2"),
    "schema.BLOCK_TOP_LEVEL_KEY_ORDER": (blocks, "BLOCK_TOP_LEVEL_KEY_ORDER"),
    "schema.BLOCK_EXEC_REF_FIELDS": (blocks, "BLOCK_EXEC_REF_FIELDS"), "schema.BLOCK_REQUIRED_FIELDS": (blocks, "BLOCK_REQUIRED_FIELDS"),
    "schema.BLOCK_FORBIDDEN_FIELDS": (blocks, "BLOCK_FORBIDDEN_FIELDS"), "schema.SCATTER_POLICIES": (blocks, "SCATTER_POLICIES"),
    "schema.EXEC_REF_KINDS": (manifest, "EXEC_REF_KINDS"),
    "ir.DTYPES": (types, "DTYPES"), "ir.BODY_KEY_HEX": (nodes, "BODY_KEY_HEX"),
}


def _same(a, b) -> bool:
    if isinstance(a, (set, frozenset)) or isinstance(b, (set, frozenset)):
        return set(a) == set(b)
    return tuple(a) == tuple(b) if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)) else a == b


def test_every_fact_has_a_python_spelling_or_is_derived():
    # ir.ACCESS_RANK is read by raptor.ir.access (rank order); every other fact is listed in _PY
    assert set(_FACTS) - set(_PY) == {"ir.ACCESS_RANK"}


@pytest.mark.parametrize("key", sorted(_PY))
def test_python_value_equals_the_compiled_value(key):
    mod, attr = _PY[key]
    compiled = _core.facts()[key]
    assert _same(getattr(mod, attr), compiled), f"{mod.__name__}.{attr} = {getattr(mod, attr)!r} but the compiled core holds {compiled!r}"


def test_access_rank_equals_the_compiled_value():
    assert tuple(access.ACCESS_CLASSES[:3]) == tuple(_core.facts()["ir.ACCESS_RANK"])
    assert sorted(access._RANK, key=access._RANK.get) == list(_core.facts()["ir.ACCESS_RANK"])


def test_compiled_facts_are_the_manifest_facts():
    assert {k: (tuple(v) if isinstance(v, list) else v) for k, v in _core.facts().items()} == _FACTS


def test_manifest_on_disk_is_the_text_compiled_in():
    assert _MANIFEST.read_text() == _core.manifest(), "raptor/ir/core.json changed without a rebuild of the module (make build)"


def _members(v):
    return frozenset(v) if isinstance(v, tuple) and len(v) > 1 else None


_SETS = {_members(v): k for k, v in _FACTS.items() if _members(v) and all(isinstance(x, str) for x in v)}
_NAMES = {k.partition(".")[2]: k for k in _FACTS}


def _literal(node):
    """The members of a literal string collection, ``True`` for a literal number or string, else None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("frozenset", "set", "tuple") and len(node.args) == 1:
        node = node.args[0]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            return frozenset(e.value for e in node.elts)
        return None
    return True if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)) and not isinstance(node.value, bool) else None


def _py_spellings():
    out = []
    for path in sorted((_PKG / "raptor").rglob("*.py")):
        for st in ast.parse(path.read_text()).body:
            tg = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, ast.AnnAssign) and st.value else []
            lit = _literal(st.value) if tg else None
            for t in tg:
                if lit is None or not isinstance(t, ast.Name):
                    continue
                if t.id in _NAMES and lit is True:
                    out.append(f"{path.relative_to(_PKG)}:{st.lineno}: {t.id} is a fact ({_NAMES[t.id]}), spelled as a literal")
                if isinstance(lit, frozenset) and lit in _SETS:
                    out.append(f"{path.relative_to(_PKG)}:{st.lineno}: {t.id} repeats the members of {_SETS[lit]}")
    return out


_RS = re.compile(r"^\s*(?:pub(?:\([a-z]+\))?\s+)?(?:const|static)\s+([A-Z][A-Z0-9_]*)\s*:\s*([^=]+?)\s*=\s*(.+?);\s*$", re.MULTILINE)


def _rs_spellings():
    out = []
    for path in sorted((_PKG / "rs").rglob("*.rs")):
        if "target" in path.parts or path.name == "nonprintable.rs":
            continue
        for m in _RS.finditer(path.read_text()):
            name, _, init = m.groups()
            line = path.read_text()[:m.start()].count("\n") + 1 + (1 if path.read_text()[m.start()] == "\n" else 0)
            strs = re.findall(r'"([^"\\]*)"', init)
            if name in _NAMES and re.fullmatch(r'-?\d+|"[^"]*"|\[.*\]', init):
                out.append(f"{path.relative_to(_PKG)}:{line}: {name} is a fact ({_NAMES[name]}), spelled as a literal")
            elif init.startswith("[") and len(strs) > 1 and frozenset(strs) in _SETS:
                out.append(f"{path.relative_to(_PKG)}:{line}: {name} repeats the members of {_SETS[frozenset(strs)]}")
    return out


def test_no_fact_is_spelled_as_a_literal_in_python():
    assert not _py_spellings()


def test_no_fact_is_spelled_as_a_literal_in_rust():
    assert not _rs_spellings()


def test_the_scan_sees_a_planted_literal(tmp_path, monkeypatch):
    """The guards are live: a literal copy of a fact, planted in a scratch package, is reported."""
    pkg = tmp_path / "raptor"
    pkg.mkdir()
    (pkg / "m.py").write_text('MAX_SCHEMA_VERSION = 4\nX = frozenset({"device", "host"})\n')
    (tmp_path / "rs").mkdir()
    (tmp_path / "rs" / "a.rs").write_text('const BODY_KEY_HEX: usize = 32;\nconst Q: [&str; 2] = ["host", "device"];\n')
    monkeypatch.setitem(globals(), "_PKG", tmp_path)
    assert len(_py_spellings()) == 2 and len(_rs_spellings()) == 2
