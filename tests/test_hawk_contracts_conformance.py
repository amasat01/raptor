# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Cross-check HAWK's severed, conformance-gated copy of the schema/ABI/layout
constants (``hawk/hawk/_contracts.py``) against every other home of the same
values — raptor's own schema module and eagle's Python + C++ sources.

``hawk`` is NEVER imported here — ``_contracts.py`` is read with :mod:`ast`
and its module-level assignments are pulled out with ``ast.literal_eval``.
eagle is read the same read-only way the existing suite already does
(``_eagle_helpers.import_eagle_module`` for Python; a regex for the C++
headers).

HAWK's ``_contracts.py`` also carries :data:`MAX_SCHEMA_VERSION` ("the MAX
twin": HAWK writes only schema-v2 documents) and the three by-value mirror
sizes ``GRefMirror`` / ``ScalarHandle`` / ``IntHandle`` — raptor itself
declares no mirror-size constants, so those three are checked against
eagle's Python ctypes structs (where one exists) and eagle's C++
``static_assert``s only; the checked-against-raptor set stays
{``SCHEMA_VERSION``, ``MAX_SCHEMA_VERSION``, ``AETHER_ABI_VERSION``}.

Resolution note: the hawk checkout is the sibling literally named ``hawk``,
whatever suffix this checkout's directory carries; :func:`hawk_root` does not
mirror :func:`_eagle_helpers.eagle_root`'s suffix matching.

``cross_repo``-marked: HARD FAILS whenever a needed sibling checkout is not
found — deselect explicitly with ``-m "not cross_repo"`` for a
repo-isolated run; this file never skips. Locates the siblings via the
``RAPTOR_EAGLE_ROOT`` / ``RAPTOR_HAWK_ROOT`` convention.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eagle_helpers import eagle_root, import_eagle_module  # noqa: E402

from raptor.schema import manifest as raptor_manifest  # noqa: E402

_CPP_SCHEMA_VERSION_RE = re.compile(
    r"inline constexpr int kPluginSchemaVersion = (\d+);"
)
_CPP_MAX_SCHEMA_VERSION_RE = re.compile(
    r"inline constexpr int kPluginMaxSchemaVersion = (\d+);"
)
_CPP_ABI_V2_RE = re.compile(r'#define EAGLE_AETHER_ABI_V2 "([^"]+)"')
_CPP_SIZEOF_RE = {
    "GRefMirror": re.compile(r"static_assert\(sizeof\(GRefMirror\) == (\d+),"),
    "ScalarHandle": re.compile(r"static_assert\(sizeof\(ScalarHandle\) == (\d+),"),
    "IntHandle": re.compile(r"static_assert\(sizeof\(IntHandle\) == (\d+),"),
}

_CONTRACTS_NAMES = (
    "SCHEMA_VERSION",
    "MAX_SCHEMA_VERSION",
    "AETHER_ABI_VERSION",
    "GREF_MIRROR_SIZE",
    "SCALAR_HANDLE_SIZE",
    "INT_HANDLE_SIZE",
)

pytestmark = pytest.mark.cross_repo


def hawk_root() -> Path | None:
    """The hawk checkout: ``RAPTOR_HAWK_ROOT`` or the workspace-root sibling
    literally named ``hawk`` — hawk carries no ``-abi``/suffix, so this
    deliberately does NOT mirror :func:`_eagle_helpers.eagle_root`'s
    suffix-matching convention."""
    env = os.environ.get("RAPTOR_HAWK_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "hawk"
    return root if root.exists() else None


def read_hawk_contracts() -> dict:
    """AST-read ``hawk/hawk/_contracts.py``'s module-level constants,
    WITHOUT importing ``hawk`` — only the literal assignments named in
    :data:`_CONTRACTS_NAMES` are evaluated.

    Raises ``FileNotFoundError`` if no hawk checkout is found — this is a
    ``cross_repo`` test; a missing sibling is a hard failure, never a
    skip."""
    root = hawk_root()
    if root is None:
        raise FileNotFoundError(
            "hawk checkout not found (set RAPTOR_HAWK_ROOT to point at it); "
            "this is a cross_repo test — deselect with -m 'not cross_repo' for "
            "a repo-isolated run"
        )
    path = root / "hawk" / "_contracts.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    values = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        name = node.targets[0].id
        if name in _CONTRACTS_NAMES:
            values[name] = ast.literal_eval(node.value)
    return values


def _require_eagle_root() -> Path:
    """:func:`_eagle_helpers.eagle_root`, but hard-failing on absence
    (``cross_repo``) instead of handing the caller ``None`` to (silently or
    not) skip on."""
    root = eagle_root()
    if root is None:
        raise FileNotFoundError(
            "eagle checkout not found (set RAPTOR_EAGLE_ROOT to point at it); "
            "this is a cross_repo test — deselect with -m 'not cross_repo' for "
            "a repo-isolated run"
        )
    return root


def _cpp_schema_version(eagle_checkout: Path) -> int:
    text = (eagle_checkout / "plugin" / "roles.h").read_text()
    m = _CPP_SCHEMA_VERSION_RE.search(text)
    assert m, "kPluginSchemaVersion not found in eagle/plugin/roles.h"
    return int(m.group(1))


def _cpp_max_schema_version(eagle_checkout: Path) -> int:
    text = (eagle_checkout / "plugin" / "roles.h").read_text()
    m = _CPP_MAX_SCHEMA_VERSION_RE.search(text)
    assert m, "kPluginMaxSchemaVersion not found in eagle/plugin/roles.h"
    return int(m.group(1))


def _cpp_abi_v2(eagle_checkout: Path) -> str:
    text = (eagle_checkout / "plugin" / "gref_abi.h").read_text()
    m = _CPP_ABI_V2_RE.search(text)
    assert m, "EAGLE_AETHER_ABI_V2 not found in eagle/plugin/gref_abi.h"
    return m.group(1)


def _cpp_sizeof(eagle_checkout: Path, struct: str) -> int:
    # the layout pins live in gref_layout.h; gref_abi.h keeps the host validation
    text = "".join((eagle_checkout / "plugin" / name).read_text()
                   for name in ("gref_abi.h", "gref_layout.h"))
    m = _CPP_SIZEOF_RE[struct].search(text)
    assert m, f"sizeof({struct}) static_assert not found in eagle/plugin/gref_abi.h or gref_layout.h"
    return int(m.group(1))


@pytest.fixture()
def hawk_contracts():
    return read_hawk_contracts()


@pytest.fixture()
def eagle_py():
    return {"abi": import_eagle_module("abi"), "roles": import_eagle_module("roles")}


def test_hawk_contracts_has_every_expected_constant(hawk_contracts):
    for name in _CONTRACTS_NAMES:
        assert name in hawk_contracts, f"hawk/_contracts.py is missing {name!r}"


def test_schema_version_four_way(hawk_contracts, eagle_py):
    """raptor / eagle(py) / eagle(C++ regex) / hawk(AST) all agree on the
    FLOOR (the four-way cross-repo pin)."""
    root = _require_eagle_root()
    assert raptor_manifest.SCHEMA_VERSION == eagle_py["roles"].SCHEMA_VERSION
    assert raptor_manifest.SCHEMA_VERSION == _cpp_schema_version(root)
    assert raptor_manifest.SCHEMA_VERSION == hawk_contracts["SCHEMA_VERSION"]


def test_max_schema_version_four_way(hawk_contracts, eagle_py):
    """raptor / eagle(py) / eagle(C++ regex) / hawk(AST) all agree on the
    CEILING — the value HAWK actually writes (the "MAX twin")."""
    root = _require_eagle_root()
    assert raptor_manifest.MAX_SCHEMA_VERSION == eagle_py["roles"].MAX_SCHEMA_VERSION
    assert raptor_manifest.MAX_SCHEMA_VERSION == _cpp_max_schema_version(root)
    assert raptor_manifest.MAX_SCHEMA_VERSION == hawk_contracts["MAX_SCHEMA_VERSION"]


def test_aether_abi_version_four_way(hawk_contracts, eagle_py):
    """raptor / eagle(py) / eagle(C++ regex) / hawk(AST) all agree on the v2
    ABI tag — HAWK never emits v1."""
    root = _require_eagle_root()
    assert raptor_manifest.AETHER_ABI_V2 == eagle_py["abi"].ABI_TAG_V2
    assert raptor_manifest.AETHER_ABI_V2 == _cpp_abi_v2(root)
    assert raptor_manifest.AETHER_ABI_V2 == hawk_contracts["AETHER_ABI_VERSION"]


@pytest.mark.parametrize(
    ("hawk_name", "cpp_struct"),
    [
        ("GREF_MIRROR_SIZE", "GRefMirror"),
        ("SCALAR_HANDLE_SIZE", "ScalarHandle"),
        ("INT_HANDLE_SIZE", "IntHandle"),
    ],
)
def test_mirror_layout_sizes_match_eagle_cpp(hawk_contracts, hawk_name, cpp_struct):
    """hawk(AST) matches eagle's C++ ``static_assert`` for all three mirrors —
    no raptor-side constant exists for these (raptor declares no
    mirror-layout vocabulary), so this leg is a 2-way check."""
    root = _require_eagle_root()
    assert hawk_contracts[hawk_name] == _cpp_sizeof(root, cpp_struct)


def test_mirror_layout_sizes_match_eagle_python_ctypes(hawk_contracts, eagle_py):
    """hawk(AST) matches eagle's Python ctypes mirrors where one exists
    (``GRefMirror``/``ScalarHandle`` — eagle ships no Python ``IntHandle``
    ctypes twin)."""
    host_launch = import_eagle_module("host_launch")
    import ctypes

    assert hawk_contracts["GREF_MIRROR_SIZE"] == ctypes.sizeof(host_launch.GRefMirror)
    assert hawk_contracts["SCALAR_HANDLE_SIZE"] == ctypes.sizeof(
        host_launch.ScalarHandle
    )
