# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""tools/upstream.json names every upstream the interop matrix certifies, and the weekly diff reads it right."""
import importlib.util
import json
import pathlib

import pytest

from raptor.conformance.interop import CERTIFIED

ROOT = pathlib.Path(__file__).resolve().parent.parent

# reads the repository's tools/, which are not in the wheel
pytestmark = pytest.mark.repo_local

TOOLCHAIN = {"cuda-bindings", "nanobind", "cibuildwheel", "cpython"}
FIELDS = {"pypi", "floor", "tested_latest", "wheel_tags", "known_gaps", "mechanisms", "changelog", "sources"}


def _watch():
    spec = importlib.util.spec_from_file_location("upstream_watch", ROOT / "tools" / "upstream_watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _upstreams():
    # read when a test runs, not at import: the installed-wheel check collects these tests without tools/
    return json.loads((ROOT / "tools" / "upstream.json").read_text())["upstreams"]


def test_every_certified_framework_and_the_toolchain_has_an_entry():
    assert set(_upstreams()) == set(CERTIFIED) | TOOLCHAIN


@pytest.mark.parametrize("name", sorted(set(CERTIFIED) | TOOLCHAIN))
def test_entry_is_complete_and_consistent(name):
    up = _upstreams()[name]
    assert FIELDS <= set(up)
    assert up["mechanisms"] and up["sources"] and up["changelog"].startswith("https://")
    assert set(up["known_gaps"]) | set(up["unverified_gaps"]) <= set(up["wheel_tags"])
    assert not set(up["known_gaps"]) & set(up["unverified_gaps"])
    assert all(up["unverified_gaps"].values()), "an unverified gap states its reason"
    w = _watch()
    assert w.vkey(up["floor"]) <= w.vkey(up["tested_latest"] or up["floor"])


def test_wheel_tags():
    w = _watch()
    assert w.wheel_tags("numpy-2.1.3-cp312-cp312-manylinux_2_17_x86_64.whl") == {"cp312"}
    assert w.wheel_tags("torch-2.14.1-cp314-cp314t-manylinux_2_28_x86_64.whl") == {"cp314t"}
    assert w.wheel_tags("warp_lang-1.18.0-py3-none-manylinux_2_28_x86_64.whl") == {"py3"}
    assert {"cp39", "cp314"} <= w.wheel_tags("x-1-cp39-abi3-manylinux_2_17_x86_64.whl")
    assert w.wheel_tags("x-1.tar.gz") == set()


def _doc(version, *files):
    return {"info": {"version": version}, "urls": [{"filename": f} for f in files]}


def test_findings_on_canned_pypi_documents():
    w = _watch()
    manifest = {"upstreams": {
        "cupy": {"pypi": "cupy", "floor": "13.6", "tested_latest": "14.0.1", "wheel_tags": ["cp314", "cp315"],
                 "known_gaps": [], "unverified_gaps": {"cp315": "assumed"}, "changelog": "https://c"},
        "cibuildwheel": {"pypi": "cibuildwheel", "floor": "3.4.1", "tested_latest": "3.4.1", "ceiling": "4",
                         "wheel_tags": ["py3"], "known_gaps": [], "changelog": "https://b"},
        "numpy": {"pypi": "numpy", "floor": "1.26", "tested_latest": "2.1.3", "wheel_tags": ["cp314"],
                  "known_gaps": [], "changelog": "https://n"},
        "cpython": {"pypi": None, "floor": "3.9", "tested_latest": "3.14", "wheel_tags": [], "known_gaps": [],
                    "changelog": "https://p"},
    }}
    pypi = {
        "cupy": _doc("14.1.0", "cupy-14.1.0-cp314-cp314-manylinux_2_17_x86_64.whl",
                     "cupy-14.1.0-cp315-cp315-manylinux_2_17_x86_64.whl"),
        "cibuildwheel": _doc("4.0.0", "cibuildwheel-4.0.0-py3-none-any.whl"),
        "numpy": _doc("2.1.3", "numpy-2.1.3-cp315-cp315-manylinux_2_17_x86_64.whl"),
    }
    text = "\n".join(w.findings(manifest, pypi))
    assert "cupy: new 14.1.0" in text
    assert "cupy: to confirm: 14.1.0 ships cp315 (manifest assumed a gap: assumed)" in text
    assert "now ships" not in text
    assert "cibuildwheel: 4.0.0 is outside the range" in text
    assert "numpy: 2.1.3 has no cp314 wheel we need" in text
    assert "cpython: new 3.15" in text
    # an unverified gap that is absent is also "to confirm", never a known gap
    pypi["cupy"] = _doc("14.0.1", "cupy-14.0.1-cp314-cp314-manylinux_2_17_x86_64.whl")
    assert "cupy: to confirm: 14.0.1 has no cp315" in "\n".join(w.findings(manifest, pypi))
    assert w.render([]).count("\n- ") == 0
