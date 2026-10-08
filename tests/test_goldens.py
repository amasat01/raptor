# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Golden conformance — every golden passes ``raptor.schema.validate_manifest``;
row01 is byte-identical to its eagle conformance-corpus source (md5 compare).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eagle_helpers import eagle_root  # noqa: E402

from raptor.schema import validate_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDENS = REPO_ROOT / "goldens"
_GOLDEN_FILES = sorted(GOLDENS.rglob("*.json"))


@pytest.mark.repo_local
def test_at_least_three_goldens_exist():
    assert len(_GOLDEN_FILES) >= 3, (
        f"expected >= 3 golden files, found {len(_GOLDEN_FILES)}: {_GOLDEN_FILES}"
    )


# A guardrail: a bare `parametrize("path", _GOLDEN_FILES, ...)` with an
# EMPTY _GOLDEN_FILES makes pytest auto-SKIP this test ("got empty parameter
# set for (path)") -- and a pytest SKIP exits 0, so it would pass silently.
# _GOLDEN_FILES can legitimately be empty in the isolated-venv leg's
# sibling-free scratch copy (goldens/ is repo-local, never packaged in the
# wheel) -- deselected there via repo_local, same as the test above. But the
# empty case can ALSO arise in a repo-rooted run (goldens/ accidentally
# emptied/deleted), where repo_local is never deselected -- so the fallback
# below guarantees at least one parametrized instance always exists, and that
# instance FAILS LOUD on emptiness rather than silently reporting "skipped".
_GOLDEN_PARAMS = _GOLDEN_FILES if _GOLDEN_FILES else [None]


def _golden_id(path: Path | None) -> str:
    return str(path.relative_to(GOLDENS)) if path is not None else "NO-GOLDENS-FOUND"


@pytest.mark.repo_local
@pytest.mark.parametrize("path", _GOLDEN_PARAMS, ids=_golden_id)
def test_golden_passes_validate_manifest(path):
    assert path is not None, (
        f"no golden files found under {GOLDENS} -- an empty parameter set must "
        "fail loud here, never silently skip"
    )
    doc = json.loads(path.read_text())
    validate_manifest(doc)  # must not raise


def _md5(path: Path) -> str:
    # usedforsecurity=False: md5 here is a byte-identity fingerprint, not crypto —
    # required on FIPS-enforcing hosts where plain md5() raises (3.9+ kwarg).
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


@pytest.mark.cross_repo
def test_row01_is_byte_identical_to_its_eagle_conformance_source():
    """cross_repo: HARD FAILS if eagle (or its conformance fixture) is not
    found — deselect with ``-m "not cross_repo"`` for a repo-isolated run;
    never a skip."""
    row01_golden = GOLDENS / "row01_neural_block_golden.sidecar.json"
    root = eagle_root()
    if root is None:
        raise FileNotFoundError(
            "eagle checkout not found (set RAPTOR_EAGLE_ROOT to point at it); "
            "this is a cross_repo test — deselect with -m 'not cross_repo' for "
            "a repo-isolated run"
        )
    source = root / "tests" / "conformance" / "row01_neural_block_golden.sidecar.json"
    if not source.exists():
        raise FileNotFoundError(f"row01 conformance source not found at {source}")
    assert row01_golden.read_bytes() == source.read_bytes()
    assert _md5(row01_golden) == _md5(source)
