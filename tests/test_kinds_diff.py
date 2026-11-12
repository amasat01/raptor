# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""tools/kinds_diff.py reports a planted change to the manifest, per kind, const, check and rule."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_local

_PKG = Path(__file__).resolve().parents[1]
_MAN = json.loads((_PKG / "raptor" / "ir" / "core.json").read_text())


def _run(*args):
    p = subprocess.run([sys.executable, str(_PKG / "tools" / "kinds_diff.py"), *args], capture_output=True, text=True)
    return p.returncode, p.stdout.splitlines()


def test_a_planted_change_is_reported_per_item(tmp_path):
    new = copy.deepcopy(_MAN)
    row = next(k for k in new["kinds"] if k.get("class") == "Const")
    row["slots"] = row["slots"] + ["extra"]                       # a changed kind: one key
    dropped = new["kinds"].pop(next(i for i, k in enumerate(new["kinds"]) if k.get("class") == "At"))
    new["kinds"].append({"name": "planted", "family": "expr"})   # an added kind
    new["consts"]["PLANTED"] = ["a"]                              # an added const
    del new["consts"]["LEAF_KINDS"]                               # a removed const
    new["consts"]["ir.BODY_KEY_HEX"] = 16                         # a changed fact
    new["checks"]["node_base"]["steps"].pop()                     # a changed check
    first_rule = sorted(new["rules"])[0]
    new["rules"][first_rule] = {"changed": True}                  # a changed rule program
    a, b = tmp_path / "old.json", tmp_path / "new.json"
    a.write_text(json.dumps(_MAN))
    b.write_text(json.dumps(new))
    rc, lines = _run(str(a), str(b))
    assert rc == 1
    assert sorted(lines) == sorted([
        "kind add planted", f"kind del {dropped['name']}", f"kind chg {row['name']}: slots",
        "const add PLANTED", "const del LEAF_KINDS", "const chg ir.BODY_KEY_HEX: value",
        "check chg node_base: steps", f"rule chg {first_rule}: " + ", ".join(sorted(set(_MAN["rules"][first_rule]) | {"changed"})),
    ])


def test_no_change_is_silent_and_a_revision_is_read_from_git():
    rc, lines = _run(str(_PKG / "raptor" / "ir" / "core.json"), str(_PKG / "raptor" / "ir" / "core.json"))
    assert (rc, lines) == (0, [])
    rc, lines = _run("HEAD")       # HEAD against the working file: the exit status is 0 or 1, never an error
    assert rc in (0, 1)
    rc, _ = _run("no-such-revision-xyz")
    assert rc not in (0, 1)
