# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What changed between two versions of the manifest raptor/ir/core.json, per item, one line each.

    python tools/kinds_diff.py OLD NEW          two paths
    python tools/kinds_diff.py REV              a git revision of the file against the working file
    python tools/kinds_diff.py REV1 REV2        two git revisions (anything that is not a file is read as a revision)

Lines:  ``kind add NAME`` | ``kind del NAME`` | ``kind chg NAME: key, key`` (the keys of the row that differ), and the same for ``const``,
``check``, ``rule``, ``record``; a change of ``key_text`` or of a top-level scalar is ``top chg KEY``. The exit status is 0 when nothing differs, 1
when something does, 2 when an argument cannot be read, so a review script can gate on it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_FILE = "raptor/ir/core.json"
_ROOT = Path(__file__).resolve().parents[1]


def load(spec: str) -> dict:
    p = Path(spec)
    if p.is_file():
        return json.loads(p.read_text())
    out = subprocess.run(["git", "-C", str(_ROOT), "show", f"{spec}:{_FILE}"], capture_output=True, text=True)
    if out.returncode:
        print(f"{spec}: neither a file nor a git revision holding {_FILE}: {out.stderr.strip()}", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(out.stdout)


def _items(man: dict, arm: str) -> dict:
    v = man.get(arm, {})
    if isinstance(v, list):
        return {(r.get("name") if isinstance(r, dict) else str(r)): r for r in v}
    return dict(v)


def _keys_differing(a, b) -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return ["value"]


def diff(old: dict, new: dict) -> list:
    lines = []
    for label, arm in (("kind", "kinds"), ("const", "consts"), ("check", "checks"), ("rule", "rules"), ("record", "records")):
        a, b = _items(old, arm), _items(new, arm)
        lines += [f"{label} add {n}" for n in sorted(set(b) - set(a))]
        lines += [f"{label} del {n}" for n in sorted(set(a) - set(b))]
        lines += [f"{label} chg {n}: {', '.join(_keys_differing(a[n], b[n]))}" for n in sorted(set(a) & set(b)) if a[n] != b[n]]
    for k in sorted(set(old) | set(new)):
        if k not in ("kinds", "consts", "checks", "rules", "records") and old.get(k) != new.get(k):
            lines.append(f"top chg {k}")
    return lines


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (1, 2):
        print(__doc__, file=sys.stderr)
        return 2
    old = load(args[0])
    new = load(args[1]) if len(args) == 2 else json.loads((_ROOT / _FILE).read_text())
    lines = diff(old, new)
    if lines:
        print("\n".join(lines))
    return 1 if lines else 0


if __name__ == "__main__":
    raise SystemExit(main())
