# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Weekly upstream watch: compare PyPI with tools/upstream.json and write the report.

    python tools/upstream_watch.py > report.md    # reads PyPI; findings are the lines starting "- "

Reports only; always exits 0. Standard library only. Nothing here edits the manifest or publishes anything.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().with_name("upstream.json")
MARKER = "<!-- upstream-watch -->"


def vkey(version: str) -> tuple[int, ...]:
    """Leading numeric parts of a version, enough to order releases."""
    return tuple(int(p) for p in re.findall(r"\d+", version.split("+")[0].split("rc")[0])[:4])


def wheel_tags(filename: str) -> set[str]:
    """The cpXY / cpXYt / py3 tag a wheel file serves (an abi3 wheel serves every newer cpXY)."""
    parts = filename[:-4].split("-")
    if not filename.endswith(".whl") or len(parts) < 5:
        return set()
    py, abi = parts[-3], parts[-2]
    if py.startswith("py"):
        return {"py3"}
    if abi.startswith("abi3"):
        n = int(py[3:])
        return {f"cp3{m}" for m in range(n, 20)}
    return {abi if abi.endswith("t") else py}


def fetch(name: str) -> dict:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as r:
        return json.load(r)


def tags_of(pypi: dict) -> set[str]:
    out: set[str] = set()
    for f in pypi["urls"]:
        out |= wheel_tags(f["filename"])
    return out


def findings(manifest: dict, pypi: dict[str, dict]) -> list[str]:
    """One line per difference. ``pypi`` maps a PyPI name to its JSON document."""
    out: list[str] = []
    for name, up in manifest["upstreams"].items():
        doc = pypi.get(up["pypi"] or "numpy")
        if doc is None:
            out.append(f"{name}: could not read PyPI")
            continue
        latest, have = doc["info"]["version"], tags_of(doc)
        if up["pypi"] is None:
            # CPython: the newest interpreter numpy ships a wheel for.
            newest = max((t for t in have if re.fullmatch(r"cp3\d+", t)), key=lambda t: int(t[3:]))
            latest = "3." + newest[3:]
            have = set()
        ceiling, tested = up.get("ceiling"), up["tested_latest"]
        if ceiling and vkey(latest) >= vkey(ceiling):
            out.append(f"{name}: {latest} is outside the range we build for (below {ceiling}); read {up['changelog']}")
        elif tested is None:
            out.append(f"{name}: newest is {latest}; no exact version is tested, the weekly run does not compare")
        elif vkey(latest) > vkey(tested):
            out.append(f"{name}: new {latest} (tested {tested}); read {up['changelog']}")
        if up["pypi"] is None:
            continue
        need = set(up["wheel_tags"])
        unverified = set(up.get("unverified_gaps", {}))
        for tag in sorted(set(up["known_gaps"]) & have):
            out.append(f"{name}: {latest} now ships {tag}; lift the Limits line")
        for tag in sorted(unverified):
            state = "ships" if tag in have else "has no"
            out.append(f"{name}: to confirm: {latest} {state} {tag} (manifest assumed a gap: {up['unverified_gaps'][tag]})")
        for tag in sorted(need - set(up["known_gaps"]) - unverified - have):
            out.append(f"{name}: {latest} has no {tag} wheel we need")
    return out


def render(lines: list[str]) -> str:
    body = "\n".join(f"- {line}" for line in lines) if lines else "Nothing new: PyPI matches the manifest."
    return f"{MARKER}\nWeekly upstream watch (tools/upstream.json against PyPI).\n\n{body}\n"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    pypi: dict[str, dict] = {}
    for up in manifest["upstreams"].values():
        for name in {up["pypi"] or "numpy"}:
            if name not in pypi:
                try:
                    pypi[name] = fetch(name)
                except OSError as exc:
                    print(f"{name}: {exc}", file=sys.stderr)
    print(render(findings(manifest, pypi)), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
