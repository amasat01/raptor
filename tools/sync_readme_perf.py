#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regenerate README.md's headline performance sentence from eagle's committed cards.

The sentence between ``<!-- hero:begin -->`` and ``<!-- hero:end -->`` is eagle's
headline (its configured device's card at the larger N, the smaller N beside it),
rendered by eagle's own ``tools/prose_claims.py`` so the number, the rounding rule
and the wording exist in one place. raptor takes no dependency on eagle: the
engine is loaded from an eagle checkout, which only this script and CI use.

Usage::

    python tools/sync_readme_perf.py --eagle-root ../eagle           # write in place
    python tools/sync_readme_perf.py --eagle-root ../eagle --check   # exit 1 on a diff
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import re
import sys

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
HERO = re.compile(r"(<!-- hero:begin -->\n)(.*?)(\n<!-- hero:end -->)", re.DOTALL)
SUBJECT = "hawk + eagle (`eagle.simulate`)"

# Exit codes shared by the docs-fresh generators.
EXIT_STALE = 1
EXIT_USAGE = 2  # argparse only
EXIT_MISSING_CARD = 3
EXIT_MISSING_TOOL = 4


class MissingTool(Exception):
    """The eagle checkout has no claim engine."""


def load_eagle(root: pathlib.Path):
    """Import eagle's claim engine from an eagle checkout."""
    path = root / "tools" / "prose_claims.py"
    if not path.is_file():
        raise MissingTool(
            f"{path} not found: pass --eagle-root of an eagle checkout "
            "(its branch must have tools/prose_claims.py)"
        )
    spec = importlib.util.spec_from_file_location("eagle_prose_claims", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    """Entry point; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eagle-root", type=pathlib.Path, default=os.environ.get("RAPTOR_EAGLE_ROOT"))
    ap.add_argument("--check", action="store_true", help="exit 1 on a diff, write nothing")
    args = ap.parse_args(argv)
    if not args.eagle_root:
        print("pass --eagle-root (or set RAPTOR_EAGLE_ROOT) to an eagle checkout", file=sys.stderr)
        return EXIT_MISSING_TOOL
    root = pathlib.Path(args.eagle_root).resolve()
    try:
        pcl = load_eagle(root)
    except MissingTool as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_MISSING_TOOL
    try:
        hero = pcl.render_hero_sentence(pcl.headline_facts(pcl.Cards(root)), SUBJECT)
    except pcl.ClaimError as exc:
        print(f"headline card missing, nothing written: {exc}", file=sys.stderr)
        return EXIT_MISSING_CARD
    old = README.read_text()
    new = HERO.sub(lambda m: m.group(1) + hero + m.group(3), old)
    if new == old:
        print("README.md's headline numbers are up to date")
        return 0
    if args.check:
        print("stale (run tools/sync_readme_perf.py --eagle-root <eagle checkout>): README.md")
        return EXIT_STALE
    README.write_text(new)
    print("wrote README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
