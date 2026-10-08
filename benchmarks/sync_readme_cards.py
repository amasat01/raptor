#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regenerate README.md's numbers block from the committed raptor cards.

Mirrors ``eagle/tools/sync_readme_cards.py``: read-only on the cards
themselves, this is the only thing allowed to write the numbers block
between ``<!-- cards:begin -->`` / ``<!-- cards:end -->`` markers in
README.md -- a reader never hand-edits either the markers' content or the
prose around them.

The three numbers currently quoted in README.md's prose (the guarded-graph
skip, the guarded-graph-vs-eager-loop ratio, the zero-copy-vs-copy ratio,
hawk's vjp/jvp speed against torch) are NOT backed by a committed card yet
-- ``early_termination_card.py``, ``zero_copy_interop_card.py`` and
``autodiff_vs_torch_card.py`` are the writers; a real run needs a visible
GPU (each writer's own ``--device`` mode, never ``--dry``). Until a card
exists, this script does not touch README.md: ``main()`` refuses (exit 1,
no write) when any of the three expected card files is absent, so a
careless run never prints a block full of blanks. README.md itself carries
no ``<!-- cards:begin -->`` marker yet either (deliberately -- its text
stays unchanged until the cards exist); this script will not invent a
place to insert one. Add the markers by hand at the point in README.md
where the numbers block belongs, THEN run this script to fill it.

Usage::

    python sync_readme_cards.py              # write README.md in place
    python sync_readme_cards.py --check       # exit 1 on a diff, write nothing
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_RAPTOR_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
import _card_common as cc  # noqa: E402

README = _RAPTOR_ROOT / "README.md"
BEGIN, END = "<!-- cards:begin -->", "<!-- cards:end -->"

#: {label: card filename}, in the order the block renders them.
CARDS = {
    "Guarded graph vs. unguarded control, vs. eager loop":
        "early_termination_card.json",
    "Zero-copy alias vs. the copying round-trip": "zero_copy_interop_card.json",
    "hawk vjp/jvp vs. torch.autograd.grad / torch.func.jvp":
        "autodiff_vs_torch_card.json",
}


def _load_card(name: str) -> dict | None:
    path = _HERE / name
    if not path.is_file():
        return None
    card = json.loads(path.read_text())
    if card.get("dry"):
        raise ValueError(
            f"{name}: a DRY card can never back README.md -- real numbers "
            f"come only from a GPU run of the matching *_card.py --device")
    return card


def render_block() -> str:
    lines = [BEGIN, "", "Measured on the machine that built this page:", ""]
    for label, filename in CARDS.items():
        card = _load_card(filename)
        assert card is not None, filename  # main() already checked every card exists
        device = (card["facts"].get("device") or {}).get("name", "unknown device")
        lines.append(f"- **{label}** ({device}):")
        lines.append(f"  ```json")
        lines.append(f"  {json.dumps(card['results'], sort_keys=True)}")
        lines.append(f"  ```")
    lines += ["", "Script md5s: " + ", ".join(
        f"`{name.replace('_card.json', '_card.py')}`="
        f"`{cc.md5(_HERE / name.replace('_card.json', '_card.py'))}`"
        for name in CARDS.values()), "", END]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="exit 1 on a diff from the committed README.md; write nothing")
    args = p.parse_args(argv)

    missing = [name for name in CARDS.values() if not (_HERE / name).is_file()]
    if missing:
        print("sync_readme_cards: no real card yet for "
             f"{', '.join(missing)} -- run the matching *_card.py --device "
             "on a GPU first (never --dry). README.md is left untouched.",
             file=sys.stderr)
        return 1

    new_block = render_block()
    text = README.read_text()
    if BEGIN not in text or END not in text:
        print(f"sync_readme_cards: README.md carries no {BEGIN} / {END} "
             "markers yet -- add them by hand at the point the numbers "
             "block belongs, then re-run. README.md is left untouched.",
             file=sys.stderr)
        return 1

    before, _, rest = text.partition(BEGIN)
    _, _, after = rest.partition(END)
    updated = before + new_block + after

    if args.check:
        if updated != text:
            print("sync_readme_cards: README.md's cards block is stale "
                 "-- re-run without --check and commit the diff.", file=sys.stderr)
            return 1
        print("sync_readme_cards: README.md's cards block matches the committed cards.")
        return 0

    README.write_text(updated)
    print(f"sync_readme_cards: wrote {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
