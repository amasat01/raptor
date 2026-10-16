#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regenerate README.md's numbers block from the committed raptor cards.

Mirrors ``eagle/tools/sync_readme_cards.py``: read-only on the cards
themselves, this is the only thing allowed to write the numbers block
between ``<!-- cards:begin -->`` / ``<!-- cards:end -->`` markers in
README.md -- a reader never hand-edits either the markers' content or the
prose around them.

Every number in the block is rendered from a committed card
(``early_termination_card.py``, ``zero_copy_interop_card.py`` and
``autodiff_vs_torch_card.py`` are the writers; a real run needs a visible
GPU, each writer's own ``--device`` mode, never ``--dry``). ``main()``
refuses (exit 3, no write) when any of the three card files is absent, exits 5
when README.md lacks the markers, and ``--check`` exits 1 when the committed
block differs from what the cards render today (CI runs it).

Usage::

    python sync_readme_cards.py              # write README.md in place
    python sync_readme_cards.py --check       # write nothing; exit 1 on a diff, 3 on a missing card
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

# Exit codes shared by the docs-fresh generators (0 fresh, 2 is argparse's).
EXIT_STALE = 1
EXIT_MISSING_CARD = 3
EXIT_NO_MARKERS = 5

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


def _x(value: float) -> str:
    return f"{value:.3g}x"


def _range(values) -> str:
    lo, hi = min(values), max(values)
    return _x(lo) if f"{lo:.3g}" == f"{hi:.3g}" else f"{lo:.3g}-{hi:.3g}x"


def _where(card: dict) -> str:
    device = (card["facts"].get("device") or {}).get("name")
    return f" ({device})" if device else ""


def render_block() -> str:
    cards = {name: _load_card(name) for name in CARDS.values()}
    assert all(c is not None for c in cards.values())  # main() checked every card exists
    et = cards["early_termination_card.json"]["results"]
    zc = cards["zero_copy_interop_card.json"]["results"]
    ad_card = cards["autodiff_vs_torch_card.json"]
    ad = sorted(ad_card["results"], key=lambda r: r["batch"])
    big = ad[-1]
    lines = [BEGIN, "", "Measured on the machine that built this page, from the committed cards "
             "under `benchmarks/`:", ""]
    lines.append(f"- **Guarded graph**{_where(cards['early_termination_card.json'])}: "
                 f"{et['batch']:,} samples over {et['steps']:,} steps, the last one stopping at step "
                 f"{et['last_termination']:,}; {_x(et['ratio_unguarded_over_guarded'])} faster per step than "
                 f"the unguarded control and {_x(et['ratio_eager_over_guarded'])} the speed of the eager loop.")
    lines.append(f"- **Zero-copy alias**{_where(cards['zero_copy_interop_card.json'])}: "
                 f"a {zc['size_mb']:g} MB buffer crosses {_x(zc['ratio'])} faster as an alias than "
                 "through the copying round-trip.")
    lines.append(f"- **hawk vjp / jvp vs torch**{_where(ad_card)}: across a batch sweep from "
                 f"{ad[0]['batch']:,} to {big['batch']:,} samples, hawk's vjp measured "
                 f"{_range([r['ratio_vjp'] for r in ad])} faster per call than `torch.autograd.grad` and its jvp "
                 f"{_range([r['ratio_jvp'] for r in ad])} faster than `torch.func.jvp`; at {big['batch']:,} samples "
                 f"{_x(big['ratio_vjp'])} and {_x(big['ratio_jvp'])}. hawk's derivatives cost "
                 f"{_range([r['hawk_vjp'] / r['hawk_primal'] for r in ad] + [r['hawk_jvp'] / r['hawk_primal'] for r in ad])}"
                 " their own primal.")
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
        return EXIT_MISSING_CARD

    new_block = render_block()
    text = README.read_text()
    if BEGIN not in text or END not in text:
        print(f"sync_readme_cards: README.md carries no {BEGIN} / {END} "
             "markers yet -- add them by hand at the point the numbers "
             "block belongs, then re-run. README.md is left untouched.",
             file=sys.stderr)
        return EXIT_NO_MARKERS

    before, _, rest = text.partition(BEGIN)
    _, _, after = rest.partition(END)
    updated = before + new_block + after

    if args.check:
        if updated != text:
            print("sync_readme_cards: README.md's cards block is stale "
                 "-- run `python benchmarks/sync_readme_cards.py` and commit the diff.", file=sys.stderr)
            return EXIT_STALE
        print("sync_readme_cards: README.md's cards block matches the committed cards.")
        return 0

    README.write_text(updated)
    print(f"sync_readme_cards: wrote {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
