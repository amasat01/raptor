# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One drawing function per tutorial figure, so a notebook's hidden "plumbing"
cell is an import and a single call rather than 15-25 lines of inline
matplotlib. Each function both draws AND shows the figure (``plt.show()``),
matching how the figures were authored inline before this extraction -- the
notebook's hidden cell keeps the exact same rendered output.

Every figure here is rendered ONCE and embedded as a raster image in the
notebook's committed output -- unlike the landing site's hand-authored SVGs,
there is no separate light/dark asset to swap. Each figure therefore uses a
transparent background plus a dark, theme-neutral ink (``_INK``) so the one
rendering reads on either background.

In dark mode the site's own theme CSS (``div.cell_output img``) draws every
notebook-output image, transparent or not, on a light-grey panel -- that is
pydata-sphinx-theme's standing behaviour for un-classed images, not a defect
in these PNGs. ``_INK`` is chosen dark enough to stay legible on THAT panel
as well as on the near-white light-mode page, rather than matching the
site's lighter ``--muted`` CSS token (which reads fine standalone but is
low-contrast against the dark-mode panel).
"""

from __future__ import annotations

#: Dark, theme-neutral ink for figure text, ticks and spines -- legible both
#: on the light-mode page and on the light-grey panel pydata-sphinx-theme
#: draws behind every notebook-output image in dark mode (see module
#: docstring).
_INK = "#555555"


def _style_axes(ax) -> None:
    """Transparent face + theme-neutral ticks/spines/labels, applied after an
    axes is created. Shared by every figure below so a reader in dark mode
    never sees a white card."""
    ax.set_facecolor("none")
    ax.tick_params(colors=_INK, labelcolor=_INK)
    for spine in ax.spines.values():
        spine.set_color(_INK)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)


def two_engines_diagram() -> None:
    """tutorials/01: one manifest, two engines that never import each other."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 1.9), facecolor="none")
    _style_axes(ax)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1.02)

    boxes = {
        "manifest": (0.5, 0.48, "manifest.json\n(written once)"),
        "host": (0.13, 0.86, "host engine\nisinstance -> True"),
        "device": (0.87, 0.86, "device engine\nisinstance -> True"),
    }
    for x, y, label in boxes.values():
        ax.text(x, y, label, ha="center", va="center", fontsize=9, color=_INK,
                bbox=dict(boxstyle="round,pad=0.45", fc="none", ec="#4C72B0"))
    ax.annotate("", xy=(0.22, 0.74), xytext=(0.43, 0.56),
                arrowprops=dict(arrowstyle="->", color="#4C72B0"))
    ax.annotate("", xy=(0.78, 0.74), xytext=(0.57, 0.56),
                arrowprops=dict(arrowstyle="->", color="#DD8452"))
    ax.set_title("One manifest, two engines that never import each other", fontsize=10, color=_INK)
    plt.show()


def manifest_key_gloss() -> None:
    """tutorials/02: a v2 manifest's top-level keys, glossed in order."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor="none")
    _style_axes(ax)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    keys = [
        ("schema_version", "which rules this document follows"),
        ("pattern", "the recognized kernel-shape vocabulary"),
        ("aether_abi", "the binary-format tag; must match schema_version"),
        ("exec_targets", "v2 only: host / device, where it may run"),
        ("exec_access", "v2 only: how it touches sample vs. batch data"),
        ("plugins", "the compiled artifact(s) this manifest describes"),
    ]
    n = len(keys)
    for i, (key, note) in enumerate(keys):
        y = 0.92 - i * (0.82 / (n - 1))
        ax.text(0.03, y, key, ha="left", va="center", fontsize=10, family="monospace",
                color=_INK, bbox=dict(boxstyle="round,pad=0.3", fc="none", ec="#4C72B0"))
        ax.text(0.34, y, note, ha="left", va="center", fontsize=8.5, color=_INK)
    ax.set_title("A v2 manifest's top-level keys, in their contractual order", fontsize=10, color=_INK)
    plt.show()


def capability_grid() -> None:
    """tutorials/03: which capabilities each kind of backend declares.

    ``jvp`` (forward-mode) is ROADMAP for a trained backend such as eagle's
    (see interop_protocols.md's "Backends" section) -- only ``forward`` and
    ``vjp`` (reverse-mode, via ``loss.backward()``) are there today.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 2.4), facecolor="none")
    _style_axes(ax)

    engines = ["ToyBackend\n(this page)", "a trained backend\n(e.g. eagle)"]
    caps = ["forward", "vjp", "jvp", "train_step"]
    grid = np.array([[1, 0, 0, 0], [1, 1, 0, 0]])

    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(caps)), caps, fontsize=8.5)
    ax.set_yticks(range(len(engines)), engines, fontsize=8.5)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            label = "yes" if grid[r, c] else ("roadmap" if caps[c] == "jvp" and r == 1 else "CapabilityError")
            ax.text(c, r, label, ha="center", va="center", fontsize=7.5,
                    color="white" if grid[r, c] else "#993333")
    ax.set_title("Declared capabilities() decide what a call may do", fontsize=10, color=_INK)
    fig.tight_layout()
    plt.show()


def rows_by_framework_bar(rows: dict) -> None:
    """tutorials/04: how many declared rows cite each framework.

    ``rows`` is ``raptor.conformance.interop.ROWS`` (passed in rather than
    imported here, so this module stays free of the lesson's own imports).
    """
    import matplotlib.pyplot as plt

    counts: dict[str, int] = {}
    for decl in rows.values():
        key = decl.framework or "— (execution invariant)"
        counts[key] = counts.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(5.5, 2.6), facecolor="none")
    _style_axes(ax)
    labels = list(counts)
    ax.barh(labels, [counts[k] for k in labels], color="#4C72B0")
    ax.set_xlabel("declared rows")
    ax.set_title(f"{len(rows)} declared rows, by framework", fontsize=10, color=_INK)
    fig.tight_layout()
    plt.show()
