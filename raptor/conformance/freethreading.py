# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threaded CPython conformance harness (the ``FT-*`` row family).

Pure standard library, importable on every interpreter. ``pytest`` is imported
function-locally, exactly as :mod:`raptor.conformance.interop` does, so the
rest of :mod:`raptor.conformance` stays importable without it.

The harness is written for reuse by every package in the family that ships a
module to be run under 3.13t / 3.14t. Its pieces:

* :func:`require_free_threaded` -- skip on a GIL build, **fail** when the GIL
  state is forced from outside (``-X gil=...`` or ``PYTHON_GIL``): a forced
  state hides a missing ``FREE_THREADED`` declaration, so a run under it
  certifies nothing.
* :func:`hammer` -- N threads released together by a barrier.
* :func:`assert_gil_free` (row class FT-1) -- import a module in a clean
  subprocess under ``-W error::RuntimeWarning`` and require that the GIL is
  still off afterwards.
* :func:`measure_loss` / :func:`canary_python` / :func:`require_not_vacuous`
  (row class FT-2) -- a *planted* race that the schedule must expose. A
  schedule that cannot lose updates on a known-racy counter cannot certify
  that a real counter is race-free, so the run is RED as vacuous, never green.
  A package supplies its own compiled canary (conventionally
  ``_unsynchronised_bump``) through :func:`measure_loss`.

FT-3..FT-6 (global state, distinct objects, shared-object misuse, GPU) are
package-specific; packages build them from :func:`hammer` and register their
row ids with :func:`declare_ft_row` / :func:`register_ft_row`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import threading
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "FT_ROWS",
    "CanaryResult",
    "FtRowDecl",
    "assert_ft_rows_complete",
    "assert_gil_free",
    "canary_python",
    "declare_ft_row",
    "free_threaded_problem",
    "hammer",
    "measure_loss",
    "register_ft_row",
    "require_free_threaded",
    "require_not_vacuous",
]

#: Default hammer shape of the vacuity canary (threads x iterations per thread).
CANARY_THREADS = 8
CANARY_ITERATIONS = 200_000
#: A canary that loses less than this fraction of its updates is vacuous.
MIN_LOST_FRACTION = 0.10


def _is_free_threaded_build() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED")) and hasattr(
        sys, "_is_gil_enabled"
    )


def free_threaded_problem() -> str | None:
    """Return why this process cannot certify free-threading, or ``None``.

    Does not need pytest. ``None`` means: a free-threaded build with the GIL
    state left to the interpreter and the modules.
    """
    if not _is_free_threaded_build():
        return "not a free-threaded CPython build (GIL build)"
    return None


def _masking_reason() -> str | None:
    flag = getattr(sys.flags, "gil", None)
    if flag is not None:
        return f"-X gil={flag} is in force (sys.flags.gil={flag})"
    if "PYTHON_GIL" in os.environ:
        return f"PYTHON_GIL={os.environ['PYTHON_GIL']!r} is set"
    return None


def require_free_threaded() -> None:
    """Skip on a GIL build; FAIL when the GIL state is forced from outside.

    ``-X gil=0`` / ``PYTHON_GIL=0`` report the GIL as off even for an
    extension that never declared support, so they mask exactly the defect
    these rows exist to catch. ``=1`` is refused too: a run with the GIL forced
    on certifies nothing either.
    """
    import pytest

    if free_threaded_problem() is not None:
        pytest.skip(free_threaded_problem())
    reason = _masking_reason()
    if reason is not None:
        pytest.fail(
            f"free-threading gate refused: {reason}; this masks a missing "
            "free-threading declaration. Unset it and rerun."
        )


def hammer(
    fn: Callable[[], object],
    threads: int = CANARY_THREADS,
    iterations: int = CANARY_ITERATIONS,
) -> None:
    """Run ``fn()`` ``iterations`` times on each of ``threads`` threads.

    All threads wait on a barrier and start together. The first exception
    raised on any thread is re-raised here after every thread has finished.
    """
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            barrier.wait()
            for _ in range(iterations):
                fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    pool = [threading.Thread(target=work) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if errors:
        raise errors[0]


_GIL_PROBE = (
    "import importlib, sys\n"
    "importlib.import_module(sys.argv[1])\n"
    "print('GIL_ENABLED=' + str(sys._is_gil_enabled()))\n"
)


def assert_gil_free(
    module_name: str, python_args: tuple[str, ...] = (), timeout: float = 120.0
) -> None:
    """FT-1: importing ``module_name`` leaves the GIL disabled.

    The import runs in a fresh subprocess under ``-W error::RuntimeWarning``
    (the interpreter's GIL re-enable notice is a ``RuntimeWarning``, so an
    undeclared extension fails the import itself) and the child then reports
    ``sys._is_gil_enabled()``. A subprocess keeps the check clean: the parent
    may already have re-enabled the GIL through some other import.
    ``python_args`` is for tests of the check itself.
    """
    if _masking_reason() is not None:
        raise AssertionError(f"assert_gil_free refused: {_masking_reason()}")
    cmd = [
        sys.executable,
        *python_args,
        "-W",
        "error::RuntimeWarning",
        "-c",
        _GIL_PROBE,
        module_name,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    out = proc.stdout.strip()
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        raise AssertionError(
            f"FT-1: importing {module_name!r} failed under "
            f"-W error::RuntimeWarning: {tail[0]}"
        )
    if out != "GIL_ENABLED=False":
        raise AssertionError(
            f"FT-1: the GIL is enabled after importing {module_name!r} ({out})"
        )


@dataclass(frozen=True)
class CanaryResult:
    """Outcome of one planted-race measurement."""

    expected: int
    observed: int

    @property
    def lost(self) -> int:
        return self.expected - self.observed

    @property
    def lost_fraction(self) -> float:
        return self.lost / self.expected if self.expected else 0.0


def measure_loss(
    bump: Callable[[], object],
    read: Callable[[], int],
    threads: int = CANARY_THREADS,
    iterations: int = CANARY_ITERATIONS,
) -> CanaryResult:
    """Hammer a deliberately unsynchronised ``bump`` and count lost updates.

    ``read`` returns the counter ``bump`` increments. A package's compiled
    canary plugs in here: ``measure_loss(core._unsynchronised_bump,
    core.canary_count)``.
    """
    before = read()
    hammer(bump, threads, iterations)
    return CanaryResult(threads * iterations, read() - before)


#: Busy-loop steps between the canary's read and its write. A bare
#: ``d[k] += 1`` races in a window a few bytecodes wide, so how much it loses
#: depends on interpreter speed and machine load (measured 3-84 % on one
#: machine); widening the window makes the planted race visible everywhere.
CANARY_GAP = 16


def canary_python(
    threads: int = CANARY_THREADS, iterations: int = CANARY_ITERATIONS,
    gap: int = CANARY_GAP,
) -> CanaryResult:
    """FT-2 canary: an unsynchronised read-modify-write of one shared value,
    with ``gap`` busy steps between the read and the write."""
    d = {"k": 0}

    def _unsynchronised_bump() -> None:
        v = d["k"]
        for _ in range(gap):
            pass
        d["k"] = v + 1

    return measure_loss(_unsynchronised_bump, lambda: d["k"], threads, iterations)


_PREFLIGHT: dict[str, CanaryResult] = {}


def require_not_vacuous(
    result: CanaryResult | None = None, min_lost: float = MIN_LOST_FRACTION
) -> CanaryResult:
    """Fail (RED, never skip) unless the canary lost at least ``min_lost``.

    With no argument the Python canary is measured once per process and
    cached. Pass a package's own :func:`measure_loss` result to certify its
    compiled canary. Call after :func:`require_free_threaded`.
    """
    import pytest

    if result is None:
        if "python" not in _PREFLIGHT:
            _PREFLIGHT["python"] = canary_python()
        result = _PREFLIGHT["python"]
    if result.lost_fraction < min_lost:
        pytest.fail(
            f"free-threading run is VACUOUS: the planted race lost only "
            f"{result.lost_fraction:.1%} of {result.expected} updates "
            f"(< {min_lost:.0%}); a schedule that cannot expose a known race "
            "cannot certify the real code. Raise the thread count."
        )
    return result


@dataclass(frozen=True)
class FtRowDecl:
    """One declared free-threading row (``owner_repo`` carries its test)."""

    owner_repo: str
    summary: str


#: row_id -> FtRowDecl. raptor-core owns the harness-level rows below;
#: packages add theirs with :func:`declare_ft_row`. Kept apart from
#: :data:`raptor.conformance.interop.ROWS` because these rows certify an
#: execution property, not an array-framework crossing.
FT_ROWS: dict[str, FtRowDecl] = {
    "FT-0-REFUSE-MASKED-GIL": FtRowDecl(
        "raptor", "the gate fails under -X gil=... and PYTHON_GIL"
    ),
    "FT-1-GIL-FREE-PURE": FtRowDecl(
        "raptor", "a pure-Python module imports with the GIL still off"
    ),
    "FT-1-GIL-REQUIRED-FAILS": FtRowDecl(
        "raptor", "FT-1 is red when the GIL is on after import"
    ),
    "FT-2-CANARY-PYTHON": FtRowDecl(
        "raptor", "the plain-Python planted race loses >= 10% of updates"
    ),
}

_COLLECTED_FT_ROWS: set[str] = set()


def declare_ft_row(row_id: str, owner_repo: str, summary: str) -> None:
    """Declare a package's FT row ahead of its test (idempotent for equal args)."""
    decl = FtRowDecl(owner_repo, summary)
    if FT_ROWS.setdefault(row_id, decl) != decl:
        raise ValueError(f"declare_ft_row: {row_id!r} already declared differently")


def register_ft_row(row_id: str):
    """Decorator marking a test as covering declared FT row ``row_id``."""
    if row_id not in FT_ROWS:
        raise KeyError(f"register_ft_row: {row_id!r} is not a declared FT row")

    def _decorate(fn):
        _COLLECTED_FT_ROWS.add(row_id)
        return fn

    return _decorate


def assert_ft_rows_complete(owner_repo: str) -> None:
    """Assert every FT row owned by ``owner_repo`` was collected."""
    declared = {r for r, d in FT_ROWS.items() if d.owner_repo == owner_repo}
    missing = declared - _COLLECTED_FT_ROWS
    assert not missing, (
        f"FT rows declared for {owner_repo!r} but never collected: "
        f"{sorted(missing)}"
    )
