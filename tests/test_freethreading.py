# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Acceptance for the free-threading harness (row classes FT-0, FT-1, FT-2).

On a GIL build every row SKIPS with a reason; the name manifest below keeps
"skipped" from ever degrading into "silently not collected".
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from raptor.conformance import freethreading as ft
from raptor.conformance.freethreading import register_ft_row

pytestmark = pytest.mark.ft

EXPECTED_TESTS = frozenset(
    {
        "test_manifest_all_tests_collected",
        "test_rows_declared_and_collected",
        "test_gate_refuses_masked_gil_x_option",
        "test_gate_refuses_masked_gil_env",
        "test_gil_free_pure_python_module",
        "test_gil_required_case_fails",
        "test_python_canary_loses_updates",
        "test_vacuous_canary_is_red",
        "test_hammer_runs_every_iteration_and_propagates_errors",
    }
)

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _child_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    env["PYTHONPATH"] = _REPO_ROOT
    env.update(extra)
    return env


_REFUSE = (
    "from raptor.conformance.freethreading import require_free_threaded;"
    "require_free_threaded()"
)


def test_manifest_all_tests_collected(request):
    here = {
        i.name for i in request.session.items if i.module is sys.modules[__name__]
    }
    assert EXPECTED_TESTS <= here, sorted(EXPECTED_TESTS - here)
    assert {n for n in here if "[" not in n} <= EXPECTED_TESTS


def test_rows_declared_and_collected():
    ft.assert_ft_rows_complete("raptor")


def test_hammer_runs_every_iteration_and_propagates_errors():
    lock_free = []
    ft.hammer(lambda: lock_free.append(1), threads=4, iterations=100)
    assert len(lock_free) == 400  # list.append is atomic on every build

    def boom():
        raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        ft.hammer(boom, threads=2, iterations=1)


@register_ft_row("FT-0-REFUSE-MASKED-GIL")
def test_gate_refuses_masked_gil_x_option():
    ft.require_free_threaded()
    for val in ("0", "1"):
        proc = subprocess.run(
            [sys.executable, "-X", f"gil={val}", "-c", _REFUSE],
            capture_output=True, text=True, env=_child_env(), check=False,
        )
        assert proc.returncode != 0 and "gate refused" in proc.stderr, proc.stderr


@register_ft_row("FT-0-REFUSE-MASKED-GIL")
def test_gate_refuses_masked_gil_env():
    ft.require_free_threaded()
    for val in ("0", "1"):
        proc = subprocess.run(
            [sys.executable, "-c", _REFUSE],
            capture_output=True, text=True,
            env=_child_env(PYTHON_GIL=val), check=False,
        )
        assert proc.returncode != 0 and "gate refused" in proc.stderr, proc.stderr


@register_ft_row("FT-1-GIL-FREE-PURE")
def test_gil_free_pure_python_module():
    ft.require_free_threaded()
    ft.assert_gil_free("raptor.conformance.freethreading")


@register_ft_row("FT-1-GIL-REQUIRED-FAILS")
def test_gil_required_case_fails():
    # Stand-in for an extension that re-enables the GIL: the child runs with
    # the GIL forced on, so the check's own verdict must be RED.
    ft.require_free_threaded()
    with pytest.raises(AssertionError, match="GIL is enabled"):
        ft.assert_gil_free("raptor.conformance.freethreading", ("-X", "gil=1"))


@register_ft_row("FT-2-CANARY-PYTHON")
def test_python_canary_loses_updates():
    ft.require_free_threaded()
    res = ft.require_not_vacuous()
    assert res.lost_fraction >= ft.MIN_LOST_FRACTION
    print(f"python canary lost {res.lost_fraction:.1%} of {res.expected}")


def test_vacuous_canary_is_red():
    ft.require_free_threaded()
    with pytest.raises(pytest.fail.Exception, match="VACUOUS"):
        ft.require_not_vacuous(ft.CanaryResult(1000, 1000))
