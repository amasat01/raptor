# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``assert_rows_complete(owner, ran=True)`` counts a row only if its test ran.

Each case runs a tiny suite of its own in a subprocess: a row declared for a
made-up owner, one test per leg, and a gate test that calls
``assert_rows_complete``. The old behaviour (``ran=False``: a row counts once
its module is imported) is the control: it passes the runs that executed
nothing, and these tests fail if the new default ever does.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import raptor

#: where ``import raptor`` resolves to, so the child imports the same one.
RAPTOR_ROOT = Path(raptor.__file__).resolve().parent.parent

_SUITE = textwrap.dedent('''
    import pytest
    from raptor.conformance.interop import ROWS, RowDecl, assert_rows_complete, register_row

    ROWS["FAKE-ROW"] = RowDecl(None, "fake", "INVARIANCE", ("cpu", "cuda"))

    @pytest.mark.rows_gate
    def test_gate():
        assert_rows_complete("fake", ran=RAN)

    @register_row("FAKE-ROW", leg="cpu")
    def test_cpu_leg():
        {cpu}

    @register_row("FAKE-ROW", leg="cuda")
    def test_cuda_leg():
        {cuda}
''')


def _run(tmp_path, *args, cpu="pass", cuda="pass", ran=True, plugin=True):
    (tmp_path / "test_suite.py").write_text(
        _SUITE.format(cpu=cpu, cuda=cuda).replace("RAN", repr(ran))
    )
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
           "--rootdir", str(tmp_path), *(["-p", "raptor.conformance.pytest_rows"] if plugin else []),
           *args, str(tmp_path / "test_suite.py")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(RAPTOR_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]))
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120)
    return run.returncode, run.stdout + run.stderr


def test_every_leg_ran_and_passed_is_green(tmp_path):
    rc, out = _run(tmp_path)
    assert rc == 0, out


def test_the_gate_runs_last_wherever_it_sits_in_the_file(tmp_path):
    """The gate is the FIRST test in the suite above; it still sees both legs."""
    rc, out = _run(tmp_path, "-v")
    assert rc == 0, out
    order = [line for line in out.splitlines() if "PASSED" in line]
    assert order[-1].split("::")[1].startswith("test_gate"), order


@pytest.mark.parametrize("deselect", ["test_cpu_leg", "test_cuda_leg"])
def test_a_deselected_leg_is_red_but_the_import_time_check_stays_green(tmp_path, deselect):
    """The old check passes a run that deselected a leg; the new one does not."""
    args = ("--deselect", f"test_suite.py::{deselect}")
    old_rc, old = _run(tmp_path, *args, ran=False)
    assert old_rc == 0, old
    new_rc, new = _run(tmp_path, *args)
    assert new_rc != 0 and "without a passing run" in new, new
    assert "FAKE-ROW" in new and deselect in new, new


def test_a_run_that_executed_only_the_gate_is_red(tmp_path):
    """Everything is imported and registered, nothing ran: the vacuous run."""
    old_rc, old = _run(tmp_path, "-k", "gate", ran=False)
    assert old_rc == 0 and "1 passed" in old, old
    new_rc, new = _run(tmp_path, "-k", "gate")
    assert new_rc != 0 and "1 failed" in new, new


@pytest.mark.parametrize("body", ["assert False", "pytest.skip('no')"])
def test_a_failed_or_skipped_leg_is_red(tmp_path, body):
    rc, out = _run(tmp_path, cuda=body)
    assert rc != 0 and "FAKE-ROW [cuda]" in out and "FAKE-ROW [cpu]" not in out, out


def test_without_the_plugin_ran_true_says_so(tmp_path):
    rc, out = _run(tmp_path, plugin=False)
    assert rc != 0 and "pytest_rows is not loaded" in out, out
