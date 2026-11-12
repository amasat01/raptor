# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The test-side gates of the free-threading rows.

:mod:`raptor.conformance.freethreading` names no test framework: it returns the reason a gate refuses (``None`` when it
passes). These two functions turn that reason into the outcome: a skip on a GIL build, a failure when the GIL state is
forced from outside or the canary is vacuous (RED, never a skip).
"""

from __future__ import annotations

import pytest

from raptor.conformance import freethreading as ft


def require_free_threaded() -> None:
    """Skip on a GIL build; FAIL when the GIL state is forced from outside."""
    problem = ft.free_threaded_problem()
    if problem is not None:
        pytest.skip(problem)
    reason = ft.masking_problem()
    if reason is not None:
        pytest.fail(
            f"free-threading gate refused: {reason}; this masks a missing "
            "free-threading declaration. Unset it and rerun."
        )


def require_not_vacuous(result=None) -> ft.CanaryResult:
    """FAIL (RED, never skip) unless the canary lost at least ``min_lost``; call after :func:`require_free_threaded`."""
    res, problem = ft.not_vacuous_problem() if result is None else ft.not_vacuous_problem(result)
    if problem is not None:
        pytest.fail(problem)
    return res
