# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""raptor's own conformance tests: the declaration is total and
self-consistent, and one test proves the roadmap-present direction goes RED.

Pure-Python: ``importlib.util.find_spec`` never imports, so this suite never
actually imports torch/cupy/jax/tensorflow even though it names them."""

from __future__ import annotations

import pytest

from raptor.conformance.interop import (
    CERTIFIED,
    ROADMAP,
    ROWS,
    UNIVERSE,
    RowDecl,
    roadmap_framework,
)


def test_declaration_never_imported_a_framework():
    import sys

    for fw in ("torch", "cupy", "warp", "jax", "tensorflow"):
        assert fw not in sys.modules, f"raptor imported {fw!r} merely by declaring it"


def test_certified_and_roadmap_cover_the_universe_exactly():
    assert set(CERTIFIED) | set(ROADMAP) == set(UNIVERSE)


def test_certified_and_roadmap_are_disjoint():
    assert not (set(CERTIFIED) & set(ROADMAP))


def test_every_row_framework_is_certified():
    """Every row's ``framework`` is either a certified array framework, or ``None``
    for a row that bridges to no framework at all (the ``execution_*`` rows — an
    eagle-internal execution-structure invariant, see RowDecl's docstring)."""
    assert all(
        row.framework is None or row.framework in CERTIFIED for row in ROWS.values()
    )


def test_every_row_has_an_owner_repo():
    assert all(row.owner_repo for row in ROWS.values())


def test_rows_are_populated_and_owned_by_expected_repos():
    """A declaration that silently empties itself must go RED.

    ``ROWS`` is PRE-DECLARED in full precisely so
    :func:`raptor.conformance.interop.assert_rows_complete` has a genuine gap
    to catch later, once each owner starts collecting against it —
    that only works if the catalogue never quietly reverts to empty."""
    assert ROWS, "ROWS must not be empty — the row catalogue is pre-declared"
    assert set(row.owner_repo for row in ROWS.values()) <= {
        "eagle",
        "learning_runtime",
        "hawk",
    }


def test_t_in_cpu_host_alias_row_declared_exactly():
    """The NEW row, pinned exactly — an explicit re-declaration, never a
    quiet one. ``T-IN-CPU-COPY`` (eagle's OWN, untouched adapter-crossing
    row) must still be present and unchanged alongside it — this is a NEW
    row, not a flip of that one."""
    assert ROWS["T-IN-CPU-COPY"] == RowDecl("torch", "eagle", "COPY")
    assert len(ROWS) == 23, sorted(ROWS)


def test_warp_rows_declared_exactly():
    """The three NEW Warp rows, pinned exactly — an explicit re-declaration,
    never a quiet one (mirrors ``test_t_in_cpu_host_alias_row_declared_exactly``
    above). Warp crosses through eagle's generic ``import_buffer`` layer, not a
    dedicated Adapter, so all three are owner=eagle."""
    assert ROWS["WP-IN-CUDA-ALIAS"] == RowDecl("warp", "eagle", "ALIAS")
    assert ROWS["WP-OUT-CUDA-ALIAS"] == RowDecl("warp", "eagle", "ALIAS")
    assert ROWS["STREAM-WARP-PRODUCER-ORDER"] == RowDecl("warp", "eagle", "STREAM")


def test_execution_rows_declared_exactly():
    """The four eagle-owned execution rows, pinned exactly — an explicit
    re-declaration, never a quiet one (mirrors
    ``test_t_in_cpu_host_alias_row_declared_exactly`` above). ``framework=None``:
    these certify an eagle-internal execution-structure invariant, not a crossing to
    an array framework (see RowDecl's docstring)."""
    assert ROWS["execution_partition_identity"] == RowDecl(None, "eagle", "INVARIANCE")
    assert ROWS["execution_host_device_twin"] == RowDecl(None, "eagle", "INVARIANCE")
    assert ROWS["execution_illegal_placement_refused"] == RowDecl(None, "eagle", "ENV")
    assert ROWS["execution_layout_selfcheck_refused"] == RowDecl(None, "eagle", "ENV")


def test_hawk_rows_declared_exactly():
    """HAWK's own four rows, pinned exactly (mirrors
    ``test_execution_rows_declared_exactly`` above). These are the rows that
    exist only BECAUSE HAWK exists — the HAWK-DIFF-POSITIVE-ROUNDTRIP row and the
    execution-contract deployment/oracle rows — declared here ahead
    of HAWK's own consumer implementation, which collects against them."""
    assert ROWS["HAWK-DIFF-POSITIVE-ROUNDTRIP"] == RowDecl(None, "hawk", "INVARIANCE")
    assert ROWS["HAWK-EXEC-DEPLOY-LOADABLE"] == RowDecl(None, "hawk", "ENV")
    assert ROWS["HAWK-EXEC-REFERENCE-HOST"] == RowDecl(None, "hawk", "INVARIANCE")
    assert ROWS["HAWK-EXEC-REFERENCE-RANKPARTITION"] == RowDecl(
        None, "hawk", "INVARIANCE"
    )


def test_roadmap_framework_raises_when_framework_is_present():
    """Calling the roadmap-absence assertion with a module that IS
    present (pytest, which every run of this suite has) must raise — proving
    the present-but-roadmap direction REDs without touching the env."""
    with pytest.raises(AssertionError):
        roadmap_framework("pytest")
