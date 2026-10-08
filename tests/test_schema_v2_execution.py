# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The schema-v2 execution axis.

Enumeration, not sampling: one valid v2 manifest per (access class x targets
combo), every individual `exec_op` value proven valid on its own, and one case
per invalid arm the acceptance line names by name (missing each required key,
unknown target, duplicate target, empty targets, unknown access, mapreduce
without op, op with non-mapreduce, unknown op, v1 with an exec key, v2 with
aether-abi/1, v1 with aether-abi/2).

``raptor.schema.validate_manifest`` is the entry point under test throughout
(not the lower-level ``manifest.check_execution_axis`` directly), so these
tests exercise the SAME call path a real producer/consumer uses.
"""

from __future__ import annotations

import pytest

from raptor.schema import validate_manifest
from raptor.schema.manifest import (
    AETHER_ABI_FIELD,
    AETHER_ABI_V1,
    AETHER_ABI_V2,
    EXEC_ACCESS_CLASSES,
    EXEC_OPS,
    MAX_SCHEMA_VERSION,
)

#: every non-empty, duplicate-free combination of the two exec_targets values.
_TARGET_COMBOS = (["device"], ["host"], ["device", "host"])


def _v2(**overrides) -> dict:
    """A minimal, otherwise-valid schema-v2 document (no `pattern`/`plugins` —
    those are validated independently and their absence keeps the key-order
    check, which requires `plugins`, out of these cases' way)."""
    doc = {
        "schema_version": 2,
        AETHER_ABI_FIELD: AETHER_ABI_V2,
        "exec_targets": ["device"],
        "exec_access": "sample_local",
    }
    doc.update(overrides)
    return doc


# --------------------------------------------------------------------------- #
# valid: enumeration of access class x targets combo, and every op on its own.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("access", sorted(EXEC_ACCESS_CLASSES))
@pytest.mark.parametrize("targets", _TARGET_COMBOS, ids=lambda t: "+".join(t))
def test_valid_v2_manifest_per_access_and_targets(access, targets):
    overrides = {"exec_targets": list(targets), "exec_access": access}
    if access == "mapreduce":
        overrides["exec_op"] = "sum"
    validate_manifest(_v2(**overrides))  # must not raise


@pytest.mark.parametrize("op", sorted(EXEC_OPS))
def test_every_mapreduce_op_is_individually_valid(op):
    validate_manifest(_v2(exec_access="mapreduce", exec_op=op))  # must not raise


def test_valid_v2_manifest_with_plugins_is_key_order_checked():
    """Full manifest shape, keys already in TOP_LEVEL_KEY_ORDER_V2 order —
    exercises the v2 key-order wiring (the `if "plugins" in doc` branch)."""
    doc = {
        "schema_version": 2,
        "pattern": "vector",
        AETHER_ABI_FIELD: AETHER_ABI_V2,
        "exec_targets": ["device"],
        "exec_access": "sample_local",
        "plugins": [],
    }
    validate_manifest(doc)  # must not raise


def test_valid_v1_manifest_with_plugins_still_loads():
    doc = {
        "schema_version": 1,
        "pattern": "vector",
        AETHER_ABI_FIELD: AETHER_ABI_V1,
        "plugins": [],
    }
    validate_manifest(doc)  # must not raise — v1 untouched by the v2 axis


def test_v1_manifest_with_no_schema_version_key_still_loads():
    validate_manifest({"pattern": "pure"})  # backward-lenient absence == v1


# --------------------------------------------------------------------------- #
# invalid: one case per named arm.
# --------------------------------------------------------------------------- #


def test_v2_missing_exec_targets():
    doc = _v2()
    del doc["exec_targets"]
    with pytest.raises(ValueError, match="exec_targets"):
        validate_manifest(doc)


def test_v2_missing_exec_access():
    doc = _v2()
    del doc["exec_access"]
    with pytest.raises(ValueError, match="exec_access"):
        validate_manifest(doc)


def test_v2_unknown_target():
    with pytest.raises(ValueError, match="exec_targets"):
        validate_manifest(_v2(exec_targets=["device", "cloud"]))


def test_v2_duplicate_target():
    with pytest.raises(ValueError, match="exec_targets"):
        validate_manifest(_v2(exec_targets=["device", "device"]))


def test_v2_empty_targets():
    with pytest.raises(ValueError, match="exec_targets"):
        validate_manifest(_v2(exec_targets=[]))


def test_v2_unknown_access():
    with pytest.raises(ValueError, match="exec_access"):
        validate_manifest(_v2(exec_access="bogus"))


def test_v2_mapreduce_without_op():
    with pytest.raises(ValueError, match="exec_op"):
        validate_manifest(_v2(exec_access="mapreduce"))


def test_v2_op_with_non_mapreduce():
    with pytest.raises(ValueError, match="exec_op"):
        validate_manifest(_v2(exec_access="sample_local", exec_op="sum"))


def test_v2_unknown_op():
    with pytest.raises(ValueError, match="exec_op"):
        validate_manifest(_v2(exec_access="mapreduce", exec_op="bogus"))


def test_v1_with_an_exec_key():
    doc = {
        "schema_version": 1,
        AETHER_ABI_FIELD: AETHER_ABI_V1,
        "exec_targets": ["device"],
    }
    with pytest.raises(ValueError, match="exec_targets"):
        validate_manifest(doc)


def test_v2_with_aether_abi_v1():
    with pytest.raises(ValueError, match=AETHER_ABI_FIELD):
        validate_manifest(_v2(**{AETHER_ABI_FIELD: AETHER_ABI_V1}))


def test_v1_with_aether_abi_v2():
    doc = {"schema_version": 1, AETHER_ABI_FIELD: AETHER_ABI_V2}
    with pytest.raises(ValueError, match=AETHER_ABI_FIELD):
        validate_manifest(doc)


def test_schema_version_3_is_still_rejected():
    """The transition bridge lifts the ceiling to v2, not to "anything" — v3+
    stays forward-strict refused."""
    with pytest.raises(ValueError, match=f"v{MAX_SCHEMA_VERSION}"):
        validate_manifest({"schema_version": 3})


def test_v2_key_order_violation_is_rejected():
    """`plugins` listed before the exec_* keys violates TOP_LEVEL_KEY_ORDER_V2."""
    doc = {
        "schema_version": 2,
        "pattern": "vector",
        AETHER_ABI_FIELD: AETHER_ABI_V2,
        "plugins": [],
        "exec_targets": ["device"],
        "exec_access": "sample_local",
    }
    with pytest.raises(ValueError, match="key order"):
        validate_manifest(doc)
