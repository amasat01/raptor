# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The sidecar JSON schema and the Python mirror agree: v1 and v3 goldens, lane space."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from raptor.schema import manifest, validate_manifest

# Reads docs/schemas and goldens from the repository; the wheel carries neither. jsonschema is a
# dev dependency, so the module must import without it: the wheel leg deselects these rows.
pytestmark = pytest.mark.repo_local

try:
    import jsonschema
except ImportError:
    jsonschema = None

REPO = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((REPO / "docs/schemas/sidecar-v1.schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA) if jsonschema else None


@pytest.fixture(autouse=True)
def _needs_jsonschema():
    if jsonschema is None:
        pytest.skip("jsonschema is not installed")
GOLD = json.loads((REPO / "goldens/vector/gravity.json").read_text())
SIDECARS = sorted(
    p
    for p in (REPO / "goldens").rglob("*.json")
    if "plugins" not in json.loads(p.read_text())
    and json.loads(p.read_text()).get("pattern") in manifest.BASE_PATTERNS
)

SPACE = {
    "axes": [
        {"label": None, "extent": None},
        {"label": "channel", "extent": 64},
        {"label": "sample", "extent": 32},
    ],
    "group": 32,
}
BAD = {
    "extra_key": {**SPACE, "stride": 1},
    "extra_axis_key": {"axes": [{"label": "a", "extent": 4, "role": "x"}], "group": 1},
    "group_zero": {**SPACE, "group": 0},
    "null_extent_not_first": {
        "axes": [{"label": "a", "extent": 4}, {"label": "b", "extent": None}],
        "group": 1,
    },
    "duplicate_label": {
        "axes": [{"label": "a", "extent": 4}, {"label": "a", "extent": 2}],
        "group": 1,
    },
    "axes_not_a_list": {"axes": {"label": "a", "extent": 4}, "group": 1},
}


def _with(space):
    d = copy.deepcopy(GOLD)
    d["launch"] = {"lane_space": space}
    return d


def test_golden_sidecars_validate_against_the_schema():
    assert len(SIDECARS) >= 2
    for path in SIDECARS:
        doc = json.loads(path.read_text())
        assert not list(VALIDATOR.iter_errors(doc)), path
        validate_manifest(doc)


def test_built_sidecars_are_among_the_goldens():
    names = {p.name for p in SIDECARS}
    assert {"hawk_flat.sidecar.json", "hawk_grid.sidecar.json", "hawk_box.sidecar.json"} <= names


V3 = json.loads((REPO / "goldens/hawk_flat.sidecar.json").read_text())


@pytest.mark.parametrize(
    "edit",
    [
        {"schema_version": 4},
        {"aether_abi": "aether-abi/1"},
        {"exec_access": "mapreduce"},
        {"exec_op": "sum"},
        {"exec_targets": []},
        {"exec_targets": ["host", "host"]},
    ],
    ids=["version", "abi_tag", "mapreduce_no_op", "op_without_mapreduce", "no_targets", "dup_targets"],
)
def test_v3_negatives_refused_both(edit):
    doc = {**V3, **edit}
    with pytest.raises(ValueError):
        validate_manifest(doc)
    assert list(VALIDATOR.iter_errors(doc))


def test_v1_must_not_carry_the_execution_axis():
    doc = {**GOLD, "exec_access": "sample_local"}
    with pytest.raises(ValueError):
        validate_manifest(doc)
    assert list(VALIDATOR.iter_errors(doc))


@pytest.mark.parametrize(
    "space",
    [
        SPACE,
        {"axes": [{"label": "x", "extent": 8}], "group": 1},
        {"axes": [{"label": None, "extent": 8}, {"label": None, "extent": 2}], "group": 2},
    ],
)
def test_lane_space_valid_both(space):
    validate_manifest(_with(space))
    assert not list(VALIDATOR.iter_errors(_with(space)))


@pytest.mark.parametrize("key", sorted(BAD))
def test_lane_space_negatives_refused_both(key):
    with pytest.raises(ValueError, match="launch.lane_space"):
        validate_manifest(_with(BAD[key]))
    if key != "duplicate_label":  # JSON Schema cannot express unique labels
        assert list(VALIDATOR.iter_errors(_with(BAD[key])))


@pytest.mark.parametrize("block", [0, "64", 1.5])
def test_launch_block_is_a_positive_integer(block):
    d = copy.deepcopy(GOLD)
    d["launch"] = {"block": block}
    assert list(VALIDATOR.iter_errors(d))


@pytest.mark.parametrize("box", [
    {"axis": 0, "T": 16},
    {"axis": 0, "T": 1, "stride": 256},
    {"axis": -1, "T": 8, "stride": 1},
    {"axis": 0, "T": 8, "stride": 0},
    {"axis": 0, "T": "8", "stride": 1},
    {"axis": 0, "T": 8, "stride": 1, "block": 4},
], ids=["no_stride", "T_one", "axis_negative", "stride_zero", "T_string", "extra_key"])
def test_launch_box_is_three_integers(box):
    d = copy.deepcopy(GOLD)
    d["launch"] = {"box": box}
    assert list(VALIDATOR.iter_errors(d))


def test_launch_box_golden_is_valid():
    d = json.loads((REPO / "goldens/hawk_box.sidecar.json").read_text())
    assert set(d["launch"]["box"]) == {"axis", "T", "stride"}
    assert not list(VALIDATOR.iter_errors(d))
    validate_manifest(d)
