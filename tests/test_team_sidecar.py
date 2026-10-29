# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Sidecar fields of team stages, index sets and local memory: the JSON schema and the Python checks agree."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from raptor.schema import validate_manifest

# Reads docs/schemas and goldens from the repository; the wheel carries neither.
pytestmark = pytest.mark.repo_local

try:
    import jsonschema
except ImportError:
    jsonschema = None

REPO = Path(__file__).resolve().parents[1]
_SCHEMA_FILE = REPO / "docs/schemas/sidecar-v1.schema.json"
SCHEMA = json.loads(_SCHEMA_FILE.read_text()) if _SCHEMA_FILE.exists() else {}
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA) if jsonschema else None


@pytest.fixture(autouse=True)
def _needs_jsonschema():
    if jsonschema is None:
        pytest.skip("jsonschema is not installed")


def _load(name):
    return json.loads((REPO / "goldens" / f"hawk_{name}.sidecar.json").read_text())


FEATURES = ["box", "team_lu", "team_gram", "scatter_ordered", "index_prefix", "index_map64"]
KEYED = {"width": 576, "axes": [["key", None]], "payload": [24, 24]}


@pytest.mark.parametrize("name", FEATURES)
def test_a_hawk_sidecar_of_each_feature_is_valid_in_both(name):
    doc = _load(name)
    assert not list(VALIDATOR.iter_errors(doc))
    validate_manifest(doc)


def test_the_goldens_carry_the_fields():
    assert _load("team_lu")["launch"]["local_source"] in ("ptx", "ptxas")
    assert _load("team_gram")["stages"][1]["team"]["grid"] == "keys"
    assert _load("scatter_ordered")["team_planes"]["tms_acc_m"]["once"] is True
    assert _load("index_prefix")["launch"]["index_set"]["prefix"] is True
    assert _load("index_map64")["launch"]["index_set"] == {"width": 64}


def _set(path, value):
    def edit(d):
        node = d
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return edit


def _drop(path):
    def edit(d):
        node = d
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return edit


# (golden, edit, a word the Python message must name)
BAD = {
    "box_no_stride": ("box", _drop(["launch", "box", "stride"]), "box"),
    "box_T_one": ("box", _set(["launch", "box", "T"], 1), "box"),
    "box_axis_negative": ("box", _set(["launch", "box", "axis"], -1), "box"),
    "box_stride_zero": ("box", _set(["launch", "box", "stride"], 0), "box"),
    "box_T_string": ("box", _set(["launch", "box", "T"], "8"), "box"),
    "box_extra_key": ("box", _set(["launch", "box", "block"], 4), "box"),
    "index_set_width": ("index_map64", _set(["launch", "index_set", "width"], 16), "index_set"),
    "index_set_extra": ("index_map64", _set(["launch", "index_set", "stride"], 1), "index_set"),
    "index_set_prefix_not_bool": ("index_prefix", _set(["launch", "index_set", "prefix"], 1), "index_set"),
    "index_set_prefix_no_count": ("index_prefix", _drop(["launch", "index_set", "count"]), "index_set"),
    "index_set_prefix_width_64": ("index_prefix", _set(["launch", "index_set", "width"], 64), "index_set"),
    "index_set_count_without_prefix": ("index_map64", _set(["launch", "index_set", "count"], "n"), "index_set"),
    "local_bytes_negative": ("team_lu", _set(["launch", "local_bytes"], -1), "local_bytes"),
    "local_bytes_string": ("team_lu", _set(["launch", "local_bytes"], "0"), "local_bytes"),
    "local_source": ("team_lu", _set(["launch", "local_source"], "nvcc"), "local_source"),
    "local_limit_zero": ("team_lu", _set(["launch", "local_limit"], 0), "local_limit"),
    "local_limit_missing": ("team_lu", _drop(["launch", "local_limit"]), "local_bytes"),
    "team_threads_zero": ("team_lu", _set(["stages", 1, "team", "threads"], 0), "threads"),
    "team_threads_missing": ("team_lu", _drop(["stages", 1, "team", "threads"]), "stages[1].team"),
    "team_extra_key": ("team_lu", _set(["stages", 1, "team", "width"], 4), "stages[1].team"),
    "team_grid_unknown": ("team_gram", _set(["stages", 1, "team", "grid"], "rows"), "grid"),
    "team_keys_without_keyed": ("team_gram", _drop(["stages", 1, "team", "keyed"]), "keyed"),
    "team_keyed_payload_zero": ("team_gram", _set(["stages", 1, "team", "keyed", "payload"], [0]), "payload"),
    "team_keyed_extra": ("team_gram", _set(["stages", 1, "team", "keyed", "rank"], 2), "keyed"),
    "team_chunks_without_status": ("scatter_ordered", _drop(["stages", 2, "team", "status"]), "status"),
    "team_targets_with_status": ("scatter_ordered", _set(["stages", 1, "team", "status"], "m"), "stages[1].team"),
    "team_lanes_with_of": ("team_lu", _set(["stages", 1, "team", "of"], "acc"), "stages[1].team"),
    "plane_dtype": ("team_lu", _set(["team_planes", "tm0_a", "dtype"], "float"), "dtype"),
    "plane_per_lane_zero": ("team_lu", _set(["team_planes", "tm0_a", "per_lane"], 0), "per_lane"),
    "plane_extra": ("team_lu", _set(["team_planes", "tm0_a", "unit"], "lane"), "tm0_a"),
    "plane_fill_float": ("scatter_ordered", _set(["team_planes", "tms_acc_k", "fill"], 0.5), "fill"),
    "plane_once_false": ("scatter_ordered", _set(["team_planes", "tms_acc_m", "once"], False), "once"),
    "plane_two_sizes": ("scatter_ordered", _set(["team_planes", "tms_acc_m", "per_target"], "acc"), "tms_acc_m"),
    "plane_keys_bad": ("scatter_ordered", _set(["team_planes", "tms_acc_p", "keys"], {"width": 0, "axes": []}), "tms_acc_p"),
    "keyed_bad": ("team_gram", _set(["keyed"], {"g": {"width": 4, "axes": [["a"]]}}), "keyed"),
    "index_planes_empty": ("box", _set(["index_planes"], []), "index_planes"),
    "index_planes_twice": ("box", _set(["index_planes"], ["x", "x"]), "index_planes"),
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_each_malformed_field_is_refused_by_name_and_by_the_schema(case):
    name, edit, word = BAD[case]
    doc = copy.deepcopy(_load(name))
    edit(doc)
    assert list(VALIDATOR.iter_errors(doc)), "the schema accepted it"
    with pytest.raises(ValueError, match=word.replace("[", r"\[").replace("]", r"\]")):
        validate_manifest(doc)


def test_a_keys_sized_plane_and_a_keyed_block_are_valid_in_both():
    doc = copy.deepcopy(_load("team_gram"))
    doc["team_planes"]["tmb0_u"]["keys"] = KEYED
    doc["keyed"] = {"tmb0_g": KEYED, "old": {"width": 3, "axes": []}}     # payload absent: older sidecar
    assert not list(VALIDATOR.iter_errors(doc))
    validate_manifest(doc)


def test_index_planes_name_declared_planes_and_refuse_an_undeclared_one():
    doc = copy.deepcopy(_load("box"))
    name = doc["arg_spec"][0][1]
    doc["index_planes"] = [name]
    assert not list(VALIDATOR.iter_errors(doc))
    validate_manifest(doc)
    doc["index_planes"] = ["not_a_plane"]
    with pytest.raises(ValueError, match="index_planes names"):
        validate_manifest(doc)
