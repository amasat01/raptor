# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The bird-neutral (neutral across the family's own eagle/hawk/raptor
packages) kernel-manifest schema — the layer hawk touches.

This module MUST stay clean of the extension family's vocabulary (declared in
:mod:`raptor.schema.blocks`, the sibling module this file never imports — the
purity direction is one-way: ``blocks`` may reference ``manifest``, never the
reverse). This is what lets hawk's writer bind against this module alone and
keep its own core-purity gate green.

Mirrors ``eagle/python/eagle/roles.py`` (the Python half of the schema-v1
protocol) for the bird-neutral subset of its vocabulary, and the C++ twin
``eagle/plugin/roles.h`` / ``eagle/plugin/plugin_registry/manifest.h``.

NOTE on the top-level key order: the full order for the EXTENDED manifest
kind (see :mod:`raptor.schema.blocks`) is ``schema_version, pattern, aether_abi,
blocks, plugins`` — spelling the literal ``blocks`` here would tie this
neutral module to the extension's carriage key, so the FULL order is housed
in :mod:`raptor.schema.blocks` instead; :data:`TOP_LEVEL_KEY_ORDER` below is
the neutral 4-key order for an ordinary (unextended) manifest.

A toy manifest document, just enough to pass :func:`check_schema_version`
and :func:`check_execution_axis`::

    doc = {
        "schema_version": 1,
        "pattern": "pure",
        "aether_abi": "aether-abi/1",
        "plugins": [],
    }
    check_schema_version(doc)        # -> 1
    check_execution_axis(doc, 1)     # -> None (no exec keys allowed at v1)
"""

from __future__ import annotations

#: raptor's own plugin-schema version — the schema triad's number of record.
#: ``eagle.roles.SCHEMA_VERSION`` derives from this value directly (an import,
#: not a copy); the C++ half (``plugin/roles.h::kPluginSchemaVersion``) stays
#: a literal, cross-checked against the derived Python value by
#: ``test_cpp_and_python_schema_version_identical`` in eagle's
#: ``python/tests/test_roles_vocab.py``.
#: hawk's own contracts module carries a second, SANCTIONED literal copy (a
#: severing law forbids it importing raptor) — guarded by eagle's own
#: structural AST scan and hawk's own import check.
SCHEMA_VERSION = 1

#: the highest ``schema_version`` this loader ACCEPTS today — a TRANSITION
#: BRIDGE: raptor accepts SCHEMA_VERSION 1, 2 AND 3 on the aether branches.
#: v2 added the execution axis; v3 is v2 with ``cells`` replacing
#: ``sample_sums`` in the sidecar (the record of reductions over samples,
#: ``{name: {op, columns}}``); the manifest shape and ABI tag are v2's.
#: Deliberately a SEPARATE constant from :data:`SCHEMA_VERSION`, not a bump of
#: it: :data:`SCHEMA_VERSION` is the cross-repo FOUR-WAY PIN against eagle's
#: C++ ``kPluginSchemaVersion`` literal and hawk's contracts-module literal
#: copy — both still ``1`` today (the twin-site bump across all four repos is
#: a separate, later work item, not this raptor-only change). Bumping
#: :data:`SCHEMA_VERSION` here would break that pin against repos this change
#: does not touch, for no benefit: :func:`check_schema_version`'s
#: forward-strict ceiling is the only thing that needs to move to let v2 load.
MAX_SCHEMA_VERSION = 3

#: entry-point symbol of every kernel artifact, generated or hand-written
#: (verbatim value of ``eagle.launch.KERNEL_NAME``).
KERNEL_NAME = "raptor_kernel"

#: the FIELD name carrying the AETHER ABI tag on a manifest/sidecar. The VALUE
#: (e.g. ``"aether-abi/1"``) stays eagle-owned; raptor only names the field.
AETHER_ABI_FIELD = "aether_abi"

#: the ``aether_abi`` WIRE VALUE a schema-v1 document must carry, when the field is
#: present (schema v1 does not require the field at all — see
#: :func:`check_execution_axis`). Named here, newly, only so the shape validator can
#: catch a version/tag mismatch; this is still not the deep ABI-tag-vs-symbol
#: agreement check, which stays eagle's loader (``eagle_abi_tag`` dlsym read).
AETHER_ABI_V1 = "aether-abi/1"

#: the ``aether_abi`` WIRE VALUE a schema-v2 document must carry, when the field is
#: present. See :data:`AETHER_ABI_V1`.
AETHER_ABI_V2 = "aether-abi/2"

#: the ``exec_targets`` vocabulary a schema-v2 manifest may name (the two
#: structural entry points device kernels and host teams launch through). MPI
#: ranks and NCCL device groups are not distinct target VALUES — they are
#: execution STRUCTURES eagle's planner may choose for an existing target, so
#: they never appear here.
EXEC_TARGETS = frozenset({"device", "host"})

#: the ``exec_access`` vocabulary a schema-v2 manifest must declare: how the
#: plugin body touches sample-local vs. cross-sample data, as classified by the
#: emitter from the trace, never guessed.
EXEC_ACCESS_CLASSES = frozenset(
    {"sample_local", "cross_sample_read", "cross_sample_write", "mapreduce"}
)

#: the reduction-operator vocabulary a schema-v2 manifest's ``exec_op`` may declare —
#: REQUIRED iff ``exec_access == "mapreduce"`` and FORBIDDEN otherwise (the
#: ``mapreduce(op)`` declared aether functor).
EXEC_OPS = frozenset({"sum", "times", "max", "land"})

#: the v2-only FLAT top-level keys naming the execution axis. Flat, not a
#: single nested ``execution`` block: the manifest is a strict-key,
#: order-contractual document a C++ text scanner reads (module NOTE above) —
#: flat keys are the format this schema already uses for every other axis
#: (``pattern``, ``aether_abi``), so the axis is flattened to match rather
#: than introducing the schema's first nested object.
EXEC_KEYS = ("exec_targets", "exec_access", "exec_op")

#: top-level key order of an ORDINARY (unextended) manifest — contractual for
#: the C++ text scanner. See the module NOTE above for why the extended
#: variant (with ``blocks``) is not spelled here.
TOP_LEVEL_KEY_ORDER = ("schema_version", "pattern", "aether_abi", "plugins")

#: top-level key order of an ORDINARY schema-v2 manifest: :data:`TOP_LEVEL_KEY_ORDER`
#: with :data:`EXEC_KEYS` inserted between ``aether_abi`` and ``plugins`` (the
#: execution axis is a new, versioned block of top-level keys). The ``blocks``-carrying
#: extended variant (see :mod:`raptor.schema.blocks`) is unaffected — v2 exec keys do
#: not currently combine with that extension.
TOP_LEVEL_KEY_ORDER_V2 = (
    "schema_version",
    "pattern",
    "aether_abi",
    *EXEC_KEYS,
    "plugins",
)

#: the base plugin-family vocabulary a bird-neutral manifest/sidecar may name
#: (schema v1). The extension family (:mod:`raptor.schema.blocks`) adds
#: exactly one more value.
BASE_PATTERNS = frozenset({"vector", "pure"})

#: the ``kind`` discriminant of an exec reference
#: (``{"kind": ..., "kernel": ...}``). One value in v1: the referenced
#: artifact is a plain plugin kernel.
EXEC_REF_KINDS = frozenset({"kernel"})

#: the manifest-entry ``format`` vocabulary (schema v1) — the artifact
#: container a plugin manifest entry names. ``host`` is a host shared library
#: (a CPU plugin); the others are device artifacts.
MANIFEST_FORMATS = frozenset({"ptx", "cubin", "fatbin", "host"})


def check_schema_version(
    meta: dict, *, name: str = "<document>", allow_legacy_version_key: bool = False
) -> int:
    """Return the artifact's plugin-schema version, rejecting one too new to load.

    ``schema_version`` absent means v1 (backward-lenient: pre-freeze artifacts
    stay loadable). A version greater than :data:`MAX_SCHEMA_VERSION` is rejected
    (forward-strict) — the transition bridge: v1, v2 AND v3 load today, v4+
    does not. Mirrors ``eagle.roles.check_schema_version``
    (``eagle/python/eagle/roles.py::check_schema_version``) and the C++
    ``kPluginSchemaVersion`` gate. ``allow_legacy_version_key`` scopes the
    legacy ``version``-key fallback to MANIFESTS only, matching the
    eagle-side asymmetry.

    This is the ONE implementation of the version compare in Python,
    family-wide: ``eagle.roles.check_schema_version`` is a re-export of this
    function, not a twin. The rejection message is BYTE-IDENTICAL to the one
    eagle emitted before that unification — including the ``upgrade eagle``
    suffix, which is corpus-pinned in two languages (rows 04/04b/04c) and
    whose re-wording is therefore a corpus + C++ event, refused here. A floor
    road emitting different words would break road-message uniformity, the
    property that makes every floor message corpus-pinned for free. (The
    rejection THRESHOLD moved from :data:`SCHEMA_VERSION` to
    :data:`MAX_SCHEMA_VERSION` — see that constant's docstring for why they
    are two constants, not one bumped in place; the message WORDING is
    unchanged, it just now names the new ceiling.)
    """
    v = meta.get(
        "schema_version", meta.get("version", 1) if allow_legacy_version_key else 1
    )
    if not isinstance(v, int) or isinstance(v, bool) or v > MAX_SCHEMA_VERSION:
        raise ValueError(
            f"{name!r} was built for plugin schema v{v}, but this loader supports "
            f"v{MAX_SCHEMA_VERSION}; upgrade eagle to load it"
        )
    return v


def check_execution_axis(meta: dict, version: int, *, name: str = "<document>") -> None:
    """Validate the schema-v2 execution axis, given the ALREADY-resolved
    ``version`` (:func:`check_schema_version`'s return).

    schema v1 (``version == 1``): none of :data:`EXEC_KEYS` may be PRESENT — the
    legacy bridge is whole-view/single-device by the AXIS'S ABSENCE, so a v1
    document naming one is a strict-key violation, never a silent upgrade. If
    :data:`AETHER_ABI_FIELD` is present it must read :data:`AETHER_ABI_V1`.

    schema v2 and v3 (``version >= 2``; :func:`check_schema_version` has already
    rejected anything newer): ``exec_targets`` and ``exec_access`` are REQUIRED ("absence
    = load refused"); ``exec_targets`` must be a non-empty list drawn from
    :data:`EXEC_TARGETS` with no duplicates; ``exec_access`` must be one of
    :data:`EXEC_ACCESS_CLASSES`; ``exec_op`` is REQUIRED iff
    ``exec_access == "mapreduce"`` and FORBIDDEN otherwise, and when present must be
    one of :data:`EXEC_OPS`. If :data:`AETHER_ABI_FIELD` is present it must read
    :data:`AETHER_ABI_V2`.

    Only a SHAPE check, matching :func:`raptor.schema.validate_manifest`'s own
    scope note: the deep tag-vs-exported-symbol agreement (the
    ``eagle_abi_tag``/``eagle_layout_sizes`` dlsym read) stays eagle's loader, not
    this module's.
    """
    tag = meta.get(AETHER_ABI_FIELD)

    if version == 1:
        if tag is not None and tag != AETHER_ABI_V1:
            raise ValueError(
                f"{name!r}: schema v1 requires {AETHER_ABI_FIELD}={AETHER_ABI_V1!r} "
                f"when present; got {tag!r}"
            )
        present = sorted(k for k in EXEC_KEYS if k in meta)
        if present:
            raise ValueError(
                f"{name!r}: schema v1 must not carry execution key(s) {present} "
                f"(the execution axis is schema-v2-only; set schema_version=2 and "
                f"{AETHER_ABI_FIELD}={AETHER_ABI_V2!r} to use it)"
            )
        return

    # version >= 2 — check_schema_version already rejected anything newer.
    if tag is not None and tag != AETHER_ABI_V2:
        raise ValueError(
            f"{name!r}: schema v2 requires {AETHER_ABI_FIELD}={AETHER_ABI_V2!r}; "
            f"got {tag!r}"
        )

    missing = [k for k in ("exec_targets", "exec_access") if k not in meta]
    if missing:
        raise ValueError(
            f"{name!r}: schema v2 requires execution key(s) {missing} "
            "(absence = load refused)"
        )

    targets = meta["exec_targets"]
    valid_targets = (
        isinstance(targets, list)
        and bool(targets)
        and all(t in EXEC_TARGETS for t in targets)
        and len(set(targets)) == len(targets)
    )
    if not valid_targets:
        raise ValueError(
            f"{name!r}: exec_targets must be a non-empty list drawn from "
            f"{sorted(EXEC_TARGETS)} with no duplicates; got {targets!r}"
        )

    access = meta["exec_access"]
    if access not in EXEC_ACCESS_CLASSES:
        raise ValueError(
            f"{name!r}: exec_access {access!r} is not a supported access class "
            f"(supported: {sorted(EXEC_ACCESS_CLASSES)})"
        )

    op_present = "exec_op" in meta
    op = meta.get("exec_op")
    if access == "mapreduce":
        if not op_present:
            raise ValueError(
                f"{name!r}: exec_access='mapreduce' requires exec_op "
                f"(one of {sorted(EXEC_OPS)})"
            )
    elif op_present:
        raise ValueError(
            f"{name!r}: exec_op is forbidden unless exec_access='mapreduce' "
            f"(exec_access={access!r})"
        )
    if op_present and op not in EXEC_OPS:
        raise ValueError(
            f"{name!r}: exec_op {op!r} is not a supported reduction op "
            f"(supported: {sorted(EXEC_OPS)})"
        )


def check_lane_space(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional ``launch.lane_space`` block (shape only; absent = valid).

    ``axes``: non-empty list of ``{label: str | None, extent: int >= 1 | None}``,
    row-major, last axis fastest. Non-null labels are unique; only the FIRST
    extent may be ``None`` (known at run time). ``group``: int >= 1, the
    compaction unit. Unknown keys are refused. Mirrors ``lane_space`` in
    ``docs/schemas/sidecar-v1.schema.json``.
    """
    launch = meta.get("launch")
    if not isinstance(launch, dict) or "lane_space" not in launch:
        return
    space = launch["lane_space"]

    def bad(why):
        raise ValueError(f"{name!r}: launch.lane_space {why}")

    def is_int(v):
        return isinstance(v, int) and not isinstance(v, bool)

    if not isinstance(space, dict) or set(space) != {"axes", "group"}:
        bad("must be an object with exactly 'axes' and 'group'")
    axes = space["axes"]
    if not isinstance(axes, list) or not axes:
        bad("axes must be a non-empty list")
    seen = set()
    for i, ax in enumerate(axes):
        if not isinstance(ax, dict) or set(ax) != {"label", "extent"}:
            bad(f"axes[{i}] must be an object with exactly 'label' and 'extent'")
        label, ext = ax["label"], ax["extent"]
        if label is not None:
            if not isinstance(label, str):
                bad(f"axes[{i}].label must be a string or null; got {label!r}")
            if label in seen:
                bad(f"axes[{i}].label {label!r} is a duplicate")
            seen.add(label)
        if ext is None:
            if i:
                bad(f"axes[{i}].extent is null; only the first axis may be run-time")
        elif not is_int(ext) or ext < 1:
            bad(f"axes[{i}].extent must be an integer >= 1 or null; got {ext!r}")
    group = space["group"]
    if not is_int(group) or group < 1:
        bad(f"group must be an integer >= 1; got {group!r}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_launch_fields(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional ``launch.box``, ``launch.index_set`` and ``launch.local_*`` fields (shape only; absent = valid).

    ``box``: exactly ``{"axis": int >= 0, "T": int >= 2, "stride": int >= 1}``.

    ``index_set``: ``{"width": 32 | 64}`` (a map of lane numbers), or the prefix form
    ``{"width": 32, "prefix": true, "count": <plane name>}`` (no map; the named plane holds how
    many leading lanes run). ``local_bytes`` (int >= 0), ``local_source`` (``"ptxas"`` | ``"ptx"``)
    and ``local_limit`` (int >= 1) come together. Unknown ``index_set`` keys are refused. Mirrors
    ``launch`` in ``docs/schemas/sidecar-v1.schema.json``.
    """
    launch = meta.get("launch")
    if not isinstance(launch, dict):
        return

    def bad(what, why):
        raise ValueError(f"{name!r}: launch.{what} {why}")

    if "box" in launch:
        box = launch["box"]
        if not isinstance(box, dict) or set(box) != {"axis", "T", "stride"}:
            bad("box", "must be an object with exactly 'axis', 'T' and 'stride'")
        for key, low in (("axis", 0), ("T", 2), ("stride", 1)):
            if not _is_int(box[key]) or box[key] < low:
                bad("box", f"{key} must be an integer >= {low}; got {box[key]!r}")

    if "index_set" in launch:
        iset = launch["index_set"]
        if not isinstance(iset, dict) or not {"width"} <= set(iset) <= {"width", "prefix", "count"}:
            bad("index_set", "must be an object with 'width' and optionally 'prefix' and 'count'")
        if iset["width"] not in (32, 64) or not _is_int(iset["width"]):
            bad("index_set", f"width must be 32 or 64; got {iset['width']!r}")
        prefix = iset.get("prefix", False)
        if not isinstance(prefix, bool):
            bad("index_set", f"prefix must be a boolean; got {prefix!r}")
        if prefix:
            if iset["width"] != 32:
                bad("index_set", "a prefix set has width 32")
            if not isinstance(iset.get("count"), str) or not iset["count"]:
                bad("index_set", "a prefix set names its count plane in 'count'")
        elif "count" in iset:
            bad("index_set", "names a 'count' plane only for a prefix set")

    keys = ("local_bytes", "local_source", "local_limit")
    present = [k for k in keys if k in launch]
    if present and len(present) != len(keys):
        bad(present[0], f"comes with {', '.join(keys)}; missing {[k for k in keys if k not in launch]}")
    if present:
        if not _is_int(launch["local_bytes"]) or launch["local_bytes"] < 0:
            bad("local_bytes", f"must be an integer >= 0; got {launch['local_bytes']!r}")
        if launch["local_source"] not in ("ptxas", "ptx"):
            bad("local_source", f"must be 'ptxas' or 'ptx'; got {launch['local_source']!r}")
        if not _is_int(launch["local_limit"]) or launch["local_limit"] < 1:
            bad("local_limit", f"must be an integer >= 1; got {launch['local_limit']!r}")


TEAM_GRIDS = frozenset({"lanes", "keys", "chunks", "targets"})
TEAM_PLANE_DTYPES = frozenset({"real", "int32"})


def _check_keyed_record(rec, where: str, bad) -> None:
    """``{width, axes, payload?}``: a keyed plane over the kept lane axes."""
    if not isinstance(rec, dict) or not {"width", "axes"} <= set(rec) <= {"width", "axes", "payload"}:
        bad(where, "must be an object with 'width', 'axes' and optionally 'payload'")
    if not _is_int(rec["width"]) or rec["width"] < 1:
        bad(where, f"width must be an integer >= 1; got {rec['width']!r}")
    axes = rec["axes"]
    if not isinstance(axes, list) or not all(
        isinstance(a, list) and len(a) == 2 and (a[0] is None or isinstance(a[0], str))
        and (a[1] is None or (_is_int(a[1]) and a[1] >= 1)) for a in axes
    ):
        bad(where, "axes must be a list of [label | null, extent | null] pairs")
    pay = rec.get("payload", [])
    if not isinstance(pay, list) or not all(_is_int(p) and p >= 1 for p in pay):
        bad(where, f"payload must be a list of integers >= 1; got {pay!r}")



def check_index_planes(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional ``index_planes`` list (shape only; absent = valid): the names of the input planes
    declared ``Index``, which bind as ``int32`` (a runtime converts an ``int64`` array whose values fit). A
    non-empty list of distinct plane names, each one a name in ``arg_spec``. Mirrors
    ``docs/schemas/sidecar-v1.schema.json``."""
    if "index_planes" not in meta:
        return
    planes = meta["index_planes"]
    if not isinstance(planes, list) or not planes or not all(isinstance(p, str) and p for p in planes):
        raise ValueError(f"{name!r}: index_planes must be a non-empty list of plane names")
    if len(set(planes)) != len(planes):
        raise ValueError(f"{name!r}: index_planes names a plane twice")
    declared = {entry[1] for entry in meta.get("arg_spec") or () if isinstance(entry, (list, tuple)) and len(entry) == 2}
    missing = [p for p in planes if p not in declared]
    if declared and missing:
        raise ValueError(f"{name!r}: index_planes names {missing}, which arg_spec does not declare")

def check_team_fields(meta: dict, *, name: str = "<document>") -> None:
    """Validate the optional team-stage fields of a split kernel (shape only; absent = valid).

    ``stages[k].team``: ``{"threads": int >= 1, "grid"?: lanes | keys | chunks | targets, ...}``;
    grid ``keys`` carries ``keyed`` (a keyed record), ``chunks`` and ``targets`` carry ``of`` (the
    ordered-add target plane) and ``chunks`` also ``status`` (the out-of-range flag plane).
    ``team_planes``: ``{plane: {"dtype": real | int32, "per_lane": int >= 1, ...}}`` with at most
    one of ``keys`` (keyed record), ``per_target`` (plane name) and ``once`` (true), and an optional
    integer ``fill``. ``keyed``: ``{plane: keyed record}``. Unknown keys of these records are
    refused. Mirrors ``docs/schemas/sidecar-v1.schema.json``.
    """
    def bad(where, why):
        raise ValueError(f"{name!r}: {where} {why}")

    stages = meta.get("stages")
    for k, st in enumerate(stages if isinstance(stages, list) else ()):
        if not isinstance(st, dict) or "team" not in st:
            continue
        where, team = f"stages[{k}].team", st["team"]
        if not isinstance(team, dict) or "threads" not in team:
            bad(where, "must be an object with 'threads'")
        if not _is_int(team["threads"]) or team["threads"] < 1:
            bad(where, f"threads must be an integer >= 1; got {team['threads']!r}")
        grid = team.get("grid", "lanes")
        if grid not in TEAM_GRIDS:
            bad(where, f"grid {grid!r} is not one of {sorted(TEAM_GRIDS)}")
        allowed = {"threads", "grid"} | {
            "lanes": set(), "keys": {"keyed"}, "chunks": {"of", "status"}, "targets": {"of"}
        }[grid]
        if set(team) - allowed:
            bad(where, f"has keys {sorted(set(team) - allowed)} that grid {grid!r} does not carry")
        if grid != "lanes" and "grid" not in team:
            bad(where, "names its grid")
        required = allowed - {"threads", "grid"}
        if required - set(team):
            bad(where, f"grid {grid!r} needs {sorted(required - set(team))}")
        if grid == "keys":
            _check_keyed_record(team["keyed"], f"{where}.keyed", bad)
        for key in required - {"keyed"}:
            if not isinstance(team[key], str) or not team[key]:
                bad(where, f"{key} must be a plane name; got {team[key]!r}")

    if "team_planes" in meta:
        planes = meta["team_planes"]
        if not isinstance(planes, dict):
            bad("team_planes", "must be an object of plane records")
        for pname, spec in planes.items():
            where = f"team_planes[{pname!r}]"
            if not isinstance(spec, dict) or not {"dtype", "per_lane"} <= set(spec) <= {
                "dtype", "per_lane", "keys", "per_target", "once", "fill"
            }:
                bad(where, "must be an object with 'dtype', 'per_lane' and optionally "
                    "'keys', 'per_target', 'once', 'fill'")
            if spec["dtype"] not in TEAM_PLANE_DTYPES:
                bad(where, f"dtype must be one of {sorted(TEAM_PLANE_DTYPES)}; got {spec['dtype']!r}")
            if not _is_int(spec["per_lane"]) or spec["per_lane"] < 1:
                bad(where, f"per_lane must be an integer >= 1; got {spec['per_lane']!r}")
            if len({"keys", "per_target", "once"} & set(spec)) > 1:
                bad(where, "sizes by at most one of 'keys', 'per_target' and 'once'")
            if "keys" in spec:
                _check_keyed_record(spec["keys"], f"{where}.keys", bad)
            if "per_target" in spec and (not isinstance(spec["per_target"], str) or not spec["per_target"]):
                bad(where, f"per_target must be a plane name; got {spec['per_target']!r}")
            if "once" in spec and spec["once"] is not True:
                bad(where, f"once must be true; got {spec['once']!r}")
            if "fill" in spec and not _is_int(spec["fill"]):
                bad(where, f"fill must be an integer; got {spec['fill']!r}")

    if "keyed" in meta:
        if not isinstance(meta["keyed"], dict):
            bad("keyed", "must be an object of keyed records")
        for kname, rec in meta["keyed"].items():
            _check_keyed_record(rec, f"keyed[{kname!r}]", bad)
