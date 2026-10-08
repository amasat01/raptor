# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``neural_block`` extension layer of the schema.

hawk never imports this module — only :mod:`raptor.schema.manifest` (the
family-neutral layer) is part of hawk's core-purity surface. This module is
free to reference ``manifest``, never the reverse.

Values below are copied VERBATIM from ``eagle/python/eagle/roles.py``, cited
by symbol below (symbol citations do not rot the way line citations do);
eagle's roles.py is not modified by this move — a later change may re-point
it at this module instead.

**The descriptor VALIDATOR lives here too.** The clause that enforces this
family's wire contract — :func:`validate_neural_block_descriptor` and the
private ``_validate_neural_block`` / ``_validate_exec_ref`` it is built from —
moved VERBATIM out of ``eagle/python/eagle/sidecar.py`` into this module,
because enforcement lives with the contract's OWNER at (never above) its
lowest-floor consumer: the descriptor wire contract is declared here and
consumed at the raptor+numpy floor, so its validator is spine content. It is
ONE implementation with two importers — a downstream package's two roads and
``eagle.sidecar.validate_sidecar``'s pattern-conditional dispatch — never a
copy. What did NOT move is everything serving the KERNEL families (``ROLES``
/ ``validate_roles``, ``parse_derivative``, the recognized-pattern and
buffer-kind loops, ``MANIFEST_FORMATS``): that vocabulary is eagle-owned and
consumed only by eagle loaders, so its validator stays in eagle for the
``neural_block`` family only — eagle stays the deep validator everywhere
else. This module imports nothing but :mod:`raptor.schema.manifest`, so the
clause runs on a numpy-less, eagle-less interpreter.
"""

from __future__ import annotations

from . import manifest

#: the neural-block family name (verbatim value from
#: ``eagle/python/eagle/roles.py::RECOGNIZED_PATTERNS``, one element of that
#: set).
NEURAL_BLOCK_PATTERN = "neural_block"

#: the full top-level key order for a NEURAL manifest — houses the literal
#: "blocks" carriage key (kept out of the neutral module, per
#: :mod:`raptor.schema.manifest`'s note).
NEURAL_TOP_LEVEL_KEY_ORDER = (
    "schema_version",
    "pattern",
    "aether_abi",
    "blocks",
    "plugins",
)

#: the declared TERMINAL-WRITE CONTRACT of a block's scatter (verbatim value
#: from ``eagle/python/eagle/roles.py::SCATTER_POLICIES``).
SCATTER_POLICIES = frozenset({"unique_write", "accumulate"})

#: the fields a ``neural_block`` descriptor MUST carry, beyond the general
#: required set (verbatim value from
#: ``eagle/python/eagle/roles.py::NEURAL_REQUIRED_FIELDS``, minus a schema-key
#: rename: ``fanin``/``fanout`` -> ``in_degree``/``out_degree``, both ends
#: renamed together, SCHEMA_VERSION stays 1).
NEURAL_REQUIRED_FIELDS = frozenset(
    {
        "in_degree",
        "out_degree",
        "input_width",
        "output_width",
        "state_width",
        "param_width",
        "scatter_policy",
        "forward_exec",
    }
)

#: the keys whose values are exec references (verbatim value from
#: ``eagle/python/eagle/roles.py::NEURAL_EXEC_REF_FIELDS``).
NEURAL_EXEC_REF_FIELDS = frozenset({"forward_exec", "vjp_exec", "jvp_exec"})

#: kernel-machinery fields a descriptor must NOT carry (verbatim value from
#: ``eagle/python/eagle/roles.py::NEURAL_FORBIDDEN_FIELDS``).
NEURAL_FORBIDDEN_FIELDS = frozenset(
    {"aether_abi", "derivative", "buffers", "mutables", "mat_shapes", "host_entry"}
)

#: the declared-buffer ``kind`` vocabulary, schema v1 (verbatim value from
#: ``eagle/python/eagle/roles.py::BUFFER_KINDS``).
BUFFER_KINDS = frozenset({"lookup"})

#: the full pattern vocabulary this extension recognizes: the neutral base
#: (:data:`raptor.schema.manifest.BASE_PATTERNS`) plus ``neural_block``.
ALL_PATTERNS = manifest.BASE_PATTERNS | {NEURAL_BLOCK_PATTERN}

# --------------------------------------------------------------------------- #
# The descriptor-family VALIDATOR. Everything below moved VERBATIM from
# ``eagle/python/eagle/sidecar.py`` — only the vocabulary SPELLING changed
# (``EXEC_REF_KINDS`` -> ``manifest.EXEC_REF_KINDS``); the checks, their order
# and every message byte are the ones the shared conformance corpus and the C++
# twin (``eagle/plugin/sidecar.h``) already pin.
# --------------------------------------------------------------------------- #

#: The scalar types a ``neural_block`` descriptor may declare (an ALLOWLIST).
#: SoftDouble is out of scope for neural blocks, and the allowlist is why that
#: stays true without a source literal naming it — the rejected value reaches
#: a message only at runtime, from the offending input.
_NEURAL_SCALAR_TYPES = ("float32", "float64")

#: Of the required neural fields, the one that may legitimately be ZERO: a block with no
#: learned parameters is meaningful, a block with no inputs or no state is not.
_NEURAL_NON_NEGATIVE_FIELDS = frozenset({"param_width"})

#: The upper bound every neural_block integer field must respect — uint32
#: representability, mirroring the runtime tier's ``_UINT32_CAP`` on the DERIVED
#: N_BLOCKS/PARAM_POOL_SIZE (``buffers.py``). The descriptor tier had no analogous
#: cap on its own fanin/fanout/*_width fields until this field was added.
_NEURAL_INT_UPPER_BOUND = 2**32

#: The required neural fields that carry integers — everything except the scatter
#: contract and the exec references. DERIVED, so it cannot drift from the required list.
_NEURAL_INT_FIELDS = (
    NEURAL_REQUIRED_FIELDS - NEURAL_EXEC_REF_FIELDS - {"scatter_policy"}
)


def _joined(vocab) -> str:
    """A stable "a, b" rendering of a vocabulary, for an error naming the valid set."""
    return ", ".join(sorted(vocab))


def _validate_exec_ref(ref, *, field: str, name: str) -> None:
    """Validate one ``{"kind": ..., "kernel": ...}`` exec reference.

    Strict shape, value-strict on ``kind``: an exec ref names WHICH artifact implements
    a leg of the block and HOW to read the reference, so an unrecognized ``kind``
    must be refused rather than dispatched on as though it were a plain kernel. The
    object is also CLOSED: an exec ref is a fixed two-key shape, so a third key — a
    digest payload, a stray typo — is refused rather than silently ignored, the same
    "closed set" discipline the requiredness and forbidden-field loops below already
    apply to the descriptor as a whole."""
    if not isinstance(ref, dict):
        raise ValueError(
            f"{name}: neural_block {field} must be an object "
            f"{{'kind': ..., 'kernel': ...}}; got {ref!r}"
        )
    unknown = sorted(set(ref) - {"kind", "kernel"})
    if unknown:
        raise ValueError(
            f"{name}: neural_block {field} carries unknown key(s) {unknown} "
            "(exec references are closed to 'kind', 'kernel')"
        )
    kind = ref.get("kind")
    if kind not in manifest.EXEC_REF_KINDS:
        raise ValueError(
            f"{name}: neural_block {field}.kind {kind!r} is not a supported exec "
            f"reference kind (supported: {_joined(manifest.EXEC_REF_KINDS)}); "
            "upgrade eagle"
        )
    target = ref.get("kernel")
    if not isinstance(target, str) or not target:
        raise ValueError(
            f"{name}: neural_block {field}.kernel must be a non-empty kernel id; "
            f"got {target!r}"
        )


def _validate_neural_block(meta: dict, *, name: str) -> None:
    """The pattern-conditional ``neural_block`` clause of the shared validator.

    A descriptor is not a kernel: it carries no ``arg_spec`` entries, no artifact of its
    own, and nothing launches it (``neural_block`` is RECOGNIZED but never
    LAUNCH-CERTIFIED). It declares the block's shape and names the kernels that
    implement it; those are ordinary ``pure`` artifacts loaded through normal doors.

    This is a CLAUSE inside the one shared validator, never a second validator: a
    downstream package's neural-block spec type — both its sidecar-loading road and its
    direct construction — funnels through here, so the two roads cannot drift apart. The
    C++ twin is ``validate_sidecar``'s identical clause in ``plugin/sidecar.h``, pinned
    message-for-message by the shared conformance corpus."""
    # The descriptor's wire identity. C++ ``parse_sidecar`` requires
    # the key at parse; Python has no parse step, so the clause requires it here.
    if not meta.get("kernel"):
        raise ValueError(
            f"{name}: a 'neural_block' descriptor must carry a non-empty 'kernel' "
            "(its wire identity)"
        )
    # ABSENCE-STRICT, unlike the general lenient check above: the leniencies exist for
    # pre-freeze backward compatibility, and a family minted after the freeze has no
    # backward to be compatible with.
    scalar_type = meta.get("scalar_type")
    if scalar_type not in _NEURAL_SCALAR_TYPES:
        raise ValueError(
            f"{name}: neural_block requires scalar_type in "
            f"({', '.join(_NEURAL_SCALAR_TYPES)}); got {scalar_type!r} "
            "(a neural block's arithmetic is native-float only)"
        )
    # Present and EMPTY. Present because the C++ ``parse_sidecar`` requires the key
    # before any validation can run, so a descriptor without it is unparseable in
    # one language and would be accepted in the other; EMPTY because a descriptor
    # binds nothing — a populated arg_spec means the producer built a kernel
    # sidecar and stamped it as a descriptor.
    if "arg_spec" not in meta:
        raise ValueError(
            f"{name}: a 'neural_block' descriptor must carry an arg_spec, present and "
            "empty ([]); the key is absent"
        )
    if meta["arg_spec"]:
        raise ValueError(
            f"{name}: a 'neural_block' descriptor must carry an EMPTY arg_spec ([]); "
            f"got {len(meta['arg_spec'])} entries (a descriptor binds nothing — the "
            "kernels it references carry their own arg_spec)"
        )
    # Presence is ITERATED over the constant, never hand-written per field, so the
    # requiredness lives in exactly one place.
    for field in sorted(NEURAL_REQUIRED_FIELDS):
        if field not in meta:
            raise ValueError(
                f"{name}: missing required neural_block field {field!r} "
                f"(schema-v1 neural_block requires {_joined(NEURAL_REQUIRED_FIELDS)})"
            )
    for field in sorted(_NEURAL_INT_FIELDS):
        value = meta[field]
        non_negative = field in _NEURAL_NON_NEGATIVE_FIELDS
        ok = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and (value >= 0 if non_negative else value > 0)
            and value < _NEURAL_INT_UPPER_BOUND
        )
        if not ok:
            what = "a non-negative integer" if non_negative else "a positive integer"
            raise ValueError(
                f"{name}: neural_block {field} must be {what} representable in a "
                f"uint32 (< 2**32); got {value!r}"
            )
    # A wire-checkable width lock: the block's outputs are committed into the
    # per-block state row, so they cannot be wider than it.
    if meta["output_width"] > meta["state_width"]:
        raise ValueError(
            f"{name}: neural_block output_width ({meta['output_width']}) must not "
            f"exceed state_width ({meta['state_width']})"
        )
    policy = meta["scatter_policy"]
    if policy not in SCATTER_POLICIES:
        raise ValueError(
            f"{name}: neural_block scatter_policy {policy!r} is not a supported "
            f"terminal-write contract (supported: {_joined(SCATTER_POLICIES)}); "
            "upgrade eagle"
        )
    for field in sorted(NEURAL_EXEC_REF_FIELDS):
        if field in meta:
            _validate_exec_ref(meta[field], field=field, name=name)
    # Forbidden kernel-machinery fields — ITERATED over the constant, same rule as
    # presence above. Each would be silently ignored on a descriptor, so a
    # producer that copies one must fail loudly.
    for field in sorted(NEURAL_FORBIDDEN_FIELDS):
        if meta.get(field):
            raise ValueError(
                f"{name}: a 'neural_block' descriptor must not carry {field!r} "
                f"(forbidden on a descriptor: {_joined(NEURAL_FORBIDDEN_FIELDS)}); it "
                "belongs on the kernel artifact the descriptor references"
            )


def validate_neural_block_descriptor(meta: dict, *, name: str) -> dict:
    """Validate a ``neural_block`` descriptor document; raise on any breach.

    THE public entry point of the descriptor family's wire contract, and the ONE
    implementation of it: a downstream package's neural-block spec type calls it
    directly on construction and when loading from a sidecar (the raptor+numpy
    floor road), and :func:`eagle.sidecar.validate_sidecar` calls it as its
    pattern-conditional ``neural_block`` clause (the eagle road). Two importers,
    no copy — which is what makes the two roads' verdicts and messages identical by
    construction rather than by parallel maintenance.

    The order is the CORPUS order and is contractual: the schema-version gate
    (:func:`raptor.schema.manifest.check_schema_version`) runs BEFORE the clause, so a
    descriptor that is both too new and malformed is refused for being too new — what
    corpus row ``row04c`` pins. ``meta`` is the descriptor as loaded from JSON (or as
    serialized in process); ``name`` names the artifact in every message. Returns
    ``meta`` unchanged, for call-site chaining.
    """
    manifest.check_schema_version(meta, name=name)
    _validate_neural_block(meta, name=name)
    return meta
