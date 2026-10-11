.. _neural-block-descriptor:

The ``neural_block`` descriptor
===============================

.. contents:: On this page
   :local:
   :depth: 2

A ``neural_block`` sidecar describes a **block**, not a kernel. It carries no
artifact of its own and no bindings; it declares the block's shape and *names* the
kernels that implement it. Those kernels are ordinary ``pure`` plugins, deployed and
loaded through the normal doors — so a neural deployment is a set of familiar
artifacts plus a descriptor that says how they fit together.

That distinction has a hard consequence, and it is the reason this variant exists as
a separate family rather than as extra keys on ``pure``:

.. important::

   ``neural_block`` is **recognized but never launch-certified**. The shared
   validator understands and checks it in full; no loader in either language will
   bind or run it. A descriptor arriving at any launching entry point is refused
   with *"is recognized but not launchable by this loader"* — deliberately **without**
   an "upgrade eagle" hint, because a newer eagle is not the remedy. Only a genuinely
   unrecognized ``pattern`` earns that hint.

Descriptor fields
-----------------

Required, from the shared base: ``schema_version``, ``pattern``, ``kernel`` (the
descriptor's own wire identity), ``scalar_type``, and ``arg_spec`` — which must be
**present and empty** (``[]``). ``scalar_type`` is stricter here than on a kernel
variant: it is required (no pre-freeze leniency — a family minted after the freeze
has no backward to be compatible with) and must be ``"float32"`` or ``"float64"``.

Required, descriptor-specific — single-sourced as
:data:`eagle.roles.NEURAL_REQUIRED_FIELDS` / ``plugin/roles.h::kNeuralRequiredFields``,
which both validators *iterate*:

.. list-table::
   :header-rows: 1
   :widths: 20 16 64

   * - Key
     - Type
     - Meaning
   * - ``in_degree``
     - int > 0
     - Number of source slots each block gathers from. Renamed from
       ``fanin`` pre-release; ``schema_version`` stayed 1 (a pre-release hard
       break, not a versioned migration).
   * - ``out_degree``
     - int > 0
     - Number of target slots each block scatters to. Renamed from
       ``fanout`` alongside ``in_degree``.
   * - ``input_width``
     - int > 0
     - Width of the gathered input vector the block consumes.
   * - ``output_width``
     - int > 0
     - Width of the result the block commits. Must not exceed ``state_width``.
   * - ``state_width``
     - int > 0
     - Width of the block's per-block state row.
   * - ``param_width``
     - int >= 0
     - Width of the block's parameter slice. Zero is legal: a block with no
       learned parameters is meaningful.
   * - ``scatter_policy``
     - ``"unique_write"`` | ``"accumulate"``
     - The declared terminal-write **contract** (see below).
   * - ``forward_exec``
     - exec reference
     - The kernel implementing the block's forward evaluation.

Optional: ``vjp_exec`` and ``jvp_exec``, two more exec references naming the
derivative artifacts. Absent is always fine; present is validated exactly as
strictly as ``forward_exec``.

An **exec reference** is ``{"kind": "kernel", "kernel": "<id>"}``. ``kind`` is a
value-strict discriminant with one v1 value: the reference names a plain plugin
kernel. It exists as a discriminant rather than a bare string because a future
aggregate or plan-bundle reference would *reshape* the object (carrying a digest
alongside ``kind``), which is a meaning change and therefore a ``schema_version``
bump — not something a v1 reader should try to interpret.

What a descriptor must **not** carry
------------------------------------

``aether_abi``, ``derivative``, ``host_entry``, and any populated ``buffers``,
``mutables`` or ``mat_shapes`` are **forbidden** on a descriptor, and a populated
``arg_spec`` is forbidden too. Every one of them is kernel machinery that a loader
would simply *ignore* on a descriptor, and a field that is silently ignored is how a
confused producer ships something that does not do what it says. ``aether_abi`` in
particular belongs on the artifacts the descriptor references — they carry their own
tags, and a copy on the descriptor is a second spelling that can drift. A
``derivative`` block belongs on the VJP artifact itself, which is exactly where it
lives when you follow ``vjp_exec`` to it.

``scatter_policy``: a contract, never a mechanism
-------------------------------------------------

``scatter_policy`` declares the **contract** of the block's terminal write — what the
committed result *means* — and never the mechanism eagle uses to commit it.

``unique_write`` says: every ``(target, slot)`` is written by exactly one source per
step. The commit is therefore a plain store, the result is deterministic with zero
atomics, and bit-exact comparison is a legitimate gate mode.

``accumulate`` is the second value: every ``(target, slot)`` may be written by more
than one source per step, and the committed result is the carried base plus an
order-unspecified sum of contributions. Its gate mode is band-tolerance, never
bit-exact — ``unique_write`` is the only value bit-exact comparison applies to.

Mechanism — atomic add, block-reduce-then-atomic, segmented reduce — is eagle's own
business, chosen on evidence and invisible to the wire. Several of those are simply
different ways to compute the *same* ``accumulate`` contract, faster or more
deterministically; promoting them to schema vocabulary would turn every performance
experiment into a schema event. Changing executors is never a schema change. A third
value will be minted only when a contract genuinely differs again from both of these,
and it will be named after that contract, together with the executor that can
actually run it.
