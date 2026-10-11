Plugin schema (v1 and v2)
=========================

.. contents:: On this page
   :local:
   :depth: 2

A plugin, in one picture
------------------------

The code generator — a separate DSL/JIT-compiler library that eagle's launch engine
serves — compiles a kernel and writes it to disk as a small set of files, a
*producer* step (see the :doc:`glossary </content/glossary>` for this page's words). Later, at deploy time, eagle reads those files back and
turns them into something you can call — a *loader* step. Neither side needs
the other's source code; they agree only on the files' shape. Concretely,
for one compiled kernel:

.. code-block:: text

   build_dir/
     escape.ptx     <- the compiled GPU code (producer output)
     escape.json    <- the "sidecar": how to call escape.ptx (producer output)

**PTX** is an intermediate, architecture-portable GPU assembly format — the
GPU driver JIT-compiles it to real machine code at load time, so one
``.ptx`` file runs on any supported GPU generation without being rebuilt for
each one.

and the sidecar itself, annotated:

.. code-block:: json

   {
     "format": "ptx",
     "schema_version": 1,
     "pattern": "pure",
     "aether_abi": "aether-abi/1",
     "kernel": "raptor_kernel",
     "per_sample": ["r_max"],
     "arg_spec": [["nsamples", "N"], ["per_sample", "r_max"], ["mutable", "fired"]]
   }

Reading it top to bottom: this artifact is PTX-format (:ref:`role-vocab`
below covers ``format``'s siblings), speaks plugin **schema** version 1
(this page's own versioning — see :ref:`aether-abi-tag` for how that differs
from ``aether_abi``), is a *pure*-shaped kernel (as opposed to *vector* — the
same two kernel shapes the code generator's docs describe), was compiled against binary
layout ``aether-abi/1``, exposes the entry point ``raptor_kernel`` (the one
name every artifact exports, see ``kernel`` below), and
``arg_spec`` is the exact, ordered list of arguments a loader must bind to
call it — each a ``[role, name]`` pair. A **role** (``nsamples``,
``per_sample``, ``mutable``, and 6 others) says *what kind* of argument a
name is; :ref:`role-vocab` below is the complete list.

When you have several kernels to deploy together, one **manifest**
(``manifest.json``) lists them all, each with its own artifact + sidecar
pair — see :ref:`the deployment manifest <deployment-manifest>` below for its
exact shape.

Words used on this page
-----------------------

Producer, loader, sink, consolidation and the ``aether_abi`` tag are
explained in the family :doc:`glossary </content/glossary>`, along with the
other terms the raptor pages share.

Now the exhaustive shapes — for daily reference once the picture above makes
sense.

Overview
--------

eagle owns the **plugin protocol**: the on-disk contract between the
producer and every loader, as sketched above. A deployment unit is:

* one **manifest** (``manifest.json``) — an ordered list of plugin artifacts;
* one **artifact** per plugin (``<id>.ptx`` / ``.cubin`` / ``.fatbin``) — the
  compiled kernel;
* one **sidecar** (``<id>.json``) per artifact — its launch signature.

This page freezes both JSON shapes as **schema v1**. The normative machine-readable
definitions are the two JSON Schema files (draft 2020-12) shipped alongside this
page:

* :download:`manifest-v1.schema.json </schemas/manifest-v1.schema.json>`
* :download:`sidecar-v1.schema.json </schemas/sidecar-v1.schema.json>`

The schema version is single-sourced from :data:`raptor.schema.manifest.SCHEMA_VERSION`
(Python; re-exported as ``eagle.roles.SCHEMA_VERSION``) ≡
``plugin/roles.h::kPluginSchemaVersion`` (C++); its current frozen value is
**1**.

The details, page by page
-------------------------

Each page below holds one part of the schema.

.. toctree::
   :maxdepth: 1
   :hidden:

   plugin_schema/manifest
   plugin_schema/sidecar
   plugin_schema/neural_block
   plugin_schema/cpp_consumer_scope
   plugin_schema/compatibility
   plugin_schema/execution_axis
   plugin_schema/reserved_and_files

- :doc:`plugin_schema/manifest` — the manifest, the file that lists a deployment's plugins.
- :doc:`plugin_schema/sidecar` — the per-kernel file that says how to call it, with its variant additions.
- :doc:`plugin_schema/neural_block` — the descriptor for neural-network blocks.
- :doc:`plugin_schema/cpp_consumer_scope` — what the C++ reference consumer covers, and the float32 capability it defers.
- :doc:`plugin_schema/compatibility` — the compatibility policy, and why the ``aether_abi`` tag is versioned separately.
- :doc:`plugin_schema/execution_axis` — the v2 execution axis, driving a v2 body from Python, and running across MPI ranks.
- :doc:`plugin_schema/reserved_and_files` — reserved capability fields and the JSON Schema files.

.. _role-vocab:

The arg_spec role vocabulary
----------------------------

``arg_spec`` is an ordered list of two-element ``[role, name]`` arrays — **role
first**. ``role`` is one of these **12 canonical arg-spec roles (schema v1)**.
This set is the single source of truth, enumerated identically in
``plugin/roles.h::kPluginArgRoles`` (C++) and :data:`eagle.roles.ROLES` (Python),
and pinned by the cross-check test ``python/tests/test_roles_vocab.py``. The
first 9 were present at the schema-v1 freeze; ``wide_in``, ``wide_out`` and
``accum_out`` were added later, additively — admitting a role is not a wire
shape change, so none of the three needed a ``schema_version`` bump.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Role
     - Meaning
   * - ``out``
     - The plugin's output sink (the vector ``outVec`` accumulate slot).
   * - ``vec_in``
     - A named vector input (from ``vector_inputs``).
   * - ``mat_in``
     - A matrix input to a **pure** or **vector** kernel (see the note below;
       vector twin — a returning kind's ``Matrix[R, C]`` vocabulary
       field, exercised by the code generator's own round-trip suite).
   * - ``per_sample``
     - A named per-sample scalar input (from ``per_sample``).
   * - ``lookup``
     - A read-only lookup buffer, bound by value as a flat handle (from
       ``buffers``).
   * - ``mutable``
     - A writable per-sample state slot — the pure output (from ``mutables``).
   * - ``terminated``
     - The per-sample terminated/active mask.
   * - ``uniform``
     - A uniform scalar parameter (from ``params``).
   * - ``nsamples``
     - The sample count.
   * - ``wide_in``
     - A named "wide" buffer input, bound by value as a flat handle — the
       same wire shape as ``lookup``. A wide buffer's true row count is a
       runtime quantity baked into the kernel's own row arithmetic, never the
       ABI, so it rides its own binding path rather than reusing
       ``lookup``/``buffers``.
   * - ``wide_out``
     - A named wide-buffer output: the VJP scatter-gradient counterpart to
       ``wide_in``, bound through the same by-value handle ABI.
   * - ``accum_out``
     - The cross-sample accumulate plane a ``cross_sample_write`` body (the
       v2 execution axis below) writes into — construction-identical to
       ``wide_out``'s plain scalar-handle ABI, kept as its own name so the
       exec-placement logic can refer to it distinctly.

The per-kind subsets (vector = 6, pure = 8 roles) union to exactly this set of
12.

.. note::

   ``mat_in`` is a **valid** schema-v1 role recognized by every loader; the
   reference C++ host ``inject()`` (``plugin/host_registry.h``, CPU) packs it
   identically to ``vec_in`` (a matrix GRef is a width-``R*C`` vector GRef, so
   it rides the same ``GRefMirror``), so a matrix pure **or** vector kernel
   host-round-trips today (``test_matrix_pure_to_host_matches_direct``
   / ``test_matrix_vector_to_host_matches_direct``). The device
   ``PluginRegistry::inject()`` (``plugin/plugin_registry/registry.h``, CUDA)
   packs it the same way via its own ``bind_matrix`` — see the
   ``mattrace`` fixture in eagle's C++ embedding example
   (``eagle``'s own ``userguide/examples`` page) for a device-side,
   driver-loaded matrix input exercised end to end.
