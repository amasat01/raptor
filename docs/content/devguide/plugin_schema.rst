Plugin schema (v1 and v2)
=========================

.. contents:: On this page
   :local:
   :depth: 2

A plugin, in one picture
------------------------

The code generator — a separate DSL/JIT-compiler library that EAGLE's launch engine
serves — compiles a kernel and writes it to disk as a small set of files, a
*producer* step. Later, at deploy time, EAGLE reads those files back and
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

A short glossary before the tables
-----------------------------------

- **Producer** — the code that compiles a kernel and writes the artifact +
  sidecar (the code generator's deploy pipeline: ``bundle.py`` / ``compile.py``).
- **Loader** — the code that reads a manifest/sidecar back and hands you a
  callable object (EAGLE's C++ ``PluginRegistry``, or the Python ``Loaded*``
  classes via ``load_manifest``).
- **Sink** — where a kernel's result goes. A *pure* kernel's sink is its
  ``mutable`` slots, written in place; a *vector* kernel's sink is a single
  accumulated output value. (The tables below name the exact JSON field and
  role for each — this is the plain-language version.)
- **Consolidation** — the one-time upload of a read-only lookup table before
  a run starts, as opposed to per-launch arguments, which travel with every
  call. (The sidecar's ``buffers`` array, in the tables below, is what gets
  consolidated.)
- **``aether_abi``** — a *separate* version tag from ``schema_version``,
  covering binary struct layout rather than JSON shape; see
  :ref:`aether-abi-tag`.

Now the exhaustive shapes — for daily reference once the picture above makes
sense.

Overview
--------

EAGLE owns the **plugin protocol**: the on-disk contract between the
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

.. _deployment-manifest:

The deployment manifest
-----------------------

Top-level keys of ``manifest.json``:

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Key
     - Type
     - Meaning
   * - ``schema_version``
     - int (``1``)
     - Plugin-schema version. Absent ⇒ treated as v1 (backward-lenient). Newer
       than the loader ⇒ rejected (forward-strict).
   * - ``version``
     - int (``1``)
     - **Legacy** manifest-version alias (always ``1``). Retained for pre-freeze
       readers; a loader falls back to it when ``schema_version`` is absent.
   * - ``pattern``
     - string
     - Plugin family of the set. The code generator's ``Bundle`` builder emits one of the
       canonical ``"vector"`` | ``"pure"`` (a bundle is pattern-homogeneous). It is
       a schema **discriminant**: an unknown value is rejected by every
       loader, Python and C++ alike. Python's loader is additionally
       **absence-strict** (an absent ``pattern`` is rejected too — dispatch needs
       it to pick ``LoadedVector``/``LoadedPure``); the two C++ registries stay
       **absence-lenient** (a pre-freeze manifest that never stamped one still
       loads — neither registry dispatches a variant class off it). See
       "Compatibility policy" below.
   * - ``aether_abi``
     - const ``"aether-abi/1"``
     - Binary-ABI tag (see :ref:`aether-abi-tag`). Orthogonal to ``schema_version``.
   * - ``plugins``
     - array
     - The set's plugin entries in declared (injection) order.

Each ``plugins`` entry:

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Key
     - Type
     - Meaning
   * - ``id``
     - string
     - Stable, unique plugin id (from the kernel function name).
   * - ``order``
     - int ≥ 0
     - Injection order == list index.
   * - ``enabled``
     - bool
     - Whether the registry loads this plugin.
   * - ``artifact``
     - string
     - Artifact filename, relative to the manifest directory.
   * - ``sidecar``
     - string
     - Sidecar filename, relative to the manifest directory.
   * - ``format``
     - ``"ptx"`` | ``"cubin"`` | ``"fatbin"``
     - Artifact format: arch-portable PTX, single-arch cubin, or multi-arch
       fatbin.

The sidecar
-----------

A sidecar is a **shared base** plus one ``pattern``-keyed variant. Two variants are
KERNEL variants — ``vector`` and ``pure``, the shapes a ``Bundle`` emits and
the only shapes any loader launches. A third, ``neural_block``, is a **descriptor**:
it names kernels rather than being one, and nothing launches it (see
:ref:`neural-block-descriptor`).

The base fields (``compile.py::_base_sidecar``) are present on every KERNEL variant;
a descriptor carries a deliberately smaller base, spelled out in its own section
below:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Key
     - Type
     - Meaning
   * - ``format``
     - ``"ptx"`` | ``"cubin"`` | ``"fatbin"``
     - Artifact format the sidecar accompanies.
   * - ``schema_version``
     - int (``1``)
     - As for the manifest (backward-lenient / forward-strict).
   * - ``pattern``
     - ``"vector"`` | ``"pure"`` | ``"neural_block"``
     - Selects the variant. Python's ``LoadedVector``/``LoadedPure`` each refuse
       the wrong family outright (and default an *absent* value to ``"vector"``
       for ``LoadedVector`` only). C++'s shared ``validate_sidecar`` treats it as
       a discriminant too — an unrecognized value is rejected, naming the
       supported set — but never dispatches a variant class (every C++ binding
       decision is driven by ``arg_spec`` alone), so it refuses only an
       unrecognized value, not a "wrong-for-this-caller" one, and stays
       absence-lenient.
   * - ``aether_abi``
     - const ``"aether-abi/1"``
     - Binary-ABI tag (see :ref:`aether-abi-tag`).
   * - ``scalar_type``
     - ``"float64"`` | ``"float32"`` | ``"softdouble"``
     - Real scalar type the kernel was compiled for.
   * - ``kernel``
     - string
     - The ``extern "C"`` entry-point symbol to resolve in the artifact.
       Always ``"raptor_kernel"`` (``eagle.KERNEL_NAME``): raptor's schema
       fixes this one name for every producer, hand-written kernels
       included, so the artifact file name, not the symbol, tells kernels
       apart.
   * - ``vector_inputs``
     - array[string]
     - Named vector inputs, in binding order.
   * - ``params``
     - array[string]
     - Named uniform scalar parameters.
   * - ``per_sample``
     - array[string]
     - Named per-sample scalar inputs.
   * - ``arg_spec``
     - array[[role, name]]
     - The ordered ``[role, name]`` binding list — **role first** (see
       :ref:`role-vocab`). Present on every variant.

Variant additions
~~~~~~~~~~~~~~~~~~

**vector** adds:

.. list-table::
   :header-rows: 1
   :widths: 20 24 56

   * - Key
     - Type
     - Meaning
   * - ``accumulate``
     - bool
     - Whether the ``out`` sink uses the ``outVec +=`` accumulate idiom.
   * - ``sink``
     - string
     - The accumulate-sink contract name (e.g. ``"outVec-scratch"``) — a
       per-kernel deterministic-reduce scratch slot, not the final total.
   * - ``vec_widths``
     - object{string→int}
     - Per-name vector width (component count) for host ``(W, N)`` allocation.
   * - ``buffers``
     - array
     - Declared read-only lookup buffers (may be empty; see below).

**pure** adds:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Key
     - Type
     - Meaning
   * - ``mutables``
     - array[object]
     - The writable per-sample state that *is* the pure output. Each element is
       ``{name, dtype ("float"|"int"|"vector"|"matrix"), width, default
       (number|null)}``, plus ``shape: [R, C]`` **only** when ``dtype ==
       "matrix"``.
   * - ``vec_widths``
     - object{string→int}
     - Per-name vector width (includes vector Mutables).
   * - ``buffers``
     - array
     - Declared read-only lookup buffers (may be empty).
   * - ``matrix_inputs``
     - array[string]
     - **Only when the kernel has matrix inputs**: the ``mat_in`` binding order.
   * - ``mat_shapes``
     - object{string→[R, C]}
     - **Only when the kernel has matrix inputs**: the ``(R, C)`` shape per
       matrix-input name.

Each ``buffers`` element (``compile.py::_buffers_field``) is
``{name: string, kind: "lookup", dtype: "float", shape: array[int], count: int}``,
where ``count == prod(shape)`` is the flat element count the registry uploads once
at consolidation and binds by value as a flat handle.

.. _neural-block-descriptor:

The ``neural_block`` descriptor
-------------------------------

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
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``scatter_policy`` declares the **contract** of the block's terminal write — what the
committed result *means* — and never the mechanism EAGLE uses to commit it.

``unique_write`` says: every ``(target, slot)`` is written by exactly one source per
step. The commit is therefore a plain store, the result is deterministic with zero
atomics, and bit-exact comparison is a legitimate gate mode.

``accumulate`` is the second value: every ``(target, slot)`` may be written by more
than one source per step, and the committed result is the carried base plus an
order-unspecified sum of contributions. Its gate mode is band-tolerance, never
bit-exact — ``unique_write`` is the only value bit-exact comparison applies to.

Mechanism — atomic add, block-reduce-then-atomic, segmented reduce — is EAGLE's own
business, chosen on evidence and invisible to the wire. Several of those are simply
different ways to compute the *same* ``accumulate`` contract, faster or more
deterministically; promoting them to schema vocabulary would turn every performance
experiment into a schema event. Changing executors is never a schema change. A third
value will be minted only when a contract genuinely differs again from both of these,
and it will be named after that contract, together with the executor that can
actually run it.

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

.. _cpp-scope-boundary:

C++ reference-consumer scope boundary
--------------------------------------

The two C++ ``PluginRegistry`` reference consumers
(``plugin/plugin_registry/registry.h``, the CUDA path; ``plugin/host_registry.h``,
the CPU path) implement the schema-v1 *binding* surface completely and
correctly, but they do not parse every field the producer emits and the Python
loader reads. Concretely, C++ never reads:

* ``vec_widths`` — the per-name vector width. The C++ registries **never
  read it**, and the underlying binding primitive is not width-limited to
  begin with: ``GRefMirror`` is documented as width-**independent**
  (``gref_abi.h``: "the struct is width-independent... width only changes in
  in-kernel indexing"), and ``bind_vector`` takes a raw device pointer plus a
  sample count — no width parameter at all. Buffer sizing is therefore
  entirely the **embedding host's contract**, invisible to the registry. A
  mis-sized binding (e.g. handing a width-4 buffer to a kernel compiled
  expecting width-3) is a **silent out-of-bounds device read** — nothing in
  ``sidecar.h`` / ``manifest.h`` / the registry can catch it, because the one
  piece of information that would (``vec_widths``) is never consulted. The
  shipped demos (``plugin/plugin_host.cpp``,
  ``plugin/graph_inject/inject_demo.cu``) happen to allocate width-3
  ``(3, N)`` buffers, but that is a property of the demo code, not a limit of
  the registry or the ABI. A width-generic embedding must size its own
  buffers correctly; the registry will not warn it if it gets this wrong.
* ``accumulate`` / ``sink`` — the vector-pattern accumulate-sink metadata.
  Both are Python-only; the C++ vector registry always accumulates.
* the base convenience arrays ``vector_inputs`` / ``params`` / ``per_sample`` —
  C++ derives everything it needs to bind purely from ``arg_spec`` (role +
  name, in order), which is a strict superset of the same information keyed by
  role. This is not a functional gap: ``sidecar.h``'s own header comment
  explains the design ("we only need two fields... kernel name and
  arg_spec").

In short: the C++ registries bind purely from ``arg_spec`` plus raw pointers
and trust the embedding host for buffer sizing (a silent-corruption hazard,
not a missing capability — see the ``vec_widths`` note above), and their
binding surface is ``double``-typed only (a separate, genuine capability
limit — see :ref:`cpp-f32-deferred` below). Treat them as the reference
implementation of the binding *protocol*, not as something that validates
every hint the producer is free to emit.

.. _cpp-f32-deferred:

float32 on the C++ registry: a named deferred capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``eagle::cuda::PluginRegistry``'s entire binding surface is ``double``-typed:
the bound-buffer struct's data pointer (``VecBinding::data``), ``bind_vector``,
``bind_uniform``, and ``consolidate`` all take ``double*`` / ``double``
(``plugin/plugin_registry/registry.h``). ``PluginRegistry::from_manifest``
therefore rejects any sidecar with ``scalar_type == "float32"`` before it ever
loads the artifact. **That rejection is correct fail-fast, not a bug** — the
guard has been present since eagle's initial commit (``fa6bf85``) and is not a
regression.

It is deliberately a *different* shape of gap from the guard sitting right next
to it in the same function, which *admits* ``scalar_type == "softdouble"``: a
SoftDouble PTX is bit-identical IEEE-754 ``float64``
(``aether``'s ``SoftDouble`` carries a ``static_assert(sizeof(SoftDouble) == 8)``,
"bit-compatible with IEEE 754 double"; arrays, uniforms, and the ``GRef`` ABI
all stay float64-shaped end to end), so it loads and launches through this same
``double``-typed surface with **zero widening** — confirmed by a load-and-run
probe (``eagle/tests/test_PluginRegistryDtype.cu``,
``PluginRegistryDtypeTest.SoftdoubleLoadsAndLaunches``). float32 is genuinely
different: an f32 artifact needs float-*typed* buffers and 4-byte uniform
packing, which changes the layout the registry hands the kernel — a real
capability gap, not a wording problem. Do not read the two rejections as the
same kind of gap: softdouble's guard was removed because the wire-layout
assumption behind it was wrong; float32's guard stays because that assumption
is right.

Nothing here blocks float32 *users* today — eagle's Python/torch device path
(:class:`eagle.LoadedVector` / :class:`eagle.LoadedPure`) already supports
float32 and is the current answer for an f32 consumer. What is deferred is
specifically *this* C++ registry's own binding surface.

**Trigger.** Build float-typed buffers and 4-byte uniform packing for
``eagle::cuda::PluginRegistry`` when the first genuine C++ consumer of an f32
artifact appears — foreseeably a future neural-engine deployment that reaches
the C++ embedding path rather than the Python/torch one. Until that trigger
fires this stays a recorded, deliberate deferral, not an oversight: the
trigger above is the condition for closing it.

**Rough size.** A float-typed twin of the ``double``-typed binding surface —
the bound-buffer struct, ``bind_vector`` / ``bind_uniform`` / ``consolidate``,
and 4-byte uniform packing through the per-plugin argument-packing path in
``inject()`` — plus tests. Small-to-medium: on the order of the existing
``double`` surface it would mirror (that surface, header comments and
``inject()`` included, is a few hundred lines), not a one-line relaxation,
because the packed-argument layout ``inject()`` builds per plugin is
width-specific end to end, not just pointer-typed.

Compatibility policy
--------------------

Validation happens at **load time**, before any launch, and follows two rules:

**Backward-lenient.**
   An artifact with **no** ``schema_version`` is treated as **v1**, so every
   pre-freeze manifest / sidecar stays loadable. For a manifest, the legacy
   ``version`` key is honored as a fallback. (Python:
   :func:`eagle.roles.check_schema_version`; C++: the ``kPluginSchemaVersion``
   gate parsing an absent tag as ``0`` ⇒ v1.)

**Forward-strict.**
   A loader **rejects**:

   * any ``arg_spec`` role outside the 9-role vocabulary
     (:func:`eagle.roles.validate_roles` / ``is_valid_role``);
   * any ``schema_version`` **greater than** the loader supports;
   * any ``scalar_type`` outside ``{"float64", "float32", "softdouble"}``
     (Python: the ``SCALAR_TYPES`` enum check in
     :class:`eagle.loaded.LoadedKernel`; C++: ``validate_sidecar`` in
     ``plugin/sidecar.h``). An absent ``scalar_type`` stays backward-lenient
     (a pre-dtype sidecar, treated as ``"float64"``) — only a *recognized*
     value is required, not a *present* one; and
   * any ``pattern`` value outside ``{"vector", "pure"}``, at **both** the
     manifest and sidecar level ("the discriminant rule": ``pattern``
     *selects* a schema variant, so every loader that reads it — even one that
     does not dispatch on it — must refuse a value it cannot name, rather than
     silently bind an unfamiliar variant as an ordinary kernel). Python:
     ``registry.py::load_manifest`` (manifest) and
     :meth:`eagle.loaded.LoadedKernel._require_pattern` (sidecar, via each
     ``Loaded*`` subclass). C++: ``PluginRegistry::from_manifest`` in both
     ``plugin/plugin_registry/registry.h`` (device) and ``plugin/host_registry.h``
     (host) for the manifest level; the shared ``validate_sidecar`` in
     ``plugin/sidecar.h`` for the sidecar level (so the public ``add_plugin``
     entry point — which bypasses manifests entirely — is covered too); and
   * any ``aether_abi`` that is **absent** (empty), not just one that is present
     but mismatched — a compatibility *gate*, not an optional hint, so
     presence is required at **every** entry point that binds a sidecar or
     manifest: the two manifest-level loads (``PluginRegistry::from_manifest``
     in ``plugin/plugin_registry/registry.h`` and ``host_registry.h::load``)
     and the sidecar-level, manifest-bypassing ``add_plugin``
     (``plugin/host_registry.h``) all reject an empty ``aether_abi`` the same
     way they reject a mismatched one. The Python peer of
     ``add_plugin`` — :class:`eagle.host_launch.HostPluginLibrary` — was the
     last absence-lenient door in either language (a private inline check,
     not :func:`eagle.abi.check_aether_abi`); harmonized to presence-required
     too, so this is now true of every loader in both languages, not
     just the C++ ones named above.

Both loaders enforce the *same* vocabulary and the *same* version gate. The C++
half is ``plugin/roles.h`` and the Python half is ``eagle/roles.py``; the
cross-check test ``python/tests/test_roles_vocab.py`` asserts the two role lists
are set-equal (and the same for the RECOGNIZED / LAUNCH-CERTIFIED
pattern sets, the buffer-``kind`` vocabulary, and the manifest-entry ``format``
vocabulary), the two ``schema_version`` constants agree, and every committed
fixture uses only canonical roles. ``eagle/python/tests/test_schema_hardening.py``
and ``eagle/tests/test_SchemaHardening.cpp`` fix the exact set of forward-strict
fields — now nine: ``schema_version`` and ``aether_abi`` (class ``gate``);
``kernel`` and ``arg_spec role`` (class ``structural``); ``pattern``,
``scalar_type``, buffer ``kind``, ``derivative.kind``, and manifest-entry
``format`` (class ``discriminant``) — so the strict/lenient split cannot erode
silently, and a CLASS mismatch (a loader implementing a private literal compare
for a classed field outside the shared validation layer) is caught too. See the
frozen-field-list note in each file for the version-bump rule and the full CLASS
definitions.

``pattern``'s *absence* handling is the one place C++ and Python genuinely
diverge, by design, not a gap:

* At the **manifest** level, Python's ``load_manifest`` is absence-**strict**
  (an absent ``pattern`` is rejected, same as an unknown one) — it must dispatch
  to ``LoadedVector`` or ``LoadedPure`` and has no third option. The two C++
  registries stay absence-**lenient** (a pre-freeze manifest that never stamped
  one still loads): neither registry dispatches a variant class off ``pattern``,
  binding is driven entirely by ``arg_spec``, so there is nothing an absent value
  would fail to gate. Pinned mechanically by the shared conformance corpus (a
  manifest with no ``pattern`` key, deliberately also carrying a duplicate
  plugin id: C++ passes the lenient gate and rejects on the duplicate id
  instead; Python's strict gate rejects first) — the ``overrides`` mechanism
  in ``eagle/tests/conformance/row13_manifest_pattern_absence_asymmetry.*``
  encodes both outcomes for one fixture, so a future change that
  accidentally harmonizes the two languages fails a test rather than
  drifting unnoticed.
* At the **sidecar** level, C++'s ``validate_sidecar`` is likewise
  absence-lenient, for the same arg_spec-driven reason (Python's shared
  validator agrees — both languages accept a ``pattern``-less sidecar at
  this layer, pinned by
  ``eagle/tests/conformance/row14_sidecar_pattern_absence_lenient.*``).
  Layered on top, Python is per-``Loaded*``-class: ``LoadedVector`` defaults
  an absent value to ``"vector"`` (lenient), while ``LoadedPure`` requires an
  exact match (no default) — this further split has no C++ counterpart to
  diverge from (C++ has no per-kind sidecar loader at all), and pinning it
  needs a real device module load (``_require_pattern`` runs after
  ``LoadedKernel.__init__``'s ``cupy.RawModule`` load), which the corpus's
  GPU-free design cannot accommodate. It is instead pinned directly —
  ``test_loaded_vector_defaults_absent_pattern_to_vector`` /
  ``test_loaded_pure_rejects_absent_pattern`` in
  ``eagle/python/tests/test_schema_hardening.py`` — against the committed
  ``gravity``/``bump`` PTX fixtures. The asymmetry itself remains locked
  behaviour, not a defect.

This is narrower than an earlier state: previously C++ did not even reject an
*unrecognized* ``pattern`` value (a manifest-only fix would also have left the
sidecar-level, manifest-bypassing ``add_plugin`` entry point open — see
``eagle/tests/test_SchemaHardening.cpp`` and ``test_PluginRegistryManifest.cu``
for the registry-level rejection tests, and ``test_HostPlugin.cpp``-adjacent
tests in ``test_SchemaHardening.cpp`` for the ``add_plugin`` path).

.. _aether-abi-tag:

The ``aether_abi`` tag is separate
-----------------------------------

``aether_abi`` (``"aether-abi/1"``) is a **distinct, ABI-stable** tag — the by-value
binary-layout contract of the ``GRef`` / ``HandleT`` POD mirrors the artifact was
built against. It is **orthogonal to**
``schema_version``: the schema version governs the *JSON* shape and the role
vocabulary, whereas ``aether_abi`` governs the *binary* struct layout a loader binds
by value. A loader rejects an ``aether_abi`` mismatch before binding any by-value
struct (Python: :func:`eagle.abi.check_aether_abi`; C++: the ``EAGLE_AETHER_ABI``
static-assert gates). The value is historical and must not change without a
coordinated C++/Python bump — pre-split artifacts remain loadable precisely
because it is stable.

.. _execution-axis-v2:

The v2 execution axis (aether-abi/2) and eagle's Python plan/exec API
-----------------------------------------------------------------------

Schema version 2 adds one new axis to the manifest: which **execution
structure(s)** — a single device kernel launch, an OpenMP team of CPU
threads, an MPI rank partition, and (planned) an NCCL device group —
may drive a plugin body, and how that body is allowed to touch data outside
its own sample. A schema-v2 manifest tags itself ``"schema_version": 2`` and
``"aether_abi": "aether-abi/2"``, and MUST additionally carry three new
top-level, flat keys (absence is a load refusal, never a silent fallback to
v1 behaviour):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``exec_targets``
     - A non-empty list drawn from ``["device", "host"]`` (no duplicates) —
       which target(s) the plugin ships an entry point for.
   * - ``exec_access``
     - How the body touches data: one of ``sample_local`` (reads/writes only
       its own sample — an ordinary elementwise kernel), ``cross_sample_read``
       (reads another sample's data, e.g. a lookup table), ``cross_sample_write``
       (writes into a location that is not its own sample's column — an
       accumulate/scatter), or ``mapreduce`` (produces a per-sample partial
       that eagle combines with a declared reduction operator).
   * - ``exec_op``
     - The reduction operator (``sum`` / ``times`` / ``max`` / ``land``) —
       **required** when ``exec_access == "mapreduce"``, and **forbidden**
       otherwise.

A v2 artifact's entry points additionally take an explicit **partition
triple** ``{base, count, nSamples}`` after their ordinary role arguments —
eagle, never the plugin, decides how a run is split and drives the *same*
body once per partition (a device kernel launch, or one CPU thread team's
tile). The ``accum_out`` role (see :ref:`role-vocab`) names the cross-sample
accumulate plane a ``cross_sample_write`` body writes into.

**Legacy v1 artifacts are unaffected.** A ``schema_version: 1`` /
``aether_abi: "aether-abi/1"`` plugin carries none of the three keys above
(carrying one is itself a strict-key violation) and is always run
whole-view, on a single structure — precisely today's behaviour, byte for
byte. ``eagle.abi.check_aether_abi`` accepts both tags; only a v2-tagged
artifact may be split across more than one partition.

Placement legality is a function of ``exec_access`` and the chosen
structure alone, and is refused at **plan time**, naming the rule, never
guessed or discovered mid-launch: ``sample_local`` / ``cross_sample_read`` /
``mapreduce`` run under any implemented structure; ``cross_sample_write``
runs on a single-device structure only (splitting it across partitions would
need a partial-accumulate combine step not yet specified) — the same
plain-language rule :func:`eagle.exec.check_placement` enforces mechanically.

Driving a v2 body from Python — ``eagle.plan`` / ``eagle.exec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`eagle.exec` is the execution-structure vocabulary: the
:data:`~eagle.exec.DeviceKernel` / :data:`~eagle.exec.HostTeam` /
:data:`~eagle.exec.RankPartition` / :data:`~eagle.exec.DeviceGroup`
singletons, plus :func:`~eagle.exec.check_placement` and
:func:`~eagle.exec.check_layout_sizes` (a self-check: a v2 artifact's
exported ``eagle_layout_sizes`` must agree with this build's own POD sizes,
or the load is refused naming the disagreeing field — a matching
``aether_abi`` tag alone cannot catch a stale or wrong-arch binding).
:mod:`eagle.plan` is where a run is PLANNED: :func:`eagle.plan.plan` maps a
plugin, a chosen structure, and an explicit partitioning to a ``Plan``, and
``Plan.run(**kw)`` drives it:

.. code-block:: python

   import eagle.exec as eexec
   from eagle import plan as eplan

   # Selection is always EXPLICIT (no residency-driven auto-choice yet) --
   # the caller names the structure every time.
   whole = eplan.plan(my_v2_kernel, structure=eexec.DeviceKernel).run(x=x, a=2.0, b=-1.5)

   # Splitting a `sample_local` body into partitions never changes the
   # per-sample answer -- eagle drives the identical body per partition.
   split = eplan.plan(
       my_v2_kernel,
       structure=eexec.DeviceKernel,
       partitions=[(0, 32, 64), (32, 32, 64)],
   ).run(x=x, a=2.0, b=-1.5)

   # The same body under HostTeam (OpenMP) instead of DeviceKernel (CUDA):
   host = eplan.plan(my_v2_kernel, structure=eexec.HostTeam).run(x=x, a=2.0, b=-1.5)

   # An illegal placement is refused at plan() time, naming the rule:
   eplan.plan(
       my_v2_kernel, structure=eexec.DeviceKernel,
       partitions=[(0, 32, 64), (32, 32, 64)], exec_access="cross_sample_write",
   )  # -> ValueError: "... exec_access 'cross_sample_write' is legal on
     #    single-device structures only ..."

``partitions=`` is a list of explicit ``(base, count, n_samples)`` triples
(contiguous, covering ``[0, n_samples)``); omit it (with ``npartitions=``
left at its default of 1) for a deferred whole-view run, whose size is
resolved from the arrays passed to ``.run()``. See
``eagle/python/tests/test_exec_contract_rows.py`` for the certification rows
this API is held to (bit-exact partition identity for a ``sample_local``
body, a ruled host/device tolerance band, and the two refusal cases above).

Running across MPI ranks — ``RankPartition``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:data:`~eagle.exec.RankPartition` is the multi-process structure. Rank *r* of
*R* runs the *r*-th **contiguous** sub-partition of one whole partition
through an **inner** structure — ``HostTeam`` or ``DeviceKernel`` — carrying
its own ``{base, count, nSamples}`` triple, with ``nSamples`` unchanged: a
rank is one more cut of the same run, not a smaller run. Inputs are
**replicated** (which is why a ``cross_sample_read`` body is legal here: the
tables it reads are whole on every rank), and outputs are **gathered**, so
every rank ends holding the whole plane, bit-identical to a single whole run.
``cross_sample_write`` stays refused — two ranks accumulating into one shared
target would need a partial-accumulate combine that is not specified.

.. code-block:: python

   # The plan's partitions stay WHOLE partitions: the rank cut happens INSIDE
   # the structure, so the same plan describes the run at any world size.
   whole = eplan.plan(
       my_v2_kernel, structure=eexec.RankPartition, inner=eexec.HostTeam,
   ).run(x=x, a=2.0, b=-1.5)          # every rank gets the WHOLE output plane

   rank, size = eexec.RankPartition.world()        # (0, 1) outside a launcher
   mine = eexec.RankPartition.local(eexec.Partition.whole(len(x)))

A ``mapreduce`` body folds its own rank's partials and then combines the *R*
partials in a fixed **rank order**
(:meth:`~eagle.exec.RankPartition.allgather_fold`) — never an
``MPI_Allreduce``, whose combine order is the implementation's business, so
two ranks could disagree in the last bits.

This structure lives in a **separate, optional** compiled extension
(``eagle._mpi``, built with ``-DEAGLE_PYTHON_MPI=ON``; ``AUTO``, the default,
builds it wherever CMake finds an MPI). ``eagle._core`` therefore never links
``libmpi``, and ``import eagle`` costs nothing to a user with no MPI
installed; where the extension is absent, every ``RankPartition`` call
refuses with a ``RuntimeError`` naming that build switch. Note that
initialising MPI is not a passive act — Open MPI installs process-wide memory
hooks — so drive this structure from a process dedicated to it (an ``mpirun``
world, or a short-lived child), not from inside a long-running session that
is also using CUDA. The certification rows are
``eagle/python/tests/test_rank_partition.py`` (single process) and the
two-rank bed ``eagle/python/tests/mpi/`` (run through its own gate,
``tests/mpi/check_rank_bed.sh``).

Reserved capability fields
--------------------------

The sidecar schema reserves an **optional** ``launch`` object for future
launch-configuration metadata:

.. code-block:: json

   "launch": {
     "block": 256,
     "shmem_bytes": 0,
     "grid": {}
   }

* ``block`` — int (reserved: thread-block size hint)
* ``shmem_bytes`` — int (reserved: dynamic shared-memory bytes hint)
* ``grid`` — object (reserved: freeform grid-configuration hint)

These fields are **reserved in v1: unpopulated, and the current loaders ignore
them.** They may be populated in a future minor revision **without a
``schema_version`` bump**. The schema keeps ``additionalProperties`` permissive on
both the ``launch`` object and the top-level object so unknown/future keys do not
fail validation of the reserved surface.

This ignore-unknown guarantee — for ``launch`` specifically, and for any
unrecognized top-level sidecar/manifest key generally — is exercised by an
executable test on both loaders, not just asserted in prose:
``eagle/python/tests/test_schema_hardening.py`` and
``eagle/tests/test_SchemaHardening.cpp``. No ``launch`` hint is populated by the
producer today; adding one is a deliberate design decision, not a casual
addition.

.. note::

   Do not confuse this reserved ``launch`` sidecar field with eagle's own,
   unrelated ``idealBlockSize`` concept in the native CUDA-graph machinery
   (``eagle/cuda/{Graph.h, CapturedGraph.h, Launcher.h, ComputeBlocks.h}``) — a
   per-captured-kernel blockDim-reconciliation cap used when composing native
   eagle graph nodes. The two share a naming convention, not a design link.

The schema files
----------------

.. dropdown:: manifest-v1.schema.json
   :icon: file-code

   .. literalinclude:: /schemas/manifest-v1.schema.json
      :language: json

.. dropdown:: sidecar-v1.schema.json
   :icon: file-code

   .. literalinclude:: /schemas/sidecar-v1.schema.json
      :language: json
