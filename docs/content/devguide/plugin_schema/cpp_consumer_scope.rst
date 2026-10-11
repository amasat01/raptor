.. _cpp-scope-boundary:

C++ reference-consumer scope boundary
======================================

.. contents:: On this page
   :local:
   :depth: 2

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
---------------------------------------------------------

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
