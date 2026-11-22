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

* ``slot_shapes`` — the value shape of each slot. The C++ registries **never read it**: ``GRefMirror`` is documented as
  width-**independent**, and ``bind_vector`` takes a raw device pointer plus a sample count. Buffer sizing is therefore entirely the
  **embedding host's contract**; a mis-sized binding is a **silent out-of-bounds device read** that nothing in the registry can
  catch, because the one piece of information that would (``slot_shapes``) is never consulted.
* ``accumulate`` / ``sink`` — the vector-pattern accumulate-sink metadata.
  Both are Python-only; the C++ vector registry always accumulates.
* the base convenience array ``params`` —
  C++ derives everything it needs to bind purely from ``arg_spec`` (role +
  name, in order), which is a strict superset of the same information keyed by
  role. This is not a functional gap: ``sidecar.h``'s own header comment
  explains the design ("we only need two fields... kernel name and
  arg_spec").

In short: the C++ registries bind purely from ``arg_spec`` plus raw pointers
and trust the embedding host for buffer sizing (a silent-corruption hazard,
not a missing capability — see the ``slot_shapes`` note above), and their
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
guard has been there from the start and is not a
regression.

It is deliberately a *different* shape of gap from the guard sitting right next
to it in the same function, which *admits* ``scalar_type == "banded"``: a
banded PTX uses an 8-byte emulated-FP64 cell (``aether``'s ``BandedReal`` carries a
``static_assert(sizeof(BandedReal) == 8)``; its values match ``float64`` within a
per-op ULP bound, not bit-identically; arrays, uniforms, and the ``GRef`` ABI
all stay float64-shaped end to end), so it loads and launches through this same
``double``-typed surface with **zero widening** — confirmed by a load-and-run
probe (``eagle/tests/test_PluginRegistryDtype.cu``,
``PluginRegistryDtypeTest.BandedLoadsAndLaunches``). float32 is genuinely
different: an f32 artifact needs float-*typed* buffers and 4-byte uniform
packing, which changes the layout the registry hands the kernel — a real
capability gap, not a wording problem. Do not read the two rejections as the
same kind of gap: banded's guard was removed because the wire-layout
assumption behind it was wrong; float32's guard stays because that assumption
is right.

Nothing here blocks float32 *users* today — eagle's Python/torch device path
(:class:`eagle.LoadedVector` / :class:`eagle.LoadedPure`) already supports
float32 and is the current answer for an f32 consumer. What is deferred is
specifically *this* C++ registry's own binding surface.

**Scope.** Float32 on the C++ registry is not supported: it would need
float-typed buffers and 4-byte uniform packing in ``eagle::cuda::PluginRegistry``.
The Python path supports float32 today.
