The sidecar
===========

.. contents:: On this page
   :local:
   :depth: 2

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
------------------

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
