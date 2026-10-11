.. _deployment-manifest:

The deployment manifest
=======================

.. contents:: On this page
   :local:
   :depth: 2

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
       :doc:`compatibility`.
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
