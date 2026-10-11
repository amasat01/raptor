Reserved capability fields
==========================

.. contents:: On this page
   :local:
   :depth: 2

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
