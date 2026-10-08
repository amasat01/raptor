raptor.schema
=============

The two-layer kernel-manifest schema. ``manifest`` is the bird-neutral core
every producer writes against; ``blocks`` is the ``neural_block`` extension a
producer never has to import; ``dtypes`` carries the scalar-type tag
vocabulary both layers share.

See :doc:`../devguide/plugin_schema` for the full manifest/sidecar reference
(field tables, the role vocabulary, and the normative JSON Schema files).

.. autosummary::
   :toctree: generated/
   :recursive:

   raptor.schema
