raptor.protocols
=================

Structural-typing contracts (``typing.Protocol``), checked by shape rather
than by inheritance. ``backend`` holds the program-level ``Backend`` /
``Executable`` pair; ``launcher`` holds the kernel-level ``KernelLauncher`` /
``Marshal`` pair, mirroring eagle's public launch/marshal surface 1:1;
``provider`` holds ``ManifestProvider``, the one kernel-provider facet the
family ships today.

.. autosummary::
   :toctree: generated/
   :recursive:

   raptor.protocols
