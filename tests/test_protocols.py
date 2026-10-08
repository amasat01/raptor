# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Protocol sanity: runtime_checkable, CapabilityError, and method
surfaces mirroring eagle's public names 1:1 / the family's own naming table.
Signatures here are pinned by review — any deviation from the locked names
would show up as a failing ``hasattr`` below."""

from __future__ import annotations

from raptor.protocols import (
    CAPABILITIES,
    Backend,
    CapabilityError,
    Executable,
    KernelLauncher,
    ManifestProvider,
    Marshal,
)

ALL_PROTOCOLS = (
    Backend,
    Executable,
    KernelLauncher,
    Marshal,
    ManifestProvider,
)


def test_capability_error_is_a_not_implemented_error():
    assert issubclass(CapabilityError, NotImplementedError)


def test_capabilities_vocabulary():
    assert CAPABILITIES == frozenset({"forward", "vjp", "jvp", "train_step"})


def test_all_protocols_are_runtime_checkable():
    for proto in ALL_PROTOCOLS:
        assert getattr(proto, "_is_runtime_protocol", False), proto


def test_backend_surface():
    assert hasattr(Backend, "capabilities")
    assert hasattr(Backend, "compile")


def test_executable_surface():
    for name in ("forward", "vjp", "jvp", "parameters"):
        assert hasattr(Executable, name)


def test_kernel_launcher_mirrors_eagle_launch_names_1to1():
    for name in ("launch", "assemble_args", "pure_origin", "pure_prepare"):
        assert hasattr(KernelLauncher, name)
    # DEFAULT_BLOCK / LaunchMixin are explicitly NOT lifted.
    assert not hasattr(KernelLauncher, "DEFAULT_BLOCK")
    assert not hasattr(KernelLauncher, "LaunchMixin")


def test_marshal_mirrors_eagle_marshal_names_1to1():
    for name in (
        "coerce_vec_inputs",
        "coerce_mat_inputs",
        "coerce_per_sample",
        "coerce_uniforms",
        "coerce_tables",
        "coerce_terminated",
        "require_n",
    ):
        assert hasattr(Marshal, name)


def test_manifest_provider_surface():
    assert hasattr(ManifestProvider, "build")


def test_executable_isinstance_smoke():
    """A minimal concrete implementation structurally satisfies Executable —
    proves the Protocol is well-formed and not accidentally abstract."""

    class _Dummy:
        def forward(self, inputs):
            return inputs

        def vjp(self, cotangents):
            return cotangents

        def jvp(self, tangents):
            return tangents

        def parameters(self):
            return ()

    assert isinstance(_Dummy(), Executable)


def test_kernel_launcher_isinstance_smoke():
    class _DummyLauncher:
        def launch(self, *a, **kw):
            ...

        def assemble_args(self, *a, **kw):
            ...

        def pure_origin(self, *a, **kw):
            ...

        def pure_prepare(self, *a, **kw):
            ...

    assert isinstance(_DummyLauncher(), KernelLauncher)
