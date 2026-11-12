# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The refusal for a toolchain or host feature that is absent here."""

from __future__ import annotations

from ..errors import RaptorError


class ToolchainAbsent(RaptorError):
    """A toolchain or host feature this build needs is absent here.

    ``toolchain`` names which one (``"cppcheck"``, ``"x86-64"``, a compiler or a device, ...).
    It is a :class:`~raptor.errors.RaptorError`, so a caller that catches that, or reads the message, is unaffected; a gate
    that wants the early exit reads the type and the name.
    """

    toolchain = ""

    def __init__(self, message: str = "", *, toolchain: str = "") -> None:
        super().__init__(message)
        if toolchain:
            self.toolchain = toolchain
