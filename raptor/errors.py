# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The family's one refusal type.

Every refusal raised by raptor, and by the pieces built on it, is a :class:`RaptorError`: a plain ``Exception`` whose
message says what was refused and why. It is defined here once; the compiled core looks it up by this name.
"""

from __future__ import annotations


class RaptorError(Exception):
    """A refusal: the input is not something the contract accepts, and the message says why."""
