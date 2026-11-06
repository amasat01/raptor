# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Scalar-type tags of the launch/artifact contract.

Values copied VERBATIM from ``eagle/python/eagle/dtypes.py``. The numpy import
is FUNCTION-LOCAL (never at module level) so this module stays import-clean on
a numpy-less interpreter — raptor has zero hard dependencies.
"""

from __future__ import annotations

#: the compiled scalar modes, in sidecar spelling (verbatim value from
#: ``eagle.dtypes.SCALAR_TYPES``).
SCALAR_TYPES = ("float64", "float32", "banded")


def np_dtype(scalar_type: str):
    """Return the numpy dtype of Real-typed arrays for a resolved ``scalar_type``.

    banded: an 8-byte emulated-FP64 cell; user-facing arrays are float64; values
    match float64 within the per-op ULP contract documented in aether, not
    bit-identically (mirrors ``eagle.dtypes.np_dtype``'s ``table`` literal).

    On the wire a banded Real plane is eagle dtype code 13 (8-byte cells, kind 'V'),
    packed and unpacked by eagle (``eagle.banded_pack`` / ``eagle.banded_unpack``);
    the user-facing numpy dtype stays float64.
    """
    import numpy as np

    table = {
        "float64": np.float64,
        "float32": np.float32,
        "banded": np.float64,
    }
    return table[scalar_type]
