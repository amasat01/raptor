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
SCALAR_TYPES = ("float64", "float32", "softdouble")


def np_dtype(scalar_type: str):
    """Return the numpy dtype of Real-typed arrays for a resolved ``scalar_type``.

    SoftDouble is bit-identical IEEE float64 — the array interface is float64
    (mirrors ``eagle.dtypes.np_dtype``'s ``table`` literal).
    """
    import numpy as np

    table = {
        "float64": np.float64,
        "float32": np.float32,
        "softdouble": np.float64,
    }
    return table[scalar_type]
