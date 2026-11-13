# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The records of the ``segmented`` dispatch strategy: its unit records. The plane names and the split itself live in the
lowering wing and its passes."""

from __future__ import annotations

from . import _gen
from .. import _core

# pure-data records of the manifest (raptor/ir/core.json, arm `records`, home 'segment')
globals().update(_gen.records(_gen.parse(_core.manifest()), __name__, 'segment'))
