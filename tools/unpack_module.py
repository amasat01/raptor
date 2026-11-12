# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Unpack the compiled module of the newest wheel in a folder into the package: ``python tools/unpack_module.py dist raptor``.

The module (``raptor/_core*.so``) is not tracked by git; ``make build`` puts it where the source tree imports it from."""

import os
import sys
import zipfile
from pathlib import Path


def main(argv) -> int:
    wheels = sorted(Path(argv[0]).glob("raptor_core-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print(f"no raptor_core wheel in {argv[0]}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(wheels[-1]) as z:
        names = [n for n in z.namelist() if n.startswith("raptor/_core.") and n.endswith((".so", ".pyd"))]
        for n in names:
            dst = Path(argv[1]) / Path(n).name
            tmp = dst.with_name(dst.name + ".new")
            tmp.write_bytes(z.read(n))
            os.replace(tmp, dst)             # a new file, never a rewrite in place: macOS kills a process that maps a changed signed binary
    print(f"{wheels[-1].name}: unpacked {', '.join(names) or 'nothing'} into {argv[1]}")
    return 0 if names else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
