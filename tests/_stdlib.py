"""The names of the standard library's top-level modules, on every supported interpreter (``sys.stdlib_module_names`` is 3.10+)."""

import os
import pkgutil
import sys
import sysconfig


def names() -> set:
    got = getattr(sys, "stdlib_module_names", None)
    if got is not None:
        return set(got)
    return {m.name for m in pkgutil.iter_modules([sysconfig.get_paths()["stdlib"], os.path.join(sysconfig.get_paths()["stdlib"], "lib-dynload")])} | set(sys.builtin_module_names) | {"__future__"}
