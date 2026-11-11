# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dialect extension protocol: a node kind is DECLARED BY DATA (a :class:`KindDecl`), never by subclassing.

Everything here is a function over a :class:`Registry` that the caller holds and passes in (no module-level table of kinds). A
declaration is plain data (JSON-shaped); :func:`declare` validates it and has the registry build ONE class for the kind, the
Python presentation of the declaration (a node holds its declaration; a pass asks :func:`is_a` and :func:`trait`)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from raptor import _core
from .nodes import KIND_TRAITS, Assign, At, HawkError, Node, Sink, WideWrite, plain_in, plain_out

#: The core kinds a declared kind may refine: name -> (family, class, role of a sink, wire kind).
def _core_kinds() -> MappingProxyType:
    """The core kinds a declared kind may refine, read from the manifest of the core (its rows marked ``refinable``)."""
    from . import nodes
    return MappingProxyType({row["name"]: (row["family"], getattr(nodes, row["class"]), row.get("role"), row.get("wire"))
                             for row in nodes._TABLE["man"]["kinds"] if row.get("refinable")})


CORE = _core_kinds()
TRAITS = KIND_TRAITS


class Registry:
    """The declared kinds of one process or one test, as a value: ``kinds`` maps a name to its :class:`KindDecl`, ``classes``
    to the class built for it, ``primitives`` to the registered :class:`~raptor.rules.primitive.PrimitiveDecl`."""

    __slots__ = ("kinds", "classes", "primitives")

    def __init__(self) -> None:
        self.kinds: dict = {}
        self.classes: dict = {}
        self.primitives: dict = {}


@dataclass(frozen=True)
class KindDecl:
    name: str
    dialect: str
    ext: str
    family: str
    refines: str | None
    wire: str
    operands: tuple          # (min, max or None, families)
    fields: tuple            # ((name, type, default, required), ...)
    result: tuple
    identity: tuple | None
    traits: tuple            # ((name, bool), ...)
    spelling: tuple | None   # (dialect, name, args)
    diff: str
    no_writer: bool = False


def load_manifest(reg: Registry, manifest: dict, *, first_party: tuple = ()) -> tuple:
    """THE way the ``kinds`` arm of a manifest is loaded, raptor's own and an extension's alike: a row that names a ``class`` is a core
    kind (its class is generated from this very row at import: it is only checked to exist), any other row is declared into ``reg``.
    Returns the declarations of the rows it declared."""
    from . import nodes
    out = []
    for row in manifest["kinds"]:
        if "class" in row:
            if not hasattr(nodes, row["class"]):
                raise HawkError(f"manifest {manifest.get('name')!r}: core kind {row.get('name')!r} names class {row['class']!r}, which is not generated")
            continue
        out.append(declare(reg, {"ext": manifest["name"], "dialect": manifest.get("dialect", ""), **row}, first_party=first_party))
    return tuple(out)


def _refuse(name: str, why: str) -> HawkError:
    return HawkError(f"kind {name!r}: {why}")


def declare(reg: Registry, data: dict, *, first_party: tuple = ()) -> KindDecl:
    """Validate ``data`` (a KindDecl as plain data) and add it to ``reg``; an equal re-declaration returns the held one. The checks
    and their refusals are the core's (``_core.kind_declare``)."""
    decl, root = _core.kind_declare(reg.kinds, CORE, data, first_party, KindDecl)
    held = reg.kinds.get(decl.name)
    if held is not None:
        if held == decl:
            return held
        raise _refuse(decl.name, f"already declared by extension {held.ext!r} with a different declaration")
    reg.kinds[decl.name] = decl
    reg.classes[decl.name] = _build_class(reg, decl, root)
    return decl


def _build_class(reg: Registry, decl: KindDecl, root: str):
    base = reg.classes[decl.refines] if decl.refines in reg.kinds else CORE[decl.refines or ("sink" if decl.family == "sink" else "node")][1]
    own = tuple(f[0] for f in decl.fields)
    ns: dict = {"__slots__": own, "decl": decl, "decl_root": root, "__module__": __name__,
                "decl_chain": (decl.name, *getattr(base, "decl_chain", ())),
                **{t: v for t, v in decl.traits}}
    if root == "node":
        ns["kind"] = decl.wire
    if decl.identity is not None:
        ns["dedup_extra"] = property(lambda self: tuple(getattr(self, n) for n in self.decl.identity))
    names = [f[0] for f in decl.fields]

    def state(self) -> dict:
        out = {"name": self.name, "role": self.role} if self.decl_root == "sink" else {}
        for f in _all_fields_of(self):
            out[f] = plain_out(getattr(self, f))
        return out

    def from_state(cls, ttype, operands, state):
        return _make(cls, ttype, operands, **{k: plain_in(v) for k, v in state.items()})

    ns.update(state=state, from_state=classmethod(from_state))
    return type("".join(p.capitalize() for p in re.split(r"[._]", decl.name)), (base,), ns)


def _all_fields_of(node) -> list:
    return [f[0] for c in type(node).__mro__ if "decl" in c.__dict__ for f in c.__dict__["decl"].fields]


def _make(cls, ttype, operands, **fields):
    """The ONE builder of a declared kind's node: the core verifies arity, families, result type and field types."""
    d = cls.decl
    core_role = CORE[_core_of(cls)][2] if cls.decl_root == "sink" else None
    ttype, operands, fields = _core.kind_check(cls, ttype, tuple(operands), fields, core_role)
    obj = object.__new__(cls)
    if cls.decl_root == "sink":
        Sink.__init__(obj, d.wire, fields.get("role", core_role), fields["name"], operands[0], operands[1] if len(operands) > 1 else None, ttype)
    elif cls.decl_root == "at":
        At.__init__(obj, operands[0], operands[1], ttype)
        object.__setattr__(obj, "kind", d.wire)   # still inside the builder: the declared wire replaces At's, before the node escapes
    else:
        Node.__init__(obj, ttype, operands)
    for fname in (f[0] for c in cls.__mro__ if "decl" in c.__dict__ for f in c.__dict__["decl"].fields):
        setattr(obj, fname, fields[fname])
    return obj


def _core_of(cls) -> str:
    for c in cls.__mro__:
        for k, v in CORE.items():
            if c is v[1] and k != "node":
                return k
    return "node"


def make_kind(reg: Registry, kind_: str, ttype, operands=(), /, **fields) -> Node:
    """A node of the declared kind ``kind_``; refuses by name an undeclared kind."""
    cls = reg.classes.get(kind_)
    if cls is None:
        raise HawkError(f"kind {kind_!r} is not declared")
    return _make(cls, ttype, operands, **fields)


def kind_name(node: Any) -> str | None:
    """The declared kind of ``node`` (its own, not a refined one's), or ``None`` for a core node."""
    chain = getattr(node, "decl_chain", None)
    return chain[0] if chain else None


def is_a(node: Any, name: str) -> bool:
    """Whether ``node`` is of kind ``name``: declared (itself or a refinement of it) or core."""
    if name in getattr(node, "decl_chain", ()):
        return True
    return name in CORE and isinstance(node, CORE[name][1])


def trait(node: Any, name: str) -> bool:
    if name not in TRAITS:
        raise HawkError(f"unknown trait {name!r}")
    return bool(getattr(node, name, False))
