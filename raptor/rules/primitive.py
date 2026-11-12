# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A user primitive and its derivative rule, as DATA: a rule is a TEMPLATE (ordinary IR over placeholder leaves), registered
per input signature into a :class:`~raptor.ir.kinds.Registry` the caller passes in. Nothing callable is stored: instantiation
substitutes a node's operands for the placeholders, structurally."""

from __future__ import annotations

from dataclasses import dataclass

from .. import _core
from ..ir.kinds import CORE, Registry
from ..ir.nodes import PRIMITIVE_KIND, Leaf, Node, Primitive
from ..errors import RaptorError
from ..ir.types import TensorType
from .core import zero_like

PREFIX = "$"                 # the name of a placeholder leaf starts with it; no plane or uniform may
DIRECTIONS = ("vjp", "jvp")
_WORDS = {"vjp": "reverse (vjp)", "jvp": "forward (jvp)"}


@dataclass(frozen=True)
class PrimitiveDecl:
    """What registration declares of one primitive: its name, its extension id and the directions it supplies a rule for."""

    name: str
    ext: str
    directions: tuple


def placeholder(tag: str, ttype: TensorType) -> Leaf:
    """The free leaf a template reads: ``in<i>``, ``out``, ``cot`` (the incoming adjoint), ``dot<i>`` (an input's tangent)."""
    return Leaf("uniform", "uniform", PREFIX + tag, ttype)


def signature_of(node: Primitive) -> tuple:
    """The input types and the output type of one primitive node: the key of its rule instance."""
    return tuple(x.ttype for x in node.inputs), node.ttype


def declare_primitive(reg: Registry, name: str, *, ext: str, directions: tuple) -> PrimitiveDecl:
    """Register a primitive's name; an equal re-registration returns the held one, a different one refuses."""
    if not isinstance(name, str) or not name:
        raise RaptorError("a primitive is registered by NAME: the name is the boundary node's identity in the walk's digest")
    if name == PRIMITIVE_KIND or name in CORE or name in reg.kinds:
        raise RaptorError(f"primitive {name!r}: the name is reserved (a core or declared kind)")
    decl = PrimitiveDecl(name, ext, tuple(directions))
    held = reg.primitives.get(name)
    if held is not None:
        if held[0] == decl:
            return decl
        raise RaptorError(f"primitive {name!r} is already registered (by extension {held[0].ext!r}, rules {held[0].directions}); a "
                        "second registration under the same name is refused because the name is what identifies the boundary "
                        "node in Walk.digest")
    reg.primitives[name] = (decl, {})
    return decl


def register_rule(reg: Registry, name: str, signature: tuple, direction: str, templates) -> None:
    """Register a rule instance for ``signature``: vjp = a tuple of templates, one per input; jvp = one template. Validates
    arity, free leaves and the type each template owes (in the core); an equal re-registration is a no-op, a different one
    refuses."""
    held = reg.primitives.get(name)
    if held is None:
        raise RaptorError(f"primitive {name!r} is not registered")
    templates, key = _core.rule_prepare(name, signature, direction, templates, PREFIX)
    instance = held[1].setdefault(signature, {})
    old = instance.get(direction)
    if old is not None and old[0] != key:
        raise RaptorError(f"primitive {name!r} is already registered with a different {direction} rule for this signature")
    instance[direction] = (key, templates)


def has_rule(reg: Registry, node: Primitive, direction: str) -> bool:
    held = reg.primitives.get(node.name)
    return held is not None and direction in held[1].get(signature_of(node), {})


def substitute(template: Node, mapping: dict) -> Node:
    """``template`` with each placeholder leaf replaced by ``mapping[name]``, through ordinary operands and through loop bodies
    alike, in ONE traversal. A subgraph that reads no placeholder is kept as it is; a loop is rebuilt with a fresh index and
    fresh carries (so two instantiations never share loop-local nodes) and its body substituted once."""
    return _core.rule_substitute(template, mapping)


def _supplied(reg: Registry, node: Primitive, direction: str):
    held = reg.primitives.get(node.name)
    if held is None or direction not in held[0].directions:
        raise RaptorError(
            f"primitive {node.name!r} supplies no {_WORDS[direction]} rule, so this direction cannot be taken. HAWK does not fall "
            "back to differentiating THROUGH a primitive's forward: the forward is inlined and the table would happily produce a "
            f"number, which is exactly the number {node.name!r} was declared to replace. Register a {direction} rule for it")
    got = held[1].get(signature_of(node), {}).get(direction)
    if got is None:
        raise RaptorError(f"primitive {node.name!r}: no {direction} rule instance for this signature is registered")
    return got[1]


def vjp_contributions(reg: Registry, node: Primitive, g: Node) -> tuple:
    """``((input, adjoint), ...)`` for one boundary node, from the registered rule instance."""
    templates = _supplied(reg, node, "vjp")
    m = {PREFIX + f"in{i}": x for i, x in enumerate(node.inputs)}
    m[PREFIX + "cot"], m[PREFIX + "out"] = g, node
    return tuple((child, substitute(t, m)) for child, t in zip(node.inputs, templates))


def jvp_tangent(reg: Registry, node: Primitive, supplied) -> Node:
    """The output's tangent for one boundary node; an input with no incoming tangent is a structural zero."""
    (template,) = _supplied(reg, node, "jvp")
    m = {PREFIX + f"in{i}": x for i, x in enumerate(node.inputs)}
    m.update({PREFIX + f"dot{i}": (t if t is not None else zero_like(x.ttype)) for i, (t, x) in enumerate(zip(supplied, node.inputs))})
    m[PREFIX + "out"] = node
    return substitute(template, m)
