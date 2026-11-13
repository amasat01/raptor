# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Lane-space FACTS of a kernel — an IR analysis.

Answers, per axis of the lane space, how every plane read and every sink KEY
moves with that axis, into a PLAIN record (strings, ints, tuples) that a
consumer reads by duck-typed attribute access — the seam style of
:mod:`hawk.ir.contraction`. It only reads: it builds no IR and changes no
emitted text.

The AXES come from the kernel's own ``split_index(minor)`` uses (one
``lane / k`` divisor each), in numpy order: an undeclared kernel has ONE axis;
one divisor ``k`` gives a runtime first axis and a static last axis of extent
``k``; nested divisors that divide one another give a chain. The flat lane is
``sum(axis_i * stride_i)`` (strides are the divisors; the last is 1), the
fastest axis last. A divisor set that is not such a chain falls back to the
single flat axis (conservative). ``labels`` name the axes; ``group`` is the
kernel's declared group, a lane count.

PER-AXIS CLASS of an index expression, from an affine algebra over the axes:

* ``invariant``      the index does not move along the axis;
* ``affine``         moves as ``coefficient * axis + rest`` (coefficient = stride);
* ``quotient``       moves through ``floor(coefficient * axis + ...)``, a
                     fractional coefficient (lanes share a value in runs);
* ``data_dependent`` moves because it carries a value READ from a plane (gather);
* ``non_affine``     any other dependence (a product of lane terms, a real
                     division, a ``floor``, a loop carry, ...).

Unknown means NOT invariant. ``gather`` on a site is True when the index
carries a plane-read value that varies with a lane axis; its per-axis ``base`` forms are then those of
the data-free part (a gather into a row the lane chooses by its coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from raptor.ir.nodes import Loop, LoopCarry, LoopIndex
from raptor.ir.nodes import KEYED_AUTO, At, Const, Leaf, Node, Op, SampleIndex, Sink, peel_clamp

INVARIANT, AFFINE, QUOTIENT = "invariant", "affine", "quotient"
DATA_DEPENDENT, NON_AFFINE = "data_dependent", "non_affine"

#: Weakest-to-strongest dependence, for joining sites of one plane.
_ORDER = (INVARIANT, AFFINE, QUOTIENT, NON_AFFINE, DATA_DEPENDENT)


@dataclass(frozen=True)
class LaneAxis:
    """One axis of the lane space: ``label`` is ``None`` when undeclared,
    ``extent`` is ``None`` for the runtime first axis, ``stride`` is the
    flat-lane distance of one step (the ``lane_<label>_stride``)."""

    label: str | None
    extent: int | None
    stride: int


@dataclass(frozen=True)
class AxisForm:
    """How one index moves along one axis. ``coefficient`` is set for
    ``affine`` / ``quotient`` (an exact ``Fraction``)."""

    cls: str
    coefficient: Fraction | None = None


@dataclass(frozen=True)
class ReadFact:
    """One ``At`` read site. ``forms``/``base`` are per-axis (axis order).
    ``base`` is the data-free part of a gather. ``offset`` is the index's
    constant term (``None`` unknown); ``lane_offset`` is that constant when
    the index is exactly ``lane + offset`` (a neighbour/stencil read)."""

    plane: str
    role: str
    forms: tuple
    base: tuple
    gather: bool
    offset: Fraction | None
    lane_offset: int | None
    #: ``(row, offset, slot)`` when the index is ``row + offset [+ clamp]`` with the
    #: data-free ``row`` invariant along the fastest axis (see :func:`_tile_site`):
    #: ``row`` names its non-constant terms, ``slot`` is the clamp's static bound or ``None``.
    tile_site: tuple | None = None


@dataclass(frozen=True)
class SinkFact:
    """One sink. ``keyed`` is False for an own-column write (it moves with
    every axis); a keyed sink's ``forms`` are its KEY's."""

    name: str
    kind: str
    keyed: bool
    forms: tuple
    gather: bool
    #: keyed sinks only (``None`` for an own-column sink); see :func:`key_facts`.
    #: ``key_origin``: invariant_axis | index_plane_sorted | index_plane_unsorted | data_dependent.
    key_origin: str | None = None
    #: static EXCLUSIVE upper bound of the key, or ``None`` (unknown = no).
    key_extent: int | None = None
    #: dense-slot form ``key = base + slot``: the slot's static bound G, else ``None``.
    slot_extent: int | None = None
    #: per-axis forms of ``base`` whenever the key has the ``base + clamp`` shape.
    slot_base_forms: tuple = ()
    #: the sum contract: the sink's own contract when it names one, else
    #: ``"compensated"`` when the sink carries a companion plane, else ``"auto"``.
    contract: str = "auto"

    def invariant_axes(self, facts: "LaneFacts") -> tuple:
        return tuple(a.label for a, f in zip(facts.axes, self.forms) if f.cls == INVARIANT)


@dataclass(frozen=True)
class LaneFacts:
    axes: tuple
    reads: tuple
    sinks: tuple
    notes: tuple = ()
    #: the kernel's declared group (a lane count, default 1).
    group: int = 1

    def axis_index(self, axis: Any) -> int:
        if isinstance(axis, int):
            return axis
        labels = [a.label for a in self.axes]
        if axis is None or axis not in labels:
            raise ValueError(f"no axis labelled {axis!r} in {labels}")
        return labels.index(axis)

    def read_classes(self, plane: str, axis: Any) -> tuple:
        """The set of per-axis classes over the read sites of ``plane``."""
        i = self.axis_index(axis)
        return tuple(sorted({r.forms[i].cls for r in self.reads if r.plane == plane},
                            key=_ORDER.index))

    def invariant_reads(self, axis: Any) -> tuple:
        """Planes ALL of whose read sites are invariant along ``axis``."""
        i = self.axis_index(axis)
        names = sorted({r.plane for r in self.reads})
        return tuple(n for n in names
                     if all(r.forms[i].cls == INVARIANT for r in self.reads if r.plane == n))


# --------------------------------------------------------------------------- #
# The value algebra: const + sum(coef_a * axis_a), plus "varies in some other way".
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _V:
    lin: tuple = ()                 # sorted ((axis, Fraction), ...), no zeros
    const: Fraction | None = Fraction(0)
    bad: frozenset = frozenset()    # axes it varies along non-affinely
    data: bool = False              # carries a plane-read value
    dbad: frozenset = frozenset()   # axes along which a carried read value varies

    def coefs(self) -> dict:
        return dict(self.lin)

    def varies(self) -> frozenset:
        return frozenset(a for a, _ in self.lin) | self.bad | self.dbad


def _mk(coefs: dict, const, bad=frozenset(), data=False, dbad=frozenset()) -> _V:
    return _V(tuple(sorted((a, c) for a, c in coefs.items() if c != 0)), const,
              frozenset(bad), data, frozenset(dbad))


def nodes_of(obj: Any) -> list:
    """Every node reachable from ``obj``'s sinks (loop bodies included)."""
    candidate = getattr(obj, "sinks", None)
    if candidate is None:
        order = getattr(obj, "order", None)
        candidate = order if order is not None else obj
    roots = list(candidate)
    seen: set = set()
    out: list = []
    stack = roots[::-1]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append(node)
        stack.extend(node.operands)
        if isinstance(node, Loop):
            stack.extend(node.body_sinks)
            stack.extend(node.body_nodes())
    return out


def divisors(nodes: list) -> list:
    """Distinct ``(k, label|None)`` of every ``lane / k`` in the nodes."""
    found: dict = {}
    for n in nodes:
        if isinstance(n, Op) and n.kind == "div" and len(n.operands) == 2 \
                and isinstance(n.operands[0], SampleIndex) \
                and isinstance(n.operands[1], Const) \
                and n.operands[1].ttype.dtype in ("i32", "i64"):
            k = int(n.operands[1].literal)
            if k > 0:
                found.setdefault(k, getattr(n.operands[1], "label", None))
    return sorted(found.items(), reverse=True)


def axes_from(divs: list) -> tuple:
    """(axes, notes) for the divisor set; a non-chain falls back to one axis."""
    ks = [k for k, _ in divs]
    chain = all(a % b == 0 for a, b in zip(ks, ks[1:])) if ks else True
    if not ks:
        return (LaneAxis(None, None, 1),), ()
    notes = ()
    if not chain:
        notes = (f"divisors {tuple(ks)} are not a chain: one flat axis (conservative)",)
        return (LaneAxis(None, None, 1),), notes
    strides = ks + [1]
    def _name(i: int) -> str | None:
        tag = divs[i][1] if i < len(divs) else None      # "lane_<label>_stride" of a declared axis
        if tag and tag.startswith("lane_") and tag.endswith("_stride"):
            return tag[len("lane_"):-len("_stride")]
        return None

    axes = [LaneAxis(_name(i), None if i == 0 else strides[i - 1] // strides[i], strides[i])
            for i in range(len(strides))]
    return tuple(axes), notes


class Eval:
    def __init__(self, axes: tuple) -> None:
        self.axes = axes
        self.n = len(axes)
        self.memo: dict = {}
        self.all = frozenset(range(self.n))

    def lane(self) -> _V:
        return _mk({i: Fraction(a.stride) for i, a in enumerate(self.axes)}, Fraction(0))

    def value(self, root: Node) -> _V:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if id(node) in self.memo:
                continue
            if not expanded:
                stack.append((node, True))
                for child in self._children(node):
                    if id(child) not in self.memo:
                        stack.append((child, False))
                continue
            self.memo[id(node)] = self._eval(node)
        return self.memo[id(root)]

    @staticmethod
    def _children(node: Node):
        if isinstance(node, At):
            return (node.raw_index,)    # the plane leaf is not a lane quantity
        if isinstance(node, (LoopIndex, LoopCarry, Leaf)):
            return ()
        return node.operands

    def _eval(self, node: Node) -> _V:
        if isinstance(node, SampleIndex):
            return self.lane()
        if isinstance(node, Const):
            if node.ttype.dtype in ("i32", "i64"):
                return _mk({}, Fraction(int(node.literal)))
            return _mk({}, None)
        if isinstance(node, LoopIndex):
            return _mk({}, None)                       # same for every lane
        if isinstance(node, LoopCarry):
            return _mk({}, None, self.all)             # conservative
        if isinstance(node, Leaf):
            if node.role in ("uniform", "nsamples"):
                return _mk({}, None)                   # the same for every lane
            return _mk({}, None, frozenset(), True, self.all)   # a per-lane input value
        if isinstance(node, At):
            idx = self.memo[id(node.raw_index)]
            return _mk({}, None, frozenset(), True, idx.varies())   # a read value
        if isinstance(node, Op):
            if node.kind == "clamp_index":           # a table read's clamp: the index's own form, held in the plane
                return self.memo[id(node.operands[0])]
            ops = [self.memo[id(o)] for o in node.operands]
            if node.kind in ("add", "sub") and len(ops) == 2:
                return self._addsub(node.kind, *ops)
            if node.kind == "mul" and len(ops) == 2:
                return self._mul(*ops)
            if node.kind == "div" and len(ops) == 2 and node.ttype.dtype in ("i32", "i64"):
                return self._div(*ops)
        # any other node: invariant iff every operand is; else non-affine
        ops = [self.memo[id(o)] for o in node.operands if id(o) in self.memo]
        lin = frozenset().union(*(frozenset(a for a, _ in o.lin) | o.bad for o in ops)) if ops else frozenset()
        return _mk({}, None, lin, any(o.data for o in ops),
                   frozenset().union(*(o.dbad for o in ops)) if ops else frozenset())

    @staticmethod
    def _lin(v: _V) -> frozenset:
        """The axes ``v`` moves along WITHOUT counting its data axes."""
        return frozenset(a for a, _ in v.lin) | v.bad

    def _addsub(self, kind: str, a: _V, b: _V) -> _V:
        sign = 1 if kind == "add" else -1
        coefs = a.coefs()
        for ax, c in b.lin:
            coefs[ax] = coefs.get(ax, Fraction(0)) + sign * c
        const = None if a.const is None or b.const is None else a.const + sign * b.const
        # a quotient survives only a shift by an axis-free term (index_form's rule)
        if any(c.denominator != 1 for c in coefs.values()) and (len(a.lin) and len(b.lin)):
            return _mk({}, None, self._lin(a) | self._lin(b), a.data or b.data, a.dbad | b.dbad)
        return _mk(coefs, const, a.bad | b.bad, a.data or b.data, a.dbad | b.dbad)

    def _mul(self, a: _V, b: _V) -> _V:
        for x, y in ((a, b), (b, a)):
            if not x.varies() and x.const is not None:    # x is a compile-time number
                k = x.const
                if k == 0:
                    return _mk({}, Fraction(0))
                if any(c.denominator != 1 for _, c in y.lin) and k.denominator != 1:
                    break
                return _mk({ax: c * k for ax, c in y.lin},
                           None if y.const is None else y.const * k, y.bad, y.data or x.data, y.dbad)
        varies = a.varies() | b.varies()
        if not varies:
            const = a.const * b.const if a.const is not None and b.const is not None else None
            return _mk({}, const, frozenset(), a.data or b.data, a.dbad | b.dbad)
        return _mk({}, None, self._lin(a) | self._lin(b), a.data or b.data, a.dbad | b.dbad)

    def _div(self, a: _V, b: _V) -> _V:
        if b.varies() or b.const is None or b.const <= 0:
            return _mk({}, None, self._lin(a) | self._lin(b), a.data or b.data, a.dbad | b.dbad)
        k = b.const
        if not a.varies():
            const = None if a.const is None or a.const < 0 else Fraction(int(a.const) // int(k))
            return _mk({}, const, frozenset(), a.data, a.dbad)
        if a.bad or a.dbad or a.const is None or a.const < 0:
            return _mk({}, None, self._lin(a), a.data, a.dbad)
        # exact part: axes whose coefficient k divides; the rest must stay below k
        whole, rest_max = {}, Fraction(0)
        for ax, c in a.lin:
            if c.denominator == 1 and c % k == 0:
                whole[ax] = c / k
                continue
            ext = self.axes[ax].extent
            if ext is None or c < 0 or c.denominator != 1:
                whole = None
                break
            rest_max += c * (ext - 1)
        if whole is not None and a.const + rest_max < k:
            return _mk(whole, Fraction(0) if a.const < k else None, frozenset(), a.data, a.dbad)
        if len(a.lin) == 1:                              # a lone axis: quotient form
            (ax, c), = a.lin
            return _mk({ax: c / k}, Fraction(int(a.const) // int(k)), frozenset(), a.data, a.dbad)
        return _mk({}, None, self._lin(a), a.data, a.dbad)


ORIGIN_AXIS, ORIGIN_SORTED = "invariant_axis", "index_plane_sorted"
ORIGIN_UNSORTED, ORIGIN_DATA = "index_plane_unsorted", "data_dependent"


def _clamp_bound(node: Node) -> int | None:
    """G when ``node`` is ``hawk.math.index(x, G)``: ``cast_i32(min(max(_, 0), G-1))``."""
    if not (isinstance(node, Op) and node.kind == "cast" and node.literal == "i32"):
        return None
    mn = node.operands[0]
    if not (isinstance(mn, Op) and mn.kind == "min" and isinstance(mn.operands[1], Const)):
        return None
    mx = mn.operands[0]
    if not (isinstance(mx, Op) and mx.kind == "max" and isinstance(mx.operands[1], Const)
            and mx.operands[1].literal == 0.0):
        return None
    top = mn.operands[1].literal
    return int(top) + 1 if float(top) == int(top) and top >= 0 else None


def _base_extent(base: _V, axes: tuple) -> int | None:
    """Static EXCLUSIVE bound of a data-free, non-negative affine ``base``
    (every axis it moves along static), else ``None``."""
    if base.bad or base.const is None or base.const < 0 or base.const.denominator != 1:
        return None
    hi = base.const
    for ax, c in base.lin:
        ext = axes[ax].extent
        if ext is None or c < 0 or c.denominator != 1:
            return None
        hi += c * (ext - 1)
    return int(hi) + 1


def _stored(node: Node) -> Node:
    """The plane read under the widening cast of an ``Index`` read (``int32`` on the wire)."""
    if isinstance(node, Op) and node.kind == "cast" and node.ttype.dtype == "i32" \
            and node.operands[0].ttype.dtype == "int32":
        return node.operands[0]
    return node


def _index_plane_read(node: Node, ev: "Eval") -> bool:
    """Whether ``node`` reads an ``Index`` plane: the own-lane leaf, or a ``Table[Index]`` read at a
    data-free index (either through the widening cast of its ``int32`` storage)."""
    stored = _stored(node)
    dtypes = ("i32", "i64") if stored is node else ("int32",)
    if isinstance(stored, Leaf):
        return stored.role == "per_sample" and stored.ttype.dtype in dtypes
    return (isinstance(stored, At) and stored.plane.ttype.dtype in dtypes
            and stored.plane.role == "lookup" and not ev.value(stored.raw_index).data)


def _split_terms(ev: "Eval", key: Node) -> tuple:
    """The additive terms of an index: ``(free, slots, planes, other)``. ``free``
    holds ``(node, value, sign)`` of data-free terms, ``slots`` ``(bound, sign)`` of
    ``hawk.math.index`` clamps, ``planes`` ``(node, sign)`` of Index-plane reads,
    ``other`` counts the rest."""
    free, slots, planes, other = [], [], [], 0
    stack = [(key, 1)]
    while stack:
        node, sign = stack.pop()
        node = peel_clamp(node)
        if isinstance(node, Op) and node.kind == "add" and len(node.operands) == 2:
            stack += [(node.operands[1], sign), (node.operands[0], sign)]
            continue
        if isinstance(node, Op) and node.kind == "sub" and len(node.operands) == 2 \
                and not ev.value(node.operands[1]).data:
            stack += [(node.operands[1], -sign), (node.operands[0], sign)]
            continue
        v = ev.value(node)
        g = _clamp_bound(node)
        if not v.data:
            free.append((node, v, sign))
        elif g is not None:
            slots.append((g, sign))
        elif _index_plane_read(node, ev):
            planes.append((_stored(node), sign))
        else:
            other += 1
    return free, slots, planes, other


def _tile_site(ev: "Eval", index: Node, axes: tuple) -> tuple | None:
    """``(row, offset, slot)`` of a read index shaped ``row + offset`` or
    ``row + offset + clamp``, with ``row`` (its non-constant data-free terms, by
    node identity) invariant along the fastest axis; else ``None``."""
    free, slots, planes, other = _split_terms(ev, index)
    if planes or other or len(slots) > 1 or (slots and slots[0][1] < 0):
        return None
    row, offset, base = [], 0, _mk({}, Fraction(0))
    for node, v, sign in free:
        base = ev._addsub("add" if sign > 0 else "sub", base, v)
        if v.varies() or v.const is None:
            row.append((id(node), sign))
        elif v.const.denominator == 1:
            offset += sign * int(v.const)
        else:
            return None
    if base.bad or axis_forms(base, len(axes))[-1].cls != INVARIANT:
        return None
    return tuple(sorted(row)), offset, slots[0][0] if slots else None


def key_facts(ev: "Eval", key: Node, axes: tuple) -> tuple:
    """``(key_origin, key_extent, slot_extent, slot_base_forms)`` of one sink key.

    The key is split into its additive terms; each is data-free (folded into
    ``base``), a ``hawk.math.index`` clamp (a SLOT of static bound G), a read of
    an Index plane (``Leaf`` per-sample at the lane's own index, or a ``Table[Index]``
    read at a data-free index), or other plane data. Unknown = no."""
    free, slots, planes, other = _split_terms(ev, peel_clamp(key))   # a table read's clamp: its shape is the key's
    base = _mk({}, Fraction(0))
    for _, v, sign in free:
        base = ev._addsub("add" if sign > 0 else "sub", base, v)
    base_forms = axis_forms(base, len(axes))
    if not slots and not planes and not other:
        if base.bad:
            return ORIGIN_DATA, None, None, ()
        return ORIGIN_AXIS, _base_extent(base, axes), None, ()
    if planes and not slots and not other and len(planes) == 1 and not base.bad:
        leaf, sign = planes[0]
        invariant = all(f.cls == INVARIANT for f in base_forms)
        if isinstance(leaf, Leaf) and leaf.sorted and sign > 0 and invariant:
            return ORIGIN_SORTED, None, None, ()
        return ORIGIN_UNSORTED, None, None, ()
    if len(slots) == 1 and not planes and not other and slots[0][1] > 0 and not base.bad:
        g = slots[0][0]
        be = _base_extent(base, axes)
        slot_ext = g if base_forms[-1].cls == INVARIANT else None
        return ORIGIN_DATA, (None if be is None else be - 1 + g), slot_ext, base_forms
    return ORIGIN_DATA, None, None, ()


def axis_forms(v: _V, n: int) -> tuple:
    out = []
    coefs = v.coefs()
    for i in range(n):
        if i in v.dbad:
            out.append(AxisForm(DATA_DEPENDENT))
        elif i in v.bad:
            out.append(AxisForm(NON_AFFINE))
        elif i in coefs:
            c = coefs[i]
            out.append(AxisForm(AFFINE if c.denominator == 1 else QUOTIENT, c))
        else:
            out.append(AxisForm(INVARIANT))
    return tuple(out)


def _compensated_names(obj: Any) -> tuple:
    """``(on, companion)``: whether the kernel's sink kind is compensated and
    the name of the companion sink it writes the correction to."""
    sink = getattr(getattr(obj, "kind", None), "sink", None)
    if getattr(sink, "id", None) != "compensated":
        return False, None
    return True, getattr(sink, "compensation", None)


def facts(obj: Any, labels: tuple | None = None, group: int = 1) -> LaneFacts:
    """The lane-space facts of a traced kernel (anything with ``sinks``), a
    :class:`~hawk.ir.walk.Walk`, or a sink tuple. ``labels`` are the declared
    axis labels in numpy order (``None`` entries stay unlabeled); ``group`` is
    the kernel's declared group, a lane count."""
    nodes = nodes_of(obj)
    axes, notes = axes_from(divisors(nodes))
    if labels is not None:
        if len(labels) != len(axes):
            raise ValueError(f"labels {labels!r} do not match {len(axes)} lane axes")
        axes = tuple(LaneAxis(lb, a.extent, a.stride) for lb, a in zip(labels, axes))
    if not isinstance(group, int) or isinstance(group, bool) or group < 1:
        raise ValueError(f"group must be a lane count >= 1, got {group!r}")
    ev = Eval(axes)
    n = len(axes)
    comp_on, companion = _compensated_names(obj)
    reads, sinks = [], []

    def contract(s: Any) -> str:
        declared = getattr(s, "contract", "auto")
        if declared == KEYED_AUTO:
            return "compensated"
        if declared not in ("auto", "plain"):
            return declared
        carries = (comp_on and s.name != companion) or (
            s.kind == "sample_sum" and getattr(s, "op", None) == "sum")
        return "compensated" if carries else "auto"

    def add_sink(s: Any) -> None:
        if s.index is None:
            sinks.append(SinkFact(s.name, s.kind, False,
                                  tuple(AxisForm(AFFINE, Fraction(a.stride)) for a in axes), False,
                                  contract=contract(s)))
            return
        v = ev.value(s.index)
        sinks.append(SinkFact(s.name, s.kind, True, axis_forms(v, n), bool(v.dbad),
                              *key_facts(ev, s.index, axes), contract=contract(s)))

    for node in nodes:
        if isinstance(node, At):
            v = ev.value(node.raw_index)
            plane = node.plane
            base = axis_forms(_mk(v.coefs(), v.const, v.bad), n)       # the data-free part
            lane_off = None
            if not v.dbad and not v.bad and v.const is not None and v.lin == ev.lane().lin:
                lane_off = int(v.const) if v.const.denominator == 1 else None
            reads.append(ReadFact(plane.name, plane.role, axis_forms(v, n), base, bool(v.dbad),
                                  v.const, lane_off, _tile_site(ev, node.raw_index, axes)))
        elif isinstance(node, Sink):
            add_sink(node)
        elif isinstance(node, Loop):
            for s in node.body_sinks:
                if isinstance(s, Sink):
                    add_sink(s)
    # deterministic order
    reads.sort(key=lambda r: (r.plane, repr(r.forms), repr(r.offset)))
    return LaneFacts(axes, tuple(reads), tuple(sinks), notes, group)


def varies_along(obj: Any, axis: int):
    """A predicate over ``obj``'s nodes (loop bodies included): whether a
    node's value may differ between lanes that differ only in lane axis
    ``axis`` (by position). Unknown = it may."""
    ev = Eval(axes_from(divisors(nodes_of(obj)))[0])
    return lambda node: axis in ev.value(node).varies()
