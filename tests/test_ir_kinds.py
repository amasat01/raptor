# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Declared kinds (raptor.ir.kinds), the structural zero and the primitive rule table (raptor.rules)."""
import dataclasses

import pytest

from raptor.errors import RaptorError
from raptor.ir import kinds as K, make
from raptor.ir.nodes import Const, Leaf, Primitive
from raptor.ir.types import TensorType
from raptor.rules import primitive as P
from raptor.rules.core import zero_like

F64 = TensorType((), "f64")
ROW = {"name": "acme.thing", "ext": "acme", "dialect": "d", "family": "expr", "operands": {"min": 1, "max": 1}, "result": ["same_as", 0],
       "fields": [{"name": "k", "type": "int"}], "identity": ["k"]}
X = Leaf("vocab_read", "per_sample", "x", F64)


def test_a_declaration_is_a_frozen_record_of_its_row():
    decl = K.declare(K.Registry(), ROW)
    assert isinstance(decl, K.KindDecl) and (decl.name, decl.ext, decl.dialect, decl.family) == ("acme.thing", "acme", "d", "expr")
    assert decl.refines is None and decl.fields[0][:2] == ("k", "int") and decl.identity == ("k",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decl.name = "other"


def test_a_declared_kind_is_named_and_a_core_node_has_no_declared_kind():
    reg = K.Registry()
    K.declare(reg, ROW)
    node = K.make_kind(reg, "acme.thing", F64, (X,), k=3)
    assert K.kind_name(node) == "acme.thing" and K.kind_name(X) is None and K.kind_name(Const(1.0, F64)) is None
    assert node.k == 3 and node.ttype == F64


def test_is_a_holds_for_the_declared_kind_a_refinement_and_a_core_kind():
    reg = K.Registry()
    K.declare(reg, ROW)
    K.declare(reg, {**ROW, "name": "acme.sub", "refines": "acme.thing", "fields": [], "identity": None})
    sub = K.make_kind(reg, "acme.sub", F64, (X,), k=1)
    assert K.is_a(sub, "acme.sub") and K.is_a(sub, "acme.thing") and K.is_a(sub, "node")
    assert not K.is_a(K.make_kind(reg, "acme.thing", F64, (X,), k=1), "acme.sub")
    from raptor.ir.nodes import Assign, At
    assert K.is_a(Assign("y", X, F64), "assign") and K.is_a(Assign("y", X, F64), "sink") and not K.is_a(X, "assign")
    assert K.is_a(X, "node") and not K.is_a(X, "acme.thing") and not K.is_a(Assign("y", X, F64), "at")


def test_load_manifest_declares_the_rows_without_a_class_and_checks_the_ones_with_one():
    reg = K.Registry()
    manifest = {"name": "acme", "dialect": "d", "kinds": [{"name": "acme.a", "family": "expr", "operands": {"min": 1, "max": 1},
                                                           "result": ["same_as", 0]}, {"name": "leaf", "class": "Leaf"}]}
    declared = K.load_manifest(reg, manifest)
    assert [d.name for d in declared] == ["acme.a"] and declared[0].ext == "acme" and set(reg.kinds) == {"acme.a"}
    with pytest.raises(RaptorError, match="not generated"):
        K.load_manifest(K.Registry(), {"name": "acme", "kinds": [{"name": "nope", "class": "NoSuchClass"}]})


def test_a_structural_zero_is_a_rank_zero_literal_or_a_splat_of_one():
    assert zero_like(F64).literal == 0.0 and zero_like(F64).ttype == F64
    z = zero_like(TensorType((2, 3), "f64"))
    assert z.kind == "splat" and z.literal == (2, 3) and z.operands[0].literal == 0.0 and z.ttype == TensorType((2, 3), "f64")
    assert zero_like(TensorType((), "f32")).ttype == TensorType((), "f32")


def _primitive():
    return Primitive("p", Const(1.0, F64), (Leaf("uniform", "uniform", "a0", F64),), F64)


def test_a_primitive_has_a_rule_only_for_a_registered_direction_and_signature():
    reg, node = K.Registry(), _primitive()
    assert not P.has_rule(reg, node, "vjp")
    P.declare_primitive(reg, "p", ext="e", directions=("vjp", "jvp"))
    assert not P.has_rule(reg, node, "vjp")                                  # declared, no instance yet
    P.register_rule(reg, "p", P.signature_of(node), "jvp", P.placeholder("dot0", F64))
    assert P.has_rule(reg, node, "jvp") and not P.has_rule(reg, node, "vjp")


def test_a_template_is_substituted_through_its_placeholders_and_shares_what_reads_none():
    constant = make("neg", (Const(2.0, F64),))
    template = make("add", (make("mul", (P.placeholder("in0", F64), P.placeholder("in0", F64))), constant))
    out = P.substitute(template, {"$in0": Const(3.0, F64)})
    assert out.kind == "add" and [o.literal for o in out.operands[0].operands] == [3.0, 3.0] and out.operands[1] is constant


def test_the_core_kinds_are_the_refinable_rows_of_the_core_table_and_a_row_may_refine_each():
    import json
    from pathlib import Path

    from raptor.ir import nodes
    table = json.loads((Path(K.__file__).parent / "core.json").read_text())
    rows = {r["name"]: r for r in table["kinds"] if r.get("refinable")}
    assert set(K.CORE) == set(rows) and len(rows) == 6
    for name, row in rows.items():
        family, cls, role, wire = K.CORE[name]
        assert (family, cls, role, wire) == (row["family"], getattr(nodes, row["class"]), row.get("role"), row.get("wire"))
    with pytest.raises(TypeError):
        K.CORE["extra"] = None                                              # read-only


def test_a_primitive_declaration_is_the_frozen_record_declare_primitive_returns():
    reg = K.Registry()
    decl = P.declare_primitive(reg, "p", ext="e", directions=["vjp", "jvp"])
    assert isinstance(decl, P.PrimitiveDecl) and (decl.name, decl.ext, decl.directions) == ("p", "e", ("vjp", "jvp"))
    assert reg.primitives["p"][0] == decl and P.declare_primitive(reg, "p", ext="e", directions=("vjp", "jvp")) == decl
    with pytest.raises(dataclasses.FrozenInstanceError):
        decl.ext = "other"


def test_a_derivative_rule_is_the_frozen_record_rule_for_returns():
    from raptor.rules import core
    rule = core.rule_for("add", 0)
    assert isinstance(rule, core.DiffRule) and callable(rule.vjp) and callable(rule.jvp) and rule.mode == "rule"
    assert core.RULES[("add", None)].mode == "rule" and all(isinstance(r, core.DiffRule) for r in core.RULES.values())
    assert core.DiffRule() == core.DiffRule(None, None, "rule")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.mode = "zero"
