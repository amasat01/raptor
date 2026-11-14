# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The op vocabulary (raptor.ir.ops, raptor.ir.types, raptor.ir.rays): each kind set against its documented members, and against the
type rule that reads it."""
import pytest

from raptor.errors import RaptorError
from raptor.ir import ops, rays, types
from raptor.ir.nodes import Const, Leaf, ROLE_ORDER
from raptor.ir.types import TensorType

F64, I32, B = TensorType((), "f64"), TensorType((), "int32"), TensorType((), "bool")
V3 = TensorType((3,), "f64")


def _leaf(t, name="x"):
    return Leaf("vocab_read", "per_sample", name, t)


def test_the_sets_hold_their_documented_members():
    assert ops.ELEMENTWISE_BINARY[:4] == ("add", "sub", "mul", "div") and "atan2" in ops.ELEMENTWISE_BINARY
    assert ops.ELEMENTWISE_TERNARY == ("fma", "clip")
    assert ops.PREDICATE == ("isnan", "isinf", "isfinite")
    assert ops.COMPARE == ("lt", "le", "gt", "ge", "eq", "ne")
    assert ops.LOGICAL == ("land", "lor", "lnot")
    assert ops.INT_UNARY == ("invert",)
    assert ops.INT_BINARY == ("floordiv", "mod", "band", "bor", "bxor", "shl", "shr", "clamp_index")
    assert ops.RANDOM_OPS == ("random_uniform", "random_normal")
    assert ops.CAST == ("cast",)
    assert {"sin", "sqrt", "neg", "erf"} <= set(ops.UNARY_MATH)


def test_every_set_is_part_of_the_one_vocabulary_without_repeats():
    parts = (ops.ELEMENTWISE_BINARY, ops.UNARY_MATH, ops.ELEMENTWISE_TERNARY, ops.PREDICATE, ops.COMPARE, ops.LOGICAL,
             ops.INT_BINARY, ops.INT_UNARY, ops.RANDOM_OPS, ops.CAST, ops.MINTED_OPS)
    assert all(set(p) <= set(ops.OP_KINDS) for p in parts)
    assert len(ops.OP_KINDS) == len(set(ops.OP_KINDS)) and {"select", "dispatch", "at", "mv", "lu"} <= set(ops.OP_KINDS)
    assert set(ops.MINTED_OPS) >= {"qr", "splat", "add_at", "clamp_index"}


@pytest.mark.parametrize("kind", ops.ELEMENTWISE_BINARY)
def test_an_elementwise_binary_keeps_the_float_type_and_broadcasts_a_scalar(kind):
    assert ops.make(kind, (_leaf(F64), _leaf(F64, "y"))).ttype == F64
    assert ops.make(kind, (_leaf(V3), _leaf(F64, "y"))).ttype == V3


@pytest.mark.parametrize("kind", ops.UNARY_MATH)
def test_an_elementwise_unary_keeps_the_operand_type(kind):
    assert ops.make(kind, (_leaf(V3),)).ttype == V3


@pytest.mark.parametrize("kind", ops.ELEMENTWISE_TERNARY)
def test_a_ternary_takes_three_operands_of_one_shape(kind):
    assert ops.make(kind, (_leaf(V3), _leaf(V3, "y"), _leaf(V3, "z"))).ttype == V3
    with pytest.raises(RaptorError):
        ops.make(kind, (_leaf(V3), _leaf(V3, "y")))


@pytest.mark.parametrize("kind", ops.COMPARE)
def test_a_comparison_is_bool_typed(kind):
    assert ops.make(kind, (_leaf(F64), _leaf(F64, "y"))).ttype == TensorType((), "bool")


@pytest.mark.parametrize("kind", ops.PREDICATE)
def test_a_class_test_is_bool_typed(kind):
    assert ops.make(kind, (_leaf(V3),)).ttype == TensorType((3,), "bool")


def test_a_connective_is_bool_typed():
    flag = _leaf(TensorType((), "bool"), "f")
    assert ops.make("land", (flag, flag)).ttype == ops.make("lnot", (flag,)).ttype == TensorType((), "bool")


@pytest.mark.parametrize("kind", [k for k in ops.INT_BINARY if k != "clamp_index"])
def test_an_integer_binary_stays_integer_and_refuses_a_float(kind):
    assert ops.make(kind, (_leaf(I32), _leaf(I32, "y"))).ttype == I32
    with pytest.raises(RaptorError):
        ops.make(kind, (_leaf(F64), _leaf(I32, "y")))


def test_an_integer_unary_stays_integer_and_a_cast_reaches_a_declared_dtype():
    assert ops.make("invert", (_leaf(I32),)).ttype == I32
    assert ops.make("cast", (_leaf(I32),), "f64").ttype == F64
    assert ops.make("cast", (_leaf(F64),), "int32").ttype == I32
    with pytest.raises(RaptorError):
        ops.make("cast", (_leaf(F64),), "no_such_dtype")


@pytest.mark.parametrize("kind", ops.RANDOM_OPS)
def test_a_draw_takes_two_integer_scalars_and_gives_a_float(kind):
    seed, counter = _leaf(TensorType((), "i32"), "seed"), _leaf(TensorType((), "i32"), "counter")
    assert ops.make(kind, (seed, counter)).ttype == F64
    with pytest.raises(RaptorError, match=f"{kind}: seed"):
        ops.make(kind, (_leaf(F64), counter))


def test_the_integer_plane_dtypes_never_mix_with_a_float_without_a_cast():
    assert types.INT_PLANE_DTYPES == ("int32", "uint32")
    for dtype in types.INT_PLANE_DTYPES:
        assert ops.promote(dtype, dtype) == dtype
        with pytest.raises(RaptorError, match="cast"):
            ops.promote(dtype, "f64")


def test_the_per_sample_roles_are_leaf_roles():
    assert types.PER_SAMPLE_ROLES == ("per_sample", "vec_in", "mat_in", "terminated", "mutable", "out")
    assert set(types.PER_SAMPLE_ROLES) - {"out"} <= set(ROLE_ORDER)


@pytest.mark.parametrize("kind,count", [("ray_tri", 70.0), ("ray_aabb", 60.0), ("ray_sphere", 45.0)])
def test_a_ray_node_runs_its_documented_operation_count(kind, count):
    v, s = _leaf(V3), _leaf(F64, "t")
    operands = {"ray_tri": (v, v, v, v, v, s), "ray_aabb": (v, v, v, v, s), "ray_sphere": (v, v, v, s, s)}[kind]
    node = ops.make(kind, operands)
    assert node.ttype == TensorType((4,), "f64") and kind in rays.RAY_OPS
    assert rays.op_count(node) == count
