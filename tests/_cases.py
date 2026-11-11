"""Replays the recorded cases of tests/data/{kinds,primitive,block}_cases.json: the specs are JSON, the outcome is a verdict string."""
from raptor.ir import kinds as K, make
from raptor.ir.loops import build_loop
from raptor.ir.nodes import Const, Leaf, LoopCarry, LoopIndex, Primitive, TapeRead
from raptor.ir.types import TensorType
from raptor.rules import primitive as P

BASE = {"name": "acme.thing", "ext": "acme", "dialect": "d", "family": "expr", "operands": {"min": 1, "max": 2, "families": ["any", "plane-leaf"]},
        "fields": [{"name": "k", "type": "int"}, {"name": "tag", "type": "str", "default": "x"}], "result": ["same_as", 0], "identity": ["k"],
        "traits": {"fresh_read": True}, "spelling": {"dialect": "d", "name": "thing", "args": ["node", "ref0"]}, "diff": "none"}
KINDS = [BASE, {"name": "acme.sc", "ext": "acme", "family": "sink", "operands": {"min": 1, "max": 1}, "result": ["carried"]},
         {"name": "acme.s", "ext": "acme", "family": "sink", "refines": "assign", "operands": {"min": 1, "max": 1}, "fields": [{"name": "q", "type": "float", "default": 1}], "result": ["carried"]},
         {"name": "acme.f", "ext": "acme", "family": "expr", "operands": {"min": 0, "max": 2}, "result": ["fixed", "f64", [2]]},
         {"name": "acme.sc2", "ext": "acme", "family": "expr", "operands": {"min": 1, "max": 1}, "result": ["scalar_of", 0]},
         {"name": "acme.c1", "ext": "acme", "family": "expr", "operands": {"min": 0, "max": 1, "families": ["any"]}, "result": ["carried"], "fields": [{"name": "t", "type": "tuple"}]}]


def out(f):
    try:
        return "ok " + str(f())
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def tt(spec):
    return None if spec is None else (spec if isinstance(spec, str) else TensorType(tuple(spec[0]), spec[1]))


def leaf(spec):
    return Leaf("uniform", "uniform", spec[3] if len(spec) > 3 else "x", TensorType(tuple(spec[1]), spec[2]))


def run_declare(data, fp):
    return out(lambda: repr(K.declare(K.Registry(), data, first_party=tuple(fp))))


def registry():
    reg = K.Registry()
    for d in KINDS:
        K.declare(reg, d)
    return reg


def run_make(kind, ttype, ops, fields):
    fields = {k: (tuple(v) if isinstance(v, list) else v["list"] if isinstance(v, dict) else v) for k, v in fields.items()}
    return out(lambda: (lambda n: [type(n).__name__, repr(n.ttype), n.kind, repr(n.state())])(K.make_kind(registry(), kind, tt(ttype), [leaf(o) for o in ops], **fields)))


def block_node(spec):
    if spec[0] == "leaf":
        return leaf(spec)
    return make(spec[0], tuple(block_node(s) for s in spec[1:]))


def run_block(payload, policy):
    from raptor.ir.block import block_shaped, ordered_case
    return out(lambda: [block_shaped(block_node(payload), policy), ordered_case(block_node(payload), policy)])


T = TensorType((), "f64")


def expr(e, loop=None):
    """['ph', tag] | ['leaf', name] | ['const', v] | [op, a, b] | ['loop', body] (body reads ['carry'], ['idx']) | ['tape'] | ['index']"""
    h = e[0]
    if h == "ph":
        return P.placeholder(e[1], T)
    if h == "leaf":
        return Leaf("uniform", "uniform", e[1], T)
    if h == "const":
        return Const(float(e[1]), T)
    if h == "carry":
        return loop[1]
    if h == "idx":
        return make("cast", (loop[0],), "f64")
    if h == "index":
        return make("cast", (LoopIndex(0, "k"),), "f64")
    if h == "tape":
        return TapeRead(expr(["loop", ["add", ["carry"], ["const", 1]]]), 0, Const(0, TensorType((), "i32")))
    if h == "loop":
        idx, c = LoopIndex(0, "k"), LoopCarry(0, 0, "c", T)
        return build_loop(idx, (c,), (Const(0.0, T),), (expr(e[1], (idx, c)),), start=0, stop=3, step=1)
    return make(h, tuple(expr(x, loop) for x in e[1:]))


def ser(roots):
    seen, rows = {}, []

    def go(n):
        if id(n) in seen:
            return seen[id(n)]
        ops = [go(o) for o in n.operands]
        seen[id(n)] = len(rows)
        rows.append([type(n).__name__, n.kind, repr(n.ttype), getattr(n, "name", "") if isinstance(n, Leaf) else "", ops])
        return seen[id(n)]
    return [[go(r) for r in roots], rows]


def run_primitive(spec):
    ins = [tt(t) for t in spec["ins"]]
    out_t = tt(spec["out"])
    node = Primitive("p", Const(1.0, out_t), tuple(Leaf("uniform", "uniform", f"a{i}", t) for i, t in enumerate(ins)), out_t)

    def go():
        reg = K.Registry()
        P.declare_primitive(reg, "p", ext="e", directions=("vjp", "jvp"))
        for _ in range(spec.get("times", 1)):
            tpl = [expr(t) for t in spec["templates"]]
            P.register_rule(reg, "p", P.signature_of(node), spec["dir"], tuple(tpl) if spec["dir"] == "vjp" else tpl[0])
        if spec["dir"] == "vjp":
            return ser([x for _, x in P.vjp_contributions(reg, node, Leaf("uniform", "uniform", "g", out_t))])
        sup = [Leaf("uniform", "uniform", "t0", ins[0]) if s else None for s in spec["sup"]]
        return ser([P.jvp_tangent(reg, node, sup)])
    return out(go)
