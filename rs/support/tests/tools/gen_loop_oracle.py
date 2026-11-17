"""Records the answers of the old Python loop rules over every loop of the recorded kernel graphs.

Usage: PYTHONPATH=<overlay holding raptor/_core> python gen_loop_oracle.py <record dir> <out.jsonl>
Each line is one loop (one per distinct body key, then two hand-made nested loops that carry their own node table): where it sits in the record (file, line, node index), the old
`classify_loop` answer (the class and its addends, or the refusal text), the old `refuse_unsupported_nesting` answer
and the old `addends` of every carry's next value. An addend that is a node of the record is written as {"n": index};
one the rule made is written as {"op": "add", "operands": [...]}.
"""

import importlib
import json
import sys
from pathlib import Path

from raptor.ir import loops as old
from raptor.ir.nodes import Const, LoopCarry, LoopIndex, Node
from raptor.ir.ops import make as old_make
from raptor.ir.types import TensorType


def decode(value, table, built):
    """The old Python value of a recorded value; nodes are built once, on first reference."""
    if not isinstance(value, dict):
        return value
    if "s" in value:
        return value["s"]
    if "i" in value:
        return int(value["i"])
    if "f" in value:
        return float.fromhex(value["x"]) if "x" in value else float(value["f"])
    if "t" in value:
        return tuple(decode(v, table, built) for v in value["t"])
    if "n" in value:
        return node(value["n"], table, built)
    if "r" in value:
        cls = getattr(importlib.import_module(value["m"]), value["r"])
        return cls(**{k: decode(v, table, built) for k, v in value["v"].items()})
    raise ValueError(f"unknown recorded value {value!r}")


def node(index, table, built):
    """The old node recorded at `index`."""
    if index not in built:
        entry = table[index]
        cls = getattr(importlib.import_module(entry["m"]), entry["c"])
        made = object.__new__(cls)
        built[index] = made
        for name, value in entry["v"].items():
            object.__setattr__(made, name, decode(value, table, built))
    return built[index]


def outcome(call):
    """What the old function answers: ('ok', value) or ('refused', its text)."""
    try:
        return ("ok", call())
    except Exception as error:  # the refusal text is the oracle
        return ("refused", str(error))


def serial(n, position):
    """An addend as the fixture writes it."""
    if id(n) in position:
        return {"n": position[id(n)]}
    return {"op": "add", "operands": [serial(o, position) for o in n.operands]}


def answers(loop, position):
    """What the old rules answer for `loop`: its terms, its nesting refusal and its class."""
    cls = outcome(lambda: old.classify_loop(loop))
    nest = outcome(lambda: old.refuse_unsupported_nesting(loop))
    row = {"terms": [[serial(t, position) for t in old.addends(n)] for n in loop.nexts]}
    row["nesting"] = nest[1] if nest[0] == "refused" else None
    if cls[0] == "refused":
        row["class"] = {"refused": cls[1]}
    else:
        row["class"] = {
            "name": cls[1].name,
            "addends": [serial(a, position) for a in cls[1].addends],
        }
    return row


def encode(value, order):
    """The recorded form of an old value; nodes are numbered through `order`."""
    if isinstance(value, Node):
        return {"n": order[id(value)]}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return {"s": value}
    if isinstance(value, int):
        return {"i": str(value)}
    if isinstance(value, float):
        return {"f": repr(value), "x": value.hex()}
    if isinstance(value, (tuple, list)):
        return {"t": [encode(v, order) for v in value]}
    cls = type(value)
    if not hasattr(value, "__dict__"):
        raise TypeError(f"no recorded form for {cls.__name__}: {value!r}")
    fields = {k: encode(v, order) for k, v in vars(value).items()}
    return {"m": cls.__module__, "r": cls.__name__, "v": fields}


def slots_of(n):
    """The slot names an old node holds, base class first."""
    names = []
    for cls in reversed(type(n).__mro__):
        names += [s for s in getattr(cls, "__slots__", ()) if s not in names]
    return names


def record_of(root):
    """The recorded node table of everything `root` reaches, children first, and the index of `root`."""
    order, table = {}, []

    def visit(n):
        if id(n) in order:
            return
        for name in slots_of(n):
            held = getattr(n, name, None)
            for item in held if isinstance(held, (tuple, list)) else (held,):
                if isinstance(item, Node):
                    visit(item)
        order[id(n)] = len(order)
        table.append(n)

    visit(root)
    entries = [
        {
            "c": type(n).__name__,
            "m": type(n).__module__,
            "v": {name: encode(getattr(n, name, None), order) for name in slots_of(n)},
        }
        for n in table
    ]
    return entries, order[id(root)], order


def nested_loops():
    """Hand-made graphs the recordings do not hold: an outer loop whose body holds a loop that breaks, and one
    that runs a recorded trip count; the old rules refuse both by name."""
    real, index = TensorType((), "f64"), TensorType((), "i32")
    one, three = Const(1.0, real), Const(3, index)
    loops = []
    for kind in ("break", "count"):
        outer_index, outer_carry = LoopIndex(0, "i", index), LoopCarry(0, 0, "t", real)
        inner_index, inner_carry = LoopIndex(1, "j", index), LoopCarry(1, 0, "s", real)
        step = old_make("add", (inner_carry, one))
        extra = {}
        if kind == "break":
            extra = {"exit_cond": old_make("gt", (inner_carry, Const(5.0, real))), "exit_values": (inner_carry,)}
        else:
            extra = {"count": three}
        inner = old.build_loop(inner_index, [inner_carry], [outer_carry], [step], start=0, stop=3, step=1, **extra)
        grown = old_make("add", (outer_carry, inner))
        loops.append(old.build_loop(outer_index, [outer_carry], [Const(0.0, real)], [grown], start=0, stop=4, step=1))
    return loops


def main():
    """Replays every recorded loop through the old rules and writes the fixture; a loop that reaches a dialect kind (declared by a manifest of the record) is skipped and counted."""
    seen, rows, skipped = set(), [], 0
    for path in sorted(Path(sys.argv[1]).glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines()):
            table = json.loads(line).get("nodes") or []
            for index, entry in enumerate(table):
                if entry["c"] != "Loop":
                    continue
                key = entry["v"]["_body_key"]["s"]
                if key in seen:
                    continue
                seen.add(key)
                built = {}
                try:
                    loop = node(index, table, built)
                except AttributeError:  # a loop that reaches a dialect kind the old package does not declare
                    skipped += 1
                    continue
                position = {id(n): i for i, n in built.items()}
                row = {"file": path.name, "line": number, "node": index}
                row.update(answers(loop, position))
                rows.append(row)
    for loop in nested_loops():
        table, index, order = record_of(loop)
        row = {"file": None, "nodes": table, "node": index}
        row.update(answers(loop, order))
        rows.append(row)
    Path(sys.argv[2]).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "loops recorded,", skipped, "skipped (a dialect kind of the record)")


if __name__ == "__main__":
    main()
