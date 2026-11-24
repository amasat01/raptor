"""Records the values of the old Python data rules, for the tests that compare the core with them.

Usage: PYTHONPATH=<overlay holding raptor/_core>:<this folder> python gen_data_oracle.py <out dir> <record dir>
Writes quantity.json: the old `Quantity` construction answer (ok, or its refusal text) for every combination of a few slugs, packs and wire lists, the wire roles including every role the manifest names. Writes structural_keys.json: the old `structural_keys` partition of every node the old package can decode in a sample of the recorded graphs, as dense numbers. Writes index_forms.json: the old `index_form` answer for the index of every table read of the recorded graphs, against the sample index and against each loop index, with and without quotients. Writes feathers.json: the old `has_feather` answer for every feather and every node of two recorded graphs. Writes param_cells.json: the old `param_cell` read for a few types, in the recorded node form. Writes constants.json: the old value of every constant a front end imports (a list becomes a list, a string a string),
and compensates.json: the old `compensates` answer for every contract the manifest names and for a text it does not.
"""

import json
import sys
from pathlib import Path

from gen_loop_oracle import node as old_node
from gen_loop_oracle import record_of
from raptor.ir.nodes import Const, Leaf, LoopIndex, RoleConst
from raptor.ir.ops import make as old_make
from raptor.ir import access, cells, compound, loops, nodes, types, walk

CONSTANTS = {
    "ACCESS_CLASSES": access.ACCESS_CLASSES,
    "CELL_PLANE": cells.CELL_PLANE,
    "COMPANION_SUFFIX": nodes.COMPANION_SUFFIX,
    "ADJOINT_PREFIX": nodes.ADJOINT_PREFIX,
    "GATHER_ROLES": nodes.GATHER_ROLES,
    "PER_SAMPLE_ROLES": types.PER_SAMPLE_ROLES,
    "COMPARE": nodes.COMPARE,
    "LOGICAL": nodes.LOGICAL,
    "PREDICATE": nodes.PREDICATE,
    "CONTRACTS": nodes.CONTRACTS,
    "SCATTER_CONTRACTS": nodes.SCATTER_CONTRACTS,
    "KEYED_AUTO": nodes.KEYED_AUTO,
    "LEAF_ROLES": nodes.LEAF_ROLES,
    "INT_PLANE_DTYPES": types.INT_PLANE_DTYPES,
    "NAMESPACE_SEP": nodes.NAMESPACE_SEP,
    "PARAM_CELL": nodes.PARAM_CELL,
}


def index_text(form):
    """The old IndexForm as the core writes it: a list of texts."""
    if form is None:
        return ["none"]
    offset = "-" if form.offset is None else f"{form.offset.numerator}/{form.offset.denominator}"
    return [
        "form",
        f"{form.coefficient.numerator}/{form.coefficient.denominator}",
        offset,
        "1" if form.anonymous else "0",
        form.symbol or "-",
        *map(str, form.labels),
    ]


def shape_of(n, depth=3):
    """The kinds of an expression down to `depth` levels: what makes two index expressions look alike."""
    kind = getattr(n, "kind", type(n).__name__)
    if depth == 0:
        return kind
    return (kind, tuple(shape_of(o, depth - 1) for o in n.operands))


def index_rows(directory, limit=300):
    """Up to `limit` rows per variable kind (sample index, loop index), one per distinct (answer, shape) of an index expression."""
    rows, seen, taken = [], set(), {True: 0, False: 0}
    for path in sorted(Path(directory).glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines()):
            table, built = json.loads(line).get("nodes") or [], {}
            loops, reads = [], []
            for index, entry in enumerate(table):
                if entry["c"] not in ("At", "Loop"):
                    continue
                try:
                    n = old_node(index, table, built)
                except AttributeError:  # a dialect kind the old package does not declare
                    continue
                (reads if entry["c"] == "At" else loops).append(n)
            place = {id(n): i for i, n in built.items()}
            for var in [None, *loops[:3]]:
                for read in reads:
                    expr = read.operands[1]
                    if id(expr) not in place:
                        continue
                    answers = [index_text(access.index_form(expr, None if var is None else var.index, quotients=q)) for q in (True, False)]
                    key = (json.dumps(answers), str(shape_of(expr)), var is None)
                    if key in seen or taken[var is None] >= limit:
                        continue
                    seen.add(key)
                    taken[var is None] += 1
                    rows.append(
                        {
                            "file": path.name,
                            "line": number,
                            "node": place[id(expr)],
                            "var": None if var is None else place[id(var.index)],
                            "answers": answers,
                        }
                    )
    return rows


def sound(n):
    """Whether everything `n` reaches was built: a node whose child failed to build is left without operands."""
    pending, seen = [n], set()
    while pending:
        cur = pending.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        try:
            pending.extend(cur.operands)
        except AttributeError:
            return False
    return True


def structural_rows(directory, budget=30000):
    """The old structural keys of each recorded graph's decodable nodes as dense numbers (equal number, equal key), for
    lines spread over the recording until `budget` nodes are held; also the number of nodes and lines covered."""
    rows, held = [], 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines()):
            if number % 7:
                continue
            table, built, usable = json.loads(line).get("nodes") or [], {}, []
            for index in range(len(table)):
                try:
                    usable.append(old_node(index, table, built))
                except (AttributeError, ImportError):  # a kind the old package does not declare
                    continue
            order = walk.post_order([n for n in usable if sound(n)])
            if not order or held + len(order) > budget:
                continue
            memo = {}
            walk.structural_keys(order, memo)
            place = {id(n): i for i, n in built.items()}
            numbers, dense = [], {}
            for n in order:
                numbers.append(dense.setdefault(memo[id(n)], len(dense)))
            held += len(order)
            rows.append(
                {
                    "file": path.name,
                    "line": number,
                    "order": [place[id(n)] for n in order],
                    "numbers": numbers,
                }
            )
    return rows


def quantity_rows():
    """The old Quantity constructor's answer over slugs x packs x wire lists."""
    roles = nodes.QUANTITY_ROLE_PAIRS[::2]
    wire_lists = [
        [],
        [("vec_in", "x")],
        [("vec_in", "x"), ("vec_in", "x")],
        [("bogus", "x")],
        [("vec_in", "it's")],
        [("vec_in", "x"), ("mat_in", "m"), ("bogus", "z")],
        *([(role, "w")] for role in roles),
    ]
    rows = []
    for slug in ("", "pos"):
        for pack in ("auto", "tuple", "plain", ""):
            for wires in wire_lists:
                made = [types.Wire(name, role) for role, name in wires]
                try:
                    compound.Quantity(slug, wires=made, pack=pack)
                    answer = None
                except Exception as error:  # the refusal text is the oracle
                    answer = str(error)
                rows.append({"slug": slug, "pack": pack, "wires": wires, "refused": answer})
    return rows


def handmade_index_rows():
    """Index expressions the recordings hold too few of: labelled strides, offsets, quotients and the sample count, each with
    its own node table. A shape the old package cannot build is skipped and the rest stand."""
    index = types.TensorType((), "i32")
    i = LoopIndex(0, "i", index)
    stride, other = RoleConst(4, "stride", index), RoleConst(2, "other", index)
    count = Leaf("nsamples", "nsamples", "n", index)

    def k(value):
        return Const(value, index)

    def op(kind, *operands):
        return old_make(kind, operands)

    exprs = [
        op("mul", i, stride),
        op("add", op("mul", i, stride), k(2)),
        op("mul", op("mul", i, stride), k(3)),
        op("sub", op("add", op("mul", i, stride), k(5)), k(1)),
        op("div", op("add", op("mul", i, k(8)), k(3)), k(4)),
        op("div", op("add", i, k(1)), k(3)),
        op("mul", op("div", i, k(4)), k(2)),
        op("mul", i, count),
        op("mul", i, i),
        op("add", op("mul", i, stride), op("mul", i, other)),
        op("sub", k(10), op("mul", i, k(2))),
        op("mul", i, k(0)),
    ]
    rows = []
    for expr in exprs:
        table, root, order = record_of(expr)
        # the variable is in the table only when the expression reaches it
        var = order.get(id(i))
        answers = [index_text(access.index_form(expr, i, quotients=q)) for q in (True, False)]
        rows.append({"file": None, "nodes": table, "node": root, "var": var, "answers": answers})
    return rows


def main():
    """Writes the two data files."""
    out = Path(sys.argv[1])
    constants = {k: list(v) if isinstance(v, tuple) else v for k, v in CONSTANTS.items()}
    (out / "constants.json").write_text(json.dumps(constants, indent=1, sort_keys=True) + "\n")
    contracts = [*nodes.CONTRACTS, *nodes.SCATTER_CONTRACTS, nodes.KEYED_AUTO, "no_such_contract"]
    answers = {c: nodes.compensates(c) for c in contracts}
    (out / "compensates.json").write_text(json.dumps(answers, indent=1, sort_keys=True) + "\n")
    scalar, vector = types.TensorType((), "f64"), types.TensorType((3,), "f64", "xs")
    cases = []
    for name, ttype in (("gain", scalar), ("offsets", vector)):
        for per_cell in (False, True):
            table, root, _ = record_of(cells.param_cell(name, ttype, per_cell))
            cases.append(
                {
                    "name": name,
                    "dtype": ttype.dtype,
                    "shape": list(ttype.shape),
                    "tag": ttype.tag,
                    "per_cell": per_cell,
                    "nodes": table,
                    "node": root,
                }
            )
    (out / "param_cells.json").write_text(json.dumps(cases, indent=1) + "\n")
    (out / "quantity.json").write_text(json.dumps(quantity_rows()) + "\n")
    (out / "structural_keys.json").write_text(json.dumps(structural_rows(sys.argv[2])) + "\n")
    (out / "index_forms.json").write_text(json.dumps(index_rows(sys.argv[2]) + handmade_index_rows()) + "\n")
    feathers = sorted(nodes.FEATHERS)
    rows = []
    for file in ("k1325.jsonl", "k1322.jsonl"):
        lines = (Path(sys.argv[2]) / file).read_text().splitlines()
        # the first record that holds a reduction over samples, so a kind with a feather of its own is covered
        number = next(i for i, text in enumerate(lines) if "ReduceSamples" in text)
        table, built = json.loads(lines[number])["nodes"], {}
        answers = []
        for index in range(len(table)):
            try:
                n = old_node(index, table, built)
            except AttributeError:  # a dialect kind the old package does not declare
                continue
            answers.append([index, [nodes.has_feather(n, f) for f in feathers]])
        rows.append({"file": file, "line": number, "answers": answers})
    (out / "feathers.json").write_text(json.dumps({"feathers": feathers, "rows": rows}) + "\n")
    # the loop class labels are local to the front end: confirm they are plain texts, no contract meaning
    assert (loops.ACCUMULATOR, loops.RECURRENCE) == ("accumulator", "recurrence")


main()
