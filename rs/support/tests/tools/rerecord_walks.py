"""Re-records the key lines and digests of `walks_small.jsonl` for the fields the plane-axes change added, with a proof.

The recorded walks come from an implementation that cannot know the new fields. For each walk the core computes today, this
tool removes exactly the fields the change added from the computed key lines (the edge rule and the axis list of a store, the
edge rule of a read) and requires the result to equal the recorded key lines, line by line. A walk that passes is re-recorded
with the computed lines and digest; a walk that differs otherwise is reported and left as recorded.

    python rerecord_walks.py --migrate-inputs rs/support/tests/data/walks_small.jsonl
    cargo build --release -p raptor_support --bin replay
    REPLAY_COMPUTED=<computed.jsonl> target/release/replay rs/support/tests/data/walks_small.jsonl
    python rerecord_walks.py rs/support/tests/data/walks_small.jsonl <computed.jsonl>
"""

import json
import re
import sys

COMPACT = {"sort_keys": True, "separators": (",", ":")}
EDGE = r"'(?:clamp|wrap|zero|unchecked)'"
AXES = r"\((?:\(.*?\),?)*\)"
STORE = re.compile(rf"^(\d+\|\('(?:accum_write|wide_write)',.*), {EDGE}, {AXES}\)\)$")
READ = re.compile(rf"^(\d+\|\('at',.*?), \(('None'), {EDGE}\)\)$")
EXTENT = re.compile(r"^(\d+)\|\('axis_extent', ('[a-z0-9]+'), \(\), None, (\(.*?\)), 0\)$")


def strip_added_fields(line):
    """The key line as it was before the change."""
    store = STORE.match(line)
    if store:
        return store.group(1) + "))"
    read = READ.match(line)
    if read:
        return f"{read.group(1)}, {read.group(2)})"
    extent = EXTENT.match(line)
    if extent:
        return f"{extent.group(1)}|('plane_extent', {extent.group(2)}, (), None, {extent.group(3)}, None)"
    return line


def rename_extent_nodes(node):
    """Renames each recorded `PlaneExtent` node of the input records to `AxisExtent` of axis 0 (a flat plane's one axis)."""
    if isinstance(node, dict):
        if node.get("c") == "PlaneExtent":
            node["c"] = "AxisExtent"
            node["v"]["axis"] = {"i": "0"}
        for value in node.values():
            rename_extent_nodes(value)
    elif isinstance(node, list):
        for value in node:
            rename_extent_nodes(value)


def migrate_inputs(walks):
    """Rewrites only the lines whose input records name a `PlaneExtent`; the other lines are kept byte for byte."""
    out, changed = [], 0
    for line in open(walks):
        if '"PlaneExtent"' in line:
            record = json.loads(line)
            assert json.dumps(record, **COMPACT) == line.rstrip("\n"), "the line is written as json.dumps writes it"
            rename_extent_nodes(record["inputs"])
            line, changed = json.dumps(record, **COMPACT), changed + 1
        out.append(line.rstrip("\n"))
    open(walks, "w").write("\n".join(out) + "\n")
    print(f"{changed} line(s) with a plane extent migrated")


def main(walks, computed):
    new = {}
    for row in open(computed):
        row = json.loads(row)
        new[row["key"]] = row
    out, total, rerecorded, findings = [], 0, 0, []
    for line in open(walks):
        record = json.loads(line)
        total += 1
        fresh = new.get(record["key"])
        if fresh is not None:
            old_lines = (record["preimage"] or "").splitlines()
            stripped = [strip_added_fields(l) for l in fresh["preimage"].splitlines()]
            if stripped == old_lines:
                # only the re-recorded lines change; every other line of the file is kept byte for byte
                for was, now in ((record["preimage"], fresh["preimage"]), (record["result"]["digest"], fresh["digest"])):
                    assert line.count(json.dumps(was)[1:-1]) >= 1, "the recorded text is in the line"
                    line = line.replace(json.dumps(was)[1:-1], json.dumps(now)[1:-1])
                rerecorded += 1
            else:
                differing = next(i for i, (a, b) in enumerate(zip(stripped, old_lines)) if a != b)
                findings.append((record["key"], differing, old_lines[differing], stripped[differing]))
        out.append(line.rstrip("\n"))
    open(walks, "w").write("\n".join(out) + "\n")
    print(f"walks total {total}, re-recorded {rerecorded}, findings {len(findings)}")
    for finding in findings:
        print("finding:", finding)


if __name__ == "__main__":
    if sys.argv[1] == "--migrate-inputs":
        migrate_inputs(sys.argv[2])
    else:
        main(*sys.argv[1:3])
