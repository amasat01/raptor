# raptor

raptor is the IR of the family: the node kinds of a numerical program, the rules that type, check and differentiate them, and the
canonical form and digest that identify a program. The kinds, their construction checks and their rules are data
(`raptor/ir/core.json`); the core that reads that data is Rust behind a Python binding. Python 3.9 or newer.

## Layout

- `raptor/` the Python package. `ir/` the node classes (generated at import from the manifest), the walk, the access class;
  `rules/` the rule programs' Python side; `schema/` the manifest and descriptor validators; `protocols/`, `conformance/` the
  contracts other packages implement. `_facts.py` serves the facts both languages share.
- `rs/core/` the dependency-free crate: canonical form (key text), digest, manifest shape validation, and the specs generated from
  `core.json` by `build.rs`. Nothing here knows about Python.
- `rs/binding/` the pyo3 crate: the rules, typing, constructor checks and substitution, working over Python node objects. A new
  front-end language adds its own binding stage over `rs/core`; it does not touch the core.
- `docs/MANIFEST.md` the manifest format. `tools/` the maintainer's tools. `tests/` the suite and the guards.

## Build, test, gate

    make build                 # maturin build into ./dist, the module unpacked into raptor/ (never `maturin develop`)
    make test                  # the whole suite
    make guards                # the fast guards: hardware-free, one-home, python-floor, node construction, tables, kinds diff
    make gate                  # build, guards, suite and `cargo test`, all run; JOBS=2 PYTHON=... MATURIN=... override defaults
    make bench                 # nanoseconds per Op / Const / At construction (tools/bench_construct.py)

The free-threaded build has no stable ABI: `maturin build --no-default-features` with a free-threaded interpreter.

## Change one thing

A fact that both languages use (a vocabulary, a version, a width) is a `consts` row named `group.NAME`; Python reads it through
`raptor/_facts.py`, Rust through the module `facts` of the core. `tests/test_one_home.py` fails if one is spelled again as a literal.

To add an attribute to a kind, edit its row in `core.json` and nothing else: `slots`, `params`, `assigns`, `dedup`, `state`, `from`,
`ctor`. Adding `note: str = ""` to `Finish` is eight small edits of that row of the JSON (18 lines added, 7 changed), and no Python or Rust file. Then `make build`.
`python tools/kinds_diff.py HEAD` lists, per kind, const, check and rule, what the working file changed against a revision.

## The four small languages of the manifest

- kinds: a row declares a kind. `{"name": "const", "class": "Const", "slots": ["literal"], "params": "literal: Plain, ttype: TensorType",
  "dedup": ["one", "r:literal"], "ctor": {"steps": [["node", "ttype", "()"]], "out": ["literal"]}}`
- checks: `["check", "value.ttype != _t", "finish {name!r}: the condition is typed {value.ttype!r}"]` refuses with the rendered text when
  the expression is true; `["let", name, expr]` binds; `["node", type_expr, operands_expr]` runs the base rules of every node. Named
  reusable checks live under `checks` (`node_base`).
- rules: programs that differentiate a kind: `"mul": [["a", ["mul", "g", "b"]], ["b", ["mul", "g", "a"]]]` gives the adjoint of each
  operand in terms of `g`, the incoming adjoint (`rules.bin_vjp`).
- consts: `"LEAF_ROLES": ["vec_in", "mat_in"]` a vocabulary the checks test membership in (`role not in SINK_ROLES`); `"ir.BODY_KEY_HEX": 32`
  a fact shared with the core.

`docs/MANIFEST.md` has the full grammar of expressions, steps, dedup items and the key-text rule.

## A pitfall

A check may use only its row's parameters, `let` names, the keys of `consts`, the functions of `BUILTINS` (`rs/binding/src/lib.rs`)
and the class names. Anything else, `str` for one, fails when the module is imported, and the message names the row and lists what
is allowed:

    Finish: unknown name str (core.json row "Finish"; a check may use its parameters [...], the keys of `consts` (50 of them), the functions [...] and the classes ...)

A manifest is checked without importing it by `_core.check_manifest(text)`.

## Rules for the package

No hardware name and no machine number anywhere in `raptor/`, `rs/` or `core.json` (`tests/test_hardware_free.py`); the Python source
runs on 3.9 (`tests/test_python_floor.py`, which also checks the family's floors agree).
