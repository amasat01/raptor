# Contributing a certification row

Add a new array framework, or a new "crossing" (a way one of raptor's
pieces hands data to or from that framework) for one already certified, to
[the interop certification matrix](/content/interop_protocols).

**Time:** ~10 min to read · **You need:** a checkout of this repository
plus the one that owns the crossing you are adding, and `pytest`.

Plugging in a new row is a two-repository process, in order — there is no
way to skip the first step.

## 1. Declare the row in raptor

`ROWS`, `CERTIFIED` and `UNIVERSE` are closed constants declared in
`raptor.conformance.interop`, in this repository. Open a pull request
against raptor that adds the row's id, framework and kind (one of `ALIAS`,
`COPY`, `STREAM`, `ENV` or `INVARIANCE` — see
{ref}`the two transfer laws <two-transfer-laws>` on the Interoperability
page) to `ROWS` (and, for a brand-new framework, to `UNIVERSE`/`CERTIFIED` or
`ROADMAP`). This repository is the declaration's only owner; no other
repository may add a row to it.

## 2. Register the test in the owning repository

Only once the row exists in raptor does the repository responsible for
that crossing register a test for it, with `@register_row(row_id)`:

```python
from raptor.conformance.interop import register_row

@register_row("T-IN-CUDA-ALIAS")
def test_my_cuda_alias():
    ...
```

`register_row` refuses an undeclared row id immediately, at decoration
time: `KeyError: "register_row: 'NOT-A-REAL-ROW' is not a declared row in
ROWS"`. A typo'd id fails loud, not silently — there is no way to certify a
crossing by registering a test for a row that step 1 never declared.

## 3. Pass the two hygiene checks your PR runs into

Your new test file must pass two checks: no skip tokens, and it must be
found by the matrix scanner.

```python
import pathlib

from raptor.conformance.interop import (
    assert_no_skip_tokens,
    discover_matrix_modules,
)

matrix_dir = pathlib.Path("tests/conformance")
modules = discover_matrix_modules(matrix_dir)
assert_no_skip_tokens(*modules)
```

`discover_matrix_modules` finds every file that calls `register_row`, by
scanning its AST (Python's own parsed syntax tree) rather than a
hand-maintained list — your new file is found automatically.
`assert_no_skip_tokens` then refuses any `pytest.skip` / `skipif` /
`importorskip` in those files outright: nothing is printed when a module
passes both checks cleanly; a banned skip token raises
`AssertionError` naming the file and the token.

> Why two checks, not one: registration happens at **import time** (when
> Python loads the test module), not when the test itself runs, so a
> "completeness gate" (a check that every declared row has a registered
> test) can otherwise not tell a row that was never collected apart from
> one that was collected and then skipped. These two checks are what keep
> that gate trustworthy.

## See also

[Interoperability](/content/interop_protocols) is the full, human-readable
matrix this declaration publishes.
[Read the interop matrix](/content/tutorials/04_plug_into_the_interop_matrix)
reads the same declaration from the user side, with no pull request
involved.
