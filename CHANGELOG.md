# Changelog

This file summarizes what changed in raptor release over release, grouped by
the work that produced each change rather than a raw commit list. Consult it
when upgrading between versions, or when you are trying to work out when a
particular contract or capability first landed. raptor is the RAPTOR
family's dependency-free protocol spine — the manifest schema, the
engine/launcher/provider contracts, and the interop certification matrix
that eagle, hawk, and other downstream packages each register their own rows
against — so most entries here are about a *contract* changing, not a
runtime feature; the "why" is usually more informative than the diff itself,
and is where this file tries to spend its words.

## 0.3.2 (2026-10-17)

**The demos moved to the packages they show.** The hybrid orbit and autodiff-against-torch examples now live in hawk; early termination and zero-copy interop live in eagle, each with its notebook, tests and benchmark card. raptor keeps only the contracts, so the `demo` extra is gone and `dependencies` stays empty.

**Four JAX FFI rows.** `JX-FFI-CPU-ALIAS`, `JX-OUT-XLA-OWNED`, `JX-JIT-INVARIANCE` and `STREAM-JAX-FFI-ORDER` certify eagle's JAX bridge: it reads XLA's buffers in place, returns XLA-owned outputs that are zero where the kernel writes nothing, gives the same answer eager, jitted, vmapped and scanned, and rides XLA's stream with no host wait. The matrix has 32 rows.

**Ten rows for framework code inside an eagle graph.** `INJ-TIER-NAMED`, `STREAM-INJECT-STREAM-ORDER`, `STREAM-INJECT-HOST-ORDER`, `T-CHILD-OWNED`, `INJ-LOOP-HOOK`, `INJ-VJP-CHAIN`, `INJ-TABLE-LIVE`, `STREAM-JAX-INJECT-ORDER`, `JX-INJECT-HOST` and `JX-LOOP-INVARIANCE` certify how eagle runs the reader's own torch or JAX code between, inside and around its launches: the tier is named, the stream fences hold, a framework graph stays alive, a per-step hook keeps the loop's answer, gradients pass through the reader's model and weights are read live. `JX-OUT-XLA-OWNED` now says that carried-state planes alias their operand. The matrix has 42 rows.

**`INJ-RUN-VJP`.** Gradients through a recorded stepping loop whose step runs the reader's model before the kernel, through the reader's vjp, match pure torch autograd. The matrix has 43 rows.

**A row now has legs, and the gate checks that its tests ran.** `RowDecl.legs` names the platforms a row must pass on (`cpu`, `cuda`; both by default), and `register_row(row_id, leg=...)` says which one a test covers. `assert_rows_complete(owner, ran=True)`, now the default, counts a row only if its test ran and passed on every declared leg, as recorded by the new `raptor.conformance.pytest_rows` plugin. Before, a row counted as soon as its module was imported, so a run that deselected or skipped every row still passed. Load the plugin from `conftest.py` and mark the gate test `rows_gate`; `ran=False` keeps the old import-time check. The portability check now pins 71 passes and 29 `repo_local` tests.

## 0.3.1 (2026-10-16)

**JAX is certified.** Five rows cover JAX arrays in the interop matrix: `JX-IN-CUDA-ALIAS`, `JX-IN-WRITE-REFUSED`, `JX-OUT-CUDA-ALIAS`, `JX-IN-CPU-COPY` and `STREAM-JAX-PRODUCER-ORDER`. JAX moves from the roadmap to the certified list in the docs, and eagle registers against the rows.

**SPDX licence metadata.** `pyproject.toml` declares `license = "Apache-2.0"` and `license-files`, the form current packaging tools expect.

**docs-fresh tells its failures apart.** `tools/docs_fresh.sh` exits with a different code for stale docs, a missing card and a missing tool, so a CI log says which one went wrong. Tests that need eagle are marked `cross_repo`, and the wheel leg of the docs-fresh tests runs without skips.

**Portability check.** Test markers are registered for the out-of-repo copy of the suite, and the check pins its pass count at 59.

**Docs.** The pages are regrouped into shorter sections, with a family glossary. The Colab and Kaggle setup cell installs the CUDA 12 or 13 extras, can test a release candidate and upgrades the family's own packages. Pages actions moved to the Node 24 releases.

## 0.3.0 (2026-10-09)

**Free-threaded CPython support, and the harness that certifies it
(2026-10-09).** The family now states one threading contract (README
"Threading", `docs/threading.md`) and raptor hosts the harness every package
tests it with: `raptor.conformance.freethreading` offers
`require_free_threaded()`, `hammer()`, `assert_gil_free()` and a planted-race
canary that turns a run which cannot expose a known race into a red, vacuous
result instead of a green one. The gate refuses to run under `-X gil=...` or
`PYTHON_GIL`, because forcing the GIL state hides a missing declaration. raptor
itself is pure Python and needs no declaration. Additive; the harness is
standard library only and `dependencies` stays empty.

## 0.2.0 (first public release)

raptor is the RAPTOR family's protocol spine: the manifest schema, the
engine/launcher/provider contracts, and the interop certification matrix
other packages register against.

**Documentation, NVIDIA Warp certification, and contribution terms
(2026-10-01).** Added the documentation site: tutorials, examples, the
interoperability reference, and the full Python API. Certified NVIDIA
Warp in the interop certification matrix (23 rows). Contributions are
welcome under Apache-2.0 with a DCO sign-off, matching the rest of the family.

**Licence and worked example (2026-09-23).** Licensed under Apache-2.0.
`examples/two_body_hybrid.py` added: a hand-written two-body
kernel with a torch-trained drag correction, traced by hawk and run by eagle
as one replayed CUDA graph, with the manifest validated by this package's
schema. The `demo` extra lists what the example needs beyond raptor itself.

**Packaging and CI finalization (2026-07-31).** The distribution name
changed to `raptor-core`; the import name stays `raptor`. This followed a
PyPI namespace check that found `raptor` already taken by an unrelated
package, resolved by prefixing the family's *distribution* names only,
leaving every `import` statement in downstream code unchanged. See the
`raptor-` naming comment at the top of `pyproject.toml` for the exact
reasoning.

**Packaging infrastructure (2026-07-30).** Version drift between
`pyproject.toml` and the installed package now fails loudly instead of
silently shipping a stale number, `py.typed` was added so type checkers pick
up raptor's stubs, and the wheel/sdist build was extended with a round-trip
check (build an sdist, build a wheel from that sdist, confirm the two
wheels match) so a packaging regression cannot slip through unnoticed.

**JVP full compatibility (2026-07-29).** Forward-mode automatic
differentiation (JVP — propagating a tangent direction through a
computation, the counterpart to the reverse-mode VJP everything before this
point supported) reached parity with the reverse-mode path across every
surface raptor's certification matrix tracks. This closed the last open
exception in the family's design record.

**Compatibility for a new downstream package, torch interop (2026-07-28).**
Extended the manifest/interop contracts to accommodate a new downstream
neural-runtime package and its torch-tensor interop path, ahead of that
package's own first release.

**Downstream compatibility and completed untangling (2026-07-23).** raptor
itself was extracted in this window as the dependency-free foundation the
rest of the family declares against, splitting apart what had previously
been entangled directly between two of the other repos.

**Initial commit (2026-07-23).** raptor's first commit: the manifest schema,
the engine/launcher/provider contracts, and the row declarations every other
family repo's certification tests register against.
