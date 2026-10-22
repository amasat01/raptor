# Concepts

A short map of why raptor is shaped the way it is. Unfamiliar words (manifest, sidecar, kernel, capability and so on) are explained in the [glossary](/content/glossary).

## Three modules, one direction of dependency

- `raptor.schema` is the kernel-manifest contract: a bird-neutral core layer
  (`raptor.schema.manifest`) that any producer can write against, plus one
  extension (`raptor.schema.blocks`, the `neural_block` descriptor family)
  that a producer never has to import. `raptor.schema.dtypes` carries the
  scalar-type vocabulary both layers share.
- `raptor.protocols` is the structural-typing contract: `Backend` /
  `Executable` for program-level execution, `KernelLauncher` / `Marshal` for
  kernel-level launch and argument marshalling, `ManifestProvider` for the
  one kernel-provider facet the family ships today. Every protocol is a
  `typing.Protocol` — an implementation satisfies it by shape, not by
  inheriting from it.
- `raptor.conformance` is the interop declaration: which array frameworks
  the family bridges to, which crossings are certified today, and the row
  catalogue every owning repository's test suite is written against. See
  [Interoperability](/content/interop_protocols) for the full matrix.

Dependencies between them run one way only: `schema.blocks` may import
`schema.manifest`, never the reverse; `conformance` references `schema` and
`protocols` by name in its documentation but never imports a framework
object. raptor itself has **zero hard dependencies** — the whole package
stays importable on an interpreter with no `numpy`, no `torch`, nothing.

## A reverted design, stated plainly

The kernel-provider protocol has exactly **one** facet today:
`ManifestProvider`, a compiled kernel described by a schema-versioned
manifest. An earlier design added a second, symbolic facet — an evaluable
trace any array namespace could interpret. It was reverted on measured
evidence (compiling and `dlopen`ing a kernel ran roughly 3.8x faster than
interpreting it on numpy, even for a trivial kernel), and its orphaned
protocol declaration was deleted rather than left standing as a promise
nothing implemented. See {ref}`the full account <reverted-design>` in
[Interoperability](/content/interop_protocols), including the narrower use
case that design may still return for.

## Schema versions are a transition bridge, not a replacement

`schema_version` 1 and 2 both load today. A v1 manifest is legacy —
whole-view, single-device, exactly the behavior that predates the execution
axis. A v2 manifest adds three flat top-level keys (`exec_targets`,
`exec_access`, `exec_op`) that declare which execution structures may run a
body and how it touches sample-local versus cross-sample data. Absence of a
required v2 key is a load refused, never a guessed default — see
[`raptor.schema.manifest.check_execution_axis`](/content/api/schema) for the
exact rules.

## For contributors

Each contract has a test that fails when the contract moves without its description.

| Contract | What it is | What pins it |
|---|---|---|
| Manifest | the `raptor.schema` fields and `validate_manifest` | `tests/test_schema_v2_execution.py` enumerates every valid and invalid execution-axis case; `tests/test_goldens.py` validates every golden; `tests/test_purity.py` keeps the core layer free of other repos' names |
| Protocols | `Backend`, `KernelLauncher`, `ManifestProvider` | `tests/test_protocols.py` pins the method names; `tests/test_packaging.py` keeps `dependencies` empty |
| Rows | the interop declaration in `raptor.conformance.interop` | `tests/test_docs_interop_protocols_sync.py` checks the published matrix against the declaration in both directions; `assert_rows_complete` fails an owner whose declared row did not run |

A change to a contract changes the test and the page in the same pull request. To add a row, follow [Contributing a certification row](/content/devguide/contributing_a_certification_row).

## Where to read next

- [Tutorials](/content/tutorials) walk through the schema, the protocols and
  the conformance matrix hands-on.
- [Interoperability](/content/interop_protocols) is the family's reference
  interop page — the row-by-row certification matrix lives there, not here.
- [Glossary](/content/glossary) explains the family's shared words.
- [API reference](/content/api/index) is the generated signature-level detail
  for everything mentioned above.
