# Foundational interoperability protocols

This page tells you which array libraries can hand data to RAPTOR, and
whether that copies.

The family is built in layers: **raptor** owns the contracts and depends on
nothing; **hawk** authors and compiles kernels; **eagle** launches, captures
and marshals them (marshals = converts an array into exactly the layout a
compiled call expects); downstream packages wire kernels into trainable programs.
Each works alone — adding a companion buys speed or a deployment option,
never a new ability the family did not already have.

## DLPack is the contract

**DLPack** is a standard, framework-neutral way for one library to hand
another an array — CPU or GPU — without copying it. Every crossing on this
page is, mechanically, a DLPack exchange: `cupy`, `torch` and NVIDIA Warp
arrays all expose `__dlpack__`/`__dlpack_device__`, and the family reads or
writes through that protocol rather than through a framework-specific code
path. That is what lets the same kernel accept a `numpy`, `cupy`, `torch` or
Warp argument and route to the right provider without the caller naming
which one.

For a runnable walkthrough rather than a table, see the
[zero-copy interop example](/content/examples/zero_copy_interop): a pointer-stable
host round-trip through `numpy`, exercised for real, plus a worked read of
what the `cupy`/`torch` device rows below certify. A CUDA-capable `torch`
build exchanges device tensors the identical way, through the same
`__dlpack__` protocol `cupy` uses — nothing about the crossing changes
between the two frameworks on the device side.

:::{admonition} Roadmap
:class: note

`jax` and `tensorflow`/`keras` are both DLPack-capable and both foreseen —
see {ref}`Certified frameworks <certified-frameworks>` below. Neither is
certified today. Treat any claim that one of them works
now as a documentation bug, not a feature: this page cites a row for every
capability it claims, and no row names either framework. NVIDIA Warp is no
longer in this category: see the certified rows below.
:::

## The rule this document obeys

Every normative claim below either **cites an executable certification row**, or
is explicitly labelled **ROADMAP** or **UNCERTIFIED**. There are no uncited
capability claims.

The rows are declared once, in `raptor.conformance.interop`, and each is
implemented as a real test in the repository on the other side of the
crossing it certifies. That declaration — not this page — is the authority. A doc-sync test asserts
that the row identifiers on this page and the declared set are identical in both
directions, so the two cannot drift; if they ever disagree, the declaration
wins. Read this document as the human-readable face of a machine-checked matrix,
never as a substitute for it.

## Dispatch is by input type

A compiled kernel is called directly, and the framework of the arguments
selects the execution path: a `numpy` argument routes to the native host
provider, a `cupy` argument routes to the `nvcc` + eagle device path. The
caller writes one call and gets host or device execution according to what
it passed.

The host path is genuinely CUDA-free: no `nvcc` (the CUDA compiler), no
`cupy`, no eagle anywhere on it. It works by compiling the kernel once,
caching the compiled result by a hash of its source, `dlopen`-ing that
compiled library (loading it at runtime the way an OS loads any shared
library), and marshalling the call's arguments (converting them into the
exact layout the compiled function expects) before calling in.

The cuda-free half of that claim is certified by running a subprocess with
`nvcc` removed from `PATH` and with `cupy` and eagle blocked at import,
compiling and executing a kernel there, and asserting inside that subprocess
that neither module was ever imported. The provider lookup is a precedence
chain — the ordered list dispatch checks before picking a provider — so
further providers can be admitted without changing the call surface; no row
certifies any provider beyond these two, and none is claimed.

## Backends: the execution substrate

A **wired training program** — a training loop assembled once from a backend
and then run many times, as opposed to one rebuilt from scratch on every call —
binds to a backend by name.

Three backends exist:

- `eagle` — compiled kernels and **captured graphs** (a GPU's sequence of
  launches recorded once, then replayed without re-issuing each one); the
  performance design point.
- `reference` — the **reference wiring engine**: a plain, uncompiled
  implementation of the same wiring, run over the **host JIT** (the CPU kernel
  path that compiles once and caches the result, with no GPU or `cupy`
  involved), used as the correctness baseline the other backends are checked
  against.
- `torch` — described next.

**TorchBackend is composition, not interpretation.** It routes on the device of
the input tensors: CPU tensors go to the reference wiring engine over the host
JIT (which is genuinely cuda-free), and CUDA tensors go to the eagle compiled
path.

Any other torch device raises `CapabilityError` naming the device and saying
ROADMAP — this covers MPS (Apple's GPU backend), ROCm (AMD's GPU compute
stack) and XLA (Google's whole-program compiler). There is deliberately **no
silent `.cpu()` fallback**: falling back would report success for a device
the family cannot actually accelerate. `T-DEVICE-FAILLOUD` certifies that
behaviour, asserting both the exception type and that its message names the
offending device.

The CPU input crossing is certified zero-copy by `T-BACKEND-CPU-IN-ALIAS` and
the CUDA input crossing by `T-BACKEND-CUDA-IN-ALIAS`.

Forward and reverse-mode run as one **program-level autograd** step — the
backward pass flows through the backend boundary exactly as it would through
plain torch ops, so a caller's `loss.backward()` works unchanged on either
leg. The end-to-end eagle-less training loop — forward, backward, and a
parameter update on CPU tensors with eagle, `cupy` and `nvcc` all unavailable —
is gated by a named test in the downstream package's own suite
(`test_torch_backend_training.py::test_eagle_less_torch_backend_training_cuda_free`)
rather than by a matrix row; it is executable and it runs, but it is a suite gate,
not a row-level certificate, and this page does not pretend otherwise.

Forward-mode differentiation (JVP) through the backend protocol is **ROADMAP**.
No row certifies it.

(certified-frameworks)=
## Certified frameworks, and what "roadmap" means mechanically

The certified set is **numpy, cupy, torch and NVIDIA Warp**. **jax and
tensorflow/keras are ROADMAP, absent by declaration.** The family's stated
horizon includes them; no row certifies them and none pretends to.

"Absent by declaration" is enforced in both directions, and neither direction is
ever a skip:

- a **certified** framework missing from the gate environment is a **failure**,
  raised by `certified_framework` — the matrix must never skip its way to green;
- a **roadmap** framework that turns out to be *installed* is **also a failure**,
  raised by `roadmap_framework` — because the moment it is present, the claim
  that nothing exercises it stops being checkable.

A skipped test returns success to the runner. A conformance matrix that skips a
framework and reports green certifies nothing, so the matrix rows are built to
be structurally incapable of skipping: a scanner discovers every file containing
a registered row and rejects skip machinery — skip tokens and availability
markers alike — on the registered functions themselves.

(reading-the-matrix)=
## Reading the matrix, in miniature

Before the full 23-row table below, here is how to read any one row. These
three all share one crossing class, **ALIAS** — pointer identity plus
write-through: the two sides report the same address, and a write on one
is visible on the other:

| row | framework | owner | class | certifies |
|---|---|---|---|---|
| `T-IN-CUDA-ALIAS` | torch | eagle | ALIAS | a CUDA tensor becomes a zero-copy device view |
| `WP-IN-CUDA-ALIAS` | warp | eagle | ALIAS | a `wp.array` becomes a zero-copy device view |
| `T-BACKEND-CPU-IN-ALIAS` | torch | downstream | ALIAS | the CPU backend's input crossing is a zero-copy view |

Four more classes appear in the full matrix: **COPY** (pointers differ, and
mutating either side afterwards leaves the other unchanged), **STREAM**
(certifies ordering, not placement), **ENV** (certifies a property of the
environment — what is absent, or what is refused), and **INVARIANCE**
(the same eagle execution body agrees across different execution
structures — no framework on either side, because there is no crossing).
{ref}`The full certification matrix <certification-matrix>` below defines
each in full and lists every one of the 23 rows; the acronyms
(ALIAS/COPY/STREAM/ENV/INVARIANCE) are each defined again, in depth, in
{ref}`The two transfer laws <two-transfer-laws>` further down this page.

(certification-matrix)=
## The certification matrix

Twenty-three rows. `framework` is what the row bridges to; `owner` is the
repository whose suite carries the assertion; `class` is how data crosses, in
the vocabulary defined in the next section.

Eight rows bridge to **no** framework — their `framework` column reads `—`.
Four of those are the `execution_*` rows, each an eagle-internal invariant of
the **heterogeneous execution contract** (the rule that one plugin body must
give equivalent results under every eagle execution structure, not only the
one it happened to be written against), not a crossing between the family
and an array framework.

The other four are the `HAWK-*` rows: rows that exist only because hawk
exists — a round-trip row and two reference-comparison rows (below).

Every row on this page is **declared before the code it certifies is
built**: the `execution_*` rows are declared here — like `CP-CAPTURE-REJECT`
and every other eagle-owned row above them — ahead of the eagle-side work
(`exec/`, `eagle.plan`, the loader) that will implement and collect them, and
the `HAWK-*` rows the same way, `owner=hawk`, ahead of hawk's own consumer
implementation. That is the family's **declare-before-build** convention: a
row may describe a capability that does not exist yet.

`HostTeam`, `DeviceKernel` and `RankPartition` below are eagle's names for
its three execution structures: a team of CPU threads, a single GPU, and a
multi-process partition, respectively — the same body can run under any of
them.

| row | framework | owner | class | certifies |
|---|---|---|---|---|
| `CP-CAPTURE-REJECT` | cupy | eagle | ENV | graph capture rejects a non-device input instead of silently allocating during capture |
| `T-IN-CUDA-ALIAS` | torch | eagle | ALIAS | a CUDA tensor becomes a zero-copy device view: pointer identity, device-ordinal agreement, and write-through |
| `T-OUT-CUDA-ALIAS` | torch | eagle | ALIAS | a caller-provided CUDA output tensor is filled in place, and the engine's result exports back pointer-equal and write-through |
| `T-IN-CPU-COPY` | torch | eagle | COPY | a CPU tensor round-trips as a torch tensor via a copy, write-isolated in both directions |
| `T-BRIDGE-COPY` | torch | downstream | COPY | the export bridge (the path that hands a result array out as a tensor) copies before handing out a tensor: the export never aliases the live replay buffer (the buffer a captured graph reuses across replays), and a later forward cannot mutate it |
| `T-BACKEND-CPU-IN-ALIAS` | torch | downstream | ALIAS | the CPU backend's input crossing is a zero-copy, write-through view of the tensor's storage |
| `T-BACKEND-CUDA-IN-ALIAS` | torch | downstream | ALIAS | the CUDA backend's forward inputs and reverse-mode cotangents alias the device engine's view |
| `T-DEVICE-FAILLOUD` | torch | downstream | ENV | an unsupported torch device raises `CapabilityError` naming the device and saying ROADMAP — never a silent fallback |
| `STREAM-TORCH-PRODUCER-ORDER` | torch | eagle | STREAM | work enqueued on a non-default torch stream before a launch is seen by that launch |
| `STREAM-CUPY-PRODUCER-ORDER` | cupy | eagle | STREAM | a device write enqueued before a graph replay, with no synchronisation, is seen by the replay — on a caller stream and on the legacy null stream (CUDA's original default stream, used when no stream is named) alike |
| `STREAM-TORCH-CONSUMER-ORDER` | torch | eagle | STREAM | a torch operation consuming the result immediately, with no synchronisation, reads the finished value |
| `STREAM-IDENTITY` | torch | eagle | STREAM | inside the launch context, the device library's current stream *is* torch's current stream |
| `WP-IN-CUDA-ALIAS` | warp | eagle | ALIAS | a `wp.array` becomes a zero-copy device view via `eagle.interop.import_buffer`: pointer identity, the legacy-DLPack access flag (`"unknown"`), and write-through |
| `WP-OUT-CUDA-ALIAS` | warp | eagle | ALIAS | the reverse crossing, `wp.from_dlpack` of an eagle-side buffer: pointer-equal and write-through in both directions |
| `STREAM-WARP-PRODUCER-ORDER` | warp | eagle | STREAM | a device write enqueued on a dedicated `wp.Stream` before a consumer import, with no synchronisation, is seen by the consumer exactly when `import_buffer` is told that consumer's stream (the DLPack stream-code contract) |
| `execution_partition_identity` | — | eagle | INVARIANCE | the same `sample_local`/`cross_sample_read` body produces bit-identical results under `DeviceKernel` (whole) vs. a 2-way partition vs. a 2-rank `RankPartition` |
| `execution_host_device_twin` | — | eagle | INVARIANCE | the same body's `HostTeam` and `DeviceKernel` results agree within a fixed host/device tolerance band — the band this row's own test defines |
| `execution_illegal_placement_refused` | — | eagle | ENV | a plan placing a body on a structure its declared `exec_access` class forbids is refused at plan time, naming the violated rule |
| `execution_layout_selfcheck_refused` | — | eagle | ENV | a loader refuses a plugin whose ABI tag agrees but whose exported layout sizes disagree (a self-check), and refuses an outright tag mismatch |
| `HAWK-DIFF-POSITIVE-ROUNDTRIP` | — | hawk | INVARIANCE | a two-wire compound quantity (one value carried as two linked slots) binds distinct slots and both wires' values round-trip on the host path — the positive twin of the matching refusal check |
| `HAWK-EXEC-DEPLOY-LOADABLE` | — | hawk | ENV | a HAWK artifact loads under `eagle.registry.load_manifest` + `eagle.plan`; a v1 loader refuses a v2 artifact and vice versa |
| `HAWK-EXEC-REFERENCE-HOST` | — | hawk | INVARIANCE | every fixture kernel under `HostTeam`/`DeviceKernel` agrees with HAWK's own serial host path (the known-correct reference every other structure is checked against), bit-identical where the inner structure is host, banded otherwise |
| `HAWK-EXEC-REFERENCE-RANKPARTITION` | — | hawk | INVARIANCE | the `RankPartition(2)` arm of the same reference comparison, bit-identical for elementwise classes when the inner structure is host |

(two-transfer-laws)=
## The two transfer laws

Every crossing between a framework and the family is declared, and the
declaration is asserted positively rather than inferred:

**ALIAS** means pointer identity plus write-through. The row asserts that the two
sides report the same address and, where the crossing is bidirectional, that a
write on one side is visible on the other with nothing in between.

**COPY** means distinctness plus write-isolation. The row asserts that the
pointers differ *and* that mutating either side afterwards leaves the other
unchanged. A pointer check alone would pass on an accidental copy; the isolation
half is what makes the row a contract.

Three further classes appear in the table. **STREAM** rows certify ordering
rather than placement. **ENV** rows certify a property of the environment —
what is absent, or what is refused. **INVARIANCE** rows certify that the same
plugin body produces equivalent results under different eagle execution
structures — bit-exact for `sample_local`/`cross_sample_read` bodies, or
within a fixed tolerance band otherwise; unlike ALIAS/COPY, an INVARIANCE row
has no framework on either side of the crossing, because there is no
crossing — both sides are eagle.

**The output direction through the export bridge is always a COPY.** The export
path copies unconditionally before wrapping a result as a tensor, whether the
underlying array is a scratch buffer or a live replay view. This is deliberate
and it is a feature: a program's buffers are reused across replays, so an export
that aliased them would be silently corrupted by the next forward pass. That
contract is certified by `T-BRIDGE-COPY`, which asserts the semantic half — a
previously exported tensor survives a later forward with different inputs
unchanged — and not merely the pointer half.

One consequence is worth stating rather than leaving as a surprise: **no backend
row can be an alias in the output direction, on either leg.** The two backend
alias rows name their direction in their own identifiers for exactly this
reason: the claim only ever covered the input crossing, never the output one.

## The stream contract

Stream correctness is certified **behaviourally**. Rows `STREAM-TORCH-PRODUCER-ORDER`,
`STREAM-CUPY-PRODUCER-ORDER`, `STREAM-TORCH-CONSUMER-ORDER` and `STREAM-IDENTITY`
manufacture an adversarial schedule — a long occupying kernel, then a data write
queued behind it, then the launch, with no synchronisation anywhere — so that a
launch which failed to ride the caller's stream would read stale data with
certainty rather than by luck. Ordering is then asserted on the result. The
ordering guarantee itself lives in the graph pipeline's launch path, which
records and waits on events so that a replay observes work already enqueued on
the caller's stream and on the legacy null stream alike.

**`__cuda_stream__` is ROADMAP.** Nothing in the pinned stack produces or
consumes it; eagle exchanges streams as a raw handle instead. Certifying the
presence of a protocol attribute that nothing calls would have certified nothing,
so the family certifies the observable ordering instead.

The full rules (stream codes, event fencing, ownership and access flags, and the
named refusals) are in
[eagle's interoperability contract](https://amasat01.github.io/eagle/content/interop_contract.html).

## Manifest key order: strict writer, tolerant reader

Top-level key order in a kernel manifest is **contractual for writers**. The
Python validator enforces it, and the **golden artifacts** (committed
reference manifests, used as fixtures across the family's tests) carry it
byte-stably, so a manifest produced through the family is reproducible.

The C++ reader is **deliberately order-agnostic**, and that asymmetry is
Postel's law applied on purpose, not drift (Postel's law: be strict in what
you write, lenient in what you read). Tightening the reader would buy no
capability: a wrong-order manifest can only come from a producer that
bypassed validation, and the bypass is the actual defect. The tolerance is
pinned by a test that feeds the reader a manifest with shuffled top-level
keys and asserts it parses, so any future tightening has to be a conscious
change of contract rather than an accident.

## Schema v2: the execution axis

`schema_version` **1 and 2 both load today** — a transition bridge, not a
replacement. A v1 manifest keeps `aether_abi: "aether-abi/1"` and must not carry
any `exec_*` key; it plans as legacy — whole-view, single-device, exactly
today's behaviour. A v2 manifest carries `aether_abi: "aether-abi/2"` plus three
new **flat** top-level keys, inserted between `aether_abi` and `plugins` in the
writer's key order:

- `exec_targets` — a non-empty list drawn from `device`/`host`, no duplicates:
  which execution structures may run the body.
- `exec_access` — one of `sample_local`, `cross_sample_read`,
  `cross_sample_write`, `mapreduce`: how the body touches sample-local vs.
  cross-sample data, declared by the **emitter** (hawk's code-generation
  stage, which writes the manifest) from the **trace** (hawk's recorded
  representation of the kernel), never guessed.
- `exec_op` — one of `sum`, `times`, `max`, `land`: **required** when
  `exec_access` is `mapreduce`, **forbidden** otherwise.

Absence of a required key is a load refused, not a default guessed. The keys
are flat rather than one nested `execution` block because the manifest is a
strict-key, order-contractual document a C++ text scanner reads — flat keys are
the shape this schema already uses everywhere else.

## What each install gets you

The **degradation law** — dropping an optional dependency may cost speed or a
deployment option, but never the ability to run the same program — governs
*backends*. A kernel provider is different — it is the program's content, not
its substrate, so provider presence sits outside the law.

| install | what runs |
|---|---|
| `raptor` alone | Validate kernel manifests (`raptor.schema`); check an implementation against the family's `typing.Protocol` contracts; read this certification declaration. No kernel compiles or runs. |
| `+ hawk` | Author and compile a kernel for a host and/or device target, and publish its manifest. Still does not run it. |
| `+ eagle`, host target | Load the compiled artifact and run it on CPU threads (`eagle.exec.HostTeam`) — no GPU, no `cupy`, no `nvcc` anywhere on this path. |
| `+ eagle`, device target, `+ cupy` | Capture and replay the kernel as a CUDA graph on the GPU. |
| `+ torch` | The `T-*` rows above: a CPU tensor crosses by copy, a CUDA tensor by zero-copy alias. |

Torch devices other than CPU and CUDA are **ROADMAP** on every row of that table.
The torch CPU and CUDA crossings are certified by `T-BACKEND-CPU-IN-ALIAS` and `T-BACKEND-CUDA-IN-ALIAS`; the refusal
to fake the rest by `T-DEVICE-FAILLOUD`.

## Going deeper (optional)

Historical context, an edge-case hazard, and the completeness gate's own
fine print — none of it needed to read the matrix above, all of it real.

### Row ownership, in transition

The numpy/cupy/torch crossing rows hawk will eventually own stay under their
current owner (above) until that migration happens, then are re-owned in one
move — nothing duplicated here for them in the meantime.

(reverted-design)=
### A reverted design, stated plainly

The kernel-provider protocol has exactly **one** facet today: a compiled kernel
described by a schema-versioned manifest.

An earlier design added a second, *symbolic* facet — an evaluable trace that any
array namespace could interpret, which would have given the code generator a
standalone interpreter and downstream packages a device-agnostic torch backend
for free. It was **reverted** on measured evidence: compiling and dlopening a
kernel ran measurably faster than interpreting it on numpy, even for a trivial
kernel, so the interpreter would have been a slow path carrying a second
semantics to keep in sync. What ships instead is the native host provider
described above, and its orphaned protocol declaration was deleted rather than
left standing as a promise nothing implemented.

That mechanism may return, scoped to what only it can do: an interpretive basis
for torch devices other than CPU and CUDA. That is **ROADMAP**.

### A sidecar nesting hazard

A separate and unrelated hazard lives in the sidecar reader (the sidecar is
the small per-kernel JSON file describing how to call it): a neural-block
descriptor embeds nested execution references whose keys share names with
top-level keys, so a nested object appearing before its parent's own key could be
read as the parent's value. That is a nesting hazard, not a key-order contract;
it is handled by bounding and masking the nested objects before the top-level
scan, and the conformance corpus is written in the adversarial order so the case
is executed rather than reasoned about.

### What the completeness gate does not certify

Each owning repository asserts that every row declared for it has a registered
test. Read the scope of that guarantee carefully, because it is narrower than it
looks.

Registration happens when the test module is **imported**, not when the test is
executed. The completeness gate therefore certifies **row presence, not row
execution**. Two consequences follow, and both are load-bearing:

- Explicitly deselecting the matrix by marker in a CI job is safe — the rows
  still register, so the gate does not go falsely red — and it is visible in the
  job definition rather than hidden in a runtime skip. That is the only sanctioned
  way a row may fail to run.
- Because deselection is safe, **at least one job must run the matrix without
  deselecting it**, or every job is green and nothing is certified. Any pipeline
  that adopts this matrix owes that job.

One further constraint: the collected-row registry is per-session, so a partial
selection of tests will fail the completeness assertion for the right reason but
at the wrong time. Suites containing the completeness assertion must therefore be
run whole, in a single process, and not distributed across worker processes.

Everything else on this page is a row, and every row is a test.
