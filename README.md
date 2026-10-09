<p align="center">
  <img src="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/brand/family_raptor.svg" alt="" height="56">
</p>
<h1 align="center">raptor</h1>
<h3 align="center">raptor: one manifest, any engine</h3>
<p align="center">A compiled GPU kernel ships with a small JSON file that says what it takes, what it returns and
where it can run. raptor defines that file, so one package can build a kernel and another can run it.</p>
<p align="center"><b>Runs on:</b> CPU, no GPU needed · <b>You need:</b> Python 3.9+, nothing else to install raptor itself</p>

<p align="center">
  <a href="https://github.com/amasat01/raptor/actions/workflows/ci.yml"><img src="https://github.com/amasat01/raptor/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://amasat01.github.io/raptor/"><img src="https://github.com/amasat01/raptor/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://github.com/amasat01/raptor/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/raptor-core/"><img src="https://img.shields.io/pypi/v/raptor-core.svg" alt="PyPI"></a>
  <a href="https://doi.org/10.5281/zenodo.23250234"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23250234.svg" alt="DOI"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_raptor_dark.svg">
    <img src="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_raptor_light.svg" alt="The RAPTOR family: hawk (write it), eagle (run it), aether (the C++/CUDA numerics underneath) and raptor (the shared contract); you are looking at raptor." width="760">
  </picture>
</p>

<p align="center">

[aether](https://amasat01.github.io/aether/) · [hawk](https://amasat01.github.io/hawk/) · [eagle](https://amasat01.github.io/eagle/) · [raptor](https://amasat01.github.io/raptor/) · [the family](https://amasat01.github.io/)

</p>

raptor holds the shared contracts of the [RAPTOR family](https://amasat01.github.io/): the manifest format and certification
contracts that let a kernel built once work with the code you already have. See
[where RAPTOR fits](https://amasat01.github.io/where_raptor_fits.html) for how the four pieces combine.

| **0** | **4** | **1** |
|:---:|:---:|:---:|
| hard dependencies (pure Python) | array libraries certified: NumPy, CuPy, PyTorch, Warp | JSON manifest per kernel bundle (a kernel's compiled artifact(s) plus its manifest) |

**See it working**: a batch of orbits, one kernel, run on CPU threads — [Demo](#demo) below has the commands
and what each printed line means.

[Docs](https://amasat01.github.io/raptor/) · [the family site](https://amasat01.github.io/)

## For package authors

```python
from raptor.schema import validate_manifest

manifest = {
    "schema_version": 2,
    "pattern": "pure",
    "aether_abi": "aether-abi/2",
    "exec_targets": ["host", "device"],
    "exec_access": "sample_local",
    "plugins": [{"id": "two_body_step", "order": 0, "enabled": True, "artifact": "two_body_step.ptx",
                "sidecar": "two_body_step.json", "format": "ptx"}],
}
validate_manifest(manifest)  # raises on a malformed or out-of-order document
```

| key | meaning |
|---|---|
| `schema_version` | which rules this document follows |
| `pattern` | which kind of kernel this is; `pure` = no side effects |
| `aether_abi` | the binary-format tag paired with `schema_version` (e.g. `"aether-abi/2"` for v2) |
| `exec_targets` | host / device: where this kernel may run |
| `exec_access` | how it touches per-sample vs. cross-sample data |
| `plugins` | the compiled artifact(s) this manifest describes (`ptx` = NVIDIA's GPU assembly format) |

Any class with a matching `build(directory)` method satisfies
`ManifestProvider` — "duck typing": judged by which methods exist, checked
with `isinstance`, never by which class it inherits from:

```python
from raptor.protocols import ManifestProvider

class Bundle:
    def build(self, directory):
        return f"{directory}/manifest.json"

isinstance(Bundle(), ManifestProvider)  # True — duck-typed, no inheritance needed
```

## Install

```bash
pip install raptor-core      # Python 3.9+, no dependencies
```

The import name is `raptor`. The rest of the family installs the same way: `pip install raptor-hawk` for the CPU
route (Linux x86_64, CPython 3.9-3.14 including free-threaded 3.13t and 3.14t, a host `g++` 11 or newer; it brings `aether-dsc`, the sealed C++ headers it
compiles against, automatically), and for the GPU route
`pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"` (use `[cuda13]` on both for CUDA 13; no `nvcc` needed).

> **NVIDIA packages come only through the extras.** `raptor-hawk[cuda12]` pulls `cuda-bindings` 12,
> `nvidia-cuda-nvrtc-cu12` and `nvidia-cuda-cccl-cu12`; `raptor-hawk[cuda13]` pulls `cuda-bindings` 13,
> `nvidia-cuda-nvrtc` 13 and `nvidia-cuda-cccl` 13; `raptor-eagle[cuda12]` / `[cuda13]` pull CuPy
> (`cupy-cuda12x` / `cupy-cuda13x`, with the CUDA headers CuPy compiles against). Without an extra pip installs no NVIDIA package: you get the CPU route, or the GPU route through a CUDA setup you already have. Pick the extra matching the CUDA
> version your driver reports (`nvidia-smi`, top right).

**Platforms:** built and tested on Linux x86_64 only so far (CPython 3.9–3.14, including free-threaded 3.13t and 3.14t), on NVIDIA GPUs from Pascal (Quadro P2000) and Turing (Tesla T4). There are no wheels for macOS, Windows or ARM yet, and WSL2 is untested. `raptor-core` and `aether-dsc` are pure Python and install anywhere. Free-threaded builds (3.13t, 3.14t) currently re-enable the GIL when `hawk` or `eagle` is imported and print a RuntimeWarning; results are correct, just not parallel.

To work on the code, from a clone: `git clone https://github.com/amasat01/raptor.git`, then `pip install -e .`. Test with `pip install pytest && pytest tests -m "not
cross_repo"` (`-m "not cross_repo"` leaves out the tests that check eagle against raptor's contracts — they need
an `eagle` checkout next to this one, and fail rather than skip without it). To set up a development machine for
the whole family, see [`tools/devbox/`](https://github.com/amasat01/raptor/tree/main/tools/devbox).

## Demo

`examples/two_body_hybrid.py` is a small end-to-end example: a batch of
orbits propagated by a hand-written physics formula plus a small learned
correction, traced into one compiled kernel that this package's manifest
describes. See [Going deeper](#going-deeper-optional) below for exactly
what it computes and prints.

```bash
# hawk and eagle come from PyPI (no nvcc needed); [demo] adds numpy and torch
pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"   # or [cuda13] on both
pip install -e .[demo]                                     # from a clone of this repository
python examples/two_body_hybrid.py --host
python examples/two_body_hybrid.py --device --steps 400 --batch 65536
```

`--host` prints the following (output of `python examples/two_body_hybrid.py --host`; the wall-per-step line varies by
machine, everything above it is exact, the same seed every run):

```
target            : host (manifest exec_targets ['host'])
batch x steps     : 1024 x 200 at dt = 0.01
training loss     : 1.230e-06
(a) max |compiled - torch eager| : 1.788e-06 (tolerance 1e-04)
(b) final-state error vs truth   : corrected 6.491e-03 vs two-body only 1.930e-01
(c) wall per step (demo measurement, not a benchmark): compiled 1.306 ms, torch eager 0.612 ms
```

## Going deeper (optional)

What the demo computes, package internals, and three scripts' own
machine-local measurements — none of it needed to install or use raptor.

<details>
<summary>What's inside</summary>

- `raptor.schema` — the kernel-manifest schema, two layers:
  `schema.manifest` (the core layer every producer writes against) and
  `schema.blocks` (the `neural_block` extension; hawk never imports it).
  `schema.dtypes` carries the scalar-type tag vocabulary.
  `raptor.schema.validate_manifest(doc)` is a minimal shape validator (key
  order, top keys, schema_version, pattern vocabulary) — not a port of
  eagle's deep `validate_sidecar`. Schema versions 1 and 2 both load
  (`manifest.MAX_SCHEMA_VERSION`): v2 adds three
  flat, versioned top-level keys — `exec_targets` (`device`/`host` list),
  `exec_access` (`sample_local` / `cross_sample_read` / `cross_sample_write`
  / `mapreduce`), `exec_op` (required iff `exec_access="mapreduce"`) — and
  requires `aether_abi: "aether-abi/2"`; v1 rejects any `exec_*` key
  outright (`manifest.check_execution_axis`).
- `raptor.protocols` — the `typing.Protocol` definitions:
  `Backend` / `Executable` (program-level execution),
  `KernelLauncher` / `Marshal` (kernel-level launch/marshal, mirroring
  eagle's public surface 1:1), `ManifestProvider` (the kernel-provider
  protocol's one facet).
- `raptor.conformance` — the interop certification declaration (`UNIVERSE` /
  `CERTIFIED` / `ROADMAP` / `ROWS`) and its reusable matrix-hygiene scanners;
  the full row-by-row certification matrix is on the
  [interop protocols page](https://amasat01.github.io/raptor/content/interop_protocols.html).
- `goldens/` — the golden conformance corpus (harvested from eagle's test
  fixtures) that `raptor.schema.validate_manifest` is exercised against.
- `tests/*cross_repo*`: the eagle conformance battery
  (`test_verbatim_against_eagle.py`, `test_goldens.py`), marked `cross_repo`
  and failing rather than skipping when a sibling checkout is absent —
  deselected here only via `-m "not cross_repo"`, run for real by the local
  gate and a downstream package's own CI. `raptor.schema.manifest.SCHEMA_VERSION`
  is the version of record; `eagle.roles.SCHEMA_VERSION` derives from it
  directly, so no standalone text-scan cross-check is kept as a redundant leg.
</details>

<details>
<summary>What the demo computes</summary>

A batch of planar orbits is propagated with **RK4** (a standard 4th-order
numerical-integration formula). The textbook two-body formula is written by
hand; the piece it misses — drag — is learned by a small (4 → 32 → 2) neural
network trained in torch on CPU. hawk traces ONE kernel that performs a
whole RK4 step with the network evaluated inline, compiles it, and publishes
it with a manifest this package's schema validates; eagle loads that
manifest and runs the kernel, either on CPU threads or as a replayed
**CUDA graph** (a GPU's sequence of launches recorded once and replayed
without re-issuing each one). The script prints how closely the compiled
rollout tracks a plain torch rollout of the same arithmetic, how much closer
to the truth the corrected orbits land than uncorrected ones, and a
wall-per-step line.
</details>

<details>
<summary>early termination: samples that finish at different times (measured when the script runs — numbers vary by machine)</summary>

`examples/early_termination.py` shows the other half of this one-kernel
idea: irregular work whose samples finish at different times, with the host
out of the loop. A batch of planar orbits ends when it crosses a collision
or an escape radius, and a device-side live count guards each step with a
conditional graph node (a graph step that can skip itself at replay time),
so the whole horizon is queued as graph replays with zero host round-trips
until every sample is done — later replays then do no work at all. Measured
against the identical captured graph with the guard removed (so the guard
is the only difference), the skip is a real wall-clock saving that scales
with how much of the horizon runs past the last termination — **this
script's own measurement, on the machine that built this page: over 13x at
a million samples over 4000 steps (13.8x in the committed card).** The script also runs
the same kernel eagerly, host-side guard check and all: at a million samples the
guarded graph and this host-guarded eager loop tie (1.01x in the card; the kernel itself
dominates there); at the smallest batches (1, 32 samples) the skip does not
pay for itself — the conditional node's own fixed cost exceeds a tiny
kernel's, so the unguarded graph is the fastest arm. The eager loop also
carries one host round-trip per two-step replay against the graph's one for
the whole captured horizon, a gap the numbers above already price in. CPU
mode (`--host`) runs the same kernel with the guard evaluated eagerly, so it
needs no GPU. `--torch` (device only) times the same RK4 two-body physics
written in idiomatic torch — eager, eager with a host check, and both as a
replayed CUDA graph — against the guarded graph and prints each arm's ratio.
Run it yourself for your own machine's numbers.

```bash
python examples/early_termination.py --host
python examples/early_termination.py --device
python examples/early_termination.py --device --torch
```
</details>

<details>
<summary>zero-copy interop (measured when the script runs — numbers vary by machine)</summary>

`examples/zero_copy_interop.py` backs the third leg of the same idea:
exchanging GPU arrays with NumPy, CuPy and PyTorch without copies, and only
where the family's own certification matrix says there is none. A
CUDA-resident torch tensor is handed straight into one hawk kernel through
eagle's zero-copy `to_cupy` view, and the kernel writes its result into a
second torch CUDA buffer through the same view — no allocation, no copy,
matching the aliasing rows of [raptor's interop matrix](https://amasat01.github.io/raptor/content/interop_protocols.html)
exactly. The same tensor read from
CPU memory instead takes the honest upload the matrix declares as a copy.
The script prints each crossing's measured pointer
identity next to the row that covers it, and a behavioural proof for the
input alias: mutating the torch tensor in place and replaying the same
kernel with no rebind shows the output tracking the change. `--device` also
times that alias path against the copying round-trip a caller without this
protocol would naturally write — **this script's own measurement, on the
machine that built this page: at 256 MB, the copies cost over 100x the alias
path (113.6x in the committed card).** CPU mode (`--host`) needs neither a GPU nor torch: it runs the same
kernel on CPU threads with plain numpy in and out.

```bash
python examples/zero_copy_interop.py --host
python examples/zero_copy_interop.py --device
```
</details>

<details>
<summary>autodiff vs. torch (measured when the script runs — numbers vary by machine)</summary>

`examples/autodiff_vs_torch.py` closes the loop on the same one-kernel
idea: hawk differentiates the traced kernel itself rather than executing a
separate tape. One RK4 two-body step is traced once, and `hawk.diff.vjp`/
`jvp` derive a reverse-mode gradient kernel (`vjp`: given a cotangent — a
weighting of the output — produce the matching weighting of the input) and
a forward-mode directional-derivative kernel (`jvp`: given a tangent — a
direction in the input — produce the matching direction in the output) from
it, published in the same manifest as the primal (the original, undifferentiated
kernel). `--device` checks both against `torch.autograd.grad` and
`torch.func.jvp` on the identical physics written in torch, and against a
float64 finite-difference Jacobian (the matrix of every output's derivative
with respect to every input, approximated numerically) so the check does
not rest on torch alone; every plane (one array per sample) in and out — the
state, the cotangent, the tangent, both derivative outputs — is a torch CUDA
tensor crossing through eagle's zero-copy `to_cupy` view, read back from
torch's own storage. It also times hawk's vjp/jvp kernels against torch's,
and each side's primal alone so the derivative's own overhead is visible —
**this script's own sweep, on the machine that built this page
(`benchmarks/autodiff_vs_torch_card.md`): across a batch sweep from 1 to a
million samples, hawk's vjp measured roughly 90-550x faster per call than
`torch.autograd.grad`, and hawk's jvp roughly 60-780x faster than
`torch.func.jvp`.** At small batch those ratios are
mostly torch's own per-call overhead; at a million samples (about 90x and
60x) they are memory traffic — one fused kernel against every intermediate
written out. hawk's derivatives cost about 1.0-1.5x their own primal, where
torch's forward-plus-backward pass costs several times its primal alone.
CPU mode (`--host`) needs no GPU or cupy; its reference is the float64
finite difference alone, with torch (if installed) added as an optional CPU
cross-check.

```bash
python examples/autodiff_vs_torch.py --host
python examples/autodiff_vs_torch.py --device
python examples/autodiff_vs_torch.py --sweep
```
</details>

---
<p align="center">
  <b>hawk</b> write it ·
  <b>eagle</b> run it ·
  <b>aether</b> the numerics core ·
  <b>raptor</b> the shared contracts (this repo) —
  <a href="https://amasat01.github.io/">the RAPTOR family</a>
</p>

Apache-2.0 (see [`LICENSE`](https://github.com/amasat01/raptor/blob/main/LICENSE) and
[`NOTICE`](https://github.com/amasat01/raptor/blob/main/NOTICE)) · cite via "Cite this repository"
(`CITATION.cff`; every tagged release is archived on Zenodo: [doi:10.5281/zenodo.23250234](https://doi.org/10.5281/zenodo.23250234)) · built to
make GPU computing accessible on modest hardware, for research and education. Collaboration is the point, and a
citation is the currency — get in touch.
