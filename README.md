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
  <a href="https://amasat01.github.io/"><b>The RAPTOR family</b></a><br>
  <a href="https://amasat01.github.io/hawk/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_hawk_raptor_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_hawk_raptor_light.svg" alt="hawk" width="430"></picture></a>
  <a href="https://amasat01.github.io/eagle/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_eagle_raptor_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_eagle_raptor_light.svg" alt="eagle" width="430"></picture></a><br>
  <a href="https://amasat01.github.io/aether/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_aether_raptor_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_aether_raptor_light.svg" alt="aether" width="430"></picture></a>
  <a href="https://amasat01.github.io/raptor/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_raptor_raptor_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/raptor/main/docs/_static/ecosystem/ecosystem_card_raptor_raptor_light.svg" alt="raptor" width="430"></picture></a>
</p>

raptor holds the shared contracts of the [RAPTOR family](https://amasat01.github.io/): the manifest format and certification
contracts that let a kernel built once work with the code you already have. See
[where RAPTOR fits](https://amasat01.github.io/where_raptor_fits.html) for how the four pieces combine.

| **0** | **4** | **1** |
|:---:|:---:|:---:|
| hard dependencies (pure Python) | array libraries certified: NumPy, CuPy, PyTorch, Warp, JAX | JSON manifest per kernel bundle (a kernel's compiled artifact(s) plus its manifest) |

**See it working**: orbits from hand-written physics plus a small learned drag correction, trained in torch and compiled into one kernel — [Demos](#demos) points to the runnable examples.

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

## Threading

Free-threaded CPython (3.13t/3.14t) is supported: the compiled modules declare GIL-free operation and the GIL stays disabled after import. Any number of threads may call module-level functions, build, compile, plan and cache concurrently. Distinct objects may be used from distinct threads without synchronisation. One stateful object (a stream, capture, launcher, graph, composer, plan, pipeline, active set, host kernel or arg block) shared by several threads is memory-safe — its calls serialise and a consumed object raises — but the ORDER of those calls is the caller's responsibility, exactly as for a NumPy array or a CuPy stream. CUDA adds two rules of its own: a stream capture is begun, filled and ended by one thread, and while any capture is open no thread may synchronise the whole device (stream-level synchronisation is fine). GPU routes are supported on free-threaded 3.14; on 3.13t the CPU route runs.

## Install

```bash
pip install raptor-core      # Python 3.9+, no dependencies
```

The import name is `raptor`. The rest of the family (hawk, eagle) and the GPU extras: [installation](https://amasat01.github.io/raptor/content/installation.html).

To work on the code, from a clone: `git clone https://github.com/amasat01/raptor.git`, then `pip install -e .`. Test with `pip install pytest && pytest tests -m "not
cross_repo"` (`-m "not cross_repo"` leaves out the tests that check eagle against raptor's contracts — they need
an `eagle` checkout next to this one, and fail rather than skip without it). To set up a development machine for
the whole family, see [`tools/devbox/`](https://github.com/amasat01/raptor/tree/main/tools/devbox).

## Demos

The runnable demos live with the packages they show off:

- [hawk](https://amasat01.github.io/hawk/content/examples.html): the hybrid orbit (analytic physics plus a learned correction in one compiled kernel) and autodiff against torch.
- [eagle](https://amasat01.github.io/eagle/content/userguide/howto.html): early termination with a conditional graph node, and zero-copy interop with NumPy, CuPy and PyTorch.

The measured headline, from eagle's committed performance cards:

<!-- hero:begin -->
hawk + eagle (`eagle.simulate`) integrates 1,000,000 RK4 oscillators, each stopping at its own step, in 613 µs on an NVIDIA H100 80GB HBM3 (199 µs for 100,000), 2.5×–600× faster than NVIDIA Warp (2.5×), JAX (76×), CuPy (600×), PyTorch with torch.compile (130×) at N = 1,000,000 (1.1×–1700× at N = 100,000), in 40 MiB of GPU memory
<!-- hero:end -->

## Going deeper (optional)

Package internals, none of it needed to install or use raptor.

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
  is the cross-repo pin (1; the highest accepted version is `MAX_SCHEMA_VERSION`, 2); `eagle.roles.SCHEMA_VERSION` derives from it
  directly, so no standalone text-scan cross-check is kept as a redundant leg.
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
