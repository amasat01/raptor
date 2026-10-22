# Contributing

The manifesto behind [How we build](https://amasat01.github.io/raptor/content/principles.html):

<!-- manifesto:begin -->
Bloat is future technical debt.<br>
Excessive abstraction hides reality.<br>
Focus and clear boundaries promote flexibility and seamless integration.<br>
Efficiency safeguards portability and scale.<br>
Performance unlocks capabilities.<br>
Rigor earns trust.<br>
Simplicity enables discovery.
<!-- manifesto:end -->

Read [How we build](https://amasat01.github.io/raptor/content/principles.html) first: the twelve rules every pull request in the
family is read against, plus the style, release and license terms.

**This repo:** the dependency-free protocol spine. The manifest schema, the engine and provider contracts, and the interop
certification matrix. Every other repo declares raptor; raptor declares nothing.

## Build and run the gate

```bash
pip install -e . pytest
python -m pytest -q -rs -m "not cross_repo"      # the suite gate (.github/workflows/ci.yml)
```

`cross_repo` selects the eagle conformance battery. It needs an `eagle` checkout next to this one and fails, not skips, without it.
The free-threading rows run with `python -m pytest -m ft -W error::RuntimeWarning` on a free-threaded build.

Docs (`cd docs`; install `pip install -r requirements.txt`, or `pip install ./site` from a checkout of the landing site for the shared `raptor-docs` tooling):

```bash
make strict     # zero warnings
make linkcheck
make nbexec     # run the tutorial notebooks in place; outputs are committed
make nbcheck    # fail on an unexecuted notebook
python -m raptor_docs.fresh --check --site ../site   # generated numbers and the capability map are current; --write re-mints
```

`make nbcheck-hosted` runs the notebooks the way a Colab user does; the rules are in eagle's
[Documentation guide](https://amasat01.github.io/eagle/content/devguide/docs_guide.html).

## Where things live

- `raptor/schema/`: the manifest contract (`manifest.py`, `blocks.py`, `dtypes.py`); `validate_manifest` is the entry point.
- `raptor/protocols/`: `Backend`, `KernelLauncher`, `ManifestProvider` as structural protocols.
- `raptor/conformance/`: the row declaration (`interop.py`) and the pytest plugin that records which rows ran.
- `tests/`: the suite; `goldens/`: manifests every golden test validates. `docs/content/`: the pages.
- A new interop row: [Contributing a certification row](https://amasat01.github.io/raptor/content/devguide/contributing_a_certification_row.html).
  What each contract pins: [Concepts, for contributors](https://amasat01.github.io/raptor/content/concepts.html).

## What a PR here must show

- `dependencies = []` in `pyproject.toml` stays empty. A new import is function-local and optional, or it does not land.
- A new framework or crossing is a certification row, declared here first and registered by the owning repo second:
  [Contributing a certification row](https://amasat01.github.io/raptor/content/devguide/contributing_a_certification_row.html).
- `tests/test_packaging.py` and `tests/test_docs_interop_protocols_sync.py` stay green: they keep package metadata and the published
  matrix in step with the code.
- The family checklist from [How we build](https://amasat01.github.io/raptor/content/principles.html), nothing more.

## Release

Record changes under `## Unreleased` in `CHANGELOG.md`; release with `python tools/version.py bump X.Y.Z` (never edit version strings by hand); the whole flow is in [RELEASING.md](https://github.com/amasat01/raptor/blob/main/RELEASING.md).
