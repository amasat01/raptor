# Contributing

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

Docs (`cd docs`; the packages are listed in `.github/workflows/docs.yml`):

```bash
make strict     # zero warnings
make linkcheck
make nbexec     # run the tutorial notebooks in place; outputs are committed
make nbcheck    # fail on an unexecuted notebook
```

`make nbcheck-hosted` runs the notebooks the way a Colab user does; read the comments above the target in `docs/Makefile` for what it
needs.

## What a PR here must show

- `dependencies = []` in `pyproject.toml` stays empty. A new import is function-local and optional, or it does not land.
- A new framework or crossing is a certification row, declared here first and registered by the owning repo second:
  [Contributing a certification row](https://amasat01.github.io/raptor/content/devguide/contributing_a_certification_row.html).
- `tests/test_packaging.py` and `tests/test_docs_interop_protocols_sync.py` stay green: they keep package metadata and the published
  matrix in step with the code.
- The family checklist from [How we build](https://amasat01.github.io/raptor/content/principles.html), nothing more.

## Release

Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, tag `vX.Y.Z`. `.github/workflows/publish.yml` does the rest; use
`workflow_dispatch` to dry-run. Publish order is in the family guide.
