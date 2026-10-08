# Contributing

Contributions — issues and pull requests — are welcome on the
[GitHub repository](https://github.com/amasat01/raptor), under the terms
below. See the top-level `CONTRIBUTING.md` for the full statement.

## Terms

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (`git commit -s`); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

## Running the tests

```bash
pip install -e . pytest
pytest tests -m "not cross_repo"
```

`cross_repo` selects the eagle conformance battery, which needs an `eagle`
checkout next to this one and fails rather than skips without it — leave it
deselected unless you have that checkout.

## Building these docs

```bash
. /path/to/your/raptor-docs-env.sh   # sphinx, myst-nb, sphinx-book-theme,
                                      # sphinx-design, sphinx-copybutton,
                                      # sphinx-autodoc-typehints
cd docs
make html       # build
make strict     # build with -W --keep-going (0 warnings required)
make nbexec     # execute every tutorial/example notebook in place
make nbcheck    # fail unless every notebook is fully executed
make linkcheck  # internal + external link check
```

Notebooks under `content/tutorials/` and `content/examples/` are executed
locally and committed **with** their outputs — `make nbexec` fills them in,
`make nbcheck` is the gate that refuses an unfilled one. CI never
re-executes them; it only renders the committed outputs.

## Gates

`pytest tests -m "not cross_repo"` is the suite gate; `tests/test_packaging.py`
and `tests/test_docs_interop_protocols_sync.py` keep the package metadata and
the published interop matrix in sync with the code that declares them. A
docstring change that touches `tests/` or any public API surface should be
run through the affected test files before it lands.
