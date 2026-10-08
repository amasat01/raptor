# Contributing

Contributions — issues and pull requests — are welcome on GitHub, under the
terms below.

## Terms

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (`git commit -s`); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

## Using the code

The code is available under the Apache License 2.0 (see `LICENSE`).
Contributions are accepted under the same terms (inbound = outbound).

## Releasing

`raptor-core` publishes to PyPI through `.github/workflows/publish.yml`,
triggered by pushing a tag `v<version>` that matches `pyproject.toml`'s
`project.version` (a mismatch fails the `check` job before anything is
published). `workflow_dispatch` runs the same build, check and test-wheel
steps without publishing — use it to dry-run a release.

1. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The workflow builds the sdist + wheel, runs `twine check --strict`, runs
   the isolated-venv portability gate against the built wheel on every
   supported CPython, then publishes via a PyPI Trusted Publisher
   (environment `pypi`; no token in this repository).

**Publish order across the family:** `raptor-core` and `aether-dsc` (in the
`aether` repo) have no family dependencies and publish first, in either
order. `raptor-hawk` (depends on `aether-dsc`) and `raptor-eagle` (depends
on `raptor-core`) publish after — they may publish in either order relative
to each other.
