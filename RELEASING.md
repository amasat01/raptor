# Releasing

How a release goes out, the same way in raptor, aether, hawk and eagle.

## What a release is

Each package has one version. Every merge to main that changes code, tests, workflows or tools is a
release of that package. A pull request that only changes docs (`docs/`, any `*.md`, `LICENSE`, `NOTICE`)
needs no new version and can merge any time.

## Steps

1. **Branch.** Name it `release-X.Y.Z` in every repo that changes, the same name everywhere. The checks in
   each repo find the matching branch in the others and test against it.
2. **Bump.** Write changes under `## Unreleased` in `CHANGELOG.md` as you go. When ready, run
   `python tools/version.py bump X.Y.Z` in each repo. It rewrites every version string and the changelog
   heading. Never edit version strings by hand. `python tools/version.py check` shows any disagreement.
3. **Green pull requests.** Every check passes, including the GPU run on Kaggle for eagle. A pull request
   without a new version fails `release-gate`. The branch must also be up to date with main.
4. **Merge in this order:** raptor, aether, hawk, eagle. Use "Squash and merge".

## What happens by itself

- Merging tags the commit `vX.Y.Z` and starts the publish run.
- The run waits until the packages it needs are on PyPI (for example eagle waits for raptor-core), then
  asks the maintainer to approve the PyPI deployment. One approval per package.
- After upload it signs the files and creates the GitHub Release, with the files attached and the notes
  taken from the changelog section.

## After the last publish

Delete the `release-X.Y.Z` branches.

## Checking a download

Each file on a GitHub Release carries a signature of where it was built:

```
gh attestation verify <file> --repo amasat01/<repo>
```

PyPI also shows the provenance of every uploaded file on its download page.
