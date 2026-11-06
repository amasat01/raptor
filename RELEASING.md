# Releasing

How a release goes out, the same way in raptor, aether, hawk and eagle.

## What a release is

Each package has one version: the top heading of its `CHANGELOG.md` (`## X.Y.Z (date)`; `## Unreleased` is a dev build).
Every pull request to main also carries one label, which only has to agree with the heading:

- `release:patch`, `release:minor`, `release:major`: the heading is the next version from the latest tag. On a release candidate, `patch` is the candidate's own release.
- `release:rc`: the heading is the next release candidate (`X.Y.ZrcN`).
- `release:none`: docs only (`docs/`, any `*.md`, `LICENSE`, `NOTICE`), no release heading; nothing is released. Any other diff needs a bump label, and a docs-only diff needs `none`.

The `release-gate` workflow (the only one that reacts to labels) refuses a missing or inconsistent label or heading, an empty pull request body and a version whose tag exists.
`python tools/next_version.py` prints the heading's version; `python tools/next_version.py patch` the version a label gives.

## Steps

0. **Upstream review.** Read the open "Upstream watch" issue and the changelogs it links. Each new feature worth adopting becomes a tracker item, or a "not now" line in the issue.
1. **One commit per repo.** Branch `release-X.Y.Z` in every repo that changes, the same name everywhere. The commit heads
   `CHANGELOG.md` with `## X.Y.Z (date)` and moves the dependency pins (`>=X.Y.Z`) to the new versions.
2. **One round of pull requests across the repos.** The checks in each repo find the matching branch in the others and build
   it at the version its heading gives, so a `>=X.Y.Z` pin resolves against the sibling's branch in the same round.
   Pins on released versions need no sibling build. Every check passes, including the GPU run on Kaggle for eagle, and the branch is up to date with main.
3. **Label and notes.** Add the release label to each pull request and write its body: the body is the GitHub Release notes.
4. **Merge in this order:** raptor, aether, hawk, eagle ("Squash and merge"), with no further checks between the merges.
   Merging tags the commit and starts publishing, which waits for the packages it needs.

## What happens by itself

- Merging tags the commit `vX.Y.Z` and starts the publish run.
- The run waits until the packages it needs are on PyPI (for example eagle waits for raptor-core), then
  asks the maintainer to approve the PyPI deployment. One approval per package.
- After upload it signs the files and creates the GitHub Release, with the files attached and the notes
  taken from the pull request body.

## After the last publish

Delete the `release-X.Y.Z` branches.

## Checking a download

Each file on a GitHub Release carries a signature of where it was built:

```
gh attestation verify <file> --repo amasat01/<repo>
```

PyPI also shows the provenance of every uploaded file on its download page.
