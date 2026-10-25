# Releasing

How a release goes out, the same way in raptor, aether, hawk and eagle.

## What a release is

Each package has one version: its latest `vX.Y.Z` git tag. Every pull request to main carries one label,
which says what the merge releases:

- `release:patch`, `release:minor`, `release:major`: the next version. On a release candidate, `patch` is the candidate's own release.
- `release:rc`: the next release candidate (`X.Y.ZrcN`).
- `release:none`: docs only (`docs/`, any `*.md`, `LICENSE`, `NOTICE`); nothing is released. Any other diff needs a bump label, and a docs-only diff needs `none`.

`release-gate` refuses a missing or inconsistent label, an empty pull request body and a version whose tag exists.
`python tools/next_version.py patch` prints the version a label gives.

## Steps

0. **Upstream review.** Read the open "Upstream watch" issue and the changelogs it links. Each new feature worth adopting becomes a tracker item, or a "not now" line in the issue.
1. **Branch.** Name it `release-X.Y.Z` in every repo that changes, the same name everywhere. The checks in
   each repo find the matching branch in the others and test against it (and, when that branch's pull request
   has a release label, against the version it will release).
2. **Label and notes.** Add the release label to each pull request and write its body: the body is the
   GitHub Release notes. No version is edited anywhere.
3. **Green pull requests.** Every check passes, including the GPU run on Kaggle for eagle. The branch must
   be up to date with main. Relabelling after the last build needs an empty commit.
4. **Merge in this order:** raptor, aether, hawk, eagle. Use "Squash and merge".

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
