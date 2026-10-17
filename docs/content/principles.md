# How we build

Fast, portable, honest, small. These twelve rules are why aether, raptor, eagle and hawk look the way they do. Every pull request in
the family is read against them.

## Manifesto

Bloat is future technical debt.<br>
Excessive abstraction hides reality.<br>
Focus and clear boundaries promote flexibility and seamless integration.<br>
Efficiency safeguards portability and scale.<br>
Performance unlocks capabilities.<br>
Rigor earns trust.<br>
Simplicity enables discovery.

## The principles

Each rule has the rule, the reason, and what a pull request must show.

### 1. Performance first, measured honestly

The fast path is the design; ergonomics grow from it. Slow code cannot be rescued by a nice API.
*A PR shows:* numbers from `-O3` builds, paired arms in one run, one variable changed at a time, raw artifacts kept. A number
appears in prose only by citing the card the run wrote.

### 2. CPU and GPU, one API

Users move between laptop and GPU without rewriting. Every feature ships on the host and the device with the same interface.
*A PR shows:* the CPU twin and the GPU path, run by the same test. No GPU-only feature lands without its CPU twin.

### 3. Thread-safe by design

Free-threaded Python runs with the GIL off, and so must we. *A PR shows:* a thread-safety test for the new feature. We never turn
the GIL back on to hide a race.

### 4. Portable

There is no single target GPU or compiler. *A PR shows:* every supported architecture still builds, with zero warnings (warnings are
errors in CI). A compiler bug is worked around in our code and pinned by a test that fails on the affected compiler.

### 5. Contracts you can check

Results agree within the math libraries' rounding guarantees, derived per operation, not bit for bit across different code paths.
The same compiled path stays bit-exact. *A PR shows:* tolerances derived, never copied from another test, and a test shown to fail
before its pass counts.

### 6. Honest limits

Users trust a tool that says what it cannot do. Our own limits live in the limits registry with their cause, the way out and the
release that lifts them. Another framework's limit is stated as theirs, with the way out, and never chased. *A PR shows:* every
skip naming its reason, and nothing that degrades silently. A weaker composition is named.

### 7. Fit the user's workflow

People already have JAX, PyTorch, CuPy/NumPy and Warp programs. We run inside them and they run inside our batches, zero-copy, both
directions. *A PR shows:* docs that teach how we fit, never the framework itself.

### 8. No bloat, ever

Small things stay fast, readable and maintainable. Reuse before adding; add nothing speculative; delete what your change makes dead;
keep one source per fact. *A PR shows:* the size it adds (lines, binary, dependencies).

### 9. Layers stay clean

A change lives in the lowest layer that owns it.

- **raptor** is the dependency-free protocol spine.
- **aether** is the header-only C++23 expression library.
- **eagle** launches and orchestrates: C++ first, then bind to Python. The plugin seam is versioned.
- **hawk** turns the user's Python into kernels whose code mirrors that Python.

*A PR shows:* the layer it touches and why no lower one owns the change.

### 10. Readable and simple

Code reads like the code around it. Names say what a thing is. A trailing underscore means private or protected and nothing
else. Comments say why, never what or when. *A PR shows:* nothing a reader must decode.

### 11. Tests are a record

The gate checks test names against a committed manifest, never a count, so a swapped test cannot hide. *A PR shows:* the manifest
re-minted in the same change when a test is added or removed. Each test name reads as a sentence about behaviour.

### 12. History is clean

`main` changes only through a reviewed pull request, squash-merged, with one plain commit line and no trailers. Commits in a
contributor's pull request carry the sign-off (`git commit -s`) the terms below ask for. Work that spans repositories uses the same branch name
in each; CI takes the companion repository at that branch if it exists, and at `main` otherwise.

## Style

The rules are the same everywhere; each repo's config holds the values.

- **Python:** `ruff` as set in the repo's `pyproject.toml`; every public name has a docstring (the docs build fails on a broken one).
- **C++/CUDA:** C++23; the repo's `.clang-format`; `camelCase` functions, `PascalCase` types; every public symbol has a Doxygen
  `@brief`. Guard CUDA-only code so the CPU build keeps the same API.
- **Docs:** the reader has a working program and decides in ten seconds. Open with what they get; show a small example early; use
  plain words and short sentences; say what another tool is right for, never that it falls short. Limits carry their marker (see rule 6).
  Every docs build is warning-free.
- **Tests:** named as sentences; one behaviour each; the CPU leg always runs; a skip names its reason.
- **Commits:** one plain line in the imperative, no trailers, one change per commit.

## Releases

A release is a tag `v<version>` that matches the version in the package; a mismatch fails CI before anything is published.
`raptor-core` and `aether-dsc` have no family dependencies and publish first, in either order. `raptor-eagle` (needs `raptor-core`)
and `raptor-hawk` (needs `aether-dsc`) follow. Each repo's `CONTRIBUTING.md` names the file to bump.

## Terms

Contributions are accepted under the Apache License 2.0, the license the code ships under. There is no Contributor License
Agreement. Every commit in a contributor's pull request carries a Developer Certificate of Origin sign-off (`git commit -s`); see
<https://developercertificate.org/>.
