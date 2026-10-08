# Examples

Four standalone notebooks, each one task, built from the scripts in
[`examples/`](https://github.com/amasat01/raptor/tree/main/examples). Unlike
the tutorials, these need [hawk](https://github.com/amasat01/hawk) (kernel
authoring) and [eagle](https://github.com/amasat01/eagle) (execution)
installed from PyPI (see [Installation](/content/installation)), plus `pip install "raptor-core[demo]"` for
`numpy`/`torch`. Every one of them also runs as a plain script from a
checkout — see each notebook's first cell for the equivalent command line.

- [Hybrid orbit: analytic physics + a learned correction](/content/examples/two_body_hybrid)
  — one traced kernel performs a whole RK4 step with gravity written by hand
  and drag learned by a small torch-trained network, compiled once and run
  on CPU threads or as a replayed CUDA graph. The same compile-once,
  train-with-`loss.backward()` pattern is covered end to end in eagle's
  [Train through your kernel with PyTorch](https://amasat01.github.io/eagle/content/userguide/tutorials/03_torch_training.html)
  and hawk's
  [capstone](https://amasat01.github.io/hawk/content/vocabulary/04_capstone.html).
- [Early termination: a batch that finishes at different times](/content/examples/early_termination)
  — a device-side live count and a conditional graph node let a captured
  CUDA graph skip replays once every sample in the batch has stopped. For
  the general trade-off — when thinning a batch like this is worth the
  overhead — see eagle's
  [Finished samples: compaction, reorder, and when not to](https://amasat01.github.io/eagle/content/userguide/tutorials/06_compaction_and_reorder.html).
- [Zero-copy interop](/content/examples/zero_copy_interop) — the
  numpy/cupy/torch crossings the certification matrix declares, exercised
  for real: a pointer-stable host round-trip and (where a CUDA-capable torch
  build is available) a zero-copy device alias.
- [Autodiff against torch](/content/examples/autodiff_vs_torch) — hawk
  differentiates the traced kernel itself (reverse- and forward-mode) rather
  than executing a separate tape, checked against `torch.autograd` and a
  float64 finite-difference Jacobian.

```{toctree}
:maxdepth: 1
:hidden:

examples/two_body_hybrid
examples/early_termination
examples/zero_copy_interop
examples/autodiff_vs_torch
```
