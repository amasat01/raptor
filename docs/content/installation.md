# Installation

raptor is pure Python with **zero hard dependencies** — nothing CUDA-specific,
no GPU required to install or import it.

## Install

```bash
pip install raptor-core
```

## Building from source

To work on the code, install from a clone:

```bash
git clone https://github.com/amasat01/raptor.git
cd raptor
pip install -e .
```

For building every repo from source and the supported-version table, see [build from source](https://amasat01.github.io/build_from_source.html).

## Requirements

- Python 3.9 or newer
- No GPU, no CUDA toolkit, no C or C++ compiler.

## Verify

```bash
python -c "import raptor; print(raptor.__version__)"
```

## Naming

The package's distribution name is `raptor-core`
— `raptor` on PyPI is an unrelated package (a Python 2-era deploy tool), so
the family's distribution names carry the `raptor-` prefix to avoid that
collision. The *import* name stays flat and unchanged regardless:
`import raptor` always works once the package is installed.

## Running the family together

raptor alone gives you the schema, the protocol contracts and the conformance
declarations — enough to validate manifests and check implementations against
the family's contracts. To compile and run a kernel end to end you also need
a kernel producer and an executor: hawk and eagle. Both are on PyPI.

```bash
pip install raptor-core
pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"   # or [cuda13] on both
```

hawk ([kernel authoring](https://github.com/amasat01/hawk)) pulls `aether-dsc`, the
sealed C++ headers it compiles against, automatically, and needs a host `g++` 11 or
newer. eagle ([execution](https://github.com/amasat01/eagle)) runs the kernels on the
GPU. No `nvcc` or CUDA toolkit is needed. hawk and eagle need Python 3.10 or newer
(hawk: Linux x86_64, CPython 3.9-3.14 including free-threaded 3.13t and 3.14t).

```{admonition} NVIDIA packages come only through the extras
:class: important
`raptor-hawk[cuda12]` pulls `cuda-bindings` 12, `nvidia-cuda-nvrtc-cu12` and
`nvidia-cuda-cccl-cu12`; `raptor-hawk[cuda13]` pulls `cuda-bindings` 13,
`nvidia-cuda-nvrtc` 13 and `nvidia-cuda-cccl` 13 (CUDA 13's wheels have no `-cu13`
suffix); `raptor-eagle[cuda12]` / `[cuda13]` pull CuPy (`cupy-cuda12x` /
`cupy-cuda13x`, with the CUDA headers CuPy compiles against). Without an extra pip installs no NVIDIA package: you get the CPU route, or the GPU route through a CUDA setup you already have. Pick the extra
matching the CUDA version your driver reports (`nvidia-smi`, top right).
```

**Platforms:** the wheels are built for Linux x86_64 (CPython versions as listed in `pyproject.toml`, including the free-threaded builds). Measured on the Quadro P2000, T4, A100, L4, H100 and RTX PRO 6000 Blackwell (sm_120), plus x86-64 and aarch64 CPUs ([performance page](https://amasat01.github.io/eagle/content/performance.html)); sm_100, sm_103 and sm_121 are shipped but not yet run on hardware. There are no wheels for macOS, Windows or ARM yet, and WSL2 is untested. `raptor-core` and `aether-dsc` are pure Python and install anywhere. On free-threaded 3.13t and 3.14t, `hawk` and `eagle` run GIL-free.

See the examples in [hawk](https://amasat01.github.io/hawk/content/examples.html) and [eagle](https://amasat01.github.io/eagle/content/userguide/howto.html) for what the pieces look like together.
