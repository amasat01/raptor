# RAPTOR development machine

`setup_devbox.sh` turns a fresh Linux machine into a working development box
for the RAPTOR family: it clones aether, raptor, eagle and hawk, builds the two
pinned environments with editable installs and compiled `_core`s, runs each
repository's quick gate and writes an env script.

```bash
git clone https://github.com/amasat01/raptor.git ~/raptor/raptor
~/raptor/raptor/tools/devbox/setup_devbox.sh            # --dry-run to see every command first
. ~/raptor/.devbox/raptor_env.sh gpu                     # or cpu
```

The environments are the ones eagle's performance cards are reproduced with
(`eagle/benchmarks/reproduce/locks/`): `gpu` (CUDA PyTorch, JAX, CuPy, Warp)
and `cpu` (CPU PyTorch, JAX, Numba), each with its own GCC 14 and nvcc 12.9, so
the host needs no compiler or CUDA toolkit. The script is idempotent: re-run it
after pulling to rebuild what changed. `--cxx-gates` also builds and runs
aether's and eagle's C++ suites (CPU mode); `--profiles cpu` skips the GPU env.

## Machine requirements

- ≥ 8 cores, 64 GB RAM, 150–200 GB NVMe (two envs + build trees use ~60 GB;
  the rest is headroom for builds, caches and card runs).
- Linux x86_64 with glibc ≥ 2.28, `git` and `curl`. No root.
- GPU optional. Without one the `gpu` env is still built (for eagle's default
  architectures) and the GPU tests are skipped; with one, use a driver ≥ 525.60.
- A rented dedicated CPU server or your own workstation both work.

## Private repositories

The optional private repo is cloned only when you ask for it and your SSH agent
can reach it:

```bash
setup_devbox.sh --extra-repo <name>=git@github.com:<owner>/<name>.git
```

It is installed editable into both envs. Credentials stay in your SSH agent;
the script writes no keys, tokens or URLs with secrets to disk.

## Separation rule

This box holds the public RAPTOR family (plus the optional private repo) and
nothing else. ESA trajectory repositories (and their data) never go on it: keep
them on the machines they are licensed for.
