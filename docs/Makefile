# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Minimal makefile for the raptor Sphinx (MyST/Markdown) documentation.
# raptor is pure Python — there is no Doxygen/breathe step.

SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile clean livehtml strict linkcheck nbexec nbcheck nbcheck-hosted nbcheck-floor nbcheck-hosted-gpu nbtest

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR) content/api/generated

# Build with warnings promoted to errors, but keep going so a single run
# reports every warning instead of stopping at the first.
strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS) $(O)

# Check every internal + external link. Named explicitly (rather than left to
# the catch-all below) so it is documented and discoverable the same way the
# sibling repos' `make linkcheck` is.
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

# Execute every tutorial notebook IN PLACE (outputs kept) — the local,
# pre-commit gate. CI never calls this; it only renders the committed outputs.
nbexec:
	python -m raptor_docs.nb execute

# Refuse to proceed unless every notebook under the notebook roots was actually executed
# (every non-blank code cell carries an execution count).
nbcheck:
	python -m raptor_docs.nb colab_setup --check
	python -m raptor_docs.nb check

# Execute every tutorial/example notebook and fail on any error, via nbval —
# a lighter-weight standalone check than nbexec (does not rewrite outputs).
nbtest:
	python -m pytest --nbval-lax -q content/tutorials

# Watch for changes and auto-recompile (requires watchdog: pip install watchdog)
livehtml: Makefile
	watchmedo shell-command -p "*.rst;*.md;*.py" -R \
	    -c "make html" \
	    -i "$(BUILDDIR)/*" \
	    --debug-force-polling .

# Catch-all target: route unknown targets to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Run every docs notebook as a hosted (Colab) user does: wheels built from the checkouts, a pinned environment,
# the setup cell included (raptor_docs.nb hosted). Needs uv, a Python 3.13 and a C++23 compiler (CXX).
NBH_PYTHON ?= python3.13
NBH_ARGS   ?=
nbcheck-hosted:
	python -m raptor_docs.nb hosted --python $(NBH_PYTHON) --lock colab --repos raptor --raptor $(abspath ..) $(NBH_ARGS)

# The same against the oldest numpy and torch that have Python 3.13 wheels (raptor_docs/nb/hosted_env/floor-cpu.txt).
nbcheck-floor:
	python -m raptor_docs.nb hosted --python $(NBH_PYTHON) --lock floor --repos raptor --raptor $(abspath ..) $(NBH_ARGS)

# The GPU runtime's lock (CUDA 12 builds: see the header of raptor_docs/nb/hosted_env/colab-gpu.txt): eagle's CUDA plugin is built for
# NBH_GPU_ARCH, the GPU-only notebooks run too, on GPU NBH_GPU_INDEX, under the flock file NBH_GPU_LOCK if set. Needs ~5 GB free.
NBH_GPU_ARCH  ?= 61
NBH_GPU_INDEX ?= 1
NBH_GPU_LOCK  ?=
nbcheck-hosted-gpu:
	python -m raptor_docs.nb hosted --python $(NBH_PYTHON) --lock colab-gpu --gpu --cuda-arch $(NBH_GPU_ARCH) --gpu-index $(NBH_GPU_INDEX) $(if $(NBH_GPU_LOCK),--gpu-lock $(NBH_GPU_LOCK)) --repos raptor --raptor $(abspath ..) $(NBH_ARGS)
