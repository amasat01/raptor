# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# The stateless gate: `make gate` (what CI runs on the CPU), `make gate-gpu` (the cross_repo conformance battery
# against the eagle and hawk sources: no GPU is needed, the name is the family's), `make gate-all` (both).
# CI calls these targets. Everything is built under $(GATE_DIR) (gitignored) in a fresh venv made by $(PYTHON);
# GATE_REUSE=1 keeps the venv for the inner loop. Siblings are inputs, never guessed:
#   EAGLE_SRC  gate-docs (README numbers) and gate-gpu       HAWK_SRC  gate-gpu
#   AETHER_SRC gate-gpu (eagle's host core is built against it, as eagle's CI does; needs CC and CXX, a C++23 compiler)
#   CUDA_PATH  optional: the CUDA toolkit aether-dsc's plugin is built with when nvcc is on PATH (else the plugin is skipped by CMake)
#   SITE_SRC   a checkout of amasat01.github.io, gate-docs (docs tools and capability data)
# Each leg is a target: `make gate-<leg>`; PYTEST_ARGS narrows the pytest legs, e.g.
#   GATE_REUSE=1 make gate-test PYTEST_ARGS="tests/test_flow.py"

GATE_DIR    ?= build/gate
PYTHON      ?= python3
GATE_REUSE  ?=
PYTEST_ARGS ?=
EAGLE_SRC   ?=
HAWK_SRC    ?=
AETHER_SRC  ?=
CUDA_PATH   ?=
GATE_JOBS   ?= 2
SITE_SRC    ?=

gate_venv = $(GATE_DIR)/venv
gate_py   = $(CURDIR)/$(gate_venv)/bin/python
gate_env  = export PATH="$(CURDIR)/$(gate_venv)/bin:$$PATH"

.PHONY: gate gate-gpu gate-all gate-venv gate-lint gate-names gate-test gate-fresh gate-docs
gate: gate-lint gate-names gate-test gate-docs
gate-all: gate gate-gpu

# The throwaway environment: this checkout with its [test] extra, ruff pinned in ci/constraints.txt
gate-venv:
	if [ -z "$(GATE_REUSE)" ] || [ ! -x "$(gate_py)" ]; then
	  rm -rf $(gate_venv)
	  mkdir -p $(GATE_DIR)
	  $(PYTHON) -m venv $(gate_venv)
	  $(gate_py) -m pip install --quiet --disable-pip-version-check -c ci/constraints.txt -e ".[test]" ruff
	fi

gate-lint: gate-venv
	$(gate_env)
	ruff check
	MANIFESTO_SOURCE=docs/content/principles.md bash tools/manifesto_check.sh CONTRIBUTING.md

# Every public name is referenced by a test or on the shrink-only allowlist
gate-names:
	$(PYTHON) -m raptor.conformance.public_names --src . --tests tests --allow ci/public_names_untested.txt --walk raptor

# The suite as CI runs it: the cross_repo battery is gate-gpu's
gate-test: gate-venv
	$(gate_py) -m pytest -q -rs -p no:cacheprovider -m "not cross_repo" $(PYTEST_ARGS)

# The generated numbers and the capability map match the committed cards
gate-fresh: gate-venv
	[ -n "$(SITE_SRC)" ] && [ -n "$(EAGLE_SRC)" ] || { echo "gate: set SITE_SRC (amasat01.github.io) and EAGLE_SRC (README numbers)"; exit 1; }
	$(gate_env)
	export RAPTOR_EAGLE_ROOT="$(abspath $(EAGLE_SRC))"
	$(gate_py) -m pip install --quiet --disable-pip-version-check "$(abspath $(SITE_SRC))"
	$(gate_py) -m raptor_docs.fresh --check --site "$(abspath $(SITE_SRC))"

# Strict docs: notebooks checked, Sphinx -W, every public name has a page
gate-docs: gate-fresh
	$(gate_env)
	grep -v '^raptor-docs' docs/requirements.txt > $(GATE_DIR)/docs-requirements.txt
	$(gate_py) -m pip install --quiet --disable-pip-version-check -r $(GATE_DIR)/docs-requirements.txt
	$(MAKE) -C docs nbcheck
	$(MAKE) -C docs strict
	$(gate_py) tools/docs_inventory_check.py docs/_build/html/objects.inv raptor

# The conformance battery that reads the eagle and hawk sources; a missing sibling is a hard failure.
# Eagle's host core is built from EAGLE_SRC against AETHER_SRC (eagle's own recipe) and installed into the gate venv.
gate-gpu: gate-venv
	[ -n "$(EAGLE_SRC)" ] && [ -n "$(HAWK_SRC)" ] && [ -n "$(AETHER_SRC)" ] || { echo "gate: set EAGLE_SRC, HAWK_SRC and AETHER_SRC"; exit 1; }
	[ -n "$${CC:-}" ] && [ -n "$${CXX:-}" ] || { echo "gate: set CC and CXX to a C++23 compiler"; exit 1; }
	$(gate_env)
	export CMAKE_BUILD_PARALLEL_LEVEL=$(GATE_JOBS)
	if ! $(gate_py) -c "import eagle._core" 2>/dev/null; then
	  # a fresh build starts from empty trees: a wheel or a cache of an earlier run never leaks into this one
	  rm -rf $(GATE_DIR)/aether-build $(GATE_DIR)/eagle-hdr $(GATE_DIR)/prefix $(GATE_DIR)/eagle-wheel
	  prefix="$(CURDIR)/$(GATE_DIR)/prefix"
	  cmake -DAETHER_CPP_MODE=ON -DCMAKE_INSTALL_PREFIX="$$prefix" -B $(GATE_DIR)/aether-build -S "$(abspath $(AETHER_SRC))"
	  cmake --install $(GATE_DIR)/aether-build
	  cmake -B $(GATE_DIR)/eagle-hdr -S "$(abspath $(EAGLE_SRC))" -DEAGLE_CPP_MODE=ON -DCMAKE_PREFIX_PATH="$$prefix" -DCMAKE_INSTALL_PREFIX="$$prefix"
	  cmake --install $(GATE_DIR)/eagle-hdr
	  $(gate_py) -m pip install --quiet --disable-pip-version-check numpy build scikit-build-core setuptools-scm nanobind
	  $(if $(CUDA_PATH),AETHER_CUDA_ROOT="$(CUDA_PATH)") $(gate_py) -m pip install --quiet --disable-pip-version-check "$(abspath $(AETHER_SRC))/dsc"
	  CMAKE_PREFIX_PATH="$$prefix" $(gate_py) -m build --wheel --no-isolation --outdir $(GATE_DIR)/eagle-wheel \
	    -Ccmake.define.EAGLE_PYTHON_CUDA_PLUGIN=OFF -Ccmake.define.EAGLE_PYTHON_MPI=OFF "$(abspath $(EAGLE_SRC))/python"
	  $(gate_py) -m pip install --quiet --disable-pip-version-check --no-deps $(GATE_DIR)/eagle-wheel/*.whl
	fi
	env -u PYTHONPATH RAPTOR_EAGLE_ROOT="$(abspath $(EAGLE_SRC))" RAPTOR_HAWK_ROOT="$(abspath $(HAWK_SRC))" \
	  $(gate_py) -m pytest -q -rs -p no:cacheprovider -m cross_repo $(PYTEST_ARGS)

# Special target that causes whole recipe to be run in one shell, failing fast
.ONESHELL:
.SHELLFLAGS := -ec
