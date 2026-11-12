# raptor: the maintainer's entry points. Paths are relative to this repository; nothing here reaches outside it.
#   make build    build the module into ./dist and unpack it into raptor/ (never `maturin develop`)
#   make test     the whole Python suite
#   make guards   the fast guards only (hardware-free, one-home, python-floor, node construction, manifest tables, kinds diff)
#   make gate     build, then guards, the suite and `cargo test`, every step run, the status that of the worst
#   make bench    the construction cost of Op, Const and At nodes
#   JOBS=2 PYTHON=path/to/python MATURIN=path/to/maturin override the defaults.
PYTHON ?= python3
MATURIN ?= maturin
JOBS ?= 2
GUARDS = tests/test_hardware_free.py tests/test_one_home.py tests/test_python_floor.py tests/test_node_construction.py tests/test_table.py tests/test_kinds_diff.py
PYTEST = PYTHONDONTWRITEBYTECODE=1 nice -n 10 $(PYTHON) -m pytest -q -p no:cacheprovider

.PHONY: build test guards cargo-test gate bench
build:
	CARGO_BUILD_JOBS=$(JOBS) nice -n 10 $(MATURIN) build --release -o dist
	$(PYTHON) tools/unpack_module.py dist raptor
	@if [ "$$(uname)" = Darwin ]; then codesign -v raptor/_core.abi3.so; fi

test:
	$(PYTEST) tests

guards:
	$(PYTEST) $(GUARDS)

cargo-test:
	CARGO_BUILD_JOBS=$(JOBS) nice -n 10 cargo test --workspace

gate: build
	@rc=0; $(MAKE) --no-print-directory guards || rc=1; $(MAKE) --no-print-directory test || rc=1; $(MAKE) --no-print-directory cargo-test || rc=1; exit $$rc

bench:
	$(PYTHON) tools/bench_construct.py
