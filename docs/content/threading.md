# Threading

This page states how the RAPTOR family behaves when several threads use it,
and how that is checked.

## The contract

Free-threaded CPython (3.13t/3.14t) is supported: the compiled modules declare GIL-free operation and the GIL stays disabled after import. Any number of threads may call module-level functions, build, compile, plan and cache concurrently. Distinct objects may be used from distinct threads without synchronisation. One stateful object (a stream, capture, launcher, graph, composer, plan, pipeline, active set, host kernel or arg block) shared by several threads is memory-safe — its calls serialise and a consumed object raises — but the ORDER of those calls is the caller's responsibility, exactly as for a NumPy array or a CuPy stream. CUDA adds two rules of its own: a stream capture is begun, filled and ended by one thread, and while any capture is open no thread may synchronise the whole device (stream-level synchronisation is fine). GPU routes are supported on free-threaded 3.14; on 3.13t the CPU route runs.

## How it is certified

raptor itself is pure Python and has nothing to declare: a pure-Python wheel
never re-enables the GIL, and its only module-level state is read-only after
import. What raptor does provide is the harness the other packages test their
claim with, in `raptor.conformance.freethreading` (standard library only):

| piece | role |
|---|---|
| `require_free_threaded()` | skips on a GIL build; **fails** if `-X gil=...` or `PYTHON_GIL` is set, since forcing the GIL state hides a missing declaration |
| `hammer(fn, threads, iterations)` | runs `fn` on N threads released together by a barrier |
| `assert_gil_free(module)` | row FT-1: imports the module in a fresh process under `-W error::RuntimeWarning` and requires the GIL to be off afterwards |
| `canary_python()`, `measure_loss()`, `require_not_vacuous()` | row FT-2: a deliberately racy counter must lose at least 10% of 8 threads x 200,000 updates, otherwise the run is red as vacuous rather than green |

Rows FT-3 to FT-6 (global state, distinct objects, shared-object misuse, GPU)
are written by each package on top of `hammer` and registered with
`declare_ft_row` / `register_ft_row`. Run them with `pytest -m ft` on a
free-threaded interpreter, in default GIL mode: never set `-X gil=0` or
`PYTHON_GIL=0`.
