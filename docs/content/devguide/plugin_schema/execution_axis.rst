.. _execution-axis-v2:

The v2 execution axis (aether-abi/2) and eagle's Python plan/exec API
=======================================================================

.. contents:: On this page
   :local:
   :depth: 2

Schema version 2 adds one new axis to the manifest: which **execution
structure(s)** — a single device kernel launch, an OpenMP team of CPU
threads, an MPI rank partition, and (planned) an NCCL device group —
may drive a plugin body, and how that body is allowed to touch data outside
its own sample. A schema-v2 manifest tags itself ``"schema_version": 2`` and
``"aether_abi": "aether-abi/2"``, and MUST additionally carry three new
top-level, flat keys (absence is a load refusal, never a silent fallback to
v1 behaviour):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``exec_targets``
     - A non-empty list drawn from ``["device", "host"]`` (no duplicates) —
       which target(s) the plugin ships an entry point for.
   * - ``exec_access``
     - How the body touches data: one of ``sample_local`` (reads/writes only
       its own sample — an ordinary elementwise kernel), ``cross_sample_read``
       (reads another sample's data, e.g. a lookup table), ``cross_sample_write``
       (writes into a location that is not its own sample's column — an
       accumulate/scatter), or ``mapreduce`` (produces a per-sample partial
       that eagle combines with a declared reduction operator).
   * - ``exec_op``
     - The reduction operator (``sum`` / ``times`` / ``max`` / ``land``) —
       **required** when ``exec_access == "mapreduce"``, and **forbidden**
       otherwise.

A v2 artifact's entry points additionally take an explicit **partition
triple** ``{base, count, nSamples}`` after their ordinary role arguments —
eagle, never the plugin, decides how a run is split and drives the *same*
body once per partition (a device kernel launch, or one CPU thread team's
tile). The ``accum_out`` role (see :ref:`role-vocab`) names the cross-sample
accumulate plane a ``cross_sample_write`` body writes into.

**Legacy v1 artifacts are unaffected.** A ``schema_version: 1`` /
``aether_abi: "aether-abi/1"`` plugin carries none of the three keys above
(carrying one is itself a strict-key violation) and is always run
whole-view, on a single structure — precisely today's behaviour, byte for
byte. ``eagle.abi.check_aether_abi`` accepts both tags; only a v2-tagged
artifact may be split across more than one partition.

Placement legality is a function of ``exec_access`` and the chosen
structure alone, and is refused at **plan time**, naming the rule, never
guessed or discovered mid-launch: ``sample_local`` / ``cross_sample_read`` /
``mapreduce`` run under any implemented structure; ``cross_sample_write``
runs on a single-device structure only (splitting it across partitions would
need a partial-accumulate combine step not yet specified) — the same
plain-language rule :func:`eagle.exec.check_placement` enforces mechanically.

Driving a v2 body from Python — ``eagle.plan`` / ``eagle.exec``
-----------------------------------------------------------------

:mod:`eagle.exec` is the execution-structure vocabulary: the
:data:`~eagle.exec.DeviceKernel` / :data:`~eagle.exec.HostTeam` /
:data:`~eagle.exec.RankPartition` / :data:`~eagle.exec.DeviceGroup`
singletons, plus :func:`~eagle.exec.check_placement` and
:func:`~eagle.exec.check_layout_sizes` (a self-check: a v2 artifact's
exported ``eagle_layout_sizes`` must agree with this build's own POD sizes,
or the load is refused naming the disagreeing field — a matching
``aether_abi`` tag alone cannot catch a stale or wrong-arch binding).
:mod:`eagle.plan` is where a run is PLANNED: :func:`eagle.plan.plan` maps a
plugin, a chosen structure, and an explicit partitioning to a ``Plan``, and
``Plan.run(**kw)`` drives it:

.. code-block:: python

   import eagle.exec as eexec
   from eagle import plan as eplan

   # Selection is always EXPLICIT (no residency-driven auto-choice yet) --
   # the caller names the structure every time.
   whole = eplan.plan(my_v2_kernel, structure=eexec.DeviceKernel).run(x=x, a=2.0, b=-1.5)

   # Splitting a `sample_local` body into partitions never changes the
   # per-sample answer -- eagle drives the identical body per partition.
   split = eplan.plan(
       my_v2_kernel,
       structure=eexec.DeviceKernel,
       partitions=[(0, 32, 64), (32, 32, 64)],
   ).run(x=x, a=2.0, b=-1.5)

   # The same body under HostTeam (OpenMP) instead of DeviceKernel (CUDA):
   host = eplan.plan(my_v2_kernel, structure=eexec.HostTeam).run(x=x, a=2.0, b=-1.5)

   # An illegal placement is refused at plan() time, naming the rule:
   eplan.plan(
       my_v2_kernel, structure=eexec.DeviceKernel,
       partitions=[(0, 32, 64), (32, 32, 64)], exec_access="cross_sample_write",
   )  # -> ValueError: "... exec_access 'cross_sample_write' is legal on
     #    single-device structures only ..."

``partitions=`` is a list of explicit ``(base, count, n_samples)`` triples
(contiguous, covering ``[0, n_samples)``); omit it (with ``npartitions=``
left at its default of 1) for a deferred whole-view run, whose size is
resolved from the arrays passed to ``.run()``. See
``eagle/python/tests/test_exec_contract_rows.py`` for the certification rows
this API is held to (bit-exact partition identity for a ``sample_local``
body, a fixed host/device tolerance band, and the two refusal cases above).

Running across MPI ranks — ``RankPartition``
--------------------------------------------

:data:`~eagle.exec.RankPartition` is the multi-process structure. Rank *r* of
*R* runs the *r*-th **contiguous** sub-partition of one whole partition
through an **inner** structure — ``HostTeam`` or ``DeviceKernel`` — carrying
its own ``{base, count, nSamples}`` triple, with ``nSamples`` unchanged: a
rank is one more cut of the same run, not a smaller run. Inputs are
**replicated** (which is why a ``cross_sample_read`` body is legal here: the
tables it reads are whole on every rank), and outputs are **gathered**, so
every rank ends holding the whole plane, bit-identical to a single whole run.
``cross_sample_write`` stays refused — two ranks accumulating into one shared
target would need a partial-accumulate combine that is not specified.

.. code-block:: python

   # The plan's partitions stay WHOLE partitions: the rank cut happens INSIDE
   # the structure, so the same plan describes the run at any world size.
   whole = eplan.plan(
       my_v2_kernel, structure=eexec.RankPartition, inner=eexec.HostTeam,
   ).run(x=x, a=2.0, b=-1.5)          # every rank gets the WHOLE output plane

   rank, size = eexec.RankPartition.world()        # (0, 1) outside a launcher
   mine = eexec.RankPartition.local(eexec.Partition.whole(len(x)))

A ``mapreduce`` body folds its own rank's partials and then combines the *R*
partials in a fixed **rank order**
(:meth:`~eagle.exec.RankPartition.allgather_fold`) — never an
``MPI_Allreduce``, whose combine order is the implementation's business, so
two ranks could disagree in the last bits.

This structure lives in a **separate, optional** compiled extension
(``eagle._mpi``, built with ``-DEAGLE_PYTHON_MPI=ON``; ``AUTO``, the default,
builds it wherever CMake finds an MPI). ``eagle._core`` therefore never links
``libmpi``, and ``import eagle`` costs nothing to a user with no MPI
installed; where the extension is absent, every ``RankPartition`` call
refuses with a ``RuntimeError`` naming that build switch. Note that
initialising MPI is not a passive act — Open MPI installs process-wide memory
hooks — so drive this structure from a process dedicated to it (an ``mpirun``
world, or a short-lived child), not from inside a long-running session that
is also using CUDA. The certification rows are
``eagle/python/tests/test_rank_partition.py`` (single process) and the
two-rank bed ``eagle/python/tests/mpi/`` (run through its own gate,
``tests/mpi/check_rank_bed.sh``).
