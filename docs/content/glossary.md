# Glossary

The words the RAPTOR family uses across raptor, [eagle](https://amasat01.github.io/eagle/) and [hawk](https://amasat01.github.io/hawk/), in plain language. Each site keeps its own glossary for its own terms and points here for the shared ones.

```{glossary}
kernel
  One compiled function that runs once per sample, on a GPU or on the CPU. raptor describes it in a small dict so that one package can build it and another can run it.

sample
  One independent item in a batch, for example one orbit or one oscillator. A kernel runs once per sample, and each sample usually has its own data column.

manifest
  The file (`manifest.json`) that lists a deployment's plugins in order. Each entry names an artifact and its sidecar.

artifact
  The compiled kernel itself, stored as a `.ptx`, `.cubin` or `.fatbin` file.

PTX
  NVIDIA's architecture-portable GPU assembly format. The driver compiles it to real machine code when it is loaded, so one `.ptx` file runs on any supported GPU generation.

sidecar
  The small JSON file that sits next to an artifact and says how to call it: the entry point, the argument list, and which kind of kernel it is.

arg_spec
  The sidecar's ordered list of `[role, name]` pairs. A loader binds arguments in exactly this order. See {ref}`the role vocabulary <role-vocab>`.

role
  What kind of argument a name is, such as `nsamples`, `per_sample` or `mutable`.

pattern
  Which shape a kernel has. A *pure* kernel writes its results in place; a *vector* kernel accumulates one output value.

producer
  The code that compiles a kernel and writes the artifact and sidecar. hawk is a producer.

loader
  The code that reads a manifest and sidecar back and hands you something you can call. eagle is a loader.

sink
  Where a kernel's result goes: the `mutable` slots of a pure kernel, or the single accumulated value of a vector kernel.

consolidation
  The one-time upload of a read-only lookup table before a run starts, as opposed to per-launch arguments that travel with every call.

schema version
  Which rules a manifest or sidecar follows (`schema_version`, currently 1 or 2). It describes the JSON shape. See {doc}`devguide/plugin_schema/compatibility`.

aether_abi
  A separate tag for the binary struct layout, so it can change without the JSON shape changing. See {ref}`aether-abi-tag`.

execution axis
  Schema v2's declaration of which execution structures may run a kernel body (`exec_targets`) and how it touches other samples' data (`exec_access`). See {ref}`execution-axis-v2`.

plan
  A kernel body bound to an execution structure and, optionally, a split of its samples into partitions, ready to run. Splitting a sample-local body into partitions never changes any sample's answer. In eagle, `eagle.plan` makes one and `eagle.deploy` returns one; see {doc}`devguide/plugin_schema/execution_axis`.

exec
  Short for execution structure: how a plan's body runs, such as one GPU kernel, a team of host threads, a set of MPI ranks or a group of devices. A schema v2 manifest lists the ones a body allows (`exec_targets`); eagle names them in `eagle.exec`.

capability
  A feature a consumer may or may not support, named in the schema so a missing one is a clear refusal, never a silent fallback.

backend
  Something that executes whole programs, described by raptor's `Backend` and `Executable` protocols.

launcher
  Something that launches one kernel and marshals its arguments, described by raptor's `KernelLauncher` and `Marshal` protocols.

marshal
  To convert an array into exactly the layout a compiled call expects.

provider
  Something that supplies kernels, described by raptor's `ManifestProvider` protocol: a compiled kernel with a schema-versioned manifest.

protocol
  A `typing.Protocol`: an implementation satisfies it by shape, without inheriting from anything in raptor.

DLPack
  The standard for sharing an array's memory between frameworks without copying. It is the contract the family's array crossings rely on. See [Interoperability](/content/interop_protocols).

certification row
  One named, executable claim about a crossing between two frameworks, such as "numpy to torch keeps the same pointer". Every owning repository's tests register against the rows. See [Interoperability](/content/interop_protocols) and {doc}`devguide/contributing_a_certification_row`.

certification matrix
  The table of all certification rows, showing which crossings are certified today.
```
