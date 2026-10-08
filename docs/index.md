```{raw} html
<div class="raptor-hero">
  <img class="raptor-reveal dark-light" src="_static/brand/wide_family_raptor.svg" alt="RAPTOR">
</div>
```

# raptor

raptor describes a compiled GPU function (a "kernel") in a small dict, so
one package can build it and another can run it.

```{image} _static/ecosystem/ecosystem_raptor_light.svg
:alt: The RAPTOR family: hawk (write it), eagle (run it), aether (the C++/CUDA numerics underneath) and raptor (the shared contract); you are looking at raptor.
:class: only-light
:align: center
```

```{image} _static/ecosystem/ecosystem_raptor_dark.svg
:alt: The RAPTOR family: hawk (write it), eagle (run it), aether (the C++/CUDA numerics underneath) and raptor (the shared contract); you are looking at raptor.
:class: only-dark
:align: center
```

[aether](https://amasat01.github.io/aether/) · [hawk](https://amasat01.github.io/hawk/) · [eagle](https://amasat01.github.io/eagle/) · [raptor](https://amasat01.github.io/raptor/) · [the family](https://amasat01.github.io/)

It is pure Python with **zero hard dependencies**, so a kernel producer such
as [hawk](https://github.com/amasat01/hawk) and an executor such as
[eagle](https://github.com/amasat01/eagle) can both build on it directly —
see [Concepts](content/concepts) for how the manifest format and the
shape-checking contracts fit together.

```python
from raptor.schema import validate_manifest

manifest = {
    "schema_version": 2,
    "pattern": "pure",
    "aether_abi": "aether-abi/2",
    "exec_targets": ["host", "device"],
    "exec_access": "sample_local",
    "plugins": [{"id": "two_body_step", "order": 0, "enabled": True, "artifact": "two_body_step.ptx",
                "sidecar": "two_body_step.json", "format": "ptx"}],
}
validate_manifest(manifest)  # raises on a malformed or out-of-order document
```

| key | meaning |
|---|---|
| `schema_version` | which rules this document follows |
| `pattern` | which kind of kernel this is; `pure` = no side effects |
| `aether_abi` | the binary-format tag paired with `schema_version` (e.g. `"aether-abi/2"` for v2) |
| `exec_targets` | host / device: where this kernel may run |
| `exec_access` | how it touches per-sample vs. cross-sample data |
| `plugins` | the compiled artifact(s) this manifest describes (`ptx` = NVIDIA's GPU assembly format) |

## Where raptor sits

```text
aether (numerics core) ---> eagle (execution) <--- raptor (shared contracts) ---> hawk (kernel authoring)
```

aether is the family's C++ numerics core (not covered on this page) that
eagle runs on; raptor depends on nothing in the family, and hawk and eagle
both declare it instead. hawk authors and compiles kernels; eagle launches,
captures and **marshals** them (marshals = converts an array into exactly
the layout a compiled call expects). Downstream packages wire kernels into
trainable programs. Each works alone — adding a companion buys speed or a
deployment option, never a new capability.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`desktop-download` Install
:link: content/installation
:link-type: doc
Pip install, zero hard dependencies, verify in one line.
:::

:::{grid-item-card} {octicon}`rocket` Quickstart
:link: content/quickstart
:link-type: doc
Build and validate a kernel manifest in under five minutes.
:::

:::{grid-item-card} {octicon}`mortar-board` Tutorials
:link: content/tutorials
:link-type: doc
What the contracts buy you, the manifest schema, the protocol contracts, and
the certification matrix.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: content/api/index
:link-type: doc
Every public class and function in `raptor.schema`, `raptor.protocols` and
`raptor.conformance`.
:::

::::

```{toctree}
:caption: Start here
:maxdepth: 1
:hidden:

content/installation
content/quickstart
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:hidden:

content/tutorials
```

```{toctree}
:caption: How-to guides
:maxdepth: 1
:hidden:

content/examples
```

```{toctree}
:caption: Explanation
:maxdepth: 1
:hidden:

content/interop_protocols
content/concepts
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

content/devguide/plugin_schema
content/api/index
```

```{toctree}
:caption: Contributing
:maxdepth: 1
:hidden:

content/contributing
content/devguide/contributing_a_certification_row
```
