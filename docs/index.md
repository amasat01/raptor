```{raw} html
<div class="raptor-hero">
  <img class="raptor-reveal dark-light" src="_static/brand/wide_family_raptor.svg" alt="RAPTOR">
</div>
```

# raptor

raptor describes a compiled GPU function (a "kernel") in a small dict, so
one package can build it and another can run it.

```{raw} html
:file: _static/ecosystem/ecosystem_cards_raptor.html
```

It is pure Python with **zero hard dependencies**, so a kernel producer such
as [hawk](https://github.com/amasat01/hawk) and an executor such as
[eagle](https://github.com/amasat01/eagle) can both build on it directly —
see [Concepts](content/concepts) for how the manifest format and the
shape-checking contracts fit together.

**What eagle and hawk do with raptor's contracts**

```{raw} html
<figure class="perf-glimpse">
<a href="https://amasat01.github.io/eagle/content/performance.html"><img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/perf/readme_gpu.svg" alt="Bar chart of wall time per tool for a million RK4 oscillators, log scale; the device, tools and numbers are on eagle's performance page." style="max-width:100%;height:auto"></a>
<figcaption>raptor has no benchmarks of its own. This is eagle running hawk kernels that follow raptor's manifest: a million RK4 oscillators, each stopping at its own step, wall time per tool, log scale. Measured by eagle's committed performance cards; the device is named on the performance page. <a href="https://amasat01.github.io/eagle/content/performance.html">Every device, workload and arm</a>.</figcaption>
</figure>
```

```{raw} html
:file: _static/family_visuals/fragments/map.html
```

*Where each tool covers the work, drawn from eagle's committed performance cards; the table behind the chart lists every value. [The radar](https://amasat01.github.io/eagle/content/performance.html) is on eagle's performance page.*

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
trainable programs. Each works alone; a companion adds speed, derivatives or a
deployment route (hawk's kernels and manifest loading need eagle to run).

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
threading
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

content/devguide/plugin_schema
content/api/index
content/glossary
```

```{toctree}
:caption: Contributing
:maxdepth: 1
:hidden:

content/contributing
```
