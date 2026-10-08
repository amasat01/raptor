# Quickstart

Check that a kernel description is valid, and that your own class fits
raptor's interface.

**Time:** ~5 min · **Runs on:** CPU, no GPU needed · **You need:** nothing
beyond raptor itself.

## 1. Validate a kernel manifest

A manifest is a small dict describing a compiled function (a "kernel"): what
it is called, what it takes, where it may run. `raptor.schema.validate_manifest`
checks that shape:

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
validate_manifest(manifest)  # returns None; raises ValueError on a breach
```

Each key's purpose is glossed, one at a time, in
[Write a manifest](/content/tutorials/02_write_a_manifest) — start there if
`schema_version`/`pattern`/`aether_abi`/`plugins` are not yet familiar.

A document that claims a schema version raptor does not understand is
rejected, not silently accepted:

```python
bad = dict(manifest, schema_version=7)
validate_manifest(bad)
# ValueError: '<document>' was built for plugin schema v7, but this loader
# supports v2; upgrade eagle to load it
```

## 2. Check an implementation against a protocol

`raptor.protocols` names the shapes a class needs to plug into the family —
one method here, `build(directory)`, is all `ManifestProvider` asks for.
Python checks that shape with `isinstance`, structurally — no inheritance
needed:

```python
from raptor.protocols import ManifestProvider

class Bundle:
    def build(self, directory):
        return str(directory) + "/manifest.json"

isinstance(Bundle(), ManifestProvider)  # True — duck-typed, no inheritance needed
```

Any object with a matching `build(directory)` method satisfies
`ManifestProvider`, whether or not it ever imports raptor.

## 3. Read the conformance declaration

`raptor.conformance.interop` says which array libraries raptor works with
today:

```python
from raptor.conformance import interop

interop.CERTIFIED   # ('numpy', 'cupy', 'torch', 'warp')
interop.ROADMAP      # ('jax', 'tensorflow') — declared, not yet built
```

See {ref}`Reading the matrix <reading-the-matrix>` in
[Interoperability](/content/interop_protocols) for a 3-row worked example of
how one of those crossings (one library handing raptor an array) is proven,
and the same page's full matrix for every row.

::::{dropdown} Going deeper (optional): the full row catalogue
:icon: telescope

`CERTIFIED`/`ROADMAP` are two of its four declaration constants; `ROWS` is
the full per-crossing catalogue every certification claim cites, and
`UNIVERSE` is every framework the family will ever bridge to:

```python
sorted(interop.ROWS)[:3]
# ['CP-CAPTURE-REJECT', 'HAWK-DIFF-POSITIVE-ROUNDTRIP', 'HAWK-EXEC-DEPLOY-LOADABLE']
```

Every entry in `ROWS` names the repository whose test suite proves it and
how data crosses — one of five classes (pointer-shared, copied, ordering,
an environment property, or a same-result-either-way invariant). The full
certification matrix defines each one.
::::

## Next

- [Tutorials](/content/tutorials) walk through the manifest schema, the
  protocol contracts and the certification matrix in depth.
- [Examples](/content/examples) show the pieces working together with a real
  compiled kernel, using hawk and eagle.
