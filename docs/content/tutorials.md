# Tutorials

Four short, sequential notebooks. Each builds on the last; none needs a GPU,
hawk or eagle — they only exercise raptor itself.

1. [What the contracts buy you](/content/tutorials/01_what_the_contracts_buy_you) —
   watch two toy engines that have never heard of each other both satisfy
   the same protocols, by shape alone.
2. [Write a manifest](/content/tutorials/02_write_a_manifest) — build a v1
   and a v2 manifest by hand, validate both, and see exactly how
   `validate_manifest` rejects a malformed one.
3. [Implement a protocol](/content/tutorials/03_implement_a_protocol) —
   implement a minimal `Backend`, `ManifestProvider` and `Marshal` from
   nothing but their protocol shape, and check them with `isinstance`.
4. [Read the interop matrix](/content/tutorials/04_plug_into_the_interop_matrix) —
   read `raptor.conformance.interop`'s declaration and check which array
   libraries this environment actually has installed. Contributing a
   certification row of your own is covered separately, in the devguide.

```{toctree}
:maxdepth: 1
:hidden:

tutorials/01_what_the_contracts_buy_you
tutorials/02_write_a_manifest
tutorials/03_implement_a_protocol
tutorials/04_plug_into_the_interop_matrix
```
