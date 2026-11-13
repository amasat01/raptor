# The manifest format of raptor: the IR extension protocol, as data
Normative: sections 1-7 and 10, language-neutral | 8-9 informative. The core kinds are `raptor/ir/core.json`; `tools/kinds_diff.py` reports what changed between two versions of it, kind by kind.
## 1. Vocabulary
An EXTENSION has an id and a version and ships a manifest (data). The manifest has the arms `kinds` and `primitives` beside
`backends` and `spellings`. A DIALECT is a named group of kinds in one extension. The core kinds (every op, leaf, sink, terminal)
are the root of every refinement. A REGISTRY is a value holding declared kinds and primitives; every raptor function that needs
one takes it as an argument, and raptor keeps no module-level table of kinds or primitives. The one default registry a process
uses is created and held by the front end (hawk).
## 2. Declaring a node kind (KindDecl), all plain data
- `name`, `dialect`, `ext`; `family`: `expr`, `sink` or `terminal`; `refines`: one core or earlier-declared kind or null
  (depth <= 3). A refining kind IS that kind for every query `is_a(node, K)`; it inherits operands, result rule, identity, feathers,
  spelling and fields of what it refines unless it states its own (fields append).
- `wire`: the kind string the node carries in the walk key (default: the refined kind's wire, else `name`). Declaration identity
  (`name`, used by `is_a`) and walk identity (`wire`) are distinct, so a refinement keeps the digest of its base.
- `operands`: `{min, max|many, families[]}` (families: `any`, `plane-leaf`). `fields`: `{name, type, default}` with type `int`,
  `float`, `bool`, `str`, `none`, `tuple` (recursively plain); a field without default is required. A sink also has the builtin
  fields `name` and `role`.
- `result`: `same_as(i)`, `scalar_of(i)`, `fixed(dtype, shape)` or `carried` (the node holds its own type).
- `identity`: the ordered field names (builtins included) that, with wire, type and operand positions, key the walk's dedup.
- `spelling`: `{dialect, name, args}`, args from `node`, `target`, `ref<i>` (the writer-side text of operand i). A kind with no
  spelling and no refined one is not renderable and a build that reaches it refuses by name.
- `diff`: `none` (default: differentiating it refuses by name) or `core:<kind>`.
- `feathers`: named properties of the kind, from the CLOSED list the core table owns (`KIND_FEATHERS`, one line of meaning each; an unknown one is refused by name). A reader asks `has_feather(node, name)` and never a wing; a refining kind has the feathers of what it refines; a feather is no part of identity or of the walk. An operation of the generic op kind states feathers in the table arm `op_feathers` (`{op: [names]}`).
## 2b. Core kinds in the same format
raptor's own manifest (`raptor/ir/core.json`, extension id `raptor`, dialect `core`) is a manifest like any other: its `kinds` arm declares the core kinds, and
`kinds.load_manifest(registry, manifest, first_party)` loads it and every extension's (simt through the same call). A row that names a `class` is a CORE kind: its
Python class is generated from the row at import and the Rust build script reads the same row; any other row is a dialect kind and goes through `declare`.
Arms added beside `kinds`: `consts` (name -> list of strings, or one string; a name written `group.NAME` is a FACT held once for both languages: a string, an integer or a list of strings, read by `raptor/_facts.py` in Python and compiled by `rs/core/build.rs` into the module `facts` of the core; see `tests/test_one_home.py`), `records` (`{name, fields: ["f", "g=default"]}`: frozen dataclasses), `checks`
(named, reusable: `{params, steps}`). Keys added to a core row (all optional for a dialect row, ignored by `declare` for it): `class`; `refinable` (a dialect kind may
refine it; with `role`, `wire`); `slots` (own `__slots__`); `wire_attr` (the class attribute `kind`); `params` (the constructor signature, with annotations);
`assigns` (the slots `ctor.out` fills, in order); `binding` (`slot`|`none`); `dedup` (`none`|`py`|`one` item|`tup` items; items `f:field`, `r:field` (repr of the
field as text), `s:text`, `k` (operand count - 1), `fx:field:v1,v2` (the field unless it is one of them)); `state`, `from`, `edges`, `from_edges`, `props` (how the
class is read back: `@k`, `@k:plain`, `op:N`, `opq:N`, `ops:N:`, `edge:k`, `ttype`, `operands`); `watch` (`all`|`raw`: the nodes the access and dispatch passes look at);
`ctor` = `{steps, out}`: the construction CHECKS, run by one generic interpreter in the binding (`rs/binding/src/lib.rs`), in the order written.
Steps: `["let", name, expr]`, `["check", expr, template]` (refuse with the rendered template when the expression is true), `["node", ttype_expr, operands_expr]`
(the base rules of every node: the named check `node_base`, then `_t` and `_o` are set), `["call", name, args...]` (a named check; `_t`/`_o` set inside come back).
Expressions: names (parameters, `let`s, constants, class names), integers, `'strings'`, `None`/`True`/`False`, tuples, `.attr`, `[i]`, `not and or`, `== != < > in
not in is is not`, and the builtins `len isinstance tt width_of rem int tuple cat opt head first_ne_ttype first_not_node coalesce orelse
auto_or_int name_of bool trip first_mismatch first_not_instance carry_fault ttype_fault literal_fault` (the list is `BUILTINS` in `rs/binding/src/lib.rs`). A template substitutes `{expr}` (str) and `{expr!r}` (repr). `check_manifest` reports an unknown name, function or check, and a wrong argument count; the message names the manifest row and lists what that row may use.
## 3. What the verifier checks
At declaration: name unique over core and all extensions; extension id `[a-z][a-z0-9_-]*`; `refines` resolves, no cycle; `family`
agrees with the refined kind; field types, defaults, collisions with builtin names; result index < `operands.min`; identity and
feathers name known things; spelling args within the operands. The writer's spelling is looked up at first render, by name.
At node build: operand count and families; each field present, typed, no extra; the type equals what `result` derives.
## 4. Canonical text of a node
The walk key of a node is (wire, dtype, shape, tag, operand positions, identity values), so a declared kind and the class it
replaces give the same key. Fields outside `identity` travel with the node (declared order) but never enter the digest. Values are
written by the key-text rule of section 10. A node
is rebuilt from (kind, type, operands, fields) by its declaration alone.
## 5. How a writer finds its spelling
A writer exposes the registration of spellings by (writer id, dialect, name). The renderer reads `spelling` from the node's
declaration and calls the writer through that name; the node has no render method and is asked nothing but its kind. A writer
without the spelling refuses: "the <writer> writer has no spelling for <dialect>.<name>". Spelling code stays in the writer's package.
## 6. Identity, collisions, reserved names
- Extension ids are unique per registry; the first-party set (`hawk`, `hawk-c`, `hawk-cuda`, and `simt`, a working label) may use
  unqualified kind names, so digests of the core kinds do not change. Any other extension spells its kinds `<ext id>.<name>`.
- A kind name is one flat namespace with the core kinds. An equal re-declaration is a no-op; a different one refuses, naming both.
- Primitive names are their own namespace; `primitive`, every core kind and every declared kind name is reserved.
## 7. User primitives and their derivative rule
- Registration declares `name`, `ext`, and the directions (`vjp`, `jvp`) it supplies; an equal re-registration is a no-op.
- The primal is IR: operand 0 of the node (inlined). Spec only: a primal may instead be a named external symbol that a
  writer spells; not built.
- A rule is a TEMPLATE: a DAG of ordinary IR whose only free leaves are placeholders `$in<i>`, `$out`, `$cot` (incoming adjoint),
  `$dot<i>` (tangent of input i). It is registered per (name, signature), signature = input
  types and output type (monomorphic). vjp = one template per input; jvp = one template. Instantiation substitutes the node's
  operands for the placeholders, structurally; an input with no tangent gets a structural zero.
- A template may contain loops. Inside a loop body it may read the placeholders, the loop's own index and carries, and constants;
  substitution goes through operands and bodies in one traversal and gives each instantiation fresh index and carries.
  Refused by name: a free leaf other than a placeholder (also inside a body), an index or carry read outside its loop, a loop
  that writes a plane (body sinks), a tape read, any other node that holds edges. A loop's carries are internal and start from
  template values, so no state crosses the template boundary and nothing needs refusing for it.
- The Primitive node holds its name only; its digest identity is ("primitive", name).
- Validation at rule registration: vjp count equals the input count ("ONE adjoint per declared input"); each template's type equals
  the type owed ("was owed"); free leaves are only the allowed placeholders; an equal re-registration is a no-op, a different one refuses.
- A missing direction refuses by name at differentiation, naming primitive and direction; there is no fallback through the primal.
## 8. Where the pieces live
- A dialect (a group of kinds in one extension) ships its own manifest; the front end loads it through `kinds.load_manifest` and asks `is_a`/`kind_name`.
- A primitive keeps its callables in the front end only to trace them, lazily at the first differentiation in a direction, into a template registered as data.
- A unit's cost is a plain dict computed when the frame is built. A quantity names its wire packing (`auto`|`tuple`), not a function.
## 9. Informative: Python presentation
raptor builds, from each declaration, one class (a subclass of the refined core class) so existing `isinstance` on core classes
keeps working; it is generated by raptor from data. A Rust core presents the same through its binding.
## 10. The key text of a value (written rule, implemented in rs/core/src/keytext.rs, no call into Python)
The dedup identity of a node (section 4, `identity` and the `dedup` recipes of core.json) is written as text. Each value kind that reaches it:
- None: `None`. bool: `True` / `False` (checked before int). int: decimal digits, `-` for negative, no padding; a value beyond 128 bits is refused by name.
- float (exact type): `nan`, `inf`, `-inf`; zero is `0.0` or `-0.0` by its sign bit; otherwise the SHORTEST digit string that reads back to the same
  double (the closest such string), with decimal point position `p` (value = 0.DIGITS x 10^p): when -4 < p <= 16 a plain decimal (`0.0001`, `123.456`,
  `1000000000000000.0`: leading `0.` and zeros when p <= 0, trailing zeros and `.0` when p >= number of digits), else `d[.ddd]e` + sign (always) + at least
  two exponent digits (`1e-05`, `1.5e+300`). Subnormals follow the same rule.
- str (exact type): enclosed in `'`, or in `"` when it holds `'` and no `"`; `\` is `\\`, the enclosing quote is escaped, `\t \n \r` are `\t \n \r`; any
  other code point that Unicode (version pinned in rs/core/src/nonprintable.rs, generated by tools/gen_nonprintable.py) does not class as printable
  (categories Cc Cf Cs Co Cn Zl Zp Zs except U+0020) is `\xNN` (< 0x100), `\uNNNN` (< 0x10000) or `\UNNNNNNNN`, lower-case hex; all else (non-ASCII included) as is.
- tuple: `(a, b)`, one element `(a,)`, empty `()`; list: `[a, b]`. Elements are separated by `, `.
- `r:<field>` items write the text of the field's value by these rules, then that text AS A str (quoted again), exactly as the old text did.
- An OBJECT may state its own key text: a class with a `__key_text__()` method returning a str is written as that str, verbatim (the text must itself be written by this section's rules for its parts: `raptor.ir.key_text(value)` applies them). The one such class in the code base is hawk's RawBlock (the literal of a `raw_device` op), whose text is `<RawBlock NAME access=ACCESS>` with ACCESS a str by the rule above, byte for byte what Python's repr gave. An `r:` item over such an object quotes this text as a str.
A `Const` or `Op` literal is checked against this rule when the node is built (`literal_fault`), and a list is refused there: a node does not change after it is built.
Every other type (numpy scalars, subclasses of these types, bytes, dtypes, objects without the method) is refused by name: "key text: a value of type T has no written rule".
The corpus of digests and the 10^6-float test (tests/test_keytext.py) pin this rule to CPython 3.15 / Unicode 17.0.0.

Types that can reach the key text, from the code (not from a corpus): the `literal` of Const (int, float, bool: hawk converts at the source), of RoleConst (int), of Op (declared `Any`: None, int, str (cast target dtype), tuple of ints (shape, index), RawBlock); the fields of dialect kinds (int, float, bool, str, none, tuple: SPEC 2); Loop dedup parts (str, int, tuples of str); Walk digest tail (str, tuples of str, TensorType and QuantitySpan records written field by field). A new literal holder must follow the hook AND be declared: the manifest key `key_text` (core.json) lists the class names that may use it (RawBlock); any other object that has the method is refused by name.
