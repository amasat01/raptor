Compatibility policy
====================

.. contents:: On this page
   :local:
   :depth: 2

Validation happens at **load time**, before any launch, and follows two rules:

**Backward-lenient.**
   An artifact with **no** ``schema_version`` is treated as **v1**, so every
   pre-freeze manifest / sidecar stays loadable. For a manifest, the legacy
   ``version`` key is honored as a fallback. (Python:
   :func:`eagle.roles.check_schema_version`; C++: the ``kPluginSchemaVersion``
   gate parsing an absent tag as ``0`` ⇒ v1.)

**Forward-strict.**
   A loader **rejects**:

   * any ``arg_spec`` role outside the 9-role vocabulary
     (:func:`eagle.roles.validate_roles` / ``is_valid_role``);
   * any ``schema_version`` **greater than** the loader supports;
   * any ``scalar_type`` outside ``{"float64", "float32", "softdouble"}``
     (Python: the ``SCALAR_TYPES`` enum check in
     :class:`eagle.loaded.LoadedKernel`; C++: ``validate_sidecar`` in
     ``plugin/sidecar.h``). An absent ``scalar_type`` stays backward-lenient
     (a pre-dtype sidecar, treated as ``"float64"``) — only a *recognized*
     value is required, not a *present* one; and
   * any ``pattern`` value outside ``{"vector", "pure"}``, at **both** the
     manifest and sidecar level ("the discriminant rule": ``pattern``
     *selects* a schema variant, so every loader that reads it — even one that
     does not dispatch on it — must refuse a value it cannot name, rather than
     silently bind an unfamiliar variant as an ordinary kernel). Python:
     ``registry.py::load_manifest`` (manifest) and
     :meth:`eagle.loaded.LoadedKernel._require_pattern` (sidecar, via each
     ``Loaded*`` subclass). C++: ``PluginRegistry::from_manifest`` in both
     ``plugin/plugin_registry/registry.h`` (device) and ``plugin/host_registry.h``
     (host) for the manifest level; the shared ``validate_sidecar`` in
     ``plugin/sidecar.h`` for the sidecar level (so the public ``add_plugin``
     entry point — which bypasses manifests entirely — is covered too); and
   * any ``aether_abi`` that is **absent** (empty), not just one that is present
     but mismatched — a compatibility *gate*, not an optional hint, so
     presence is required at **every** entry point that binds a sidecar or
     manifest: the two manifest-level loads (``PluginRegistry::from_manifest``
     in ``plugin/plugin_registry/registry.h`` and ``host_registry.h::load``)
     and the sidecar-level, manifest-bypassing ``add_plugin``
     (``plugin/host_registry.h``) all reject an empty ``aether_abi`` the same
     way they reject a mismatched one. The Python peer of
     ``add_plugin`` — :class:`eagle.host_launch.HostPluginLibrary` — was the
     last absence-lenient door in either language (a private inline check,
     not :func:`eagle.abi.check_aether_abi`); harmonized to presence-required
     too, so this is now true of every loader in both languages, not
     just the C++ ones named above.

Both loaders enforce the *same* vocabulary and the *same* version gate. The C++
half is ``plugin/roles.h`` and the Python half is ``eagle/roles.py``; the
cross-check test ``python/tests/test_roles_vocab.py`` asserts the two role lists
are set-equal (and the same for the RECOGNIZED / LAUNCH-CERTIFIED
pattern sets, the buffer-``kind`` vocabulary, and the manifest-entry ``format``
vocabulary), the two ``schema_version`` constants agree, and every committed
fixture uses only canonical roles. ``eagle/python/tests/test_schema_hardening.py``
and ``eagle/tests/test_SchemaHardening.cpp`` fix the exact set of forward-strict
fields — now nine: ``schema_version`` and ``aether_abi`` (class ``gate``);
``kernel`` and ``arg_spec role`` (class ``structural``); ``pattern``,
``scalar_type``, buffer ``kind``, ``derivative.kind``, and manifest-entry
``format`` (class ``discriminant``) — so the strict/lenient split cannot erode
silently, and a CLASS mismatch (a loader implementing a private literal compare
for a classed field outside the shared validation layer) is caught too. See the
frozen-field-list note in each file for the version-bump rule and the full CLASS
definitions.

``pattern``'s *absence* handling is the one place C++ and Python genuinely
diverge, by design, not a gap:

* At the **manifest** level, Python's ``load_manifest`` is absence-**strict**
  (an absent ``pattern`` is rejected, same as an unknown one) — it must dispatch
  to ``LoadedVector`` or ``LoadedPure`` and has no third option. The two C++
  registries stay absence-**lenient** (a pre-freeze manifest that never stamped
  one still loads): neither registry dispatches a variant class off ``pattern``,
  binding is driven entirely by ``arg_spec``, so there is nothing an absent value
  would fail to gate. Pinned mechanically by the shared conformance corpus (a
  manifest with no ``pattern`` key, deliberately also carrying a duplicate
  plugin id: C++ passes the lenient gate and rejects on the duplicate id
  instead; Python's strict gate rejects first) — the ``overrides`` mechanism
  in ``eagle/tests/conformance/row13_manifest_pattern_absence_asymmetry.*``
  encodes both outcomes for one fixture, so a future change that
  accidentally harmonizes the two languages fails a test rather than
  drifting unnoticed.
* At the **sidecar** level, C++'s ``validate_sidecar`` is likewise
  absence-lenient, for the same arg_spec-driven reason (Python's shared
  validator agrees — both languages accept a ``pattern``-less sidecar at
  this layer, pinned by
  ``eagle/tests/conformance/row14_sidecar_pattern_absence_lenient.*``).
  Layered on top, Python is per-``Loaded*``-class: ``LoadedVector`` defaults
  an absent value to ``"vector"`` (lenient), while ``LoadedPure`` requires an
  exact match (no default) — this further split has no C++ counterpart to
  diverge from (C++ has no per-kind sidecar loader at all), and pinning it
  needs a real device module load (``_require_pattern`` runs after
  ``LoadedKernel.__init__``'s ``cupy.RawModule`` load), which the corpus's
  GPU-free design cannot accommodate. It is instead pinned directly —
  ``test_loaded_vector_defaults_absent_pattern_to_vector`` /
  ``test_loaded_pure_rejects_absent_pattern`` in
  ``eagle/python/tests/test_schema_hardening.py`` — against the committed
  ``gravity``/``bump`` PTX fixtures. The asymmetry itself remains locked
  behaviour, not a defect.

This is narrower than an earlier state: previously C++ did not even reject an
*unrecognized* ``pattern`` value (a manifest-only fix would also have left the
sidecar-level, manifest-bypassing ``add_plugin`` entry point open — see
``eagle/tests/test_SchemaHardening.cpp`` and ``test_PluginRegistryManifest.cu``
for the registry-level rejection tests, and ``test_HostPlugin.cpp``-adjacent
tests in ``test_SchemaHardening.cpp`` for the ``add_plugin`` path).

.. _aether-abi-tag:

The ``aether_abi`` tag is separate
-----------------------------------

``aether_abi`` (``"aether-abi/1"``) is a **distinct, ABI-stable** tag — the by-value
binary-layout contract of the ``GRef`` / ``HandleT`` POD mirrors the artifact was
built against. It is **orthogonal to**
``schema_version``: the schema version governs the *JSON* shape and the role
vocabulary, whereas ``aether_abi`` governs the *binary* struct layout a loader binds
by value. A loader rejects an ``aether_abi`` mismatch before binding any by-value
struct (Python: :func:`eagle.abi.check_aether_abi`; C++: the ``EAGLE_AETHER_ABI``
static-assert gates). The value is historical and must not change without a
coordinated C++/Python bump — pre-split artifacts remain loadable precisely
because it is stable.
