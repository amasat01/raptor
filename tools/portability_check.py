#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""raptor/tools/portability_check.py — the isolated-venv portability gate,
raptor's own leg.

Self-contained by design: raptor holds zero eagle/hawk path knowledge outside
a `cross_repo`-marked test, and this tool is not one. This file knows only
raptor's OWN tree (its grandparent directory) plus whatever wheelhouse/scratch
paths and leak-route overrides are handed to it on the command line. It never
hardcodes a sibling repo's path — that knowledge belongs to a separate,
workspace-wide cross-repo driver, the one place allowed to know about every
repo.

raptor has zero hard dependencies and no conftest.py (dependency-free by
design) — so route 3 (conftest sibling injection) does not apply here; routes
1 and 2 still do, and are demonstrated the same way as the other legs'
scripts:
  1. system-site skbc/editable finder -> venv built WITHOUT
     `--system-site-packages` (RED demo: `--system-site`)
  2. PYTHONPATH                       -> every interpreter call is `python -I`
     PLUS the four leak vars explicitly unset (RED demo: `--no-isolate`
     combined with `--leak-pythonpath`)

The `cross_repo`-marked files HARD FAIL rather than skip when eagle/hawk are
absent — deselected here via `-m "not cross_repo"`, the same expression
raptor's own CI uses; the gate that runs the full battery (including
`cross_repo`) is a separate invocation, not this leg.

Also deselected here (never in the repo-rooted suite): `repo_local`-marked
tests. Those certify the REPOSITORY (docs/, pyproject.toml, schema source
read AS TEXT, or a dev-env package-presence assumption like `test_v2_
present_roadmap_is_red`'s `find_spec("torch")`) rather than the installed
DISTRIBUTION a wheel actually ships -- a wheel legitimately does not carry
`docs/`, `goldens/`, or an unrelated dev tool's presence, so failing on
their absence here would not be evidence against portability. raptor's
repo-rooted suite (CI, dev runs) NEVER deselects `repo_local` and must stay
fully passed / 0 skipped; this leg is the only place it is deselected, and
the count of what gets deselected is enumerated by collection and pinned
below -- never a hand-maintained list, so a test someone mismarks
`repo_local` to cheaply green this leg is caught as a RED on
EXPECTED_REPO_LOCAL_COUNT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "raptor"
# PACKAGE_NAME is the IMPORT name (flat, unchanged); DEFAULT_TARGET_SPEC is a
# pip INSTALL spec and carries the DIST name. A bare "raptor" here would
# resolve to an unrelated py2 package on any index.
DEFAULT_TARGET_SPEC = "raptor-core"
LEAK_VARS = ["PYTHONPATH", "EAGLE_PYTHON", "RAPTOR_PATH", "RAPTOR_ROOT"]
DEFAULT_MARKER_EXPR = "not cross_repo and not repo_local and not ft"
# Pinned by commissioning measurement (venv leg): 20 passed, 0 failed,
# 0 skipped, 0 errors, 28 deselected (18 cross_repo + 10 repo_local).
#
# RE-MINTED after landing the zero-dep floor test
# (`tests/test_blocks_zero_dep_floor.py`, the acceptance row for
# `raptor.schema.blocks`'s numpy/eagle/cupy-less import + entry-point call):
# it is neither `cross_repo` nor `repo_local` -- it IS the portability claim
# this leg exists to certify -- so it runs here and adds one pass.
# RE-MINTED 2026-09-23 (venv leg): 56 passed, 0 failed, 0 skipped, 0 errors,
# 32 deselected (22 cross_repo + 10 repo_local).
# RE-MINTED 2026-09-24 (venv leg): 57 passed, 0 failed, 0 skipped, 0 errors,
# 46 deselected (37 cross_repo + 9 repo_local). The +1 pass is
# test_roadmap_framework_raises_when_framework_is_present, which now probes
# pytest (present in every run) instead of torch, so it no longer assumes a
# dev env and left repo_local.
# RE-MINTED 2026-10-02 (venv leg): 58 passed, 0 failed, 0 skipped, 0 errors,
# 46 deselected. The +1 pass is test_warp_rows_declared_exactly (the NVIDIA Warp
# rows of the interop matrix), a declaration check with no framework import.
# 2026-10-09 (venv leg): still 58 passed, 0 skipped. tests/test_freethreading.py
# is marked `ft` throughout; its rows need a free-threaded interpreter and skip
# on a GIL build, so this leg deselects `ft` (DEFAULT_MARKER_EXPR) and ci.yml's
# ft-stress job runs them on 3.13t and 3.14t.
DEFAULT_EXPECTED_PASS_COUNT = 58
# Pinned by commissioning measurement, collecting `-m repo_local` against the
# live tree (10 node IDs: the 9 tests that used to fail in the venv leg --
# test_pyproject_declares_zero_hard_deps_and_name_raptor,
# test_roadmap_framework_raises_when_framework_is_present,
# test_at_least_three_goldens_exist,
# test_protocols_doc_{exists,rows_match_declaration,is_wired_into_the_toctree},
# test_{manifest,dtypes,blocks}_py_is_{purity_clean,the_extension_layer...} --
# plus test_golden_passes_validate_manifest, newly repo_local per a later
# guardrail addition). 2026-09-24: 9 --
# test_roadmap_framework_raises_when_framework_is_present left repo_local (it
# probes pytest now, not torch).
# A drift in this count (either direction) REDs -- update ONLY with a
# reviewed reason, never to silence a RED.
EXPECTED_REPO_LOCAL_COUNT = 9

# GPU-safe: find_spec only, never imports/executes the target or eagle.
# argv[3], if given, is a test_cwd to probe route 3 with: it is inserted onto
# sys.path and its conftest.py (if any) is imported for side effects ONLY
# (the same sys.path.insert(0, ROOT) a checkout-rooted pytest run would
# trigger) before the find_spec checks below run -- this is what lets the
# SAME canary invocation catch route 3 (python -I does not stop a conftest's
# own explicit sys.path mutation, only the interpreter's automatic ones).
# raptor itself has no conftest.py (route 3 does not apply to this leg), but
# the probe is harmless (the import simply fails) and kept identical to the
# other two legs' scripts for one code path across all three.
CANARY_SRC = """
import hashlib, importlib.util, json, sys
pkg = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else ""
test_cwd = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
if test_cwd:
    sys.path.insert(0, test_cwd)
    try:
        import conftest  # noqa: F401  (side effects only -- route 3 probe)
    except BaseException:
        pass
result = {}
result["tag"] = tag
result["test_cwd_probed"] = test_cwd
result["executable"] = sys.executable
result["in_venv"] = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
result["isolated_flag"] = sys.flags.isolated
result["sys_prefix"] = sys.prefix
result["sys_base_prefix"] = getattr(sys, "base_prefix", sys.prefix)
finders = [type(f).__module__ + "." + type(f).__name__ for f in sys.meta_path]
result["meta_path"] = finders
result["skbc_finder_present"] = any(
    "skbc" in f.lower() or "scikitbuild" in f.lower() for f in finders
)
try:
    spec = importlib.util.find_spec("eagle")
    result["eagle_find_spec_origin"] = spec.origin if spec else None
except BaseException as e:
    result["eagle_find_spec_origin"] = "RAISED " + type(e).__name__ + ": " + str(e)
try:
    spec = importlib.util.find_spec(pkg)
    origin = spec.origin if spec else None
except BaseException as e:
    origin = "RAISED " + type(e).__name__ + ": " + str(e)
result["target_find_spec_origin"] = origin
if origin and origin not in ("built-in", "frozen") and "RAISED" not in origin:
    try:
        with open(origin, "rb") as fh:
            result["target_md5"] = hashlib.md5(fh.read()).hexdigest()
    except OSError as e:
        result["target_md5"] = "UNREADABLE: " + str(e)
else:
    result["target_md5"] = None
print("CANARY_JSON:" + json.dumps(result))
"""


def log(*a):
    print(*a, file=sys.stderr)


def run(cmd, **kw):
    log("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def clean_env(overrides=None):
    env = os.environ.copy()
    for v in LEAK_VARS:
        env.pop(v, None)
    # Plain pytest summary lines: the parser reads them.
    env["NO_COLOR"] = "1"
    env["PY_COLORS"] = "0"
    if overrides:
        env.update(overrides)
    return env


def build_venv(venv_dir: Path, base_python: str, system_site: bool):
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    cmd = [base_python, "-m", "venv"]
    if system_site:
        cmd.append("--system-site-packages")
    cmd.append(str(venv_dir))
    r = run(cmd, env=clean_env(), capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stdout, r.stderr)
        raise SystemExit(f"venv creation failed rc={r.returncode}")


def pip_install(venv_dir: Path, wheelhouse: str, spec: str, env: dict):
    pip = venv_dir / "bin" / "pip"
    return run(
        [str(pip), "install", "--no-index", "--find-links", wheelhouse, spec],
        env=env,
        capture_output=True,
        text=True,
    )


def sibling_free_copy(scratch_dir: Path) -> Path:
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    shutil.copytree(
        REPO_ROOT / "tests",
        scratch_dir / "tests",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    conftest = REPO_ROOT / "conftest.py"
    if conftest.exists():
        shutil.copy(conftest, scratch_dir / "conftest.py")
    return scratch_dir


def run_canary(
    venv_python: Path, tag: str, isolate: bool, env: dict, test_cwd: str = ""
) -> dict:
    cmd = [str(venv_python)]
    if isolate:
        cmd.append("-I")
    cmd += ["-c", CANARY_SRC, PACKAGE_NAME, tag, test_cwd]
    r = run(cmd, env=env, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("CANARY_JSON:")]
    if not lines:
        return {
            "error": "no canary output",
            "rc": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    return json.loads(lines[-1][len("CANARY_JSON:") :])


def evaluate_assertions(canary: dict, venv_dir: Path) -> dict:
    site_prefix = str(venv_dir)
    origin = canary.get("target_find_spec_origin")
    return {
        "in_venv": bool(canary.get("in_venv")),
        "isolated": canary.get("isolated_flag") == 1,
        "no_skbc_finder": not canary.get("skbc_finder_present", True),
        "eagle_absent": canary.get("eagle_find_spec_origin") is None,
        "target_under_site_packages": bool(origin) and origin.startswith(site_prefix),
        "target_origin": origin,
        "target_md5": canary.get("target_md5"),
    }


def run_pytest(
    venv_python: Path, cwd: Path, marker_expr: str, isolate: bool, env: dict
):
    cmd = [str(venv_python)]
    if isolate:
        cmd.append("-I")
    cmd += ["-m", "pytest", "-q", "-rs", "--no-header"]
    if marker_expr:
        cmd += ["-m", marker_expr]
    return run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)


def enumerate_repo_local(
    venv_python: Path, cwd: Path, isolate: bool, env: dict
) -> dict:
    """Discover every `repo_local`-marked node ID by COLLECTION (never a
    hand-maintained list) and pin the count against EXPECTED_REPO_LOCAL_COUNT.
    A mismatch (either direction -- someone deselecting a real portability
    test via `repo_local`, or a fix removing one that should stay) is
    reported as `pin_ok: false`, which the caller must treat as RED."""
    cmd = [str(venv_python)]
    if isolate:
        cmd.append("-I")
    cmd += ["-m", "pytest", "--collect-only", "-q", "--no-header", "-m", "repo_local"]
    r = run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    node_ids = sorted(
        l.strip()
        for l in r.stdout.splitlines()
        if "::" in l and not l.strip().startswith(("=", "!"))
    )
    return {
        "rc": r.returncode,
        "node_ids": node_ids,
        "count": len(node_ids),
        "expected_count": EXPECTED_REPO_LOCAL_COUNT,
        "pin_ok": len(node_ids) == EXPECTED_REPO_LOCAL_COUNT,
    }


# pytest's real final summary line is NOT reliably wrapped in "=...=": it IS
# when collection succeeds ("=== 52 passed in 0.15s ==="), it is NOT when
# collection is interrupted by errors ("2 skipped, ... 19 errors in 1.73s",
# bare -- the previous "starts and ends with '=' " scan matched the
# `=== warnings summary ===` BANNER instead in that case and silently
# reported 0 errors on a many-error run. Anchor on the "in <N>.<N>s" suffix
# every terminal summary line carries instead, `=`-wrapped or not.
_FINAL_SUMMARY_RE = re.compile(
    r"^\s*=*\s*(?P<body>.*?\bin\s+[\d.]+s(?:\s*\([^)]*\))?)\s*=*\s*$"
)
_ERROR_COLLECTING_RE = re.compile(r"ERROR collecting (\S+)")
_COUNT_RES = {
    "n_failed": re.compile(r"(\d+)\s+failed"),
    "n_passed": re.compile(r"(\d+)\s+passed"),
    "n_skipped": re.compile(r"(\d+)\s+skipped"),
    "n_errors": re.compile(r"(\d+)\s+error"),
    "n_deselected": re.compile(r"(\d+)\s+deselected"),
    "n_xfailed": re.compile(r"(\d+)\s+xfailed"),
    "n_xpassed": re.compile(r"(\d+)\s+xpassed"),
    "n_warnings": re.compile(r"(\d+)\s+warning"),
}


class SummaryParseError(RuntimeError):
    """pytest's terminal summary line could not be located in stdout. Raised
    rather than defaulting every counter to 0 -- a counter that reads 0
    because nothing matched is indistinguishable from a genuine 0, and that
    ambiguity is itself the defect."""


def summarize_pytest(result) -> dict:
    if result is None:
        return {}
    lines = result.stdout.splitlines()

    collect_error_files = sorted(
        {m.group(1) for l in lines if (m := _ERROR_COLLECTING_RE.search(l))}
    )
    modulenotfound = sorted({l.strip() for l in lines if "ModuleNotFoundError" in l})

    final_line = None
    for l in reversed(lines):
        m = _FINAL_SUMMARY_RE.match(l)
        if m:
            final_line = m.group("body").strip()
            break
    if final_line is None:
        raise SummaryParseError(
            "pytest's terminal summary line was not found in stdout -- refusing "
            "to default n_passed/n_failed/n_skipped/n_errors/n_deselected to 0. "
            f"Last 5 stdout lines: {lines[-5:]!r}"
        )

    counts = {key: 0 for key in _COUNT_RES}
    for key, rx in _COUNT_RES.items():
        m = rx.search(final_line)
        if m:
            counts[key] = int(m.group(1))

    return {
        "rc": result.returncode,
        "final_summary_line": final_line,
        **counts,
        "collect_error_files": collect_error_files,
        "n_collect_errors": len(collect_error_files),
        "collect_error_count_matches_summary": (
            len(collect_error_files) == counts["n_errors"]
        ),
        "modulenotfound_lines": modulenotfound,
    }


def compute_verdict(
    pytest_summary: dict, expected_pass_count, extra_invariants=None
) -> dict:
    """The per-leg contract enforced by the RECORD, not by eye: 0 errors, 0
    skips, pass count == pinned baseline (when one is pinned) -- AND rc must
    agree. If rc signals a problem while the counters claim clean, that is
    NEVER trusted as green -- it is flagged INCONSISTENT, because that is
    exactly the shape a broken parser produces (this happened once already:
    rc=2, n_collect_errors=0).

    `extra_invariants`, when given, is a {name: bool} map of OTHER computed
    judgements (e.g. `repo_local`'s count pin) that must ALL be true for
    GREEN -- a judgement computed and recorded but never checked here is the
    same non-propagation defect (found a second and third time in this
    artifact's sibling legs: `file_count_pin_ok` was computed, printed, and
    never read by this function either)."""
    if not pytest_summary:
        return {"verdict": "NOT_RUN"}
    rc = pytest_summary.get("rc")
    n_errors = pytest_summary.get("n_errors", 0)
    n_skipped = pytest_summary.get("n_skipped", 0)
    n_failed = pytest_summary.get("n_failed", 0)
    n_passed = pytest_summary.get("n_passed", 0)
    pass_count_ok = expected_pass_count is None or n_passed == expected_pass_count
    invariants_ok = all((extra_invariants or {}).values())
    counters_clean = (
        n_errors == 0
        and n_skipped == 0
        and n_failed == 0
        and pass_count_ok
        and invariants_ok
    )
    rc_ok = rc == 0
    if rc_ok and counters_clean:
        verdict = "GREEN"
    elif (not rc_ok) and counters_clean:
        # rc says problem, counters say clean -- do NOT trust; this is the
        # exact shape a broken parser produces.
        verdict = "INCONSISTENT"
    else:
        verdict = "RED"
    return {
        "verdict": verdict,
        "counters_clean": counters_clean,
        "rc_ok": rc_ok,
        "pass_count_ok": pass_count_ok,
        "expected_pass_count": expected_pass_count,
        "extra_invariants": extra_invariants or {},
        "invariants_ok": invariants_ok,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheelhouse", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--base-python", default=sys.executable)
    ap.add_argument("--system-site", action="store_true")
    ap.add_argument(
        "--no-venv",
        action="store_true",
        help="skip venv entirely, run the canary directly with base_python "
        "(the in_venv self-assertion's RED demo -- no install, no pytest)",
    )
    ap.add_argument("--no-isolate", action="store_true")
    ap.add_argument("--leak-pythonpath", default=None)
    ap.add_argument("--checkout-rooted", action="store_true")
    ap.add_argument("--set-env", action="append", default=[])
    ap.add_argument("--marker-expr", default=DEFAULT_MARKER_EXPR)
    ap.add_argument("--target-spec", default=DEFAULT_TARGET_SPEC)
    ap.add_argument(
        "--expected-pass-count",
        type=int,
        default=DEFAULT_EXPECTED_PASS_COUNT,
        help="pinned per-leg baseline; omit/None when no green state is pinned yet",
    )
    ap.add_argument("--tag", default="run")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--reuse-venv", action="store_true", help="skip venv (re)build")
    ap.add_argument("--log-out", default=None, help="write full pytest stdout here")
    args = ap.parse_args()

    scratch = Path(args.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    venv_dir = scratch / "venv"

    overrides = {}
    if args.leak_pythonpath is not None:
        overrides["PYTHONPATH"] = args.leak_pythonpath
    for kv in args.set_env:
        k, _, v = kv.partition("=")
        overrides[k] = v
    env = clean_env(overrides)

    base_ver = run([args.base_python, "--version"], capture_output=True, text=True)

    if args.no_venv:
        venv_python = Path(args.base_python)
        install_result = None
        pytest_install_result = None
        args.skip_tests = True
    else:
        if not args.reuse_venv:
            build_venv(venv_dir, args.base_python, args.system_site)
        venv_python = venv_dir / "bin" / "python"
        install_result = pip_install(venv_dir, args.wheelhouse, args.target_spec, env)
        pytest_install_result = pip_install(venv_dir, args.wheelhouse, "pytest", env)
        # A failed install is named here, not left to surface downstream as
        # an empty pytest stdout the summary parser can only refuse.
        for what, r in (("target", install_result), ("pytest", pytest_install_result)):
            if r.returncode != 0:
                log(r.stdout[-1500:], r.stderr[-1500:])
                raise SystemExit(
                    f"{what} install from wheelhouse {args.wheelhouse!r} failed "
                    f"rc={r.returncode} (index-free: the wheelhouse must carry "
                    f"every wheel the venv needs, pytest included)"
                )

    isolate = not args.no_isolate

    if args.checkout_rooted:
        test_cwd = REPO_ROOT
    else:
        test_cwd = sibling_free_copy(scratch / "sibling_free")

    canary = run_canary(venv_python, args.tag, isolate, env, str(test_cwd))
    assertions = evaluate_assertions(canary, venv_dir)

    pytest_result = None
    repo_local = None
    if not args.skip_tests:
        repo_local = enumerate_repo_local(venv_python, test_cwd, isolate, env)
        pytest_result = run_pytest(
            venv_python, test_cwd, args.marker_expr, isolate, env
        )
        if args.log_out:
            Path(args.log_out).write_text(
                "STDOUT:\n" + pytest_result.stdout +
                "\nSTDERR:\n" + pytest_result.stderr
            )

    record = {
        "leg": PACKAGE_NAME,
        "tag": args.tag,
        "base_python": args.base_python,
        "base_python_version": base_ver.stdout.strip() or base_ver.stderr.strip(),
        "venv_dir": str(venv_dir),
        "system_site": args.system_site,
        "isolate": isolate,
        "leak_pythonpath": args.leak_pythonpath,
        "extra_env": overrides,
        "checkout_rooted": args.checkout_rooted,
        "target_spec": args.target_spec,
        "marker_expr": args.marker_expr,
        "no_venv": args.no_venv,
        "install_rc": install_result.returncode if install_result else None,
        "install_stderr_tail": (
            install_result.stderr[-1500:] if install_result else None
        ),
        "pytest_install_rc": (
            pytest_install_result.returncode if pytest_install_result else None
        ),
        "canary": canary,
        "assertions": assertions,
        "pytest_cwd": str(test_cwd),
        "repo_local": repo_local,
        "pytest": (pytest_summary := summarize_pytest(pytest_result)),
        "verdict": compute_verdict(
            pytest_summary,
            args.expected_pass_count,
            {"repo_local_pin_ok": repo_local["pin_ok"]} if repo_local else {},
        ),
    }
    print(json.dumps(record, indent=2))

    # The verdict must reach the CALLER through the exit status, not only
    # through printed JSON -- a status a consumer cannot act on is
    # decorative (this is the same defect class the summary-line parser had,
    # one level up: rc=2 sat beside a printed "n_collect_errors: 0" that no
    # exit code carried forward). GREEN=0; RED=1; INCONSISTENT=2 (the
    # dangerous case -- must be non-zero); NOT_RUN=0 (a --skip-tests run,
    # e.g. a self-assertion demo, has no pytest verdict to fail on -- the
    # driver checks the `assertions` field directly for those).
    sys.exit({"GREEN": 0, "RED": 1, "INCONSISTENT": 2, "NOT_RUN": 0}.get(
        record["verdict"]["verdict"], 3
    ))


if __name__ == "__main__":
    main()
