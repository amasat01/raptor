# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The docs-fresh generators say which failure it is, by exit code.

0 fresh, 1 stale, 3 a card the docs cite is missing, 4 the eagle checkout or its
claim engine is missing, 2 usage only. Each case runs a temporary copy of the
generator and README. The eagle-backed cases need ``RAPTOR_EAGLE_ROOT`` and skip
without it. A grep test keeps the workflows from reporting every failure as
"out of date".
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parent.parent
if not (_REPO / "tools" / "sync_readme_perf.py").is_file():  # the suite also runs from a copy of tests/ outside the repository
    pytest.skip("needs the repository's tools/", allow_module_level=True)
_EAGLE = os.environ.get("RAPTOR_EAGLE_ROOT")


def _run(script, *args, cwd, env=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True,
                          text=True, cwd=cwd, env=env)


@pytest.fixture()
def copy(tmp_path):
    """A copy of the generators, cards and README."""
    root = tmp_path / "raptor"
    shutil.copytree(_REPO / "benchmarks", root / "benchmarks",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "tools").mkdir()
    shutil.copy2(_REPO / "tools" / "sync_readme_perf.py", root / "tools")
    shutil.copy2(_REPO / "README.md", root)
    return root


@pytest.fixture()
def eagle(tmp_path):
    if not _EAGLE or not (pathlib.Path(_EAGLE) / "tools" / "prose_claims.py").is_file():
        pytest.skip("needs RAPTOR_EAGLE_ROOT pointing at an eagle checkout")
    root = tmp_path / "eagle"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(pathlib.Path(_EAGLE) / "tools" / "prose_claims.py", root / "tools")
    shutil.copytree(pathlib.Path(_EAGLE) / "benchmarks", root / "benchmarks",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return root


def _perf(copy, *args):
    return _run(copy / "tools" / "sync_readme_perf.py", "--check", *args, cwd=copy)


def _cards(copy, *args):
    return _run(copy / "benchmarks" / "sync_readme_cards.py", "--check", *args, cwd=copy)


# README's own cards block (benchmarks/sync_readme_cards.py)
def test_cards_block_fresh_exits_0(copy):
    assert _cards(copy).returncode == 0


def test_cards_block_stale_exits_1_and_names_the_file(copy):
    readme = copy / "README.md"
    text = readme.read_text()
    i = text.index("<!-- cards:begin -->") + len("<!-- cards:begin -->")
    readme.write_text(text[:i] + "\nhand edit\n" + text[text.index("<!-- cards:end -->"):])
    res = _cards(copy)
    assert res.returncode == 1
    assert "README.md" in res.stderr


def test_cards_block_missing_card_exits_3_and_names_it(copy):
    (copy / "benchmarks" / "zero_copy_interop_card.json").unlink()
    res = _cards(copy)
    assert res.returncode == 3
    assert "zero_copy_interop_card.json" in res.stderr


def test_cards_block_usage_error_exits_2(copy):
    assert _cards(copy, "--no-such-flag").returncode == 2


# README's headline (tools/sync_readme_perf.py)
def test_headline_fresh_exits_0(copy, eagle):
    assert _perf(copy, "--eagle-root", str(eagle)).returncode == 0


def test_headline_stale_exits_1_and_names_the_file(copy, eagle):
    readme = copy / "README.md"
    text = readme.read_text()
    i = text.index("<!-- hero:begin -->\n") + len("<!-- hero:begin -->\n")
    readme.write_text(text[:i] + "hand edit" + text[text.index("\n<!-- hero:end -->"):])
    res = _perf(copy, "--eagle-root", str(eagle))
    assert res.returncode == 1
    assert "README.md" in res.stdout


def test_headline_missing_card_exits_3_and_names_it(copy, eagle):
    (eagle / "benchmarks" / "perf_card" / "card_nvidia-h100-80gb-hbm3.json").unlink()
    res = _perf(copy, "--eagle-root", str(eagle))
    assert res.returncode == 3
    assert "benchmarks/perf_card/card_" in res.stderr


def test_headline_eagle_without_claim_engine_exits_4_and_names_the_path(copy, tmp_path):
    bare = tmp_path / "bare_eagle"
    bare.mkdir()
    res = _perf(copy, "--eagle-root", str(bare))
    assert res.returncode == 4
    assert "prose_claims.py" in res.stderr


def test_headline_without_eagle_root_exits_4(copy):
    env = {k: v for k, v in os.environ.items() if k != "RAPTOR_EAGLE_ROOT"}
    res = _run(copy / "tools" / "sync_readme_perf.py", "--check", cwd=copy, env=env)
    assert res.returncode == 4
    assert "--eagle-root" in res.stderr


def test_headline_usage_error_exits_2(copy):
    assert _perf(copy, "--no-such-flag").returncode == 2


def test_workflows_say_out_of_date_only_in_the_stale_branch():
    for wf in sorted((_REPO / ".github" / "workflows").glob("*.yml")):
        assert "out of date" not in wf.read_text(), wf.name
    for line in (_REPO / "tools" / "docs_fresh.sh").read_text().splitlines():
        if "out of date" in line:
            assert line.lstrip().startswith("1)"), line
