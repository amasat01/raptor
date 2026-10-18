# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The publish/deploy workflows may only fetch, verify and upload."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "ci_publish_subset.py"

# reads the repository's workflows and tools/, which are not in the wheel
pytestmark = pytest.mark.repo_local


def _guard():
    spec = importlib.util.spec_from_file_location("ci_publish_subset", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_current_workflows_pass():
    assert _guard().main([]) == 0


@pytest.mark.parametrize("run", [
    "python -m build --outdir dist",
    "python -m pip install --quiet -e .",
    "pip install .",
    "python -m pytest -q",
    "make -C docs strict",
    "sphinx-build docs out",
])
def test_build_step_is_red(tmp_path, run):
    wf = tmp_path / "publish.yml"
    wf.write_text(pytest.importorskip("yaml").safe_dump({"jobs": {"j": {"steps": [{"name": "s", "run": run}]}}}))
    assert _guard().violations(wf)


def test_setup_python_is_red(tmp_path):
    wf = tmp_path / "docs.yml"
    wf.write_text(pytest.importorskip("yaml").safe_dump({"jobs": {"j": {"steps": [{"uses": "actions/setup-python@v7"}]}}}))
    assert _guard().violations(wf)
