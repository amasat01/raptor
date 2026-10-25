# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for building the raptor documentation."""

import os
import sys

# -- Make raptor importable for autosummary + notebook execution.
sys.path.insert(0, os.path.abspath('..'))            # the `raptor` package root

# -- Project information -----------------------------------------------------

project = 'raptor'
copyright = '2026, Alessandro Masat'
author = 'Alessandro Masat'

# The version is the latest git tag (setuptools-scm), never written into the tree.
import setuptools_scm

release = version = setuptools_scm.get_version(root="..", relative_to=__file__, fallback_version="0.0.0")

# -- General configuration ---------------------------------------------------

# raptor has no C++ to document, so there is deliberately no breathe / doxygen
# here. myst_nb pulls in MyST and adds executable Jupyter notebooks; autodoc +
# autosummary generate the Python API reference from the installed package.
extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'raptor_docs.sphinx',
]

# MyST options — colon fences + deflists, plus dollar/amsmath for the little
# math the tutorials carry ($a = -\mu r/|r|^3$). The PTX-heavy essays keep their
# '$' tokens inside fenced code blocks, which MyST never parses as math.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

# -- Notebook execution (myst_nb) --------------------------------------------
# Tutorials are executed LOCALLY (`make nbexec`) and committed WITH their outputs. CI has
# no CUDA toolchain and never installs hawk/eagle (raptor itself has zero
# hard dependencies — see the GitHub Actions `ci.yml`/`docs.yml` jobs), so it
# must never attempt to execute them: `nb_execution_mode = "off"` there,
# rendering exactly the committed outputs. Locally, "auto" re-executes a
# notebook myst-nb considers incomplete (see `make nbexec`/`make nbcheck`,
# the actual gates — this is a convenience for `make html` during authoring,
# not the thing that enforces freshness).
nb_execution_mode = "off" if os.environ.get("CI") else "auto"
nb_execution_raise_on_error = True
nb_execution_timeout = 3600
nb_merge_streams = True

# -- Python API reference (autosummary/autodoc) ------------------------------
autosummary_generate = True
autosummary_imported_members = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# myst_nb registers .md (MyST) and .ipynb (executable notebooks) on its own; .rst
# stays the Sphinx default. We deliberately leave source_suffix unset so
# autosummary writes only .rst stubs (an explicit .md mapping makes it emit
# duplicate .md + .rst stubs for every API entry).

# Templates and exclusions
templates_path = ['_templates']
exclude_patterns = ['_templates', '_build', 'Thumbs.db', '.DS_Store', 'README.md']

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
# The skin (font, raptor-*.css, favicon links) comes from raptor_docs.sphinx; this site only sets its accent and page styles.
html_css_files = ['site-accent.css', 'brand.css']

html_theme_options = {
    "repository_url": "https://github.com/amasat01/raptor",
    # One transparent logo file works on light AND dark pages (RAPTOR brand kit).
    "logo": {
        "image_light": "_static/brand/family_raptor.svg",
        "image_dark": "_static/brand/family_raptor.svg",
        "alt_text": "RAPTOR",
    },
}

# -----------------------------------------------------------------------------
# linkcheck
# -----------------------------------------------------------------------------
linkcheck_ignore = []

html_title = f"raptor — one manifest, any engine ({version})"
