# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for building the raptor documentation."""

import os
import re
import sys

# -- Make raptor importable for autosummary + notebook execution.
sys.path.insert(0, os.path.abspath('..'))            # the `raptor` package root

# -- Project information -----------------------------------------------------

project = 'raptor'
copyright = '2026, Alessandro Masat'
author = 'Alessandro Masat'

# Read version from pyproject.toml (the single source of truth). raptor is a
# pure-Python package, so there is no CMakeLists.txt to read (unlike the CUDA
# libraries in the ecosystem); fall back to a hardcoded 0.2.0 if it can't be read.
_version = "0.2.0"
_pyproject = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
_pattern = re.compile(r'^\s*version\s*=\s*["\'](\d+\.\d+\.\d+)["\']')
try:
    with open(_pyproject) as _f:
        for _line in _f:
            _m = _pattern.match(_line)
            if _m:
                _version = _m.group(1)
                break
except FileNotFoundError:
    pass

version = _version
release = _version

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
# Tutorials/examples are executed LOCALLY (`make nbexec`, on a GPU where the
# example notebooks need hawk+eagle) and committed WITH their outputs. CI has
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
# raptor-tokens.css (kit) -> site-accent.css (this site's --accent) ->
# raptor-theme.css (shared skin, derives everything from --accent) ->
# raptor-reveal.css (kit) -> brand.css (skeleton rules that CONSUME the vars
# raptor-theme.css sets; no longer sets them itself, see its own header).
html_css_files = ['raptor-tokens.css', 'site-accent.css', 'raptor-theme.css',
                   'raptor-reveal.css', 'brand.css']

html_theme_options = {
    "repository_url": "https://github.com/amasat01/raptor",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "collapse_navigation": True,
    "navigation_with_keys": True,
    # Colab + download launch buttons on notebook pages (no Binder: not configured).
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "classic",
    },
    # One transparent logo file works on light AND dark pages (RAPTOR brand kit).
    "logo": {
        "image_light": "_static/brand/family_raptor.svg",
        "image_dark": "_static/brand/family_raptor.svg",
        "alt_text": "RAPTOR",
    },
    # Harmonised code themes for light/dark (pydata-sphinx-theme keys).
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    # Family frame: purple "part of RAPTOR" chip in the footer, every site
    # (raptor-theme.css's .raptor-family-chip; theme's own extension point).
    "extra_footer": (
        '<div class="raptor-family-chip">part of '
        '<a href="https://amasat01.github.io/">RAPTOR</a></div>'
    ),
}

# RAPTOR brand: favicons + home-screen icon. Leave html_favicon unset: the
# hook below writes the icon links itself (SVG where supported, favicon.ico
# for Safari/older tools, 180 px icon for iOS home screens).
_RAPTOR_ICONS = [
    ("icon", "brand/favicon.ico", 'sizes="any"'),
    ("icon", "brand/favicon.svg", 'type="image/svg+xml"'),
    ("apple-touch-icon", "brand/app_icon_180.png", ""),
]


def _raptor_icons(app, pagename, templatename, context, doctree):
    pathto = context.get("pathto")
    if pathto is None:
        return
    links = "".join(
        f'<link rel="{rel}" href="{pathto("_static/" + path, 1)}" {extra}>\n'
        for rel, path, extra in _RAPTOR_ICONS
    )
    context["metatags"] = context.get("metatags", "") + links


def setup(app):
    app.connect("html-page-context", _raptor_icons)

# -----------------------------------------------------------------------------
# linkcheck
# -----------------------------------------------------------------------------
linkcheck_ignore = []

html_title = f"raptor — one manifest, any engine ({version})"
