# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Minimal makefile for the raptor Sphinx (MyST/Markdown) documentation.
# raptor is pure Python — there is no Doxygen/breathe step.

SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build
NBROOTS       = content/tutorials content/examples

.PHONY: help Makefile clean livehtml strict linkcheck nbexec nbcheck nbtest

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR) content/api/generated

# Build with warnings promoted to errors, but keep going so a single run
# reports every warning instead of stopping at the first.
strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS) $(O)

# Check every internal + external link. Named explicitly (rather than left to
# the catch-all below) so it is documented and discoverable the same way the
# sibling repos' `make linkcheck` is.
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

# Execute every tutorial/example notebook IN PLACE (outputs kept) — the local,
# pre-commit gate. CI never calls this; it only renders the committed outputs.
nbexec:
	python _scripts/nb_execute.py $(NBROOTS)

# Refuse to proceed unless every notebook under NBROOTS was actually executed
# (every non-blank code cell carries an execution count).
nbcheck:
	python _scripts/nb_check_filled.py $(NBROOTS)

# Execute every tutorial/example notebook and fail on any error, via nbval —
# a lighter-weight standalone check than nbexec (does not rewrite outputs).
nbtest:
	python -m pytest --nbval-lax -q $(NBROOTS)

# Watch for changes and auto-recompile (requires watchdog: pip install watchdog)
livehtml: Makefile
	watchmedo shell-command -p "*.rst;*.md;*.py" -R \
	    -c "make html" \
	    -i "$(BUILDDIR)/*" \
	    --debug-force-polling .

# Catch-all target: route unknown targets to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
