"""Sphinx configuration for the swarp documentation site.

The site is prose-only — there is no autodoc — so nothing here imports ``swarp`` and the
build needs neither torch nor warp. That is what lets ``.github/workflows/docs.yml`` build
in seconds from ``uv sync --only-group docs``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]

project = "swarp"
author = "Davide De Benedittis"
copyright = "2025, Davide De Benedittis"  # noqa: A001
version = release = _PYPROJECT["version"]

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

# -- MyST ---------------------------------------------------------------------------
myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "substitution",
]
myst_heading_anchors = 3

# -- HTML ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# The RTD sidebar is dark, so it wants the light-marks variant of the logo.
html_logo = "../img/swarp_dark.svg"
html_favicon = "../img/swarp.svg"
html_title = f"swarp {version}"
html_show_sourcelink = False

html_theme_options = {
    # Show the wordmark under the glyph: the logo alone carries no "swarp" text.
    "logo_only": False,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_external_links": True,
}

html_context = {
    "display_github": True,
    "github_user": "ddebenedittis",
    "github_repo": "swarp",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- linkcheck ----------------------------------------------------------------------
# The Pages URL does not resolve until the first deployment, and GitHub 429s anonymous
# link checks often enough to make the builder flaky.
linkcheck_ignore = [
    r"https://ddebenedittis\.github\.io/swarp.*",
    r"https://github\.com/ddebenedittis/swarp.*",
]
