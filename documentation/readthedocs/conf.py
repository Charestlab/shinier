from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "documentation" / "readthedocs" / "_build" / "matplotlib"))

project = "SHINIER"
author = "Nicolas Dupuis-Roy and Mathias Salvas-Hebert"
copyright = "2026, Nicolas Dupuis-Roy and Mathias Salvas-Hebert"

try:
    from shinier import __version__
except Exception:
    __version__ = "0.2.0"

version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "includehidden": True,
}

autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_class_signature = "separated"
autodoc_typehints = "none"
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_ivar = False
napoleon_custom_sections = ["Runtime Attributes"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3
suppress_warnings = [
    "myst.xref_missing",
    "misc.highlighting_failure",
]
