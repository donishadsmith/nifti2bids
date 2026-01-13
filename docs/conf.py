# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinxext"))

from github_link import make_linkcode_resolve

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nifti2bids"
copyright = "2025, nifti2bids Developers"
author = "Donisha Smith"

import nifti2bids

# The full version, including alpha/beta/rc tags
release = nifti2bids.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# Generate the API documentation when building
autosummary_generate = True
autodoc_default_options = {"members": False, "inherited-members": False}
numpydoc_show_class_members = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "none"

# Remove module name in signature
add_module_names = True

napoleon_google_docstring = False
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_include_private_with_doc = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "nifti2bids",
    "https://github.com/donishadsmith/nifti2bids/blob/{revision}/{package}/{path}#L{lineno}",
)
