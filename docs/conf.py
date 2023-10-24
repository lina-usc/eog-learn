import sys
import os

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EOGLearn"
copyright = "2023, Scott Huberty"
author = "Scott Huberty"
release = "0.1"


# Point Sphinx.ext.autodoc to the our python modules
# In this case, 1 parent directory from this dir
sys.path.insert(0, os.path.abspath("../"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx_design",
    "sphinxemoji.sphinxemoji",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]

# Enable numref for automatically numbering Figures, i.e "Fig 1"
numfig = True

# NumPyDoc configuration -----------------------------------------------------
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = {}
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_validate = True
# Only generate documentation for public members
autodoc_default_flags = ["members", "undoc-members", "inherited-members"]
