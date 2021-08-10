# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import sphinx
from pathlib import Path
from importlib import metadata

DOCS_DIR = Path(".").resolve()
SRC_DIR = Path("../src").resolve()
sys.path.insert(0, SRC_DIR)


# -- Project information -----------------------------------------------------

project = 'Flax-extra'
copyright = '2021, Andrei Nesterov'
author = 'Andrei Nesterov'
version = metadata.version(project.lower())


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Setup for Sphinx API Docs -----------------------------------------------
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    main([
        "-f",
        "--separate",
        "--module-first",
        "-d", "1",
        "-H", project,
        "-A", author,
        "-V", version,
        "-o", str(DOCS_DIR / "_apidoc"),
        str(SRC_DIR / project.replace("-","_").lower())
    ])

def setup(app):
    app.connect('builder-inited', run_apidoc)
