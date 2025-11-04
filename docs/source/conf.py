# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'telemetry_anomdet'
copyright = '2025, Julio Anzaldo'
author = 'Julio Anzaldo'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # automatically document code
    'sphinx.ext.napoleon',    # supports Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',    # links to source
    'sphinx.ext.autosummary',
]

autodoc_member_order = 'bysource'
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

root_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_baseurl = "https://julioanzaldo.github.io/telemetry-anomdet/"