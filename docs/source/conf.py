# docs/source/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'telemetry-anomdet'
author = 'Julio Anzaldo'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']