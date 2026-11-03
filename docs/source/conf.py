# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
sys.path.insert(0, os.path.abspath('../../src'))

project = 'telemetry_anomdet'
copyright = '2026, Julio Anzaldo'
author = 'Julio Anzaldo'

# Per-page meta descriptions and link preview text are derived from each page's
# own opening paragraph by sphinxext-opengraph, so there is no site-wide
# description to maintain here.

# Read from the installed package so this never drifts from pyproject.toml.
try:
    release = _pkg_version('telemetry-anomdet')
except PackageNotFoundError:  # docs built without installing the project
    release = '0.0.0'

# -- Versioned builds ---------------------------------------------------------
#
# One source tree produces every version of the site. TAD_DOCS_SLOT names the
# subdirectory this build is deployed into:
#
#   unset   -> the site root, which is always the current release
#   dev     -> /dev/, built from the dev branch
#   v0.2.0  -> /v0.2.0/, an archived release built from its tag
#
# The root keeps its existing URLs, so llms.txt, robots.txt, the sitemap, the
# README badge and CITATION.cff all stay correct without edits. Only the
# archives live at new paths.
# Unset means a local or PR build: no switcher, and no network fetch of the
# version list. Set it to "root" for the release build that lands at the site
# root, or to the subdirectory name for anything else.
_SITE_ROOT = 'https://julioanzaldo.github.io/telemetry-anomdet/'
_SLOT_RAW = os.environ.get('TAD_DOCS_SLOT', '').strip('/')
_VERSIONED = bool(_SLOT_RAW)
_SLOT = '' if _SLOT_RAW in ('', 'root') else _SLOT_RAW
_SLOT_PATH = f'{_SLOT}/' if _SLOT else ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # automatically document code
    'sphinx.ext.napoleon',    # supports Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',    # links to source
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',  # link types to the upstream project's docs
    'sphinx_sitemap',          # sitemap.xml for crawlers
    'sphinxext.opengraph',     # link preview cards
    'notfound.extension',      # a 404 page that works at any URL depth
]

autodoc_member_order = 'bysource'
autosummary_generate = True

# -- Cross-project links ------------------------------------------------------
# Turns np.ndarray, pd.DataFrame and friends in signatures into links to the
# upstream docs. Fetched at build time, so keep the timeout short: a slow or
# unreachable inventory should warn and move on, not stall CI.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}
intersphinx_timeout = 10

templates_path = ['_templates']
exclude_patterns = []

root_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['llm_copy.js']
html_baseurl = f"{_SITE_ROOT}{_SLOT_PATH}"

# Copy extra files (e.g. llms.txt) verbatim to the built site root.
html_extra_path = ['_extra']

# Navbar brand text. Without this the theme uses "<project> <release>
# documentation", which wraps in the corner.
html_title = 'telemetry_anomdet'
html_short_title = 'telemetry_anomdet'

html_theme_options = {
    # Top navigation. Sections come from the toctree captions in index.rst;
    # anything past the first few collapses into a "More" dropdown rather than
    # wrapping the bar.
    'header_links_before_dropdown': 4,
    'navbar_align': 'left',
    'show_nav_level': 1,
    # Right-hand "On this page" list, two levels deep so long guides are
    # navigable without scrolling to find the section headings.
    'show_toc_level': 2,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/JulioAnzaldo/telemetry-anomdet',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/telemetry-anomdet/',
            'icon': 'fa-solid fa-box',
        },
    ],
    'use_edit_page_button': True,
    'footer_start': ['copyright'],
    'footer_end': ['theme-version'],
}

if _VERSIONED:
    # The switcher list is always read from the root copy, never from this
    # build's own. An archived version then offers every release published
    # after it, instead of freezing the list as it stood on its release day.
    #
    # Only on deploy builds: the theme fetches this URL while building, so
    # wiring it up unconditionally would make every local build reach the
    # network and warn when it cannot.
    html_theme_options['switcher'] = {
        'json_url': f'{_SITE_ROOT}switcher.json',
        'version_match': 'dev' if _SLOT == 'dev' else release,
    }
    html_theme_options['navbar_start'] = ['navbar-logo', 'version-switcher']
    # Tells a reader on an archived or dev page that they are not on the
    # current release, which is the whole point of publishing older versions.
    html_theme_options['show_version_warning_banner'] = True

html_context = {
    'github_user': 'JulioAnzaldo',
    'github_repo': 'telemetry-anomdet',
    'github_version': 'main',
    'doc_path': 'docs/source',
}

# -- Discoverability ----------------------------------------------------------

# sitemap.xml. Single-version site, so no language or version in the URLs.
sitemap_url_scheme = '{link}'

# Keep the sitemap to pages worth landing on, and in agreement with robots.txt:
# advertising a URL that robots.txt disallows is reported as a conflict. The
# viewcode listings and the index pages carry no content of their own.
sitemap_excludes = [
    '404.html',
    'search.html',
    'genindex.html',
    'py-modindex.html',
    '_modules/*',
]

# Link preview cards, for when a page is pasted into chat or a social post.
# opengraph derives a per-page description from the page's own text and emits
# the <meta name="description"> tag itself, so do not add a second one here.
ogp_site_url = html_baseurl
ogp_site_name = 'telemetry-anomdet'
ogp_description_length = 200
ogp_type = 'website'
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary" />',
]

# 404 page. GitHub Pages serves /404.html for any missing path, so the page's
# own asset links must be absolute rather than relative to where it was built.
notfound_urls_prefix = f'/telemetry-anomdet/{_SLOT_PATH}'