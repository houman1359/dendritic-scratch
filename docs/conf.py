import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
release = '0.1.0'



# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dendritic_modeling'
copyright = '2024, Research and Engineering at Kempner'
author = 'Research and Engineering at Kempner'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    'sphinx.ext.autodoc',     # Automatically document from docstrings
    'sphinx.ext.napoleon',    # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',    # Add links to highlighted source code
    'sphinx.ext.autosummary', # Generate summary tables for modules/classes
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "nbsphinx",
    'myst_parser',            # Support for Markdown files (if using Markdown)  
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.githubpages', 
    'sphinx_copybutton',
    'sphinxcontrib.bibtex'
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "smartquotes",
    "linkify",
    "substitution",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None

bibtex_bibfiles = ['refs.bib']


# Enable syntax highlighting
pygments_style = 'default'  # You can also use 'default' or other pygments styles


# Figure numbers
numfig = True

numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' 
html_static_path = ['_static']
