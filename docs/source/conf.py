# docs/conf.py

import os, sys
import glob
import shutil
import re
from pathlib import Path
import nbsphinx
import sphinx_rtd_theme
from importlib.metadata import PackageNotFoundError, version as pkg_version

#from sheap.RegionHandler.RegionBuilder import RegionBuilder


sys.path.insert(0, os.path.abspath('../..'))

project   = 'sheap'
author    = 'F. Avila-Vera'

def resolve_release():
    # On RTD, this is the tag/branch being built, e.g., "v0.1.2" or "latest"
    v = os.environ.get("READTHEDOCS_VERSION")
    vt = os.environ.get("READTHEDOCS_VERSION_TYPE")  # "tag" | "branch" | "unknown"
    if v and vt == "tag":
        return v.lstrip("v")
    # Fallback: use installed package version
    try:
        return pkg_version("sheap")
    except PackageNotFoundError:
        # Last resort: keep a sane default
        return "0.0.0"
release = resolve_release()
version = ".".join(release.split(".")[:2])
autodoc_inherit_docstrings = True

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    "nbsphinx",
    "sphinxcontrib.jquery",
    "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
#    "sphinx_gallery.gen_gallery",
#    "sphinx_search.extension",
]

# Treat notebooks and markdown via myst-nb
# source_suffix = {
#     '.rst':    'restructuredtext',
#     '.ipynb':  'myst-nb',
#     '.md':     'myst-nb',
# }

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    # …any others you like…
]

# Always run notebooks at build time
#nb_execution_mode = "force"
#nb_execution_timeout = 300
nb_execution_mode = "off"  # <- Don't execute notebooks

nbsphinx_execute = "never"  # <- Explicitly prevent nbsphinx from executing

autodoc_mock_imports = [
    "jax", "jax.numpy", "optax", "numpyro",
    "astropy", "pandas", "auto_uncertainties",
    # etc…
]
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_typehints   = 'description'
napoleon_google_docstring = True
autodoc_class_content = "both"   # include both class‐ and __init__‐docstrings

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','api/sheap.SuportData*', 'api/sheap.Core.Core.rst'
                    ,'**/Core/Core.*'
                    ,'api/sheap.SuportData.rst'
                    ,'**/SuportData/*']
templates_path   = ['_templates']


html_theme      = 'sphinx_rtd_theme'
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme_options = {
#     'collapse_navigation': False,
#     'navigation_depth':     4,
#     'sticky_navigation':    True,
# }

html_static_path = ['_static']
html_css_files = [
  'custom.css',
]
pygments_style = "sphinx"
add_module_names = False

nbsphinx_execute = "never"
language = "en"
master_doc = "index"
# For the getting started ##

# with open("../README.rst", "rt") as f:
#         lines = f.readlines()
#         for i, line in enumerate(lines):
#             if "#sheap" in line:
#                 break
#         #print(i,line) 
#         lines = lines[i:]
#         lines = ["Getting Started with sheap","=====\n"] + lines
#         text = "\n".join(lines)

# with open("getting_started.rst", "wt") as f:
#     f.write(text)
##########################
with open("../../README.rst", "rt") as f:
    lines = f.readlines()

# 1) Find where the first real section starts (title + underline)
start = 0
for i in range(len(lines) - 1):
    title = lines[i].rstrip("\n")
    underline = lines[i+1].rstrip("\n")
    if (
        len(underline) >= 3
        and set(underline) == {underline[0]}
        and len(underline) >= len(title)
        and underline[0] in ("=", "-", "~", "^")
    ):
        start = i
        break

# 2) Split out the body from that section onward
body_lines = lines[start:]

# 3) Prepare to build a new body with the first section swapped and all underlines → '-'
new_body = []
first_replaced = False

i = 0
while i < len(body_lines):
    line = body_lines[i]
    # If this is the first section title (underlined with '=')
    if (not first_replaced
        and i + 1 < len(body_lines)
        and re.match(r"=+", body_lines[i+1].rstrip("\n"))
    ):
        # Replace title + its underline
        new_title = "sheap (Spectral Handling and Estimation of AGN)\n"
        new_uline = "-" * (len(new_title.rstrip("\n"))) + "\n"
        new_body.append(new_title)
        new_body.append(new_uline)
        first_replaced = True
        i += 2
        continue

    # If this line *is* an underline of any repeated punctuation, swap to '-'
    if re.match(r"^([=~^\-])\1+$", line.rstrip("\n")):
        new_body.append("-" * len(line.rstrip("\n")) + "\n")
        i += 1
        continue

    # Otherwise, just copy the line
    new_body.append(line)
    i += 1

with open("../../README.rst", "rt") as f:
    lines = f.readlines()

# 1) Find where the first real section starts (title + underline)
start = 0
for i in range(len(lines) - 1):
    title = lines[i].rstrip("\n")
    underline = lines[i+1].rstrip("\n")
    if (
        len(underline) >= 3
        and set(underline) == {underline[0]}
        and len(underline) >= len(title)
        and underline[0] in ("=", "-", "~", "^")
    ):
        start = i
        break

# 2) Split out the body from that section onward
body_lines = lines[start:]

# 3) Prepare to build a new body with the first section swapped and all underlines → '-'
new_body = []
first_replaced = False

i = 0
while i < len(body_lines):
    line = body_lines[i]
    # If this is the first section title (underlined with '=')
    if (not first_replaced
        and i + 1 < len(body_lines)
        and re.match(r"=+", body_lines[i+1].rstrip("\n"))
    ):
        # Replace title + its underline
        new_title = "sheap (Spectral Handling and Estimation of AGN)\n"
        new_uline = "-" * (len(new_title.rstrip("\n"))) + "\n"
        new_body.append(new_title)
        new_body.append(new_uline)
        first_replaced = True
        i += 2
        continue

    # If this line *is* an underline of any repeated punctuation, swap to '-'
    if re.match(r"^([=~^\-])\1+$", line.rstrip("\n")):
        new_body.append("-" * len(line.rstrip("\n")) + "\n")
        i += 1
        continue

    # Otherwise, just copy the line
    new_body.append(line)
    i += 1

# 4) Prepend the “Getting Started with SHEAP” heading
hdr = "Getting Started with sheap\n" + "=" * len("Getting Started with sheap") + "\n\n"

with open("getting_started.rst", "wt") as f:
    f.write(hdr)
    f.write("".join(new_body))
    
######

if not os.path.exists("tutorials"):
    os.makedirs("tutorials")

for src_file in glob.glob("../../tutorials/*.ipynb"):
    dst_file = os.path.join("tutorials", src_file.split("/")[-1])
    shutil.copy(src_file, "tutorials/")

# add index file to `tutorials` path, `:orphan:` is used to
# tell sphinx that this rst file needs not to be appeared in toctree
with open("../../tutorials/index.rst", "rt") as f1:
    with open("tutorials/index.rst", "wt") as f2:
        #f2.write(":orphan:\n\n")
        f2.write(f1.read())

html_logo = "_static/sheap_nobck_logo.png"

# # logo
# html_logo = "_static/img/pyro_logo_wide.png"

# # logo
# html_favicon = "_static/img/favicon/favicon.ico"

# # The theme to use for HTML and HTML Help pages.  See the documentation for
# # a list of builtin themes.
# #
# html_theme = "sphinx_rtd_theme"

# # Theme options are theme-specific and customize the look and feel of a theme
# # further.  For a list of options available for each theme, see the
# # documentation.
# #
# # html_theme_options = {}

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
# html_style = "css/pyro.css"

# # Custom sidebar templates, must be a dictionary that maps document names
# # to template names.
# #
# # The default sidebars (for documents that don't match any pattern) are
# # defined by theme itself.  Builtin themes are using these templates by
# # default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# # 'searchbox.html']``.
# #
# # html_sidebars = {}


# # -- Options for HTMLHelp output ---------------------------------------------

# # Output file base name for HTML help builder.
# htmlhelp_basename = "numpyrodoc"


# # -- Options for LaTeX output ------------------------------------------------

# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     #
#     # 'papersize': 'letterpaper',
#     # The font size ('10pt', '11pt' or '12pt').
#     #
#     # 'pointsize': '10pt',
#     # Additional stuff for the LaTeX preamble.
#     #
#     "preamble": r"""
#         \usepackage{pmboxdraw}
#         \usepackage{alphabeta}
#         """,
#     # Latex figure (float) alignment
#     #
#     # 'figure_align': 'htbp',
# }

# # Grouping the document tree into LaTeX files. List of tuples
# # (source start file, target name, title,
# #  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     (master_doc, "NumPyro.tex", "NumPyro Documentation", "Uber AI Labs", "manual")
# ]

# # -- Options for manual page output ------------------------------------------

# # One entry per manual page. List of tuples
# # (source start file, name, description, authors, manual section).
# man_pages = [(master_doc, "NumPyro", "NumPyro Documentation", [author], 1)]

# # -- Options for Texinfo output ----------------------------------------------

# # Grouping the document tree into Texinfo files. List of tuples
# # (source start file, target name, title, author,
# #  dir menu entry, description, category)
# texinfo_documents = [
#     (
#         master_doc,
#         "NumPyro",
#         "NumPyro Documentation",
#         author,
#         "NumPyro",
#         "Pyro PPL on Numpy",
#         "Miscellaneous",
#     )
# ]


# # -- Extension configuration -------------------------------------------------

# # -- Options for intersphinx extension ---------------------------------------

# # Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3/", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "jax": ("https://jax.readthedocs.io/en/latest/", None),
#     "pyro": ("https://docs.pyro.ai/en/stable/", None),
# }


# # -- Suppress warnings in Sphinx 7.3.5

# suppress_warnings = ["config.cache"]