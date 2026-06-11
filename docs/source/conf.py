#  Python library for processing AQI data from the CHMI OpenData portal.
#  Copyright (C) 2026 chickendrop89

#  This library is free software; you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

project = "czech_air_quality"
copyright = "chickendrop89, 2025"
author = "chickendrop89"
release = "2.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    'sphinx_sitemap'
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    "typehints": "both",
}

autodoc_typehints = "both"
autodoc_typehints_format = "short"
typehints_fully_qualified = False
typehints_use_rtype = True

html_extra_path = ['robots.txt', 'custom_sitemap.xml']
html_baseurl = 'https://czech-air-quality.readthedocs.io/en/latest/'
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": ""
}

pygments_style = "sphinx"
language = "en"
