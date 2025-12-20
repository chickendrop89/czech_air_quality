#  Provides a python client for simply retrieving
#  and processing air quality data from the CHMI OpenData portal.
#  Copyright (C) 2025 chickendrop89

#  This library is free software; you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.

# Need to use _warn() in this library
# pylint: disable=wrong-import-position

"""
Provides a python client for simply retrieving 
and processing air quality data from the CHMI OpenData portal.
"""

import warnings
from importlib.metadata import version

def _warn(message: str) -> None:
    """
    Wrapper for python warnings generated
    by this library for the developer
    """
    warnings.warn(
        message,
        category=RuntimeWarning,
        stacklevel=2
    )

__version__ = version("czech_air_quality")

from .exceptions import (
    AirQualityError,
    DataDownloadError,
    StationNotFoundError,
    PollutantNotReportedError,
    CacheError
)

from .api import (
    AirQuality
)

__all__ = [
    "AirQuality",
    "AirQualityError",
    "DataDownloadError",
    "StationNotFoundError",
    "PollutantNotReportedError",
    "CacheError",
    "__version__",
]
