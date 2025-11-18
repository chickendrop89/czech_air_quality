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

"""
Provides a python client for simply retrieving 
and processing air quality data from the CHMI OpenData portal.
"""

__version__ = "1.0.1"

from .airquality import (
    AirQuality,
    AirQualityError,
    DataDownloadError,
    StationNotFoundError,
    PollutantNotReportedError,
)

__all__ = [
    "AirQuality",
    "AirQualityError",
    "DataDownloadError",
    "StationNotFoundError",
    "PollutantNotReportedError",
    "__version__",
]
