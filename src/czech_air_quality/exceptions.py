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

"""
Exceptions of the library.
"""

class AirQualityError(Exception):
    """Base exception for the czech_air_quality library."""

class StationNotFoundError(AirQualityError):
    """Raised when a city or station cannot be found."""

class PollutantNotReportedError(AirQualityError):
    """Raised when the nearest station does not report data for the requested pollutant."""

class DataDownloadError(AirQualityError):
    """Raised when data cannot be downloaded or is invalid."""

class CacheError(AirQualityError):
    """Raised when there is an error related to caching data."""
