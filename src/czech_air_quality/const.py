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
Data constants of the library.
"""

from . import __version__

###########################

AQ_DATA_URL  = "https://opendata.chmi.cz/air_quality/now/data/airquality_1h_avg_CZ.csv"
METADATA_URL = "https://opendata.chmi.cz/air_quality/now/metadata/metadata.json"

ETAG_URLS = {
    "aq_data_etag": AQ_DATA_URL,
    "metadata_etag": METADATA_URL
}

###########################

CACHE_DIR_NAME = "czech_air_quality"
CACHE_FILE_NAME = "airquality_opendata_cache.json"
CACHE_METADATA_KEY = "__cache_metadata__"

TIMESTAMP_KEY = "timestamp"
ETAGS_KEY = "etags"

###########################

USER_AGENT = f"python-czech_air_quality/{__version__}"
REQUEST_HEADERS = {
     "User-Agent": USER_AGENT,
     "Accept": "text/csv, application/json, application/octet-stream"
}
REQUEST_TIMEOUT = 20
NOMINATIM_TIMEOUT = 10

###########################

# Threshold for pollutant values considered as Error/N/A
# Value that is X or above will be accepted.
# If a station reports "1.0" value for a pollutant, it is likely an error.
CHMI_ERROR_THRESHOLD = 1.1

# Maximum number of neighbour stations to consider merging 
# pollutants from
CHMI_NEIGHBOUR_LIMIT = 20

###########################

EAQI_LEVELS = {
    0: "Error/N/A",
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
    6: "Extremely Poor",
}

EAQI_BANDS = {
    # Concentration breakpoints (µg/m³) and the corresponding EAQI level (1-6)
    # Format: (EAQI_Level, Upper_Concentration_Limit)
    "PM10": [
        (1, 20),    # Good: 0-20
        (2, 40),    # Fair: 20-40
        (3, 50),    # Moderate: 40-50
        (4, 100),   # Poor: 50-100
        (5, 150),   # Very Poor: 100-150
        (6, float("inf"))  # Extremely Poor: ≥150
    ],
    "PM2_5": [
        (1, 10),    # Good: 0-10
        (2, 20),    # Fair: 10-20
        (3, 25),    # Moderate: 20-25
        (4, 50),    # Poor: 25-50
        (5, 75),    # Very Poor: 50-75
        (6, float("inf"))  # Extremely Poor: ≥75
    ],
    "O3": [
        (1, 50),    # Good: 0-50
        (2, 100),   # Fair: 50-100
        (3, 130),   # Moderate: 100-130
        (4, 240),   # Poor: 130-240
        (5, 380),   # Very Poor: 240-380
        (6, float("inf"))  # Extremely Poor: ≥380
    ],
    "NO2": [
        (1, 40),    # Good: 0-40
        (2, 90),    # Fair: 40-90
        (3, 120),   # Moderate: 90-120
        (4, 230),   # Poor: 120-230
        (5, 340),   # Very Poor: 230-340
        (6, float("inf"))  # Extremely Poor: ≥340
    ],
    "SO2": [
        (1, 100),   # Good: 0-100
        (2, 200),   # Fair: 100-200
        (3, 350),   # Moderate: 200-350
        (4, 500),   # Poor: 350-500
        (5, 750),   # Very Poor: 500-750
        (6, float("inf"))  # Extremely Poor: ≥750
    ],
}
