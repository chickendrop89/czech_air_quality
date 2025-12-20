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
Public API.
"""

import logging
from datetime import datetime
from functools import wraps

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from .processing import AirQualityCalculations
from . import const, _warn
from . import data_manager
from . import (
    StationNotFoundError,
    PollutantNotReportedError,
    DataDownloadError
)

_LOGGER = logging.getLogger(__name__)


class AirQuality(AirQualityCalculations):
    """
    A client for retrieving air quality data from CHMI.
    """
    def __init__(
        self,
        auto_load=True,
        region_filter=None,
        use_nominatim=True,
        neighbour_station_limit=const.CHMI_NEIGHBOUR_LIMIT,
        nominatim_timeout=const.NOMINATIM_TIMEOUT,
        request_timeout=const.REQUEST_TIMEOUT,
        disable_caching=False
    ):
        """
        Initialize the Air Quality client.

        :param auto_load: If True, load/download data immediately during initialization.
                         If False, data loads on first method call.
        :type auto_load: bool
        :param region_filter: Limit stations to specific region (case-insensitive).
        :type region_filter: str, optional
        :param use_nominatim: If True, enable Nominatim geocoding for city name lookups.
                             If False, only exact station name matches are accepted.
        :type use_nominatim: bool
        :param neighbour_station_limit: Maximum number of nearby stations to check when 
                                        searching for pollutant data. Useful
                                        with use_nominatim=True to find data if nearest
                                        station doesn't report requested pollutant.
        :type neighbour_station_limit: int
        :param nominatim_timeout: Timeout in seconds for Nominatim geocoding requests
        :type nominatim_timeout: int
        :param request_timeout: Timeout in seconds for CHMI data download requests
        :type request_timeout: int
        :param disable_caching: If True, skip all caching and always download fresh data
        :type disable_caching: bool

        Caching Strategy:

        1. Cache Hit (Recent): If cached data is < 20 minutes old, use it immediately
        2. ETag Validation: If cache is > 20 minutes old, perform HTTP-HEAD request with ETag
           - If server returns 304 (Not Modified), trust cache for another 20 minutes
           - If server returns 200 (Modified), download full fresh data
        3. Network Error: If network unavailable but cache exists, use stale cache with warning
        4. Force Refresh: `force_fetch_fresh()` bypasses age check but respects ETags
        """
        super().__init__()

        self._region_filter = region_filter.lower() if region_filter else None
        self._data = {}
        self._all_stations = []
        self._component_lookup = {}
        self._id_registration_to_component = {}
        self._locality_code_to_station = {}
        self._city_coordinate_cache = {}
        self._data_manager = data_manager.DataManager(
            disable_caching=disable_caching,
            request_timeout=request_timeout
        )

        self._use_nominatim = use_nominatim
        self._neighbour_station_limit = neighbour_station_limit
        self._nominatim_timeout = nominatim_timeout

        if self._use_nominatim:
            self._geolocator = Nominatim(
                user_agent=const.USER_AGENT
            )
            self._rate_limited_geocode = RateLimiter(
                self._geolocator.geocode,
                min_delay_seconds=1.0,
                max_retries=0
            )
        else:
            self._geolocator = None
            self._rate_limited_geocode = None

        if auto_load:
            self._data_manager.ensure_latest_data()
            self._load_and_parse_data()


    @classmethod
    def get_all_station_names(cls) -> list[str | None]:
        """
        Get all known air quality station names by creating temporary a client instance.

        :return: List of station names, or empty list if data cannot be retrieved
        :rtype: ``list[str | None]``
        """
        try:
            temp_instance = cls()
            return [station["Name"] for station in temp_instance.all_stations]
        except (DataDownloadError, StationNotFoundError) as e:
            _warn(f"Failed to get all station names: {e}")
            return []

    @staticmethod
    def _ensure_loaded(func):
        """
        Ensure data is fresh and loaded before executing a public method.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._data_manager.raw_data_json:
                self._data_manager.ensure_latest_data()
                self._load_and_parse_data()
            return func(self, *args, **kwargs)

        return wrapper


    @property
    def actualized_time(self) -> datetime:
        """
        Timestamp when data was last updated by the CHMI source.

        :rtype: ``datetime``
        """
        return self._data_manager.actualized_time

    @property
    def is_data_fresh(self) -> bool:
        """
        Check if cached data is still valid via ETag validation.

        :return: ``True`` if cached data is current; ``False`` if needs refresh
        :rtype: ``bool``
        """
        return self._data_manager.is_data_fresh()

    @property
    def all_stations(self) -> list[dict]:
        """
        Get all available air quality stations.

        :return: List of station dictionaries, filtered by region if set
        :rtype: ``list[dict]``
        """
        return self._all_stations

    @property
    def component_lookup(self) -> dict[str, tuple[str, str, str]]:
        """
        Map of pollutant codes to (code, name, unit) tuples.

        :return: Dictionary with pollutant code as key
        :rtype: ``dict[str, tuple[str, str, str]]``
        """
        return self._component_lookup

    @property
    def raw_data(self) -> dict:
        """
        Raw parsed data from the JSON source.

        :return: Dictionary containing localities and measurements
        :rtype: ``dict``
        """
        return self._data


    @_ensure_loaded
    def find_nearest_station(self, city_name: str) -> tuple[dict, float]:
        """
        Find air quality station nearest to a city.

        If ``use_nominatim=True``, geocodes the city name to coordinates and
        calculates distances to all stations. Otherwise, matches exact station names only 
        (e.g. "Prague - Letná").

        :param city_name: Name of the city or exact station name
        :type city_name: str

        :return: Tuple of (``station_dict``, ``distance_km``) with station metadata and distance
        :rtype: ``tuple[dict, float]``

        :raises StationNotFoundError: If city/station not found or no nearby stations exist
        """
        return self._get_nearest_station_to_city(city_name)


    @_ensure_loaded
    def get_air_quality_report(self, city_name: str) -> dict:
        """
        Get comprehensive air quality report with EAQI (European Air Quality Index) for a city.

        :param city_name: City name to search for
        :type city_name: str
        :return: Air quality report dictionary with keys:

            - **city_searched (str)**: Original search term
            - **station_name (str)**: Name of station providing data
            - **station_code (str)**: Station locality code
            - **region (str)**: Region name
            - **distance_km (str)**: Distance from city to station in kilometers
            - **air_quality_index_code (int)**: EAQI level (0-6, 0 if no data)
            - **air_quality_index_description (str)**: Human description (e.g., 'Good', 'Poor')
            - **actualized_time_utc (str)**: ISO format UTC timestamp of data
            - **measurements (list[dict])**: List of pollutant measurements:

              - **pollutant_code (str)**: Code like 'PM10', 'O3'
              - **pollutant_name (str)**: Full name
              - **unit (str)**: Unit of measurement
              - **value (float|None)**: Numeric value
              - **sub_aqi (int)**: Sub-index level for this pollutant
              - **formatted_measurement (str)**: Display string

            - ``Error (str)``: Error message if lookup failed

        :rtype: ``dict``
        """
        try:
            station_data, distance_km = self._get_nearest_station_to_city(city_name)
        except StationNotFoundError as exc:
            return {"city_searched": city_name, "Error": str(exc)}

        if not self._station_has_valid_data(station_data):
            _LOGGER.debug(
                "Station %s has no valid data. Attempting to find alternative station...",
                station_data.get("Name"),
            )
            nearby_stations = self._get_nearby_stations_sorted(city_name)

            for alt_station, alt_distance in nearby_stations[1:]:
                if self._station_has_valid_data(alt_station):
                    _LOGGER.debug(
                        "Fallback: Using station %s at %.2f km (1st station had no valid data).",
                        alt_station.get("Name"),
                        alt_distance,
                    )
                    station_data = alt_station
                    distance_km = alt_distance
                    break

        return self._format_station_data(station_data, distance_km, city_name)


    @_ensure_loaded
    def get_pollutant_measurement(self, city_name: str, pollutant_code: str) -> dict:
        """
        Get measurement data for a specific pollutant at the nearest station.

        :param city_name: City name to search for
        :type city_name: str
        :param pollutant_code: Pollutant code to retrieve (case-insensitive):

            - 'PM10': Particulate matter < 10 µm
            - 'PM2_5': Fine particulate matter < 2.5 µm  
            - 'O3': Ozone
            - 'NO2': Nitrogen dioxide
            - 'SO2': Sulfur dioxide

        :type pollutant_code: str
        :return: Measurement dictionary with keys:

            - **city_searched (str)**: Original search term
            - **station_name (str)**: Station name(s) that provided the measurement
            - **pollutant_code (str)**: Normalized pollutant code
            - **pollutant_name (str)**: Full pollutant name
            - **unit (str)**: Unit of measurement (e.g., 'µg/m³')
            - **value (float|None)**: Numeric measurement value
            - **measurement_status (str)**: Status string (e.g., 'Measured', 'No Data')
            - **formatted_measurement (str)**: Display string (e.g., '12.5 µg/m³')

        :rtype: ``dict``
        :raises StationNotFoundError: If city not found or no nearby stations exist
        :raises PollutantNotReportedError: If pollutant is not measured at any station
        """
        nearby_stations = self._get_nearby_stations_sorted(city_name)
        station_data, _ = self._get_nearest_station_to_city(city_name)

        pollutant_code_upper = pollutant_code.upper()
        stations_tried: list[str] = [station_data.get("Name") or ""]

        result = self._try_get_pollutant_from_station(
            station_data,
            station_data,
            pollutant_code_upper,
            city_name,
            stations_tried
        )

        if result:
            return result

        for alt_station, _ in nearby_stations:
            if alt_station.get("Name") in stations_tried:
                continue

            alt_result = self._try_get_pollutant_from_station(
                alt_station,
                station_data,
                pollutant_code_upper,
                city_name,
                stations_tried
            )

            if alt_result:
                return alt_result

            stations_tried.append(
                alt_station.get("Name") or ""
            )

        raise PollutantNotReportedError(
            f"Pollutant code '{pollutant_code_upper}' is not being measured"
            f" at any available station near '{city_name}'."
        )


    @_ensure_loaded
    def get_air_quality_index(self, city_name: str) -> tuple[int, str]:
        """
        Get EAQI for a city using the 0-6 scale.

        Returns the highest sub-index across all measured pollutants (PM10, PM2_5, O3, NO2, SO2).

        :param city_name: Name of the city
        :type city_name: str
        :return: Tuple of (EAQI level 0-6, description)
        :rtype: ``tuple[int, str]``
        """
        aqi_level = self._get_aqi(city_name)

        return aqi_level, const.EAQI_LEVELS.get(aqi_level, "Error/N/A")


    def force_fetch_fresh(self) -> None:
        """
        Force fetching fresh data from the source without waiting for the internal cache timer.

        Bypasses the normal 20-minute cache timeout and immediately requests fresh data from CHMI.
        Still uses cached data if server returns 304 (Not Modified) via ETag validation.

        :raises DataDownloadError: If network error occurs and no cache is available
        """
        self._data_manager.ensure_latest_data(force_fetch=True)
        self._load_and_parse_data()

