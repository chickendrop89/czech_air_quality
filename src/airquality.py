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
Air Quality Data Processor
"""

import json
import logging
from datetime import datetime
from functools import wraps

from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderServiceError
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests

import src.const
from src.data_manager import DataManager

_LOGGER = logging.getLogger(__name__)

def _ensure_loaded(func):
    """Ensure data is fresh and loaded before executing a public method"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.ensure_data_loaded()
        return func(self, *args, **kwargs)

    return wrapper


class AirQualityError(Exception):
    """Base exception for the czech-air-quality library."""

class DataDownloadError(AirQualityError):
    """Raised when data cannot be downloaded or is invalid."""

class StationNotFoundError(AirQualityError):
    """Raised when a city or station cannot be found."""

class PollutantNotReportedError(AirQualityError):
    """Raised when the nearest station does not report data for the requested pollutant."""

class AirQuality:
    """
    A client for retrieving and parsing air quality data from CHMI
    (Czech Hydrometeorological Institute).

    Data is cached locally until the data on the CHMI server changes
    """

    @classmethod
    def get_all_station_names(cls) -> list[str | None]:
        """
        Get all known air quality station names.

        :return: List of station names, or empty list if unavailable
        :rtype: list[str | None]
        """
        try:
            temp_instance = cls()
            return [station["Name"] for station in temp_instance.all_stations]
        except (DataDownloadError, StationNotFoundError) as e:
            _LOGGER.error("Failed to get all station names: %s", e)
            return []


    def __init__(
        self,
        region_filter=None,
        disable_caching=False,
        auto_load=True,
        use_nominatim=True,
        nominatim_timeout=src.const.NOMINATIM_TIMEOUT,
        request_timeout=src.const.REQUEST_TIMEOUT
    ):
        """
        Initialize the Air Quality client.

        :param region_filter: Limit stations to specific region (case-insensitive)
        :type region_filter: str, optional
        :param disable_caching: Skip caching, force fresh download every time
        :type disable_caching: bool
        :param auto_load: Load/download data immediately
        :type auto_load: bool
        :param use_nominatim: Enable Nominatim geocoding for city lookups (True) or only exact station names (False)
        :type use_nominatim: bool
        :param nominatim_timeout: Geocoding timeout in seconds
        :type nominatim_timeout: int
        :param request_timeout: HTTP request timeout in seconds
        :type request_timeout: int
        """
        self._region_filter = region_filter.lower() if region_filter else None
        self._use_nominatim = use_nominatim
        self._nominatim_timeout = nominatim_timeout

        self._data = {}
        self._all_stations = []
        self._component_lookup = {}
        self._id_registration_to_component = {}
        self._locality_code_to_station = {}
        self._city_coordinate_cache = {}

        self._data_manager = DataManager(
            disable_caching=disable_caching,
            request_timeout=request_timeout
        )

        if self._use_nominatim:
            self._geolocator = Nominatim(
                user_agent=src.const.USER_AGENT
            )
            self._rate_limited_geocode = RateLimiter(
                self._geolocator.geocode,
                min_delay_seconds=1.0
            )
        else:
            self._geolocator = None
            self._rate_limited_geocode = None

        if auto_load:
            self._data_manager.ensure_latest_data()
            self._load_and_parse_data()


    @property
    def actualized_time(self) -> datetime:
        """
        Timestamp when data was last updated by the CHMI source.

        :rtype: datetime
        """
        return self._data_manager.actualized_time

    @property
    def is_data_fresh(self) -> bool:
        """
        Check if cached data is still valid via ETag validation.

        :return: `True` if cached data is current; `False` if needs refresh
        :rtype: bool
        """
        return self._data_manager.is_data_fresh()

    @property
    def all_stations(self) -> list[dict]:
        """
        Get all available air quality stations.

        :return: List of station dictionaries, filtered by region if set
        :rtype: list[dict]
        """
        return self._all_stations

    @property
    def component_lookup(self) -> dict[str, tuple[str, str, str]]:
        """
        Map of pollutant codes to (code, name, unit) tuples.

        :return: Dictionary with pollutant code as key
        :rtype: dict[str, tuple[str, str, str]]
        """
        return self._component_lookup

    @property
    def raw_data(self) -> dict:
        """
        Raw parsed data from the JSON source.

        :return: Dictionary containing localities and measurements
        :rtype: dict
        """
        return self._data

    @property
    def last_download_status(self) -> str:
        """
        Status message from the last download attempt.

        :rtype: str
        """
        return self._data_manager.last_download_status


    def ensure_data_loaded(self) -> None:
        """
        Ensure data is loaded and fresh via ETag validation.

        Re-downloads if any server resource has been modified.
        """
        if not self._data_manager.raw_data_json or not self.is_data_fresh:
            self._data_manager.ensure_latest_data()
            self._load_and_parse_data()


    def get_city_coordinates(self, city_name: str) -> tuple[float, float] | None:
        """
        Get geographic coordinates for a city.
        Uses local cache, then Nominatim geocoding if enabled.

        :param city_name: City name to geocode
        :type city_name: str
        :return: (latitude, longitude) tuple or None
        :rtype: tuple[float, float] | None
        """
        if city_name in self._city_coordinate_cache:
            _LOGGER.info("Coordinates for '%s' retrieved from local cache.", city_name)
            return self._city_coordinate_cache[city_name]

        if not self._use_nominatim or self._rate_limited_geocode is None:
            _LOGGER.debug("Nominatim geocoding disabled. Cannot lookup coordinates for '%s'.", city_name)
            return None

        _LOGGER.info("Attempting external geocoding for '%s'...", city_name)
        try:
            search_query = f"{city_name}, Czechia"
            location = self._rate_limited_geocode(
                search_query,
                timeout=self._nominatim_timeout
            )

            if location:
                coords = (
                    location.latitude,
                    location.longitude
                )
                self._city_coordinate_cache[city_name] = coords
                _LOGGER.info("Successfully geocoded '%s'.", city_name)
                return coords

            _LOGGER.warning("External geocoding failed for '%s'.", city_name)
            return None

        except (requests.exceptions.RequestException, GeocoderServiceError) as exc:
            _LOGGER.error("Geocoding service error for '%s': %s", city_name, exc)
            return None


    @_ensure_loaded
    def find_nearest_station(self, city_name: str) -> tuple[dict, float]:
        """
        Find air quality station nearest to a city.

        If use_nominatim is False, only exact station name matches are returned.
        If use_nominatim is True, attempts to find coordinates and locate nearest station.

        :param city_name: City name to search for
        :type city_name: str
        :return: (station_dict, distance_km) tuple
        :rtype: Tuple[dict, float]
        :raises StationNotFoundError: If city or nearby stations not found
        """
        city_name_lower = city_name.lower()

        for station in self._all_stations:
            if city_name_lower in station["Name"].lower():
                _LOGGER.info("Direct Match Found: %s", station["Name"])
                return station, 0.0

        if not self._use_nominatim:
            raise StationNotFoundError(
                f"No exact station match found for '{city_name}', and nominatim geocoding is disabled."
            )

        _LOGGER.info(
            "No direct station for '%s'. Attempting coordinate lookup...", city_name
        )

        city_coords = self.get_city_coordinates(city_name)
        if not city_coords:
            raise StationNotFoundError(
                f"Could not find geographic coordinates for '{city_name}'. Please use a known station name or a different city."
            )

        nearest_station = None
        min_distance_km = float("inf")

        for station in self._all_stations:
            try:
                station_coords = (float(station["Lat"]), float(station["Lon"]))
                distance = geodesic(city_coords, station_coords).km

                if distance < min_distance_km:
                    min_distance_km, nearest_station = distance, station
            except ValueError:
                _LOGGER.debug(
                    "Skipping station %s due to invalid coordinates.",
                    station.get("Name"),
                )
                continue

        if nearest_station:
            _LOGGER.info(
                "Nearest Station Found: %s at %.2f km.",
                nearest_station["Name"],
                min_distance_km,
            )
            return nearest_station, min_distance_km

        raise StationNotFoundError(
            f"No air quality stations could be found after searching for '{city_name}'."
        )


    @_ensure_loaded
    def get_air_quality_index(self, city_name: str) -> int:
        """
        Get overall EAQI for the nearest station to a city.
        EAQI is the maximum sub-index of all reported pollutants.

        :param city_name: City name to search for
        :type city_name: str
        :return: EAQI value (0-100+) or -1 if no data
        :rtype: int
        """
        station_data, _ = self.find_nearest_station(city_name)
        measurements = self._get_station_measurements(station_data)

        max_aqi = 0

        for meas in measurements:
            code = meas.get("ComponentCode")
            value_str = meas.get("value")

            if not code or not self._is_valid_measurement(value_str):
                continue

            try:
                value_float = float(value_str)  # type: ignore
                sub_aqi = self._calculate_e_aqi_subindex(code, value_float)
                max_aqi = max(max_aqi, sub_aqi)
            except (ValueError, TypeError):
                _LOGGER.debug(
                    "Skipping AQI calculation for %s due to invalid value: %s",
                    code,
                    value_str,
                )
                continue

        return max_aqi if max_aqi > 0 else -1


    @_ensure_loaded
    def get_station_capabilities(self, city_name: str) -> list[str | None]:
        """
        Get pollutant codes measured by the nearest station.

        :param city_name: City name to search for
        :type city_name: str
        :return: List of pollutant codes (e.g., ['PM10', 'O3'])
        :rtype: list[str | None]
        :raises StationNotFoundError: If station not found
        """
        station_data, _ = self.find_nearest_station(city_name)
        measurements = self._get_station_measurements(station_data)

        return [
            meas.get("ComponentCode")
            for meas in measurements
            if meas.get("ComponentCode")
        ]


    @_ensure_loaded
    def get_air_quality_report(self, city_name: str) -> dict:
        """
        Get comprehensive air quality report with EAQI (European Air Quality Index) for a city.

        :param city_name: City name to search for (e.g., 'Prague', 'Brno')
        :type city_name: str
        :return: Air quality report dictionary with keys:
                - city_searched (str): Original search term
                - station_name (str): Name of station providing data
                - station_code (str): Station locality code
                - region (str): Czech region name
                - distance_km (str): Distance from city to station (e.g., "12.34")
                - air_quality_index_code (int): EAQI value (0-500+, -1 if no data)
                - air_quality_index_description (str): Human description (e.g., 'Good', 'Poor')
                - actualized_time_utc (str): ISO format UTC timestamp of data
                - measurements (list[dict]): List of pollutant measurements with:
                    * pollutant_code (str): Code like 'PM10', 'O3'
                    * pollutant_name (str): Full name
                    * unit (str): Unit of measurement
                    * value (float|None): Numeric value
                    * sub_aqi (int): Sub-index for this pollutant (-1 if no data)
                    * formatted_measurement (str): Display string
                - Error (str): [Optional] Error message if lookup failed
        :rtype: dict

        **EAQI Scale:**
        - 0-25: Good (favorable air quality)
        - 26-50: Fair (acceptable)
        - 51-75: Poor (members of at-risk groups should limit outdoor exposure)
        - 76-100: Very Poor (general population should reduce outdoor exposure)
        - 100+: Extremely Poor (significant health risk for all)
        """
        try:
            station_data, distance_km = self.find_nearest_station(city_name)
        except StationNotFoundError as exc:
            return {"city_searched": city_name, "Error": str(exc)}

        if not self._station_has_valid_data(station_data):
            _LOGGER.info(
                "Station %s has no valid data. Attempting to find alternative station...",
                station_data.get("Name"),
            )
            nearby_stations = self._get_nearby_stations_sorted(city_name, limit=5)

            for alt_station, alt_distance in nearby_stations[1:]:
                if self._station_has_valid_data(alt_station):
                    _LOGGER.info(
                        "Fallback: Using station %s at %.2f km (primary station had no valid data).",
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

        :param city_name: City name to search for (e.g., 'Prague', 'Brno')
        :type city_name: str
        :param pollutant_code: Pollutant code to retrieve (case-insensitive):
                              - 'PM10': Particulate matter < 10 µm
                              - 'PM2.5': Fine particulate matter < 2.5 µm  
                              - 'O3': Ozone
                              - 'NO2': Nitrogen dioxide
                              - 'SO2': Sulfur dioxide
        :type pollutant_code: str
        :return: Measurement dictionary with keys:
                - city_searched (str): Original search term
                - station_name (str): Station name(s) that provided the measurement (comma-separated if multiple)
                - pollutant_code (str): Pollutant code (uppercase with underscores)
                - pollutant_name (str): Full pollutant name
                - unit (str): Measurement unit (e.g., 'µg/m³')
                - value (float): Numeric measurement value
                - measurement_status (str): Status ('Measured', 'No Data Available', etc.)
                - formatted_measurement (str): Human-readable value with unit
        :rtype: dict
        :raises StationNotFoundError: If no city/station found for the search term
        :raises PollutantNotReportedError: If none of the 5 nearest stations report this pollutant
        """
        station_data, _ = self.find_nearest_station(city_name)
        pollutant_code_normalized = pollutant_code.upper().replace(".", "_")
        nearby_stations = self._get_nearby_stations_sorted(city_name, limit=5)
        stations_tried = [str(s.get("Name", "")) for s, _ in nearby_stations]

        for alt_station, _ in nearby_stations:
            result = self._try_get_pollutant_from_station(
                alt_station, station_data, pollutant_code_normalized,
                city_name, stations_tried
            )
            if result:
                return result

        raise PollutantNotReportedError(
            f"No nearby station found that reports data for {pollutant_code_normalized}."
        )

    def _try_get_pollutant_from_station(self, alt_station: dict, primary_station: dict,
            pollutant_code: str, city_name: str, stations_tried: list[str]) -> dict | None:
        """
        Try to get a pollutant measurement from a specific station.

        :param alt_station: Station to check
        :param primary_station: Primary station (for logging fallback)
        :param pollutant_code: Normalized pollutant code to find
        :param city_name: City searched for (for result dict)
        :param stations_tried: List of stations already tried (for logging)
        :return: Measurement result dict or None if not found
        """
        station_name = alt_station.get("Name")
        measurements = self._get_station_measurements(alt_station)

        for measurement in measurements:
            if measurement.get("ComponentCode") != pollutant_code:
                continue

            value = measurement.get("value")
            value_float, measurement_str, status = self._process_measurement_value(
                value, measurement.get("Unit", "N/A")
            )

            if value_float is not None and value_float >= 0:
                if alt_station != primary_station:
                    _LOGGER.info(
                        "Fallback: Using station %s for pollutant %s (searched through: %s).",
                        station_name, pollutant_code, ", ".join(stations_tried),
                    )

                return {
                    "city_searched": city_name,
                    "station_name": station_name,
                    "pollutant_code": pollutant_code,
                    "pollutant_name": measurement.get("ComponentName", pollutant_code),
                    "unit": measurement.get("Unit", "N/A"),
                    "value": value_float,
                    "measurement_status": status,
                    "formatted_measurement": measurement_str,
                }

        return None

    def _process_measurement_value(self, value: str | None,
            unit: str) -> tuple[float | None, str, str]:
        """
        Process and validate a measurement value.

        :param value: Raw measurement value
        :param unit: Unit of measurement
        :return: (value_float, display_string, status) tuple
        """
        if self._is_valid_measurement(value):
            try:
                value_float = float(value)  # type: ignore
                return value_float, f"{value} {unit}", "Measured"
            except (ValueError, TypeError):
                return None, "N/A", "Invalid Value Format"

        return None, "N/A", "No Data Available"


    def _load_and_parse_data(self) -> None:
        """
        Parse raw JSON data into internal structures.

        :raises DataDownloadError: If raw data is empty or invalid JSON
        """
        raw_json = self._data_manager.raw_data_json
        if not raw_json:
            self._data = {}
            raise DataDownloadError("Cannot parse data: raw JSON string is empty.")

        try:
            raw_dict = json.loads(raw_json)
            raw_dict.pop(src.const.CACHE_METADATA_KEY, None)
            self._data = raw_dict
        except json.JSONDecodeError as exc:
            _LOGGER.error("Error decoding JSON data during final load: %s", exc)
            raise DataDownloadError(f"Data is corrupted/invalid JSON: {exc}") from exc

        self._id_registration_to_component = self._data.get(
            "id_registration_to_component", {}
        )
        self._component_lookup = {
            str(id_reg): (comp["ComponentCode"], comp["ComponentName"], comp["Unit"])
            for id_reg, comp in self._id_registration_to_component.items()
        }
        self._all_stations = self._collect_stations()


    def _collect_stations(self) -> list[dict]:
        """
        Collect stations from data, applying region filter.

        :return: List of station dictionaries with coordinates
        :rtype: list[dict]
        """
        stations = []

        for locality in self._data.get("Localities", []):
            region_name = locality.get("Region", "")
            if self._region_filter and region_name.lower() != self._region_filter:
                continue

            if locality.get("Lat") and locality.get("Lon"):
                station = {
                    "Name": locality.get("Name"),
                    "LocalityCode": locality.get("LocalityCode"),
                    "Region": region_name,
                    "Lat": locality.get("Lat"),
                    "Lon": locality.get("Lon"),
                    "IdRegistrations": locality.get("IdRegistrations", []),
                }
                stations.append(station)
                self._locality_code_to_station[locality.get("LocalityCode")] = station

        if self._region_filter:
            _LOGGER.info(
                "Station list filtered to region: %s (%d stations found)",
                self._region_filter,
                len(stations),
            )
        return stations


    def _is_valid_measurement(self, value: float | str | None) -> bool:
        """
        Check if a measurement value is valid.

        Stations return negative values like -5009, -5003, -9999 to indicate
        missing or invalid data. Any negative value should be treated as invalid.

        :param value: Measurement value (string or float)
        :type value: float | str | None
        :return: True if value is valid (non-negative), False otherwise
        :rtype: bool
        """
        if value is None or value == "":
            return False

        try:
            value_float = float(value)
            return value_float >= src.const.CHMI_ERROR_THRESHOLD
        except (ValueError, TypeError):
            return False


    def _get_nearby_stations_sorted(self, city_name: str, limit: int = 5) -> list[tuple[dict, float]]:
        """
        Get a list of nearby stations sorted by distance.

        Returns multiple stations for fallback purposes.

        :param city_name: City name to search for
        :type city_name: str
        :param limit: Maximum number of stations to return
        :type limit: int
        :return: List of (station_dict, distance_km) tuples sorted by distance
        :rtype: list[tuple[dict, float]]
        """
        city_name_lower = city_name.lower()
        stations_with_distance = []

        direct_match = None
        for station in self._all_stations:
            if city_name_lower in station["Name"].lower():
                direct_match = station
                stations_with_distance.append((station, 0.0))
                break

        if direct_match and not self._use_nominatim:
            return stations_with_distance[:limit]

        if self._use_nominatim:
            city_coords = self.get_city_coordinates(city_name)
            if not city_coords:
                return stations_with_distance[:limit]

            for station in self._all_stations:
                if direct_match and station["Name"] == direct_match["Name"]:
                    continue
                try:
                    station_coords = (float(station["Lat"]), float(station["Lon"]))
                    distance = geodesic(city_coords, station_coords).km
                    stations_with_distance.append((station, distance))
                except ValueError:
                    continue

            direct_matches = [s for s in stations_with_distance if s[1] == 0.0]
            other_stations = [s for s in stations_with_distance if s[1] > 0.0]
            other_stations.sort(key=lambda x: x[1])

            return direct_matches + other_stations[:limit - len(direct_matches)]

        return stations_with_distance[:limit]


    def _station_has_valid_data(self, station_data: dict) -> bool:
        """
        Check if a station has at least one valid measurement.

        :param station_data: Station dictionary
        :type station_data: dict
        :return: True if station has valid measurements, False otherwise
        :rtype: bool
        """
        measurements = self._get_station_measurements(station_data)
        for meas in measurements:
            value = meas.get("value")
            if self._is_valid_measurement(value):
                return True
        return False


    def _station_supports_pollutant(self, station_data: dict, pollutant_code: str) -> bool:
        """
        Check if a station measures a specific pollutant with valid data.

        :param station_data: Station dictionary
        :type station_data: dict
        :param pollutant_code: Pollutant code (e.g., 'PM10', 'NO2')
        :type pollutant_code: str
        :return: True if station has valid data for this pollutant, False otherwise
        :rtype: bool
        """
        measurements = self._get_station_measurements(station_data)
        pollutant_code_upper = pollutant_code.upper()

        for meas in measurements:
            if meas.get("ComponentCode") == pollutant_code_upper:
                value = meas.get("value")
                if self._is_valid_measurement(value):
                    return True
        return False


    def _calculate_e_aqi_subindex(self, pollutant_code: str, concentration: float) -> int:
        """
        Calculate EAQI sub-index for a pollutant.

        Uses linear interpolation between concentration breakpoints.

        :param pollutant_code: Pollutant code (e.g., 'PM10', 'O3')
        :type pollutant_code: str
        :param concentration: Measured concentration in µg/m³
        :type concentration: float
        :return: AQI sub-index or -1 if invalid
        :rtype: int
        """
        if concentration is None or concentration < 0:
            return -1

        bands = src.const.EAQI_BANDS.get(pollutant_code.upper())
        if not bands:
            return -1

        # Finds which two breakpoints the concentration falls between,
        # then uses linear interpolation to calculate the AQI value.
        if concentration < bands[0][1]:
            aqi_low, conc_low = 0, 0
            aqi_high, conc_high = bands[0]
        else:
            aqi_low, conc_low = bands[0]
            aqi_high, conc_high = bands[0]

            for i in range(1, len(bands)):
                aqi_curr, conc_curr = bands[i]

                if concentration < conc_curr:
                    aqi_low, conc_low = bands[i - 1]
                    aqi_high, conc_high = aqi_curr, conc_curr
                    break

                if i == len(bands) - 1:
                    aqi_low, conc_low = aqi_curr, conc_curr
                    aqi_high, conc_high = (
                        bands[i],
                        concentration,
                    )
            else:
                aqi_low, conc_low = bands[-1]
                return aqi_low

        if conc_high - conc_low == 0:
            return aqi_high if concentration >= conc_high else aqi_low

        aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (
            concentration - conc_low
        ) + aqi_low

        return int(round(aqi))


    def _get_aqi_description(self, aqi_value: int) -> str:
        """
        Get text description for EAQI value.

        :param aqi_value: EAQI value
        :type aqi_value: int
        :return: Description string (e.g., 'Good', 'Poor')
        :rtype: str
        """
        if aqi_value <= 0:
            return "N/A"

        for limit, description in sorted(src.const.EAQI_LEVELS.items()):
            if aqi_value <= limit:
                return description

        return src.const.EAQI_LEVELS[max(src.const.EAQI_LEVELS.keys())]


    def _get_station_measurements(self, station_data: dict) -> list[dict]:
        """
        Get all measurements for a station from the data.

        :param station_data: Station dictionary
        :type station_data: dict
        :return: List of measurement dictionaries
        :rtype: list[dict]
        """
        measurements_list = []
        measurements_dict = self._data.get("Measurements", {})

        for id_reg in station_data.get("IdRegistrations", []):
            id_reg_str = str(id_reg)
            if id_reg_str in measurements_dict:
                comp_info = self._id_registration_to_component.get(id_reg_str, {})

                meas_list = measurements_dict.get(id_reg_str, [])
                latest_value = None
                if meas_list:
                    latest_value = meas_list[-1].get("value")

                measurements_list.append(
                    {
                        "ComponentCode": comp_info.get("ComponentCode"),
                        "ComponentName": comp_info.get("ComponentName"),
                        "Unit": comp_info.get("Unit"),
                        "value": latest_value,
                        "idRegistration": id_reg,
                    }
                )
        return measurements_list


    def _format_station_data(self, station_data: dict,
        distance_km: float, city_searched: str) -> dict:
        """
        Format raw station data into public-facing report structure.

        :param station_data: Station dictionary
        :type station_data: dict
        :param distance_km: Distance from city to station
        :type distance_km: float
        :param city_searched: Original city name searched
        :type city_searched: str
        :return: Formatted report dictionary
        :rtype: dict
        """
        overall_aqi_value = self.get_air_quality_index(city_searched)
        measurements_list = self._get_station_measurements(station_data)
        nearby_stations = self._get_nearby_stations_sorted(city_searched, limit=5)

        measurements = []
        stations_used = [station_data.get("Name", "")]
        added_pollutants = set()

        for meas in measurements_list:
            code = meas.get("ComponentCode")
            added_pollutants.add(code)

            measurement_data = self._build_measurement_entry(
                meas, code, station_data, nearby_stations, stations_used
            )
            measurements.append(measurement_data)

        extra_pollutants = self._find_extra_pollutants(
            nearby_stations, station_data, added_pollutants
        )
        for code, (alt_station, alt_meas) in extra_pollutants.items():
            measurement_data = self._build_extra_pollutant_entry(
                alt_meas, alt_station, station_data, stations_used
            )
            measurements.append(measurement_data)

        combined_station_name = ", ".join(stations_used)

        return {
            "city_searched": city_searched,
            "station_name": combined_station_name,
            "station_code": station_data.get("LocalityCode"),
            "region": station_data.get("Region"),
            "distance_km": f"{distance_km:.2f}",
            "air_quality_index_code": overall_aqi_value,
            "air_quality_index_description": self._get_aqi_description(overall_aqi_value),
            "actualized_time_utc": self._data_manager.actualized_time.isoformat(),
            "measurements": measurements,
        }


    def _build_measurement_entry(self, meas: dict, code: str | None,
            primary_station: dict, nearby_stations: list[tuple[dict, float]],
            stations_used: list[str]) -> dict:
        """
        Build a measurement entry, using fallback stations if needed.

        :param meas: Measurement from primary station
        :param code: Pollutant code
        :param primary_station: Primary station data
        :param nearby_stations: List of nearby stations for fallback
        :param stations_used: List accumulator for station names
        :return: Formatted measurement dictionary
        """
        name = meas.get("ComponentName", code)
        unit = meas.get("Unit", "N/A")
        value = meas.get("value")
        value_float = None
        status_text = "N/A"
        sub_aqi = -1

        if self._is_valid_measurement(value):
            value_float, status_text, sub_aqi = self._process_valid_measurement(
                str(value), unit, code
            )
        elif code:
            value_float, status_text, sub_aqi, _ = self._find_measurement_fallback(
                code, unit, primary_station, nearby_stations, stations_used
            )

        if value_float is None:
            status_text = "No Data Available"

        return {
            "pollutant_code": code,
            "pollutant_name": name,
            "unit": unit,
            "value": value_float,
            "sub_aqi": sub_aqi,
            "formatted_measurement": status_text,
        }


    def _process_valid_measurement(self, value:
            str, unit: str, code: str | None) -> tuple[float | None, str, int]:
        """
        Process a valid measurement value.

        :return: (value_float, status_text, sub_aqi) tuple
        """
        try:
            value_float = float(value)
            status_text = f"{value} {unit}"
            sub_aqi = self._calculate_e_aqi_subindex(code, value_float) if code else -1
            return value_float, status_text, sub_aqi
        except (ValueError, TypeError):
            return None, "Invalid Value Format", -1


    def _find_measurement_fallback(self, code: str, unit: str,
            primary_station: dict, nearby_stations: list[tuple[dict, float]],
            stations_used: list[str]) -> tuple[float | None, str, int, str | None]:
        """
        Search nearby stations for a measurement with fallback logic.

        :return: (value_float, status_text, sub_aqi, used_station_name) tuple
        """
        primary_name = primary_station.get("Name")
        value_float = None
        status_text = "No Data Available"
        sub_aqi = -1
        used_station_name = primary_name

        for alt_station, _ in nearby_stations:
            alt_station_name = alt_station.get("Name")
            if alt_station_name == primary_name:
                continue

            alt_measurements = self._get_station_measurements(alt_station)
            for alt_meas in alt_measurements:
                if alt_meas.get("ComponentCode") == code:
                    alt_value = alt_meas.get("value")
                    if self._is_valid_measurement(alt_value):
                        value_float, status_text, sub_aqi = self._process_valid_measurement(
                            str(alt_value), unit, code
                        )
                        if value_float is not None:
                            used_station_name = alt_station_name
                            if alt_station_name and alt_station_name not in stations_used:
                                stations_used.append(alt_station_name)
                            _LOGGER.info(
                                "Fallback: Using station %s for pollutant %s in report",
                                alt_station_name, code,
                            )
                            return value_float, status_text, sub_aqi, used_station_name

            if value_float is not None:
                break

        return value_float, status_text, sub_aqi, used_station_name


    def _find_extra_pollutants(self, nearby_stations: list[tuple[dict, float]],
            primary_station: dict, added_pollutants: set) -> dict[str, tuple[dict, dict]]:
        """
        Find pollutants available in nearby stations but not in primary station.

        :return: Dictionary mapping pollutant code to (station, measurement)
        """
        extra_pollutants = {}

        for alt_station, _ in nearby_stations:
            if alt_station.get("Name") == primary_station.get("Name"):
                continue

            alt_measurements = self._get_station_measurements(alt_station)
            for alt_meas in alt_measurements:
                code = alt_meas.get("ComponentCode")
                if code and code not in added_pollutants and code not in extra_pollutants:
                    value = alt_meas.get("value")
                    if self._is_valid_measurement(value):
                        extra_pollutants[code] = (alt_station, alt_meas)

        return extra_pollutants


    def _build_extra_pollutant_entry(self, meas: dict, alt_station: dict,
            primary_station: dict, stations_used: list[str]) -> dict:
        """
        Build a measurement entry for an extra pollutant from nearby station.

        :param meas: Measurement from alternative station
        :param alt_station: Station providing the measurement
        :param primary_station: Primary station (for logging)
        :param stations_used: List accumulator for station names
        :return: Formatted measurement dictionary
        """
        code = meas.get("ComponentCode")
        name = meas.get("ComponentName", code)
        unit = meas.get("Unit", "N/A")
        value = meas.get("value")
        alt_station_name = alt_station.get("Name")

        status_text = "N/A"
        value_float = None
        sub_aqi = -1

        try:
            value_float = float(value)  # type: ignore
            status_text = f"{value} {unit}"
            if code:
                sub_aqi = self._calculate_e_aqi_subindex(code, value_float)

            if alt_station_name and alt_station_name not in stations_used:
                stations_used.append(alt_station_name)

            _LOGGER.info(
                "Fallback: Adding pollutant %s from station %s (not measured by primary station %s).",
                code, alt_station_name, primary_station.get("Name"),
            )
        except (ValueError, TypeError):
            status_text = "Invalid Value Format"

        return {
            "pollutant_code": code,
            "pollutant_name": name,
            "unit": unit,
            "value": value_float,
            "sub_aqi": sub_aqi,
            "formatted_measurement": status_text,
        }
