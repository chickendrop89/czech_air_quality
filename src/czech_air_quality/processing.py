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
Data processing of air quality.
"""

import json
import logging
import requests

from geopy.distance import geodesic
from geopy.exc import GeocoderServiceError, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter

from .data_manager import DataManager
from . import const, _warn
from . import DataDownloadError, StationNotFoundError

_LOGGER = logging.getLogger(__name__)


class AirQualityCalculations:
    """
    Helper class for air quality data processing and calculations.
    """
    def __init__(self):
        # Attributes initialized in AirQuality.__init__
        self._data_manager: DataManager
        self._region_filter: str | None
        self._use_nominatim: bool
        self._neighbour_station_limit: int
        self._nominatim_timeout: int
        self._data: dict
        self._all_stations: list[dict]
        self._component_lookup: dict
        self._id_registration_to_component: dict
        self._locality_code_to_station: dict
        self._city_coordinate_cache: dict
        self._rate_limited_geocode: RateLimiter | None


    def _load_and_parse_data(self) -> None:
        """
        Parse raw JSON data into internal structures.

        :raises DataDownloadError: If raw data is empty or invalid JSON
        """
        raw_json: str | None = self._data_manager.raw_data_json
        if not raw_json:
            self._data = {}
            raise DataDownloadError("Cannot parse data: raw JSON string is empty.")

        try:
            raw_dict = json.loads(raw_json)
            raw_dict.pop(const.CACHE_METADATA_KEY, None)
            self._data = raw_dict
        except json.JSONDecodeError as exc:
            raise DataDownloadError(f"Data is corrupted/invalid JSON: {exc}") from exc

        self._id_registration_to_component = self._data.get(
            "id_registration_to_component", {}
        )
        self._component_lookup = {
            str(id_reg): (comp.get("ComponentCode"), comp.get("ComponentName"), comp.get("Unit"))
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
            _LOGGER.debug(
                "Station list filtered to region: %s (%d stations found)",
                self._region_filter,
                len(stations),
            )
        return stations


    def _is_valid_measurement(self, value: float | str | None) -> bool:
        """
        Check if a measurement value is valid.

        Stations return 1.0 or negative values like -5009, -5003, -9999 to indicate
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
            return value_float >= const.CHMI_ERROR_THRESHOLD
        except (ValueError, TypeError):
            return False


    def _get_nearby_stations_sorted(self, city_name: str,
            limit: int | None = None) -> list[tuple[dict, float]]:
        """
        Get a list of nearby stations sorted by distance.

        Returns multiple stations for fallback purposes.

        :param city_name: City name to search for
        :type city_name: str
        :param limit: Maximum number of stations to return
        :type limit: int | None
        :return: List of (station_dict, distance_km) tuples sorted by distance
        :rtype: list[tuple[dict, float]]
        """
        city_name_lower = city_name.lower()
        stations_with_distance = []

        if limit is None:
            limit = self._neighbour_station_limit

        for station in self._all_stations:
            if city_name_lower in station["Name"].lower():
                stations_with_distance.append((station, 0.0))
                break

        if stations_with_distance and not self._use_nominatim:
            return stations_with_distance[:limit]

        if self._use_nominatim:
            city_coords = self._get_city_coordinates_internal(city_name)
            if not city_coords:
                return stations_with_distance[:limit]

            for station in self._all_stations:
                if stations_with_distance and station["Name"] == stations_with_distance[0][0]["Name"]:
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


    def _calculate_e_aqi_subindex(self, pollutant_code: str, concentration: float | None) -> int:
        """
        Calculate EAQI sub-index for a pollutant

        Returns the EAQI level (0-6) based on concentration thresholds.
        Level 0 is reserved for invalid data.

        :param pollutant_code: Pollutant code (e.g., 'PM10', 'O3')
        :type pollutant_code: str
        :param concentration: Measured concentration in µg/m³
        :type concentration: float
        :return: EAQI level (0-6), 0 if invalid
        :rtype: int
        """
        if concentration is None or concentration < 0:
            return 0

        bands = const.EAQI_BANDS.get(pollutant_code.upper())
        if not bands or len(bands) == 0:
            return 0

        for level, threshold in bands:
            if concentration <= threshold:
                return level

        return bands[-1][0]


    def _get_aqi_description(self, aqi_value: int) -> str:
        """
        Get text description for EAQI value

        :param aqi_value: EAQI value (0-6)
        :type aqi_value: int
        :return: Description string (e.g., 'Good', 'Poor')
        :rtype: str
        """
        if aqi_value < 0 or aqi_value > 6:
            return const.EAQI_LEVELS.get(0, "Error/N/A")

        return const.EAQI_LEVELS.get(aqi_value, "Error/N/A")


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


    def _get_aqi(self, city_name: str) -> int:
        """
        Get overall EAQI for the nearest station to a city.

        EAQI is the maximum sub-index of all reported pollutants.
        Includes pollutants from nearby stations if not available in primary station.

        EAQI levels:
        - 0 = Error/N/A
        - 1 = Good
        - 2 = Fair
        - 3 = Moderate
        - 4 = Poor
        - 5 = Very Poor
        - 6 = Extremely Poor

        :param city_name: City name to search for
        :type city_name: str
        :return: EAQI level (0-6)
        :rtype: int
        """
        station_data, _ = self._get_nearest_station_to_city(city_name)
        nearby_stations = self._get_nearby_stations_sorted(city_name)
        measurements = self._get_station_measurements(station_data)

        max_aqi = 0
        added_pollutants = set()

        for meas in measurements:
            code = meas.get("ComponentCode")
            value_str = meas.get("value")

            if not code or not self._is_valid_measurement(value_str):
                continue

            added_pollutants.add(code)

            try:
                value_float = float(value_str)  # type: ignore[arg-type]
                sub_aqi = self._calculate_e_aqi_subindex(code, value_float)
                max_aqi = max(max_aqi, sub_aqi)
            except (ValueError, TypeError):
                _LOGGER.debug(
                    "Skipping AQI calculation for %s due to invalid value: %s",
                    code,
                    value_str,
                )
                continue

        extra_pollutants = self._find_extra_pollutants(
            nearby_stations,
            station_data,
            added_pollutants
        )

        for code, (_, meas) in extra_pollutants.items():
            value_str = meas.get("value")
            if not self._is_valid_measurement(value_str):
                continue

            try:
                value_float = float(value_str)  # type: ignore[arg-type]
                sub_aqi = self._calculate_e_aqi_subindex(code, value_float)
                max_aqi = max(max_aqi, sub_aqi)
            except (ValueError, TypeError):
                _LOGGER.debug(
                    "Skipping AQI calculation for %s due to invalid value: %s",
                    code,
                    value_str,
                )
                continue

        return max_aqi if max_aqi > 0 else 0


    def _try_get_pollutant_from_station(self, alt_station: dict, primary_station: dict,
            pollutant_code: str, city_name: str, stations_tried: list[str]) -> dict | None:
        """
        Try to get a pollutant measurement from a specific station.

        :param alt_station: Station to check
        :param primary_station: Primary station
        :param pollutant_code: Normalized pollutant code to find
        :param city_name: City searched for
        :param stations_tried: List of stations already tried
        :return: Measurement result dict or None if not found or unrealistic
        """
        station_name = alt_station.get("Name")
        measurements = self._get_station_measurements(alt_station)

        for measurement in measurements:
            if measurement.get("ComponentCode") != pollutant_code:
                continue

            value = measurement.get("value")
            value_float = None
            status = "No Data Available"

            if self._is_valid_measurement(value):
                try:
                    value_float = float(value)  # type: ignore[arg-type]
                    _, status, _ = self._process_valid_measurement(
                        str(value),
                        measurement.get("Unit", "N/A"),
                        pollutant_code
                    )
                except (ValueError, TypeError):
                    status = "Invalid Value Format"

            if self._is_valid_measurement(value_float):
                if alt_station != primary_station:
                    _LOGGER.debug(
                        "Fallback: Using station %s for pollutant %s (searched through: %s).",
                        station_name,
                        pollutant_code,
                        ", ".join(stations_tried)
                    )

                return {
                    "city_searched": city_name,
                    "station_name": station_name,
                    "pollutant_code": pollutant_code,
                    "pollutant_name": measurement.get("ComponentName", pollutant_code),
                    "unit": measurement.get("Unit", "N/A"),
                    "value": value_float,
                    "measurement_status": status,
                    "formatted_measurement": status,
                }
            else:
                _LOGGER.debug(
                    "Skipping suspiciously low value (%s) from %s. %s",
                    value_float, alt_station.get("Name"),
                    "Searching for better data"
                )
                continue

        return None


    def _get_nearest_station_to_city(self, city_name: str) -> tuple[dict, float]:
        """
        Get the nearest air quality station to a city.

        :param city_name: City name to search for
        :type city_name: str
        :return: (station_dict, distance_km) tuple
        :rtype: Tuple[dict, float]
        :raises StationNotFoundError: If city or nearby stations not found
        """
        nearby_station = self._get_nearby_stations_sorted(city_name, 1)

        if nearby_station:
            station, distance = nearby_station[0]
            _LOGGER.debug(
                "Nearest Station Found: %s at %.2f km.",
                station["Name"],
                distance,
            )
            return station, distance

        if not self._use_nominatim:
            raise StationNotFoundError(
                f"""No exact station match found for '{city_name}',
                and nominatim geocoding is disabled."""
            )

        raise StationNotFoundError(
            f"No air quality stations could be found after searching for '{city_name}'."
        )


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
        overall_aqi_value = self._get_aqi(city_searched)
        measurements_list = self._get_station_measurements(station_data)
        nearby_stations = self._get_nearby_stations_sorted(city_searched)

        measurements = []
        stations_used = [station_data.get("Name", "")]
        added_pollutants = set()

        for meas in measurements_list:
            code = meas.get("ComponentCode")
            added_pollutants.add(code)

            measurement_data = self._build_measurement_entry(
                meas,
                code,
                station_data,
                nearby_stations,
                stations_used
            )
            measurements.append(measurement_data)

        extra_pollutants = self._find_extra_pollutants(
            nearby_stations,
            station_data,
            added_pollutants
        )

        for code, (alt_station, alt_meas) in extra_pollutants.items():
            measurement_data = self._build_extra_pollutant_entry(
                alt_meas,
                alt_station,
                station_data,
                stations_used
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
                str(value),
                unit,
                code
            )
        elif code:
            value_float, status_text, sub_aqi, _ = self._find_measurement_fallback(
                code,
                unit,
                primary_station,
                nearby_stations,
                stations_used
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
                    if not alt_value:
                        continue

                    alt_value_float = float(alt_value)
                    if self._is_valid_measurement(alt_value_float):
                        try:
                            value_float, status_text, sub_aqi = self._process_valid_measurement(
                                str(alt_value), unit, code
                            )
                            if value_float is not None:
                                used_station_name = alt_station_name

                                if alt_station_name and alt_station_name not in stations_used:
                                    stations_used.append(alt_station_name)

                                _LOGGER.debug(
                                    "Fallback: Using station %s for pollutant %s in report",
                                    alt_station_name, code,
                                )
                                return value_float, status_text, sub_aqi, used_station_name
                        except (ValueError, TypeError):
                            continue

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
                        try:
                            extra_pollutants[code] = (alt_station, alt_meas)
                        except (ValueError, TypeError):
                            continue

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
            value_float = float(value) # type: ignore[arg-type]
            status_text = f"{value} {unit}"
            if code:
                sub_aqi = self._calculate_e_aqi_subindex(code, value_float)

            if alt_station_name and alt_station_name not in stations_used:
                stations_used.append(alt_station_name)

            _LOGGER.debug(
                "Fallback: Adding pollutant %s from station %s (not measured by 1st station %s).",
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


    def _get_city_coordinates_internal(self, city_name: str) -> tuple[float, float] | None:
        """
        Internal implementation for getting geographic coordinates for a city.
        Uses local cache, then Nominatim geocoding if enabled.

        :param city_name: City name to geocode
        :type city_name: str
        :return: (latitude, longitude) tuple or None
        :rtype: tuple[float, float] | None
        """
        if city_name in self._city_coordinate_cache:
            _LOGGER.debug("Coordinates for '%s' retrieved from local cache.", city_name)
            return self._city_coordinate_cache[city_name]

        if not self._use_nominatim or self._rate_limited_geocode is None:
            _LOGGER.debug("Nominatim geocoding disabled. Cannot lookup '%s'.", city_name)
            return None

        _LOGGER.debug("Attempting geocoding for '%s'.", city_name)
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
                _LOGGER.debug("Successfully geocoded '%s'.", city_name)
                return coords

            _warn(f"Nominatim geocoding failed for '{city_name}'.")
            return None

        except GeocoderUnavailable as exc:
            _warn(f"Nominatim unavailable. Falling back to exact station name matching: {exc}")
            return None
        except (requests.exceptions.RequestException, GeocoderServiceError) as exc:
            _warn(f"Geocoding service error for '{city_name}': {exc}")
            return None
