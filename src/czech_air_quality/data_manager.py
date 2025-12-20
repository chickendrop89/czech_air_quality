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
Data management.
"""

from datetime import (
    datetime,
    timezone,
    timedelta
)
import json
import os
import csv
import io
import tempfile
import logging
import requests

from . import const, _warn
from . import DataDownloadError, CacheError

_LOGGER = logging.getLogger(__name__)


class DataManager:
    """
    Manages data caching, downloading, and parsing for air quality data.
    Handles ETag-based conditional downloads, local caching, and data combination.
    """
    def __init__(self, disable_caching: bool = False,
            request_timeout: int = const.REQUEST_TIMEOUT):
        """
        Initialize the DataManager.

        :param disable_caching: Skip caching, force fresh download every time
        :type disable_caching: bool
        :param request_timeout: HTTP request timeout in seconds
        :type request_timeout: int
        """
        self._actualized_time = datetime.min

        self._raw_data_json: str | None = None
        self._raw_metadata_json: dict | None = None
        self._raw_aq_csv_str: str | None = None

        self._request_timeout = request_timeout
        self._disable_caching = disable_caching
        self._cache_dir_path = os.path.join(
            tempfile.gettempdir(),
            const.CACHE_DIR_NAME
        )
        self._cache_file_path = os.path.join(
            self._cache_dir_path,
            const.CACHE_FILE_NAME
        )
        self._etags = {}


    @property
    def raw_data_json(self) -> str | None:
        """Get raw JSON data string."""
        return self._raw_data_json

    @property
    def actualized_time(self) -> datetime:
        """Get timestamp when data was last actualized."""
        return self._actualized_time


    def ensure_latest_data(self, force_fetch: bool = False) -> None:
        """
        Ensure data is loaded and fresh using a freshness check strategy.

        The freshness check works as follows:
        1. Load Cache: Attempt to load data from the local cache file.
        2. Freshness Check: If cache is present, check if its age exceeds 20 minutes.
        3. ETag Validation: If cache is expired, perform a HTTP-HEAD ETag check.
           - If server returns 304 (Not Modified), trust cache for another 20 minutes.
           - If server returns 200 (Modified), download full data.
        4. Force Fetch/No Cache: If caching is disabled or force_fetch is True,
           download fresh data immediately.
        5. Offline Fallback: If network is unavailable, use stale cache if available.

        :param force_fetch: If True, bypass all cache logic and download fresh data
        :type force_fetch: bool
        :raises DataDownloadError: If no data can be retrieved from downloads or cache
        """
        cache_loaded = self._load_from_cache()

        if self._disable_caching or force_fetch:
            _LOGGER.debug("Caching disabled. Forcing fresh download.")
            self._download_data()
            return

        if not cache_loaded:
            _LOGGER.debug("No cache file found. Downloading fresh data.")
            self._download_data()
            return

        cache_age = self._get_cache_file_age()
        cache_expired = (cache_age is None) or (cache_age > timedelta(minutes=20))

        if not cache_expired:
            _LOGGER.debug("Cache is recent (Age: %s).", cache_age)
            return

        _LOGGER.debug("Cache expired (Age: %s). Doing an ETag check", cache_age)

        if self.is_data_fresh():
            _LOGGER.debug(
                "ETag is still fresh. %s",
                "Marking the data as fresh for another 20 minutes."
            )
            self._refresh_cache_validation_timestamp()
        else:
            _LOGGER.debug("ETag is stale. Downloading new data.")
            try:
                self._download_data()
            except DataDownloadError:
                if self._raw_data_json:
                    _warn("Download failed, but stale cache is available. Using it")
                else:
                    raise

        if not self._raw_data_json:
            raise DataDownloadError(
                "Could not retrieve any air quality data from downloads or cache."
            )


    def is_data_fresh(self) -> bool:
        """
        Check if cached data is fresh via HTTP-HEAD ETag validation with the server.

        :return: True if all resources return 304 (Not Modified) or caching is disabled;
                 False if any resource is 200 (Modified) or if network errors occur
        :rtype: bool
        """

        if self._disable_caching:
            return True

        if not self._raw_data_json:
            return False

        is_modified = False
        all_304_or_ok = True

        _LOGGER.debug(
            "Checking server ETag freshness for metadata at %s.", 
            datetime.now(timezone.utc).isoformat()
        )

        try:
            for etag_key, url in const.ETAG_URLS.items():
                response = self._perform_conditional_head(url, etag_key)

                if response.status_code == 200:
                    is_modified = True
                    _LOGGER.debug("Resource %s was modified (200). Full download required.", url)
                    break

                if response.status_code not in (200, 304):
                    all_304_or_ok = False
                    _warn(f"Server check failed for {url}: Status {response.status_code}")
                    break

            if is_modified:
                return False

            if all_304_or_ok:
                return True

        except requests.exceptions.RequestException as exc:
            _warn(f"ETag freshness check failed due to network error: {exc}. Using cached data")
            return False

        return False


    def _load_from_cache(self) -> bool:
        """
        Load data, ETags, and raw components (metadata/CSV) from cache file.

        :return: True if successfully loaded all required components, False otherwise
        :rtype: bool
        :raises OSError: If cache file cannot be read or lacks permissions
        :raises json.JSONDecodeError: If cache file contains invalid JSON
        """
        if self._disable_caching:
            return False
        try:
            with open(self._cache_file_path, "r", encoding="utf-8") as file:
                cache_data = json.load(file)

            metadata    = cache_data.pop(const.CACHE_METADATA_KEY, {})
            cache_time  = metadata.get(const.TIMESTAMP_KEY)

            self._etags = metadata.get(const.ETAGS_KEY, {})
            self._raw_metadata_json = cache_data.get("raw_metadata_json")
            self._raw_aq_csv_str = cache_data.get("raw_aq_csv_str")
            self._raw_data_json = cache_data.get("combined_data")

            if not cache_time or not self._raw_data_json or not self._raw_metadata_json or not self._raw_aq_csv_str:
                self._etags = {}
                return False

            try:
                self._actualized_time = datetime.fromisoformat(cache_time)
            except ValueError as exc:
                _LOGGER.debug("Invalid cache timestamp format: %s", exc)
                return False

            _LOGGER.debug(
                "Loaded data and ETags from cache. Timestamp: %s. Awaiting network validation.",
                self._actualized_time.strftime("%Y-%m-%d %H:%M"),
            )
            return True

        except (json.JSONDecodeError, OSError) as exc:
            _LOGGER.debug("Cache load failed: %s", exc)
            self._actualized_time = datetime.min

            self._raw_metadata_json = None
            self._raw_aq_csv_str = None
            self._raw_data_json = None
            return False


    def _save_to_cache(self, data_json_str: str) -> None:
        """
        Save raw JSON data with timestamp and ETags to cache file.

        :param data_json_str: Combined JSON string to cache
        :type data_json_str: str
        """
        if not data_json_str or self._disable_caching:
            return

        try:
            timestamp = datetime.now(timezone.utc).isoformat()

            self._write_cache_file(
                data_json_str,
                self._raw_metadata_json,
                self._raw_aq_csv_str,
                timestamp
            )

            _LOGGER.debug("Fresh data and ETags saved to cache.")
        except CacheError as exc:
            _warn(f"Could not save data to cache file: {exc}")


    def _write_cache_file(self, combined_data_json: str,
        metadata_json: dict | None, aq_csv_str: str | None, timestamp: str) -> None:
        """
        Write cache data to the file system.

        :param combined_data_json: Combined JSON data string
        :type combined_data_json: str
        :param metadata_json: Parsed metadata dictionary (may be None during initial load)
        :type metadata_json: dict | None
        :param aq_csv_str: Raw CSV data string (may be None during initial load)
        :type aq_csv_str: str | None
        :param timestamp: ISO format timestamp of cache creation
        :type timestamp: str
        :raises CacheError: If file system operations fail
        """
        try:
            cache_data = {
                "combined_data": combined_data_json,
                "raw_metadata_json": metadata_json,
                "raw_aq_csv_str": aq_csv_str,
                const.CACHE_METADATA_KEY: {
                    const.TIMESTAMP_KEY: timestamp,
                    const.ETAGS_KEY: self._etags,
                }
            }

            os.makedirs(
                self._cache_dir_path,
                exist_ok=True,
                mode=0o700 # rwx only for the owner (dir)
            )

            with open(self._cache_file_path, "w", encoding="utf-8") as file:
                json.dump(cache_data, file, ensure_ascii=False)

            # rwx only for the owner (file)
            os.chmod(self._cache_file_path, 0o600)

        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError(f"File system error during cache write: {exc}") from exc


    def _download_data(self) -> None:
        """
        Download latest air quality data.

        :raises DataDownloadError: If download fails and no cache available
        """
        _LOGGER.debug("Attempting to download fresh data.")

        download_results = {}
        is_modified = False

        raw_metadata_content: dict | None = None
        raw_aq_csv_content: str | None = None

        try:
            for etag_key, url in const.ETAG_URLS.items():
                response = self._perform_conditional_download(url, etag_key)

                if response.status_code == 304:
                    _LOGGER.debug("Resource %s not modified. Using cached version.", url)
                    download_results[etag_key] = {"status": 304}
                    continue

                response.raise_for_status()
                is_modified = True

                content = response.text if "csv" in url else response.json()

                if etag_key == "metadata_etag":
                    if isinstance(content, dict):
                        raw_metadata_content = content
                    else:
                        _warn(f"Invalid metadata format from {url}")

                elif etag_key == "aq_data_etag":
                    if isinstance(content, str):
                        raw_aq_csv_content = content
                    else:
                        _warn(f"Invalid CSV format from {url}")

                download_results[etag_key] = {
                    "status": response.status_code,
                    "content": content
                }

            if not is_modified:
                _LOGGER.debug("All resources are unmodified. Using existing data.")
                return

            metadata_data: dict | None = raw_metadata_content or self._raw_metadata_json
            aq_csv_str: str | None = raw_aq_csv_content or self._raw_aq_csv_str

            if metadata_data is None or aq_csv_str is None:
                raise DataDownloadError(
                    "Failed to download required data files. "
                    "At least one file is missing or invalid."
                )

            self._raw_metadata_json = metadata_data
            self._raw_aq_csv_str = aq_csv_str

            combined_data = self._combine_downloaded_data(
                metadata_data,
                aq_csv_str
            )
            timestamp = datetime.now(timezone.utc)

            self._actualized_time = timestamp
            self._raw_data_json = json.dumps(
                combined_data, 
                ensure_ascii=False
            )

            _LOGGER.debug(
                "Download successful. Data downloaded at %s.",
                timestamp.strftime("%Y-%m-%d %H:%M"),
            )
            self._save_to_cache(self._raw_data_json)

        except requests.exceptions.RequestException as exc:
            if not self._raw_data_json and not self._load_from_cache():
                raise DataDownloadError(
                    f"Failed to download and no cache data is available: {exc}"
                ) from exc

            _warn(f"Download failed: {exc}. Falling back to cached data.")

        except json.JSONDecodeError as exc:
            if not self._raw_data_json and not self._load_from_cache():
                raise DataDownloadError(
                    f"Downloaded data is invalid and no cache data is available: {exc}"
                ) from exc

            _warn(f"Downloaded data is invalid: {exc}. Falling back to cached data.")


    def _combine_downloaded_data(self, metadata_json: dict, aq_csv_str: str) -> dict:
        """
        Combine metadata and CSV data into a unified structure for caching and processing.

        Creates a structured dictionary containing:
        - Localities with station information (coordinates, region)
        - ID registration to component mappings (pollutant codes, names, units)
        - Measurements organized by ID registration with values and timestamps

        :param metadata_json: Parsed metadata JSON from CHMI (must be a dict)
        :type metadata_json: dict
        :param aq_csv_str: Raw CSV string with air quality measurements
        :type aq_csv_str: str
        :return: Combined data dictionary with Localities, Measurements, and component mappings
        :rtype: dict
        :raises DataDownloadError: If inputs are invalid types or malformed
        """

        if not isinstance(metadata_json, dict):
            raise DataDownloadError("Metadata JSON is not a valid dictionary")
        if not isinstance(aq_csv_str, str):
            raise DataDownloadError("AQ CSV data is not a valid string")

        combined = {
            "Actualized": datetime.now(timezone.utc).isoformat(),
            "Localities": [],
            "Measurements": {},
            "id_registration_to_component": {},
        }

        if "data" in metadata_json and "Localities" in metadata_json["data"]:
            for locality in metadata_json["data"]["Localities"]:
                locality_code = locality.get("LocalityCode")
                locality_name = locality.get("Name")
                localization = locality.get("Localization", {})

                station_entry = {
                    "LocalityCode": locality_code,
                    "Name": locality_name,
                    "Region": locality.get("BasicInfo", {}).get("Region", ""),
                    "Lat": localization.get("LatAsNumber"),
                    "Lon": localization.get("LonAsNumber"),
                    "IdRegistrations": [],
                }

                for program in locality.get("MeasuringPrograms", []):
                    for measurement in program.get("Measurements", []):
                        id_reg = measurement.get("IdRegistration")
                        if id_reg:
                            station_entry["IdRegistrations"].append(id_reg)
                            combined["id_registration_to_component"][str(id_reg)] = {
                                "ComponentCode": measurement.get("ComponentCode"),
                                "ComponentName": measurement.get("ComponentName"),
                                "Unit": measurement.get("UnitAsUNICODE")
                                or measurement.get("UnitAsASCII", ""),
                            }

                combined["Localities"].append(station_entry)

        aq_reader = csv.DictReader(io.StringIO(aq_csv_str))

        for row in aq_reader:
            normalized_row = {k.strip(): v for k, v in row.items()}
            id_reg     = normalized_row.get("idRegistration", "").strip()
            start_time = normalized_row.get("startTime", "").strip()
            id_value_type = normalized_row.get("idValueType", "").strip()
            value         = normalized_row.get("value", "").strip()

            if id_reg:
                if id_reg not in combined["Measurements"]:
                    combined["Measurements"][id_reg] = []

                combined["Measurements"][id_reg].append({
                        "startTime": start_time,
                        "idValueType": id_value_type,
                        "value": value,
                    }
                )

        return combined


    def _perform_conditional_head(self, url: str, etag_key: str) -> requests.Response:
        """
        Perform a HEAD request with ETag conditional headers to check for modification.

        :param url: URL to check for updates
        :type url: str
        :param etag_key: Key for storing/retrieving ETag value
        :type etag_key: str
        :return: Response object (200=modified, 304=not modified)
        :rtype: requests.Response
        """
        headers = const.REQUEST_HEADERS.copy()

        if etag_key in self._etags:
            headers["If-None-Match"] = self._etags[etag_key]

        response = requests.head(url,
            headers=headers,
            timeout=self._request_timeout
        )

        return response


    def _perform_conditional_download(self, url: str, etag_key: str) -> requests.Response:
        """
        Perform GET request with ETag conditional headers.

        :param url: URL to download from
        :type url: str
        :param etag_key: Key for storing/retrieving ETag
        :type etag_key: str
        :return: Response object with status code and content
        :rtype: requests.Response
        """
        headers = const.REQUEST_HEADERS.copy()

        if etag_key in self._etags:
            headers["If-None-Match"] = self._etags[etag_key]

        response = requests.get(url,
            headers=headers,
            timeout=self._request_timeout
        )
        new_etag = response.headers.get("ETag")

        if new_etag:
            self._etags[etag_key] = new_etag

        return response


    def _refresh_cache_validation_timestamp(self) -> None:
        """
        Updates the cache file timestamp and _actualized_time after a successful
        304 (Not Modified) ETag validation, resetting the 20-minute cache timer.
        """
        if self._disable_caching or not self._raw_data_json:
            return

        try:
            current_timestamp = datetime.now(timezone.utc)

            self._write_cache_file(
                self._raw_data_json,
                self._raw_metadata_json,
                self._raw_aq_csv_str,
                current_timestamp.isoformat()
            )

            self._actualized_time = current_timestamp

            _LOGGER.debug(
                "Cache verified fresh and timestamp updated to %s",
                current_timestamp.strftime("%H:%M")
            )
        except (json.JSONDecodeError, OverflowError) as exc:
            _warn(f"Could not update cache timestamp: {exc}")


    def _get_cache_file_age(self) -> timedelta | None:
        """
        Reads the file timestamp of a cache file.

        :return: the current age of a file
        :rtype: timedelta | None
        """
        if not os.path.exists(self._cache_file_path):
            return None

        try:
            file_mod_timestamp = os.path.getmtime(self._cache_file_path)
            file_mod_datetime = datetime.fromtimestamp(
                timestamp=file_mod_timestamp,
                tz=timezone.utc
            )
            return datetime.now(timezone.utc) - file_mod_datetime
        except OSError as exc:
            _warn(f"Could not read cache file modification time: {exc}")
            return None
