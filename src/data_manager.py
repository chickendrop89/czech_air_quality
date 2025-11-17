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
Data management, caching, and download manager
"""

from datetime import datetime,timezone
import json
import os
import csv
import io
import tempfile
import logging
import requests

# Can't import DataDownloadError directly
import src
import src.const


_LOGGER = logging.getLogger(__name__)

class DataManager:
    """
    Manages data caching, downloading, and parsing for air quality data.
    Handles ETag-based conditional downloads, local caching, and data combination.
    """

    def __init__(self, disable_caching: bool = False, request_timeout: int = src.const.REQUEST_TIMEOUT):
        """
        Initialize the DataManager.

        :param disable_caching: Skip caching, force fresh download every time
        :type disable_caching: bool
        :param request_timeout: HTTP request timeout in seconds
        :type request_timeout: int
        """
        self._request_timeout = request_timeout
        self._raw_data_json = None
        self._actualized_time = datetime.min
        self._last_download_status = "Not yet run"
        self._etags = {}
        self._disable_caching = disable_caching
        self._cache_file_path = os.path.join(
            tempfile.gettempdir(),
            src.const.CACHE_FILE_NAME
        )


    @property
    def raw_data_json(self) -> str | None:
        """Get raw JSON data string."""
        return self._raw_data_json

    @property
    def actualized_time(self) -> datetime:
        """Get timestamp when data was last actualized."""
        return self._actualized_time

    @property
    def last_download_status(self) -> str:
        """Get status message from last download attempt."""
        return self._last_download_status


    def ensure_latest_data(self) -> None:
        """
        Ensure raw data is loaded and fresh.
        Loads from cache if available, verifies freshness via ETags, downloads if needed.

        :raises DataDownloadError: If no data can be retrieved from cache or download
        """
        cache_is_fresh = self._load_from_cache()

        if self._disable_caching:
            _LOGGER.info("Caching disabled. Forcing fresh download.")
            self._download_data()
        elif not cache_is_fresh:
            self._download_data()
        elif not self.is_data_fresh():
            _LOGGER.info("Cached data is stale according to ETag check. Downloading fresh data.")
            self._download_data()

        if not self._raw_data_json:
            raise src.DataDownloadError(
                "Could not retrieve any air quality data from download or cache."
            )


    def is_data_fresh(self) -> bool:
        """
        Check if cached data is fresh via ETag validation.

        Performs conditional GET requests against metadata files.

        :return: True if all resources are 304 (Not Modified) or caching disabled; 
                 False if any resource is 200 (Modified) or network error
        :rtype: bool
        """

        if self._disable_caching:
            return True

        if not self._raw_data_json:
            return False

        is_modified = False
        all_304_or_ok = True

        _LOGGER.info(
            "Checking server ETag freshness for metadata at %s.", 
            datetime.now(timezone.utc).isoformat()
        )

        try:
            for etag_key, url in src.const.ETAG_URLS.items():
                response = self._perform_conditional_download(url, etag_key)

                if response.status_code == 200:
                    is_modified = True
                    _LOGGER.info("Resource %s was modified (200). Full download required.", url)
                    break

                if response.status_code not in (200, 304):
                    all_304_or_ok = False
                    _LOGGER.warning(
                        "Server check failed for %s: Status %d",
                        url,
                        response.status_code,
                    )
                    break

            if is_modified:
                return False

            if all_304_or_ok:
                return True

        except requests.exceptions.RequestException as exc:
            _LOGGER.error(
                "Server ETag freshness check failed due to network error: %s", exc
            )
            return False

        return False


    def _load_from_cache(self) -> bool:
        """
        Load data and ETags from cache file.

        :return: True if successfully loaded, False otherwise
        :rtype: bool
        :raises OSError: If cache file cannot be read
        :raises json.JSONDecodeError: If cache file is corrupted
        """

        if self._disable_caching:
            return False
        try:
            with open(self._cache_file_path, "r", encoding="utf-8") as file:
                cache_data = json.load(file)

            metadata    = cache_data.pop(src.const.CACHE_METADATA_KEY, {})
            cache_time  = metadata.get(src.const.TIMESTAMP_KEY)
            self._etags = metadata.get(src.const.ETAGS_KEY, {})

            if cache_time:
                self._actualized_time = datetime.fromisoformat(cache_time)
            else:
                return False

            cache_data[src.const.CACHE_METADATA_KEY] = metadata
            self._raw_data_json = json.dumps(cache_data, ensure_ascii=False)
            self._last_download_status = f"Loaded data from cache (pending ETag validation). Actualized at {self._actualized_time.strftime('%Y-%m-%d %H:%M')}"

            _LOGGER.info(
                "Loaded data and ETags from cache. Timestamp: %s. Awaiting network validation.",
                self._actualized_time.strftime("%Y-%m-%d %H:%M"),
            )
            return True

        except (json.JSONDecodeError, OSError) as exc:
            _LOGGER.debug("Cache load failed: %s", exc)
            self._actualized_time = datetime.min
            return False


    def _save_to_cache(self, data_json_str: str) -> None:
        """
        Save raw JSON data with timestamp and ETags to cache file.

        :param data_json_str: JSON string to cache
        :type data_json_str: str
        """
        if not data_json_str or self._disable_caching:
            return

        try:
            cache_data = json.loads(data_json_str)
            metadata = {
                src.const.TIMESTAMP_KEY: datetime.now(timezone.utc).isoformat(),
                src.const.ETAGS_KEY: self._etags,
            }
            cache_data[src.const.CACHE_METADATA_KEY] = metadata

            os.makedirs(
                os.path.dirname(self._cache_file_path),
                exist_ok=True
            )

            with open(self._cache_file_path, "w", encoding="utf-8") as file:
                json.dump(cache_data, file, ensure_ascii=False)

            _LOGGER.info("Fresh data and ETags saved to cache.")
        except (OSError, json.JSONDecodeError) as exc:
            _LOGGER.warning("Could not save data to cache file: %s", exc)


    def _download_data(self) -> None:
        """
        Download latest air quality data.

        :raises DataDownloadError: If download fails and no cache available
        """

        _LOGGER.info(
            "Attempting to download fresh data from OpenData CHMI endpoints..."
        )

        download_results = {}
        is_modified = False

        try:
            for etag_key, url in src.const.ETAG_URLS.items():
                response = self._perform_conditional_download(url, etag_key)

                if response.status_code == 304:
                    _LOGGER.info(
                        "Resource %s not modified (304). Using cached version.", url
                    )
                    download_results[etag_key] = {"status": 304}
                else:
                    response.raise_for_status()
                    is_modified = True

                    download_results[etag_key] = {
                        "status": response.status_code,
                        "content": response.text if "csv" in url else response.json(),
                    }

            if not is_modified and self._raw_data_json:
                timestamp = datetime.now(timezone.utc)
                self._actualized_time = timestamp
                self._last_download_status = f"Success. Cache refreshed at {timestamp.strftime('%Y-%m-%d %H:%M')}"
                self._save_to_cache(self._raw_data_json)
                _LOGGER.info("All resources were Not Modified (304). Using existing data.")
                return

            metadata_data = download_results.get("metadata_etag", {}).get("content")
            aq_csv_str = download_results.get("aq_data_etag", {}).get("content")

            if metadata_data is None or aq_csv_str is None:
                raise src.DataDownloadError(
                    "Failed to download required data files. At least one file is missing or invalid."
                )

            combined_data = self._combine_downloaded_data(
                metadata_data, aq_csv_str
            )

            timestamp = datetime.now(timezone.utc)
            self._raw_data_json = json.dumps(combined_data, ensure_ascii=False)
            self._actualized_time = timestamp
            self._last_download_status = (
                f"Success. Data downloaded at {timestamp.strftime('%Y-%m-%d %H:%M')}"
            )
            _LOGGER.info(
                "Download successful. Data downloaded at %s.",
                timestamp.strftime("%Y-%m-%d %H:%M"),
            )
            self._save_to_cache(self._raw_data_json)

        except requests.exceptions.RequestException as exc:
            self._last_download_status = f"Download failed: {exc}"

            if not self._raw_data_json and not self._load_from_cache():
                raise src.DataDownloadError(
                    f"Failed to download and no cache data is available: {exc}"
                ) from exc

            _LOGGER.warning("Download failed: %s. Falling back to cached data.", exc)

        except json.JSONDecodeError as exc:
            self._last_download_status = (
                f"Download successful but data parse failed: {exc}"
            )

            if not self._raw_data_json and not self._load_from_cache():
                raise src.DataDownloadError(
                    f"Downloaded data is invalid and no cache data is available: {exc}"
                ) from exc

            _LOGGER.warning(
                "Downloaded data is invalid: %s. Falling back to cached data.", exc
            )


    def _combine_downloaded_data(
        self, metadata_json: dict, aq_csv_str: str
    ) -> dict:
        """f
        Combine metadata and CSV data into unified structure.

        :param metadata_json: Parsed metadata JSON
        :type metadata_json: dict
        :param aq_csv_str: Raw CSV string
        :type aq_csv_str: str
        :return: Combined data dictionary
        :rtype: dict
        :raises DataDownloadError: If inputs are invalid types
        """

        if not isinstance(metadata_json, dict):
            raise src.DataDownloadError("Metadata JSON is not a valid dictionary")
        if not isinstance(aq_csv_str, str):
            raise src.DataDownloadError("AQ CSV data is not a valid string")

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
            id_reg = normalized_row.get("idRegistration", "").strip()
            start_time = normalized_row.get("startTime", "").strip()
            id_value_type = normalized_row.get("idValueType", "").strip()
            value = normalized_row.get("value", "").strip()

            if id_reg:
                if id_reg not in combined["Measurements"]:
                    combined["Measurements"][id_reg] = []

                combined["Measurements"][id_reg].append(
                    {
                        "startTime": start_time,
                        "idValueType": id_value_type,
                        "value": value,
                    }
                )

        return combined


    def _perform_conditional_download(self, url: str, etag_key: str) -> requests.Response:
        """
        Perform GET request with ETag conditional headers.

        :param url: URL to download
        :type url: str
        :param etag_key: Key for storing/retrieving ETag
        :type etag_key: str
        :return: Response object
        :rtype: requests.Response
        """

        headers = {
            "User-Agent": src.const.USER_AGENT,
            "Accept": "text/csv, application/json, application/octet-stream"
        } # accept "text/csv" for futureproofing

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
