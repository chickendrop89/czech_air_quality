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

# pylint: disable=protected-access
# pylint: disable=wildcard-import, unused-wildcard-import

"""
Unit tests for constants and configuration values.
"""

import unittest
from urllib.parse import urlparse

# Discouraged, but let's do it for the sake of readibility
from src.czech_air_quality.const import *
from src.czech_air_quality import __version__


class TestURLConstants(unittest.TestCase):
    """Test URL endpoints."""

    def test_aq_data_url_is_valid(self):
        """Test that AQ_DATA_URL is a valid URL."""
        result = urlparse(AQ_DATA_URL)
        self.assertEqual(result.scheme, "https")
        self.assertIn("chmi.cz", result.netloc)


    def test_metadata_url_is_valid(self):
        """Test that METADATA_URL is a valid URL."""
        result = urlparse(METADATA_URL)
        self.assertEqual(result.scheme, "https")
        self.assertIn("chmi.cz", result.netloc)


    def test_urls_are_different(self):
        """Test that AQ_DATA_URL and METADATA_URL are different."""
        self.assertNotEqual(AQ_DATA_URL, METADATA_URL)


class TestTimeoutConstants(unittest.TestCase):
    """Test timeout configuration values."""

    def test_nominatim_timeout_positive(self):
        """Test that NOMINATIM_TIMEOUT is positive."""
        self.assertGreater(NOMINATIM_TIMEOUT, 0)


    def test_request_timeout_positive(self):
        """Test that REQUEST_TIMEOUT is positive."""
        self.assertGreater(REQUEST_TIMEOUT, 0)


    def test_nominatim_timeout_reasonable(self):
        """Test that NOMINATIM_TIMEOUT is reasonable (not excessive)."""
        self.assertLess(NOMINATIM_TIMEOUT, 60)


    def test_request_timeout_reasonable(self):
        """Test that REQUEST_TIMEOUT is reasonable (not excessive)."""
        self.assertLess(REQUEST_TIMEOUT, 120)


    def test_request_timeout_greater_than_nominatim(self):
        """Test that REQUEST_TIMEOUT >= NOMINATIM_TIMEOUT."""
        self.assertGreaterEqual(REQUEST_TIMEOUT, NOMINATIM_TIMEOUT)


class TestUserAgent(unittest.TestCase):
    """Test USER_AGENT string."""

    def test_user_agent_contains_library_name(self):
        """Test that USER_AGENT includes library name."""
        self.assertIn("czech_air_quality", USER_AGENT)


    def test_user_agent_contains_version(self):
        """Test that USER_AGENT includes version number."""
        self.assertIn(__version__, USER_AGENT)


    def test_user_agent_format(self):
        """Test USER_AGENT has standard format."""
        parts = USER_AGENT.split("/")
        self.assertEqual(len(parts), 2)


class TestEAQIBands(unittest.TestCase):
    """Test EAQI_BANDS structure and values."""

    def test_eaqi_bands_has_required_pollutants(self):
        """Test that EAQI_BANDS includes required pollutants."""
        required_pollutants = ["PM10", "PM2_5", "O3", "NO2", "SO2"]

        for pollutant in required_pollutants:
            self.assertIn(pollutant, EAQI_BANDS)


    def test_eaqi_bands_are_lists(self):
        """Test that EAQI_BANDS values are lists of tuples."""
        for _, bands in EAQI_BANDS.items():
            self.assertIsInstance(bands, list)
            self.assertGreater(len(bands), 0)

            for band in bands:
                self.assertIsInstance(band, tuple)
                self.assertEqual(len(band), 2)


    def test_eaqi_bands_values_ascending(self):
        """Test that EAQI band concentration limits are ascending."""
        for pollutant, bands in EAQI_BANDS.items():
            prev_concentration = -1

            for _, concentration in bands:
                self.assertGreater(
                    concentration,
                    prev_concentration,
                    msg=f"{pollutant} bands not in ascending order",
                )
                prev_concentration = concentration


    def test_eaqi_bands_aqi_values_positive(self):
        """Test that EAQI AQI values are positive."""
        for pollutant, bands in EAQI_BANDS.items():
            for aqi_value, _ in bands:
                self.assertGreater(
                    aqi_value, 0, msg=f"{pollutant} has non-positive AQI value"
                )


    def test_eaqi_bands_aqi_values_ascending(self):
        """Test that AQI values increase with concentration."""
        for pollutant, bands in EAQI_BANDS.items():
            prev_aqi = 0

            for aqi_value, _ in bands:
                self.assertGreaterEqual(
                    aqi_value, prev_aqi, msg=f"{pollutant} AQI values not ascending"
                )
                prev_aqi = aqi_value


    def test_pm10_bands_first_value(self):
        """Test PM10 specific band values (sanity check)."""
        pm10_bands = EAQI_BANDS["PM10"]
        # First band should be (1, 20) - Level 1 at 20 µg/m³
        self.assertEqual(pm10_bands[0], (1, 20))


    def test_o3_bands_first_value(self):
        """Test O3 specific band values (sanity check)."""
        o3_bands = EAQI_BANDS["O3"]
        # First band should be (1, 50) - Level 1 at 50 µg/m³
        self.assertEqual(o3_bands[0], (1, 50))


class TestEAQILevels(unittest.TestCase):
    """Test EAQI_LEVELS mapping."""

    def test_eaqi_levels_structure(self):
        """Test EAQI_LEVELS has proper structure."""
        self.assertIsInstance(EAQI_LEVELS, dict)
        self.assertGreater(len(EAQI_LEVELS), 0)


    def test_eaqi_levels_keys_are_numbers(self):
        """Test EAQI_LEVELS keys are numeric."""
        for key in EAQI_LEVELS:
            self.assertIsInstance(key, int)


    def test_eaqi_levels_values_are_strings(self):
        """Test EAQI_LEVELS values are strings."""
        for value in EAQI_LEVELS.values():
            self.assertIsInstance(value, str)


    def test_eaqi_levels_has_required_descriptions(self):
        """Test EAQI_LEVELS includes required descriptions."""
        required_descriptions = ["Good", "Fair", "Poor", "Very Poor"]

        all_descriptions = list(EAQI_LEVELS.values())

        for description in required_descriptions:
            self.assertIn(description, all_descriptions)


    def test_eaqi_levels_keys_ascending(self):
        """Test EAQI_LEVELS keys are in ascending order."""
        keys = sorted(EAQI_LEVELS.keys())
        self.assertEqual(keys, sorted(EAQI_LEVELS.keys()))


class TestCacheConstants(unittest.TestCase):
    """Test cache configuration constants."""

    def test_cache_file_name_not_empty(self):
        """Test CACHE_FILE_NAME is not empty."""
        self.assertGreater(len(CACHE_FILE_NAME), 0)


    def test_cache_file_name_has_extension(self):
        """Test CACHE_FILE_NAME has file extension."""
        self.assertTrue(CACHE_FILE_NAME.endswith(".json"))


    def test_cache_metadata_key_not_empty(self):
        """Test CACHE_METADATA_KEY is not empty."""
        self.assertGreater(len(CACHE_METADATA_KEY), 0)


    def test_etag_urls_structure(self):
        """Test ETAG_URLS has required keys."""
        required_keys = ["aq_data_etag", "metadata_etag"]

        for key in required_keys:
            self.assertIn(key, ETAG_URLS)


    def test_etag_urls_point_to_data_urls(self):
        """Test ETAG_URLS values match data URLs."""
        self.assertEqual(ETAG_URLS["aq_data_etag"], AQ_DATA_URL)
        self.assertEqual(ETAG_URLS["metadata_etag"], METADATA_URL)


class TestMiscConstants(unittest.TestCase):
    """Test miscellaneous constants."""

    def test_chmi_error_threshold(self):
        """Test CHMI_ERROR_THRESHOLD is defined."""
        self.assertIsNotNone(CHMI_ERROR_THRESHOLD)

    def test_timestamp_key_not_empty(self):
        """Test TIMESTAMP_KEY is not empty."""
        self.assertGreater(len(TIMESTAMP_KEY), 0)

    def test_etags_key_not_empty(self):
        """Test ETAGS_KEY is not empty."""
        self.assertGreater(len(ETAGS_KEY), 0)


if __name__ == "__main__":
    unittest.main()
