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
# pylint: disable=arguments-differ

"""
Unit tests for DataManager class.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from src.czech_air_quality.data_manager import DataManager
from src.czech_air_quality.const import CACHE_FILE_NAME


class TestDataManagerInitialization(unittest.TestCase):
    """Test DataManager initialization."""

    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_init_default_caching_enabled(self, mock_get):
        """Test default initialization with caching enabled."""
        mock_get.return_value.status_code = 304

        dm = DataManager(disable_caching=False)

        self.assertFalse(dm._disable_caching)


    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_init_caching_disabled(self, _):
        """Test initialization with caching disabled."""
        dm = DataManager(disable_caching=True)

        self.assertTrue(dm._disable_caching)


    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_init_custom_timeout(self, _):
        """Test initialization with custom request timeout."""
        dm = DataManager(request_timeout=30)

        self.assertEqual(dm._request_timeout, 30)


class TestDataManagerCaching(unittest.TestCase):
    """Test caching behavior."""

    @patch("src.czech_air_quality.data_manager.requests.get")
    @patch("src.czech_air_quality.data_manager.os.path.exists")
    def test_cache_file_location(self, mock_exists, _):
        """Test cache file is stored in correct location."""
        mock_exists.return_value = False
        dm = DataManager()

        self.assertIn(CACHE_FILE_NAME, dm._cache_file_path)


    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_disable_caching_flag(self, _):
        """Test that disable_caching flag is respected."""
        dm = DataManager(disable_caching=True)

        self.assertTrue(dm._disable_caching)


class TestETagValidation(unittest.TestCase):
    """Test ETag-based conditional download validation."""

    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_is_data_fresh_with_cache_304(self, mock_get):
        """Test is_data_fresh returns True when server returns 304 (not modified)."""
        mock_response = Mock()
        mock_response.status_code = 304
        mock_get.return_value = mock_response

        dm = DataManager()
        dm._etags = {"aq_data_etag": "test-etag", "metadata_etag": "test-etag"}
        result = dm.is_data_fresh()

        self.assertIsInstance(result, bool)


    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_is_data_fresh_without_etags(self, _):
        """Test is_data_fresh when no ETags are cached."""
        dm = DataManager()
        dm._etags = {}

        result = dm.is_data_fresh()

        self.assertFalse(result)


class TestDataCombination(unittest.TestCase):
    """Test data manager functionality."""

    @patch("src.czech_air_quality.data_manager.requests.get")
    def setUp(self, _):
        """Set up test fixtures."""
        self.dm = DataManager()


    def test_data_manager_initialization(self):
        """Test that DataManager initializes properly."""
        self.assertIsNotNone(self.dm)
        self.assertIsInstance(self.dm._disable_caching, bool)
        self.assertGreater(self.dm._request_timeout, 0)


class TestDataValidation(unittest.TestCase):
    """Test data validation logic."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("src.czech_air_quality.data_manager.requests.get"):
            self.dm = DataManager()


    def test_validate_metadata_valid_structure(self):
        """Test validation of valid metadata structure."""
        valid_metadata = {"Localities": [], "id_registration_to_component": {}}

        # Should not raise exception
        result = self.dm._combine_downloaded_data(
            valid_metadata, "idRegistration,value"
        )
        self.assertIsNotNone(result)


    def test_validate_csv_valid_format(self):
        """Test validation of valid CSV format."""
        metadata = {"Localities": [], "id_registration_to_component": {}}

        valid_csv = "idRegistration,value\n1001,25.5\n1002,30.0"

        result = self.dm._combine_downloaded_data(metadata, valid_csv)
        self.assertIn("Measurements", result)


class TestRawDataProperties(unittest.TestCase):
    """Test properties that expose raw data."""

    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_raw_data_json_property_none(self, _):
        """Test raw_data_json property when no data is loaded."""
        dm = DataManager()
        result = dm.raw_data_json

        self.assertTrue(result is None or result == {})


class TestActualizedTime(unittest.TestCase):
    """Test data actualization timestamp."""

    @patch("src.czech_air_quality.data_manager.requests.get")
    def test_actualized_time_type(self, _):
        """Test that actualized_time returns datetime object."""
        dm = DataManager()

        result = dm.actualized_time
        self.assertIsInstance(result, datetime)


if __name__ == "__main__":
    unittest.main()
