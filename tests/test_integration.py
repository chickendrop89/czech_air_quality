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

"""
Integration tests for the czech_air_quality library.
"""

import unittest
from unittest.mock import patch
from datetime import datetime, timezone

from czech_air_quality.api import (
    AirQuality,
    StationNotFoundError,
)


class TestStationFinding(unittest.TestCase):
    """Test station finding and nearby station logic."""

    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_find_nearest_station_not_found(self, mock_dm):
        """Test exception when station not found."""
        mock_dm.return_value.raw_data_json = '{"Localities": []}'
        mock_dm.return_value.is_data_fresh.return_value = True
        mock_dm.return_value._raw_data_json = '{"Localities": []}'

        aq = AirQuality(auto_load=False)
        aq._use_nominatim = False
        aq._data = {"Localities": []}

        aq._all_stations = [
            {
                "Name": "Praha - Vinohrady",
                "LocalityCode": "1",
                "Region": "Praha",
                "Latitude": 50.0755,
                "Longitude": 14.4378,
                "IdRegistrations": ["1001", "1002"],
            },
            {
                "Name": "Brno",
                "LocalityCode": "2",
                "Region": "Jihomoravský",
                "Latitude": 49.1951,
                "Longitude": 16.6068,
                "IdRegistrations": ["2001", "2002"],
            },
            {
                "Name": "Ostrava",
                "LocalityCode": "3",
                "Region": "Moravskoslezský",
                "Latitude": 49.8209,
                "Longitude": 18.2625,
                "IdRegistrations": ["3001", "3002"],
            },
        ]

        with self.assertRaises(StationNotFoundError):
            aq.find_nearest_station("NonexistentCity")


class TestAirQualityProperties(unittest.TestCase):
    """Test AirQuality class properties and data access."""

    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_all_stations_property(self, mock_dm):
        """Test all_stations property returns list."""
        mock_dm.return_value.raw_data_json = None
        aq = AirQuality(auto_load=False)

        result = aq.all_stations
        self.assertIsInstance(result, list)


    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_component_lookup_property(self, mock_dm):
        """Test component_lookup property returns dict."""
        mock_dm.return_value.raw_data_json = None
        aq = AirQuality(auto_load=False)

        result = aq.component_lookup
        self.assertIsInstance(result, dict)


    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_raw_data_property(self, mock_dm):
        """Test raw_data property returns dict."""
        mock_dm.return_value.raw_data_json = None
        aq = AirQuality(auto_load=False)

        result = aq.raw_data
        self.assertIsInstance(result, dict)


    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_actualized_time_property(self, mock_dm):
        """Test actualized_time property returns datetime."""
        mock_dm.return_value.raw_data_json = None
        mock_dm.return_value.actualized_time = datetime.now(timezone.utc)

        aq = AirQuality(auto_load=False)

        result = aq.actualized_time
        self.assertIsInstance(result, datetime)


    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_is_data_fresh_property(self, mock_dm):
        """Test is_data_fresh property returns bool."""
        mock_dm.return_value.raw_data_json = None
        mock_dm.return_value.is_data_fresh.return_value = True

        aq = AirQuality(auto_load=False)

        result = aq.is_data_fresh
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
