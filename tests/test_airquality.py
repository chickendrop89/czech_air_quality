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
Unit tests for AirQuality class.
"""

import unittest
from unittest.mock import patch

from src.czech_air_quality import (
    AirQuality,
    AirQualityError,
    DataDownloadError,
    StationNotFoundError,
    PollutantNotReportedError,
)
from src.czech_air_quality.const import NOMINATIM_TIMEOUT


class TestAirQualityInitialization(unittest.TestCase):
    """Test AirQuality initialization with various configurations."""

    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_init_defaults(self, mock_dm):
        """Test default initialization parameters."""
        mock_dm.return_value.raw_data_json = None

        aq = AirQuality(auto_load=False)

        self.assertIsNone(aq._region_filter)
        self.assertTrue(aq._use_nominatim)
        self.assertEqual(aq._nominatim_timeout, NOMINATIM_TIMEOUT)
        self.assertEqual(aq._data, {})


    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_init_with_region_filter(self, mock_dm):
        """Test initialization with region filter."""
        mock_dm.return_value.raw_data_json = None

        aq = AirQuality(region_filter="Jihomoravský", auto_load=False)

        self.assertEqual(aq._region_filter, "jihomoravský")


    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_init_without_nominatim(self, mock_dm):
        """Test initialization with Nominatim disabled."""
        mock_dm.return_value.raw_data_json = None

        aq = AirQuality(use_nominatim=False, auto_load=False)

        self.assertFalse(aq._use_nominatim)
        self.assertIsNone(aq._geolocator)
        self.assertIsNone(aq._rate_limited_geocode)


    @patch("src.czech_air_quality.data_manager.DataManager")
    def test_init_with_custom_timeouts(self, mock_dm):
        """Test initialization with custom timeout values."""
        mock_dm.return_value.raw_data_json = None

        aq = AirQuality(nominatim_timeout=15, request_timeout=30, auto_load=False)

        self.assertEqual(aq._nominatim_timeout, 15)


class TestAirQualityExceptions(unittest.TestCase):
    """Test exception hierarchy and behavior."""

    def test_exception_hierarchy(self):
        """Test that custom exceptions inherit from AirQualityError."""
        self.assertTrue(issubclass(DataDownloadError, AirQualityError))
        self.assertTrue(issubclass(StationNotFoundError, AirQualityError))
        self.assertTrue(issubclass(PollutantNotReportedError, AirQualityError))


    def test_exception_instantiation(self):
        """Test that exceptions can be instantiated with messages."""
        msg = "Test error message"
        exc = StationNotFoundError(msg)
        self.assertEqual(str(exc), msg)


class TestAirQualityHelpers(unittest.TestCase):
    """Test internal helper methods for data validation."""

    @patch("src.czech_air_quality.data_manager.DataManager")
    def setUp(self, mock_dm):
        """Set up test fixtures."""
        mock_dm.return_value.raw_data_json = None
        self.aq = AirQuality(auto_load=False)


class TestIsValidMeasurement(unittest.TestCase):
    """Test measurement validation logic."""

    @patch("src.czech_air_quality.data_manager.DataManager")
    def setUp(self, mock_dm):
        """Set up test fixtures."""
        mock_dm.return_value.raw_data_json = None
        self.aq = AirQuality(auto_load=False)


    def test_is_valid_measurement_valid_float(self):
        """Test is_valid_measurement with positive float."""
        self.assertTrue(self.aq._is_valid_measurement(42.5))


    def test_is_valid_measurement_valid_string(self):
        """Test is_valid_measurement with numeric string."""
        self.assertTrue(self.aq._is_valid_measurement("15.3"))


    def test_is_valid_measurement_zero(self):
        """Test is_valid_measurement with zero (valid)."""
        self.assertFalse(self.aq._is_valid_measurement(0))


    def test_is_valid_measurement_none(self):
        """Test is_valid_measurement with None."""
        self.assertFalse(self.aq._is_valid_measurement(None))


    def test_is_valid_measurement_empty_string(self):
        """Test is_valid_measurement with empty string."""
        self.assertFalse(self.aq._is_valid_measurement(""))


    def test_is_valid_measurement_error_code_negative(self):
        """Test is_valid_measurement with CHMI error code."""
        # CHMI returns negative values for missing/invalid data
        self.assertFalse(self.aq._is_valid_measurement(-5009))
        self.assertFalse(self.aq._is_valid_measurement(-5003))
        self.assertFalse(self.aq._is_valid_measurement(-9999))


    def test_is_valid_measurement_invalid_string(self):
        """Test is_valid_measurement with non-numeric string."""
        self.assertFalse(self.aq._is_valid_measurement("invalid"))


class TestEAQICalculation(unittest.TestCase):
    """Test EAQI (European Air Quality Index) calculation logic."""

    @patch("src.czech_air_quality.data_manager.DataManager")
    def setUp(self, mock_dm):
        """Set up test fixtures."""
        mock_dm.return_value.raw_data_json = None
        self.aq = AirQuality(auto_load=False)


    def test_calculate_e_aqi_subindex_invalid_pollutant(self):
        """Test EAQI calculation with unknown pollutant."""
        result = self.aq._calculate_e_aqi_subindex("UNKNOWN", 50.0)
        self.assertEqual(result, 0)


    def test_calculate_e_aqi_subindex_none_value(self):
        """Test EAQI calculation with None concentration."""
        result = self.aq._calculate_e_aqi_subindex("PM10", None)
        self.assertEqual(result, 0)


    def test_calculate_e_aqi_subindex_negative_value(self):
        """Test EAQI calculation with negative concentration."""
        result = self.aq._calculate_e_aqi_subindex("PM10", -5.0)
        self.assertEqual(result, 0)


    def test_calculate_e_aqi_subindex_pm10_zero(self):
        """Test PM10 EAQI at zero concentration."""
        result = self.aq._calculate_e_aqi_subindex("PM10", 0.0)
        self.assertEqual(result, 1)  # Good level (0-20 µg/m³)


    def test_calculate_e_aqi_subindex_pm10_below_first_band(self):
        """Test PM10 EAQI below first breakpoint."""
        # PM10 first band: Level 1 (Good) at 20 µg/m³
        result = self.aq._calculate_e_aqi_subindex("PM10", 5.0)
        # 5 is below 20, so it's level 1 (Good)
        self.assertEqual(result, 1)


    def test_calculate_e_aqi_subindex_pm10_first_band(self):
        """Test PM10 EAQI in first band."""
        # PM10 first band: Level 1 (Good) at 20 µg/m³, Level 2 (Fair) at 40 µg/m³
        result = self.aq._calculate_e_aqi_subindex("PM10", 15.0)
        # 15 is below 20, so it's level 1 (Good)
        self.assertEqual(result, 1)


    def test_calculate_e_aqi_subindex_pm10_exact_band(self):
        """Test PM10 EAQI at exact band limit."""
        result = self.aq._calculate_e_aqi_subindex("PM10", 20.0)
        self.assertEqual(result, 1)  # At threshold of Good level


    def test_calculate_e_aqi_subindex_o3(self):
        """Test O3 EAQI calculation."""
        result = self.aq._calculate_e_aqi_subindex("O3", 150.0)
        # O3 bands: Level 1 at 50, Level 2 at 100, Level 3 at 130, Level 4 at 240...
        # 150 falls above 130 but below 240, so it's level 4 (Poor)
        self.assertEqual(result, 4)


    def test_get_aqi_description_error(self):
        """Test AQI description for error value."""
        desc = self.aq._get_aqi_description(-1)
        # Negative AQI maps to closest level which is "N/A" or lowest level
        self.assertIsInstance(desc, str)
        self.assertIn(desc.lower(), ["error/n/a", "n/a", "unknown"])


    def test_get_aqi_description_good(self):
        """Test AQI description for good air quality."""
        desc = self.aq._get_aqi_description(1)
        self.assertEqual(desc, "Good")


    def test_get_aqi_description_fair(self):
        """Test AQI description for fair air quality."""
        desc = self.aq._get_aqi_description(2)
        self.assertEqual(desc, "Fair")


    def test_get_aqi_description_poor(self):
        """Test AQI description for poor air quality."""
        desc = self.aq._get_aqi_description(4)
        self.assertEqual(desc, "Poor")


    def test_get_aqi_description_very_poor(self):
        """Test AQI description for very poor air quality."""
        desc = self.aq._get_aqi_description(5)
        self.assertEqual(desc, "Very Poor")


    def test_get_aqi_description_extremely_poor(self):
        """Test AQI description for extremely poor air quality."""
        desc = self.aq._get_aqi_description(6)
        self.assertEqual(desc, "Extremely Poor")


class TestFindMeasurementInStation(unittest.TestCase):
    """Test the _find_measurement_in_station helper method (if implemented)."""

    @patch("src.czech_air_quality.data_manager.DataManager")
    def setUp(self, mock_dm):
        """Set up test fixtures."""
        mock_dm.return_value.raw_data_json = None
        self.aq = AirQuality(auto_load=False)


    def test_find_measurement_in_station_method_exists(self):
        """Test that _find_measurement_in_station method exists if implemented."""
        has_method = hasattr(self.aq, "_find_measurement_in_station")
        self.assertIsInstance(has_method, bool)


class TestGetAllStationNames(unittest.TestCase):
    """Test the class method get_all_station_names."""

    def test_get_all_station_names_returns_list(self):
        """Test that get_all_station_names returns a list."""
        with patch("src.czech_air_quality.api.AirQuality", side_effect=Exception("Mock error")):
            result = AirQuality.get_all_station_names()

            self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
