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
Pytest configuration and shared fixtures for tests.
"""

from unittest.mock import Mock, patch
from datetime import datetime, timezone
import tempfile
import pytest

@pytest.fixture
def mock_data_manager():
    """Provide a mock DataManager instance."""
    with patch("src.data_manager.DataManager") as mock_dm:
        instance = Mock()
        instance.raw_data_json = None
        instance.actualized_time = datetime.now(timezone.utc)
        instance.is_data_fresh.return_value = True
        instance.ensure_latest_data = Mock()

        mock_dm.return_value = instance
        yield instance


@pytest.fixture
def mock_stations():
    """Provide mock station data"""
    return [
        {
            "Name": "Ostrava-Fifejdy",
            "LocalityCode": "TOFF",
            "Region": "Moravskoslezský",
            "Lat": 49.83918762207031,
            "Lon": 18.263689041137695,
            "IdRegistrations": [40555, 40557, 40560, 40559, 40561, 1648691],
        },
        {
            "Name": "Beroun",
            "LocalityCode": "SBER",
            "Region": "Středočeský",
            "Lat": 49.95792770385742,
            "Lon": 14.058300018310547,
            "IdRegistrations": [40851, 40853, 40852, 40848, 40854, 1648793],
        },
        {
            "Name": "Bělotín",
            "LocalityCode": "MBEL",
            "Region": "Olomoucký",
            "Lat": 49.58708190917969,
            "Lon": 17.80422019958496,
            "IdRegistrations": [1379571, 1379567, 1648706],
        },
    ]


@pytest.fixture
def mock_measurements():
    """Provide mock measurement data"""
    return {
        "40555": {
            "ComponentCode": "SO2",
            "ComponentName": "oxid siřičitý",
            "Unit": "µg∙m⁻³",
            "value": "1.3",
        },
        "40557": {
            "ComponentCode": "NO2",
            "ComponentName": "oxid dusičitý",
            "Unit": "µg∙m⁻³",
            "value": "27.0",
        },
        "40560": {
            "ComponentCode": "NOx",
            "ComponentName": "oxidy dusíku",
            "Unit": "µg∙m⁻³",
            "value": "113.2",
        },
        "40559": {
            "ComponentCode": "O3",
            "ComponentName": "přízemní ozon",
            "Unit": "µg∙m⁻³",
            "value": "1.0",
        },
        "40561": {
            "ComponentCode": "PM10",
            "ComponentName": "částice PM10",
            "Unit": "µg∙m⁻³",
            "value": "58.3",
        },
        "1648691": {
            "ComponentCode": "INDX",
            "ComponentName": "Index kvality ovzduší",
            "Unit": "",
            "value": "3.0",
        },
        "40851": {
            "ComponentCode": "NO2",
            "ComponentName": "oxid dusičitý",
            "Unit": "µg∙m⁻³",
            "value": "14.9",
        },
        "40853": {
            "ComponentCode": "NOx",
            "ComponentName": "oxidy dusíku",
            "Unit": "µg∙m⁻³",
            "value": "20.3",
        },
        "40852": {
            "ComponentCode": "CO",
            "ComponentName": "oxid uhelnatý",
            "Unit": "µg∙m⁻³",
            "value": "349.0",
        },
        "40848": {
            "ComponentCode": "PM2_5",
            "ComponentName": "jemné částice PM2,5",
            "Unit": "µg∙m⁻³",
            "value": "14.8",
        },
        "40854": {
            "ComponentCode": "PM10",
            "ComponentName": "částice PM10",
            "Unit": "µg∙m⁻³",
            "value": "21.3",
        },
        "1648793": {
            "ComponentCode": "INDX",
            "ComponentName": "Index kvality ovzduší",
            "Unit": "",
            "value": "2.0",
        },
        "1379571": {
            "ComponentCode": "NO2",
            "ComponentName": "oxid dusičitý",
            "Unit": "µg∙m⁻³",
            "value": "7.2",
        },
        "1379567": {
            "ComponentCode": "PM10",
            "ComponentName": "částice PM10",
            "Unit": "µg∙m⁻³",
            "value": "15.1",
        },
        "1648706": {
            "ComponentCode": "INDX",
            "ComponentName": "Index kvality ovzduší",
            "Unit": "",
            "value": "1.0",
        },
    }


@pytest.fixture
def mock_metadata():
    """Provide mock metadata structure"""
    return {
        "Localities": [
            {
                "Name": "Ostrava-Fifejdy",
                "LocalityCode": "TOFF",
                "Region": "Moravskoslezský",
                "Lat": 49.83918762207031,
                "Lon": 18.263689041137695,
                "IdRegistrations": [40555, 40557, 40560, 40559, 40561, 1648691],
                "MeasuringPrograms": [],
            },
            {
                "Name": "Beroun",
                "LocalityCode": "SBER",
                "Region": "Středočeský",
                "Lat": 49.95792770385742,
                "Lon": 14.058300018310547,
                "IdRegistrations": [40851, 40853, 40852, 40848, 40854, 1648793],
                "MeasuringPrograms": [],
            },
            {
                "Name": "Bělotín",
                "LocalityCode": "MBEL",
                "Region": "Olomoucký",
                "Lat": 49.58708190917969,
                "Lon": 17.80422019958496,
                "IdRegistrations": [1379571, 1379567, 1648706],
                "MeasuringPrograms": [],
            },
        ],
        "id_registration_to_component": {
            "40555": {
                "ComponentCode": "SO2",
                "ComponentName": "oxid siřičitý",
                "Unit": "µg∙m⁻³",
            },
            "40557": {
                "ComponentCode": "NO2",
                "ComponentName": "oxid dusičitý",
                "Unit": "µg∙m⁻³",
            },
            "40560": {
                "ComponentCode": "NOx",
                "ComponentName": "oxidy dusíku",
                "Unit": "µg∙m⁻³",
            },
            "40559": {
                "ComponentCode": "O3",
                "ComponentName": "přízemní ozon",
                "Unit": "µg∙m⁻³",
            },
            "40561": {
                "ComponentCode": "PM10",
                "ComponentName": "částice PM10",
                "Unit": "µg∙m⁻³",
            },
            "1648691": {
                "ComponentCode": "INDX",
                "ComponentName": "Index kvality ovzduší",
                "Unit": "",
            },
            "40851": {
                "ComponentCode": "NO2",
                "ComponentName": "oxid dusičitý",
                "Unit": "µg∙m⁻³",
            },
            "40853": {
                "ComponentCode": "NOx",
                "ComponentName": "oxidy dusíku",
                "Unit": "µg∙m⁻³",
            },
            "40852": {
                "ComponentCode": "CO",
                "ComponentName": "oxid uhelnatý",
                "Unit": "µg∙m⁻³",
            },
            "40848": {
                "ComponentCode": "PM2_5",
                "ComponentName": "jemné částice PM2,5",
                "Unit": "µg∙m⁻³",
            },
            "40854": {
                "ComponentCode": "PM10",
                "ComponentName": "částice PM10",
                "Unit": "µg∙m⁻³",
            },
            "1648793": {
                "ComponentCode": "INDX",
                "ComponentName": "Index kvality ovzduší",
                "Unit": "",
            },
            "1379571": {
                "ComponentCode": "NO2",
                "ComponentName": "oxid dusičitý",
                "Unit": "µg∙m⁻³",
            },
            "1379567": {
                "ComponentCode": "PM10",
                "ComponentName": "částice PM10",
                "Unit": "µg∙m⁻³",
            },
            "1648706": {
                "ComponentCode": "INDX",
                "ComponentName": "Index kvality ovzduší",
                "Unit": "",
            },
        },
    }


@pytest.fixture
def temp_cache_dir():
    """Provide a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def pytest_configure(config):
    """Configure pytest options and register custom markers."""
    config.addinivalue_line("testpaths", "tests")
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
