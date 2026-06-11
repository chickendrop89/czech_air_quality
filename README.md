# czech_air_quality
![PyPI - Version](https://img.shields.io/pypi/v/czech_air_quality?logo=python&logoColor=white) ![PyPI - Downloads](https://img.shields.io/pypi/dm/czech_air_quality?logo=python&logoColor=white) ![PyPI - Typing](https://img.shields.io/pypi/types/czech_air_quality?logo=python&logoColor=white)

Python library for retrieving and processing current air quality data from the CHMI `OpenData` portal, that provides data hourly.

Features:
- Fetch air quality report/pollutant measurements and EAQI data of a location.
- Resolve locations to the nearest physical weather station automatically using Nominatim.
- Ability to automatically fetch multiple close stations to get measurements of all pollutants at a location
- Caching mechanism

---

## Installation
```bash
pip install czech_air_quality
```

**Requirements:**
- `Python` 3.10+
- `requests` >= 2.28.0
- `geopy` >= 2.3.0

---

## Quick Start
```python
from czech_air_quality import AirQuality

client = AirQuality()
aqi_level, description = client.get_air_quality_index("Prague")
print(f"AQI: {aqi_level} ({description})")
# Output: "AQI: 3 (Moderate)"
```

## API documentation
References/docs can be found below, this library also 
supports full typing hints:

https://czech-air-quality.readthedocs.io

## Data Source
Data from CHMI (Czech Hydrometeorological Institute) OpenData portal, updated hourly.

- Metadata: https://opendata.chmi.cz/air_quality/now/metadata/metadata.json
- Measurements: https://opendata.chmi.cz/air_quality/now/data/airquality_1h_avg_CZ.csv
