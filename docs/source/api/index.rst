API References
=============

.. meta::
   :description: API references of the czech_air_quality python library.

.. toctree::
   :maxdepth: 2

   main
   exceptions

Caching Strategy:
~~~~~~~~~~~~~~~~~
To prevent overloading the OpenData servers on every call, and to improve performance,
this library uses a caching mechanism with the following strategy:

- **Cache Hit (Recent):** If cached data is =<20 minutes old, use it immediately
- **ETag Validation:** If cache is >=20 minutes old, perform HTTP-HEAD request with ETag
    - If server returns **304 (Not Modified)**, trust cache for another 20 minutes
    - If server returns **200 (Modified)**, download full data
- **Network Error:** If network unavailable but cache exists, use stale cache with warning

Caching is also done with Nominatim, however it's more basic.

This can be customized:
    - **Force Refreshing:** ``force_fetch_fresh()`` bypasses the age check
    - **Disabling Caching:** By using ``disable_caching=True`` to always fetch fresh data

EAQI scale
~~~~~~~~~~
- EAQI reported by this library is the highest sub-index among all pollutants.

.. code-block:: python
    :linenos:

    EAQI_BANDS = {
        # Concentration breakpoints (µg/m³) and the corresponding EAQI level (1-6)
        # Format: (EAQI_Level, Upper_Concentration_Limit)
        "PM10": [
            (1, 20),    # Good: 0-20
            (2, 40),    # Fair: 20-40
            (3, 50),    # Moderate: 40-50
            (4, 100),   # Poor: 50-100
            (5, 150),   # Very Poor: 100-150
            (6, float("inf"))  # Extremely Poor: ≥150
        ],
        "PM2_5": [
            (1, 10),    # Good: 0-10
            (2, 20),    # Fair: 10-20
            (3, 25),    # Moderate: 20-25
            (4, 50),    # Poor: 25-50
            (5, 75),    # Very Poor: 50-75
            (6, float("inf"))  # Extremely Poor: ≥75
        ],
        "O3": [
            (1, 50),    # Good: 0-50
            (2, 100),   # Fair: 50-100
            (3, 130),   # Moderate: 100-130
            (4, 240),   # Poor: 130-240
            (5, 380),   # Very Poor: 240-380
            (6, float("inf"))  # Extremely Poor: ≥380
        ],
        "NO2": [
            (1, 40),    # Good: 0-40
            (2, 90),    # Fair: 40-90
            (3, 120),   # Moderate: 90-120
            (4, 230),   # Poor: 120-230
            (5, 340),   # Very Poor: 230-340
            (6, float("inf"))  # Extremely Poor: ≥340
        ],
        "SO2": [
            (1, 100),   # Good: 0-100
            (2, 200),   # Fair: 100-200
            (3, 350),   # Moderate: 200-350
            (4, 500),   # Poor: 350-500
            (5, 750),   # Very Poor: 500-750
            (6, float("inf"))  # Extremely Poor: ≥750
        ],
    }

Usable regions
~~~~~~~~~~~~~~
- Lowercase strings can be used as well

.. code-block:: python
    :linenos:

    regions = [
        "Jihomoravský",
        "Jihočeský",
        "Karlovarský",
        "Královéhradecký",
        "Liberecký",
        "Moravskoslezský",
        "Olomoucký",
        "Pardubický",
        "Plzeňský",
        "Praha",
        "Středočeský",
        "Ústecký",
        "Vysočina",
        "Zlínský"
    ]

Debugging
~~~~~~~~~
- This library uses the standard Python logging module for debugging.

.. code-block:: python
    :linenos:

    import logging
    logging.getLogger("czech_air_quality").setLevel(logging.DEBUG)
