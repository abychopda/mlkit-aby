"""
Weather data acquisition and aggregation.

This module retrieves historical weather observations from the Open-Meteo
API and aggregates hourly observations into the daily values used for
downstream target creation and feature engineering.

Open-Meteo exposes two relevant services:

* The **archive** API, which serves quality-controlled reanalysis data but
  lags real time by several days.
* The **forecast** API, which can serve recent past observations via its
  ``past_days`` parameter and therefore covers the gap the archive leaves.

Both are wrapped here behind a single interface so that callers do not need
to know which service a given date range comes from.
"""

from __future__ import annotations

import pandas as pd
import requests

# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# --------------------------------------------------------------------------
# Study location
# --------------------------------------------------------------------------

# Sydney, NSW, Australia, as specified by the project requirements.
#
# The timezone matters: Open-Meteo returns timestamps in the requested
# timezone, and daily aggregation must follow *local* calendar days. Using
# the default (UTC) would shift each Sydney day by 10-11 hours and silently
# mix observations from two different local days into one aggregate.
SYDNEY_LATITUDE = -33.8688
SYDNEY_LONGITUDE = 151.2093
SYDNEY_TIMEZONE = "Australia/Sydney"

# --------------------------------------------------------------------------
# Hourly variables
# --------------------------------------------------------------------------

# The union of every hourly variable required by the two target definitions.
# Both targets are derived from this single download, so the raw data only
# needs to be retrieved once.
REQUIRED_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "wind_gusts_10m",
    "snowfall",
]

# --------------------------------------------------------------------------
# Daily aggregation specifications
# --------------------------------------------------------------------------
#
# Each specification maps an output column name to a
# ``(source_column, aggregation_function)`` pair.
#
# The aggregation happens *before* the target formulas are applied. This
# ordering matters because the formulas are non-linear (they use MIN, MAX
# and ABS), so aggregating afterwards would produce different values. It
# also matches the constants used in the formulas, which are scaled to
# daily quantities: 20 mm of precipitation and 15 cm of snowfall are
# plausible daily totals but would almost never be reached within a single
# hour.

#: Aggregations required to evaluate the CCI and WHI formulas.
TARGET_DAILY_AGGREGATIONS: dict[str, tuple[str, str]] = {
    # Typical conditions drive comfort.
    "temperature_2m": ("temperature_2m", "mean"),
    "relative_humidity_2m": ("relative_humidity_2m", "mean"),
    "wind_speed_10m": ("wind_speed_10m", "mean"),
    "cloud_cover": ("cloud_cover", "mean"),
    # Total rainfall, rather than an average rate, is what makes a day wet.
    "precipitation": ("precipitation", "sum"),
    # Hazard is driven by the worst moment of the day, not the average one,
    # so the peak gust is the meaningful quantity.
    "wind_gusts_10m": ("wind_gusts_10m", "max"),
    "snowfall": ("snowfall", "sum"),
}

#: Additional aggregations retained as candidate model features.
#:
#: The target definitions deliberately use daily means, which discard
#: within-day variation. A day that swings between 12 C and 32 C and a day
#: that sits flat at 22 C share the same mean but are not the same day.
#: That information is preserved here, as features, rather than by
#: complicating the target definition.
FEATURE_DAILY_AGGREGATIONS: dict[str, tuple[str, str]] = {
    "temperature_2m_max": ("temperature_2m", "max"),
    "temperature_2m_min": ("temperature_2m", "min"),
    "relative_humidity_2m_max": ("relative_humidity_2m", "max"),
    "relative_humidity_2m_min": ("relative_humidity_2m", "min"),
    "wind_speed_10m_max": ("wind_speed_10m", "max"),
    "cloud_cover_max": ("cloud_cover", "max"),
    "cloud_cover_min": ("cloud_cover", "min"),
    "precipitation_max": ("precipitation", "max"),
    "precipitation_hours": ("precipitation", lambda s: (s > 0).sum()),
}


def fetch_hourly_weather(
    start_date: str,
    end_date: str,
    latitude: float = SYDNEY_LATITUDE,
    longitude: float = SYDNEY_LONGITUDE,
    variables: list[str] | None = None,
    timezone: str = SYDNEY_TIMEZONE,
    source: str = "archive",
    timeout: int = 120,
) -> pd.DataFrame:
    """
    Retrieve hourly weather observations from the Open-Meteo API.

    Args:
        start_date (str):
            First date to retrieve, as ``YYYY-MM-DD`` (inclusive).

        end_date (str):
            Last date to retrieve, as ``YYYY-MM-DD`` (inclusive).

        latitude (float):
            Latitude of the location. Defaults to Sydney.

        longitude (float):
            Longitude of the location. Defaults to Sydney.

        variables (list[str] | None):
            Hourly variables to request. Defaults to
            :data:`REQUIRED_HOURLY_VARIABLES`.

        timezone (str):
            Timezone used for the returned timestamps. Daily aggregation
            depends on this, so it should match the location.

        source (str):
            Either ``"archive"`` for the historical reanalysis API, or
            ``"forecast"`` for the forecast API, which also serves recent
            observations the archive has not yet published.

        timeout (int):
            Request timeout in seconds.

    Returns:
        pd.DataFrame:
            One row per hour, with a ``time`` column of local timestamps
            followed by one column per requested variable.

    Raises:
        ValueError:
            If ``source`` is not ``"archive"`` or ``"forecast"``, or if the
            API response does not contain hourly data.

        requests.HTTPError:
            If the API returns an unsuccessful status code.
    """
    if source not in {"archive", "forecast"}:
        raise ValueError("source must be either 'archive' or 'forecast'.")

    if variables is None:
        variables = REQUIRED_HOURLY_VARIABLES

    url = ARCHIVE_URL if source == "archive" else FORECAST_URL

    parameters = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        # The API expects a single comma-separated string of variables.
        "hourly": ",".join(variables),
        "timezone": timezone,
    }

    response = requests.get(url, params=parameters, timeout=timeout)
    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(
            f"Open-Meteo response contained no hourly data: {payload}"
        )

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])

    return hourly


def aggregate_to_daily(
    hourly: pd.DataFrame,
    aggregations: dict[str, tuple[str, str]] | None = None,
    time_column: str = "time",
) -> pd.DataFrame:
    """
    Aggregate hourly weather observations into daily values.

    Args:
        hourly (pd.DataFrame):
            Hourly observations, as returned by :func:`fetch_hourly_weather`.

        aggregations (dict[str, tuple[str, str]] | None):
            Mapping of output column name to a
            ``(source_column, aggregation_function)`` pair. Defaults to
            :data:`TARGET_DAILY_AGGREGATIONS`.

        time_column (str):
            Name of the column holding hourly timestamps.

    Returns:
        pd.DataFrame:
            One row per calendar day, with a ``date`` column followed by one
            column per entry in ``aggregations``.

    Raises:
        KeyError:
            If ``time_column`` or any requested source column is absent.
    """
    if aggregations is None:
        aggregations = TARGET_DAILY_AGGREGATIONS

    if time_column not in hourly.columns:
        raise KeyError(f"Column '{time_column}' not found in the input frame.")

    missing = {
        source
        for source, _ in aggregations.values()
        if source not in hourly.columns
    }
    if missing:
        raise KeyError(
            f"Columns required for aggregation are missing: {sorted(missing)}"
        )

    # Group by local calendar date rather than by a fixed 24-hour offset.
    daily_groups = hourly.groupby(hourly[time_column].dt.date)

    daily = pd.DataFrame(
        {
            output_column: daily_groups[source_column].agg(aggregation)
            for output_column, (source_column, aggregation) in aggregations.items()
        }
    )

    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"

    return daily.reset_index()
