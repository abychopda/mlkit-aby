"""
Feature pipeline construction.

This module assembles the complete set of model inputs from raw hourly
weather observations, in a single function driven by a declarative recipe.

The reason for centralising this is training/serving skew. A model is
trained on features built one way and served features built another; if the
two ever diverge — a different lag, an extra rolling window, a different
column order — the model does not fail, it simply receives the wrong inputs
and returns confidently wrong predictions. Building both from the same
function, driven by a recipe stored alongside the trained model, removes the
possibility.

A recipe is a plain dictionary so that it can be serialised into the model
artefact and read back at prediction time:

    recipe = {
        "base_variables": ["temperature_2m", "cci", ...],
        "lags": [1, 2, 3, 7],
        "windows": [3, 7, 30],
        "statistics": ["mean", "std"],
        "derived_indices": ["cci"],
        "within_day": True,
        "within_day_lags": [1],
    }
"""

from __future__ import annotations

import pandas as pd

from mlkit_aby.helpers import targets, timeseries, weather

#: Recipe used by the Climate Comfort Index experiments.
DEFAULT_RECIPE: dict = {
    # Variables from which lag and rolling features are derived.
    "base_variables": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation",
        "cci",
    ],
    "lags": [1, 2, 3, 7],
    "windows": [3, 7, 30],
    "statistics": ["mean", "std"],
    # Same-day index values to compute. These are legitimate features: each
    # is derived only from weather already observed on that day.
    "derived_indices": ["cci"],
    # Within-day spread, which the daily means used by the targets discard.
    "within_day": True,
    "within_day_lags": [1],
}


def build_daily_features(
    hourly: pd.DataFrame,
    recipe: dict | None = None,
) -> pd.DataFrame:
    """
    Build the complete daily feature frame from hourly observations.

    Every feature produced is computable from information available on or
    before its own row's date, so the output is safe to use as model input
    without further screening.

    Args:
        hourly (pd.DataFrame):
            Hourly observations, as returned by
            :func:`mlkit_aby.helpers.weather.fetch_hourly_weather`.

        recipe (dict | None):
            Feature recipe. Defaults to :data:`DEFAULT_RECIPE`.

    Returns:
        pd.DataFrame:
            One row per day, with a ``date`` column followed by the daily
            weather aggregates, any derived indices, and the lag, rolling,
            calendar and within-day features the recipe specifies.

            Early rows contain missing values where a lag or rolling window
            reaches back beyond the start of the input. Callers are expected
            to drop them, or to supply enough leading history that they fall
            outside the period of interest.

    Raises:
        KeyError:
            If a requested base variable is not available.
    """
    if recipe is None:
        recipe = DEFAULT_RECIPE

    # 1. Aggregate hourly observations to daily values, using the
    #    aggregation appropriate to each variable.
    daily = weather.aggregate_to_daily(hourly, weather.TARGET_DAILY_AGGREGATIONS)

    # 2. Derive same-day index values. These are computed from the same day's
    #    weather, so they are known at prediction time and may be used as
    #    features for a future day's value.
    if "cci" in recipe.get("derived_indices", []):
        daily["cci"] = targets.climate_comfort_index(daily)

    if "whi" in recipe.get("derived_indices", []):
        daily["whi"] = targets.weather_hazard_index(daily)
        daily["whc"] = targets.weather_hazard_category(daily["whi"])

    # 3. Lagged copies of each base variable.
    features = timeseries.add_lag_features(
        daily, recipe["base_variables"], recipe["lags"]
    )

    # 4. Rolling window summaries of each base variable.
    features = timeseries.add_rolling_features(
        features,
        recipe["base_variables"],
        recipe["windows"],
        statistics=recipe["statistics"],
    )

    # 5. Cyclical calendar encoding.
    features = timeseries.add_calendar_features(features)

    # 6. Within-day variation, which the daily means discard.
    if recipe.get("within_day"):
        within_day = weather.aggregate_to_daily(
            hourly, weather.FEATURE_DAILY_AGGREGATIONS
        )
        features = features.merge(within_day, on="date", how="left")

        features["temperature_range"] = (
            features["temperature_2m_max"] - features["temperature_2m_min"]
        )

        within_day_columns = list(weather.FEATURE_DAILY_AGGREGATIONS) + [
            "temperature_range"
        ]

        if recipe.get("within_day_lags"):
            features = timeseries.add_lag_features(
                features, within_day_columns, recipe["within_day_lags"]
            )

    return features


def required_history_days(recipe: dict | None = None) -> int:
    """
    Calculate how many days of history a recipe needs to produce one row.

    At prediction time the service must fetch enough preceding weather to
    compute every feature for the requested date. This derives that number
    from the recipe itself, so that changing the recipe automatically
    changes how much data is fetched, rather than relying on a constant
    somewhere that has to be remembered.

    Args:
        recipe (dict | None):
            Feature recipe. Defaults to :data:`DEFAULT_RECIPE`.

    Returns:
        int:
            Number of days of history required, inclusive of the requested
            date itself.
    """
    if recipe is None:
        recipe = DEFAULT_RECIPE

    # A rolling window of n days spans the current day plus n-1 before it;
    # a lag of n reaches n days back, so needs n+1 days inclusive.
    longest_window = max(recipe.get("windows", [1]))
    longest_lag = max(recipe.get("lags", [0]) + recipe.get("within_day_lags", [0]))

    return max(longest_window, longest_lag + 1)
