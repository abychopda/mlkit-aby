"""Tests for the feature pipeline module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit_aby.helpers import features


@pytest.fixture
def synthetic_hourly() -> pd.DataFrame:
    """Ninety days of hourly observations covering every required variable."""
    timestamps = pd.date_range("2024-01-01", periods=90 * 24, freq="h")
    rng = np.random.default_rng(0)
    n = len(timestamps)
    return pd.DataFrame(
        {
            "time": timestamps,
            "temperature_2m": 20 + 5 * np.sin(np.arange(n) / 24) + rng.normal(size=n),
            "relative_humidity_2m": np.clip(60 + rng.normal(scale=10, size=n), 0, 100),
            "wind_speed_10m": np.abs(rng.normal(12, 4, size=n)),
            "cloud_cover": np.clip(rng.uniform(0, 100, size=n), 0, 100),
            "precipitation": np.clip(rng.exponential(0.2, size=n) - 0.15, 0, None),
            "wind_gusts_10m": np.abs(rng.normal(30, 10, size=n)),
            "snowfall": np.zeros(n),
        }
    )


class TestBuildDailyFeatures:
    def test_one_row_per_day(self, synthetic_hourly):
        result = features.build_daily_features(synthetic_hourly)
        assert len(result) == 90

    def test_contains_each_feature_family(self, synthetic_hourly):
        result = features.build_daily_features(synthetic_hourly)
        assert "cci" in result.columns                       # derived index
        assert "temperature_2m_lag1" in result.columns       # lag
        assert "cci_roll7_mean" in result.columns            # rolling
        assert "day_of_year_sin" in result.columns           # calendar
        assert "temperature_range" in result.columns         # within-day

    def test_recipe_controls_the_output(self, synthetic_hourly):
        recipe = dict(features.DEFAULT_RECIPE, lags=[1], windows=[3], within_day=False)
        result = features.build_daily_features(synthetic_hourly, recipe)

        assert "temperature_2m_lag1" in result.columns
        assert "temperature_2m_lag7" not in result.columns
        assert "cci_roll30_mean" not in result.columns
        assert "temperature_range" not in result.columns

    def test_warm_up_rows_are_missing_not_fabricated(self, synthetic_hourly):
        # A thirty-day mean cannot exist on day one. It must be missing
        # rather than silently computed from partial history.
        result = features.build_daily_features(synthetic_hourly)
        assert result.loc[:28, "cci_roll30_mean"].isna().all()
        assert result.loc[29:, "cci_roll30_mean"].notna().all()

    def test_no_future_leakage(self, synthetic_hourly):
        # The same truncation check applied to the assembled pipeline: a row
        # built with later data present must equal the row built without it.
        full = features.build_daily_features(synthetic_hourly)

        cutoff_day = 60
        truncated_hourly = synthetic_hourly[
            synthetic_hourly["time"] < pd.Timestamp("2024-01-01") + pd.Timedelta(days=cutoff_day + 1)
        ]
        truncated = features.build_daily_features(truncated_hourly)

        pd.testing.assert_series_equal(
            full.loc[cutoff_day],
            truncated.loc[cutoff_day],
            check_names=False,
        )


class TestRequiredHistoryDays:
    def test_default_recipe(self):
        # The thirty-day rolling window is the binding constraint.
        assert features.required_history_days() == 30

    def test_longest_window_wins(self):
        recipe = dict(features.DEFAULT_RECIPE, windows=[3, 90], lags=[1])
        assert features.required_history_days(recipe) == 90

    def test_longest_lag_wins_when_larger(self):
        # A lag of 365 needs 366 days inclusive of the current one.
        recipe = dict(features.DEFAULT_RECIPE, windows=[3], lags=[1, 365])
        assert features.required_history_days(recipe) == 366

    def test_enough_history_actually_produces_a_complete_row(self, synthetic_hourly):
        # The contract that matters: fetching exactly this many days must be
        # sufficient to build one fully populated row.
        needed = features.required_history_days()
        window = synthetic_hourly[
            synthetic_hourly["time"] >= synthetic_hourly["time"].max()
            - pd.Timedelta(days=needed - 1)
        ]
        result = features.build_daily_features(window)
        assert result.iloc[-1].notna().all()
