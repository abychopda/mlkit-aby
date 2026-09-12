"""Tests for the weather acquisition and aggregation module."""

from __future__ import annotations

import pandas as pd
import pytest

from mlkit_aby.helpers import weather


@pytest.fixture
def two_days_hourly() -> pd.DataFrame:
    """Forty-eight hours of synthetic observations across two local days."""
    timestamps = pd.date_range("2024-01-01 00:00", periods=48, freq="h")
    return pd.DataFrame(
        {
            "time": timestamps,
            # Day one averages 10, day two averages 20.
            "temperature_2m": [10.0] * 24 + [20.0] * 24,
            # One millimetre every hour, so 24 mm per day in total.
            "precipitation": [1.0] * 48,
            # A single gust spike on day one.
            "wind_gusts_10m": [5.0] * 23 + [90.0] + [5.0] * 24,
        }
    )


class TestAggregateToDaily:
    def test_produces_one_row_per_day(self, two_days_hourly):
        daily = weather.aggregate_to_daily(
            two_days_hourly,
            aggregations={"temperature_2m": ("temperature_2m", "mean")},
        )
        assert len(daily) == 2

    def test_mean_aggregation(self, two_days_hourly):
        daily = weather.aggregate_to_daily(
            two_days_hourly,
            aggregations={"temperature_2m": ("temperature_2m", "mean")},
        )
        assert list(daily["temperature_2m"]) == pytest.approx([10.0, 20.0])

    def test_sum_aggregation(self, two_days_hourly):
        daily = weather.aggregate_to_daily(
            two_days_hourly,
            aggregations={"precipitation": ("precipitation", "sum")},
        )
        assert list(daily["precipitation"]) == pytest.approx([24.0, 24.0])

    def test_max_aggregation_captures_the_spike(self, two_days_hourly):
        # The point of using max for gusts: the peak survives aggregation
        # where a mean would dilute it away.
        daily = weather.aggregate_to_daily(
            two_days_hourly,
            aggregations={"wind_gusts_10m": ("wind_gusts_10m", "max")},
        )
        assert list(daily["wind_gusts_10m"]) == pytest.approx([90.0, 5.0])

    def test_date_column_is_returned(self, two_days_hourly):
        daily = weather.aggregate_to_daily(
            two_days_hourly,
            aggregations={"temperature_2m": ("temperature_2m", "mean")},
        )
        assert daily.columns[0] == "date"
        assert pd.api.types.is_datetime64_any_dtype(daily["date"])

    def test_missing_time_column_raises(self, two_days_hourly):
        with pytest.raises(KeyError, match="time"):
            weather.aggregate_to_daily(two_days_hourly.drop(columns=["time"]))

    def test_missing_source_column_raises(self, two_days_hourly):
        with pytest.raises(KeyError, match="missing"):
            weather.aggregate_to_daily(
                two_days_hourly,
                aggregations={"snowfall": ("snowfall", "sum")},
            )


class TestFetchHourlyWeather:
    def test_rejects_unknown_source(self):
        with pytest.raises(ValueError, match="archive"):
            weather.fetch_hourly_weather(
                "2024-01-01", "2024-01-02", source="nonsense"
            )


class TestConstants:
    def test_target_aggregations_cover_every_required_variable(self):
        # Every variable the two target formulas need must have a defined
        # daily aggregation, otherwise the targets cannot be calculated.
        assert set(weather.TARGET_DAILY_AGGREGATIONS) == set(
            weather.REQUIRED_HOURLY_VARIABLES
        )
