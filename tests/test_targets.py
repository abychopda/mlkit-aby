"""Tests for the weather-derived target variable module."""

from __future__ import annotations

import pandas as pd
import pytest

from mlkit_aby.helpers import targets


@pytest.fixture
def ideal_day() -> pd.DataFrame:
    """A day sitting exactly on every CCI reference point."""
    return pd.DataFrame(
        {
            "temperature_2m": [22.0],
            "relative_humidity_2m": [50.0],
            "wind_speed_10m": [10.0],
            "cloud_cover": [0.0],
            "precipitation": [0.0],
        }
    )


@pytest.fixture
def calm_day() -> pd.DataFrame:
    """A day with no hazardous conditions at all."""
    return pd.DataFrame(
        {
            "precipitation": [0.0],
            "wind_gusts_10m": [0.0],
            "cloud_cover": [0.0],
            "snowfall": [0.0],
            "temperature_2m": [22.0],
        }
    )


class TestClimateComfortIndex:
    def test_ideal_day_scores_one_hundred(self, ideal_day):
        assert targets.climate_comfort_index(ideal_day).iloc[0] == pytest.approx(100.0)

    def test_component_scores_are_bounded(self):
        # Conditions far beyond every reference point must floor at zero
        # rather than going negative.
        extreme = pd.DataFrame(
            {
                "temperature_2m": [-40.0],
                "relative_humidity_2m": [100.0],
                "wind_speed_10m": [200.0],
                "cloud_cover": [100.0],
                "precipitation": [500.0],
            }
        )
        components = targets.climate_comfort_components(extreme)
        assert (components >= 0).all().all()
        assert (components <= 1).all().all()

    def test_index_stays_within_range(self):
        extreme = pd.DataFrame(
            {
                "temperature_2m": [-40.0, 22.0],
                "relative_humidity_2m": [100.0, 50.0],
                "wind_speed_10m": [200.0, 10.0],
                "cloud_cover": [100.0, 0.0],
                "precipitation": [500.0, 0.0],
            }
        )
        cci = targets.climate_comfort_index(extreme)
        assert cci.between(0, 100).all()

    def test_weights_sum_to_one(self):
        assert sum(targets.CCI_WEIGHTS.values()) == pytest.approx(1.0)

    def test_known_value(self):
        # Hand-calculated: temp 32 C -> 0.5, humidity 75% -> 0.5,
        # wind 30 km/h -> 0.5, cloud 50% -> 0.5, rain 10 mm -> 0.5.
        # Every component is 0.5, so the weighted sum is 0.5 and CCI is 50.
        day = pd.DataFrame(
            {
                "temperature_2m": [32.0],
                "relative_humidity_2m": [75.0],
                "wind_speed_10m": [30.0],
                "cloud_cover": [50.0],
                "precipitation": [10.0],
            }
        )
        assert targets.climate_comfort_index(day).iloc[0] == pytest.approx(50.0)

    def test_missing_column_raises_key_error(self, ideal_day):
        with pytest.raises(KeyError, match="CCI"):
            targets.climate_comfort_index(ideal_day.drop(columns=["cloud_cover"]))


class TestWeatherHazardIndex:
    def test_calm_day_scores_zero(self, calm_day):
        assert targets.weather_hazard_index(calm_day).iloc[0] == pytest.approx(0.0)

    def test_hazard_scores_are_capped_at_one(self):
        extreme = pd.DataFrame(
            {
                "precipitation": [500.0],
                "wind_gusts_10m": [400.0],
                "cloud_cover": [100.0],
                "snowfall": [200.0],
                "temperature_2m": [-50.0],
            }
        )
        components = targets.weather_hazard_components(extreme)
        assert (components <= 1).all().all()
        assert targets.weather_hazard_index(extreme).iloc[0] == pytest.approx(100.0)

    def test_weights_sum_to_one(self):
        assert sum(targets.WHI_WEIGHTS.values()) == pytest.approx(1.0)

    def test_known_value(self):
        # 15 mm rain -> 0.5, 50 km/h gusts -> 0.5, 50% cloud -> 0.5,
        # 7.5 cm snow -> 0.5, 34.5 C -> 0.5. Weighted sum 0.5, WHI 50.
        day = pd.DataFrame(
            {
                "precipitation": [15.0],
                "wind_gusts_10m": [50.0],
                "cloud_cover": [50.0],
                "snowfall": [7.5],
                "temperature_2m": [34.5],
            }
        )
        assert targets.weather_hazard_index(day).iloc[0] == pytest.approx(50.0)

    def test_missing_column_raises_key_error(self, calm_day):
        with pytest.raises(KeyError, match="WHI"):
            targets.weather_hazard_index(calm_day.drop(columns=["snowfall"]))


class TestWeatherHazardCategory:
    def test_band_assignment(self):
        whi = pd.Series([0.0, 24.9, 25.0, 49.9, 50.0, 74.9, 75.0, 100.0])
        assert list(targets.weather_hazard_category(whi)) == [0, 0, 1, 1, 2, 2, 3, 3]

    def test_boundaries_are_left_closed(self):
        # A value sitting exactly on a threshold belongs to the higher band.
        whi = pd.Series([25.0, 50.0, 75.0])
        assert list(targets.weather_hazard_category(whi)) == [1, 2, 3]

    def test_returns_integer_dtype(self):
        result = targets.weather_hazard_category(pd.Series([10.0, 60.0]))
        assert result.dtype == "int64"

    def test_labels_map_correctly(self):
        whc = targets.weather_hazard_category(pd.Series([10.0, 30.0, 60.0, 90.0]))
        labels = targets.label_weather_hazard_category(whc)
        assert list(labels) == [
            "Low Risk",
            "Moderate Risk",
            "High Risk",
            "Extreme Risk",
        ]
