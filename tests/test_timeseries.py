"""Tests for the time series feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit_aby.helpers import timeseries


@pytest.fixture
def counter_series() -> pd.DataFrame:
    """A series whose value equals its position, making shifts easy to verify.

    Row 0 is 2024-01-01 with value 0, row 1 is 2024-01-02 with value 1, and
    so on. Any lag or horizon can therefore be checked by simple arithmetic.
    """
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "value": np.arange(40, dtype=float),
        }
    )


class TestAddLagFeatures:
    def test_lag_pulls_from_the_past(self, counter_series):
        result = timeseries.add_lag_features(counter_series, ["value"], [1, 3])
        # On row 10 (value 10), lag1 must be 9 and lag3 must be 7.
        assert result.loc[10, "value_lag1"] == 9.0
        assert result.loc[10, "value_lag3"] == 7.0

    def test_early_rows_are_missing(self, counter_series):
        result = timeseries.add_lag_features(counter_series, ["value"], [3])
        assert result.loc[:2, "value_lag3"].isna().all()

    def test_zero_lag_rejected(self, counter_series):
        # A lag of zero would be the current value; a negative one would be
        # the future. Both are refused so a typo cannot create a leak.
        with pytest.raises(ValueError, match="positive integers"):
            timeseries.add_lag_features(counter_series, ["value"], [0])

    def test_negative_lag_rejected(self, counter_series):
        with pytest.raises(ValueError, match="positive integers"):
            timeseries.add_lag_features(counter_series, ["value"], [-1])

    def test_missing_column_raises(self, counter_series):
        with pytest.raises(KeyError, match="missing"):
            timeseries.add_lag_features(counter_series, ["absent"], [1])


class TestAddRollingFeatures:
    def test_window_closes_at_current_row(self, counter_series):
        result = timeseries.add_rolling_features(
            counter_series, ["value"], [3], statistics=["mean"]
        )
        # Row 10 covers rows 8, 9 and 10, whose values average to 9.
        assert result.loc[10, "value_roll3_mean"] == pytest.approx(9.0)

    def test_incomplete_windows_are_missing(self, counter_series):
        result = timeseries.add_rolling_features(
            counter_series, ["value"], [7], statistics=["mean"]
        )
        assert result.loc[:5, "value_roll7_mean"].isna().all()
        assert not np.isnan(result.loc[6, "value_roll7_mean"])

    def test_multiple_statistics(self, counter_series):
        result = timeseries.add_rolling_features(
            counter_series, ["value"], [5], statistics=["mean", "std", "max"]
        )
        for statistic in ["mean", "std", "max"]:
            assert f"value_roll5_{statistic}" in result.columns

    def test_zero_window_rejected(self, counter_series):
        with pytest.raises(ValueError, match="positive integers"):
            timeseries.add_rolling_features(counter_series, ["value"], [0])


class TestAddCalendarFeatures:
    def test_expected_columns_created(self, counter_series):
        result = timeseries.add_calendar_features(counter_series)
        for column in [
            "day_of_year_sin",
            "day_of_year_cos",
            "month_sin",
            "month_cos",
        ]:
            assert column in result.columns

    def test_encoding_is_bounded(self, counter_series):
        result = timeseries.add_calendar_features(counter_series)
        assert result["day_of_year_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()

    def test_year_boundary_is_continuous(self):
        # The point of the cyclical encoding: 31 December and 1 January must
        # be close together, not 364 units apart.
        boundary = pd.DataFrame(
            {"date": pd.to_datetime(["2024-12-31", "2025-01-01"])}
        )
        result = timeseries.add_calendar_features(boundary)

        distance = np.hypot(
            result["day_of_year_sin"].diff().iloc[1],
            result["day_of_year_cos"].diff().iloc[1],
        )
        # Adjacent days sit about 1/366 of a revolution apart.
        assert distance < 0.05


class TestMakeSupervisedFrame:
    def test_target_comes_from_the_future(self, counter_series):
        result = timeseries.make_supervised_frame(
            counter_series, target_column="value", horizon=3
        )
        # Row 10 has value 10, so its three-day-ahead target must be 13.
        assert result.loc[10, "value_h3"] == 13.0

    def test_target_date_matches_the_horizon(self, counter_series):
        result = timeseries.make_supervised_frame(
            counter_series, target_column="value", horizon=7
        )
        offset = result.loc[0, "target_date"] - result.loc[0, "date"]
        assert offset == pd.Timedelta(days=7)

    def test_unresolvable_rows_are_dropped(self, counter_series):
        result = timeseries.make_supervised_frame(
            counter_series, target_column="value", horizon=3
        )
        # The final three days have no future value available.
        assert len(result) == len(counter_series) - 3

    def test_custom_target_name(self, counter_series):
        result = timeseries.make_supervised_frame(
            counter_series, target_column="value", horizon=1, target_name="y"
        )
        assert "y" in result.columns

    def test_zero_horizon_rejected(self, counter_series):
        with pytest.raises(ValueError, match="positive integers"):
            timeseries.make_supervised_frame(
                counter_series, target_column="value", horizon=0
            )


class TestChronologicalSplit:
    @pytest.fixture
    def supervised(self, counter_series):
        return timeseries.make_supervised_frame(
            counter_series, target_column="value", horizon=3
        )

    def test_splits_are_ordered_in_time(self, supervised):
        train, validation, test = timeseries.chronological_split(
            supervised,
            validation_start="2024-01-20",
            test_start="2024-01-30",
            horizon=3,
        )
        assert train["date"].max() < validation["date"].min()
        assert validation["date"].max() < test["date"].min()

    def test_embargo_removes_the_boundary_rows(self, supervised):
        horizon = 3
        train, _, _ = timeseries.chronological_split(
            supervised,
            validation_start="2024-01-20",
            test_start="2024-01-30",
            horizon=horizon,
        )
        # Every training target must land strictly before the validation
        # period begins, which is exactly what the embargo guarantees.
        assert train["target_date"].max() < pd.Timestamp("2024-01-20")

    def test_no_row_appears_in_two_splits(self, supervised):
        train, validation, test = timeseries.chronological_split(
            supervised,
            validation_start="2024-01-20",
            test_start="2024-01-30",
            horizon=3,
        )
        all_dates = pd.concat([train["date"], validation["date"], test["date"]])
        assert all_dates.is_unique

    def test_rejects_reversed_boundaries(self, supervised):
        with pytest.raises(ValueError, match="after validation_start"):
            timeseries.chronological_split(
                supervised,
                validation_start="2024-01-30",
                test_start="2024-01-20",
                horizon=3,
            )


class TestNoFutureLeakage:
    """The definitive leakage check.

    If a feature for row ``i`` genuinely uses only information available up
    to day ``i``, then computing it on the full series and computing it on
    the series truncated at row ``i`` must give the same answer. Any feature
    that consults the future would change once the future is removed.
    """

    def test_features_are_unchanged_by_truncation(self, counter_series):
        def build_features(df: pd.DataFrame) -> pd.DataFrame:
            result = timeseries.add_lag_features(df, ["value"], [1, 3, 7])
            result = timeseries.add_rolling_features(
                result, ["value"], [3, 7], statistics=["mean", "std"]
            )
            return timeseries.add_calendar_features(result)

        full = build_features(counter_series)

        for cutoff in [10, 20, 30]:
            truncated = build_features(counter_series.iloc[: cutoff + 1])

            feature_columns = [
                column for column in full.columns if column != "date"
            ]

            pd.testing.assert_series_equal(
                full.loc[cutoff, feature_columns],
                truncated.loc[cutoff, feature_columns],
                check_names=False,
            )

    def test_target_does_change_with_truncation(self, counter_series):
        # The mirror image of the test above: the target is *supposed* to
        # depend on the future, so removing the future must remove it. If
        # this test ever passed silently, the target would not be a forecast.
        full = timeseries.make_supervised_frame(
            counter_series, target_column="value", horizon=3
        )
        truncated = timeseries.make_supervised_frame(
            counter_series.iloc[:11], target_column="value", horizon=3
        )
        assert 10 in full.index
        assert 10 not in truncated.index
