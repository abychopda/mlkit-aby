"""Unit tests for mlkit_aby.helpers.eda."""

from __future__ import annotations

import pandas as pd
import pytest

from mlkit_aby.helpers import eda


def test_dataset_overview_counts(sample_df):
    overview = eda.dataset_overview(sample_df)
    assert overview.loc[0, "rows"] == len(sample_df)
    assert overview.loc[0, "columns"] == sample_df.shape[1]
    assert overview.loc[0, "duplicate_rows"] == 1
    assert overview.loc[0, "missing_values"] == int(sample_df.isna().sum().sum())


def test_missing_summary_sorted_descending(sample_df):
    summary = eda.missing_summary(sample_df)
    assert list(summary.columns) == ["missing_count", "missing_percentage"]
    assert summary["missing_count"].is_monotonic_decreasing
    assert summary.loc["points", "missing_count"] == 0


def test_data_types_summary(sample_df):
    summary = eda.data_types_summary(sample_df)
    assert list(summary.columns) == ["data_type", "unique_values"]
    assert summary.loc["position", "unique_values"] == 3


class TestNumericalSummary:
    def test_all_numeric_columns(self, sample_df):
        summary = eda.numerical_summary(sample_df)
        assert "age" in summary.index
        assert "points" in summary.index
        assert "position" not in summary.index

    def test_single_feature(self, sample_df):
        summary = eda.numerical_summary(sample_df, feature="points")
        assert list(summary.index) == ["points"]

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError):
            eda.numerical_summary([1, 2, 3])

    def test_non_string_feature_raises_type_error(self, sample_df):
        with pytest.raises(TypeError):
            eda.numerical_summary(sample_df, feature=123)

    def test_missing_feature_raises_value_error(self, sample_df):
        with pytest.raises(ValueError):
            eda.numerical_summary(sample_df, feature="nope")

    def test_non_numeric_feature_raises_value_error(self, sample_df):
        with pytest.raises(ValueError):
            eda.numerical_summary(sample_df, feature="position")


class TestFindHighlyCorrelatedFeatures:
    def test_detects_correlated_pair(self, correlated_df):
        result = eda.find_highly_correlated_features(correlated_df, threshold=0.9)
        pair = {result.iloc[0]["feature_1"], result.iloc[0]["feature_2"]}
        assert pair == {"x", "x_scaled"}

    def test_invalid_threshold_raises(self, correlated_df):
        with pytest.raises(ValueError):
            eda.find_highly_correlated_features(correlated_df, threshold=1.5)

    def test_too_few_numeric_columns_raises(self):
        df = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            eda.find_highly_correlated_features(df)


def test_calculate_correlation_matrix_is_square_and_absolute(correlated_df):
    matrix = eda.calculate_correlation_matrix(correlated_df)
    assert matrix.shape[0] == matrix.shape[1]
    assert (matrix.values >= 0).all()
    assert (matrix.values <= 1 + 1e-9).all()
