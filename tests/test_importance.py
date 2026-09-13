"""Tests for the feature importance module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit_aby.helpers import importance, modeling


@pytest.fixture
def signal_and_noise():
    """Two features that matter and one that is pure noise."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["strong", "weak", "noise"])
    y = 10 * X["strong"] + 1 * X["weak"] + rng.normal(scale=0.1, size=300)
    return X, y


class TestLinearCoefficientImportance:
    def test_ranks_by_absolute_magnitude(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_linear_regression(X, y)
        result = importance.linear_coefficient_importance(model, list(X.columns))
        assert result.loc[0, "feature"] == "strong"
        assert result["abs_coefficient"].is_monotonic_decreasing

    def test_rejects_non_linear_model(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_random_forest_regressor(X, y, n_estimators=5)
        with pytest.raises(AttributeError, match="coef_"):
            importance.linear_coefficient_importance(model, list(X.columns))

    def test_rejects_mismatched_feature_names(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_linear_regression(X, y)
        with pytest.raises(ValueError, match="feature names"):
            importance.linear_coefficient_importance(model, ["only_one"])


class TestTreeFeatureImportance:
    def test_identifies_the_strong_feature(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_random_forest_regressor(X, y, n_estimators=50)
        result = importance.tree_feature_importance(model, list(X.columns))
        assert result.loc[0, "feature"] == "strong"

    def test_importances_sum_to_one(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_random_forest_regressor(X, y, n_estimators=50)
        result = importance.tree_feature_importance(model, list(X.columns))
        assert result["importance"].sum() == pytest.approx(1.0)

    def test_rejects_non_tree_model(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_linear_regression(X, y)
        with pytest.raises(AttributeError, match="feature_importances_"):
            importance.tree_feature_importance(model, list(X.columns))


class TestPermutationFeatureImportance:
    def test_noise_feature_scores_near_zero(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_linear_regression(X, y)
        result = importance.permutation_feature_importance(
            model, X, y, n_repeats=5
        ).set_index("feature")
        assert result.loc["strong", "importance_mean"] > result.loc["noise", "importance_mean"]
        assert result.loc["noise", "importance_mean"] < 0.05

    def test_sorted_descending(self, signal_and_noise):
        X, y = signal_and_noise
        model = modeling.train_linear_regression(X, y)
        result = importance.permutation_feature_importance(model, X, y, n_repeats=5)
        assert result["importance_mean"].is_monotonic_decreasing
