"""Tests for the regression and multiclass helpers in the modeling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit_aby.helpers import modeling


@pytest.fixture
def linear_data():
    """A noiseless linear relationship: y = 3x1 - 2x2 + 5."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["x1", "x2"])
    y = 3 * X["x1"] - 2 * X["x2"] + 5
    return X, y


@pytest.fixture
def imbalanced_classes():
    """Four classes where class 3 appears only a handful of times."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 150 + [1] * 120 + [2] * 25 + [3] * 5)
    # Give the model a genuine signal to learn.
    X["a"] = X["a"] + y * 3
    return X, y


class TestRegression:
    def test_linear_regression_recovers_coefficients(self, linear_data):
        X, y = linear_data
        model = modeling.train_linear_regression(X, y)
        assert model.coef_ == pytest.approx([3.0, -2.0], abs=1e-6)

    def test_ridge_shrinks_coefficients(self, linear_data):
        X, y = linear_data
        ridge = modeling.train_ridge_regression(X, y, alpha=100.0)
        ols = modeling.train_linear_regression(X, y)
        # The L2 penalty must pull the fitted coefficients towards zero.
        assert np.abs(ridge.coef_).sum() < np.abs(ols.coef_).sum()

    def test_random_forest_fits(self, linear_data):
        X, y = linear_data
        model = modeling.train_random_forest_regressor(X, y, n_estimators=20)
        assert len(model.predict(X)) == len(y)

    def test_evaluate_regression_on_perfect_predictions(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        metrics = modeling.evaluate_regression(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_evaluate_regression_known_values(self):
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([12.0, 18.0, 33.0])
        metrics = modeling.evaluate_regression(y_true, y_pred)
        # Errors are -2, +2, -3: MAE is 7/3, RMSE is sqrt(17/3).
        assert metrics["mae"] == pytest.approx(7 / 3)
        assert metrics["rmse"] == pytest.approx(np.sqrt(17 / 3))

    def test_reproducible_with_fixed_seed(self, linear_data):
        X, y = linear_data
        first = modeling.train_random_forest_regressor(X, y, n_estimators=20)
        second = modeling.train_random_forest_regressor(X, y, n_estimators=20)
        assert first.predict(X) == pytest.approx(second.predict(X))


class TestClassification:
    def test_classifier_fits_and_predicts(self, imbalanced_classes):
        X, y = imbalanced_classes
        model = modeling.train_random_forest_classifier(X, y, n_estimators=20)
        assert set(model.predict(X)).issubset({0, 1, 2, 3})

    def test_evaluate_classification_reports_absent_classes(self):
        # Class 3 never appears in either the truth or the predictions, but
        # must still be reported, otherwise a silently missing rare class
        # would look like it had been handled.
        y_true = pd.Series([0, 0, 1, 1, 2])
        y_pred = pd.Series([0, 0, 1, 1, 2])
        metrics = modeling.evaluate_classification(y_true, y_pred, labels=[0, 1, 2, 3])
        assert "recall_class_3" in metrics
        assert metrics["recall_class_3"] == 0.0

    def test_macro_f1_punishes_majority_only_prediction(self):
        # The core reason accuracy is not used as the headline metric: a
        # model that always answers "0" is 80% accurate and useless.
        y_true = pd.Series([0] * 80 + [1] * 20)
        y_pred = pd.Series([0] * 100)
        metrics = modeling.evaluate_classification(y_true, y_pred, labels=[0, 1])
        assert metrics["accuracy"] == pytest.approx(0.8)
        assert metrics["macro_f1"] < 0.5
        assert metrics["balanced_accuracy"] == pytest.approx(0.5)

    def test_perfect_classification(self):
        y = pd.Series([0, 1, 2, 3])
        metrics = modeling.evaluate_classification(y, y, labels=[0, 1, 2, 3])
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)
