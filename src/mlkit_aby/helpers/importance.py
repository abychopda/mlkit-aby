"""
Feature importance analysis.

This module extracts feature importance from fitted models in a consistent
tabular form, so that different model families can be compared and so that
importance can be reported alongside a deployed model.

Three views are provided, and they answer different questions:

* **Coefficients** (linear models) - the modelled effect of a one unit
  change in a feature, holding the others fixed.
* **Impurity importance** (tree ensembles) - how much each feature reduced
  error across the trees. Fast, but biased towards high-cardinality and
  continuous features.
* **Permutation importance** (any model) - how much performance degrades
  when a feature is randomly shuffled. Slower, but measures what the model
  actually relies on, and is computed on held-out data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def linear_coefficient_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract coefficients from a fitted linear model.

    Coefficients are only comparable across features when those features
    share a scale, so this is most meaningful on standardised inputs.

    Args:
        model:
            Fitted linear model exposing a ``coef_`` attribute.

        feature_names (list[str]):
            Names of the model input features, in column order.

    Returns:
        pd.DataFrame:
            Columns ``feature``, ``coefficient`` and ``abs_coefficient``,
            sorted by absolute magnitude.

    Raises:
        AttributeError:
            If the model has no ``coef_`` attribute.

        ValueError:
            If the number of coefficients does not match the number of
            feature names.
    """
    if not hasattr(model, "coef_"):
        raise AttributeError("Model has no coef_ attribute; is it a linear model?")

    coefficients = np.ravel(model.coef_)
    _check_length(coefficients, feature_names, "coefficients")

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def tree_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract impurity-based feature importance from a fitted tree ensemble.

    Args:
        model:
            Fitted tree-based model exposing ``feature_importances_``.

        feature_names (list[str]):
            Names of the model input features, in column order.

    Returns:
        pd.DataFrame:
            Columns ``feature`` and ``importance``, sorted descending.
            Importances sum to 1.

    Raises:
        AttributeError:
            If the model has no ``feature_importances_`` attribute.

        ValueError:
            If the number of importances does not match the number of
            feature names.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "Model has no feature_importances_ attribute; is it tree-based?"
        )

    importances = model.feature_importances_
    _check_length(importances, feature_names, "importances")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_feature_importance(
    model,
    X,
    y,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate permutation importance on held-out data.

    Each feature is shuffled in turn and the resulting drop in performance
    is recorded. A feature the model genuinely depends on causes a large
    drop; an ignored feature causes none.

    This should be computed on validation or test data rather than training
    data, since the question is what the model relies on when generalising.

    Args:
        model:
            Fitted model exposing a ``predict`` method.

        X:
            Features to evaluate against.

        y:
            Observed target values.

        scoring (str | None):
            Scikit-learn scoring name. Defaults to the model's own scorer.

        n_repeats (int):
            Number of shuffles per feature. More repeats give a more stable
            estimate at proportionally more cost.

        random_state (int):
            Seed used for reproducibility.

    Returns:
        pd.DataFrame:
            Columns ``feature``, ``importance_mean`` and ``importance_std``,
            sorted by mean importance descending.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    feature_names = (
        list(X.columns) if hasattr(X, "columns") else [f"feature_{i}" for i in range(X.shape[1])]
    )

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _check_length(values, feature_names: list[str], description: str) -> None:
    """Raise if a value array does not align with the feature name list."""
    if len(values) != len(feature_names):
        raise ValueError(
            f"Model has {len(values)} {description} but "
            f"{len(feature_names)} feature names were supplied."
        )
