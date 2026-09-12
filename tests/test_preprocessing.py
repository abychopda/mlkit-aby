"""Unit tests for mlkit_aby.helpers.preprocessing."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from mlkit_aby.helpers import preprocessing

CATEGORICAL = ["position"]


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "points": [10.0, 20.0, 30.0, 40.0],
            "position": ["G", "F", "G", "C"],
        }
    )


def test_fit_one_hot_encoder_returns_fitted_encoder():
    encoder = preprocessing.fit_one_hot_encoder(_training_frame(), CATEGORICAL)
    assert isinstance(encoder, OneHotEncoder)
    check_is_fitted(encoder)  # raises if not fitted


def test_transform_preserves_rows_and_numeric_columns():
    train = _training_frame()
    encoder = preprocessing.fit_one_hot_encoder(train, CATEGORICAL)

    transformed = preprocessing.transform_with_one_hot_encoder(train, encoder, CATEGORICAL)

    assert len(transformed) == len(train)
    assert "points" in transformed.columns
    assert {"position_G", "position_F", "position_C"}.issubset(transformed.columns)
    assert "position" not in transformed.columns


def test_transform_handles_unseen_category():
    train = _training_frame()
    encoder = preprocessing.fit_one_hot_encoder(train, CATEGORICAL)

    unseen = pd.DataFrame({"points": [5.0], "position": ["PG"]})
    transformed = preprocessing.transform_with_one_hot_encoder(unseen, encoder, CATEGORICAL)

    one_hot_cols = [c for c in transformed.columns if c.startswith("position_")]
    assert transformed[one_hot_cols].sum().sum() == 0
