"""
Data preprocessing utility functions.

This module contains reusable functions for transforming cleaned datasets
into machine learning ready data. Typical preprocessing tasks include
encoding categorical variables, scaling numerical features, splitting
datasets, and preparing feature matrices and target variables.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def fit_one_hot_encoder(
    X_train: pd.DataFrame,
    categorical_features: list[str],
) -> OneHotEncoder:
    """
    Fit a OneHotEncoder using only the training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature data.

    categorical_features : list[str]
        Names of categorical columns to encode.

    Returns
    -------
    OneHotEncoder
        Fitted encoder.
    """

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    encoder.fit(X_train[categorical_features])

    return encoder


def transform_with_one_hot_encoder(
    X: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_features: list[str],
) -> pd.DataFrame:
    """
    Transform categorical features using a fitted OneHotEncoder
    and return the complete numerical feature matrix.

    Numerical features are retained without scaling.
    """

    numerical_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    X_numerical = X[numerical_features].reset_index(drop=True)

    X_categorical = encoder.transform(
        X[categorical_features]
    )

    categorical_feature_names = (
        encoder.get_feature_names_out(
            categorical_features
        )
    )

    X_categorical = pd.DataFrame(
        X_categorical,
        columns=categorical_feature_names,
    )

    X_transformed = pd.concat(
        [
            X_numerical,
            X_categorical,
        ],
        axis=1,
    )

    return X_transformed