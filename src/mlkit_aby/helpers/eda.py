"""
Exploratory Data Analysis (EDA) utility functions.

This module contains reusable functions for summarising datasets, inspecting
data quality, analysing feature distributions, identifying missing values,
and generating descriptive statistics to support exploratory data analysis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact structural summary of a dataset.

    Args:
        df: Dataset to summarise.

    Returns:
        A one-row DataFrame containing the row count, column count, duplicate
        row count, and total number of missing values.
    """
    return pd.DataFrame(
        {
            "rows": [df.shape[0]],
            "columns": [df.shape[1]],
            "duplicate_rows": [int(df.duplicated().sum())],
            "missing_values": [int(df.isna().sum().sum())],
        }
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values for every column in a dataset.

    Args:
        df: Dataset to inspect.

    Returns:
        A DataFrame indexed by column name with missing counts and percentages,
        sorted from the most to the fewest missing values.
    """
    missing_count = df.isna().sum()
    return (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_percentage": (missing_count / len(df) * 100).round(2),
            }
        )
        .sort_values("missing_count", ascending=False)
    )


def data_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the data type and distinct-value count of each column.

    Args:
        df: Dataset to inspect.

    Returns:
        A DataFrame indexed by column name with its data type and the number of
        distinct non-missing values, sorted by data type then column name.
    """
    return (
        pd.DataFrame(
            {
                "data_type": df.dtypes.astype(str),
                "unique_values": df.nunique(dropna=True),
            }
        )
        .sort_values(["data_type", "unique_values"])
    )


# def numeric_feature_summary(df: pd.DataFrame, feature_name: str, target_name: str,) -> pd.DataFrame:
#     """Summarise a numeric feature for every target class.

#     Args:
#         df: Dataset containing the feature and target columns.
#         feature_name: Name of the numeric feature to summarise.
#         target_name: Name of the target column used for grouping.

#     Returns:
#         A DataFrame with count, mean, median, standard deviation, minimum, and
#         maximum for the feature in each target class.

#     Raises:
#         KeyError: If the feature or target column is not present in the dataset.
#     """
#     required_columns = {feature_name, target_name}
#     missing_columns = required_columns.difference(df.columns)
#     if missing_columns:
#         raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

#     return df.groupby(target_name)[feature_name].agg(
#         ["count", "mean", "median", "std", "min", "max"]
#     )


def numerical_summary(
    df: pd.DataFrame,
    feature: str | None = None,
    include: str = "number",
) -> pd.DataFrame:
    """
    Generate descriptive statistics for numerical features in a DataFrame.

    The function can either summarise all numerical features or a single
    numerical feature when its name is provided.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing the features to summarise.

        feature (str | None, optional):
            Name of a specific numerical feature to summarise. If None,
            descriptive statistics are generated for all numerical features.
            Defaults to None.

        include (str, optional):
            Data type category to include in the summary. Defaults to
            numerical columns.

    Returns:
        pd.DataFrame:
            A DataFrame containing descriptive statistics such as count,
            mean, standard deviation, minimum, 25th percentile, median,
            75th percentile, and maximum.

    Raises:
        TypeError:
            If ``df`` is not a pandas DataFrame.

        TypeError:
            If ``feature`` is provided but is not a string.

        ValueError:
            If the specified feature does not exist in the DataFrame.

        ValueError:
            If the specified feature is not numerical.

        ValueError:
            If the DataFrame does not contain any numerical columns.

    Examples:
        >>> numerical_summary(df)
        >>> numerical_summary(df, feature="ORtg")
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if feature is not None and not isinstance(feature, str):
        raise TypeError("feature must be a string or None.")

    if feature is not None:
        if feature not in df.columns:
            raise ValueError(
                f"Feature '{feature}' does not exist in the DataFrame."
            )

        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(
                f"Feature '{feature}' is not numerical."
            )

        return df[[feature]].describe().T

    summary = df.describe(include=include).T

    if summary.empty:
        raise ValueError(
            "The DataFrame does not contain any numerical columns."
        )

    return summary



def find_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Identify pairs of numerical features with high absolute correlation.

    This function calculates the pairwise absolute correlation between
    numerical features and returns feature pairs whose correlation is
    greater than or equal to the specified threshold. Only the upper
    triangle
    of the correlation matrix is considered to avoid returning duplicate
    feature pairs.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing the features to analyse.

        threshold (float, optional):
            Minimum absolute correlation required for a feature pair to be
            included in the results. Must be between 0 and 1.
            Defaults to 0.90.

    Returns:
        pd.DataFrame:
            A DataFrame containing the columns ``feature_1``, ``feature_2``,
            and ``correlation``, sorted from highest to lowest correlation.

    Raises:
        ValueError:
            If the threshold is not between 0 and 1.
        ValueError:
            If the DataFrame does not contain at least two numerical features.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    numeric_features = df.select_dtypes(include="number").columns

    if len(numeric_features) < 2:
        raise ValueError(
            "The DataFrame must contain at least two numerical features."
        )

    correlation_matrix = df[numeric_features].corr().abs()

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1,
        ).astype(bool)
    )

    high_corr_pairs = (
        upper_triangle
        .stack()
        .reset_index()
    )

    high_corr_pairs.columns = [
        "feature_1",
        "feature_2",
        "correlation",
    ]

    high_corr_pairs = (
        high_corr_pairs[
            high_corr_pairs["correlation"] >= threshold
        ]
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )

    return high_corr_pairs


def calculate_correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the absolute correlation matrix for numerical features.

    Args:
        df (pd.DataFrame):
            Input DataFrame containing numerical features.

    Returns:
        pd.DataFrame:
            Absolute pairwise correlation matrix.
    """
    numeric_features = df.select_dtypes(include="number").columns

    return df[numeric_features].corr().abs()