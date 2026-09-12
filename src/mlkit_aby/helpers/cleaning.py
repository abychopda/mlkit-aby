"""
Data cleaning utility functions.

This module provides reusable functions for cleaning raw datasets, handling
missing values, correcting inconsistent data formats, removing duplicates,
and preparing data for downstream preprocessing and modelling.
"""

from __future__ import annotations

import pandas as pd


def remove_features(
    df: pd.DataFrame,
    columns_to_remove: list[str],
) -> pd.DataFrame:
    """
    Remove specified features from a DataFrame.

    This function removes the columns provided in ``columns_to_remove``
    while leaving the original DataFrame unchanged.

    Args:
        df (pd.DataFrame):
            Input DataFrame from which features will be removed.

        columns_to_remove (list[str]):
            List of column names to remove.

    Returns:
        pd.DataFrame:
            A copy of the input DataFrame with the specified columns removed.

    Raises:
        TypeError:
            If ``columns_to_remove`` is not a list of strings.
    """
    if not isinstance(columns_to_remove, list):
        raise TypeError("columns_to_remove must be a list of strings.")

    if not all(isinstance(column, str) for column in columns_to_remove):
        raise TypeError("All values in columns_to_remove must be strings.")

    return df.drop(
        columns=columns_to_remove,
        errors="ignore",
    ).copy()
