"""
Input/output utility functions for the NBA Career Longevity Prediction project.

This module provides reusable functions for loading and saving datasets used
throughout the machine learning experimentation pipeline. The functions are
designed to support consistent and reproducible data access across multiple
experiments and notebooks.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_dataset(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    This function reads a CSV file and returns its contents as a pandas
    DataFrame. It performs basic validation to ensure that the specified
    file exists before attempting to load it.

    Args:
        file_path (str | Path):
            Path to the CSV file.

    Returns:
        pd.DataFrame:
            The loaded dataset.

    Raises:
        FileNotFoundError:
            If the specified file does not exist.

        pd.errors.EmptyDataError:
            If the CSV file is empty.

        pd.errors.ParserError:
            If the CSV file cannot be parsed.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path, **kwargs)

def _ensure_directory_exists(directory_path: str | Path) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        directory_path (str | Path):
            Directory to create.

    Returns:
        None
    """

    Path(directory_path).mkdir(parents=True, exist_ok=True)

def save_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    index: bool = False
) -> None:
    """
    Save a DataFrame to a CSV file.

    If the parent directory does not exist, it will be created automatically.

    Args:
        df (pd.DataFrame):
            DataFrame to save.

        output_path (str | Path):
            Destination path for the CSV file.

        index (bool, optional):
            Whether to write the DataFrame index.
            Defaults to False.

    Returns:
        None
    """

    output_path = Path(output_path)

    _ensure_directory_exists(output_path.parent)

    df.to_csv(output_path, index=index)