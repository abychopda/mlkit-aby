"""Unit tests for mlkit_aby.helpers.io."""

from __future__ import annotations

import pandas as pd
import pytest

from mlkit_aby.helpers import io


def test_save_then_load_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "data.csv"

    io.save_dataframe(df, path)
    loaded = io.load_dataset(path)

    pd.testing.assert_frame_equal(loaded, df)


def test_save_dataframe_creates_missing_parent_directory(tmp_path):
    df = pd.DataFrame({"a": [1]})
    path = tmp_path / "nested" / "deeper" / "out.csv"

    io.save_dataframe(df, path)

    assert path.exists()


def test_save_dataframe_omits_index_by_default(tmp_path):
    df = pd.DataFrame({"a": [1, 2]}, index=[10, 11])
    path = tmp_path / "out.csv"

    io.save_dataframe(df, path)

    assert path.read_text().splitlines()[0] == "a"


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_dataset(tmp_path / "missing.csv")
