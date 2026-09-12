"""Tests for the cleaning helper module."""

from __future__ import annotations

import pandas as pd
import pytest

from mlkit_aby.helpers import cleaning


class TestRemoveFeatures:
    def test_drops_listed_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = cleaning.remove_features(df, ["b"])
        assert list(result.columns) == ["a", "c"]

    def test_original_unchanged(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        cleaning.remove_features(df, ["b"])
        assert list(df.columns) == ["a", "b"]

    def test_unknown_columns_ignored(self):
        df = pd.DataFrame({"a": [1]})
        result = cleaning.remove_features(df, ["does_not_exist"])
        assert list(result.columns) == ["a"]

    def test_non_list_raises_type_error(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError):
            cleaning.remove_features(df, "a")

    def test_non_string_entries_raise_type_error(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError):
            cleaning.remove_features(df, ["a", 1])
