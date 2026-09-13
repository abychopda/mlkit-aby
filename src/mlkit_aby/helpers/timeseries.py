"""
Time series feature engineering and dataset construction.

This module builds supervised learning datasets from a daily time series,
for problems where the goal is to predict a value some fixed number of days
into the future.

Preventing target leakage is the central concern. For a row dated ``D``,
every feature must be computable from information available on or before
``D``, while the target lives at ``D + horizon``. Two conventions enforce
this:

* Feature functions only ever shift **backwards** (``shift(+n)``), bringing
  past values forward into the present row.
* :func:`make_supervised_frame` is the only function that shifts
  **forwards** (``shift(-horizon)``), and it does so solely to construct the
  target column.

Anything that violates those conventions is a leak, and a model trained on
leaked data will score well while being useless in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Add lagged copies of the specified columns.

    A lag of ``n`` brings the value from ``n`` days earlier onto the current
    row, which is always safe with respect to leakage.

    Args:
        df (pd.DataFrame):
            Daily observations, sorted or sortable by ``date_column``.

        columns (list[str]):
            Columns to lag.

        lags (list[int]):
            Lag distances in days. Must all be positive.

        date_column (str):
            Column used to order the series.

    Returns:
        pd.DataFrame:
            A copy of the input with one additional column per
            column/lag combination, named ``{column}_lag{n}``.

    Raises:
        ValueError:
            If any lag is not a positive integer, since a non-positive lag
            would pull future values into the present row.

        KeyError:
            If any requested column is absent.
    """
    _validate_positive(lags, name="lags")
    _check_columns(df, [date_column, *columns])

    result = df.sort_values(date_column).copy()

    for column in columns:
        for lag in lags:
            result[f"{column}_lag{lag}"] = result[column].shift(lag)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    statistics: list[str] | None = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Add rolling window statistics for the specified columns.

    Windows are closed at the current row, so a 7-day mean on row ``D``
    summarises days ``D-6`` through ``D`` inclusive. That is safe: every
    observation in the window is in the past or present.

    Args:
        df (pd.DataFrame):
            Daily observations, sorted or sortable by ``date_column``.

        columns (list[str]):
            Columns to summarise.

        windows (list[int]):
            Window lengths in days. Must all be positive.

        statistics (list[str] | None):
            Statistics to compute for each window. Defaults to
            ``["mean", "std"]``.

        date_column (str):
            Column used to order the series.

    Returns:
        pd.DataFrame:
            A copy of the input with one additional column per
            column/window/statistic combination, named
            ``{column}_roll{window}_{statistic}``.

    Raises:
        ValueError:
            If any window is not a positive integer.

        KeyError:
            If any requested column is absent.
    """
    if statistics is None:
        statistics = ["mean", "std"]

    _validate_positive(windows, name="windows")
    _check_columns(df, [date_column, *columns])

    result = df.sort_values(date_column).copy()

    for column in columns:
        for window in windows:
            # min_periods=window means an incomplete window yields a missing
            # value rather than a statistic computed from partial history.
            # Those rows are dropped later, which is preferable to training
            # on a 7-day mean that actually covers two days.
            rolling = result[column].rolling(window=window, min_periods=window)

            for statistic in statistics:
                result[f"{column}_roll{window}_{statistic}"] = rolling.agg(statistic)

    return result


def add_calendar_features(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Add cyclical calendar features derived from the date.

    Calendar features are never leaky: the date of a future day is known
    arbitrarily far in advance.

    Day of year and month are encoded as sine and cosine pairs rather than
    raw integers. A raw encoding implies that 31 December and 1 January are
    364 units apart, when in seasonal terms they are adjacent. The paired
    trigonometric encoding places them next to each other on a circle, which
    is what a model needs in order to learn seasonality correctly.

    Args:
        df (pd.DataFrame):
            Daily observations containing a datetime column.

        date_column (str):
            Column holding the date.

    Returns:
        pd.DataFrame:
            A copy of the input with added calendar features.

    Raises:
        KeyError:
            If ``date_column`` is absent.
    """
    _check_columns(df, [date_column])

    result = df.copy()
    dates = pd.to_datetime(result[date_column])

    day_of_year = dates.dt.dayofyear
    month = dates.dt.month

    # 366 rather than 365 so that leap years do not push the encoding past a
    # full revolution.
    result["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    result["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 366)

    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)

    return result


def make_supervised_frame(
    df: pd.DataFrame,
    target_column: str,
    horizon: int,
    date_column: str = "date",
    target_name: str | None = None,
) -> pd.DataFrame:
    """
    Attach a future-dated target to each row.

    This is the only function in the module that shifts forwards in time.
    For a row dated ``D``, the target is the value of ``target_column`` on
    day ``D + horizon``, so the row reads as "given what is known on D,
    predict this".

    Rows near the end of the series have no future value available and are
    dropped, along with any row whose target is missing.

    Args:
        df (pd.DataFrame):
            Daily observations, sorted or sortable by ``date_column``.

        target_column (str):
            Column whose future value is to be predicted.

        horizon (int):
            Number of days ahead to predict. Must be positive.

        date_column (str):
            Column used to order the series.

        target_name (str | None):
            Name for the created target column. Defaults to
            ``{target_column}_h{horizon}``.

    Returns:
        pd.DataFrame:
            A copy of the input with an added target column and an added
            ``target_date`` column recording which day the target refers to,
            with rows lacking a target removed.

    Raises:
        ValueError:
            If ``horizon`` is not a positive integer.

        KeyError:
            If ``target_column`` or ``date_column`` is absent.
    """
    _validate_positive([horizon], name="horizon")
    _check_columns(df, [date_column, target_column])

    if target_name is None:
        target_name = f"{target_column}_h{horizon}"

    result = df.sort_values(date_column).copy()

    # The negative shift is what makes this the target rather than a
    # feature: it pulls a future value backwards onto the present row.
    result[target_name] = result[target_column].shift(-horizon)
    result["target_date"] = result[date_column].shift(-horizon)

    return result.dropna(subset=[target_name]).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    validation_start: str,
    test_start: str,
    horizon: int,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time series into training, validation and test sets by date.

    Splits are chronological rather than random. Weather is strongly
    autocorrelated, so a random split would place consecutive days on
    opposite sides of the boundary and let the model see near-copies of the
    test data during training. A chronological split mirrors how the model
    is actually used: learn from the past, predict forward.

    An embargo of ``horizon`` days is removed from the end of the training
    and validation sets. Without it, the final training rows would carry
    targets dated inside the following period, leaking across the boundary.

    Args:
        df (pd.DataFrame):
            Supervised frame, as produced by :func:`make_supervised_frame`.

        validation_start (str):
            First date of the validation period, as ``YYYY-MM-DD``.

        test_start (str):
            First date of the test period, as ``YYYY-MM-DD``.

        horizon (int):
            Forecast horizon in days, used as the embargo length.

        date_column (str):
            Column used to order and split the series.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Training, validation and test frames.

    Raises:
        ValueError:
            If ``test_start`` is not after ``validation_start``, or if
            ``horizon`` is not positive.
    """
    _validate_positive([horizon], name="horizon")
    _check_columns(df, [date_column])

    validation_start_date = pd.Timestamp(validation_start)
    test_start_date = pd.Timestamp(test_start)

    if test_start_date <= validation_start_date:
        raise ValueError("test_start must fall after validation_start.")

    result = df.sort_values(date_column).copy()
    dates = pd.to_datetime(result[date_column])

    # The embargo: the last `horizon` days before each boundary are dropped,
    # because their targets fall on the far side of it.
    embargo = pd.Timedelta(days=horizon)

    train = result[dates < validation_start_date - embargo]
    validation = result[
        (dates >= validation_start_date) & (dates < test_start_date - embargo)
    ]
    test = result[dates >= test_start_date]

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _validate_positive(values: list[int], name: str) -> None:
    """Raise if any value is not a positive integer.

    Non-positive lags and windows are rejected rather than silently
    accepted, because a lag of zero or less would reach into the future.
    """
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"All {name} must be positive integers; received {value!r}."
            )


def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise a descriptive KeyError if any required column is missing."""
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Required columns are missing: {missing}")
