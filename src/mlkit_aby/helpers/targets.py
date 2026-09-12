"""
Weather-derived target variables.

This module implements the two target variables used in this project:

* **Climate Comfort Index (CCI)** - a continuous score from 0 to 100
  describing how suitable conditions are for outdoor activity. Higher is
  more comfortable.
* **Weather Hazard Category (WHC)** - a four-level ordinal category derived
  by banding a continuous Weather Hazard Index (WHI), also scored from 0 to
  100. Higher is more hazardous.

Both are computed from **daily** weather values. The component formulas are
non-linear, so applying them to hourly observations and then averaging would
give different results from aggregating first. See
:mod:`mlkit_aby.helpers.weather` for the aggregation step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Climate Comfort Index
# --------------------------------------------------------------------------

#: Weight applied to each CCI component score. Weights sum to 1.0, so the
#: index spans the full 0-100 range.
CCI_WEIGHTS = {
    "temperature_score": 0.35,
    "humidity_score": 0.20,
    "wind_score": 0.15,
    "cloud_score": 0.15,
    "rain_score": 0.15,
}

#: Interpretation bands for the CCI, used for reporting rather than modelling.
CCI_BANDS = [
    (80, 100, "Excellent outdoor conditions"),
    (60, 80, "Comfortable conditions"),
    (40, 60, "Moderate comfort"),
    (20, 40, "Poor comfort"),
    (0, 20, "Very poor comfort"),
]


def climate_comfort_components(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the five component scores that make up the CCI.

    Each component is scaled to the range 0-1, where 1 represents ideal
    conditions for that variable. The reference points encode what
    "comfortable" means: 22 C, 50% relative humidity, a 10 km/h breeze, no
    cloud and no rain.

    Args:
        daily (pd.DataFrame):
            Daily weather values containing ``temperature_2m``,
            ``relative_humidity_2m``, ``wind_speed_10m``, ``cloud_cover``
            and ``precipitation``.

    Returns:
        pd.DataFrame:
            One column per component score, aligned to the input index.

    Raises:
        KeyError:
            If any required column is absent.
    """
    required = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation",
    ]
    _check_columns(daily, required, target="CCI")

    components = pd.DataFrame(index=daily.index)

    # Comfort falls away linearly on either side of 22 C, reaching zero at a
    # 20 degree departure (so at 2 C or 42 C).
    components["temperature_score"] = _clip_lower_zero(
        1 - (daily["temperature_2m"] - 22).abs() / 20
    )

    # Both very dry and very humid air are penalised, centred on 50%.
    components["humidity_score"] = _clip_lower_zero(
        1 - (daily["relative_humidity_2m"] - 50).abs() / 50
    )

    # A light breeze of 10 km/h scores best; still air and strong wind are
    # both penalised, though the wide divisor makes this a gentle penalty.
    components["wind_score"] = _clip_lower_zero(
        1 - (daily["wind_speed_10m"] - 10).abs() / 40
    )

    # Cloud cover is a percentage, so it maps directly onto a 0-1 penalty.
    components["cloud_score"] = _clip_lower_zero(1 - daily["cloud_cover"] / 100)

    # Total daily rainfall, with 20 mm treated as fully uncomfortable.
    components["rain_score"] = _clip_lower_zero(1 - daily["precipitation"] / 20)

    return components


def climate_comfort_index(daily: pd.DataFrame) -> pd.Series:
    """
    Calculate the Climate Comfort Index (CCI).

    The index is the weighted sum of the five component scores, rescaled to
    a 0-100 range.

    Args:
        daily (pd.DataFrame):
            Daily weather values. See :func:`climate_comfort_components`
            for the required columns.

    Returns:
        pd.Series:
            CCI values in the range 0-100, named ``cci``.
    """
    components = climate_comfort_components(daily)

    weighted_sum = sum(
        components[component] * weight
        for component, weight in CCI_WEIGHTS.items()
    )

    return (100 * weighted_sum).rename("cci")


# --------------------------------------------------------------------------
# Weather Hazard Index and Category
# --------------------------------------------------------------------------

#: Weight applied to each WHI hazard score. Weights sum to 1.0.
WHI_WEIGHTS = {
    "rain_hazard": 0.30,
    "wind_hazard": 0.30,
    "cloud_hazard": 0.20,
    "snow_hazard": 0.10,
    "temperature_hazard": 0.10,
}

#: Lower bound of each Weather Hazard Category, in WHI units.
WHC_THRESHOLDS = [0, 25, 50, 75]

#: Human-readable label for each Weather Hazard Category class.
WHC_LABELS = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
    3: "Extreme Risk",
}


def weather_hazard_components(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the five hazard scores that make up the WHI.

    Each hazard is scaled to the range 0-1, where 1 represents the most
    severe condition the formula recognises. Unlike the comfort scores,
    these are capped rather than floored: conditions beyond the reference
    point are all treated as equally severe.

    Args:
        daily (pd.DataFrame):
            Daily weather values containing ``precipitation``,
            ``wind_gusts_10m``, ``cloud_cover``, ``snowfall`` and
            ``temperature_2m``.

    Returns:
        pd.DataFrame:
            One column per hazard score, aligned to the input index.

    Raises:
        KeyError:
            If any required column is absent.
    """
    required = [
        "precipitation",
        "wind_gusts_10m",
        "cloud_cover",
        "snowfall",
        "temperature_2m",
    ]
    _check_columns(daily, required, target="WHI")

    components = pd.DataFrame(index=daily.index)

    # 30 mm of rain in a day, or more, counts as a full rain hazard.
    components["rain_hazard"] = _clip_upper_one(daily["precipitation"] / 30)

    # A peak gust of 100 km/h, or more, counts as a full wind hazard.
    components["wind_hazard"] = _clip_upper_one(daily["wind_gusts_10m"] / 100)

    components["cloud_hazard"] = _clip_upper_one(daily["cloud_cover"] / 100)

    # 15 cm of snow, or more, counts as a full snow hazard. In Sydney this
    # term is expected to be zero for every observation.
    components["snow_hazard"] = _clip_upper_one(daily["snowfall"] / 15)

    # Departure from 22 C in either direction, saturating at 25 degrees.
    components["temperature_hazard"] = _clip_upper_one(
        (daily["temperature_2m"] - 22).abs() / 25
    )

    return components


def weather_hazard_index(daily: pd.DataFrame) -> pd.Series:
    """
    Calculate the continuous Weather Hazard Index (WHI).

    Args:
        daily (pd.DataFrame):
            Daily weather values. See :func:`weather_hazard_components`
            for the required columns.

    Returns:
        pd.Series:
            WHI values in the range 0-100, named ``whi``.
    """
    components = weather_hazard_components(daily)

    weighted_sum = sum(
        components[component] * weight
        for component, weight in WHI_WEIGHTS.items()
    )

    return (100 * weighted_sum).rename("whi")


def weather_hazard_category(whi: pd.Series) -> pd.Series:
    """
    Convert a Weather Hazard Index into its ordinal category.

    Bands are left-closed and right-open, so a WHI of exactly 25 is
    Moderate Risk rather than Low Risk.

    Args:
        whi (pd.Series):
            Continuous WHI values.

    Returns:
        pd.Series:
            Integer class labels 0-3, named ``whc``. See :data:`WHC_LABELS`
            for the corresponding descriptions.
    """
    # np.inf closes the final band; -np.inf guards against any value that
    # floating point arithmetic nudges marginally below zero.
    bin_edges = [-np.inf, *WHC_THRESHOLDS[1:], np.inf]

    categories = pd.cut(
        whi,
        bins=bin_edges,
        labels=list(WHC_LABELS.keys()),
        right=False,
    )

    return categories.astype("int64").rename("whc")


def label_weather_hazard_category(whc: pd.Series) -> pd.Series:
    """
    Map Weather Hazard Category class numbers to their text labels.

    Args:
        whc (pd.Series):
            Integer class labels 0-3.

    Returns:
        pd.Series:
            Descriptive labels such as ``"Moderate Risk"``.
    """
    return whc.map(WHC_LABELS).rename("whc_label")


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _check_columns(
    df: pd.DataFrame,
    required: list[str],
    target: str,
) -> None:
    """Raise a descriptive KeyError if any required column is missing."""
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(
            f"Columns required to calculate {target} are missing: {missing}"
        )


def _clip_lower_zero(values: pd.Series) -> pd.Series:
    """Apply ``MAX(0, x)`` - used by the comfort component scores."""
    return values.clip(lower=0)


def _clip_upper_one(values: pd.Series) -> pd.Series:
    """Apply ``MIN(x, 1)`` - used by the hazard component scores."""
    return values.clip(upper=1)
