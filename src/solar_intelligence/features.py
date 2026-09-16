from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "ALLSKY_SFC_SW_DWN"
WEATHER_COLUMNS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR", "CLOUD_AMT"]
FEATURE_COLUMNS = [
    TARGET,
    *WEATHER_COLUMNS,
    "day_sin",
    "day_cos",
]


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    day = data.index.dayofyear.astype(float)
    data["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    data["day_cos"] = np.cos(2 * np.pi * day / 365.25)
    return data


def make_inference_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build model features from observed days without requiring a future target."""
    missing = [column for column in [TARGET, *WEATHER_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = add_calendar_features(frame)
    data = data.dropna(subset=FEATURE_COLUMNS)
    if data.empty:
        raise ValueError("No complete observations are available for inference.")
    return data[FEATURE_COLUMNS].astype(float)


def make_next_day_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Use today's measured conditions to estimate tomorrow's solar resource."""
    features = make_inference_features(frame)
    target_next_day = frame[TARGET].shift(-1).reindex(features.index)
    valid = target_next_day.notna()
    return features.loc[valid], target_next_day.loc[valid].astype(float)


def chronological_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must leave a test set.")
    if len(x) != len(y):
        raise ValueError("Features and target must have equal length.")

    train_end = int(len(x) * train_fraction)
    validation_end = int(len(x) * (train_fraction + validation_fraction))
    return (
        x.iloc[:train_end],
        y.iloc[:train_end],
        x.iloc[train_end:validation_end],
        y.iloc[train_end:validation_end],
        x.iloc[validation_end:],
        y.iloc[validation_end:],
    )
