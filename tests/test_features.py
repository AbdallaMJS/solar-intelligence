import pandas as pd
import pytest

from solar_intelligence.features import chronological_split, make_next_day_dataset


def sample_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": [4.0 + i * 0.1 for i in range(12)],
            "T2M": [20 + i for i in range(12)],
            "RH2M": [50 + i for i in range(12)],
            "WS2M": [2.0] * 12,
            "PRECTOTCORR": [0.0] * 12,
            "CLOUD_AMT": [10 + i for i in range(12)],
        },
        index=index,
    )


def test_next_day_target_is_shifted_forward():
    x, y = make_next_day_dataset(sample_frame())
    assert len(x) == 11
    assert y.iloc[0] == pytest.approx(4.1)
    assert x.iloc[0]["ALLSKY_SFC_SW_DWN"] == pytest.approx(4.0)


def test_calendar_features_are_bounded():
    x, _ = make_next_day_dataset(sample_frame())
    assert x["day_sin"].between(-1, 1).all()
    assert x["day_cos"].between(-1, 1).all()


def test_chronological_split_preserves_order():
    x, y = make_next_day_dataset(sample_frame())
    parts = chronological_split(x, y, train_fraction=0.6, validation_fraction=0.2)
    x_train, _, x_val, _, x_test, _ = parts
    assert x_train.index.max() < x_val.index.min() < x_test.index.min()


def test_missing_required_columns_rejected():
    with pytest.raises(ValueError):
        make_next_day_dataset(pd.DataFrame({"T2M": [1, 2, 3]}))
