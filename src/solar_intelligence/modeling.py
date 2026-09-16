from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import chronological_split, make_next_day_dataset

RANDOM_STATE = 42


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def candidate_models() -> dict[str, object]:
    return {
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=250,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_evaluate(frame: pd.DataFrame, output_dir: str | Path = "artifacts") -> dict:
    """Compare a persistence baseline and learned models on chronological splits."""
    x, y = make_next_day_dataset(frame)
    x_train, y_train, x_val, y_val, x_test, y_test = chronological_split(x, y)

    validation = {}
    persistence_val = x_val["ALLSKY_SFC_SW_DWN"].to_numpy()
    validation["persistence_baseline"] = regression_metrics(y_val, persistence_val)

    models = candidate_models()
    for name, model in models.items():
        model.fit(x_train, y_train)
        validation[name] = regression_metrics(y_val, model.predict(x_val))

    learned_names = list(models)
    best_name = min(learned_names, key=lambda name: validation[name]["mae"])
    best_model = models[best_name]

    x_train_full = pd.concat([x_train, x_val])
    y_train_full = pd.concat([y_train, y_val])
    best_model.fit(x_train_full, y_train_full)

    test_prediction = best_model.predict(x_test)
    persistence_test = x_test["ALLSKY_SFC_SW_DWN"].to_numpy()
    test_metrics = {
        best_name: regression_metrics(y_test, test_prediction),
        "persistence_baseline": regression_metrics(y_test, persistence_test),
    }

    importance = permutation_importance(
        best_model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )
    feature_importance = sorted(
        [
            {"feature": feature, "importance": round(float(score), 6)}
            for feature, score in zip(x.columns, importance.importances_mean)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output / "model.joblib")

    predictions = pd.DataFrame(
        {
            "date": x_test.index.astype(str),
            "actual": y_test.to_numpy(),
            "prediction": np.asarray(test_prediction),
            "persistence": persistence_test,
        }
    )
    predictions.to_csv(output / "test_predictions.csv", index=False)

    report = {
        "selected_model": best_name,
        "validation_metrics": validation,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "sample_counts": {
            "train": len(x_train),
            "validation": len(x_val),
            "test": len(x_test),
        },
        "note": "Metrics are valid only for the dataset and split used in this run.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
