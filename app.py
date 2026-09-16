from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from solar_intelligence.data import PowerRequest, fetch_power_daily
from solar_intelligence.features import FEATURE_COLUMNS, make_next_day_dataset

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"

st.set_page_config(page_title="Solar Intelligence", page_icon="☀️", layout="wide")
st.title("☀️ Solar Intelligence")
st.caption("Explainable next-day solar-resource forecasting with NASA POWER data")

st.markdown(
    "This demo is intentionally tied to the project's Abu Dhabi reference location. "
    "It uses today's measured meteorological conditions to estimate the next day's "
    "surface solar irradiation."
)

if not MODEL_PATH.exists():
    st.info(
        "No trained model artifact is committed. Run `python scripts/run_experiment.py` first, "
        "review the real metrics, and then launch this app locally."
    )
else:
    model = joblib.load(MODEL_PATH)
    if METRICS_PATH.exists():
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"**Selected model:** {report['selected_model']}")
        st.json(report["test_metrics"], expanded=False)

    st.subheader("Generate a forecast from recent NASA POWER observations")
    start = st.text_input("Start date (YYYYMMDD)", value="20250101")
    end = st.text_input("End date (YYYYMMDD)", value="20250430")

    if st.button("Fetch data and estimate next day", type="primary"):
        request = PowerRequest(24.4539, 54.3773, start, end)
        try:
            frame = fetch_power_daily(request)
            x, _ = make_next_day_dataset(frame)
            latest = x.iloc[[-1]][FEATURE_COLUMNS]
            prediction = float(model.predict(latest)[0])
            st.metric("Estimated next-day solar irradiation", f"{prediction:.2f} kWh/m²/day")
            st.dataframe(pd.DataFrame(latest).T if latest.ndim == 1 else latest, use_container_width=True)
            st.caption(
                "This is a portfolio experiment, not an operational energy forecast. "
                "The model is evaluated only on the historical distribution used in training."
            )
        except Exception as exc:
            st.error(str(exc))
