# ☀️ Solar Intelligence

Explainable machine-learning project for **next-day solar-resource forecasting in Abu Dhabi, UAE** using real NASA POWER meteorological data.

The project is designed to show a complete applied-ML workflow: data acquisition, leakage-aware feature engineering, chronological validation, baseline comparison, model selection, held-out testing, permutation-based explainability, reproducible artifacts, and a small Streamlit demo.

> **Status:** training pipeline ready; experiment results are intentionally not fabricated. The repository will display final metrics only after a verified run.

## Research question
Can supervised-learning models improve on a simple persistence baseline when estimating tomorrow's daily surface solar irradiation from today's observed weather and seasonal context?

## Data
The pipeline fetches daily values directly from the **NASA POWER API** for the Abu Dhabi reference location (24.4539° N, 54.3773° E). See `DATA_CARD.md` for provenance, variables, split policy, and limitations.

Features include current-day solar irradiation, temperature, relative humidity, wind speed, precipitation, cloud amount, and cyclic day-of-year features. The prediction target is the **next day's** `ALLSKY_SFC_SW_DWN` value.

## Models compared
- Persistence baseline: tomorrow ≈ today's solar irradiation
- Ridge regression
- Random forest regression
- Histogram gradient boosting

The learned model with the lowest validation MAE is retrained on train + validation data and evaluated once on the held-out chronological test period.

## Evaluation
The experiment reports:
- MAE
- RMSE
- R²
- persistence-baseline comparison
- permutation feature importance
- held-out predictions saved to CSV

No performance number belongs in this README until `scripts/run_experiment.py` has actually produced and verified it.

## Project structure
```text
src/solar_intelligence/
  data.py          NASA POWER acquisition
  features.py      target construction + chronological splits
  modeling.py      baselines, model comparison, metrics, explainability
scripts/
  run_experiment.py
app.py             Streamlit reviewer demo
tests/             unit tests for temporal logic
DATA_CARD.md        provenance and limitations
.github/workflows/ci.yml
```

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/run_experiment.py --start-year 2018
```

After a successful experiment:
```bash
streamlit run app.py
```

Generated CSV/model/metric artifacts are intentionally ignored by Git. Only verified summary results should later be added to the repository.

## Why chronological splitting matters
Random train/test shuffling would allow future seasonal patterns to leak into an earlier forecasting task. This project keeps observations in time order so training, validation, and test periods imitate a more realistic forecasting setup.

## Responsible interpretation
This is an educational portfolio experiment, **not an operational energy forecast**. A single-location historical model can fail under unusual weather, distribution shift, different geographies, or future climate conditions. It should not be used for grid operations, investment, or safety-critical decisions.

## Next milestone
Run the full experiment, verify CI, inspect error cases and feature importance, then add a concise results table and one visualization based only on genuine outputs.
