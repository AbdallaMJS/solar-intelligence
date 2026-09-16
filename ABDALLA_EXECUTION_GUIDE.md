# Abdalla Execution Guide — Solar Intelligence

This file is a **student execution guide**, not evidence that the experiment has already been completed. The branch was prepared with AI-assisted development support. Abdalla should personally run the experiment, inspect the data/results, and complete `PROJECT_JOURNAL.md` in his own words.

## Goal
Reproduce the Abu Dhabi next-day solar-resource forecasting experiment from real NASA POWER data and verify every reported metric independently.

## 1. Work on the execution branch
```bash
git clone https://github.com/AbdallaMJS/solar-intelligence.git
cd solar-intelligence
git checkout abdalla-execution
```

## 2. Create a clean environment and install
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Then:
```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 3. Quality checks
```bash
ruff check src tests scripts app.py
pytest -q
```
Save the terminal output.

## 4. Run the reproducible experiment
Use an explicit date range so the experiment does not change as new NASA data appears:
```bash
python scripts/run_experiment.py --start-year 2018 --end-year 2025
```

Expected local outputs include:
- `data/abu_dhabi_power.csv`
- `artifacts/metrics.json`
- `artifacts/test_predictions.csv`
- `artifacts/model.joblib`

Do not invent results if the API fails or data are incomplete. Record exactly what happened.

## 5. Verify the data
Abdalla should inspect and record:
- row count;
- first/last dates;
- feature columns;
- missing-value counts;
- chronological train/validation/test sample counts.

He should be able to explain why chronological splitting is required for time-ordered data.

## 6. Independently verify metrics
Read `artifacts/metrics.json` and `artifacts/test_predictions.csv`. Independently recompute held-out:
- MAE;
- RMSE;
- R².

The recomputed values should match the saved report after rounding. Compare the selected learned model with the persistence baseline. If the learned model does not beat persistence, report that honestly and discuss why.

## 7. Inspect errors and feature importance
Review at least the 10-15 largest held-out absolute errors. Record any recurring pattern you can support from the data. Review permutation importance and explain that importance is model-specific predictive evidence, not proof of causality.

## 8. Generate final evidence
Create:
- `RESULTS.md` with only verified metrics;
- `results/test_predictions.png` showing actual vs selected model vs persistence;
- a concise README results summary;
- completed `PROJECT_JOURNAL.md` entries.

Do not commit raw downloaded data, trained model binaries, or temporary artifacts that are intentionally ignored.

Suggested final student commit:
```text
Document verified Solar Intelligence experiment results
```

## 9. Run the reviewer demo
```bash
streamlit run app.py
```
Confirm Abdalla can explain why the application is a statistical next-day prototype rather than an operational weather forecast.

## Interview check
Abdalla should be able to answer without notes:
1. What is the target variable and how is it shifted to the next day?
2. Why use a persistence baseline?
3. What is the difference between MAE and RMSE?
4. Why can R² be useful but insufficient by itself?
5. Why does chronological splitting matter?
6. What does permutation importance tell you, and what does it not tell you?
