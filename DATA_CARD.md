# Data Card — NASA POWER Daily Point Data

## Source
This project retrieves daily meteorological and solar-resource observations from the NASA POWER API rather than redistributing a copied dataset.

Reference location for the main experiment: **Abu Dhabi, UAE (24.4539° N, 54.3773° E)**.

Variables used:
- `ALLSKY_SFC_SW_DWN` — all-sky surface shortwave downward irradiance; also used as the next-day prediction target
- `T2M` — temperature at 2 m
- `RH2M` — relative humidity at 2 m
- `WS2M` — wind speed at 2 m
- `PRECTOTCORR` — corrected precipitation
- `CLOUD_AMT` — cloud amount

## Intended use
The dataset is used for an educational, non-operational machine-learning experiment. The task predicts the next day's solar-resource value from the current day's observations plus seasonal calendar features.

## Split policy
Rows are kept in chronological order. The pipeline uses an earlier training period, a later validation period, and the latest held-out test period. Random shuffling is intentionally avoided because it can leak future information into a forecasting experiment.

## Missing values
NASA POWER missing-value sentinels are converted to missing values and rows required by the feature pipeline are dropped before modeling.

## Limitations
- A single geographic point does not represent all of the UAE.
- Historical reanalysis/observation products are not identical to operational weather forecasts.
- Weather patterns can shift over time, producing distribution drift.
- The project should not be used for power-grid or financial decisions.
- Model performance must be interpreted only for the exact time period and split used in a verified run.
