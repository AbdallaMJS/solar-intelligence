from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DEFAULT_PARAMETERS = (
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "CLOUD_AMT",
)
MISSING_SENTINEL = -999.0


@dataclass(frozen=True)
class PowerRequest:
    latitude: float
    longitude: float
    start: str
    end: str
    parameters: tuple[str, ...] = DEFAULT_PARAMETERS

    def validate(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        if len(self.start) != 8 or len(self.end) != 8:
            raise ValueError("Dates must use YYYYMMDD format.")
        if self.start > self.end:
            raise ValueError("Start date must not be after end date.")


def fetch_power_daily(request: PowerRequest, timeout: int = 60) -> pd.DataFrame:
    """Fetch daily meteorological and solar variables from the NASA POWER API."""
    request.validate()
    params = {
        "parameters": ",".join(request.parameters),
        "community": "RE",
        "longitude": request.longitude,
        "latitude": request.latitude,
        "start": request.start,
        "end": request.end,
        "format": "JSON",
    }
    response = requests.get(POWER_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    parameter_block = payload.get("properties", {}).get("parameter", {})
    if not parameter_block:
        raise RuntimeError("NASA POWER response did not contain parameter data.")

    rows: dict[str, dict[str, float]] = {}
    for parameter, values in parameter_block.items():
        for day, value in values.items():
            rows.setdefault(day, {})[parameter] = value

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    frame = frame.sort_index().replace(MISSING_SENTINEL, pd.NA)
    return frame.apply(pd.to_numeric, errors="coerce")


def fetch_abu_dhabi_history(start_year: int = 2018, end_year: int | None = None) -> pd.DataFrame:
    """Convenience loader for the project's reference location in Abu Dhabi, UAE."""
    end_year = end_year or date.today().year - 1
    request = PowerRequest(
        latitude=24.4539,
        longitude=54.3773,
        start=f"{start_year}0101",
        end=f"{end_year}1231",
    )
    return fetch_power_daily(request)
