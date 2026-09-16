from __future__ import annotations

import argparse
from pathlib import Path

from solar_intelligence.data import fetch_abu_dhabi_history
from solar_intelligence.modeling import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Solar Intelligence experiment")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--data-out", default="data/abu_dhabi_power.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    frame = fetch_abu_dhabi_history(args.start_year, args.end_year)
    data_path = Path(args.data_out)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path)

    report = train_and_evaluate(frame, args.artifacts)
    print("Selected model:", report["selected_model"])
    print("Test metrics:", report["test_metrics"])
    print("Artifacts written to", args.artifacts)


if __name__ == "__main__":
    main()
