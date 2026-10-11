"""Run the CHAP train/predict contract against the bundled AMFT data."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chap_model import N_SAMPLES, predict_model, train_model  # noqa: E402
from convert_amft_data import convert_amft_csv  # noqa: E402


OUTPUT_DIR = ROOT / "data" / "private" / "chap_smoke_test"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted_path = OUTPUT_DIR / "chap_data.csv"
    data = convert_amft_csv(ROOT / "data" / "sample" / "data.csv", converted_path)
    periods = sorted(data["time_period"].unique())
    if len(periods) < 7:
        raise ValueError("Smoke test needs at least seven monthly periods")

    forecast_periods = periods[-3:]
    historic = data[~data["time_period"].isin(forecast_periods)].copy()
    future = data[data["time_period"].isin(forecast_periods)].drop(columns=["disease_cases"])
    historic_path = OUTPUT_DIR / "historic.csv"
    future_path = OUTPUT_DIR / "future.csv"
    model_path = OUTPUT_DIR / "model.pkl"
    predictions_path = OUTPUT_DIR / "predictions.csv"
    historic.to_csv(historic_path, index=False)
    future.to_csv(future_path, index=False)

    train_model(str(historic_path), str(model_path))
    predict_model(str(model_path), str(historic_path), str(future_path), str(predictions_path))
    predictions = pd.read_csv(predictions_path)
    sample_columns = [column for column in predictions if column.startswith("sample_")]
    if len(sample_columns) != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} sample columns, found {len(sample_columns)}")
    if not predictions[sample_columns].notna().all().all():
        raise ValueError("Prediction samples contain missing values")
    print(
        f"CHAP smoke test passed: {len(predictions):,} forecasts with "
        f"{len(sample_columns)} samples each."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CHAP smoke test failed: {error}", file=sys.stderr)
        raise
