"""Convert the bundled AMFT schema to the monthly DHIS2 CHAP schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CHAP_COVARIATES = [
    "mean_temperature",
    "rainfall",
    "mean_relative_humidity",
    "average_gpp",
    "malaria_confirmed_u5",
    "pneumonia_cases_u5",
    "diarrhea_u5",
    "low_birth_weight_babies",
    "screened_u5",
    "reporting_rate",
    "population_u5",
]


def convert_amft_csv(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(input_path)
    required = {"time_period", "Acut_Malnutrition", *CHAP_COVARIATES}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing AMFT columns required for CHAP conversion: {missing}")

    if "District" in source.columns:
        location = source["District"].astype("string").str.strip()
    elif "Region_District" in source.columns:
        location = source["Region_District"].astype("string").str.split("|").str[-1].str.strip()
    else:
        raise ValueError("AMFT data must contain District or Region_District")

    converted = pd.DataFrame({
        "time_period": pd.to_datetime(source["time_period"], errors="coerce").dt.strftime("%Y-%m"),
        "location": location,
        "disease_cases": pd.to_numeric(source["Acut_Malnutrition"], errors="coerce"),
    })
    for covariate in CHAP_COVARIATES:
        converted[covariate] = pd.to_numeric(source[covariate], errors="coerce")

    converted["location"] = converted["location"].replace("", pd.NA)
    converted = (
        converted.dropna(subset=["time_period", "location"])
        .sort_values(["location", "time_period"])
        .reset_index(drop=True)
    )
    if converted.duplicated(["location", "time_period"]).any():
        raise ValueError("Converted data contains duplicate location-month rows")
    if converted.empty:
        raise ValueError("No valid location-month rows remain after CHAP conversion")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output_path, index=False)
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    arguments = parser.parse_args()
    result = convert_amft_csv(arguments.input_csv, arguments.output_csv)
    print(f"Wrote {len(result):,} CHAP rows to {arguments.output_csv}")
