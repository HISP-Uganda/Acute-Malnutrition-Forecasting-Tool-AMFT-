import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


MODEL_VERSION = "chap-rf-v22-finite-chap-samples"
RISK_LEVELS = ["Monitor", "Alert", "Respond"]
RISK_ORDER = {
    "Monitor": 0,
    "Alert": 1,
    "Respond": 2,
}
RISK_EXPORT_ORDER = {
    "No Data": 0,
    "Monitor": 1,
    "Alert": 2,
    "Respond": 3,
}
LEGACY_DIARRHEA_COVARIATES = ("diarrhea_acute", "diarrhea_persistent")
CLIMATE_COVARIATES = [
    "mean_temperature",
    "rainfall",
    "mean_relative_humidity",
    "average_gpp",
]
CHILD_HEALTH_COVARIATES = [
    "malaria_confirmed_u5",
    "pneumonia_cases_u5",
    "diarrhea_u5",
    "low_birth_weight_babies",
    "screened_u5",
]
RATE_COVARIATES = ["reporting_rate"]
SLOW_MOVING_COVARIATES = ["population_u5"]
DEFAULT_COVARIATES = [
    *CLIMATE_COVARIATES,
    *CHILD_HEALTH_COVARIATES,
    *RATE_COVARIATES,
    *SLOW_MOVING_COVARIATES,
]
CORE_COVARIATES = [
    "mean_temperature",
    "rainfall",
    "average_gpp",
    "malaria_confirmed_u5",
    "diarrhea_u5",
    "screened_u5",
    "reporting_rate",
    "population_u5",
]
SKEWED_COVARIATES = {
    *CHILD_HEALTH_COVARIATES,
    *SLOW_MOVING_COVARIATES,
}
MISSING_FLAG_SUFFIX = "_missing"
OUTLIER_FLAG_SUFFIX = "_outlier"
COUNT_COVARIATES = CHILD_HEALTH_COVARIATES
COVARIATE_ALIASES = {
    "malaria_confirmed_u5": (
        "malaria_confirmed_u5",
        "malaria_confirmed",
    ),
    "pneumonia_cases_u5": (
        "pneumonia_cases_u5",
        "pneumonia_cases",
    ),
    "diarrhea_u5": (
        "diarrhea_u5",
        "diarrhea",
        *LEGACY_DIARRHEA_COVARIATES,
    ),
    "screened_u5": (
        "screened_u5",
        "screening_assessment",
        "screening_assessments",
        "screened_children",
        "children_screened",
        "number_screened",
        "screening_assessment_u5",
    ),
    "population_u5": (
        "population_u5",
        "population",
        "under5population",
        "u5population",
    ),
    "reporting_rate": (
        "reporting_rate",
        "facility_reporting_rate",
        "district_reporting_rate",
        "reporting_completeness",
        "reporting_completeness_pct",
        "hf_reporting_rate",
    ),
}
BASE_FEATURES = [
    "Month_Sin",
    "Month_Cos",
    "Lag1",
    "Lag2",
    "Lag3",
    "Roll3_Mean",
    "Seasonal_Lag12",
]
N_SAMPLES = 100
GAM_RATE_SCALE = 1_000.0
GAM_DETECTION_CI_WIDTH_MAX = 125.0
MIN_WITHIN_HISTORY_MONTHS = 9


def parse_time_period(value: str) -> pd.Timestamp:
    value = str(value).strip()
    for fmt in ("%Y-%m", "%B %Y", "%b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(value, format=fmt))
        except (TypeError, ValueError):
            pass
    return pd.to_datetime(value, errors="coerce")


def classify_risk(value: float, p90: float, p95: float) -> str:
    if pd.isna(value):
        return "No Data"
    if value < p90:
        return "Monitor"
    if value < p95:
        return "Alert"
    return "Respond"


def encode_risk_level(value: str) -> int:
    return int(RISK_EXPORT_ORDER.get(str(value), 0))


def env_flag_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def log_transform_covariate(name: str, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if name in SKEWED_COVARIATES:
        clipped = numeric.clip(lower=0)
        return np.log1p(clipped)
    return numeric


def sanitize_covariate_values(name: str, values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if name in COUNT_COVARIATES:
        return numeric.mask(numeric < 0)
    if name == "reporting_rate":
        numeric = numeric.mask((numeric < 0) | (numeric > 100))
        return numeric.where(numeric <= 1, numeric / 100.0)
    if name == "population_u5":
        return numeric.mask(numeric <= 0)
    if name == "rainfall":
        return numeric.mask(numeric < 0)
    if name == "mean_relative_humidity":
        return numeric.mask((numeric < 0) | (numeric > 100))
    if name == "average_gpp":
        return numeric.mask(numeric < 0)
    return numeric


def _cap_series_with_mad(series: pd.Series, z: float = 5.0, min_obs: int = 4) -> tuple[pd.Series, pd.Series]:
    valid = series.dropna()
    flags = pd.Series(False, index=series.index, dtype=bool)
    if len(valid) < min_obs:
        return series, flags
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad < 1e-6:
        return series, flags
    lower = median - z * mad
    upper = median + z * mad
    flags = series.notna() & ((series < lower) | (series > upper))
    return series.clip(lower=lower, upper=upper), flags


def cap_outliers_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> tuple[pd.Series, pd.Series]:
    capped = pd.Series(index=df.index, dtype="float64")
    flags = pd.Series(0.0, index=df.index, dtype="float64")
    for _, index in df.groupby(group_col).groups.items():
        clipped, group_flags = _cap_series_with_mad(df.loc[index, value_col])
        capped.loc[index] = clipped
        flags.loc[index] = group_flags.astype(float)
    return capped, flags


def cap_outliers_causally(df: pd.DataFrame, group_col: str, date_col: str, value_col: str) -> tuple[pd.Series, pd.Series]:
    """Cap a row using only earlier observations from its location."""
    ordered = df.sort_values([date_col, group_col]).copy()
    capped = pd.Series(np.nan, index=df.index, dtype="float64")
    flags = pd.Series(0.0, index=df.index, dtype="float64")
    history: dict[str, list[float]] = {}

    for _, batch in ordered.groupby(date_col, sort=True):
        for idx in batch.index:
            value = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
            group_key = str(df.loc[idx, group_col])
            prior_values = pd.Series(history.get(group_key, []), dtype="float64").dropna()
            if pd.isna(value) or len(prior_values) < 4:
                capped.loc[idx] = value
                continue
            median = float(prior_values.median())
            mad = float((prior_values - median).abs().median())
            if mad < 1e-6:
                capped.loc[idx] = value
                continue
            lower = median - 5.0 * mad
            upper = median + 5.0 * mad
            flags.loc[idx] = float(value < lower or value > upper)
            capped.loc[idx] = float(np.clip(value, lower, upper))

        for idx in batch.index:
            value = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
            if pd.notna(value):
                history.setdefault(str(df.loc[idx, group_col]), []).append(float(value))

    return capped, flags


def flag_outliers_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    flags = pd.Series(0.0, index=df.index, dtype="float64")
    for _, index in df.groupby(group_col).groups.items():
        _, group_flags = _cap_series_with_mad(df.loc[index, value_col])
        flags.loc[index] = group_flags.astype(float)
    return flags


def sanitize_target_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.mask(numeric < 0)


def add_target_qc_flags(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df["target_duplicate_qc"] = df.duplicated(subset=[group_col, "Date"], keep=False).astype(float)
    df["target_outlier_qc"] = flag_outliers_by_group(df, group_col, target_col)
    return df


def wilson_interval_per_1000(cases: pd.Series, assessed: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return 95% Wilson interval bounds for an observed detection rate."""
    cases = pd.to_numeric(cases, errors="coerce")
    assessed = pd.to_numeric(assessed, errors="coerce")
    valid = cases.notna() & cases.ge(0) & assessed.gt(0) & cases.le(assessed)
    proportion = (cases / assessed).where(valid)
    z = 1.959963984540054
    denominator = 1.0 + (z ** 2 / assessed.where(valid))
    center = (proportion + z ** 2 / (2.0 * assessed.where(valid))) / denominator
    half_width = z * np.sqrt(
        (proportion * (1.0 - proportion) + z ** 2 / (4.0 * assessed.where(valid)))
        / assessed.where(valid)
    ) / denominator
    return (center - half_width) * GAM_RATE_SCALE, (center + half_width) * GAM_RATE_SCALE


def add_gam_rate_target(df: pd.DataFrame, target_col: str, assessed_col: str) -> pd.DataFrame:
    """Create an observed-only GAM detection-rate target without filling missing GAM."""
    df = df.copy()
    observed = pd.to_numeric(df[target_col], errors="coerce")
    assessed = pd.to_numeric(df[assessed_col], errors="coerce") if assessed_col in df.columns else pd.Series(np.nan, index=df.index)
    duplicate = df.get("target_duplicate_qc", pd.Series(0.0, index=df.index)).astype(bool)
    basic_valid = observed.notna() & observed.ge(0) & assessed.gt(0) & observed.le(assessed) & ~duplicate
    ci_lower, ci_upper = wilson_interval_per_1000(observed, assessed)
    ci_width = ci_upper - ci_lower
    imprecise = basic_valid & ci_width.gt(GAM_DETECTION_CI_WIDTH_MAX)
    valid = basic_valid & ~imprecise

    reason = pd.Series("", index=df.index, dtype="object")
    reason.loc[observed.isna()] = "missing_gam"
    reason.loc[observed.notna() & assessed.isna()] = "missing_children_assessed"
    reason.loc[observed.notna() & assessed.notna() & assessed.le(0)] = "invalid_children_assessed"
    reason.loc[observed.notna() & assessed.gt(0) & observed.gt(assessed)] = "gam_exceeds_children_assessed"
    reason.loc[duplicate] = "duplicate_district_month"
    reason.loc[imprecise] = "imprecise_detection_rate"

    df[f"{target_col}_observed"] = observed
    df["children_assessed_observed"] = assessed
    df["gam_detection_ci_lower"] = ci_lower
    df["gam_detection_ci_upper"] = ci_upper
    df["gam_detection_ci_width"] = ci_width
    df["target_precision_qc"] = imprecise.astype(float)
    df["target_valid"] = valid.astype(float)
    df["target_exclusion_reason"] = reason
    df["target_is_observed"] = valid.astype(float)
    df["target_training_weight"] = valid.astype(float)
    df[target_col] = (observed / assessed * GAM_RATE_SCALE).where(valid)
    return df


def _median_or_nan(history: list[float]) -> float:
    return float(np.median(np.asarray(history, dtype=float))) if history else np.nan


def fill_group_month_median(
    df: pd.DataFrame,
    values: pd.Series,
    group_col: str,
    date_col: str,
) -> pd.Series:
    """Causally fill covariates from prior location, seasonal, or global values."""
    ordered = df.sort_values([date_col, group_col])
    filled = pd.Series(np.nan, index=df.index, dtype="float64")
    group_month_history: dict[tuple[str, int], list[float]] = {}
    group_history: dict[str, list[float]] = {}
    global_history: list[float] = []

    for current_date, batch in ordered.groupby(date_col, sort=True):
        for idx in batch.index:
            group_key = str(ordered.loc[idx, group_col])
            month_key = int(pd.Timestamp(current_date).month)
            estimate = _median_or_nan(group_month_history.get((group_key, month_key), []))
            if pd.isna(estimate):
                estimate = _median_or_nan(group_history.get(group_key, []))
            if pd.isna(estimate):
                estimate = _median_or_nan(global_history)
            filled.loc[idx] = estimate

        for idx in batch.index:
            value = pd.to_numeric(values.loc[idx], errors="coerce")
            if pd.notna(value):
                group_key = str(ordered.loc[idx, group_col])
                month_key = int(pd.Timestamp(current_date).month)
                group_month_history.setdefault((group_key, month_key), []).append(float(value))
                group_history.setdefault(group_key, []).append(float(value))
                global_history.append(float(value))
    return filled


def covariate_lag_feature_names(available_columns: list[str] | set[str]) -> list[str]:
    return []


def impute_modeled_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present_covariates = [col for col in DEFAULT_COVARIATES if col in df.columns]

    for col in present_covariates:
        values = sanitize_covariate_values(col, df[col])
        df[col] = values
        df[f"{col}{MISSING_FLAG_SUFFIX}"] = values.isna().astype(float)
        df[col], df[f"{col}{OUTLIER_FLAG_SUFFIX}"] = cap_outliers_causally(df, "location", "Date", col)

    for col in [covariate for covariate in CLIMATE_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0)

    for col in [covariate for covariate in CHILD_HEALTH_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0).clip(lower=0)

    for col in [covariate for covariate in RATE_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill(limit=1))
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0).clip(lower=0, upper=1)

    for col in [covariate for covariate in SLOW_MOVING_COVARIATES if covariate in df.columns]:
        df[col] = df.groupby("location")[col].transform(lambda series: series.ffill())
        df[col] = df[col].fillna(fill_group_month_median(df, df[col], "location", "Date"))
        df[col] = df[col].fillna(0.0).clip(lower=0)

    return df


def resolve_covariate_series(df: pd.DataFrame, covariate: str) -> pd.Series | None:
    if covariate == "diarrhea_u5":
        source = pick_column(df, [covariate, "diarrhea"])
        if source is not None:
            return pd.to_numeric(df[source], errors="coerce")

        legacy_sources = [
            pick_column(df, [legacy_covariate])
            for legacy_covariate in LEGACY_DIARRHEA_COVARIATES
        ]
        legacy_sources = [source_name for source_name in legacy_sources if source_name is not None]
        if not legacy_sources:
            return None

        legacy_values = pd.concat(
            [pd.to_numeric(df[source_name], errors="coerce") for source_name in legacy_sources],
            axis=1,
        )
        return legacy_values.sum(axis=1, min_count=1)

    source = pick_column(df, [covariate, *COVARIATE_ALIASES.get(covariate, ())])
    if source is None:
        return None
    return pd.to_numeric(df[source], errors="coerce")


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_dataframe(df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input CSV is empty.")

    time_col = pick_column(df, ["time_period", "Time_Period"])
    location_col = pick_column(df, ["location", "Region_District", "District", "district"])
    target_col = pick_column(df, ["disease_cases", "Acut_Malnutrition"])

    missing = []
    if time_col is None:
        missing.append("time_period")
    if location_col is None:
        missing.append("location")
    if require_target and target_col is None:
        missing.append("disease_cases")
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    norm = pd.DataFrame(
        {
            "time_period": df[time_col].astype(str).str.strip(),
            "location": df[location_col].astype(str).str.strip(),
        }
    )
    norm["Date"] = norm["time_period"].apply(parse_time_period)
    norm = norm.dropna(subset=["Date"]).copy()

    if target_col is not None:
        norm["disease_cases"] = sanitize_target_values(df.loc[norm.index, target_col])
    elif require_target:
        raise ValueError("Observed inputs must include disease_cases.")
    else:
        norm["disease_cases"] = np.nan

    covariates = []
    for covariate in DEFAULT_COVARIATES:
        values = resolve_covariate_series(df, covariate)
        if values is None:
            continue
        norm[covariate] = pd.to_numeric(values.loc[norm.index], errors="coerce")
        covariates.append(covariate)

    if require_target and "screened_u5" not in norm.columns:
        raise ValueError("Observed inputs must include screened_u5 (children assessed) to calculate the GAM detection-rate target.")

    norm = norm.sort_values(["location", "Date"]).reset_index(drop=True)
    norm = add_target_qc_flags(norm, group_col="location", target_col="disease_cases")
    if require_target:
        norm = add_gam_rate_target(norm, target_col="disease_cases", assessed_col="screened_u5")
    norm = impute_modeled_covariates(norm)
    for covariate in covariates:
        norm[covariate] = log_transform_covariate(covariate, norm[covariate])
    return norm


def infer_covariate_columns(df: pd.DataFrame) -> list[str]:
    covariates = [column for column in CORE_COVARIATES if column in df.columns]
    flags = [
        f"{column}{MISSING_FLAG_SUFFIX}"
        for column in CORE_COVARIATES
        if f"{column}{MISSING_FLAG_SUFFIX}" in df.columns
    ]
    outlier_flags = [
        f"{column}{OUTLIER_FLAG_SUFFIX}"
        for column in CORE_COVARIATES
        if f"{column}{OUTLIER_FLAG_SUFFIX}" in df.columns
    ]
    return covariates + flags + outlier_flags


def transform_target(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log1p(np.clip(values, a_min=0, a_max=None))


def inverse_target(values: float | np.ndarray, scaler: dict | None = None) -> float | np.ndarray:
    """Restore the causal log1p GAM target; scaler is retained for API compatibility."""
    return np.expm1(np.asarray(values))


def build_fallback_scaler(cases: list[float] | np.ndarray) -> dict:
    # Random forests are scale-invariant. Identity values avoid using future
    # observations to normalize an earlier training or validation row.
    return {
        "feature_mean": 0.0,
        "feature_std": 1.0,
        "target_mean": 0.0,
        "target_std": 1.0,
    }


def pad_history(values: list[float], min_length: int, fill_value: float) -> list[float]:
    history = list(values)
    seed = float(fill_value)
    if not history:
        history = [seed]
    while len(history) < min_length:
        history.insert(0, float(history[0]))
    return history


def build_training_features(
    df: pd.DataFrame,
    covariate_cols: list[str],
    horizon: int = 1,
) -> tuple[pd.DataFrame, list[str], dict]:
    if horizon not in (1, 3):
        raise ValueError("Direct forecast horizon must be 1 or 3 months.")
    rows = []
    scalers = {}
    derived_covariate_features = []

    for location in df["location"].unique():
        location_df = df[df["location"] == location].sort_values("Date").reset_index(drop=True)
        cases = location_df["disease_cases"].astype(float).to_numpy()
        if len(location_df) < 4:
            continue

        eligible_indices = [
            i for i in range(2, len(location_df) - horizon)
            if np.isfinite(cases[i - 2:i + 1]).all() and np.isfinite(cases[i + horizon])
        ]
        if not eligible_indices:
            continue

        scalers[location] = build_fallback_scaler([])

        for i in eligible_indices:
            target_index = i + horizon
            hist = cases[:i + 1][np.isfinite(cases[:i + 1])]
            roll3 = hist[-3:]
            current_date = location_df.loc[i, "Date"]

            same_month_hist = location_df.loc[
                (location_df.index <= i) & (location_df["Date"].dt.month == location_df.loc[target_index, "Date"].month),
                "disease_cases",
            ].dropna().astype(float)
            seasonal_lag = float(same_month_hist.iloc[-1]) if not same_month_hist.empty else float(hist[-1])

            row = {
                "location": location,
                "Date": location_df.loc[target_index, "Date"],
                "Forecast_Horizon": horizon,
                "Month_Sin": np.sin(2 * np.pi * location_df.loc[target_index, "Date"].month / 12),
                "Month_Cos": np.cos(2 * np.pi * location_df.loc[target_index, "Date"].month / 12),
                "Lag1": float(hist[-1]),
                "Lag2": float(hist[-2]),
                "Lag3": float(hist[-3]),
                "Roll3_Mean": float(np.mean(roll3)),
                "Seasonal_Lag12": seasonal_lag,
                "Target": float(cases[target_index]),
                "Target_Observed": float(location_df.loc[target_index, "disease_cases_observed"]) if pd.notna(location_df.loc[target_index, "disease_cases_observed"]) else np.nan,
                "Target_Is_Observed": float(location_df.loc[target_index, "target_is_observed"]) if "target_is_observed" in location_df.columns else 1.0,
                "Target_Training_Weight": float(location_df.loc[target_index, "target_training_weight"]) if "target_training_weight" in location_df.columns else 1.0,
                "Target_Transformed": float(transform_target(cases[target_index])),
            }
            for covariate in covariate_cols:
                row[covariate] = float(location_df.loc[i - 1, covariate])
            rows.append(row)

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        raise ValueError("Not enough history to build training features. Each location needs at least 4 observations.")

    feature_cols = BASE_FEATURES + covariate_cols + derived_covariate_features
    return feature_df, feature_cols, scalers


def make_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=1,
        min_samples_split=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def train_model(train_data_path: str, model_path: str) -> None:
    df = normalize_dataframe(pd.read_csv(train_data_path), require_target=True)
    covariate_cols = infer_covariate_columns(df)
    models = {}
    for horizon in (1, 3):
        try:
            feature_df, feature_cols, scalers = build_training_features(df, covariate_cols, horizon=horizon)
        except ValueError:
            continue
        model = make_regressor()
        sample_weight = feature_df["Target_Training_Weight"].fillna(1.0).to_numpy()
        model.fit(feature_df[feature_cols].to_numpy(), feature_df["Target_Transformed"].to_numpy(), sample_weight=sample_weight)
        models[str(horizon)] = {
            "feature_cols": feature_cols,
            "covariate_cols": covariate_cols,
            "scalers": scalers,
            "model": model,
        }
    if not models:
        raise ValueError("Not enough valid GAM history to train a direct 1- or 3-month model.")

    artifact = {
        "model_version": MODEL_VERSION,
        "models": models,
        "fallback_samples": build_fallback_samples(df["disease_cases"]),
    }
    with open(model_path, "wb") as file_obj:
        pickle.dump(artifact, file_obj)


def last_known_covariates(location_df: pd.DataFrame, covariate_cols: list[str]) -> dict:
    values = {}
    for covariate in covariate_cols:
        if covariate not in location_df.columns:
            values[covariate] = 0.0
            continue
        series = location_df[covariate].dropna()
        values[covariate] = float(series.iloc[-1]) if not series.empty else 0.0
    return values


def scale_feature_row(row: dict, scaler: dict) -> dict:
    return row.copy()


def sample_from_forest(model: RandomForestRegressor, feature_vector: np.ndarray, scaler: dict) -> np.ndarray:
    tree_predictions = np.array([tree.predict(feature_vector)[0] for tree in model.estimators_], dtype=float)
    tree_predictions = inverse_target(tree_predictions, scaler)
    tree_predictions = np.clip(tree_predictions, a_min=0, a_max=None)
    if len(tree_predictions) >= N_SAMPLES:
        positions = np.linspace(0, len(tree_predictions) - 1, N_SAMPLES).round().astype(int)
        return tree_predictions[positions]
    if len(tree_predictions) == 0:
        return np.zeros(N_SAMPLES)
    return np.resize(tree_predictions, N_SAMPLES)


def build_fallback_samples(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return finite training-derived samples for CHAP-required output rows."""
    valid = pd.to_numeric(pd.Series(values), errors="coerce").dropna().clip(lower=0).to_numpy(dtype=float)
    if len(valid) == 0:
        return np.zeros(N_SAMPLES, dtype=float)
    return np.quantile(valid, np.linspace(0.01, 0.99, N_SAMPLES)).astype(float)


def direct_model_samples(
    model_setup: dict,
    location: str,
    location_hist: pd.DataFrame,
    hist_cases: list[float],
    target_date: pd.Timestamp,
) -> np.ndarray:
    hist = np.asarray(hist_cases, dtype=float)
    same_month_values = [
        hist_cases[position]
        for position, date_value in enumerate(location_hist["Date"])
        if date_value.month == target_date.month
    ]
    feature_row = {
        "Month_Sin": np.sin(2 * np.pi * target_date.month / 12),
        "Month_Cos": np.cos(2 * np.pi * target_date.month / 12),
        "Lag1": float(hist[-1]),
        "Lag2": float(hist[-2]),
        "Lag3": float(hist[-3]),
        "Roll3_Mean": float(np.mean(hist[-3:])),
        "Seasonal_Lag12": float(same_month_values[-1]) if same_month_values else float(hist[-1]),
    }
    feature_row.update(last_known_covariates(location_hist, model_setup["covariate_cols"]))
    scaler = model_setup["scalers"].get(location, build_fallback_scaler([]))
    feature_vector = np.array([[feature_row[column] for column in model_setup["feature_cols"]]], dtype=float)
    return sample_from_forest(model_setup["model"], feature_vector, scaler)


def build_prediction_rows(
    historic_df: pd.DataFrame,
    future_df: pd.DataFrame,
    model_artifact: dict,
    include_risk_output: bool = False,
) -> pd.DataFrame:
    rows = []
    fallback_samples = np.asarray(model_artifact.get("fallback_samples", np.zeros(N_SAMPLES)), dtype=float)
    fallback_samples = np.nan_to_num(fallback_samples, nan=0.0, posinf=0.0, neginf=0.0)
    if len(fallback_samples) != N_SAMPLES:
        fallback_samples = np.resize(fallback_samples, N_SAMPLES)

    locations = sorted(set(future_df["location"]))
    for location in locations:
        location_hist = historic_df[
            (historic_df["location"] == location) & historic_df["disease_cases"].notna()
        ].sort_values("Date").reset_index(drop=True)
        location_future = future_df[future_df["location"] == location].sort_values("Date").reset_index(drop=True)
        if location_future.empty:
            continue

        hist_cases = location_hist["disease_cases"].dropna().astype(float).tolist()
        last_date = location_hist["Date"].iloc[-1] if hist_cases else pd.NaT

        for i in range(len(location_future)):
            future_row = location_future.iloc[i]
            current_date = future_row["Date"]
            horizon = (
                (current_date.year - last_date.year) * 12 + current_date.month - last_date.month
                if pd.notna(last_date) else 0
            )
            samples = fallback_samples.copy()
            wd_risk = "No Data"
            is_direct_forecast = False

            if len(hist_cases) >= 3:
                if horizon in (1, 3):
                    model_setup = model_artifact["models"].get(str(horizon))
                    if model_setup is not None:
                        samples = direct_model_samples(
                            model_setup, location, location_hist, hist_cases, current_date
                        )
                        is_direct_forecast = True
                elif horizon == 2:
                    # CHAP requires finite samples at every requested timestamp.
                    # Bridge H2 from independent H1/H3 models without recursion.
                    h1_model = model_artifact["models"].get("1")
                    h3_model = model_artifact["models"].get("3")
                    if h1_model is not None and h3_model is not None:
                        h1_samples = direct_model_samples(
                            h1_model,
                            location,
                            location_hist,
                            hist_cases,
                            last_date + pd.DateOffset(months=1),
                        )
                        h3_samples = direct_model_samples(
                            h3_model,
                            location,
                            location_hist,
                            hist_cases,
                            last_date + pd.DateOffset(months=3),
                        )
                        samples = 0.5 * (h1_samples + h3_samples)

            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            point_forecast = float(np.mean(samples))
            if include_risk_output and is_direct_forecast and len(hist_cases) >= MIN_WITHIN_HISTORY_MONTHS:
                p90, p95 = np.percentile(np.asarray(hist_cases, dtype=float), [90, 95])
                wd_risk = classify_risk(point_forecast, p90, p95)

            row = {
                "time_period": str(future_row["time_period"]),
                "location": location,
                "_point_forecast_internal": point_forecast,
                "_wd_risk_internal": wd_risk,
            }
            row.update({f"sample_{index}": float(samples[index]) for index in range(len(samples))})
            if include_risk_output:
                row.update(
                    {
                        "point_forecast": point_forecast,
                        "operational_alert": wd_risk,
                        "operational_alert_code": encode_risk_level(wd_risk),
                    }
                )
            rows.append(row)

    if not rows:
        raise ValueError("No predictions were generated. Check that future locations exist in the historic/training data and each has at least 3 observations.")
    predictions = pd.DataFrame(rows)
    return predictions.drop(
        columns=[col for col in ["_point_forecast_internal", "_wd_risk_internal"] if col in predictions.columns]
    )


def predict_model(
    model_path: str,
    historic_data_path: str,
    future_data_path: str,
    output_path: str,
    include_risk_output: bool | None = None,
) -> None:
    if include_risk_output is None:
        include_risk_output = env_flag_true("CHAP_INCLUDE_RISK_OUTPUT")
    with open(model_path, "rb") as file_obj:
        artifact = pickle.load(file_obj)
    if artifact.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Model version {artifact.get('model_version', 'unknown')} is incompatible with {MODEL_VERSION}. "
            "Retrain the CHAP model with the current code."
        )

    historic_df = normalize_dataframe(pd.read_csv(historic_data_path), require_target=True)
    future_df = normalize_dataframe(pd.read_csv(future_data_path), require_target=False)

    required_covariates = sorted({
        covariate
        for model_setup in artifact["models"].values()
        for covariate in model_setup["covariate_cols"]
    })
    for covariate in required_covariates:
        if covariate not in future_df.columns:
            if covariate.endswith(MISSING_FLAG_SUFFIX):
                future_df[covariate] = 1.0
            elif covariate.endswith(OUTLIER_FLAG_SUFFIX):
                future_df[covariate] = 0.0
            else:
                future_df[covariate] = np.nan

    predictions = build_prediction_rows(
        historic_df,
        future_df,
        artifact,
        include_risk_output=include_risk_output,
    )
    predictions.to_csv(output_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CHAP-compatible acute malnutrition forecasting model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_data")
    train_parser.add_argument("model")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("model")
    predict_parser.add_argument("historic_data")
    predict_parser.add_argument("future_data")
    predict_parser.add_argument("out_file")
    predict_parser.add_argument(
        "--include-risk-output",
        action="store_true",
        help="Append point forecast and direct district operational-alert columns to the CHAP prediction output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        train_model(args.train_data, args.model)
        return

    if args.command == "predict":
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        predict_model(
            args.model,
            args.historic_data,
            args.future_data,
            args.out_file,
            include_risk_output=args.include_risk_output,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
