import calendar
import hashlib
from itertools import product
import json
import os
import warnings
from io import StringIO

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = APP_DIR if os.path.isdir(os.path.join(APP_DIR, "data", "sample")) else os.path.dirname(APP_DIR)
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "Acute_Malnutrition_data_district.csv")
DEFAULT_GEOJSON_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "uganda-districts_ug.geojson")
MODEL_SCHEMA_VERSION = "anomaly-4level-v5"

RISK_LEVELS = ["Low", "Moderate", "High", "Extreme"]
RISK_ORDER = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Extreme": 3,
}
OPERATIONAL_ALERT_LEVELS = ["Monitor", "Alert", "Respond"]
OPERATIONAL_ALERT_ORDER = {
    "Monitor": 0,
    "Alert": 1,
    "Respond": 3,
}
IPC_LEVELS = ["Acceptable", "Alert", "Serious", "Critical", "Extremely Critical"]
IPC_ORDER = {
    "Acceptable": 0,
    "Alert": 1,
    "Serious": 2,
    "Critical": 3,
    "Extremely Critical": 4,
}
RISK_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#ffd11a",
    "High": "#f08a24",
    "Extreme": "#d85c4a",
    "Monitor": "#2ecc71",
    "Alert": "#ffd11a",
    "Respond": "#d85c4a",
    "Acceptable": "#d4edda",
    "Serious": "#fde8d0",
    "Critical": "#f8d7da",
    "Extremely Critical": "#f5c6cb",
    "No Data": "#bdbdbd",
}

ANOMALY_GUIDANCE = {
    "Low": "Within the usual range",
    "Moderate": "Higher than usual",
    "High": "Unusually high",
    "Extreme": "Exceptionally high",
}

SKEWED_COVS = [
    "malaria_confirmed",
    "pneumonia_cases",
    "pregnant_women_with_Anaemia",
    "diarrhea_acute",
    "low_birth_weight_babies",
    "diarrhea_persistent",
    "population",
]

FEATURE_COLS_NEW = [
    "mean_temperature",
    "rainfall",
    "mean_relative_humidity",
    "average_gpp",
    "malaria_confirmed",
    "pneumonia_cases",
    "pregnant_women_with_Anaemia",
    "diarrhea_acute",
    "low_birth_weight_babies",
    "diarrhea_persistent",
    "population",
]

NICE_NAMES = {
    "mean_temperature": "Mean Temperature",
    "rainfall": "Rainfall",
    "mean_relative_humidity": "Relative Humidity",
    "average_gpp": "Avg GPP",
    "malaria_confirmed": "Malaria Confirmed",
    "pneumonia_cases": "Pneumonia Cases",
    "pregnant_women_with_Anaemia": "Pregnant Women w/ Anaemia",
    "diarrhea_acute": "Acute Diarrhoea",
    "low_birth_weight_babies": "Low Birth Weight Babies",
    "diarrhea_persistent": "Persistent Diarrhoea",
    "population": "Population",
    "Acut_Malnutrition": "GAM Caseload (SAM + MAM)",
}

DISPLAY_LABELS = {
    "Date": "Month",
    "time_period": "Reported Period",
    "District_Name": "District",
    "Region_Label": "Region",
    "Region_District": "Region | District",
    "wd_risk": "Within-District Anomaly",
    "xd_risk": "Between-Districts Anomaly",
    "WD_Risk": "Within-District Anomaly",
    "XD_Risk": "Between-Districts Anomaly",
    "Severity_Prevalence_Pct": "GAM Prevalence (%)",
    "Severity_Phase": "IPC AMN Phase",
    "Lower_Severity_Phase": "Lower-Bound IPC AMN Phase",
    "Severity_Basis": "Severity Basis",
    "Caseload_per_100000_Pop": "GAM Caseload per 100,000 Population",
    "Predicted_Caseload_per_100000_Pop": "Forecast GAM Caseload per 100,000 Population",
    "Denominator_Source": "Severity Source",
    "Predicted": "Forecast GAM Caseload",
    "Lower_80": "Lower 80% Bound",
    "Upper_80": "Upper 80% Bound",
    "Operational_Alert": "Operational Alert",
    "Operational_Alert_Why": "Alert Rule",
    "Months_Reported": "Months Reported",
    "Completeness_Pct": "Completeness (%)",
    "Zero_GAM_Months": "Zero GAM Months",
    "Outlier_GAM_Months": "Outlier GAM Months",
    "Missing_Population": "Missing Population",
    "Missing_Severity": "Missing Severity Rows",
    "Districts_Reported": "Districts Reported",
    "Total_GAM_Caseload": "Total GAM Caseload",
    "Zero_GAM_Districts": "Districts with Zero GAM",
    "Reporting_Completeness_Pct": "Reporting Completeness (%)",
}

_EXCLUDE_FROM_FEATURES = {
    "Region_District",
    "District_Name",
    "Region_Label",
    "Display_Name",
    "District",
    "Region",
    "time_period",
    "location",
    "Date",
    "Acut_Malnutrition",
    "wd_risk",
    "xd_risk",
    "wd_score",
    "xd_score",
    "wd_p50",
    "wd_p75",
    "wd_p90",
    "wd_p95",
    "xd_p50",
    "xd_p75",
    "xd_p90",
    "xd_p95",
    "Severity_Prevalence_Pct",
    "Severity_Phase",
    "Severity_Basis",
    "Caseload_per_100000_Pop",
    "Denominator_Source",
    "Operational_Alert",
    "Operational_Alert_Why",
    "month",
    "quarter",
}


# =============================================================================
# FIXED RISK ENCODER
# =============================================================================

class RiskLabelEncoder:
    """Fixed-order encoder. sklearn LabelEncoder sorts alphabetically."""

    classes_ = np.array(RISK_LEVELS)

    def transform(self, values):
        return np.array([RISK_ORDER[v] for v in values], dtype=int)

    def inverse_transform(self, values):
        return np.array([RISK_LEVELS[int(v)] for v in values])


# =============================================================================
# UTILITIES
# =============================================================================

def parse_date(s: str) -> pd.Timestamp:
    s = str(s).strip()
    for fmt in ("%Y-%m", "%B %Y", "%b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            pass
    return pd.to_datetime(s, errors="coerce")


def display_label(col: str) -> str:
    if col in DISPLAY_LABELS:
        return DISPLAY_LABELS[col]
    if col in NICE_NAMES:
        return NICE_NAMES[col]
    return str(col).replace("_", " ")


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: display_label(c) for c in df.columns})


def build_phase_count_table(values: pd.Series, levels: list[str] | None = None, label: str = "Classification") -> pd.DataFrame:
    levels = levels or RISK_LEVELS
    counts = values.value_counts().reindex(levels).fillna(0).astype(int).reset_index()
    counts.columns = [label, "District Count"]
    return counts


def render_phase_pie_chart(counts: pd.DataFrame, level_col: str = "Level"):
    chart_df = counts[counts["District Count"] > 0].copy()
    if chart_df.empty:
        st.info("No districts in the current selection.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_df[level_col],
                values=chart_df["District Count"],
                hole=0.35,
                sort=False,
                marker=dict(colors=[RISK_COLORS.get(level, "#bdbdbd") for level in chart_df[level_col]]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} districts (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)


def build_alert_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in RISK_LEVELS:
        rows.append({
            "Indicator": "Within-District Anomaly",
            "Level": level,
            "District Count": int(df["wd_risk"].value_counts().reindex(RISK_LEVELS).fillna(0).get(level, 0)),
        })
    for level in RISK_LEVELS:
        rows.append({
            "Indicator": "Between-Districts Anomaly",
            "Level": level,
            "District Count": int(df["xd_risk"].value_counts().reindex(RISK_LEVELS).fillna(0).get(level, 0)),
        })
    for level in OPERATIONAL_ALERT_LEVELS:
        rows.append({
            "Indicator": "Operational Alert",
            "Level": level,
            "District Count": int(df["Operational_Alert"].value_counts().reindex(OPERATIONAL_ALERT_LEVELS).fillna(0).get(level, 0)),
        })
    return pd.DataFrame(rows)


def ordered_levels_for_col(col: str) -> list[str]:
    if col in {"Operational_Alert", "Composite_Risk"}:
        return OPERATIONAL_ALERT_LEVELS
    if col in {"Severity_Phase", "Lower_Severity_Phase"}:
        return IPC_LEVELS
    return RISK_LEVELS


def classify_risk(value: float, p50: float, p75: float, p90: float, p95: float) -> str:
    """Map anomaly intensity to a percentile-based four-band alert scale."""
    if pd.isna(value):
        return "No Data"
    if value < p75:
        return "Low"
    if value < p90:
        return "Moderate"
    if value < p95:
        return "High"
    return "Extreme"


def classify_ipc_amn_phase(prevalence_pct: float) -> str:
    """Classify GAM prevalence with IPC AMN phase thresholds."""
    if pd.isna(prevalence_pct):
        return "No Data"
    if prevalence_pct < 5:
        return "Acceptable"
    if prevalence_pct < 10:
        return "Alert"
    if prevalence_pct < 15:
        return "Serious"
    if prevalence_pct < 30:
        return "Critical"
    return "Extremely Critical"


def risk_score(label: str) -> int:
    return RISK_ORDER.get(label, -1)


def operational_alert_score(label: str) -> int:
    return OPERATIONAL_ALERT_ORDER.get(label, -1)


def severity_score(label: str) -> int:
    return {
        "Acceptable": 0,
        "Alert": 1,
        "Serious": 2,
        "Critical": 3,
        "Extremely Critical": 3,
    }.get(label, -1)


def derive_operational_alert(
    wd_risk: str,
    xd_risk: str,
    severity_phase: str,
    lower_severity_phase: str = "No Data",
) -> str:
    """Map within- and between-district anomaly combinations to an operational alert."""
    if wd_risk not in RISK_LEVELS or xd_risk not in RISK_LEVELS:
        return "No Data"
    if wd_risk == "Low" and xd_risk in {"Low", "Moderate"}:
        return "Monitor"
    if (
        (wd_risk == "Low" and xd_risk in {"High", "Extreme"})
        or (wd_risk == "Moderate" and xd_risk in {"Low", "Moderate"})
    ):
        return "Alert"
    if wd_risk in {"High", "Extreme"} or (wd_risk == "Moderate" and xd_risk in {"High", "Extreme"}):
        return "Respond"
    return "No Data"


def explain_operational_alert(
    wd_risk: str,
    xd_risk: str,
    severity_phase: str,
    lower_severity_phase: str = "No Data",
) -> str:
    alert = derive_operational_alert(wd_risk, xd_risk, severity_phase, lower_severity_phase)
    if alert == "Monitor":
        return f"Monitor because within-district anomaly is {wd_risk} and between-districts anomaly is {xd_risk}."
    if alert == "Alert":
        return f"Alert because within-district anomaly is {wd_risk} and between-districts anomaly is {xd_risk}."
    if alert == "Respond":
        return f"Respond because within-district anomaly is {wd_risk} and between-districts anomaly is {xd_risk}."
    return "Operational alert unavailable because one or both anomaly classifications are missing."


def normalise_operational_alert_labels(values: pd.Series) -> pd.Series:
    return values.replace({
        "Routine": "Monitor",
        "Watch": "Alert",
        "Prepare": "Respond",
    })


def ensure_observed_alert_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Operational_Alert" not in df.columns:
        df["Operational_Alert"] = df.apply(
            lambda row: derive_operational_alert(
                row.get("wd_risk", "No Data"),
                row.get("xd_risk", "No Data"),
                row.get("Severity_Phase", "No Data"),
            ),
            axis=1,
        )
    else:
        df["Operational_Alert"] = normalise_operational_alert_labels(df["Operational_Alert"])
    if "Operational_Alert_Why" not in df.columns:
        df["Operational_Alert_Why"] = df.apply(
            lambda row: explain_operational_alert(
                row.get("wd_risk", "No Data"),
                row.get("xd_risk", "No Data"),
                row.get("Severity_Phase", "No Data"),
            ),
            axis=1,
        )
    return df


def ensure_forecast_alert_columns(fc_df: pd.DataFrame) -> pd.DataFrame:
    fc_df = fc_df.copy()
    if fc_df.empty:
        return fc_df
    if "Operational_Alert" not in fc_df.columns:
        fc_df["Operational_Alert"] = fc_df.apply(
            lambda row: derive_operational_alert(
                row.get("WD_Risk", "No Data"),
                row.get("XD_Risk", "No Data"),
                row.get("Severity_Phase", "No Data"),
                row.get("Lower_Severity_Phase", "No Data"),
            ),
            axis=1,
        )
    else:
        fc_df["Operational_Alert"] = normalise_operational_alert_labels(fc_df["Operational_Alert"])
    if "Operational_Alert_Why" not in fc_df.columns:
        fc_df["Operational_Alert_Why"] = fc_df.apply(
            lambda row: explain_operational_alert(
                row.get("WD_Risk", "No Data"),
                row.get("XD_Risk", "No Data"),
                row.get("Severity_Phase", "No Data"),
                row.get("Lower_Severity_Phase", "No Data"),
            ),
            axis=1,
        )
    if "Composite_Risk" not in fc_df.columns:
        fc_df["Composite_Risk"] = fc_df["Operational_Alert"]
    return fc_df


def normalise_col_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_matching_col(columns: list[str], aliases: list[str]) -> str | None:
    norm_map = {normalise_col_name(col): col for col in columns}
    for alias in aliases:
        col = norm_map.get(alias)
        if col is not None:
            return col
    return None


def get_severity_setup(df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    whz_col = find_matching_col(columns, [
        "gamwhz", "gamwhzpct", "gamwhzpercent", "gamwhzprevalence",
        "gamprevalence", "gampct", "gampercent", "globalacutemalnutritionprevalence",
    ])
    muac_col = find_matching_col(columns, [
        "gammuac", "gammuacpct", "gammuacpercent", "gammuacprevalence",
    ])
    total_pop_col = find_matching_col(columns, ["population", "totalpopulation"])

    if whz_col:
        return {
            "observed_method": "gam_whz",
            "prevalence_col": whz_col,
            "denominator_col": None,
            "population_col": total_pop_col,
            "basis": f"GAM prevalence from `{whz_col}`",
            "note": "Standards-based severity uses direct GAM prevalence, aligned to IPC AMN thresholds.",
            "forecast_method": "unavailable",
            "forecast_note": "Forecast severity phase unavailable because the model predicts GAM caseload and the dataset does not include a forecast denominator.",
        }
    if muac_col:
        return {
            "observed_method": "gam_muac",
            "prevalence_col": muac_col,
            "denominator_col": None,
            "population_col": total_pop_col,
            "basis": f"GAM MUAC prevalence from `{muac_col}`",
            "note": "Standards-based severity uses direct GAM MUAC prevalence. IPC recommends MUAC-based phase use only where WHZ is unavailable and supported by convergence of evidence.",
            "forecast_method": "unavailable",
            "forecast_note": "Forecast severity phase unavailable because the model predicts GAM caseload and the dataset does not include a forecast denominator.",
        }
    return {
        "observed_method": "unavailable",
        "prevalence_col": None,
        "denominator_col": None,
        "population_col": total_pop_col,
        "basis": "Unavailable",
        "note": "Standards-based severity requires direct GAM prevalence (WHZ/MUAC). This dataset currently supports anomaly-based warning labels and GAM caseload forecasting, but not standards-based severity calculation.",
        "forecast_method": "unavailable",
        "forecast_note": "Forecast severity phase unavailable because the model predicts GAM caseload and the dataset does not provide a direct GAM prevalence forecast target.",
    }


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def get_denominator_source_label(severity_setup: dict) -> str:
    method = severity_setup["observed_method"]
    if method == "gam_whz":
        return f"Reported GAM prevalence `{severity_setup['prevalence_col']}`"
    if method == "gam_muac":
        return f"Reported GAM MUAC prevalence `{severity_setup['prevalence_col']}`"
    return "Unavailable"


def get_denominator_source_short(severity_setup: dict) -> str:
    method = severity_setup["observed_method"]
    if method in {"gam_whz", "gam_muac"}:
        return "Direct GAM"
    return "Unavailable"


def build_methodology_df(df: pd.DataFrame) -> pd.DataFrame:
    severity_setup = get_severity_setup(df)
    rows = [
        {
            "Component": "Outcome",
            "Definition": "GAM caseload from `Acut_Malnutrition`, interpreted as SAM + MAM.",
        },
        {
            "Component": "Within-district anomaly",
            "Definition": "Compares each district with its own historical distribution.",
        },
        {
            "Component": "Between-districts anomaly",
            "Definition": "Compares districts against peer and seasonal historical patterns.",
        },
        {
            "Component": "Operational alert",
            "Definition": "Uses a fixed combination rule based on within-district and between-district anomaly levels.",
        },
        {
            "Component": "Data note",
            "Definition": severity_setup["note"],
        },
    ]
    return pd.DataFrame(rows)


def build_alert_actions_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Anomaly Level": "Low",
            "Percentile": "<75th percentile",
            "Interpretation": "Within the usual range",
        },
        {
            "Anomaly Level": "Moderate",
            "Percentile": "75th-<90th percentile",
            "Interpretation": "Higher than usual",
        },
        {
            "Anomaly Level": "High",
            "Percentile": "90th-<95th percentile",
            "Interpretation": "Unusually high",
        },
        {
            "Anomaly Level": "Extreme",
            "Percentile": ">=95th percentile",
            "Interpretation": "Exceptionally high",
        },
    ])


def build_operational_alert_rules_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Operational Alert": "Monitor",
            "Combination Rule": "Within = Low AND Between = (Low OR Moderate)",
            "Action": "Routine monitoring",
        },
        {
            "Operational Alert": "Alert",
            "Combination Rule": "Within = Low AND Between = (High OR Extreme), OR Within = Moderate AND Between = (Low OR Moderate)",
            "Action": "Heightened surveillance",
        },
        {
            "Operational Alert": "Respond",
            "Combination Rule": "Within = High OR Extreme, OR Within = Moderate AND Between = (High OR Extreme)",
            "Action": "Immediate response",
        },
    ])


def _count_iqr_outliers(values: pd.Series) -> int:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 4:
        return 0
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0
    upper = q3 + 1.5 * iqr
    return int((vals > upper).sum())


def compute_data_quality_tables(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    severity_setup = get_severity_setup(df)
    pop_col = severity_setup["population_col"]
    districts = sorted(df["District_Name"].unique())
    expected_months = len(pd.period_range(df["Date"].min(), df["Date"].max(), freq="M"))
    total_districts = len(districts)

    district_rows = []
    for district in districts:
        d = df[df["District_Name"] == district].sort_values("Date")
        pop_missing = 0
        if pop_col is not None and pop_col in d.columns:
            pop_missing = int(pd.to_numeric(d[pop_col], errors="coerce").isna().sum())

        district_rows.append({
            "District": district,
            "Months_Reported": int(d["Date"].nunique()),
            "Completeness_Pct": round(d["Date"].nunique() / expected_months * 100, 1) if expected_months else np.nan,
            "Zero_GAM_Months": int((pd.to_numeric(d["Acut_Malnutrition"], errors="coerce").fillna(0) == 0).sum()),
            "Outlier_GAM_Months": _count_iqr_outliers(d["Acut_Malnutrition"]),
            "Missing_Population": pop_missing,
            "Missing_Severity": int(d["Severity_Prevalence_Pct"].isna().sum()),
            "Denominator_Source": d["Denominator_Source"].iloc[0] if "Denominator_Source" in d.columns and not d.empty else "Unavailable",
        })

    district_quality = pd.DataFrame(district_rows).sort_values(
        ["Completeness_Pct", "Missing_Severity", "Outlier_GAM_Months"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    monthly = (
        df.groupby("Date")
        .agg(
            Districts_Reported=("District_Name", "nunique"),
            Total_GAM_Caseload=("Acut_Malnutrition", "sum"),
            Zero_GAM_Districts=("Acut_Malnutrition", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) == 0).sum())),
            Missing_Severity=("Severity_Prevalence_Pct", lambda s: int(s.isna().sum())),
        )
        .reset_index()
        .sort_values("Date")
    )
    monthly["Reporting_Completeness_Pct"] = (monthly["Districts_Reported"] / total_districts * 100).round(1) if total_districts else np.nan
    monthly["Date"] = monthly["Date"].dt.strftime("%Y-%m")

    duplicate_rows = int(df.duplicated(subset=["District_Name", "Date"]).sum())
    summary = {
        "expected_months": expected_months,
        "median_completeness": float(district_quality["Completeness_Pct"].median()) if not district_quality.empty else 0.0,
        "zero_gam_months": int(district_quality["Zero_GAM_Months"].sum()) if not district_quality.empty else 0,
        "outlier_gam_months": int(district_quality["Outlier_GAM_Months"].sum()) if not district_quality.empty else 0,
        "missing_severity_rows": int(df["Severity_Prevalence_Pct"].isna().sum()),
        "duplicate_rows": duplicate_rows,
    }
    return summary, district_quality, monthly


def apply_log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def transform_regression_target(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log1p(np.clip(values, a_min=0, a_max=None))


def inverse_regression_target(values: float | np.ndarray, scaler: dict) -> float | np.ndarray:
    restored = np.expm1(np.asarray(values) * scaler["target_std"] + scaler["target_mean"])
    restored = np.clip(restored, a_min=0, a_max=None)
    if np.isscalar(values):
        return float(restored)
    return restored


def guess_name_col(columns: list[str]) -> str:
    keywords = ["district", "dist", "name", "admin", "adm"]
    non_geom = [c for c in columns if c != "geometry"]
    for kw in keywords:
        for col in non_geom:
            if kw in col.lower():
                return col
    return non_geom[0] if non_geom else ""


def safe_geojson(gdf: gpd.GeoDataFrame) -> dict:
    g = gdf.copy()
    g["geometry"] = g["geometry"].simplify(0.01, preserve_topology=True)
    for col in g.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(g[col]):
            g[col] = g[col].astype(str)
        elif g[col].dtype == object:
            g[col] = g[col].fillna("Unknown").astype(str)
        elif pd.api.types.is_float_dtype(g[col]):
            g[col] = g[col].fillna(0).replace([np.inf, -np.inf], 0)
    return json.loads(g.to_json(na="drop", show_bbox=False))


def colour_risk_df(df_styler, columns):
    risk_bg = {
        "Low": RISK_COLORS["Low"],
        "Moderate": RISK_COLORS["Moderate"],
        "High": RISK_COLORS["High"],
        "Extreme": RISK_COLORS["Extreme"],
        "Monitor": RISK_COLORS["Monitor"],
        "Alert": RISK_COLORS["Alert"],
        "Respond": RISK_COLORS["Respond"],
        "Extremely Critical": "#f5c6cb",
        "Critical": "#f8d7da",
        "Serious": "#fde8d0",
        "Acceptable": "#d4edda",
        "No Data": "#f0f0f0",
    }
    risk_fg = {
        "Low": "#0b2e13",
        "Moderate": "#4d3d00",
        "High": "#ffffff",
        "Extreme": "#ffffff",
        "Monitor": "#0b2e13",
        "Alert": "#4d3d00",
        "Respond": "#ffffff",
        "Extremely Critical": "#58151c",
        "Critical": "#721c24",
        "Serious": "#8a4500",
        "Acceptable": "#155724",
        "No Data": "#555",
    }

    def _style(val):
        if val in risk_bg:
            return f"background-color:{risk_bg[val]};color:{risk_fg[val]};font-weight:bold"
        return ""

    return df_styler.map(_style, subset=columns)


def colour_alert_summary_df(df: pd.DataFrame):
    risk_bg = {
        "Low": RISK_COLORS["Low"],
        "Moderate": RISK_COLORS["Moderate"],
        "High": RISK_COLORS["High"],
        "Extreme": RISK_COLORS["Extreme"],
        "Monitor": RISK_COLORS["Monitor"],
        "Alert": RISK_COLORS["Alert"],
        "Respond": RISK_COLORS["Respond"],
        "No Data": "#f0f0f0",
    }
    risk_fg = {
        "Low": "#0b2e13",
        "Moderate": "#4d3d00",
        "High": "#ffffff",
        "Extreme": "#ffffff",
        "Monitor": "#0b2e13",
        "Alert": "#4d3d00",
        "Respond": "#ffffff",
        "No Data": "#555",
    }

    def _style_row(row):
        level = row.get("Level", row.get("Anomaly Level", row.get("Operational Alert")))
        if level in risk_bg:
            style = f"background-color:{risk_bg[level]};color:{risk_fg[level]};font-weight:bold"
            return [style] * len(row)
        return [""] * len(row)

    return df.style.apply(_style_row, axis=1)


def compute_per_class_metrics(all_true: list[int], all_pred: list[int]) -> dict:
    conf = confusion_matrix(all_true, all_pred, labels=list(range(len(RISK_LEVELS))))
    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(len(RISK_LEVELS))),
        target_names=RISK_LEVELS,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for i, lvl in enumerate(RISK_LEVELS):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        tn = conf.sum() - tp - fp - fn

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        lr_pos = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float("inf")
        lr_neg = (1 - sensitivity) / specificity if specificity > 0 else float("inf")

        rows.append({
            "Risk Level": lvl,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision (PPV)": precision,
            "NPV": npv,
            "F1": report.get(lvl, {}).get("f1-score", 0.0),
            "LR+": lr_pos,
            "LR-": lr_neg,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Support": int(report.get(lvl, {}).get("support", 0)),
        })
    return {"per_class": rows, "conf_matrix": conf, "report": report}


def compute_spearman_correlations(data_slice: pd.DataFrame):
    target = "Acut_Malnutrition"
    candidates = [c for c in FEATURE_COLS_NEW if c in data_slice.columns]
    rows = []

    for col in candidates:
        pair = data_slice[[target, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 6 or pair[target].nunique() < 2 or pair[col].nunique() < 2:
            continue
        rho = pair[target].corr(pair[col], method="spearman")
        if pd.notna(rho):
            rows.append({
                "Variable": col,
                "Display": NICE_NAMES.get(col, col),
                "Spearman rho": float(rho),
                "Abs rho": float(abs(rho)),
                "N": int(len(pair)),
            })

    return pd.DataFrame(rows).sort_values("Abs rho", ascending=False).reset_index(drop=True)


def draw_spearman_radial(corr_df: pd.DataFrame):
    if corr_df.empty:
        return None

    plot_df = corr_df.head(10).copy()
    n = len(plot_df)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = plot_df["Abs rho"].to_numpy()
    signed = plot_df["Spearman rho"].to_numpy()
    labels = plot_df["Display"].tolist()

    fig, ax = plt.subplots(
        figsize=(7.1, 7.1),
        subplot_kw={"projection": "polar"},
        facecolor="#f7f4ef",
    )
    ax.set_facecolor("#f7f4ef")

    max_r = max(0.40, float(vals.max()) + 0.06)
    for rv in np.arange(0.1, max_r, 0.1):
        ax.plot(
            np.linspace(0, 2 * np.pi, 360),
            [rv] * 360,
            color="#2f2f2f",
            lw=0.55,
            alpha=0.15,
            zorder=1,
        )
        ax.text(
            np.pi / 2,
            rv + 0.006,
            f"r={rv:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555",
            fontfamily="monospace",
            alpha=0.80,
        )

    pos_color = "#08aeca"
    neg_color = "#ef4056"
    width = 2 * np.pi / n * 0.58

    def wrap_label(label: str, width: int = 12) -> str:
        words = str(label).split()
        if not words:
            return str(label)
        lines = []
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if len(trial) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        if len(lines) > 2:
            lines = lines[:2]
            if not lines[-1].endswith("..."):
                lines[-1] = lines[-1][: max(0, width - 3)].rstrip() + "..."
        return "\n".join(lines)

    for angle, rho_abs, rho, label in zip(angles, vals, signed, labels):
        color = pos_color if rho >= 0 else neg_color
        for glow_width, alpha in [(width * 2.7, 0.05), (width * 1.8, 0.10), (width, 0.95)]:
            ax.bar(
                angle,
                rho_abs,
                width=glow_width,
                bottom=0.035,
                color=color,
                alpha=alpha,
                edgecolor="none",
                zorder=3,
            )
        ax.scatter(
            angle,
            rho_abs + 0.035,
            s=70,
            color=color,
            edgecolors="#111",
            linewidths=1.1,
            zorder=6,
        )
        ax.text(
            angle,
            rho_abs + 0.075,
            f"{rho:+.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
            fontfamily="monospace",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground="#f7f4ef")],
        )
        short = wrap_label(label)
        ax.text(
            angle,
            max(rho_abs + 0.11, 0.18),
            short,
            ha="center",
            va="center",
            fontsize=7.4,
            color="#303040",
            fontweight="bold",
            fontfamily="monospace",
            linespacing=1.05,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="#f7f4ef")],
        )

    for size, alpha in [(430, 0.04), (260, 0.10), (130, 0.22), (70, 0.95)]:
        ax.scatter(0, 0, s=size, color="#f5c518", alpha=alpha, zorder=9)
    ax.text(
        0,
        0,
        "malnutrition",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#101010",
        fontfamily="monospace",
        zorder=10,
    )

    ax.set_ylim(0, max_r + 0.14)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    fig.subplots_adjust(top=0.96, bottom=0.05, left=0.05, right=0.95)
    return fig


def render_compare_time_series(df: pd.DataFrame, reference_label: str = "All Districts"):
    st.subheader(f"Compare Districts or Regions Against {reference_label} Reference")
    compare_mode = st.radio(
        "Compare by",
        ["Districts", "Regions"],
        horizontal=True,
        key="compare_mode",
    )

    national = (
        df.groupby("Date")["Acut_Malnutrition"]
        .agg(Total="sum", Mean="mean", Median="median")
        .reset_index()
        .sort_values("Date")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=national["Date"],
        y=national["Total"],
        mode="lines",
        name=f"{reference_label} - Total (right axis)",
        yaxis="y2",
        line=dict(color="#34495e", width=2.2, dash="dash"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Total: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=national["Date"],
        y=national["Mean"],
        mode="lines",
        name=f"{reference_label} - Mean (left axis)",
        yaxis="y1",
        line=dict(color="#95a5a6", width=2.8),
        hovertemplate="<b>%{x|%b %Y}</b><br>Mean per district: %{y:,.1f}<extra></extra>",
    ))

    palette = px.colors.qualitative.Bold
    if compare_mode == "Districts":
        options = sorted(df["Display_Name"].unique())
        defaults = options[: min(3, len(options))]
        selected = st.multiselect("Select districts", options=options, default=defaults, key="compare_districts")
        for i, display_name in enumerate(selected):
            sub = df[df["Display_Name"] == display_name].sort_values("Date")
            fig.add_trace(go.Scatter(
                x=sub["Date"],
                y=sub["Acut_Malnutrition"],
                mode="lines+markers",
                name=display_name,
                yaxis="y1",
                line=dict(color=palette[i % len(palette)], width=2),
                marker=dict(size=5),
                hovertemplate="<b>%{x|%b %Y}</b><br>Cases: %{y:,.0f}<extra></extra>",
            ))
        y1_title = "Cases per District"
    else:
        options = sorted(df["Region_Label"].dropna().unique())
        defaults = options[: min(3, len(options))]
        selected = st.multiselect("Select regions", options=options, default=defaults, key="compare_regions")
        region_month = (
            df[df["Region_Label"].isin(selected)]
            .groupby(["Date", "Region_Label"])["Acut_Malnutrition"]
            .agg("mean")
            .reset_index()
            .sort_values("Date")
        )
        for i, region in enumerate(selected):
            sub = region_month[region_month["Region_Label"] == region]
            fig.add_trace(go.Scatter(
                x=sub["Date"],
                y=sub["Acut_Malnutrition"],
                mode="lines+markers",
                name=region,
                yaxis="y1",
                line=dict(color=palette[i % len(palette)], width=2.2),
                marker=dict(size=5),
                hovertemplate="<b>%{x|%b %Y}</b><br>Mean cases per district: %{y:,.1f}<extra></extra>",
            ))
        y1_title = "Mean Cases per District in Region"

    fig.update_layout(
        height=520,
        hovermode="x unified",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title=y1_title, showgrid=True, gridcolor="#ececec"),
        yaxis2=dict(title=f"Total - {reference_label}", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_historical_risk_trend(df: pd.DataFrame, scope_label: str = "All Districts"):
    st.subheader(f"Historical Anomaly Trend - {scope_label}")
    risk_source = st.radio(
        "Anomaly definition",
        ["Within-District Anomaly", "Between-Districts Anomaly"],
        horizontal=True,
        key="risk_trend_source",
    )
    risk_col = "wd_risk" if risk_source.startswith("Within") else "xd_risk"

    monthly = (
        df.groupby(["Date", risk_col]).size()
        .unstack(fill_value=0)
        .reindex(columns=RISK_LEVELS, fill_value=0)
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    total = monthly[RISK_LEVELS].sum(axis=1).replace(0, np.nan)
    for lvl in RISK_LEVELS:
        monthly[f"{lvl}_pct"] = monthly[lvl] / total * 100

    monthly["_n0"] = 0
    cumulative = monthly["_n0"]
    for i, lvl in enumerate(RISK_LEVELS, start=1):
        cumulative = cumulative + monthly[lvl]
        monthly[f"_n{i}"] = cumulative

    focus_level = RISK_LEVELS[-1]
    monthly[f"{focus_level}_6m"] = monthly[f"{focus_level}_pct"].rolling(6, min_periods=1, center=True).mean()
    total_max = float(monthly[f"_n{len(RISK_LEVELS)}"].max())
    monthly[f"{focus_level}_6m_scaled"] = monthly[f"{focus_level}_6m"] / 100 * total_max

    fig = go.Figure()
    for i, lvl in enumerate(RISK_LEVELS, start=1):
        bottom = f"_n{i - 1}"
        top = f"_n{i}"
        color = RISK_COLORS[lvl]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=monthly["Date"],
            y=monthly[top],
            mode="lines",
            line=dict(color=color, width=0.8),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([monthly["Date"], monthly["Date"][::-1]]),
            y=pd.concat([monthly[top], monthly[bottom][::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.78)",
            line=dict(color="rgba(0,0,0,0)"),
            name=lvl,
            customdata=np.stack([
                monthly[lvl],
                monthly[f"{lvl}_pct"].fillna(0).round(1),
            ], axis=-1).tolist() * 2,
            hovertemplate="<b>%{x|%b %Y}</b><br>" + lvl + ": %{customdata[0]} districts (%{customdata[1]}%)<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=monthly["Date"],
        y=monthly[f"{focus_level}_6m_scaled"],
        mode="lines",
        name=f"{focus_level} % (6m avg)",
        line=dict(color="#c0392b", width=2.4, dash="dash"),
        customdata=monthly[f"{focus_level}_6m"].fillna(0),
        hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{focus_level} trend: %{{customdata:.1f}}%<extra></extra>",
    ))

    peak_series = monthly[f"{focus_level}_pct"].dropna() if not monthly.empty else pd.Series(dtype=float)
    if not peak_series.empty:
        peak_idx = peak_series.idxmax()
        peak_date = monthly.loc[peak_idx, "Date"]
        peak_pct = monthly.loc[peak_idx, f"{focus_level}_pct"]
        fig.add_vline(x=peak_date, line_dash="dot", line_color="#e74c3c", line_width=1.8, opacity=0.75)
        fig.add_annotation(
            x=peak_date,
            y=total_max * 1.04,
            text=f"Peak: {peak_pct:.0f}% {focus_level}",
            showarrow=False,
            font=dict(size=10, color="#c0392b"),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="#e74c3c",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        height=500,
        xaxis_title="",
        yaxis_title="Number of Districts",
        hovermode="x unified",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="v", x=1.01, y=1, xanchor="left"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if monthly.empty:
        latest_month = "No data"
        avg_e = np.nan
        peak_e = np.nan
        recent = np.nan
        prior = np.nan
    else:
        latest_month = monthly["Date"].iloc[-1].strftime("%b %Y")
        avg_e = monthly[f"{focus_level}_pct"].mean()
        peak_e = monthly[f"{focus_level}_pct"].max()
        recent = monthly[f"{focus_level}_pct"].tail(3).mean()
        prior = (
            monthly[f"{focus_level}_pct"].iloc[-9:-3].mean()
            if len(monthly) >= 9 else monthly[f"{focus_level}_pct"].mean()
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest month", latest_month)
    k2.metric(f"Avg % {focus_level}", f"{avg_e:.1f}%" if pd.notna(avg_e) else "N/A")
    k3.metric(f"Peak {focus_level}", f"{peak_e:.1f}%" if pd.notna(peak_e) else "N/A")
    k4.metric(
        "Recent 3m trend",
        "Rising" if pd.notna(recent) and pd.notna(prior) and recent > prior else "Falling" if pd.notna(recent) and pd.notna(prior) else "N/A",
        delta=f"{recent - prior:+.1f}pp" if pd.notna(recent) and pd.notna(prior) else None,
        delta_color="inverse",
    )


# =============================================================================
# DATA LOADING AND DISTRICT RISK CLASSIFICATION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, _hash: str) -> pd.DataFrame:
    content = file_bytes.decode("utf-8")
    return load_data_from_df(pd.read_csv(StringIO(content)))


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return load_data_from_df(pd.read_csv(path))


def load_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["time_period", "Acut_Malnutrition", "Region_District"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = df["time_period"].apply(parse_date)
    df = df.dropna(subset=["Date", "Acut_Malnutrition"]).copy()

    raw_rd = df["Region_District"].astype(str).str.strip()
    df["Region_District"] = raw_rd

    if "District" in df.columns:
        df["District_Name"] = df["District"].astype(str).str.strip()
    else:
        df["District_Name"] = raw_rd.str.split("|").str[-1].str.strip()

    has_pipe = raw_rd.str.contains("|", regex=False)
    df["Region_Label"] = ""
    df.loc[has_pipe, "Region_Label"] = raw_rd[has_pipe].str.split("|").str[0].str.strip()
    if "Region" in df.columns:
        df.loc[~has_pipe, "Region_Label"] = df.loc[~has_pipe, "Region"].astype(str).str.strip()
    else:
        df.loc[~has_pipe, "Region_Label"] = "Unknown"

    df["Display_Name"] = df["Region_Label"] + " - " + df["District_Name"]
    df = df.sort_values(["District_Name", "Date"]).reset_index(drop=True)

    severity_setup = get_severity_setup(df)
    df["Severity_Basis"] = severity_setup["basis"]
    df["Severity_Prevalence_Pct"] = np.nan
    df["Severity_Phase"] = "No Data"
    df["Caseload_per_100000_Pop"] = np.nan
    df["Denominator_Source"] = get_denominator_source_label(severity_setup)
    df["Operational_Alert"] = "No Data"
    df["Operational_Alert_Why"] = ""

    if severity_setup["population_col"] is not None:
        total_pop = pd.to_numeric(df[severity_setup["population_col"]], errors="coerce").replace(0, np.nan)
        df["Caseload_per_100000_Pop"] = pd.to_numeric(df["Acut_Malnutrition"], errors="coerce") / total_pop * 100000

    if severity_setup["observed_method"] in {"gam_whz", "gam_muac"}:
        prevalence = pd.to_numeric(df[severity_setup["prevalence_col"]], errors="coerce")
        df["Severity_Prevalence_Pct"] = prevalence
        df["Severity_Phase"] = prevalence.apply(classify_ipc_amn_phase)

    df[["wd_p50", "wd_p75", "wd_p90", "wd_p95"]] = np.nan
    df["wd_risk"] = "No Data"

    for district in df["District_Name"].unique():
        mask = df["District_Name"] == district
        d = df[mask].sort_values("Date")
        for idx in d.index:
            past = d.loc[d["Date"] < df.loc[idx, "Date"], "Acut_Malnutrition"]
            if len(past) >= 2:
                p50, p75, p90, p95 = np.percentile(past, [50, 75, 90, 95])
            else:
                v = df.loc[idx, "Acut_Malnutrition"]
                p50 = p75 = p90 = p95 = v
            df.loc[idx, ["wd_p50", "wd_p75", "wd_p90", "wd_p95"]] = p50, p75, p90, p95
            df.loc[idx, "wd_risk"] = classify_risk(df.loc[idx, "Acut_Malnutrition"], p50, p75, p90, p95)

    df[["xd_p50", "xd_p75", "xd_p90", "xd_p95"]] = np.nan
    df["xd_risk"] = "No Data"

    for date in sorted(df["Date"].unique()):
        month = date.month
        curr_idx = df[df["Date"] == date].index
        past_all = df[df["Date"] < date]

        hist = past_all[past_all["Date"].dt.month == month]["Acut_Malnutrition"]
        if len(hist) < 3:
            season_months = [(month - 1) % 12 + 1, month, month % 12 + 1]
            hist = past_all[past_all["Date"].dt.month.isin(season_months)]["Acut_Malnutrition"]
        if len(hist) < 3:
            hist = past_all["Acut_Malnutrition"]
        if len(hist) < 3:
            hist = df["Acut_Malnutrition"]

        p50, p75, p90, p95 = np.percentile(hist, [50, 75, 90, 95])
        for idx in curr_idx:
            val = df.loc[idx, "Acut_Malnutrition"]
            df.loc[idx, ["xd_p50", "xd_p75", "xd_p90", "xd_p95"]] = p50, p75, p90, p95
            df.loc[idx, "xd_risk"] = classify_risk(val, p50, p75, p90, p95)

    df["wd_score"] = df["wd_risk"].map(RISK_ORDER).fillna(0).astype(int)
    df["xd_score"] = df["xd_risk"].map(RISK_ORDER).fillna(0).astype(int)
    df["Operational_Alert"] = df.apply(
        lambda row: derive_operational_alert(
            row["wd_risk"],
            row["xd_risk"],
            row["Severity_Phase"],
        ),
        axis=1,
    )
    df["Operational_Alert_Why"] = df.apply(
        lambda row: explain_operational_alert(
            row["wd_risk"],
            row["xd_risk"],
            row["Severity_Phase"],
        ),
        axis=1,
    )
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter

    return df.reset_index(drop=True)


# =============================================================================
# FEATURE ENGINEERING  (pruned)
# =============================================================================
#
# Removed lower-importance / redundant engineered features:
# Lag2_Z, Lag3_Z, Roll6_Max, Roll12_Mean, Roll12_Max, Trend, Acceleration,
# Hist_P75, Lag1_Over_P75, Lag1_Over_P90, Lag1_Ratio_P75, Lag1_Ratio_P90.
# Hist_P90 is retained because it helps separate High from Extreme conditions.

@st.cache_data(show_spinner=False)
def build_features(_df: pd.DataFrame):
    records = []

    for district in _df["District_Name"].unique():
        d = _df[_df["District_Name"] == district].sort_values("Date").reset_index(drop=True)
        adm = d["Acut_Malnutrition"].astype(float).values

        for i in range(3, len(d)):
            hist = adm[:i]
            roll3 = adm[i - 3:i]
            roll6 = adm[max(0, i - 6):i]

            hist_mu = float(np.mean(hist))
            hist_sd = float(np.std(hist)) if np.std(hist) > 1e-6 else 1.0
            hist_p90, hist_p95 = np.percentile(hist, [90, 95])

            lag1 = float(adm[i - 1])
            lag2 = float(adm[i - 2])
            lag3 = float(adm[i - 3])

            same_month_hist = d.loc[
                (d.index < i) & (d["Date"].dt.month == d.loc[i, "Date"].month),
                "Acut_Malnutrition",
            ].astype(float)
            seasonal_lag_12 = float(same_month_hist.iloc[-1]) if len(same_month_hist) else lag1

            recent_scores = d.loc[i - 3:i - 1, "wd_score"].astype(float).values
            recent_high = np.isin(recent_scores, [2, 3]).sum()

            row = {
                "District": district,
                "Date": d.loc[i, "Date"],
                # Calendar
                "Month": d.loc[i, "Date"].month,
                "Quarter": d.loc[i, "Date"].quarter,
                "Month_Sin": np.sin(2 * np.pi * d.loc[i, "Date"].month / 12),
                "Month_Cos": np.cos(2 * np.pi * d.loc[i, "Date"].month / 12),
                # Raw lags
                "Lag1": lag1,
                "Lag2": lag2,
                "Lag3": lag3,
                # Standardised lag
                "Lag1_Z": (lag1 - hist_mu) / hist_sd,
                # Rolling windows
                "Roll3_Mean": float(np.mean(roll3)),
                "Roll3_Std": float(np.std(roll3)),
                "Roll6_Mean": float(np.mean(roll6)),
                # Seasonal reference
                "Seasonal_Lag12": seasonal_lag_12,
                # Historical distribution
                "Hist_Mean": hist_mu,
                "Hist_Std": hist_sd,
                "Hist_P90": hist_p90,
                "Hist_P95": hist_p95,
                # Threshold exceedance / ratio
                "Lag1_Over_P95": lag1 - hist_p95,
                "Lag1_Ratio_P95": lag1 / (hist_p95 + 1e-6),
                # Risk history
                "Recent_WD_High_Count": recent_high,
                "WD_Score_Lag": d.loc[i - 1, "wd_score"],
                "XD_Score_Lag": d.loc[i - 1, "xd_score"],
                "Target_Adm": adm[i],
                "Target_WD_Risk": d.loc[i, "wd_risk"],
                "Target_XD_Risk": d.loc[i, "xd_risk"],
            }

            # Passthrough numeric covariates, lagged by one month.
            for col in d.columns:
                if col not in row and col not in _EXCLUDE_FROM_FEATURES:
                    if pd.api.types.is_numeric_dtype(d[col]):
                        row[col] = d.loc[i - 1, col]

            records.append(row)

    feat_df = pd.DataFrame(records).dropna().reset_index(drop=True)
    feat_df = apply_log_transform(feat_df, SKEWED_COVS)

    scale_cols = [
        "Lag1", "Lag2", "Lag3",
        "Roll3_Mean", "Roll6_Mean",
        "Seasonal_Lag12",
        "Hist_Mean", "Hist_Std",
        "Hist_P90", "Hist_P95",
        "Lag1_Over_P95",
    ]

    scalers = {}
    feat_df["Target_Adm_Transformed"] = np.nan
    for district in feat_df["District"].unique():
        mask = feat_df["District"] == district
        baseline = feat_df.loc[mask, "Lag1"].astype(float)
        feature_mu = float(baseline.mean())
        feature_std = float(baseline.std())
        if feature_std < 1e-6:
            feature_std = 1.0

        target_log = transform_regression_target(feat_df.loc[mask, "Target_Adm"].astype(float))
        target_mu = float(target_log.mean())
        target_std = float(target_log.std())
        if target_std < 1e-6:
            target_std = 1.0

        scalers[district] = {
            "feature_mean": feature_mu,
            "feature_std": feature_std,
            "target_mean": target_mu,
            "target_std": target_std,
        }

        for col in scale_cols:
            if col in feat_df.columns:
                feat_df.loc[mask, col] = (feat_df.loc[mask, col] - feature_mu) / feature_std
        feat_df.loc[mask, "Target_Adm_Transformed"] = (target_log - target_mu) / target_std

    for col in feat_df.columns:
        if col not in {"District", "Date", "Target_WD_Risk", "Target_XD_Risk"}:
            if pd.api.types.is_integer_dtype(feat_df[col]):
                feat_df[col] = feat_df[col].astype("float64")

    targets = {
        "District",
        "Date",
        "Target_Adm",
        "Target_Adm_Transformed",
        "Target_WD_Risk",
        "Target_XD_Risk",
    }
    feature_cols = [c for c in feat_df.columns if c not in targets]
    return feat_df, feature_cols, scalers


# =============================================================================
# WALK-FORWARD CV AND MODELS
# =============================================================================

@st.cache_data(show_spinner=False)
def wf_splits(_feat_df: pd.DataFrame, n_splits: int = 6, min_months: int = 12):
    dates = sorted(_feat_df["Date"].unique())
    n = len(dates)
    if n < min_months + 2:
        return []

    cutoff = dates[0] + pd.DateOffset(months=min_months)
    min_i = next((i for i, d in enumerate(dates) if d >= cutoff), n - 1)
    step = max(1, (n - min_i) // n_splits)

    splits = []
    for i in range(n_splits):
        ci = min_i + i * step
        if ci >= n - 1:
            break
        te = min(ci + step, n - 1)
        tr_i = _feat_df[_feat_df["Date"].isin(dates[:ci])].index
        te_i = _feat_df[_feat_df["Date"].isin(dates[ci:te])].index
        if len(tr_i) >= 10 and len(te_i) >= 5:
            splits.append((tr_i, te_i))
    return splits


def make_classifier(target: str) -> RandomForestClassifier:
    if target == "WD":
        class_weight = {0: 0.8, 1: 2.0, 2: 3.8, 3: 5.8}
        max_depth = 16
        min_leaf = 1
    else:
        class_weight = {0: 1.0, 1: 1.5, 2: 2.6, 3: 4.0}
        max_depth = 14
        min_leaf = 2

    return RandomForestClassifier(
        n_estimators=700,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        min_samples_split=3,
        max_features="sqrt",
        class_weight=class_weight,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )


def make_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=700,
        max_depth=14,
        min_samples_leaf=1,
        min_samples_split=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def predict_with_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    adjusted = proba / thresholds.reshape(1, -1)
    return adjusted.argmax(axis=1)


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray, target: str) -> np.ndarray:
    if target == "WD":
        grids = [
            np.linspace(0.55, 0.85, 4),
            np.linspace(0.26, 0.52, 4),
            np.linspace(0.12, 0.34, 5),
            np.linspace(0.08, 0.24, 5),
        ]
    else:
        grids = [
            np.linspace(0.38, 0.62, 4),
            np.linspace(0.24, 0.46, 4),
            np.linspace(0.16, 0.36, 5),
            np.linspace(0.10, 0.26, 5),
        ]

    best = (
        np.array([0.68, 0.36, 0.22, 0.12])
        if target == "WD"
        else np.array([0.50, 0.34, 0.24, 0.14])
    )
    best_score = -np.inf

    for thr_values in product(*grids):
        thr = np.array(thr_values)
        pred = predict_with_thresholds(proba, thr)
        recalls = recall_score(
            y_true,
            pred,
            labels=list(range(len(RISK_LEVELS))),
            average=None,
            zero_division=0,
        )
        precision_high = precision_score(
            np.isin(y_true, [2, 3]).astype(int),
            np.isin(pred, [2, 3]).astype(int),
            zero_division=0,
        )
        if target == "WD":
            score = (
                0.12 * recalls[0]
                + 0.20 * recalls[1]
                + 0.32 * recalls[2]
                + 0.36 * recalls[3]
                + 0.05 * precision_high
            )
        else:
            score = recalls.mean() + 0.05 * precision_high
        if score > best_score:
            best_score = score
            best = thr
    return best


def full_proba(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    full = np.zeros((len(X), len(RISK_LEVELS)))
    for pos, cls in enumerate(model.classes_):
        full[:, cls] = proba[:, pos]
    return full


@st.cache_data(show_spinner=False)
def run_classification(_feat_df: pd.DataFrame, feature_cols: list[str], target: str = "WD"):
    target_col = f"Target_{target}_Risk"
    le = RiskLabelEncoder()
    valid = _feat_df[_feat_df[target_col].isin(RISK_LEVELS)].copy().reset_index(drop=True)
    valid["y"] = le.transform(valid[target_col])

    X = valid[feature_cols].values
    y = valid["y"].values
    splits = wf_splits(valid)
    if not splits:
        return {"error": f"Not enough data for {target} risk classification."}

    cv_acc, cv_f1, cv_bal_acc, cv_macro_recall = [], [], [], []
    all_true, all_pred = [], []

    for tr_i, te_i in splits:
        tr_p = valid.index.get_indexer(tr_i)
        tr_p = tr_p[tr_p >= 0]
        te_p = valid.index.get_indexer(te_i)
        te_p = te_p[te_p >= 0]
        if not len(tr_p) or not len(te_p):
            continue

        model = make_classifier(target)
        model.fit(X[tr_p], y[tr_p])
        thresholds = tune_thresholds(y[tr_p], full_proba(model, X[tr_p]), target)
        p = predict_with_thresholds(full_proba(model, X[te_p]), thresholds)

        cv_acc.append(accuracy_score(y[te_p], p))
        cv_f1.append(f1_score(y[te_p], p, average="weighted", zero_division=0))
        cv_bal_acc.append(balanced_accuracy_score(y[te_p], p))
        cv_macro_recall.append(recall_score(y[te_p], p, average="macro", zero_division=0))
        all_true.extend(y[te_p].tolist())
        all_pred.extend(p.tolist())

    final_model = make_classifier(target)
    final_model.fit(X, y)
    thresholds = tune_thresholds(y, full_proba(final_model, X), target)

    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    pcm = compute_per_class_metrics(all_true, all_pred)
    return {
        "cv_acc": float(np.mean(cv_acc)),
        "cv_acc_std": float(np.std(cv_acc)),
        "cv_f1": float(np.mean(cv_f1)),
        "cv_bal_acc": float(np.mean(cv_bal_acc)),
        "cv_macro_recall": float(np.mean(cv_macro_recall)),
        "feat_imp": feat_imp,
        "model": final_model,
        "thresholds": thresholds,
        "le": le,
        "n_folds": len(splits),
        "target": target,
        "per_class": pcm["per_class"],
        "all_true": all_true,
        "all_pred": all_pred,
    }


@st.cache_data(show_spinner=False)
def run_regression_cv(_feat_df: pd.DataFrame, feature_cols: list[str], _scalers: dict):
    valid = _feat_df.dropna(subset=["Target_Adm_Transformed"]).copy().reset_index(drop=True)
    X = valid[feature_cols].values
    y = valid["Target_Adm_Transformed"].values
    splits = wf_splits(valid)
    if not splits:
        return {"error": "Not enough data for regression CV."}

    mae_cv, rmse_cv, resid_rows = [], [], []
    for tr_i, te_i in splits:
        tr_p = valid.index.get_indexer(tr_i)
        tr_p = tr_p[tr_p >= 0]
        te_p = valid.index.get_indexer(te_i)
        te_p = te_p[te_p >= 0]
        if not len(tr_p) or not len(te_p):
            continue

        model = make_regressor()
        model.fit(X[tr_p], y[tr_p])
        pred_transformed = model.predict(X[te_p])
        meta = valid.iloc[te_p]
        actual_raw = meta["Target_Adm"].values
        pred_raw = np.array([
            inverse_regression_target(pred_transformed[j], _scalers[meta.iloc[j]["District"]])
            for j in range(len(te_p))
        ])

        mae_cv.append(mean_absolute_error(actual_raw, pred_raw))
        rmse_cv.append(np.sqrt(mean_squared_error(actual_raw, pred_raw)))

        for j in range(len(te_p)):
            residual = actual_raw[j] - pred_raw[j]
            resid_rows.append({
                "District": meta.iloc[j]["District"],
                "Date": meta.iloc[j]["Date"],
                "XD_Risk": meta.iloc[j]["Target_XD_Risk"],
                "Actual": actual_raw[j],
                "Predicted": pred_raw[j],
                "Residual": residual,
                "Signed_Error": pred_raw[j] - actual_raw[j],
                "Absolute_Error": abs(residual),
            })

    final_model = make_regressor()
    final_model.fit(X, y)
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    residuals = pd.DataFrame(resid_rows)

    return {
        "cv_mae": float(np.mean(mae_cv)),
        "cv_mae_std": float(np.std(mae_cv)),
        "cv_rmse": float(np.mean(rmse_cv)),
        "cv_mean_error": float(residuals["Signed_Error"].mean()) if not residuals.empty else np.nan,
        "cv_median_ae": float(residuals["Absolute_Error"].median()) if not residuals.empty else np.nan,
        "cv_p90_ae": float(residuals["Absolute_Error"].quantile(0.90)) if not residuals.empty else np.nan,
        "feat_imp": feat_imp,
        "residuals": residuals,
        "n_folds": len(splits),
        "model": final_model,
        "target_transform": "log1p raw caseload, z-scored within district",
    }


# =============================================================================
# FORECASTING
# =============================================================================

class Forecaster:
    HORIZON = 3
    # Engineered features computed at inference time. Keep in sync with build_features.
    BASE = [
        "Month", "Quarter", "Month_Sin", "Month_Cos",
        "Lag1", "Lag2", "Lag3",
        "Lag1_Z",
        "Roll3_Mean", "Roll3_Std", "Roll6_Mean",
        "Seasonal_Lag12",
        "Hist_Mean", "Hist_Std", "Hist_P90", "Hist_P95",
        "Lag1_Over_P95",
        "Lag1_Ratio_P95",
        "Recent_WD_High_Count", "WD_Score_Lag", "XD_Score_Lag",
    ]

    def fit(self, feat_df: pd.DataFrame, feature_cols: list[str], scalers: dict):
        self.feature_cols = feature_cols
        self.scalers = scalers
        self.extra = [c for c in feature_cols if c not in self.BASE]

        valid = feat_df.dropna(subset=["Target_Adm_Transformed"]).copy()
        self.extra_vals = {}
        for d in valid["District"].unique():
            last = valid[valid["District"] == d].iloc[-1]
            self.extra_vals[d] = {c: float(last.get(c, 0.0)) for c in self.extra}

        self.reg = make_regressor()
        self.reg.fit(valid[feature_cols].values, valid["Target_Adm_Transformed"].values)

        # Restore classifier-driven within-district anomaly prediction for forecasts.
        self.wd_clf = None
        self.wd_thresholds = None
        self.wd_le = None
        wd_valid = valid[valid["Target_WD_Risk"].isin(RISK_LEVELS)].copy()
        if not wd_valid.empty:
            self.wd_le = RiskLabelEncoder()
            wd_y = self.wd_le.transform(wd_valid["Target_WD_Risk"])
            self.wd_clf = make_classifier("WD")
            self.wd_clf.fit(wd_valid[feature_cols].values, wd_y)
            self.wd_thresholds = tune_thresholds(wd_y, full_proba(self.wd_clf, wd_valid[feature_cols].values), "WD")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        severity_setup = get_severity_setup(df)

        for district in df["District_Name"].unique():
            d = df[df["District_Name"] == district].sort_values("Date")
            if len(d) < 3:
                continue

            scaler = self.scalers.get(district)
            if scaler is None:
                continue
            mu = scaler["feature_mean"]
            sg = max(scaler["feature_std"], 1e-6)

            hist_adm = list(d["Acut_Malnutrition"].astype(float).values)
            hist_wd = list(d["wd_score"].values)
            hist_xd = list(d["xd_score"].values)
            last_date = d["Date"].iloc[-1]
            extras = self.extra_vals.get(district, {c: 0.0 for c in self.extra})
            total_pop_col = severity_setup["population_col"]
            total_pop = np.nan
            if total_pop_col is not None and total_pop_col in d.columns:
                total_pop = pd.to_numeric(d[total_pop_col], errors="coerce").dropna().iloc[-1] if not d[total_pop_col].dropna().empty else np.nan

            for step in range(1, self.HORIZON + 1):
                fd = last_date + pd.DateOffset(months=step)
                hist = np.array(hist_adm, dtype=float)
                roll3 = hist[-3:]
                roll6 = hist[-6:] if len(hist) >= 6 else roll3
                hist_mu = float(hist.mean())
                hist_sd = float(hist.std()) if hist.std() > 1e-6 else 1.0
                #p90, p95 = np.percentile(hist, [90, 95])
                p75, p90, p95 = np.percentile(hist, [75, 90, 95])
                lag1, lag2, lag3 = hist[-1], hist[-2], hist[-3]

                base_raw = {
                    # Calendar
                    "Month": fd.month,
                    "Quarter": fd.quarter,
                    "Month_Sin": np.sin(2 * np.pi * fd.month / 12),
                    "Month_Cos": np.cos(2 * np.pi * fd.month / 12),
                    # Raw lags
                    "Lag1": (lag1 - mu) / sg,
                    "Lag2": (lag2 - mu) / sg,
                    "Lag3": (lag3 - mu) / sg,
                    # Standardised lag
                    "Lag1_Z": (lag1 - hist_mu) / hist_sd,
                    # Rolling windows
                    "Roll3_Mean": (roll3.mean() - mu) / sg,
                    "Roll3_Std": roll3.std(),
                    "Roll6_Mean": (roll6.mean() - mu) / sg,
                    # Seasonal reference
                    "Seasonal_Lag12": (hist[-12] - mu) / sg if len(hist) >= 12 else (lag1 - mu) / sg,
                    # Historical distribution
                    "Hist_Mean": (hist_mu - mu) / sg,
                    "Hist_Std": (hist_sd - mu) / sg,
                    "Hist_P90": (p90 - mu) / sg,
                    "Hist_P95": (p95 - mu) / sg,
                    # Threshold exceedance / ratio
                    "Lag1_Over_P95": (lag1 - p95) / sg,
                    "Lag1_Ratio_P95": lag1 / (p95 + 1e-6),
                    # Risk history
                    "Recent_WD_High_Count": np.isin(hist_wd[-3:], [2, 3]).sum(),
                    "WD_Score_Lag": hist_wd[-1],
                    "XD_Score_Lag": hist_xd[-1],
                }

                fv = np.array([[{**base_raw, **extras}.get(c, 0.0) for c in self.feature_cols]])
                pred_transformed = float(self.reg.predict(fv)[0])
                pred_raw = round(inverse_regression_target(pred_transformed, scaler))

                tree_preds = np.array([t.predict(fv)[0] for t in self.reg.estimators_])
                tree_preds_raw = inverse_regression_target(tree_preds, scaler)
                lo, hi = np.percentile(tree_preds_raw, [10, 90])
                lo = max(0, round(lo))
                hi = max(lo, round(hi))

                p50 = np.percentile(hist, 50)
                if self.wd_clf is not None and self.wd_thresholds is not None and self.wd_le is not None:
                    wd_code = int(predict_with_thresholds(full_proba(self.wd_clf, fv), self.wd_thresholds)[0])
                    wd_risk = self.wd_le.inverse_transform([wd_code])[0]
                else:
                    wd_risk = classify_risk(pred_raw, p50, p75, p90, p95)
                pred_rate = pred_raw / total_pop * 100000 if pd.notna(total_pop) and total_pop > 0 else np.nan
                lower_severity_phase = "No Data"
                severity_phase = "No Data"
                rows.append({
                    "District": district,
                    "Date": fd,
                    "Month_Year": fd.strftime("%B %Y"),
                    "Step": step,
                    "Predicted": int(pred_raw),
                    "Lower_80": int(lo),
                    "Upper_80": int(hi),
                    "WD_Risk": wd_risk,
                    "XD_Risk": "No Data",
                    "Composite_Risk": "No Data",
                    "Operational_Alert": "No Data",
                    "Operational_Alert_Why": "",
                    "Severity_Prevalence_Pct": np.nan,
                    "Severity_Phase": severity_phase,
                    "Lower_Severity_Phase": lower_severity_phase,
                    "Severity_Basis": severity_setup["forecast_note"],
                    "Predicted_Caseload_per_100000_Pop": pred_rate,
                })

                hist_adm.append(pred_raw)
                hist_wd.append(RISK_ORDER.get(wd_risk, 0))
                hist_xd.append(0)

        result = pd.DataFrame(rows)
        if result.empty:
            return result

        all_hist_vals = df["Acut_Malnutrition"].dropna().values
        for month_date in result["Date"].unique():
            mask = result["Date"] == month_date
            preds = result.loc[mask, "Predicted"].values
            if len(preds) >= 3:
                xp50, xp75, xp90, xp95 = np.percentile(preds, [50, 75, 90, 95])
            else:
                xp50, xp75, xp90, xp95 = np.percentile(all_hist_vals, [50, 75, 90, 95])
            for idx in result[mask].index:
                pred = result.loc[idx, "Predicted"]
                xd_risk = classify_risk(pred, xp50, xp75, xp90, xp95)
                wd_risk = result.loc[idx, "WD_Risk"]
                severity_phase = result.loc[idx, "Severity_Phase"]
                lower_severity_phase = result.loc[idx, "Lower_Severity_Phase"]
                operational_alert = derive_operational_alert(wd_risk, xd_risk, severity_phase, lower_severity_phase)
                operational_alert_why = explain_operational_alert(wd_risk, xd_risk, severity_phase, lower_severity_phase)
                result.loc[idx, "XD_Risk"] = xd_risk
                result.loc[idx, "Operational_Alert"] = operational_alert
                result.loc[idx, "Operational_Alert_Why"] = operational_alert_why
                result.loc[idx, "Composite_Risk"] = operational_alert

        return result


@st.cache_data(show_spinner=False)
def run_forecast(_feat_df, feature_cols, _scalers, _df, data_key):
    fc = Forecaster().fit(_feat_df, feature_cols, _scalers)
    return fc.predict(_df)


# =============================================================================
# MAPS
# =============================================================================

def load_and_match_geodf(geo_file, df: pd.DataFrame, name_col: str):
    gdf = gpd.read_file(geo_file)
    gdf = gdf.rename(columns={name_col: "District_Name"})
    gdf["District_Name"] = gdf["District_Name"].astype(str).str.strip()

    csv_lower = {n.lower(): n for n in df["District_Name"].unique()}
    remap = {
        gn: csv_lower[gn.lower()]
        for gn in gdf["District_Name"].unique()
        if gn.lower() in csv_lower and gn != csv_lower[gn.lower()]
    }
    if remap:
        gdf["District_Name"] = gdf["District_Name"].replace(remap)

    matched = len(set(gdf["District_Name"]) & set(df["District_Name"]))
    return gdf, matched


def render_map(gdf, data, risk_col, title, hover_cols=None):
    merged = gdf.merge(data, on="District_Name", how="left")
    levels = ordered_levels_for_col(risk_col)
    if risk_col in merged.columns:
        merged[risk_col] = pd.Categorical(merged[risk_col], categories=levels, ordered=True)
    gj = safe_geojson(merged)
    hover_cols = hover_cols or []
    hover_data = {c: True for c in hover_cols if c in merged.columns}
    hover_data[risk_col] = True

    fig = px.choropleth_mapbox(
        merged,
        geojson=gj,
        locations=merged.index,
        color=risk_col,
        color_discrete_map=RISK_COLORS,
        category_orders={risk_col: levels},
        labels={risk_col: display_label(risk_col)},
        zoom=5.1,
        center={"lat": 1.37, "lon": 32.29},
        mapbox_style="carto-positron",
        hover_name="District_Name",
        hover_data=hover_data,
        title=title,
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=360)
    st.plotly_chart(fig, use_container_width=True)


def render_delta_map(gdf, fc_map, risk_col, months):
    if len(months) < 2:
        return

    first, last = months[0], months[-1]
    first_df = fc_map[fc_map["Month_Year"] == first][["District_Name", risk_col]].rename(columns={risk_col: "Risk_First"})
    last_df = fc_map[fc_map["Month_Year"] == last][["District_Name", risk_col]].rename(columns={risk_col: "Risk_Last"})
    delta = first_df.merge(last_df, on="District_Name", how="inner")
    score_order = OPERATIONAL_ALERT_ORDER if risk_col in {"Operational_Alert", "Composite_Risk"} else RISK_ORDER
    delta["Score_First"] = delta["Risk_First"].map(score_order).fillna(0).astype(int)
    delta["Score_Last"] = delta["Risk_Last"].map(score_order).fillna(0).astype(int)
    delta["Delta"] = delta["Score_Last"] - delta["Score_First"]
    delta["Delta_Label"] = delta["Delta"].apply(lambda x: f"+{x}" if x > 0 else str(x))
    delta["Direction"] = delta["Delta"].apply(lambda x: "Worsening" if x > 0 else ("Improving" if x < 0 else "No change"))

    merged = gdf.merge(delta, on="District_Name", how="left")
    merged["Delta_Label"] = merged["Delta_Label"].fillna("0")
    color_map = {
        "-4": "#08306b",
        "-3": "#08519c",
        "-2": "#3182bd",
        "-1": "#9ecae1",
        "0": "#e0e0e0",
        "+1": "#fdae6b",
        "+2": "#e6550d",
        "+3": "#a63603",
        "+4": "#7f2704",
    }
    gj = safe_geojson(merged)
    fig = px.choropleth_mapbox(
        merged,
        geojson=gj,
        locations=merged.index,
        color="Delta_Label",
        color_discrete_map=color_map,
        zoom=5.1,
        center={"lat": 1.37, "lon": 32.29},
        mapbox_style="carto-positron",
        hover_name="District_Name",
        hover_data={"Risk_First": True, "Risk_Last": True, "Direction": True, "Delta_Label": True},
        title=f"Risk Change: {first} to {last}",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=380)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# APP
# =============================================================================

def inject_app_styles():
    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: #eef4f8;
            padding: 0.35rem;
            border-radius: 0.9rem;
            border: 1px solid #d4e0e8;
            margin-bottom: 0.85rem;
        }

        button[data-baseweb="tab"] {
            min-height: 2.8rem;
            padding: 0.55rem 1rem;
            border-radius: 0.7rem;
            border: 1px solid transparent;
            background: #f8fbfd;
            color: #234;
            font-size: 1rem;
            font-weight: 700;
            transition: all 0.18s ease;
        }

        button[data-baseweb="tab"]:hover {
            background: #e3eef6;
            color: #12344d;
            border-color: #b8cfde;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: #1f5f8b;
            color: #ffffff;
            border-color: #1f5f8b;
            box-shadow: 0 6px 16px rgba(31, 95, 139, 0.18);
        }

        button[data-baseweb="tab"] p {
            font-size: 1rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Acute Malnutrition Forecasting Tool",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_app_styles()
    st.title("Acute Malnutrition Forecasting Tool")
    st.caption("A tool for district anomaly monitoring, forecasting, and operational alerts")

    with st.sidebar:
        st.header("Data")
        csv_mode = st.radio("CSV source", ["Use local CSV path", "Upload CSV"], horizontal=False)
        csv_path = st.text_input("CSV path", value=DEFAULT_CSV_PATH, disabled=(csv_mode != "Use local CSV path"))
        csv_file = None
        if csv_mode == "Upload CSV":
            csv_file = st.file_uploader("Upload CSV", type="csv")

        st.header("GeoJSON")
        geo_mode = st.radio("GeoJSON source", ["Use local GeoJSON path", "Upload GeoJSON"], horizontal=False)
        geo_path = st.text_input("GeoJSON path", value=DEFAULT_GEOJSON_PATH, disabled=(geo_mode != "Use local GeoJSON path"))
        geo_file = None
        geo_source = geo_path if geo_mode == "Use local GeoJSON path" else None
        if geo_mode == "Upload GeoJSON":
            geo_file = st.file_uploader("Upload district GeoJSON", type=["json", "geojson"])
            geo_source = geo_file
        geo_name_col = None
        if geo_source:
            try:
                preview = gpd.read_file(geo_source)
                if hasattr(geo_source, "seek"):
                    geo_source.seek(0)
                non_geom = [c for c in preview.columns if c != "geometry"]
                auto_guess = guess_name_col(list(preview.columns))
                geo_name_col = st.selectbox(
                    "District name column in GeoJSON",
                    options=non_geom,
                    index=non_geom.index(auto_guess) if auto_guess in non_geom else 0,
                )
            except Exception as e:
                st.warning(f"Could not preview GeoJSON: {e}")

        st.markdown("---")
        st.markdown("**Risk rule**")
        st.caption("Within-district compares each district with its own history. Between-districts compares districts with peers. Anomaly classification uses Low (<75th percentile), Moderate (75th-<90th percentile), High (90th-<95th percentile), and Extreme (>=95th percentile).")

    try:
        if csv_mode == "Upload CSV":
            if csv_file is None:
                st.info("Upload your CSV to start.")
                return
            raw = csv_file.read()
            data_key = file_hash(raw)
            df = load_data_from_bytes(raw, data_key)
        else:
            data_key = csv_path
            df = load_data_from_path(csv_path)
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        return

    df = ensure_observed_alert_columns(df)

    with st.spinner("Engineering district-relative features..."):
        feat_df, feature_cols, scalers = build_features(df)

    if (
        st.session_state.get("_data_key") != data_key
        or st.session_state.get("_model_schema_version") != MODEL_SCHEMA_VERSION
    ):
        with st.spinner("Training Random Forest models and generating 3-month forecasts..."):
            st.session_state["reg_eval"] = run_regression_cv(feat_df, feature_cols, scalers)
            st.session_state["clf_wd"] = run_classification(feat_df, feature_cols, target="WD")
            st.session_state["clf_xd"] = run_classification(feat_df, feature_cols, target="XD")
            st.session_state["fc_df"] = run_forecast(feat_df, feature_cols, scalers, df, data_key)
            st.session_state["_data_key"] = data_key
            st.session_state["_model_schema_version"] = MODEL_SCHEMA_VERSION

    reg_eval = st.session_state["reg_eval"]
    clf_wd = st.session_state["clf_wd"]
    clf_xd = st.session_state["clf_xd"]
    fc_df = ensure_forecast_alert_columns(st.session_state["fc_df"])
    st.session_state["fc_df"] = fc_df

    district_scope = (
        df[["District_Name", "Display_Name"]]
        .drop_duplicates()
        .sort_values("Display_Name")
        .reset_index(drop=True)
    )
    all_districts = district_scope["District_Name"].tolist()
    all_display_districts = district_scope["Display_Name"].tolist()
    display_to_district = dict(zip(district_scope["Display_Name"], district_scope["District_Name"]))
    district_to_display = dict(zip(district_scope["District_Name"], district_scope["Display_Name"]))

    selected_display_districts = st.multiselect(
        "District filter (applies across the tool)",
        ["All"] + all_display_districts,
        default=["All"],
        key="global_district_filter",
    )
    use_all_districts = (not selected_display_districts) or ("All" in selected_display_districts)
    active_display_districts = all_display_districts if use_all_districts else selected_display_districts
    active_districts = all_districts if use_all_districts else [display_to_district[name] for name in active_display_districts]
    scope_label = "All Districts" if use_all_districts else active_display_districts[0] if len(active_display_districts) == 1 else f"{len(active_display_districts)} Selected Districts"

    view_df = df[df["District_Name"].isin(active_districts)].copy()
    view_feat_df = feat_df[feat_df["District"].isin(active_districts)].copy()
    view_fc_df = fc_df[fc_df["District"].isin(active_districts)].copy()
    method_df = build_methodology_df(view_df)
    dq_summary, dq_district, dq_monthly = compute_data_quality_tables(view_df)

    if use_all_districts:
        scoped_reg_eval = reg_eval
        scoped_clf_wd = clf_wd
        scoped_clf_xd = clf_xd
    else:
        scoped_reg_eval = run_regression_cv(view_feat_df, feature_cols, scalers)
        scoped_clf_wd = run_classification(view_feat_df, feature_cols, target="WD")
        scoped_clf_xd = run_classification(view_feat_df, feature_cols, target="XD")

    st.success(
        f"{len(view_df):,} records | {view_df['District_Name'].nunique()} districts | "
        f"{view_df['Date'].min().strftime('%b %Y')} to {view_df['Date'].max().strftime('%b %Y')}"
    )
    if not use_all_districts:
        st.caption(f"Current district filter: {', '.join(active_display_districts)}")
    st.info("Streamlit displays the early-warning anomaly scale: Low, Moderate, High, Extreme.")

    tab_explore, tab_drivers, tab_forecast, tab_eval = st.tabs([
        "Exploratory Analysis",
        "Drivers & Context",
        "Forecasts",
        "Model Checks",
    ])

    with tab_explore:
        with st.expander("Run Summary", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{len(view_df):,}")
            c2.metric("Districts", view_df["District_Name"].nunique())
            c3.metric("Months", view_df["Date"].nunique())
            c4.metric("Feature rows", f"{len(view_feat_df):,}")

        with st.expander("Data Quality & Method", expanded=True):
            mtab, dtab, mtab2 = st.tabs(["Methodology", "District Quality", "Monthly Quality"])

            with mtab:
                k1, k2, k3 = st.columns(3)
                k1.metric("Numerator", "Global Acute Malnutrition")
                k2.metric("Anomaly Levels", "4 levels")
                k3.metric("Expected months", dq_summary["expected_months"])
                st.caption("Early-warning labels are anomaly-based and use district-relative percentile thresholds.")

                left, right = st.columns([1.0, 2.0])
                with left:
                    st.markdown("**Anomaly Scale**")
                    st.dataframe(
                        colour_risk_df(build_alert_actions_df().style, ["Anomaly Level"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                with right:
                    st.markdown("**How the Tool Works**")
                    st.dataframe(method_df, use_container_width=True, hide_index=True)
                st.markdown("**Operational Alert Rules**")
                st.dataframe(
                    colour_risk_df(build_operational_alert_rules_df().style, ["Operational Alert"]),
                    use_container_width=True,
                    hide_index=True,
                )

            with dtab:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Median completeness", f"{dq_summary['median_completeness']:.1f}%")
                q2.metric("Zero GAM months", f"{dq_summary['zero_gam_months']:,}")
                q3.metric("Outlier GAM months", f"{dq_summary['outlier_gam_months']:,}")
                q4.metric("Duplicate district-months", f"{dq_summary['duplicate_rows']:,}")
                st.dataframe(rename_for_display(dq_district), use_container_width=True, hide_index=True, height=360)

            with mtab2:
                q5, q6 = st.columns(2)
                q5.metric("Missing severity rows", f"{dq_summary['missing_severity_rows']:,}")
                q6.metric("Latest month completeness", f"{dq_monthly['Reporting_Completeness_Pct'].iloc[-1]:.1f}%" if not dq_monthly.empty else "0.0%")
                st.dataframe(rename_for_display(dq_monthly), use_container_width=True, hide_index=True, height=320)

        latest = view_df.sort_values("Date").groupby("District_Name").last().reset_index()
        with st.expander("Current Situation", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Within-District Anomaly**")
                wd_table = build_phase_count_table(latest["wd_risk"], RISK_LEVELS, "Level")
                render_phase_pie_chart(wd_table, "Level")
            with c2:
                st.markdown("**Between-Districts Anomaly**")
                xd_table = build_phase_count_table(latest["xd_risk"], RISK_LEVELS, "Level")
                render_phase_pie_chart(xd_table, "Level")
            with c3:
                st.markdown("**Operational Alert**")
                op_table = build_phase_count_table(latest["Operational_Alert"], OPERATIONAL_ALERT_LEVELS, "Level")
                render_phase_pie_chart(op_table, "Level")

        with st.expander("Top Districts by Current Operational Alert", expanded=True):
            alert_cols = [
                "District_Name",
                "wd_risk",
                "xd_risk",
                "Operational_Alert",
                "Operational_Alert_Why",
            ]
            alert_latest = latest[alert_cols].copy()
            alert_latest["_alert_rank"] = alert_latest["Operational_Alert"].map(OPERATIONAL_ALERT_ORDER).fillna(-1)
            alert_latest = (
                alert_latest
                .sort_values(["_alert_rank", "District_Name"], ascending=[False, True])
                .drop(columns=["_alert_rank"])
            )
            alert_latest = rename_for_display(alert_latest)
            st.dataframe(
                colour_risk_df(
                    alert_latest.head(15).style,
                    ["Within-District Anomaly", "Between-Districts Anomaly", "Operational Alert"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Observed Anomalies & Operational Alert", expanded=True):
            if not geo_source or geo_name_col is None:
                st.info("Provide your district GeoJSON in the sidebar to show observed maps.")
            else:
                try:
                    if hasattr(geo_source, "seek"):
                        geo_source.seek(0)
                    gdf, matched = load_and_match_geodf(geo_source, view_df, geo_name_col)
                    if hasattr(geo_source, "seek"):
                        geo_source.seek(0)
                    st.caption(f"Matched {matched} GeoJSON districts to CSV districts.")
                    if matched == 0:
                        st.error("No matching district names. Select the correct GeoJSON district name column.")
                    else:
                        latest_period = latest["Date"].max().strftime("%B %Y") if not latest.empty else "Latest Available Month"
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            render_map(
                                gdf,
                                latest[["District_Name", "wd_risk", "Acut_Malnutrition"]],
                                "wd_risk",
                                f"Within-District Anomaly - {latest_period}",
                                ["Acut_Malnutrition"],
                            )
                        with m2:
                            render_map(
                                gdf,
                                latest[["District_Name", "xd_risk", "Acut_Malnutrition"]],
                                "xd_risk",
                                f"Between-Districts Anomaly - {latest_period}",
                                ["Acut_Malnutrition"],
                            )
                        with m3:
                            if "Operational_Alert" in latest.columns:
                                render_map(
                                    gdf,
                                    latest[[c for c in ["District_Name", "Operational_Alert", "Operational_Alert_Why", "Acut_Malnutrition"] if c in latest.columns]],
                                    "Operational_Alert",
                                    f"Operational Alert - {latest_period}",
                                    [c for c in ["Acut_Malnutrition", "Operational_Alert_Why"] if c in latest.columns],
                                )
                            else:
                                st.info("Operational Alert is not available for the current observed dataset.")
                except Exception as e:
                    st.error(f"Observed map error: {e}")
                    st.exception(e)

        with st.expander("Raw Classified Data", expanded=False):
            show_cols = [
                "Region_District", "District_Name", "Region_Label", "time_period", "Date",
                "Acut_Malnutrition", "wd_risk", "xd_risk",
                "Caseload_per_100000_Pop", "Operational_Alert", "Operational_Alert_Why",
            ] + [c for c in FEATURE_COLS_NEW if c in df.columns]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(rename_for_display(view_df[show_cols]), use_container_width=True, height=420, hide_index=True)
            st.download_button("Download classified CSV", view_df[show_cols].to_csv(index=False), "classified_acute_malnutrition.csv")

    with tab_forecast:
        if view_fc_df.empty:
            with st.expander("3-Month District Forecast", expanded=True):
                st.warning("No forecast produced.")
        else:
            with st.expander("3-Month District Forecast", expanded=True):
                st.success(", ".join(view_fc_df["Month_Year"].unique().tolist()))
                st.caption("Forecast maps and tables use the early-warning anomaly scale and Operational Alert.")
                resid_source = scoped_reg_eval.get("residuals", pd.DataFrame()).copy() if "error" not in scoped_reg_eval else pd.DataFrame()

                hindcast_plot = pd.DataFrame()
                hindcast_months = 0
                if use_all_districts or len(active_districts) > 1:
                    if not resid_source.empty:
                        hindcast_plot = (
                            resid_source
                            .groupby("Date")[["Actual", "Predicted"]]
                            .sum()
                            .reset_index()
                            .sort_values("Date")
                        )
                else:
                    selected_forecast_district = active_districts[0]
                    selected_forecast_display = district_to_display.get(selected_forecast_district, selected_forecast_district)
                    if not resid_source.empty:
                        hindcast_plot = (
                            resid_source[resid_source["District"] == selected_forecast_district][["Date", "Actual", "Predicted"]]
                            .sort_values("Date")
                            .reset_index(drop=True)
                        )

                max_hindcast = int(hindcast_plot["Date"].nunique()) if not hindcast_plot.empty else 0
                if max_hindcast > 0:
                    hindcast_months = st.slider(
                        "Hindcast months before forecast",
                        min_value=0,
                        max_value=max_hindcast,
                        value=min(6, max_hindcast),
                        step=1,
                        help="Shows out-of-sample historical predictions before the forward 3-month forecast.",
                    )
                    if hindcast_months > 0:
                        hindcast_plot = hindcast_plot.tail(hindcast_months).copy()
                        st.caption("Hindcast shows model predictions for prior months before the forward forecast.")
                    else:
                        hindcast_plot = hindcast_plot.iloc[0:0].copy()

                if use_all_districts or len(active_districts) > 1:
                    hist_plot = (
                        view_df
                        .groupby("Date")["Acut_Malnutrition"]
                        .sum()
                        .reset_index()
                    )
                    hist_label = "Historical"
                    fc_plot = (
                        view_fc_df
                        .groupby("Date")[["Predicted", "Lower_80", "Upper_80"]]
                        .sum()
                        .reset_index()
                    )
                    chart_title = scope_label
                    yaxis_title = "Total GAM caseload"
                    forecast_marker = None
                else:
                    hist_window = max(18, hindcast_months + 6) if hindcast_months > 0 else 18
                    hist_plot = (
                        view_df[view_df["District_Name"] == selected_forecast_district]
                        .sort_values("Date")
                        .tail(hist_window)[["Date", "Acut_Malnutrition", "wd_risk"]]
                        .reset_index(drop=True)
                    )
                    hist_label = "Historical"
                    fc_plot = (
                        view_fc_df[view_fc_df["District"] == selected_forecast_district]
                        .sort_values("Date")[["Date", "Predicted", "Lower_80", "Upper_80", "Operational_Alert", "WD_Risk"]]
                        .reset_index(drop=True)
                    )
                    chart_title = selected_forecast_display
                    yaxis_title = "Cases"
                    forecast_color_col = "Operational_Alert" if "Operational_Alert" in fc_plot.columns else "WD_Risk" if "WD_Risk" in fc_plot.columns else None
                    forecast_marker = dict(
                        symbol="diamond",
                        size=11,
                        color=[RISK_COLORS.get(r, "#999") for r in fc_plot[forecast_color_col]] if forecast_color_col else "#999",
                    )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_plot["Date"],
                    y=hist_plot["Acut_Malnutrition"],
                    mode="lines+markers",
                    name=hist_label,
                    marker=dict(color=[RISK_COLORS.get(r, "#999") for r in hist_plot["wd_risk"]]) if "wd_risk" in hist_plot.columns else None,
                ))
                if not hindcast_plot.empty:
                    fig.add_trace(go.Scatter(
                        x=hindcast_plot["Date"],
                        y=hindcast_plot["Predicted"],
                        mode="lines+markers",
                        name="Hindcast",
                        line=dict(color="#6c757d", dash="dot", width=2.5),
                        marker=dict(color="#6c757d", size=8),
                        customdata=hindcast_plot["Actual"],
                        hovertemplate="<b>%{x|%b %Y}</b><br>Hindcast: %{y:,.0f}<br>Actual: %{customdata:,.0f}<extra></extra>",
                    ))
                fig.add_trace(go.Scatter(
                    x=list(fc_plot["Date"]) + list(fc_plot["Date"])[::-1],
                    y=list(fc_plot["Upper_80"]) + list(fc_plot["Lower_80"])[::-1],
                    fill="toself",
                    fillcolor="rgba(52,152,219,0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="80% interval",
                ))
                fig.add_trace(go.Scatter(
                    x=fc_plot["Date"],
                    y=fc_plot["Predicted"],
                    mode="lines+markers",
                    name="Forecast",
                    marker=forecast_marker,
                    line=dict(color="#3498db", dash="dash", width=3),
                ))
                fig.update_layout(height=460, title=chart_title, yaxis_title=yaxis_title, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("District Forecast Table", expanded=True):
                display = view_fc_df.copy()
                display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
                st.caption("Operational Alert uses the within-district and between-district anomaly combination rule.")
                risk_cols = [c for c in ["WD_Risk", "XD_Risk", "Operational_Alert"] if c in display.columns]
                display_cols = [
                    "District", "Date", "Predicted", "Lower_80", "Upper_80",
                    "WD_Risk", "XD_Risk", "Operational_Alert", "Operational_Alert_Why",
                ]
                display_cols = [c for c in display_cols if c in display.columns]
                display_view = rename_for_display(display[display_cols])
                display_risk_cols = [display_label(c) for c in risk_cols if c in display_cols]
                st.dataframe(colour_risk_df(display_view.style, display_risk_cols), use_container_width=True, hide_index=True, height=480)
                if use_all_districts:
                    download_name = "acute_malnutrition_forecast.csv"
                elif len(active_districts) == 1:
                    download_name = f"acute_malnutrition_forecast_{active_districts[0]}.csv"
                else:
                    download_name = "acute_malnutrition_forecast_selected_districts.csv"
                st.download_button("Download forecast CSV", display.to_csv(index=False), download_name)

            with st.expander("Forecast Anomaly & Operational Alert Maps", expanded=True):
                if not use_all_districts:
                    st.caption(f"Showing forecast maps for: {', '.join(active_display_districts)}.")
                if not geo_source or geo_name_col is None:
                    st.info("Provide your district GeoJSON in the sidebar to show forecast maps.")
                else:
                    try:
                        if hasattr(geo_source, "seek"):
                            geo_source.seek(0)
                        gdf, matched = load_and_match_geodf(geo_source, view_df, geo_name_col)
                        if hasattr(geo_source, "seek"):
                            geo_source.seek(0)
                        st.caption(f"Matched {matched} GeoJSON districts to CSV districts.")
                        if matched == 0:
                            st.error("No matching district names. Select the correct GeoJSON district name column.")
                        else:
                            fc_map = view_fc_df.rename(columns={"District": "District_Name"}).copy()
                            if fc_map.empty:
                                st.info("No forecast map data available for the selected district.")
                            else:
                                months = (
                                    fc_map.sort_values("Date")["Month_Year"]
                                    .drop_duplicates()
                                    .tolist()[:3]
                                )
                                map_cols = [
                                    ("WD_Risk", "Within-District Anomaly"),
                                    ("XD_Risk", "Between-Districts Anomaly"),
                                    ("Operational_Alert", "Operational Alert"),
                                ]
                                available_cols = [(risk_col, col_label) for risk_col, col_label in map_cols if risk_col in fc_map.columns]
                                if not available_cols:
                                    st.info("No forecast anomaly or operational alert columns available for mapping.")
                                else:
                                    for i, month in enumerate(months, start=1):
                                        st.markdown(f"**Horizon {i}: {month}**")
                                        row_cols = st.columns(len(available_cols))
                                        md = fc_map[fc_map["Month_Year"] == month]
                                        for col, (risk_col, col_label) in zip(row_cols, available_cols):
                                            hover_cols = ["Predicted", "Lower_80", "Upper_80"]
                                            if risk_col == "Operational_Alert":
                                                hover_cols = [c for c in ["Predicted", "Lower_80", "Upper_80", "Operational_Alert_Why"] if c in md.columns]
                                            with col:
                                                render_map(
                                                    gdf,
                                                    md[[c for c in ["District_Name", risk_col, "Predicted", "Lower_80", "Upper_80", "Severity_Prevalence_Pct", "Predicted_Caseload_per_100000_Pop", "Operational_Alert_Why"] if c in md.columns]],
                                                    risk_col,
                                                    f"{col_label} - {month}",
                                                    hover_cols,
                                                )
                    except Exception as e:
                        st.error(f"Forecast map error: {e}")
                        st.exception(e)

    with tab_drivers:
        with st.expander("Spearman Correlation Portrait", expanded=True):
            st.caption("Rank-based correlation between GAM caseload (SAM + MAM) and covariates. Blue means positive association; red means negative association.")
            corr_slice = view_df
            if len(active_districts) == 1:
                st.caption(f"{len(corr_slice)} monthly records for {active_display_districts[0]}.")

            corr_df = compute_spearman_correlations(corr_slice)
            fig_corr = draw_spearman_radial(corr_df)
            if fig_corr is None:
                st.info("Not enough numeric variation for Spearman correlation in the selected scope.")
            else:
                st.pyplot(fig_corr, use_container_width=True)
                plt.close(fig_corr)
                show_corr = corr_df[["Display", "Spearman rho", "N"]].rename(columns={"Display": "Variable"})
                st.dataframe(show_corr.round(3), use_container_width=True, hide_index=True)

        with st.expander("Compare Districts or Regions Against National Reference", expanded=True):
            render_compare_time_series(view_df, scope_label)

        with st.expander(f"Historical Anomaly Trend - {scope_label}", expanded=True):
            render_historical_risk_trend(view_df, scope_label)

        with st.expander("Historical Trend", expanded=True):
            nat = view_df.groupby("Date")["Acut_Malnutrition"].sum().reset_index()
            fig = px.line(nat, x="Date", y="Acut_Malnutrition", markers=True, title=f"Total GAM Caseload (SAM + MAM) - {scope_label}")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    with tab_eval:
        with st.expander("Regression: Case Count Forecast", expanded=True):
            if "error" in scoped_reg_eval:
                st.warning(scoped_reg_eval["error"])
            else:
                mean_error = scoped_reg_eval.get("cv_mean_error", np.nan)
                p90_ae = scoped_reg_eval.get("cv_p90_ae", np.nan)
                target_transform = scoped_reg_eval.get("target_transform", "raw caseload scaling")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("MAE", f"{scoped_reg_eval['cv_mae']:.1f}", delta=f"+/- {scoped_reg_eval['cv_mae_std']:.1f}", delta_color="off")
                r2.metric("RMSE", f"{scoped_reg_eval['cv_rmse']:.1f}")
                r3.metric("Mean Error", f"{mean_error:.1f}" if pd.notna(mean_error) else "N/A")
                r4.metric("P90 Abs Error", f"{p90_ae:.1f}" if pd.notna(p90_ae) else "N/A")
                r5.metric("CV folds", scoped_reg_eval["n_folds"])
                st.caption(
                    f"Regression target transform: {target_transform}. "
                    "Mean error above 0 means the model tends to overpredict."
                )
                if not use_all_districts:
                    st.caption(f"Model checks are filtered to: {', '.join(active_display_districts)}")

                resid = scoped_reg_eval["residuals"].copy()
                if not use_all_districts and not resid.empty:
                    resid = resid[resid["District"].isin(active_districts)].copy()
                if not resid.empty:
                    tabs = st.tabs(["Predicted vs Actual", "Residuals", "Largest Errors", "Feature Importance"])

                    with tabs[0]:
                        fig = px.scatter(
                            resid,
                            x="Actual",
                            y="Predicted",
                            color="XD_Risk",
                            color_discrete_map=RISK_COLORS,
                            category_orders={"XD_Risk": RISK_LEVELS},
                            hover_data=["District", "Date", "Absolute_Error"],
                            title=f"Predicted vs Actual - {scope_label}",
                        )
                        max_val = max(resid["Actual"].max(), resid["Predicted"].max()) * 1.05
                        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)

                    with tabs[1]:
                        fig = px.scatter(
                            resid,
                            x="Actual",
                            y="Signed_Error",
                            color="XD_Risk",
                            color_discrete_map=RISK_COLORS,
                            category_orders={"XD_Risk": RISK_LEVELS},
                            hover_data=["District", "Date", "Predicted", "Absolute_Error"],
                            title=f"Signed Error vs Actual - {scope_label}",
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(height=450, yaxis_title="Predicted - Actual")
                        st.plotly_chart(fig, use_container_width=True)

                    with tabs[2]:
                        largest_errors = (
                            resid.sort_values("Absolute_Error", ascending=False)
                            .head(15)
                            .copy()
                        )
                        largest_errors["Date"] = pd.to_datetime(largest_errors["Date"]).dt.strftime("%Y-%m")
                        st.dataframe(
                            largest_errors[[
                                "District", "Date", "Actual", "Predicted",
                                "Signed_Error", "Absolute_Error", "XD_Risk",
                            ]].round(2),
                            use_container_width=True,
                            hide_index=True,
                        )

                    with tabs[3]:
                        fig_fi = px.bar(
                            scoped_reg_eval["feat_imp"].head(15),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title=f"Top Features - Regression ({scope_label})",
                        )
                        fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=430)
                        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("Classification: Anomaly Sensitivity", expanded=True):
            tabs = st.tabs(["Within-District Anomaly", "Between-Districts Anomaly"])
            for tab, clf, label in [(tabs[0], scoped_clf_wd, "Within-District Anomaly"), (tabs[1], scoped_clf_xd, "Between-Districts Anomaly")]:
                with tab:
                    if "error" in clf:
                        st.warning(clf["error"])
                        continue
                    if not use_all_districts:
                        st.caption(f"Metrics filtered to: {', '.join(active_display_districts)}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{clf['cv_acc'] * 100:.1f}%")
                    c2.metric("Balanced Accuracy", f"{clf['cv_bal_acc'] * 100:.1f}%")
                    c3.metric("Macro Recall", f"{clf['cv_macro_recall'] * 100:.1f}%")
                    c4.metric("Weighted F1", f"{clf['cv_f1']:.3f}")

                    pc = pd.DataFrame(clf["per_class"])
                    show = pc.copy()
                    for col in ["Sensitivity", "Specificity", "Precision (PPV)", "NPV", "F1"]:
                        show[col] = show[col].apply(lambda v: f"{v * 100:.1f}%")
                    st.dataframe(show, use_container_width=True, hide_index=True)

                    st.download_button(
                        f"Download {label} metrics CSV",
                        pc.to_csv(index=False),
                        f"{label.lower().replace('-', '_')}_metrics.csv",
                    )

                    fig_fi = px.bar(
                        clf["feat_imp"].head(15),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Top Features - {label}",
                    )
                    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=430)
                    st.plotly_chart(fig_fi, use_container_width=True)


if __name__ == "__main__":
    main()
