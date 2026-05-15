import calendar
import hashlib
import json
import os
import warnings
from io import StringIO

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import geopandas as gpd
import matplotlib.patches as mpatches
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

DEFAULT_CSV_PATH = "/Users/sarahawino/Downloads/Updated_with_region_district.csv"

RISK_LEVELS = ["Normal", "Alert", "Alarm", "Emergency"]
RISK_ORDER = {"Normal": 0, "Alert": 1, "Alarm": 2, "Emergency": 3}
RISK_COLORS = {
    "Normal": "#2ecc71",
    "Alert": "#f1c40f",
    "Alarm": "#e67e22",
    "Emergency": "#e74c3c",
    "No Data": "#bdbdbd",
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
    "Acut_Malnutrition": "Acute Malnutrition",
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
    "wd_p75",
    "wd_p90",
    "wd_p95",
    "xd_p75",
    "xd_p90",
    "xd_p95",
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


def classify_risk(value: float, p75: float, p90: float, p95: float) -> str:
    if pd.isna(value):
        return "No Data"
    if value <= p75:
        return "Normal"
    if value <= p90:
        return "Alert"
    if value <= p95:
        return "Alarm"
    return "Emergency"


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def apply_log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


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
        "Emergency": "#f8d7da",
        "Alarm": "#fde8d0",
        "Alert": "#fff3cd",
        "Normal": "#d4edda",
        "No Data": "#f0f0f0",
    }
    risk_fg = {
        "Emergency": "#721c24",
        "Alarm": "#8a4500",
        "Alert": "#856404",
        "Normal": "#155724",
        "No Data": "#555",
    }

    def _style(val):
        if val in risk_bg:
            return f"background-color:{risk_bg[val]};color:{risk_fg[val]};font-weight:bold"
        return ""

    return df_styler.map(_style, subset=columns)


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


def draw_spearman_radial(corr_df: pd.DataFrame, title: str):
    if corr_df.empty:
        return None

    plot_df = corr_df.head(12).copy()
    n = len(plot_df)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vals = plot_df["Abs rho"].to_numpy()
    signed = plot_df["Spearman rho"].to_numpy()
    labels = plot_df["Display"].tolist()

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": "polar"},
        facecolor="#f7f4ef",
    )
    ax.set_facecolor("#f7f4ef")

    max_r = max(0.42, float(vals.max()) + 0.10)
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
            fontsize=8,
            color="#555",
            fontfamily="monospace",
            alpha=0.80,
        )

    pos_color = "#08aeca"
    neg_color = "#ef4056"
    width = 2 * np.pi / n * 0.58

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
            rho_abs + 0.082,
            f"{rho:+.3f}",
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
            fontfamily="monospace",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground="#f7f4ef")],
        )
        short = label if len(label) <= 15 else label.replace(" ", "\n")
        ax.text(
            angle,
            max(rho_abs + 0.16, 0.22),
            short,
            ha="center",
            va="center",
            fontsize=9,
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

    ax.set_ylim(0, max_r + 0.23)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(color=pos_color, label="Positive Spearman rho"),
            mpatches.Patch(color=neg_color, label="Negative Spearman rho"),
            mpatches.Patch(color="#f5c518", label="Acute malnutrition target"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        framealpha=0.70,
        fontsize=9,
        ncol=3,
    )
    fig.text(
        0.5,
        0.965,
        title,
        ha="center",
        fontsize=15,
        fontweight="bold",
        color="#1a1a2e",
        fontfamily="monospace",
        path_effects=[pe.withStroke(linewidth=4, foreground="#f7f4ef")],
    )
    plt.tight_layout()
    return fig


def render_compare_time_series(df: pd.DataFrame):
    st.subheader("Compare Districts or Regions Against National Reference")
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
        name="All Districts - Total (right axis)",
        yaxis="y2",
        line=dict(color="#34495e", width=2.2, dash="dash"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Total: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=national["Date"],
        y=national["Mean"],
        mode="lines",
        name="All Districts - Mean (left axis)",
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
        yaxis2=dict(title="Total - All Districts", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_historical_risk_trend(df: pd.DataFrame):
    st.subheader("Historical Risk Trend - All Districts")
    risk_source = st.radio(
        "Risk definition",
        ["Within-District Risk", "Between-Districts Risk"],
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
    monthly["_n1"] = monthly["Normal"]
    monthly["_n2"] = monthly["_n1"] + monthly["Alert"]
    monthly["_n3"] = monthly["_n2"] + monthly["Alarm"]
    monthly["_n4"] = monthly["_n3"] + monthly["Emergency"]
    monthly["Emergency_6m"] = monthly["Emergency_pct"].rolling(6, min_periods=1, center=True).mean()
    total_max = float(monthly["_n4"].max())
    monthly["Emergency_6m_scaled"] = monthly["Emergency_6m"] / 100 * total_max

    fig = go.Figure()
    for lvl, bottom, top in [
        ("Normal", "_n0", "_n1"),
        ("Alert", "_n1", "_n2"),
        ("Alarm", "_n2", "_n3"),
        ("Emergency", "_n3", "_n4"),
    ]:
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
        y=monthly["Emergency_6m_scaled"],
        mode="lines",
        name="Emergency % (6m avg)",
        line=dict(color="#c0392b", width=2.4, dash="dash"),
        customdata=monthly["Emergency_6m"].fillna(0),
        hovertemplate="<b>%{x|%b %Y}</b><br>Emergency trend: %{customdata:.1f}%<extra></extra>",
    ))

    if not monthly.empty:
        peak_idx = monthly["Emergency_pct"].idxmax()
        peak_date = monthly.loc[peak_idx, "Date"]
        peak_pct = monthly.loc[peak_idx, "Emergency_pct"]
        fig.add_vline(x=peak_date, line_dash="dot", line_color="#e74c3c", line_width=1.8, opacity=0.75)
        fig.add_annotation(
            x=peak_date,
            y=total_max * 1.04,
            text=f"Peak: {peak_pct:.0f}% Emergency",
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

    latest_month = monthly["Date"].iloc[-1].strftime("%b %Y")
    avg_e = monthly["Emergency_pct"].mean()
    peak_e = monthly["Emergency_pct"].max()
    recent = monthly["Emergency_pct"].tail(3).mean()
    prior = monthly["Emergency_pct"].iloc[-9:-3].mean() if len(monthly) >= 9 else monthly["Emergency_pct"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest month", latest_month)
    k2.metric("Avg % Emergency", f"{avg_e:.1f}%")
    k3.metric("Peak Emergency", f"{peak_e:.1f}%")
    k4.metric(
        "Recent 3m trend",
        "Rising" if recent > prior else "Falling",
        delta=f"{recent - prior:+.1f}pp",
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

    df[["wd_p75", "wd_p90", "wd_p95"]] = np.nan
    df["wd_risk"] = "No Data"

    for district in df["District_Name"].unique():
        mask = df["District_Name"] == district
        d = df[mask].sort_values("Date")
        for idx in d.index:
            past = d.loc[d["Date"] < df.loc[idx, "Date"], "Acut_Malnutrition"]
            if len(past) >= 2:
                p75, p90, p95 = np.percentile(past, [75, 90, 95])
            else:
                v = df.loc[idx, "Acut_Malnutrition"]
                p75 = p90 = p95 = v
            df.loc[idx, ["wd_p75", "wd_p90", "wd_p95"]] = p75, p90, p95
            df.loc[idx, "wd_risk"] = classify_risk(df.loc[idx, "Acut_Malnutrition"], p75, p90, p95)

    df[["xd_p75", "xd_p90", "xd_p95"]] = np.nan
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

        p75, p90, p95 = np.percentile(hist, [75, 90, 95])
        for idx in curr_idx:
            val = df.loc[idx, "Acut_Malnutrition"]
            df.loc[idx, ["xd_p75", "xd_p90", "xd_p95"]] = p75, p90, p95
            df.loc[idx, "xd_risk"] = classify_risk(val, p75, p90, p95)

    df["wd_score"] = df["wd_risk"].map(RISK_ORDER).fillna(0).astype(int)
    df["xd_score"] = df["xd_risk"].map(RISK_ORDER).fillna(0).astype(int)
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
# Hist_P90 is retained because it supports the Alarm decision boundary.

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
    feat_df["Target_Adm_Scaled"] = np.nan
    for district in feat_df["District"].unique():
        mask = feat_df["District"] == district
        baseline = feat_df.loc[mask, "Lag1"].astype(float)
        mu = float(baseline.mean())
        sigma = float(baseline.std())
        if sigma < 1e-6:
            sigma = 1.0
        scalers[district] = {"mean": mu, "std": sigma}

        for col in scale_cols:
            if col in feat_df.columns:
                feat_df.loc[mask, col] = (feat_df.loc[mask, col] - mu) / sigma
        feat_df.loc[mask, "Target_Adm_Scaled"] = (feat_df.loc[mask, "Target_Adm"] - mu) / sigma

    for col in feat_df.columns:
        if col not in {"District", "Date", "Target_WD_Risk", "Target_XD_Risk"}:
            if pd.api.types.is_integer_dtype(feat_df[col]):
                feat_df[col] = feat_df[col].astype("float64")

    targets = {
        "District",
        "Date",
        "Target_Adm",
        "Target_Adm_Scaled",
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
        class_weight = {0: 0.7, 1: 3.0, 2: 5.0, 3: 6.0}
        max_depth = 16
        min_leaf = 1
    else:
        class_weight = {0: 1.0, 1: 1.8, 2: 2.8, 3: 3.8}
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
            np.linspace(0.65, 0.95, 5),
            np.linspace(0.12, 0.42, 6),
            np.linspace(0.06, 0.30, 6),
            np.linspace(0.05, 0.28, 6),
        ]
    else:
        grids = [
            np.linspace(0.40, 0.70, 4),
            np.linspace(0.25, 0.55, 5),
            np.linspace(0.18, 0.50, 6),
            np.linspace(0.12, 0.45, 7),
        ]

    best = np.array([0.78, 0.28, 0.18, 0.16]) if target == "WD" else np.array([0.50, 0.45, 0.40, 0.34])
    best_score = -np.inf

    for t0 in grids[0]:
        for t1 in grids[1]:
            for t2 in grids[2]:
                for t3 in grids[3]:
                    thr = np.array([t0, t1, t2, t3])
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
                            0.10 * recalls[0]
                            + 0.26 * recalls[1]
                            + 0.32 * recalls[2]
                            + 0.32 * recalls[3]
                            + 0.04 * precision_high
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
    valid = _feat_df.dropna(subset=["Target_Adm_Scaled"]).copy().reset_index(drop=True)
    X = valid[feature_cols].values
    y = valid["Target_Adm_Scaled"].values
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
        pred_scaled = model.predict(X[te_p])
        meta = valid.iloc[te_p]
        actual_raw = meta["Target_Adm"].values
        pred_raw = np.array([
            pred_scaled[j] * _scalers[meta.iloc[j]["District"]]["std"]
            + _scalers[meta.iloc[j]["District"]]["mean"]
            for j in range(len(te_p))
        ])

        mae_cv.append(mean_absolute_error(actual_raw, pred_raw))
        rmse_cv.append(np.sqrt(mean_squared_error(actual_raw, pred_raw)))

        for j in range(len(te_p)):
            resid_rows.append({
                "District": meta.iloc[j]["District"],
                "Date": meta.iloc[j]["Date"],
                "XD_Risk": meta.iloc[j]["Target_XD_Risk"],
                "Actual": actual_raw[j],
                "Predicted": max(0, pred_raw[j]),
                "Error": actual_raw[j] - pred_raw[j],
            })

    final_model = make_regressor()
    final_model.fit(X, y)
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "cv_mae": float(np.mean(mae_cv)),
        "cv_mae_std": float(np.std(mae_cv)),
        "cv_rmse": float(np.mean(rmse_cv)),
        "feat_imp": feat_imp,
        "residuals": pd.DataFrame(resid_rows),
        "n_folds": len(splits),
        "model": final_model,
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

        valid = feat_df.dropna(subset=["Target_Adm_Scaled"]).copy()
        self.extra_vals = {}
        for d in valid["District"].unique():
            last = valid[valid["District"] == d].iloc[-1]
            self.extra_vals[d] = {c: float(last.get(c, 0.0)) for c in self.extra}

        self.reg = make_regressor()
        self.reg.fit(valid[feature_cols].values, valid["Target_Adm_Scaled"].values)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for district in df["District_Name"].unique():
            d = df[df["District_Name"] == district].sort_values("Date")
            if len(d) < 3:
                continue

            scaler = self.scalers.get(district)
            if scaler is None:
                continue
            mu = scaler["mean"]
            sg = max(scaler["std"], 1e-6)

            hist_adm = list(d["Acut_Malnutrition"].astype(float).values)
            hist_wd = list(d["wd_score"].values)
            hist_xd = list(d["xd_score"].values)
            last_date = d["Date"].iloc[-1]
            extras = self.extra_vals.get(district, {c: 0.0 for c in self.extra})

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
                pred_s = float(self.reg.predict(fv)[0])
                pred_raw = max(0.0, round(pred_s * sg + mu))

                tree_preds = np.array([t.predict(fv)[0] for t in self.reg.estimators_])
                pi_half = 1.28 * tree_preds.std() * sg
                lo = max(0, round(pred_raw - pi_half))
                hi = max(0, round(pred_raw + pi_half))

                wd_risk = classify_risk(pred_raw, p75, p90, p95)
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
                xp75, xp90, xp95 = np.percentile(preds, [75, 90, 95])
            else:
                xp75, xp90, xp95 = np.percentile(all_hist_vals, [75, 90, 95])
            for idx in result[mask].index:
                pred = result.loc[idx, "Predicted"]
                xd_risk = classify_risk(pred, xp75, xp90, xp95)
                wd_risk = result.loc[idx, "WD_Risk"]
                result.loc[idx, "XD_Risk"] = xd_risk
                result.loc[idx, "Composite_Risk"] = max([wd_risk, xd_risk], key=lambda r: RISK_ORDER.get(r, 0))

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
        category_orders={risk_col: RISK_LEVELS},
        zoom=5.1,
        center={"lat": 1.37, "lon": 32.29},
        mapbox_style="carto-positron",
        hover_name="District_Name",
        hover_data=hover_data,
        title=title,
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_delta_map(gdf, fc_map, risk_col, months):
    if len(months) < 2:
        return

    first, last = months[0], months[-1]
    first_df = fc_map[fc_map["Month_Year"] == first][["District_Name", risk_col]].rename(columns={risk_col: "Risk_First"})
    last_df = fc_map[fc_map["Month_Year"] == last][["District_Name", risk_col]].rename(columns={risk_col: "Risk_Last"})
    delta = first_df.merge(last_df, on="District_Name", how="inner")
    delta["Score_First"] = delta["Risk_First"].map(RISK_ORDER).fillna(0).astype(int)
    delta["Score_Last"] = delta["Risk_Last"].map(RISK_ORDER).fillna(0).astype(int)
    delta["Delta"] = delta["Score_Last"] - delta["Score_First"]
    delta["Delta_Label"] = delta["Delta"].apply(lambda x: f"+{x}" if x > 0 else str(x))
    delta["Direction"] = delta["Delta"].apply(lambda x: "Worsening" if x > 0 else ("Improving" if x < 0 else "No change"))

    merged = gdf.merge(delta, on="District_Name", how="left")
    merged["Delta_Label"] = merged["Delta_Label"].fillna("0")
    color_map = {
        "-3": "#08519c",
        "-2": "#3182bd",
        "-1": "#9ecae1",
        "0": "#e0e0e0",
        "+1": "#fdae6b",
        "+2": "#e6550d",
        "+3": "#a63603",
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
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=500)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# APP
# =============================================================================

def main():
    st.set_page_config(page_title="ACUTE MALNUTRITION FORECASTING TOOL (AMFT)", layout="wide")
    st.title("ACUTE MALNUTRITION FORECASTING TOOL (AMFT)")
    st.caption("District-based risk mapping, Random Forest regression forecast, and recall-focused classification")

    with st.sidebar:
        st.header("Data")
        csv_mode = st.radio("CSV source", ["Use local CSV path", "Upload CSV"], horizontal=False)
        csv_path = st.text_input("CSV path", value=DEFAULT_CSV_PATH, disabled=(csv_mode != "Use local CSV path"))
        csv_file = None
        if csv_mode == "Upload CSV":
            csv_file = st.file_uploader("Upload CSV", type="csv")

        st.header("GeoJSON")
        geo_file = st.file_uploader("Upload district GeoJSON", type=["json", "geojson"])
        geo_name_col = None
        if geo_file is not None:
            try:
                preview = gpd.read_file(geo_file)
                geo_file.seek(0)
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
        st.caption("Within-district compares each district with its own history. Between-districts compares districts with peers.")

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

    st.success(
        f"{len(df):,} records | {df['District_Name'].nunique()} districts | "
        f"{df['Date'].min().strftime('%b %Y')} to {df['Date'].max().strftime('%b %Y')}"
    )

    with st.spinner("Engineering district-relative features..."):
        feat_df, feature_cols, scalers = build_features(df)

    if st.session_state.get("_data_key") != data_key:
        with st.spinner("Training Random Forest models and generating 3-month forecasts..."):
            st.session_state["reg_eval"] = run_regression_cv(feat_df, feature_cols, scalers)
            st.session_state["clf_wd"] = run_classification(feat_df, feature_cols, target="WD")
            st.session_state["clf_xd"] = run_classification(feat_df, feature_cols, target="XD")
            st.session_state["fc_df"] = run_forecast(feat_df, feature_cols, scalers, df, data_key)
            st.session_state["_data_key"] = data_key

    reg_eval = st.session_state["reg_eval"]
    clf_wd = st.session_state["clf_wd"]
    clf_xd = st.session_state["clf_xd"]
    fc_df = st.session_state["fc_df"]

    tab_data, tab_forecast, tab_maps, tab_eval = st.tabs([
        "Data",
        "Forecasts",
        "District Maps",
        "Model Evaluation",
    ])

    with tab_data:
        st.header("Data Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Districts", df["District_Name"].nunique())
        c3.metric("Months", df["Date"].nunique())
        c4.metric("Feature rows", f"{len(feat_df):,}")

        st.subheader("Current Risk Counts")
        latest = df.sort_values("Date").groupby("District_Name").last().reset_index()
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Within-District Risk**")
            st.dataframe(latest["wd_risk"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int), use_container_width=True)
        with cc2:
            st.markdown("**Between-Districts Risk**")
            st.dataframe(latest["xd_risk"].value_counts().reindex(RISK_LEVELS).fillna(0).astype(int), use_container_width=True)

        st.subheader("Spearman Correlation Portrait")
        st.caption("Rank-based correlation between acute malnutrition and covariates. Blue means positive association; red means negative association.")
        corr_all_tab, corr_dist_tab = st.tabs(["All Districts", "Single District"])

        with corr_all_tab:
            corr_df = compute_spearman_correlations(df)
            fig_corr = draw_spearman_radial(corr_df, "Spearman Correlation Portrait - All Districts")
            if fig_corr is None:
                st.info("Not enough numeric covariate data for Spearman correlation.")
            else:
                st.pyplot(fig_corr, use_container_width=True)
                plt.close(fig_corr)
                show_corr = corr_df[["Display", "Spearman rho", "N"]].rename(columns={"Display": "Variable"})
                st.dataframe(show_corr.round(3), use_container_width=True, hide_index=True)

        with corr_dist_tab:
            district_options = sorted(df["Display_Name"].unique())
            selected_display = st.selectbox("Select district for Spearman correlation", district_options, key="spearman_district")
            district_slice = df[df["Display_Name"] == selected_display]
            st.caption(f"{len(district_slice)} monthly records for {selected_display}.")
            corr_dist = compute_spearman_correlations(district_slice)
            fig_dist = draw_spearman_radial(corr_dist, f"Spearman Correlation Portrait - {selected_display}")
            if fig_dist is None:
                st.info("Not enough variation in this district for Spearman correlation.")
            else:
                st.pyplot(fig_dist, use_container_width=True)
                plt.close(fig_dist)
                show_dist = corr_dist[["Display", "Spearman rho", "N"]].rename(columns={"Display": "Variable"})
                st.dataframe(show_dist.round(3), use_container_width=True, hide_index=True)

        st.markdown("---")
        render_compare_time_series(df)

        st.markdown("---")
        render_historical_risk_trend(df)

        st.markdown("---")
        st.subheader("Raw Classified Data")
        show_cols = [
            "Region_District", "District_Name", "Region_Label", "time_period", "Date",
            "Acut_Malnutrition", "wd_risk", "xd_risk",
        ] + [c for c in FEATURE_COLS_NEW if c in df.columns]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, height=420)
        st.download_button("Download classified CSV", df[show_cols].to_csv(index=False), "classified_acute_malnutrition.csv")

        st.markdown("---")
        st.subheader("Historical Trend")
        nat = df.groupby("Date")["Acut_Malnutrition"].sum().reset_index()
        fig = px.line(nat, x="Date", y="Acut_Malnutrition", markers=True, title="Total Acute Malnutrition Cases")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab_forecast:
        st.header("3-Month District Forecast")
        if fc_df.empty:
            st.warning("No forecast produced.")
        else:
            st.success(", ".join(fc_df["Month_Year"].unique().tolist()))

            nat_hist = df.groupby("Date")["Acut_Malnutrition"].sum().reset_index()
            nat_fc = fc_df.groupby("Date")[["Predicted", "Lower_80", "Upper_80"]].sum().reset_index()
            nat_fc["Month_Year"] = nat_fc["Date"].dt.strftime("%B %Y")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nat_hist.tail(18)["Date"],
                y=nat_hist.tail(18)["Acut_Malnutrition"],
                mode="lines+markers",
                name="Historical",
            ))
            fig.add_trace(go.Scatter(
                x=list(nat_fc["Date"]) + list(nat_fc["Date"])[::-1],
                y=list(nat_fc["Upper_80"]) + list(nat_fc["Lower_80"])[::-1],
                fill="toself",
                fillcolor="rgba(52,152,219,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="80% interval",
            ))
            fig.add_trace(go.Scatter(
                x=nat_fc["Date"],
                y=nat_fc["Predicted"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#3498db", dash="dash", width=3),
            ))
            fig.update_layout(height=460, yaxis_title="Total cases", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("District Forecast Table")
            display = fc_df.copy()
            display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
            risk_cols = ["WD_Risk", "XD_Risk", "Composite_Risk"]
            st.dataframe(colour_risk_df(display.style, risk_cols), use_container_width=True, hide_index=True, height=480)
            st.download_button("Download forecast CSV", fc_df.to_csv(index=False), "acute_malnutrition_forecast.csv")

            st.subheader("Single District")
            districts = sorted(df["District_Name"].unique())
            sel = st.selectbox("District", districts)
            hist = df[df["District_Name"] == sel].sort_values("Date").tail(18)
            fcd = fc_df[fc_df["District"] == sel].sort_values("Date")
            if not fcd.empty:
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=hist["Date"],
                    y=hist["Acut_Malnutrition"],
                    mode="lines+markers",
                    name="Historical",
                    marker=dict(color=[RISK_COLORS.get(r, "#999") for r in hist["wd_risk"]]),
                ))
                fig_d.add_trace(go.Scatter(
                    x=fcd["Date"],
                    y=fcd["Predicted"],
                    mode="lines+markers",
                    name="Forecast",
                    marker=dict(symbol="diamond", size=11, color=[RISK_COLORS.get(r, "#999") for r in fcd["Composite_Risk"]]),
                    line=dict(color="#3498db", dash="dash", width=3),
                ))
                fig_d.update_layout(title=sel, height=440, yaxis_title="Cases")
                st.plotly_chart(fig_d, use_container_width=True)

    with tab_maps:
        st.header("District-Based Maps")
        if geo_file is None or geo_name_col is None:
            st.info("Upload your district GeoJSON in the sidebar to show maps.")
        else:
            try:
                geo_file.seek(0)
                gdf, matched = load_and_match_geodf(geo_file, df, geo_name_col)
                geo_file.seek(0)
                st.caption(f"Matched {matched} GeoJSON districts to CSV districts.")
                if matched == 0:
                    st.error("No matching district names. Select the correct GeoJSON district name column.")
                else:
                    latest = (
                        df.sort_values("Date")
                        .groupby("District_Name")
                        .last()
                        .reset_index()
                    )
                    st.subheader("Current Observed Risk")
                    m1, m2 = st.columns(2)
                    with m1:
                        render_map(
                            gdf,
                            latest[["District_Name", "wd_risk", "Acut_Malnutrition"]],
                            "wd_risk",
                            "Current Within-District Risk",
                            ["Acut_Malnutrition"],
                        )
                    with m2:
                        render_map(
                            gdf,
                            latest[["District_Name", "xd_risk", "Acut_Malnutrition"]],
                            "xd_risk",
                            "Current Between-Districts Risk",
                            ["Acut_Malnutrition"],
                        )

                    if not fc_df.empty:
                        st.subheader("Forecast Risk Maps")
                        fc_map = fc_df.rename(columns={"District": "District_Name"}).copy()
                        months = fc_map["Month_Year"].unique().tolist()
                        risk_choice = st.radio(
                            "Map risk type",
                            ["WD_Risk", "XD_Risk", "Composite_Risk"],
                            horizontal=True,
                            format_func=lambda x: {
                                "WD_Risk": "Within-District",
                                "XD_Risk": "Between-Districts",
                                "Composite_Risk": "Composite",
                            }[x],
                        )
                        selected_month = st.selectbox("Forecast month", months)
                        md = fc_map[fc_map["Month_Year"] == selected_month]
                        render_map(
                            gdf,
                            md[["District_Name", risk_choice, "Predicted", "Lower_80", "Upper_80"]],
                            risk_choice,
                            f"{risk_choice.replace('_', ' ')} - {selected_month}",
                            ["Predicted", "Lower_80", "Upper_80"],
                        )
                        st.subheader("3-Month Risk Change")
                        render_delta_map(gdf, fc_map, risk_choice, months)
            except Exception as e:
                st.error(f"Map error: {e}")
                st.exception(e)

    with tab_eval:
        st.header("Model Evaluation")

        st.subheader("Regression: Case Count Forecast")
        if "error" in reg_eval:
            st.warning(reg_eval["error"])
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("MAE", f"{reg_eval['cv_mae']:.1f}", delta=f"+/- {reg_eval['cv_mae_std']:.1f}", delta_color="off")
            r2.metric("RMSE", f"{reg_eval['cv_rmse']:.1f}")
            r3.metric("CV folds", reg_eval["n_folds"])

            resid = reg_eval["residuals"]
            if not resid.empty:
                fig = px.scatter(
                    resid,
                    x="Actual",
                    y="Predicted",
                    color="XD_Risk",
                    color_discrete_map=RISK_COLORS,
                    category_orders={"XD_Risk": RISK_LEVELS},
                    hover_data=["District", "Date"],
                    title="Predicted vs Actual",
                )
                max_val = max(resid["Actual"].max(), resid["Predicted"].max()) * 1.05
                fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Classification: Risk Sensitivity")
        tabs = st.tabs(["Within-District", "Between-Districts"])
        for tab, clf, label in [(tabs[0], clf_wd, "Within-District"), (tabs[1], clf_xd, "Between-Districts")]:
            with tab:
                if "error" in clf:
                    st.warning(clf["error"])
                    continue
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
