# Acute Malnutrition Forecasting Tool (AMFT)

AMFT is a Streamlit-based decision-support tool for district-level acute malnutrition monitoring in Uganda. It combines anomaly detection, short-term forecasting, geospatial visualization, data quality checks, and model evaluation to support early warning and operational planning.

The current version focuses on three connected outputs:

- 4-level anomaly classification for within-district and between-district monitoring
- 3-level operational alerts derived from anomaly combinations
- 3-month district forecasts of GAM caseloads with uncertainty bounds


## Core Capabilities

- Load district monthly GAM caseload data from a local path or file upload
- Load district boundaries from a local GeoJSON path or file upload
- Classify observed GAM caseloads using district-relative and peer-relative anomaly rules
- Generate operational alerts from within-district and between-district anomaly combinations
- Forecast district GAM caseloads for the next 3 months
- Display forecast uncertainty with 80% intervals
- Compare districts or regions against a national reference trend
- Explore Spearman correlations between GAM caseload and covariates
- Review regression and classification model performance
- Download classified observed data and forecast outputs

## Risk Framework

### 1. Anomaly Classification

Observed and forecast anomaly labels use percentile thresholds:

| Level | Threshold | Meaning |
|---|---|---|
| `Low` | Below 75th percentile | Within the usual range |
| `Moderate` | 75th to below 90th percentile | Higher than usual |
| `High` | 90th to below 95th percentile | Unusually high |
| `Extreme` | 95th percentile and above | Exceptionally high |

### 2. Operational Alert

Operational alerts are derived from the combination of within-district and between-district anomaly levels:

| Operational Alert | Rule | Typical Action |
|---|---|---|
| `Monitor` | Within = `Low` and Between = `Low` or `Moderate` | Routine monitoring |
| `Alert` | Within = `Low` and Between = `High` or `Extreme`, or Within = `Moderate` and Between = `Low` or `Moderate` | Heightened surveillance |
| `Respond` | Within = `High` or `Extreme`, or Within = `Moderate` and Between = `High` or `Extreme` | Immediate response |

### 3. IPC AMN Severity

If the dataset includes direct GAM prevalence columns such as WHZ- or MUAC-based prevalence, the app can also compute observed IPC AMN severity phases:

- `Acceptable`
- `Alert`
- `Serious`
- `Critical`
- `Extremely Critical`

If prevalence data are not present, the app still supports anomaly-based early warning and GAM caseload forecasting.

## How the App Works

The application follows this workflow:

1. Load monthly district GAM data.
2. Parse dates and standardize district labels.
3. Calculate within-district anomalies from each district's own historical distribution.
4. Calculate between-district anomalies from peer district and seasonal patterns.
5. Engineer lagged, rolling, trend, seasonal, and district-relative features.
6. Train Random Forest models for regression and anomaly classification.
7. Generate 3-month recursive district forecasts.
8. Convert forecast outputs into anomaly labels and operational alerts.
9. Visualize current conditions, historical patterns, drivers, forecasts, and model checks.

## App Sections

The Streamlit app is organized into four main tabs:

### Exploratory Analysis

- Run summary
- Data quality and methodology
- Current situation summaries
- Top districts by current operational alert
- Observed anomaly and operational alert maps
- Raw classified data table

### Drivers & Context

- Spearman correlation portrait
- District or regional comparison against the national reference
- Historical anomaly trend
- Historical GAM trend

### Forecasts

- 3-month district forecast chart
- Hindcast view for prior out-of-sample predictions
- Forecast table with one row per district and all three monthly projections side by side; optional uncertainty bounds and anomaly details
- Forecast anomaly and operational alert maps

### Model Checks

- Regression metrics such as MAE, RMSE, mean error, and P90 absolute error
- Predicted vs actual plots
- Residual analysis
- Largest forecast error review
- Feature importance charts
- Classification metrics for within-district and between-district anomaly models

## System Architecture

The tool is designed for integration with DHIS2 and the Climate Health Analytics Platform (CHAP).

![System setup in the DHIS2 eco system](Screenshots/architecture.png)

## Installation

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

The default local address is:

```text
http://localhost:8501
```

## Running With Sample Data

The app points to these local sample paths by default:

- `data/sample/data.csv`
- `data/sample/district_map.geojson`

Both files are included in this checkout. Despite the CSV filename, its rows cover January 2021 through December 2025. Select `name` as the GeoJSON district column; its district labels match the CSV.

Missing caseloads are excluded from observed analysis rather than treated as zero. Feature rows with missing covariates are excluded from training. Districts with too little usable history may have no forecast, and sparse histories limit forecast reliability.

All forward forecasts use a common cutoff: the latest valid reporting month in the source CSV, including rows with missing caseloads. For this sample the cutoff is December 2025, so the chart, table, maps, and CSV export forecast January, February, and March 2026. Districts without an observed caseload at the cutoff are excluded from forecasts and shown as `No Data` on maps. District filtering keeps the same cutoff. Historical hindcasts remain separate.

In the sidebar you can choose either:

- `Use local CSV path` / `Use local GeoJSON path`
- `Upload CSV` / `Upload GeoJSON`

When a GeoJSON is provided, the app asks you to select the district name column used for matching.

## Data Requirements

### Required columns

The loader requires `Region_District`, `time_period`, and `Acut_Malnutrition`. `District` and `Region` are optional: when absent, labels are derived from `Region_District`. Numeric covariates are optional, and the supplied sample uses the following schema:

| Column | Description |
|---|---|
| `Region_District` | Region and district label, for example `Acholi|Agago District` |
| `District` | District name |
| `Region` | Region name |
| `time_period` | Monthly date such as `2020-01` |
| `Acut_Malnutrition` | GAM caseload, interpreted in the app as SAM + MAM |
| `mean_temperature` | Mean temperature |
| `rainfall` | Rainfall |
| `mean_relative_humidity` | Relative humidity |
| `average_gpp` | Vegetation productivity |
| `malaria_confirmed_u5` | Confirmed malaria cases in children under five |
| `pneumonia_cases_u5` | Pneumonia cases in children under five |
| `diarrhea_u5` | Diarrhoea cases in children under five |
| `low_birth_weight_babies` | Low birth weight babies |
| `screened_u5` | Children under five screened |
| `reporting_rate` | Reporting rate, in the source file's units |
| `population_u5` | District population under five |

The app also supports the older covariates `malaria_confirmed`, `pneumonia_cases`, `pregnant_women_with_Anaemia`, `diarrhea_acute`, `diarrhea_persistent`, and `population` when supplied. Under-five columns retain their own names and labels. `population_u5` is a model covariate, not a substitute for total population; the per-100,000 population outputs require `population` or `totalpopulation`.

### Optional severity columns

Observed IPC AMN severity can be calculated if the dataset includes a direct GAM prevalence column. The app checks for common WHZ- and MUAC-based naming variants such as:

- `gam_whz`
- `gam_whz_pct`
- `gam_muac`
- `gam_muac_pct`

### GeoJSON requirements

- Upload or point to a district-level GeoJSON
- Select the GeoJSON column that contains district names
- Ensure district names match the CSV closely enough for joining

## Model Approach

### Regression

A `RandomForestRegressor` is used to forecast district GAM caseloads for the next 3 months.

### Classification

Two `RandomForestClassifier` models are used to evaluate:

- within-district anomaly levels
- between-district anomaly levels

### Features

The model pipeline uses:

- climate, disease, maternal-child health, and population covariates
- lagged values
- rolling summaries
- seasonality features
- district-relative thresholds
- recent anomaly history
- trend and acceleration signals

## Outputs You Can Download

- classified observed CSV
- forecast CSV
- district summary CSV with one row per district (matches the displayed table)
- within-district anomaly metrics CSV
- between-district anomaly metrics CSV

## Screens

### Current Situation and Observed Maps

The app shows side-by-side summaries for:

- within-district anomaly
- between-districts anomaly
- operational alert

It also renders observed district maps when a GeoJSON is provided.

![Observed maps and current situation](Screenshots/Screenshot2png.png)

### Historical Anomaly Trend

The historical anomaly view shows how district counts move across anomaly levels over time.

![Historical anomaly trend](Screenshots/Screenshot3.png)

### Spearman Correlation Portrait

The drivers view includes a rank-based correlation portrait between GAM caseload and key covariates.

![Spearman correlation portrait](Screenshots/Screenshot4.png)

### Forecasts

The forecast tab includes district forecast charts, forecast tables, and forecast maps for each horizon.

![Forecast view](Screenshots/Screenshot6.png)

## Troubleshooting

### NumPy `incompatible architecture` on Apple Silicon

If the traceback says `have 'x86_64', need 'arm64'`, an Intel NumPy binary is being loaded by ARM Python. Create a fresh project environment and install dependencies with that environment's Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m streamlit run app.py
```

If `.venv` already exists and has the same architecture error, use a new environment directory such as `.venv-arm64` in the first two commands. This leaves the existing environment intact.

Activate the environment again in each new terminal before running the app. Use the Streamlit command above to start the web interface.

### `streamlit: command not found`

```bash
python -m pip install -r requirements.txt
```

or

```bash
python -m pip install streamlit
```

### Port already in use

```bash
python -m streamlit run app.py --server.port 8502
```

### Missing package errors

```bash
python -m pip install -r requirements.txt
```

## Project Structure

```text
.
├── app.py
├── app/
├── data/
│   └── sample/
├── docs/
├── impact_metrics/
├── Screenshots/
└── requirements.txt
```

## Additional Documentation

- Technical notes: [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)
- Impact framing: [impact_metrics/IMPACT_METRICS.md](impact_metrics/IMPACT_METRICS.md)

## Notes and Limitations

- Forecast accuracy depends on data quality, completeness, and reporting consistency.
- The anomaly labels are percentile-based early-warning indicators, not direct clinical diagnosis.
- Severity phases require direct GAM prevalence data and may be unavailable in many datasets.
- GeoJSON district naming mismatches can prevent map joins.
- Model outputs should be interpreted alongside field surveillance and program context.
