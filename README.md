# AMFT Technical Reference

AMFT monitors district-level GAM caseload anomalies and generates short-term case-count forecasts. The implementation is in `app.py`, the single Streamlit entry point. See the [README](../README.md) for installation and usage.

This reference combines the code walkthrough, implementation details, model evaluation, and operational impact measures.

- [Codebase walkthrough](#codebase-walkthrough)
- [Setup and inputs](#1-setup-and-helper-functions)
- [Model training and evaluation](#5-model-training-and-evaluation)
- [Forecasting and uncertainty](#6-forecasting-and-uncertainty)
- [Verification](#verification)
- [Operational impact and field evidence](#operational-impact-and-field-evidence)

## Codebase walkthrough

The eight sections below match the physical order of `app.py`, the application entry point. Each section appears once, with its related helpers and implementations grouped together. Search for `SECTION 1.` through `SECTION 8.` to follow the code walkthrough from setup to application outputs. All definitions are available before the final call to `main()`, so loaders in section 2 can call anomaly rules defined in section 3.

| Section | What to highlight | Functions and objects to show |
| --- | --- | --- |
| [**1. Setup and helper functions**](#1-setup-and-helper-functions) | Libraries, `data/sample/data.csv`, `data/sample/district_map.geojson`, contextual variables, labels, colours, date parsing, column-name matching, and shared display helpers. | Configuration constants, `RiskLabelEncoder`, `parse_date`, `display_label`, `find_matching_col` |
| [**2. Data loading and quality checks**](#2-data-loading-and-quality-checks) | Read local or uploaded CSVs; check required columns; prepare district names and dates; preserve the shared source cutoff; drop missing targets; report completeness, duplicates, zero counts, missing fields, and unusual values. | `load_data_from_bytes`, `load_data_from_path`, `load_data_from_df`, `build_methodology_df`, `compute_data_quality_tables` |
| [**3. Anomaly and operational alert rules**](#3-anomaly-and-operational-alert-rules) | Historical percentile comparisons produce **Low, Moderate, High, Extreme**. Within-district and between-district anomaly combinations produce **Monitor, Alert, Respond**, with explanations. Direct prevalence fields can provide separate observed severity labels. | `classify_risk`, `derive_operational_alert`, `explain_operational_alert`, `get_severity_setup`, `classify_ipc_amn_phase` |
| [**4. Target preparation and feature engineering**](#4-target-preparation-and-feature-engineering) | Use GAM case counts as the target; apply log transformation and district standardization; create lagged observations, rolling summaries, seasonality, historical percentiles, and lagged contextual variables. Drop incomplete feature rows. | `apply_log_transform`, `transform_regression_target`, `inverse_regression_target`, `build_features` |
| [**5. Model training and evaluation**](#5-model-training-and-evaluation) | Random Forest regression and classification, class weights and probability thresholds, chronological walk-forward validation, MAE/RMSE, bias, residuals, classification metrics, and feature importance. | `wf_splits`, `make_classifier`, `make_regressor`, `tune_thresholds`, `run_classification`, `run_regression_cv`, `compute_per_class_metrics` |
| [**6. Forecasting and uncertainty**](#6-forecasting-and-uncertainty) | Recursively project the next three calendar months using one shared cutoff. Earlier predictions feed later steps. Tree-prediction percentiles provide the displayed 80% bounds. Exclude districts without a cutoff observation or usable training features; derive forecast anomaly and alert labels. | `Forecaster.fit`, `Forecaster.predict`, `run_forecast` |
| [**7. Analysis and geospatial visualization**](#7-analysis-and-geospatial-visualization) | Spearman correlations, district/region comparisons, GAM trends, anomaly distributions, boundary matching, observed/forecast maps, full-extent fitting, and grey **No Data** areas. | `compute_spearman_correlations`, `draw_spearman_radial`, `render_compare_time_series`, `render_historical_risk_trend`, `load_and_match_geodf`, `render_map`, `fit_district_map_layout` |
| [**8. Application flow and user outputs**](#8-application-flow-and-user-outputs) | How `main()` connects the workflow; Streamlit caching and session state; sidebar inputs and district filters; four tabs; one district per row with three projections; optional forecast details; CSV downloads. | `inject_app_styles`, `main`, `build_district_forecast_table`, `st.cache_data`, `st.session_state`, `st.download_button` |

Use the section comments for navigation and this table for speaking notes. Function docstrings are the appropriate place for a function's purpose, inputs, outputs, or assumptions; inline comments should explain decisions that are not obvious from the code.

## 1. Setup and helper functions

The application uses pandas/NumPy for data processing, scikit-learn for Random Forest models, GeoPandas/Shapely for district boundaries, Plotly/Matplotlib for charts, and Streamlit for the interface. Configuration defines covariates, display labels, risk ordering, colours, and a model schema version.

Default inputs are `data/sample/data.csv` and `data/sample/district_map.geojson`. Run from the project root:

```bash
python -m streamlit run app.py
```

Shared helpers parse dates, match optional column aliases, hash uploaded files, encode anomaly labels in a fixed order, and format or colour tables. Date parsing accepts year-month and named-month forms, with a general pandas fallback. GeoJSON preparation is grouped with maps in section 7; forecast table construction is in section 8.

## 2. Data loading and quality checks

`load_data_from_df()` requires `Region_District`, `time_period`, and `Acut_Malnutrition`. Optional `District` and `Region` fields supply labels; otherwise labels are derived from `Region_District`, typically `Region|District`. Names are trimmed and observations sorted by district and date.

The latest parsed source date is retained as `forecast_cutoff` before rows missing dates or caseloads are dropped. Missing GAM targets are not filled with zero. Missing feature values cause engineered rows to be excluded from training; covariates are not imputed. Missing calendar months are not inserted.

`compute_data_quality_tables()` reports district/monthly completeness, zero GAM counts, duplicate district-month records, missing population/prevalence fields, and upper-tail outliers above Q3 + 1.5 × IQR. These summaries use the loaded observations after missing dates and targets have been removed. Reporting duplicates and outliers does not automatically exclude them.

The bundled CSV contains 8,760 rows for 146 districts over January 2021–December 2025. Of these, 555 have missing caseloads, leaving 8,205 observed rows before feature engineering. The GeoJSON contains 146 district features; its `name` column matches CSV district labels.

## 3. Anomaly and operational alert rules

### Anomaly levels

`classify_risk()` applies strict upper boundaries:

| Anomaly | Threshold |
| --- | --- |
| Low | Below p75 |
| Moderate | p75 to below p90 |
| High | p90 to below p95 |
| Extreme | p95 and above |
| No Data | Missing value |

Observed within-district thresholds use that district's earlier observations. With fewer than two prior observations, all thresholds equal the current value; under the strict comparisons, this fallback labels that observation Extreme. Sparse-history classifications therefore need careful interpretation.

Observed between-district thresholds start with earlier observations from the same calendar month across districts. Sparse pools fall back to a broader seasonal pool, then all earlier observations, then the full input dataset. The final fallback can include later observations.

### Operational alerts

`derive_operational_alert()` combines anomaly labels:

| Alert | Rule | Action displayed |
| --- | --- | --- |
| Monitor | Within Low and Between Low or Moderate | Routine monitoring |
| Alert | Within Low and Between High or Extreme; or Within Moderate and Between Low or Moderate | Heightened surveillance |
| Respond | Within High or Extreme; or Within Moderate and Between High or Extreme | Immediate response |
| No Data | One or both anomaly labels unavailable | Alert unavailable |

`explain_operational_alert()` supplies the reason. Operational alerts depend on the two anomaly labels; severity arguments in the function signature do not change the combination rule.

### Optional observed severity

`get_severity_setup()` identifies direct WHZ-based GAM prevalence, or MUAC-based prevalence when WHZ is unavailable. `classify_ipc_amn_phase()` maps the supplied percentages to the application's observed severity labels. Severity remains separate from anomaly labels and operational alerts. Forward forecasts predict counts and do not produce prevalence severity phases.

When total `population` or `totalpopulation` exists, a separate display rate is calculated as GAM cases ÷ total population × 100,000. `population_u5` remains a contextual covariate and is not substituted for total population or used to construct the regression target.

## 4. Target preparation and feature engineering

The regression target is the raw `Acut_Malnutrition` case count, interpreted as SAM + MAM. `transform_regression_target()` clips negative values at zero for the transformation and applies `log1p`. The log target is then standardized within district. `inverse_regression_target()` restores predictions to non-negative case counts.

`build_features()` starts at each district's fourth observed row and creates:

- Month, quarter, and sine/cosine seasonality.
- The previous three observed GAM values and a standardized most-recent lag.
- Three-observation rolling mean/standard deviation and six-observation rolling mean.
- The most recent earlier observation from the same calendar month, falling back to the most recent observation.
- Historical mean, standard deviation, p90/p95, and exceedance/ratio relative to p95.
- Recent High/Extreme within-district counts and lagged within/between anomaly scores.
- Numeric covariates from the previous observed row, excluding identifiers and target-derived fields listed in configuration.

Selected skewed covariates are clipped at zero and log-transformed. Selected engineered features use district scaling; non-finite or near-zero standard deviations are replaced by 1. Rows containing missing features are dropped.

Windows follow recorded observations, so a three-observation window does not always represent three consecutive calendar months. The current feature set does not include engineered trend/acceleration or 12-observation rolling summaries.

## 5. Model training and evaluation

The regression factory creates a 700-tree Random Forest with maximum depth 14, minimum leaf size 1, minimum split size 3, square-root feature selection, and random seed 42.

Classification also uses 700 trees. Within-district classification uses maximum depth 16 and minimum leaf size 1; between-district classification uses depth 14 and leaf size 2. Class weights and tuned probability thresholds adjust predictions. `RiskLabelEncoder` preserves Low, Moderate, High, Extreme ordering.

`wf_splits()` creates expanding chronological training windows and later test blocks, requesting up to six folds after an initial 12-month span. Folds require at least 10 training rows and five test rows; insufficient data produce an interface message.

Regression outputs include MAE, RMSE, signed mean error, median/P90 absolute error, residuals, and feature importance. Signed error is predicted minus actual; positive mean error indicates overprediction. Classification outputs include accuracy, balanced accuracy, macro recall, weighted F1, per-class sensitivity/specificity/precision/NPV, likelihood ratios, and confusion-matrix counts.

The train/test folds are chronological, but district scaling is calculated before those splits. Together with the observed peer-label fallback to the full dataset, this means preprocessing and label construction are not fully isolated within each validation fold. The displayed hindcasts use later test-block predictions, rather than a complete three-step recursive backtest.

## 6. Forecasting and uncertainty

`Forecaster.fit()` trains a shared regressor and a within-district anomaly classifier. Contextual covariates are taken from each district's latest usable training row and held fixed over the forecast horizon.

`Forecaster.predict()` uses one common cutoff and predicts the next three calendar months. Eligible districts must have an observed caseload in the cutoff month, at least three observed values, and a fitted district scaler from usable feature rows. Missing cutoff observations are excluded, rather than forecasting from each district's individual last date. District filtering preserves the common cutoff.

Predictions are recursive: each predicted case count and within-district anomaly score is appended to the history used by later steps. Future features are rebuilt at each step. At inference, the seasonal reference uses the value 12 observations back when available. Future between-district lag scores are appended as zero until peer classifications are calculated after recursive prediction.

Within-district forecast anomalies come from the classifier, with a percentile-based fallback. Between-district forecast anomalies compare predicted counts across eligible districts for each forecast month. With fewer than three predictions, thresholds use historical case counts. The two labels determine the operational alert.

For the bundled sample, the cutoff is December 2025 and all forward displays and exports use January, February, and March 2026. Historical hindcasts are shown separately.

### Displayed bounds

The 10th and 90th percentiles of individual tree predictions, transformed back to case counts and rounded, provide `Lower_80` and `Upper_80`. Bounds are non-negative. These are empirical tree-spread bounds; they do not guarantee 80% coverage or fully propagate uncertainty through recursive steps.

The multi-district chart sums district lower and upper bounds. That sum is not an independently estimated aggregate interval. No state-space model or MAE-weighted blend is implemented.

## 7. Analysis and geospatial visualization

The Drivers & Context tab shows Spearman correlations, district/region comparisons, historical anomaly distributions, and GAM totals. Correlations use complete numeric pairs with sufficient variation. The historical anomaly trend's centered six-row smoother is a retrospective display feature.

`load_and_match_geodf()` matches trimmed district names, including case-insensitive remapping to CSV labels. `safe_geojson()` simplifies polygons and prepares serializable properties. The bundled boundaries use geographic longitude/latitude coordinates; external boundaries should use the same coordinate system.

Observed and monthly forecast maps show within-district anomalies, between-district anomalies, and operational alerts. Plotly choropleths fit the full district extent to each panel, with legends below and grey No Data polygons where classifications are unavailable. District hover details and zoom controls remain available.

Observed maps use each district's latest available observation; the common map heading uses the latest date across those observations, so some districts can have older data. Forecast maps use the common future months and do not forecast stale districts.

`render_delta_map()` defines a risk-change map helper but is not currently called by the interface.

## 8. Application flow and user outputs

`main()` coordinates sidebar inputs, loading, features, model evaluation, forecasts, global district filtering, and four tabs:

- **Exploratory Analysis:** quality summaries, methodology, current alerts, observed maps, and classified records.
- **Drivers & Context:** correlations, comparisons, and historical charts.
- **Forecasts:** historical/hindcast/forward charts, one district per table row with three projections, optional bounds/anomaly details, and monthly maps.
- **Model Checks:** regression diagnostics, classification metrics, and feature importance.

Downloads include classified observations, detailed forecast records, district summaries matching the displayed table, and per-class classification metrics.

Streamlit caching reuses loading, feature, evaluation, and forecast computations. Session state retains trained results and refreshes them when the data key or `MODEL_SCHEMA_VERSION` changes. Uploaded data use a content hash; local inputs use their path. When replacing a local CSV at the same path, clear Streamlit's cache and start a new session or restart the app to refresh both loaded data and retained model results.

## Verification

Run the existing regression checks from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests cover sparse/constant histories, finite sample features, input-sensitive feature caching, shared cutoffs, under-five covariate handling, and No Data map rendering. Operational impact measurement is described in [Operational impact and field evidence](#operational-impact-and-field-evidence).

## Operational impact and field evidence

Use these measures to describe the public health value of the application.

### Operational Impact

| Metric | Suggested Measurement |
|---|---|
| Districts monitored | Number of districts included in the dataset |
| Time period covered | First and last month of available data |
| Forecast horizon | Number of months forecasted |
| High-anomaly districts identified | Districts forecast with High or Extreme within-district or between-district anomalies; report each definition separately |
| Districts requiring operational review | Forecast district counts by Monitor, Alert, and Respond |
| Forecast coverage | Eligible forecast districts divided by districts in the dataset; report exclusions for missing cutoff observations or unusable training history |
| Regions monitored | Number of regions represented |
| Data indicators integrated | Number of climate, disease, nutrition, and population covariates |

### Model Performance Impact

| Metric | Why it matters |
|---|---|
| Sensitivity | Ability to detect true high-risk periods |
| Specificity | Ability to avoid false alarms |
| Balanced accuracy | Fairer measure when classes are imbalanced |
| Macro recall | Treats each risk class equally |
| MAE | Average forecast error in case counts |
| RMSE | Penalizes larger forecast errors |

### Public Health Use Cases

- Early identification of districts with increasing anomalies or operational alerts requiring surveillance or response.
- Prioritization of nutrition surveillance.
- Targeted district or regional review.
- Comparison of district trends against national patterns.
- Support for planning outreach, commodities, staffing, and field validation.

### Example Impact Statement

The bundled dataset covers 146 districts over 60 months, from January 2021 through December 2025. The tool provides January–March 2026 GAM case-count forecasts for districts with observed December caseloads and usable training history. By combining district-specific anomaly models, peer comparisons, and spatial mapping, it supports review of districts requiring intensified nutrition surveillance and response. Dataset coverage and forecast coverage should be reported separately.

This describes the tool's intended use. Measured improvements in decisions, response times, or programme outcomes require evidence from field use.

### Evidence to Add Later

After field use, document:

- number of decisions supported;
- number of high-risk districts reviewed;
- response time improvement;
- program actions triggered;
- user feedback from nutrition/public health teams;
- comparison with routine reporting or field alerts.
