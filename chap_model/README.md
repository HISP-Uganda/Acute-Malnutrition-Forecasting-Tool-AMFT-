# AMFT model for DHIS2 CHAP

This repository includes a CHAP-compatible acute-malnutrition model beside the Streamlit application. Its root-level layout and model behavior follow [acute-malnutrition-RF-model](https://github.com/omielp/acute-malnutrition-RF-model):

- `MLproject`: external-model metadata and entry points
- `MLProject.yaml`: equivalent metadata for model-template seeding
- `chap_model.py`: preprocessing, training, sampling, and prediction
- `train.py` and `predict.py`: thin CHAP command wrappers
- `pyproject.toml` and `uv.lock`: reproducible UV environment

The Streamlit app remains in `app.py` and continues to use its existing raw-caseload workflow. The CHAP model is an independent deployment interface.

## Input contract

Training and historic CSV files use:

- `time_period` in monthly `YYYY-MM` format
- `location`
- `disease_cases`, containing reported GAM cases
- available covariates declared in `MLproject`, including `screened_u5` for assessed children under five

The CHAP target is observed GAM caseload (SAM + MAM). `disease_cases` remains in case-count units throughout target preparation, and predictions are restored to case counts after the model's internal `log1p` transformation. `screened_u5` is an optional predictor of screening activity; it is not used to divide the target.

The declared covariates are:

- `mean_temperature`
- `rainfall`
- `mean_relative_humidity`
- `average_gpp`
- `malaria_confirmed_u5`
- `pneumonia_cases_u5`
- `diarrhea_u5`
- `low_birth_weight_babies`
- `screened_u5`
- `reporting_rate`
- `population_u5`

Prediction receives historic data with the observed fields and future data containing `time_period`, `location`, and available covariates. Missing covariates are handled causally with a one-month forward fill followed by past-only location-month, location, and global medians. Missing and outlier indicator features preserve data-quality information.

The target is excluded when GAM is missing, negative, or non-finite, or a location-month is duplicated. Missing targets are never imputed. Suspicious target spikes are flagged for quality review and retained as observed counts. Missing or invalid screening values are handled as covariate-quality issues and do not exclude GAM counts; counts are also retained when they exceed reported screening volume.

## Forecast behavior

The model fits separate 400-tree Random Forest regressors for horizons +1 and +3. The +2 month averages paired draws from the independent +1 and +3 models. It does not feed forecast values back into later horizons.

Default prediction output contains:

- `time_period`
- `location`
- `sample_0` through `sample_99`

Each sample is derived from an individual tree prediction and represents GAM cases. Sparse locations receive a finite training-derived fallback distribution in the same case-count units because CHAP requires finite samples for every requested row.

Set `CHAP_INCLUDE_RISK_OUTPUT=1` to append `point_forecast` in GAM cases, `operational_alert`, and `operational_alert_code`. Direct horizons with at least nine valid historic months compare forecast counts with district historical case-count p90/p95 thresholds:

- `Monitor`: below p90
- `Alert`: p90 to below p95
- `Respond`: p95 or above

The +2 bridge and locations without sufficient history remain `No Data` in enriched alert fields.

Saved models trained on the previous detection-rate target are incompatible with the case-count model and must be retrained. Prediction rejects them with a retraining message to prevent mixing units.

## Within-district and between-district anomalies

CHAP uses `RandomForestRegressor` for GAM forecasting. It does not train or run a `RandomForestClassifier` for anomaly prediction.

Within-district alerts are the optional p90/p95 comparisons described above: the mean forecast is compared with that district's valid historical GAM case counts. Despite its name, `operational_alert` represents this within-district comparison alone. It is not produced by combining within-district and between-district anomaly labels.

Between-district anomaly detection is not implemented in CHAP. The prediction output does not contain separate within-district or between-district anomaly-label columns. Default output contains only `time_period`, `location`, and predictive samples; enriched output adds the point forecast and within-district alert fields.

The Streamlit implementation in `app.py` has a different forecast workflow. It uses a `RandomForestClassifier` for within-district labels (`Low`, `Moderate`, `High`, `Extreme`), with a percentile fallback when the classifier is unavailable. Between-district labels compare predicted GAM counts across eligible districts for each forecast month, with a historical-count fallback when fewer than three predictions are available. Streamlit then combines both labels into an operational alert. Its Model Checks also evaluate a between-district classifier, but that classifier does not generate the forward between-district labels.

Matching these Streamlit methods in CHAP would require implementing the within-district classifier, between-district peer comparisons, and combined alert rule. See the [comparison in the main README](../README.md#chap-and-streamlit-anomaly-methods).

## Convert the bundled data

From the repository root:

```bash
python chap_model/convert_amft_data.py data/sample/data.csv data/private/chap_amft.csv
```

The converter maps `Acut_Malnutrition` to `disease_cases`, maps `District` to `location`, and retains all declared covariates. It preserves missing targets and covariates so the model's quality-control and causal preprocessing rules can handle them.

## Run entry points directly

```bash
python train.py train.csv model.pkl
python predict.py model.pkl historic.csv future.csv predictions.csv
CHAP_INCLUDE_RISK_OUTPUT=1 python predict.py model.pkl historic.csv future.csv predictions_enriched.csv
```

For an end-to-end test with the bundled dataset:

```bash
python chap_model/smoke_test.py
```

Generated files are written under `data/private/chap_smoke_test/`, which is ignored by Git.

## Run through CHAP

After installing CHAP Core, evaluate this repository as a local external model:

```bash
chap eval \
  --model-name ./ \
  --dataset-csv data/private/chap_amft.csv \
  --output-file data/private/amft_chap_evaluation.nc \
  --plot \
  --backtest-params.n-splits 3 \
  --backtest-params.n-periods 3
```

When hosted in a public Git repository, CHAP can resolve the root `MLproject`. A DHIS2 Modeling App deployment can register or seed the repository as a model template using `MLProject.yaml`.
