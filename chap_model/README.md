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
- `screened_u5`, containing assessed children under five
- the covariates declared in `MLproject`

The CHAP target is the observed GAM detection rate per 1,000 assessed children:

```text
disease_cases / screened_u5 * 1,000
```

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

The target is excluded when GAM is missing or negative, assessed children are missing or non-positive, GAM exceeds assessed children, a location-month is duplicated, or its 95% Wilson interval is wider than 125 detections per 1,000.

## Forecast behavior

The model fits separate 400-tree Random Forest regressors for horizons +1 and +3. The +2 month averages paired draws from the independent +1 and +3 models. It does not feed forecast values back into later horizons.

Default prediction output contains:

- `time_period`
- `location`
- `sample_0` through `sample_99`

Each sample is derived from an individual tree prediction. Sparse locations receive a finite training-derived fallback distribution because CHAP requires finite samples for every requested row.

Set `CHAP_INCLUDE_RISK_OUTPUT=1` to append `point_forecast`, `operational_alert`, and `operational_alert_code`. Direct horizons with at least nine valid historic months use district p90/p95 thresholds:

- `Monitor`: below p90
- `Alert`: p90 to below p95
- `Respond`: p95 or above

The +2 bridge and locations without sufficient history remain `No Data` in enriched alert fields.

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
