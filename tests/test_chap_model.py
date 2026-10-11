import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from chap_model import (
    BASE_FEATURES,
    CORE_COVARIATES,
    N_SAMPLES,
    build_prediction_rows,
    build_training_features,
    classify_risk,
    infer_covariate_columns,
    normalize_dataframe,
    predict_model,
    train_model,
)


class CausalPreprocessingTests(unittest.TestCase):
    def test_future_covariate_values_do_not_change_earlier_normalized_rows(self):
        historic = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02", "2020-03", "2020-04"],
                "location": ["A"] * 4,
                "mean_temperature": [20.0, None, 22.0, None],
                "population_u5": [1000.0, None, None, None],
            }
        )
        future = pd.DataFrame(
            {
                "time_period": ["2020-05"],
                "location": ["A"],
                "mean_temperature": [9999.0],
                "population_u5": [999999.0],
            }
        )

        baseline = normalize_dataframe(historic, require_target=False)
        combined = normalize_dataframe(pd.concat([historic, future], ignore_index=True), require_target=False)
        earlier_combined = combined[combined["time_period"].isin(historic["time_period"])].reset_index(drop=True)

        assert_frame_equal(baseline, earlier_combined)

    def test_later_gam_values_do_not_change_earlier_training_features(self):
        base = pd.DataFrame(
            {
                "time_period": [f"2020-{month:02d}" for month in range(1, 8)],
                "location": ["A"] * 7,
                "disease_cases": [10, 12, 11, 15, 14, 16, 13],
                "screened_u5": [1000] * 7,
            }
        )
        changed = pd.concat(
            [
                base,
                pd.DataFrame(
                    {
                        "time_period": ["2020-08"],
                        "location": ["A"],
                        "disease_cases": [900],
                        "screened_u5": [1000],
                    }
                ),
            ],
            ignore_index=True,
        )

        base_norm = normalize_dataframe(base, require_target=True)
        changed_norm = normalize_dataframe(changed, require_target=True)
        base_features, _, _ = build_training_features(base_norm, infer_covariate_columns(base_norm))
        changed_features, _, _ = build_training_features(changed_norm, infer_covariate_columns(changed_norm))
        earlier_changed = changed_features[changed_features["Date"] <= pd.Timestamp("2020-07-01")].reset_index(drop=True)

        assert_frame_equal(base_features.reset_index(drop=True), earlier_changed)

    def test_training_uses_only_the_lean_feature_schema(self):
        data = pd.DataFrame(
            {
                "time_period": [f"2020-{month:02d}" for month in range(1, 7)],
                "location": ["A"] * 6,
                "disease_cases": [10, 12, 11, 15, 14, 16],
                "screened_u5": [1000] * 6,
                "rainfall": [10, 20, 30, 40, 50, 60],
                "pneumonia_cases_u5": [1, 2, 3, 4, 5, 6],
            }
        )
        normalized = normalize_dataframe(data, require_target=True)
        features, feature_cols, _ = build_training_features(normalized, infer_covariate_columns(normalized))
        expected = BASE_FEATURES + [
            covariate for covariate in CORE_COVARIATES if covariate in normalized.columns
        ] + [
            f"{covariate}_missing" for covariate in CORE_COVARIATES if f"{covariate}_missing" in normalized.columns
        ] + [
            f"{covariate}_outlier" for covariate in CORE_COVARIATES if f"{covariate}_outlier" in normalized.columns
        ]

        self.assertEqual(feature_cols, expected)
        self.assertNotIn("pneumonia_cases_u5", feature_cols)
        self.assertNotIn("rainfall_Lag2", feature_cols)
        self.assertFalse(features[feature_cols].isna().any().any())

    def test_chap_entrypoints_accept_optional_covariates(self):
        train = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"],
                "location": ["A"] * 5,
                "disease_cases": [10, 12, 11, 15, 14],
                "mean_temperature": [20.0, 21.0, 20.0, 22.0, 21.0],
            }
        )
        historic = pd.concat(
            [
                train,
                pd.DataFrame(
                    {
                        "time_period": ["2020-01", "2020-02"],
                        "location": ["B", "B"],
                        "disease_cases": [5, 6],
                    }
                ),
            ],
            ignore_index=True,
        )
        future = pd.DataFrame(
            {"time_period": ["2020-06", "2020-03"], "location": ["A", "B"]}
        )

        with TemporaryDirectory() as directory:
            workdir = Path(directory)
            train_path = workdir / "train.csv"
            future_path = workdir / "future.csv"
            model_path = workdir / "model.pkl"
            output_path = workdir / "predictions.csv"
            enriched_output_path = workdir / "predictions_enriched.csv"
            train.to_csv(train_path, index=False)
            historic.to_csv(workdir / "historic.csv", index=False)
            future.to_csv(future_path, index=False)

            train_model(str(train_path), str(model_path))
            predict_model(str(model_path), str(workdir / "historic.csv"), str(future_path), str(output_path))

            predictions = pd.read_csv(output_path)
            expected_columns = ["time_period", "location"] + [f"sample_{index}" for index in range(100)]
            self.assertEqual(predictions.columns.tolist(), expected_columns)
            self.assertEqual(len(predictions), 2)
            self.assertTrue(predictions.loc[predictions["location"] == "A"].filter(like="sample_").ge(0).all().all())
            self.assertTrue(np.isfinite(predictions.filter(like="sample_").to_numpy(dtype=float)).all())

            predict_model(
                str(model_path),
                str(workdir / "historic.csv"),
                str(future_path),
                str(enriched_output_path),
                include_risk_output=True,
            )
            enriched = pd.read_csv(enriched_output_path)
            self.assertTrue({"point_forecast", "operational_alert", "operational_alert_code"}.issubset(enriched.columns))
            self.assertEqual(enriched.loc[enriched["location"] == "B", "operational_alert"].iloc[0], "No Data")

    def test_chap_uses_separate_direct_one_and_three_month_models(self):
        periods = pd.date_range("2020-01-01", periods=18, freq="MS").strftime("%Y-%m").tolist()
        train = pd.DataFrame(
            {
                "time_period": periods,
                "location": ["A"] * len(periods),
                "disease_cases": list(range(10, 28)),
                "screened_u5": [1000] * len(periods),
            }
        )
        future = pd.DataFrame(
            {
                "time_period": ["2021-07", "2021-08", "2021-09"],
                "location": ["A", "A", "A"],
            }
        )
        with TemporaryDirectory() as directory:
            workdir = Path(directory)
            train_path = workdir / "train.csv"
            model_path = workdir / "model.pkl"
            historic_path = workdir / "historic.csv"
            future_path = workdir / "future.csv"
            output_path = workdir / "predictions.csv"
            enriched_output_path = workdir / "predictions_enriched.csv"
            train.to_csv(train_path, index=False)
            train.to_csv(historic_path, index=False)
            future.to_csv(future_path, index=False)

            train_model(str(train_path), str(model_path))
            predict_model(str(model_path), str(historic_path), str(future_path), str(output_path))
            predictions = pd.read_csv(output_path)

            samples = predictions.filter(like="sample_")
            self.assertTrue(samples.iloc[0].notna().all())
            self.assertTrue(samples.iloc[1].notna().all())
            self.assertTrue(samples.iloc[2].notna().all())
            self.assertTrue(np.isfinite(samples.to_numpy(dtype=float)).all())

            predict_model(
                str(model_path),
                str(historic_path),
                str(future_path),
                str(enriched_output_path),
                include_risk_output=True,
            )
            enriched = pd.read_csv(enriched_output_path)
            self.assertEqual(
                enriched.loc[enriched["time_period"] == "2021-08", "operational_alert"].iloc[0],
                "No Data",
            )

    def test_missing_or_invalid_gam_is_not_imputed_or_used_for_training(self):
        data = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08"],
                "location": ["A"] * 8,
                "disease_cases": [10, None, 12, -1, 15, 16, 17, 18],
                "screened_u5": [1000, 1000, 1200, 1000, 1500, 1600, 1700, 1800],
            }
        )

        normalized = normalize_dataframe(data, require_target=True)
        self.assertEqual(normalized.loc[0, "disease_cases"], 10.0)
        self.assertTrue(pd.isna(normalized.loc[1, "disease_cases"]))
        self.assertTrue(pd.isna(normalized.loc[3, "disease_cases"]))
        self.assertEqual(normalized.loc[1, "target_exclusion_reason"], "missing_gam")
        self.assertEqual(normalized.loc[3, "target_exclusion_reason"], "negative_gam")

        features, _, _ = build_training_features(normalized, infer_covariate_columns(normalized))
        self.assertNotIn(pd.Timestamp("2020-02-01"), features["Date"].tolist())
        self.assertNotIn(pd.Timestamp("2020-04-01"), features["Date"].tolist())

    def test_target_and_lags_remain_counts_regardless_of_screening(self):
        data = pd.DataFrame(
            {
                "time_period": [f"2020-{month:02d}" for month in range(1, 7)],
                "location": ["A"] * 6,
                "disease_cases": [1, 10, 12, 1300, 15, 16],
                "screened_u5": [1, 2000, None, 1000, 0, -1],
            }
        )

        normalized = normalize_dataframe(data, require_target=True)
        self.assertEqual(normalized["disease_cases"].tolist(), data["disease_cases"].tolist())
        self.assertTrue(normalized["target_valid"].eq(1).all())
        self.assertEqual(normalized.loc[3, "target_outlier_qc"], 1.0)

        features, _, _ = build_training_features(normalized, infer_covariate_columns(normalized))
        self.assertEqual(features["Target"].tolist(), [1300, 15, 16])
        self.assertEqual(features.loc[0, ["Lag1", "Lag2", "Lag3"]].tolist(), [12, 10, 1])
        np.testing.assert_allclose(features["Target_Transformed"], np.log1p([1300, 15, 16]))

        without_screening = normalize_dataframe(data.drop(columns="screened_u5"), require_target=True)
        without_features, _, _ = build_training_features(without_screening, [])
        assert_frame_equal(features[[*BASE_FEATURES, "Target"]], without_features[[*BASE_FEATURES, "Target"]])

    def test_duplicate_and_nonfinite_counts_are_excluded(self):
        data = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-01", "2020-02", "2020-03", "2020-04"],
                "location": ["A"] * 5,
                "disease_cases": [10, 12, np.inf, -np.inf, 0],
            }
        )
        normalized = normalize_dataframe(data, require_target=True)
        self.assertTrue(normalized.loc[:3, "disease_cases"].isna().all())
        self.assertEqual(normalized["target_exclusion_reason"].tolist(), [
            "duplicate_district_month", "duplicate_district_month", "nonfinite_gam", "nonfinite_gam", "",
        ])
        self.assertEqual(normalized.loc[4, "disease_cases"], 0)
        self.assertEqual(normalized.loc[4, "target_valid"], 1)

    def test_forecast_samples_and_point_forecasts_are_case_counts(self):
        train = pd.DataFrame(
            {
                "time_period": pd.date_range("2020-01-01", periods=12, freq="MS").strftime("%Y-%m"),
                "location": ["A"] * 12,
                "disease_cases": [250] * 12,
                "screened_u5": [5000] * 12,
            }
        )
        future = pd.DataFrame(
            {"time_period": ["2021-01", "2021-02", "2021-03"], "location": ["A"] * 3}
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train.to_csv(root / "train.csv", index=False)
            future.to_csv(root / "future.csv", index=False)
            train_model(str(root / "train.csv"), str(root / "model.pkl"))
            predict_model(
                str(root / "model.pkl"), str(root / "train.csv"), str(root / "future.csv"),
                str(root / "predictions.csv"), include_risk_output=True,
            )
            predictions = pd.read_csv(root / "predictions.csv")
            self.assertEqual(len(predictions), 3)
            np.testing.assert_allclose(predictions.filter(like="sample_").to_numpy(), 250)
            np.testing.assert_allclose(predictions["point_forecast"], 250)

            # Sparse locations also receive fallback samples on the count scale.
            with open(root / "model.pkl", "rb") as model_file:
                artifact = pickle.load(model_file)
            np.testing.assert_allclose(artifact["fallback_samples"], 250)

    def test_forecast_alerts_compare_counts_with_historic_count_thresholds(self):
        historic = normalize_dataframe(pd.DataFrame({
            "time_period": pd.date_range("2020-01-01", periods=10, freq="MS").strftime("%Y-%m"),
            "location": ["A"] * 10,
            "disease_cases": list(range(100, 110)),
            "screened_u5": [100_000] * 10,
        }), require_target=True)
        future = normalize_dataframe(pd.DataFrame({
            "time_period": ["2020-11", "2021-01"], "location": ["A"] * 2,
        }), require_target=False)
        artifact = {"models": {"1": {}, "3": {}}}
        for count, expected, code in [(108, "Monitor", 1), (108.3, "Alert", 2), (109, "Respond", 3)]:
            with self.subTest(count=count), patch(
                "chap_model.direct_model_samples", return_value=np.full(N_SAMPLES, count),
            ):
                predictions = build_prediction_rows(historic, future, artifact, include_risk_output=True)
                self.assertEqual(predictions["operational_alert"].tolist(), [expected] * 2)
                self.assertEqual(predictions["operational_alert_code"].tolist(), [code] * 2)

    def test_saved_rate_model_requires_retraining(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.pkl"
            with open(model_path, "wb") as model_file:
                pickle.dump({"model_version": "chap-rf-v22-finite-chap-samples"}, model_file)
            with self.assertRaisesRegex(ValueError, "Retrain the CHAP model"):
                predict_model(str(model_path), "historic.csv", "future.csv", str(root / "out.csv"))

    def test_operational_alert_thresholds_use_p90_p95(self):
        self.assertEqual(classify_risk(89.9, 90, 95), "Monitor")
        self.assertEqual(classify_risk(90.0, 90, 95), "Alert")
        self.assertEqual(classify_risk(95.0, 90, 95), "Respond")


if __name__ == "__main__":
    unittest.main()
