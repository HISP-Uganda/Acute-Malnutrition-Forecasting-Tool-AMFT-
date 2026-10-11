import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from chap_model import (
    BASE_FEATURES,
    CORE_COVARIATES,
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
                "screened_u5": [1000, 1200, 1100, 1500, 1400],
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
                        "screened_u5": [1000, 1000],
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

    def test_missing_or_unreliable_gam_is_not_imputed_or_used_for_training(self):
        data = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08"],
                "location": ["A"] * 8,
                "disease_cases": [10, None, 12, 1300, 15, 16, 17, 18],
                "screened_u5": [1000, 1000, 1200, 1000, 1500, 1600, 1700, 1800],
            }
        )

        normalized = normalize_dataframe(data, require_target=True)
        self.assertEqual(normalized.loc[0, "disease_cases"], 10.0)
        self.assertTrue(pd.isna(normalized.loc[1, "disease_cases"]))
        self.assertTrue(pd.isna(normalized.loc[3, "disease_cases"]))
        self.assertEqual(normalized.loc[1, "target_exclusion_reason"], "missing_gam")
        self.assertEqual(normalized.loc[3, "target_exclusion_reason"], "gam_exceeds_children_assessed")

        features, _, _ = build_training_features(normalized, infer_covariate_columns(normalized))
        self.assertNotIn(pd.Timestamp("2020-02-01"), features["Date"].tolist())
        self.assertNotIn(pd.Timestamp("2020-04-01"), features["Date"].tolist())

    def test_imprecise_detection_rate_is_flagged_without_using_population(self):
        data = pd.DataFrame(
            {
                "time_period": ["2020-01", "2020-02"],
                "location": ["A", "A"],
                "disease_cases": [1, 10],
                "screened_u5": [1, 1000],
                "population_u5": [1_000_000, 1],
            }
        )

        normalized = normalize_dataframe(data, require_target=True)
        self.assertTrue(pd.isna(normalized.loc[0, "disease_cases"]))
        self.assertEqual(normalized.loc[0, "target_exclusion_reason"], "imprecise_detection_rate")
        self.assertEqual(normalized.loc[0, "target_precision_qc"], 1.0)
        self.assertEqual(normalized.loc[1, "disease_cases"], 10.0)
        self.assertEqual(normalized.loc[1, "target_precision_qc"], 0.0)

    def test_operational_alert_thresholds_use_p90_p95(self):
        self.assertEqual(classify_risk(89.9, 90, 95), "Monitor")
        self.assertEqual(classify_risk(90.0, 90, 95), "Alert")
        self.assertEqual(classify_risk(95.0, 90, 95), "Respond")


if __name__ == "__main__":
    unittest.main()
