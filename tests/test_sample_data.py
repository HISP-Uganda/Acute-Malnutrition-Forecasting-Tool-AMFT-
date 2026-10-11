import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from shapely.geometry import box

import app


class SampleDataTests(unittest.TestCase):
    def load_history(self, caseloads):
        count = len(caseloads)
        return app.load_data_from_df(pd.DataFrame({
            "Region_District": ["Test Region|Test District"] * count,
            "time_period": pd.date_range("2024-01-01", periods=count, freq="MS").strftime("%Y-%m"),
            "Acut_Malnutrition": caseloads,
            "malaria_confirmed_u5": np.arange(count) + 10,
            "pneumonia_cases_u5": np.arange(count) + 5,
            "diarrhea_u5": np.arange(count) + 3,
            "screened_u5": np.arange(count) + 100,
            "reporting_rate": np.linspace(80, 95, count),
            "population_u5": [1000] * count,
        }))

    def test_sparse_history_forecasts_without_nan(self):
        # Four observations leave just one training row: the reported crash.
        df = self.load_history([8, 0, 8, 2])
        features, columns, scalers = app.build_features(df)
        self.assertEqual(len(features), 1)
        self.assertTrue(np.isfinite(features[columns + ["Target_Adm_Transformed"]]).all().all())
        with (
            patch.object(app, "make_regressor", side_effect=lambda: RandomForestRegressor(n_estimators=5, random_state=42)),
            patch.object(app, "make_classifier", side_effect=lambda target: RandomForestClassifier(n_estimators=5, random_state=42)),
        ):
            forecast = app.Forecaster().fit(features, columns, scalers).predict(df)
        self.assertEqual(forecast["Step"].tolist(), [1, 2, 3])
        self.assertTrue(np.isfinite(forecast[["Predicted", "Lower_80", "Upper_80"]]).all().all())
        self.assertEqual(forecast["Predicted"].tolist(), [2, 2, 2])

    def test_constant_history_remains_finite(self):
        features, columns, scalers = app.build_features(self.load_history([0] * 8))
        restored = app.inverse_regression_target(features["Target_Adm_Transformed"].to_numpy(), scalers["Test District"])
        np.testing.assert_allclose(restored, 0)
        self.assertTrue(np.isfinite(features[columns]).all().all())

    def test_under_five_covariates_keep_their_meaning(self):
        df = self.load_history([2, 4, 1, 6, 3, 8, 2, 5])
        features, columns, scalers = app.build_features(df)
        restored = app.inverse_regression_target(features["Target_Adm_Transformed"].to_numpy(), scalers["Test District"])
        np.testing.assert_allclose(restored, features["Target_Adm"])
        correlation_columns = app.compute_spearman_correlations(df)["Variable"].tolist()
        for column in ["malaria_confirmed_u5", "pneumonia_cases_u5", "diarrhea_u5", "screened_u5", "population_u5"]:
            self.assertIn(column, columns)
            if df[column].nunique() > 1:
                self.assertIn(column, correlation_columns)
            self.assertIn("Under 5", app.display_label(column))
        self.assertIsNone(app.get_severity_setup(df)["population_col"])
        self.assertEqual(features["reporting_rate"].iloc[0], df["reporting_rate"].iloc[2])

    def test_feature_cache_respects_input_data(self):
        first, _, _ = app.build_features(self.load_history([1, 2, 3, 4]))
        second, _, _ = app.build_features(self.load_history([1, 2, 3, 9]))
        self.assertEqual(first["Target_Adm"].tolist(), [4])
        self.assertEqual(second["Target_Adm"].tolist(), [9])

    def test_common_cutoff_excludes_stale_districts_even_when_filtered(self):
        raw = pd.DataFrame({
            "Region_District": ["Test|Current"] * 4 + ["Test|Stale"] * 4,
            "time_period": ["2025-09", "2025-10", "2025-11", "2025-12"]
                + ["2025-08", "2025-09", "2025-10", "2025-11"],
            "Acut_Malnutrition": [8, 0, 8, 2, 6, 3, 2, 4],
        })
        df = app.load_data_from_df(raw)
        features, columns, scalers = app.build_features(df)
        with (
            patch.object(app, "make_regressor", side_effect=lambda: RandomForestRegressor(n_estimators=5, random_state=42)),
            patch.object(app, "make_classifier", side_effect=lambda target: RandomForestClassifier(n_estimators=5, random_state=42)),
        ):
            model = app.Forecaster().fit(features, columns, scalers)
            forecast = model.predict(df)
            self.assertTrue(model.predict(df[df["District_Name"] == "Stale"]).empty)
        self.assertEqual(forecast["District"].unique().tolist(), ["Current"])
        self.assertEqual(forecast["Month_Year"].tolist(), ["January 2026", "February 2026", "March 2026"])
        self.assertEqual(forecast["Step"].tolist(), [1, 2, 3])

    def test_cutoff_preserves_month_with_missing_caseload(self):
        df = app.load_data_from_df(pd.DataFrame({
            "Region_District": ["Test|Stale"] * 5,
            "time_period": ["2025-08", "2025-09", "2025-10", "2025-11", "2025-12"],
            "Acut_Malnutrition": [6, 3, 2, 4, np.nan],
        }))
        self.assertEqual(df["Date"].max(), pd.Timestamp("2025-11-01"))
        self.assertEqual(df.attrs["forecast_cutoff"], pd.Timestamp("2025-12-01"))
        model = app.Forecaster()
        self.assertTrue(model.predict(df).empty)

    def test_map_shows_unforecast_district_as_no_data(self):
        gdf = app.gpd.GeoDataFrame(
            {"District_Name": ["Current", "Stale"]},
            geometry=[box(32, 0, 33, 1), box(33, 0, 34, 1)], crs="EPSG:4326",
        )
        data = pd.DataFrame({"District_Name": ["Current"], "WD_Risk": ["Low"]})
        with patch.object(app.st, "plotly_chart") as chart:
            app.render_map(gdf, data, "WD_Risk", "January 2026")
        figure = chart.call_args.args[0]
        no_data = next(trace for trace in figure.data if trace.name == "No Data")
        self.assertEqual(list(no_data.locations), [1])
        self.assertIn(app.RISK_COLORS["No Data"], str(no_data.colorscale))

    def test_default_sample_has_finite_training_data_and_matching_districts(self):
        self.assertTrue(Path(app.DEFAULT_CSV_PATH).is_file())
        self.assertTrue(Path(app.DEFAULT_GEOJSON_PATH).is_file())
        df = app.load_data_from_path(app.DEFAULT_CSV_PATH)
        features, columns, scalers = app.build_features(df)
        self.assertGreater(len(features), 0)
        self.assertTrue(np.isfinite(features[columns + ["Target_Adm_Transformed"]]).all().all())
        self.assertTrue(all(np.isfinite(list(scaler.values())).all() for scaler in scalers.values()))
        _, matched = app.load_and_match_geodf(app.DEFAULT_GEOJSON_PATH, df, "name")
        self.assertEqual(matched, df["District_Name"].nunique())


if __name__ == "__main__":
    unittest.main()
