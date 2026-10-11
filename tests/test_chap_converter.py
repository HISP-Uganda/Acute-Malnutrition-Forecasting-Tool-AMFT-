import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CONVERTER_DIR = Path(__file__).resolve().parents[1] / "chap_model"
sys.path.insert(0, str(CONVERTER_DIR))

from convert_amft_data import CHAP_COVARIATES, convert_amft_csv  # noqa: E402


class ChapConverterTests(unittest.TestCase):
    def test_converter_preserves_model_covariates_and_missing_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = pd.DataFrame({
                "District": ["Alpha", "Alpha"],
                "time_period": ["2024-01", "2024-02"],
                "Acut_Malnutrition": [4, np.nan],
                **{
                    covariate: [index + 1, index + 2]
                    for index, covariate in enumerate(CHAP_COVARIATES)
                },
            })
            input_path = root / "amft.csv"
            output_path = root / "chap.csv"
            source.to_csv(input_path, index=False)

            result = convert_amft_csv(input_path, output_path)

            self.assertEqual(
                result.columns.tolist(),
                ["time_period", "location", "disease_cases", *CHAP_COVARIATES],
            )
            self.assertEqual(result["location"].tolist(), ["Alpha", "Alpha"])
            self.assertTrue(pd.isna(result.loc[1, "disease_cases"]))
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
