"""跨参数残差单子图组合图、筛选规则和配套 TXT 的回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import residual_evaluation as evaluation


class ResidualEvaluationTests(unittest.TestCase):
    """锁定默认周期、距离/台站筛选及散点视觉编码。"""

    @staticmethod
    def _computation():
        observations = pd.DataFrame(
            {
                "Sta_ID": ["HN_near", "EI_near", "HN_far", "EI_far"],
                "Instrument_Type": ["HN", "EI", "HN", "EI"],
                "lon": [87.0, 87.1, 88.0, 88.1],
                "lat": [28.0, 28.1, 29.0, 29.1],
                "PGA": [10.0, 20.0, 30.0, 40.0],
                "PGV": [1.0, 2.0, 3.0, 4.0],
            }
        )
        return {
            "params": [-1.0, -2.0],
            "infos": {
                -1.0: {"label": "PGA", "unit": "cm/s2"},
                -2.0: {"label": "PGV", "unit": "cm/s"},
            },
            "obs": observations,
            "preds": {
                "PGA": np.array([11.0, 18.0, 36.0, 32.0]),
                "PGV": np.array([1.1, 1.8, 3.6, 3.2]),
            },
            "ress": {
                "PGA": np.array([0.1, -0.1, 0.2, -0.2]),
                "PGV": np.array([0.1, -0.1, 0.2, -0.2]),
            },
            "aeqs": {
                "PGA": (
                    np.array([50.0, 150.0, 250.0, 350.0]),
                    np.array([40.0, 140.0, 240.0, 340.0]),
                ),
                "PGV": (
                    np.array([50.0, 150.0, 250.0, 350.0]),
                    np.array([40.0, 140.0, 240.0, 340.0]),
                ),
            },
            "source_columns": {"PGA": "PGA_RotD50", "PGV": "PGV_RotD50"},
        }

    def test_cea2019_default_parameter_order(self):
        self.assertEqual(
            evaluation.DEFAULT_CEA2019_EVALUATION_PARAMS,
            (
                -1.0,
                -2.0,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.40,
                0.50,
                0.75,
                1.00,
                1.50,
                2.00,
                3.00,
                4.00,
                5.00,
                6.00,
            ),
        )

    def test_gb18306_default_parameter_order(self):
        self.assertEqual(
            evaluation.DEFAULT_GB18306_EVALUATION_PARAMS,
            (-1.0, -2.0, "Intensity"),
        )

    def test_distance_and_station_filters_use_requested_axis(self):
        station_table, summary, metadata = (
            evaluation.build_residual_evaluation_tables(
                self._computation(),
                distance_range=(None, 200.0),
                station_type="强震仪",
                axis="长轴",
            )
        )
        self.assertEqual(metadata["station_type"], "HN")
        self.assertEqual(metadata["distance_label"], "< 200 km")
        self.assertEqual(set(station_table["Sta_ID"]), {"HN_near"})
        self.assertEqual(summary["N"].tolist(), [1, 1])

        far_table, _, _ = evaluation.build_residual_evaluation_tables(
            self._computation(),
            distance_range=(200.0, None),
            station_type="EI",
            axis="短轴",
        )
        self.assertEqual(set(far_table["Sta_ID"]), {"EI_far"})

    def test_combined_plot_writes_png_and_structured_txt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "evaluation.png"
            result = evaluation.plot_residual_evaluation_combined(
                self._computation(),
                outpath=plot_path,
                model_name="TEST",
                distance_range=(None, 300.0),
                station_type="all",
            )
            table_path = plot_path.with_suffix(".txt")
            self.assertTrue(plot_path.is_file())
            self.assertTrue(table_path.is_file())
            self.assertEqual(result["metadata"]["scatter_colormap"], "Spectral_r")
            self.assertEqual(
                result["metadata"]["plot_layout"],
                "single_axes_half_violin_box_scatter",
            )
            self.assertEqual(
                result["metadata"]["scatter_markers"]["HN"],
                "circle (o), no outline",
            )
            self.assertEqual(
                result["metadata"]["scatter_markers"]["EI"],
                "triangle_up (^), no outline",
            )
            self.assertEqual(
                result["metadata"]["parameter_annotations"],
                "Median, Mean, Sigma, RMS",
            )
            self.assertEqual(
                result["metadata"]["sample_count_location"],
                "figure subtitle",
            )
            text = table_path.read_text(encoding="utf-8-sig")
            self.assertIn("[SUMMARY]", text)
            self.assertIn("[STATION_DATA]", text)
            self.assertIn("# scatter_colormap: Spectral_r", text)
            self.assertIn("PGA_RotD50", text)

    def test_symmetric_y_limit_rounds_up_to_half_unit(self):
        computation = self._computation()
        computation["ress"]["PGA"] = np.array([2.1, -0.2, 0.1, 0.0])
        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluation.plot_residual_evaluation_combined(
                computation,
                outpath=Path(temp_dir) / "symmetric.png",
                model_name="CEA2019",
                figsize_cm=(20.0, 12.0),
                symmetric_y_step=0.5,
            )
        self.assertEqual(result["metadata"]["y_limits"], [-2.5, 2.5])
        self.assertEqual(result["metadata"]["figure_width_cm"], 20.0)


if __name__ == "__main__":
    unittest.main()
