import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from CB14_site_correct import _parse_CB14_site_correct_factor_all_period
from Vs30_site_correction import (
    cb14_site_factor_actual_over_reference,
    correct_observations_to_reference_vs30,
    prepare_site_plot_observations,
    query_station_vs30,
    solve_cb14_a1100,
)


class CB14ReferenceSiteTests(unittest.TestCase):
    def test_a1100_fixed_point_converges_through_user_package(self):
        observed_gal = 112.2
        actual_vs30 = 328.377594
        a1100, iterations, _ = solve_cb14_a1100(
            observed_gal, 328.377594, 500.0
        )
        actual_over_1100 = _parse_CB14_site_correct_factor_all_period(
            Vs30=actual_vs30,
            PGAr=a1100,
            Vref=1100.0,
            Region_name="CH",
            Use_Basin=False,
        )
        pga_factor = float(
            actual_over_1100.loc[
                actual_over_1100["period"] == -1.0, "Site_corrected"
            ].iloc[0]
        )
        self.assertLessEqual(iterations, 10)
        self.assertAlmostEqual(a1100, observed_gal / pga_factor)

    def test_factor_matches_existing_cb14_module(self):
        reference_pga = 105.0
        factors = _parse_CB14_site_correct_factor_all_period(
            Vs30=320.0,
            PGAr=reference_pga,
            Vref=500.0,
            Region_name="CH",
            Use_Basin=False,
        )
        for period in (-1.0, -2.0, 0.3, 6.0):
            expected = float(
                factors.loc[
                    np.isclose(factors["period"], period), "Site_corrected"
                ].iloc[0]
            )
            actual = cb14_site_factor_actual_over_reference(
                period, reference_pga, 320.0, 500.0
            )
            self.assertEqual(actual, expected)

    def test_observations_are_divided_by_actual_over_reference_factor(self):
        vs30 = 280.0
        pga_h = 112.2
        pga_rotd50 = 125.0
        raw = pd.DataFrame(
            {
                "Sta_ID": ["S1"],
                "longi": [100.0],
                "lati": [30.0],
                "Vs30(m/s)": [vs30],
                "PGA_RotD50": [pga_rotd50],
                "PGA_H": [pga_h],
                "EPA_H": [0.8 * pga_h],
                "PGV_RotD50": [13.0],
                "PGV_H": [12.0],
                "EPV_H": [9.0],
                "pSa(T=0.30s)_RotD50": [170.0],
                "pSa(T=0.30s)_H": [150.0],
            }
        )
        corrected, audit = correct_observations_to_reference_vs30(
            raw, [-1, -2, 0.3], verbose=False
        )
        self.assertEqual(corrected.attrs["site_reference_vs30"], 500.0)
        a1100 = float(audit.loc[0, "CB14_A1100_gal"])
        self.assertEqual(audit.loc[0, "PGA_driver_column"], "PGA_RotD50")
        self.assertEqual(float(audit.loc[0, "PGA_driver_raw_gal"]), pga_rotd50)
        self.assertLessEqual(int(audit.loc[0, "CB14_A1100_iterations"]), 10)
        for period, column in (
            (-1.0, "PGA_RotD50"),
            (-1.0, "PGA_H"),
            (-1.0, "EPA_H"),
            (-2.0, "PGV_RotD50"),
            (-2.0, "EPV_H"),
            (0.3, "pSa(T=0.30s)_RotD50"),
            (0.3, "pSa(T=0.30s)_H"),
        ):
            factor = cb14_site_factor_actual_over_reference(
                period, a1100, vs30, 500.0
            )
            self.assertAlmostEqual(
                float(corrected.loc[0, column]),
                float(raw.loc[0, column]) / factor,
            )
            self.assertAlmostEqual(
                float(corrected.loc[0, f"{column}_raw"]),
                float(raw.loc[0, column]),
            )

    def test_chunked_vs30_query_uses_nearest_grid_point(self):
        grid = pd.DataFrame(
            {
                "longi": [100.0, 100.1, 100.0, 100.1],
                "lati": [30.1, 30.1, 30.0, 30.0],
                "Vs30(m/s)": [300.0, 400.0, 500.0, 600.0],
            }
        )
        stations = pd.DataFrame(
            {"longi": [100.01, 100.09], "lati": [30.01, 30.09]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vs30.csv"
            grid.to_csv(path, index=False)
            result = query_station_vs30(stations, path, verbose=False)
        np.testing.assert_allclose(result["Vs30(m/s)"], [500.0, 400.0])

    def test_plot_switch_restores_raw_values_without_mutating_corrected_data(self):
        corrected = pd.DataFrame(
            {
                "Sta_ID": ["S1"],
                "PGA_RotD50": [80.0],
                "PGA_RotD50_raw": [100.0],
                "PGV_RotD50": [9.0],
                "PGV_RotD50_raw": [12.0],
            }
        )
        corrected.attrs.update(
            site_reference_vs30=500.0,
            site_correction_model="CB14",
            site_use_basin=False,
        )

        corrected_plot = prepare_site_plot_observations(corrected, "corrected")
        raw_plot = prepare_site_plot_observations(corrected, "raw")

        self.assertEqual(float(corrected_plot.loc[0, "PGA_RotD50"]), 80.0)
        self.assertEqual(float(raw_plot.loc[0, "PGA_RotD50"]), 100.0)
        self.assertEqual(float(raw_plot.loc[0, "PGV_RotD50"]), 12.0)
        self.assertEqual(float(corrected.loc[0, "PGA_RotD50"]), 80.0)
        self.assertEqual(
            corrected_plot.attrs["site_plot_observations"], "corrected"
        )
        self.assertEqual(raw_plot.attrs["site_plot_observations"], "raw")
        self.assertNotIn("site_reference_vs30", raw_plot.attrs)

        with self.assertRaisesRegex(ValueError, "plot_observations"):
            prepare_site_plot_observations(corrected, "original")


if __name__ == "__main__":
    unittest.main()
