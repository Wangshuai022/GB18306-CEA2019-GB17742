import contextlib
import io
import unittest
import warnings

import numpy as np

from fault_distance_azimuth_single_multi import rjb_distance_km, rx_distance_km
from GB18306_epicenter_inversion import fault_mesh_points
from Leonard2014_fitted_by_SMD_crust import l14_fitted
from mesh_single_rectangular_finite_fault import (
    build_fault_grid,
    local_offset_to_lonlat,
)


class FaultToolIntegrationTests(unittest.TestCase):
    @staticmethod
    def _build_quietly(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return build_fault_grid(*args, **kwargs)

    def test_l14_and_fault_modules_are_importable_from_project(self):
        result = l14_fitted(7.0, "NF", "板间")
        self.assertGreater(result["L"], 0.0)
        self.assertGreater(result["W"], 0.0)

    def test_build_fault_grid_default_dhypo_is_057_width(self):
        result = self._build_quietly(
            100.0,
            30.0,
            20.0,
            30.0,
            60.0,
            30.0,
            15.0,
            dx=1.0,
            dy=1.0,
        )
        self.assertAlmostEqual(result["dhypo"], 0.57 * 15.0, places=12)

    def test_inversion_fault_mesh_default_dhypo_is_057_width(self):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fault_mesh_points(
                100.0,
                30.0,
                20.0,
                30.0,
                60.0,
                90.0,
                6.5,
                dx=1.0,
                dy=1.0,
                verbose=False,
            )
        self.assertAlmostEqual(result["dhypo_km"], 0.57 * result["W_km"], places=12)

    def test_surface_exposure_keeps_current_downshift_policy(self):
        result = self._build_quietly(
            87.45,
            28.5,
            10.0,
            187.0,
            49.0,
            60.0,
            30.0,
            dhypo=17.1,
            dx=1.0,
            dy=1.0,
        )
        self.assertTrue(result["surface_exposed"])
        self.assertAlmostEqual(np.min(result["depth_matrix"]), 0.0, places=12)
        self.assertAlmostEqual(
            result["final_hypo_depth"],
            result["original_hypo_depth"] + result["depth_shift_km"],
            places=12,
        )

    def test_rx_sign_and_rjb_inside_rectangle(self):
        top_lon, top_lat = local_offset_to_lonlat(
            0.0, 0.0, np.array([-10.0, 10.0]), np.array([0.0, 0.0])
        )
        site_lon, site_lat = local_offset_to_lonlat(
            0.0, 0.0, np.array([0.0, 0.0]), np.array([-10.0, 10.0])
        )
        rx = rx_distance_km(top_lon, top_lat, site_lon, site_lat)
        self.assertGreater(rx[0], 0.0)
        self.assertLess(rx[1], 0.0)
        np.testing.assert_allclose(np.abs(rx), 10.0, atol=2.0e-5)

        east, north = np.meshgrid([-10.0, 10.0], [-5.0, 5.0])
        fault_lon, fault_lat = local_offset_to_lonlat(0.0, 0.0, east, north)
        rjb = rjb_distance_km(fault_lon, fault_lat, 0.0, 0.0)
        self.assertAlmostEqual(float(rjb[0]), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
