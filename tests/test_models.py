import unittest

import numpy as np
import pandas as pd

from CEA2019_class import CEA2019
from CEA2019_epicenter_inversion import invert_epicenter_cea2019
from CEA2019_pre import period_label, predict_period_values
from CEA2019_vs_Obs import load_obs_data as load_cea_obs
from ellipse_fields import CEA2019EllipseField, GB18306EllipseField
from GB17742_class import GB17742_2020_Cal_instrument_intensity
from GB18306_class import GB18306_2015_IntensityCal, GB18306_2015_PGA_PGV_GMMs
from GB18306_epicenter_inversion import invert_epicenter_gb18306, load_station_data
from GB18306_Pre import predict_pga_pgv
from GB18306_vs_Obs import load_obs_data as load_gb_obs


class ClassRoundTripTests(unittest.TestCase):
    def test_gb18306_forward_inverse_round_trip(self):
        intensity = GB18306_2015_IntensityCal()
        gmm = GB18306_2015_PGA_PGV_GMMs()
        for region in ("东部区", "中部区", "新疆区", "青藏区"):
            for axis in ("长轴", "短轴"):
                for magnitude in (5.8, 6.5, 6.8):
                    for distance in (1.0, 10.0, 100.0, 300.0):
                        value = intensity.calculate(magnitude, distance, region, axis)[
                            "mean"
                        ]
                        recovered = intensity.invert_R(value, magnitude, region, axis)[
                            "mean"
                        ]
                        self.assertAlmostEqual(recovered, distance, places=9)

                        pga, pgv = gmm.calculate(magnitude, distance, region, axis)
                        self.assertAlmostEqual(
                            gmm.invert_R_from_aE(pga[0], magnitude, region, axis)[0],
                            distance,
                            places=9,
                        )
                        self.assertAlmostEqual(
                            gmm.invert_R_from_vE(pgv[0], magnitude, region, axis)[0],
                            distance,
                            places=9,
                        )

    def test_corrected_middle_short_axis_pgv_coefficient(self):
        params = GB18306_2015_PGA_PGV_GMMs()._get_params(6.8, "中部区", "短轴", "vE")
        self.assertEqual(params["C"], -1.559)

    def test_cea2019_forward_inverse_round_trip(self):
        for region in ("东部", "中部", "新疆", "青藏"):
            for axis in ("长轴", "短轴"):
                cal = CEA2019(region, axis)
                for period in (-1.0, -2.0, 0.3, 1.0, 3.0, 6.0):
                    for distance in (1.0, 10.0, 100.0, 300.0):
                        value = cal.calculate(period, 6.8, distance)[0]
                        recovered = cal.invert_R(period, 6.8, value)[0]
                        self.assertAlmostEqual(recovered, distance, places=9)

    def test_gb17742_static_and_instance_calls(self):
        cls_value = GB17742_2020_Cal_instrument_intensity.cal_Intensity(100, 10)
        instance_value = GB17742_2020_Cal_instrument_intensity().cal_Intensity(100, 10)
        self.assertEqual(cls_value, instance_value)
        with self.assertRaises(ValueError):
            GB17742_2020_Cal_instrument_intensity.cal_Intensity(0, 10)


class SharedFieldTests(unittest.TestCase):
    def setUp(self):
        self.lon = 87.45
        self.lat = 28.50
        self.strike = 187.0
        self.sta_lon = np.array([87.45, 87.60, 88.00])
        self.sta_lat = np.array([28.50, 28.70, 29.00])

    def test_gb_pre_uses_shared_field_exactly(self):
        field = GB18306EllipseField("青藏区", 6.8, extent=400)
        result = field.predict(
            self.lon,
            self.lat,
            self.strike,
            self.sta_lon,
            self.sta_lat,
        )
        pga, pgv = predict_pga_pgv(
            self.lon,
            self.lat,
            self.strike,
            "青藏区",
            6.8,
            self.sta_lon,
            self.sta_lat,
            extent=400,
            verbose=False,
        )
        np.testing.assert_allclose(pga, result[1], rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(pgv, result[2], rtol=0, atol=0, equal_nan=True)

    def test_cea_pre_uses_shared_field_exactly(self):
        periods = (-1.0, -2.0, 0.3, 1.0, 3.0, 6.0)
        field = CEA2019EllipseField("青藏区", 6.8, extent=400)
        shared = field.predict(
            periods,
            self.lon,
            self.lat,
            self.strike,
            self.sta_lon,
            self.sta_lat,
        )
        pre = predict_period_values(
            self.lon,
            self.lat,
            self.strike,
            "青藏区",
            6.8,
            periods,
            self.sta_lon,
            self.sta_lat,
            extent=400,
        )
        for period in periods:
            np.testing.assert_allclose(
                pre[period_label(period)],
                shared[period][0],
                rtol=0,
                atol=0,
                equal_nan=True,
            )


class ObservationLoadingTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame(
            {
                "Sta_ID": ["A", "B"],
                "longi": [87.4, 87.5],
                "lati": [28.5, 28.6],
                "I": [7.0, 6.5],
                "EPA_H": [100.0, 80.0],
                "PGA_H": [999.0, 999.0],
                "EPV_H": [10.0, 8.0],
                "PGV_H": [99.0, 99.0],
                "pSa(T=0.30s)_H": [150.0, np.nan],
            }
        )

    def test_epa_epv_priority_is_preserved(self):
        gb = load_gb_obs(self.base, (-1, -2, "Intensity"))
        np.testing.assert_array_equal(gb["PGA"].values, [100.0, 80.0])
        np.testing.assert_array_equal(gb["PGV"].values, [10.0, 8.0])
        self.assertEqual(gb.attrs["source_columns"]["PGA"], "EPA_H")
        self.assertEqual(gb.attrs["source_columns"]["PGV"], "EPV_H")

    def test_parameter_nan_is_kept_for_per_parameter_comparison(self):
        cea = load_cea_obs(self.base, (-1, 0.3))
        self.assertTrue(np.isnan(cea.loc[1, "PSA(T=0.30s)"]))
        self.assertEqual(cea.attrs["source_columns"]["PGA"], "EPA_H")

    def test_gb_single_parameter_loader_does_not_require_other_columns(self):
        pga_only = self.base[["Sta_ID", "longi", "lati", "PGA_H"]]
        loaded = load_station_data(pga_only, mode="pga")
        self.assertEqual(len(loaded), 2)
        self.assertTrue(loaded["I"].isna().all())
        self.assertTrue(loaded["pgv"].isna().all())


class InversionGuardTests(unittest.TestCase):
    @staticmethod
    def fault_mesh():
        lon = np.tile(np.array([87.40, 87.45, 87.50]), (3, 1))
        lat = np.tile(np.array([28.55, 28.50, 28.45])[:, None], (1, 3))
        return lon, lat

    def test_cea_joint_inversion_rejects_no_joint_observations(self):
        data = pd.DataFrame(
            {
                "Sta_ID": ["A", "B"],
                "lon": [87.44, 87.48],
                "lat": [28.51, 28.49],
                "EPA_H": [100.0, np.nan],
                "EPV_H": [np.nan, 10.0],
            }
        )
        fault_lon, fault_lat = self.fault_mesh()
        with self.assertRaisesRegex(ValueError, "没有可用台站"):
            invert_epicenter_cea2019(
                data=data,
                Ms=6.8,
                region="青藏区",
                hypo=(87.45, 28.50, 10.0),
                strike=187.0,
                dip=49.0,
                rake=-78.0,
                invert_GMIMs=(-1, -2),
                fault_lon_mat=fault_lon,
                fault_lat_mat=fault_lat,
                local_refine=0,
                verbose=False,
            )

    def test_gb_pga_inversion_accepts_pga_only_data(self):
        sta_lon = np.array([87.44, 87.48, 87.52])
        sta_lat = np.array([28.51, 28.49, 28.53])
        field = GB18306EllipseField("青藏区", 6.8)
        pga = field.predict(87.45, 28.50, 187.0, sta_lon, sta_lat)[1]
        data = pd.DataFrame(
            {
                "Sta_ID": ["A", "B", "C"],
                "lon": sta_lon,
                "lat": sta_lat,
                "PGA_H": pga,
            }
        )
        fault_lon, fault_lat = self.fault_mesh()
        result = invert_epicenter_gb18306(
            data=data,
            Ms=6.8,
            region="青藏区",
            hypo=(87.45, 28.50, 10.0),
            strike=187.0,
            dip=49.0,
            rake=-78.0,
            mode="pga",
            fault_lon_mat=fault_lon,
            fault_lat_mat=fault_lat,
            local_refine=0,
            verbose=False,
        )
        self.assertEqual(result["n_used"], 3)
        self.assertAlmostEqual(result["lon"], 87.45, places=12)
        self.assertAlmostEqual(result["lat"], 28.50, places=12)
        self.assertEqual(result["observation_columns"]["PGA"], "PGA_H")


if __name__ == "__main__":
    unittest.main()
