"""V1.0 发布契约：把用户约定的功能直接锁定到公共代码入口。"""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import CEA2019_epicenter_inversion_Vs30 as cea_vs30
import CEA2019_vs_Obs as cea_obs
import GB18306_epicenter_inversion_Vs30 as gb_vs30
import GB18306_vs_Obs as gb_obs
import residual_evaluation as evaluation
import Vs30_site_correction as site

ROOT = Path(__file__).resolve().parents[1]


def _call_has_keyword(module, function_name, called_name, keyword):
    """检查指定函数内对目标函数的调用是否显式传递某个关键字。"""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = getattr(child.func, "id", None) or getattr(
                child.func, "attr", None
            )
            if name == called_name:
                return any(item.arg == keyword for item in child.keywords)
    return False


class ReleaseContractTests(unittest.TestCase):
    """验证 README 中可由用户直接观察和调用的 V1.0 功能。"""

    def test_observation_priority_is_exact_and_has_no_bare_fallback(self):
        self.assertEqual(
            gb_obs.obs_col_candidates(gb_obs.param_info(-1)),
            ["PGA_RotD50", "PGA_H", "EPA_RotD50", "EPA_H"],
        )
        self.assertEqual(
            gb_obs.obs_col_candidates(gb_obs.param_info(-2)),
            ["PGV_RotD50", "PGV_H", "EPV_RotD50", "EPV_H"],
        )
        self.assertEqual(
            cea_obs.obs_col_candidates(cea_obs.param_info(-1)),
            ["PGA_RotD50", "PGA_H", "EPA_RotD50", "EPA_H"],
        )
        self.assertEqual(
            cea_obs.obs_col_candidates(cea_obs.param_info(-2)),
            ["PGV_RotD50", "PGV_H", "EPV_RotD50", "EPV_H"],
        )
        self.assertEqual(
            cea_obs.obs_col_candidates(cea_obs.param_info(1.0)),
            ["pSa(T=1.00s)_RotD50", "pSa(T=1.00s)_H"],
        )

    def test_vs_obs_public_plot_functions_expose_site_plot_switch(self):
        for function in (
            gb_obs.plot_gb18306_vs_obs,
            cea_obs.plot_cea2019_vs_obs,
        ):
            parameters = inspect.signature(function).parameters
            for name in ("plot_observations", "site_correction_kwargs"):
                self.assertIn(
                    name,
                    parameters,
                    f"{function.__module__}.{function.__name__} 缺少 {name}",
                )

    def test_all_prediction_observation_apps_expose_evaluation_options(self):
        functions = (
            gb_obs.plot_gb18306_vs_obs,
            cea_obs.plot_cea2019_vs_obs,
            gb_vs30.invert_epicenter_gb18306_vs30,
            cea_vs30.invert_epicenter_cea2019_vs30,
        )
        required = {
            "evaluation_path",
            "evaluation_table_path",
            "evaluation_distance_range",
            "evaluation_station_type",
            "evaluation_figsize_cm",
        }
        for function in functions:
            self.assertTrue(
                required.issubset(inspect.signature(function).parameters),
                function.__qualname__,
            )

    def test_vs30_wrappers_default_to_corrected_and_forward_switch(self):
        cases = (
            (
                gb_vs30,
                "invert_epicenter_gb18306_vs30",
                "plot_gb18306_vs_obs",
            ),
            (
                cea_vs30,
                "invert_epicenter_cea2019_vs30",
                "plot_cea2019_vs_obs",
            ),
        )
        for module, function_name, plot_name in cases:
            function = getattr(module, function_name)
            parameter = inspect.signature(function).parameters[
                "plot_observations"
            ]
            self.assertEqual(parameter.default, "corrected")
            self.assertTrue(
                _call_has_keyword(
                    module,
                    function_name,
                    plot_name,
                    "plot_observations",
                ),
                f"{function_name} 没有把绘图开关传给 {plot_name}",
            )

    def test_vs30_wrapper_sends_corrected_data_to_inversion_and_mode_to_plot(self):
        raw = pd.DataFrame(
            {
                "Sta_ID": ["S1"],
                "longi": [87.5],
                "lati": [28.5],
                "PGA_RotD50": [12.0],
            }
        )
        corrected = raw.copy()
        corrected["PGA_RotD50_raw"] = corrected["PGA_RotD50"]
        corrected["PGA_RotD50"] = [10.0]
        corrected.attrs.update(
            site_reference_vs30=500.0,
            site_correction_model="CB14",
            site_use_basin=False,
        )
        audit = pd.DataFrame(
            {"Sta_ID": ["S1"], "Vs30_reference_mps": [500.0]}
        )
        result = {
            "epicenter": (87.6, 28.6),
            "plot_GMIMs": [-1.0],
            "mesh": {
                "lon_mat": [[87.4, 87.6]],
                "lat_mat": [[28.4, 28.6]],
            },
            "table": pd.DataFrame({"Sta_ID": ["S1"]}),
        }
        with (
            mock.patch.object(
                gb_vs30,
                "correct_observations_to_reference_vs30",
                return_value=(corrected, audit),
            ),
            mock.patch.object(
                gb_vs30, "invert_epicenter_gb18306", return_value=result
            ) as inversion,
            mock.patch.object(
                gb_vs30,
                "attach_site_audit_to_result",
                side_effect=lambda value, *args, **kwargs: value,
            ),
            mock.patch.object(gb_vs30, "plot_gb18306_vs_obs") as plot,
        ):
            gb_vs30.invert_epicenter_gb18306_vs30(
                raw,
                Ms=6.8,
                region="青藏区",
                hypo=(87.45, 28.5, 10.0),
                strike=187.0,
                dip=49.0,
                rake=-78.0,
                mode="pga",
                plot_GMIMs=[-1],
                plot_observations="raw",
                plot_path="unused.png",
                verbose=False,
            )
        self.assertIs(inversion.call_args.kwargs["data"], corrected)
        self.assertIs(plot.call_args.kwargs["data"], corrected)
        self.assertEqual(plot.call_args.kwargs["plot_observations"], "raw")

        cea_result = {
            "epicenter": (87.6, 28.6),
            "plot_GMIMs": [-1.0],
            "mesh": result["mesh"],
            "table": pd.DataFrame({"Sta_ID": ["S1"]}),
        }
        with (
            mock.patch.object(
                cea_vs30,
                "correct_observations_to_reference_vs30",
                return_value=(corrected, audit),
            ),
            mock.patch.object(
                cea_vs30, "invert_epicenter_cea2019", return_value=cea_result
            ) as inversion,
            mock.patch.object(
                cea_vs30,
                "attach_site_audit_to_result",
                side_effect=lambda value, *args, **kwargs: value,
            ),
            mock.patch.object(cea_vs30, "plot_cea2019_vs_obs") as plot,
        ):
            cea_vs30.invert_epicenter_cea2019_vs30(
                raw,
                Ms=6.8,
                region="青藏区",
                hypo=(87.45, 28.5, 10.0),
                strike=187.0,
                dip=49.0,
                rake=-78.0,
                invert_GMIMs=[-1],
                plot_GMIMs=[-1],
                plot_observations="raw",
                plot_path="unused.png",
                verbose=False,
            )
        self.assertIs(inversion.call_args.kwargs["data"], corrected)
        self.assertIs(plot.call_args.kwargs["data"], corrected)
        self.assertEqual(plot.call_args.kwargs["plot_observations"], "raw")

    def test_site_engine_uses_500_and_apps_do_not_import_pynga_directly(self):
        self.assertEqual(site.DEFAULT_REFERENCE_VS30, 500.0)
        for filename in (
            "Vs30_site_correction.py",
            "GB18306_epicenter_inversion_Vs30.py",
            "CEA2019_epicenter_inversion_Vs30.py",
        ):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(
                any(name == "pynga" or name.startswith("pynga.") for name in imports),
                f"{filename} 不应直接调用 pynga",
            )

    def test_key_public_functions_have_parameter_and_return_documentation(self):
        functions = (
            gb_obs.load_obs_data,
            gb_obs.compute_vs_obs,
            gb_obs.plot_gb18306_vs_obs,
            cea_obs.load_obs_data,
            cea_obs.plot_cea2019_vs_obs,
            gb_obs.plot_gb18306_residual_evaluation,
            cea_obs.plot_cea2019_residual_evaluation,
            evaluation.build_residual_evaluation_tables,
            evaluation.plot_residual_evaluation_combined,
            site.query_station_vs30,
            site.correct_observations_to_reference_vs30,
            site.prepare_observations_for_site_plot,
            gb_vs30.invert_epicenter_gb18306_vs30,
            cea_vs30.invert_epicenter_cea2019_vs30,
        )
        for function in functions:
            doc = inspect.getdoc(function) or ""
            self.assertIn("Parameters", doc, function.__qualname__)
            self.assertIn("Returns", doc, function.__qualname__)


if __name__ == "__main__":
    unittest.main()
