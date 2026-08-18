"""vs_Obs 图件与配套数据表输出契约。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import CEA2019_vs_Obs as cea
import GB18306_vs_Obs as gb

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ROOT / "20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt"


class PairedPlotTableOutputTests(unittest.TestCase):
    """公共绘图接口必须同时留下可复用的逐台站TXT数据。"""

    def _assert_readable_table(self, png_path: Path) -> None:
        txt_path = png_path.with_suffix(".txt")
        self.assertTrue(png_path.is_file(), png_path)
        self.assertTrue(txt_path.is_file(), txt_path)

        table = pd.read_csv(txt_path, sep="\t", encoding="utf-8-sig")
        self.assertGreater(len(table), 0)
        self.assertTrue(
            {"Sta_ID", "Sta_longi", "Sta_lati", "Instrument_Type"}.issubset(
                table.columns
            )
        )
        for suffix in ("_obs", "_pred", "_res"):
            self.assertTrue(any(col.endswith(suffix) for col in table.columns))
        self.assertTrue(any(col.startswith("Repi_long_") for col in table.columns))
        self.assertTrue(any(col.startswith("Repi_short_") for col in table.columns))

    def test_gb18306_plot_automatically_writes_same_stem_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = Path(temp_dir) / "gb_plot.png"
            gb.plot_gb18306_vs_obs(
                data=OBSERVATIONS,
                macro_epicenter=(87.612, 28.823),
                Ms=6.8,
                region="青藏区",
                strike=349.0,
                params=(-1,),
                grid_n=8,
                extent=150.0,
                max_dist=100.0,
                outpath=png_path,
            )
            self._assert_readable_table(png_path)

    def test_cea2019_plot_automatically_writes_same_stem_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = Path(temp_dir) / "cea_plot.png"
            cea.plot_cea2019_vs_obs(
                data=OBSERVATIONS,
                macro_epicenter=(87.5597, 28.8978),
                Ms=6.8,
                region="青藏",
                strike=187.0,
                params=(-1,),
                grid_n=8,
                extent=150.0,
                max_dist=100.0,
                outpath=png_path,
            )
            self._assert_readable_table(png_path)


if __name__ == "__main__":
    unittest.main()
