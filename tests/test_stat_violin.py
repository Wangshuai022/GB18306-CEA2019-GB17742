"""残差分布图统计标注的回归测试。"""

from __future__ import annotations

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stat_violin import half_violin_box_scatter


class ResidualAnnotationTests(unittest.TestCase):
    """锁定 vs_Obs 第四排共用的统计定义与显示字段。"""

    def test_annotation_contains_n_mean_median_sigma_and_rms(self):
        """统计框必须同时显示 N、均值、中位数、总体标准差和 RMS。"""
        data = np.array([1.0, 2.0, 3.0])
        fig, ax = plt.subplots()
        try:
            half_violin_box_scatter(
                ax,
                data,
                x=0.0,
                color="#1f77b4",
                value_fmt="{:.2f}",
            )
            self.assertEqual(len(ax.texts), 1)
            label = ax.texts[0].get_text()
            self.assertIn("$N$ = 3", label)
            self.assertIn("$\\mu$ = 2.00", label)
            self.assertIn("$m$ = 2.00", label)
            self.assertIn("$\\sigma$ = 0.82", label)
            self.assertIn("RMS = 2.16", label)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
