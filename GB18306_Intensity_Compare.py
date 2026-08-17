# -*- coding: utf-8 -*-
"""
烈度衰减对比：GB18306 直接烈度 vs PGA/PGV -> GB/T 17742-2020 仪器烈度
=====================================================================
功能：
    1) 用 GB18306-2015 青藏区（或任意分区）长轴/短轴衰减关系，
       方法一：直接算烈度 I = A + B*Ms + C*lg(R + R0)
       方法二：先算 PGA(gal)、PGV(cm/s)，再按 GB/T 17742-2020 换算仪器烈度
    2) 出 2 排图（长短轴分列）：
       第一排：两种方法的烈度衰减曲线（含 ±1σ 阴影）
       第二排：差异 Δ = I(GB17742) - I(GB18306)，只画中值，Y 轴 0.2 度一档，上限 0
    3) 同时导出 CSV（中值 + ±1σ + 差异）

运行方式（在本目录下）：
    python run_GB18306_compare.py

依赖：numpy、pandas、matplotlib（本地 GB18306_class.py、GB17742_class.py 同目录）
"""

import os
import sys
import math

import numpy as np
import pandas as pd

# ---------------- 绘图标准化（plot-style） ----------------
import matplotlib

matplotlib.use("Agg")  # 无弹窗
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,  # ASCII "-" 代替 U+2212
        "mathtext.default": "regular",
        "font.size": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)
CM2IN = 1 / 2.54

# ---------------- 调用本目录下的包 ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from GB18306_class import (  # noqa: E402
    GB18306_2015_IntensityCal,
    GB18306_2015_PGA_PGV_GMMs,
)
from GB17742_class import (  # noqa: E402
    GB17742_2020_Cal_instrument_intensity as Cal,
)

# ================= 可调参数 =================
REGION = "青藏区"  # 东部区 / 中部区 / 新疆区 / 青藏区
MS = 7.5  # 面波震级 Ms
R_MIN, R_MAX = 1.0, 400.0  # 距离范围 km
N_R = 200  # 距离点数（对数均匀）
AXES = ["长轴", "短轴"]

adjust = 0.5
# ============================================

DIST = np.logspace(math.log10(R_MIN), math.log10(R_MAX), N_R)

cal_i = GB18306_2015_IntensityCal()
gmm = GB18306_2015_PGA_PGV_GMMs()


def main():
    # ---------- 计算 ----------
    data = {}
    for axis in AXES:
        I1_mid, I1_lo, I1_up = [], [], []
        I2_mid, I2_lo, I2_up = [], [], []
        for R in DIST:
            # 方法一：GB18306 直接烈度
            r = cal_i.calculate(M=MS, R=R, region=REGION, axis_type=axis)
            I1_mid.append(r["mean"])
            I1_lo.append(r["lower_1sigma"])
            I1_up.append(r["upper_1sigma"])

            # 方法二：PGA/PGV -> GB17742 仪器烈度
            (aE, aE_lo, aE_up), (vE, vE_lo, vE_up) = gmm.calculate(
                MS, R, REGION, axis
            )
            I2_mid.append(Cal.cal_Intensity(aE, vE))
            I2_lo.append(Cal.cal_Intensity(aE_lo, vE_lo))
            I2_up.append(Cal.cal_Intensity(aE_up, vE_up))

        data[axis] = {
            "I1_mid": np.array(I1_mid),
            "I1_lo": np.array(I1_lo),
            "I1_up": np.array(I1_up),
            "I2_mid": np.array(I2_mid) + adjust,
            "I2_lo": np.array(I2_lo) + adjust,
            "I2_up": np.array(I2_up) + adjust,
            "diff_mid": np.array(I2_mid) - np.array(I1_mid) + adjust,
            "diff_lo": np.array(I2_lo) - np.array(I1_up) + adjust,
            "diff_up": np.array(I2_up) - np.array(I1_lo) + adjust,
        }

    # ---------- 导出 CSV ----------
    df = pd.DataFrame({"Distance_km": np.round(DIST, 2)})
    for axis in AXES:
        d = data[axis]
        tag = "Long" if axis == "长轴" else "Short"
        df[f"I_GB18306_{tag}"] = np.round(d["I1_mid"], 2)
        df[f"I_GB18306_{tag}_lo"] = np.round(d["I1_lo"], 2)
        df[f"I_GB18306_{tag}_up"] = np.round(d["I1_up"], 2)
        df[f"I_GB17742_{tag}"] = np.round(d["I2_mid"], 2)
        df[f"I_GB17742_{tag}_lo"] = np.round(d["I2_lo"], 2)
        df[f"I_GB17742_{tag}_up"] = np.round(d["I2_up"], 2)
        df[f"Diff_{tag}"] = np.round(d["diff_mid"], 2)

    os.makedirs("Test_output", exist_ok=True)
    csv_path = os.path.join(
        BASE, f"./Test_output/GB18306_Intensity_Compare_{REGION}_Ms{MS}.csv"
    )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ---------- 绘图：2 排 × 2 列（标准化布局） ----------
    fig = plt.figure(figsize=(16 * CM2IN, 11 * CM2IN), constrained_layout=True)
    axes = fig.subplots(2, 2)
    colors = {"GB18306": "#2471A3", "GB17742": "#C0392B"}
    xt = [1, 2, 5, 10, 20, 50, 100, 200, 400]
    titles = [
        ["(a) 长轴 · 烈度衰减", "(b) 短轴 · 烈度衰减"],
        ["(c) 长轴 · 差异 ΔI", "(d) 短轴 · 差异 ΔI"],
    ]

    for j, axis in enumerate(AXES):
        d = data[axis]

        # 第一排：两种方法衰减曲线（±1σ 阴影）
        ax = axes[0][j]
        ax.fill_between(
            DIST,
            d["I1_lo"],
            d["I1_up"],
            color=colors["GB18306"],
            alpha=0.15,
            lw=0,
        )
        ax.plot(
            DIST,
            d["I1_mid"],
            color=colors["GB18306"],
            lw=2.2,
            label="GB18306 直接烈度",
        )
        ax.fill_between(
            DIST,
            d["I2_lo"],
            d["I2_up"],
            color=colors["GB17742"],
            alpha=0.15,
            lw=0,
        )
        ax.plot(
            DIST,
            d["I2_mid"],
            color=colors["GB17742"],
            lw=2.2,
            label="PGA/PGV→GB17742",
        )
        ax.set_title(titles[0][j], fontsize=10, pad=6)
        ax.set_xscale("log")
        ax.set_xlim(R_MIN, R_MAX)
        ax.set_ylim(1, 12)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(x) for x in xt])
        ax.set_yticks(range(1, 13))
        ax.set_ylabel("烈度")
        ax.legend(frameon=False, loc="best", fontsize=8)

        # 第二排：差异（只画中值，Y 轴 0.2 度一档，上限 0）
        ax = axes[1][j]
        ax.plot(
            DIST,
            d["diff_mid"],
            color="#7D3C98",
            lw=2.2,
            label="I(GB17742) - I(GB18306)",
        )
        ax.axhline(0, color="black", lw=1, ls=":", alpha=0.7)
        ax.axhline(-0.5, color="g", lw=1, ls="--", alpha=0.95)
        ax.axhline(-1, color="r", lw=1.5, ls="--", alpha=0.95)
        ax.set_title(titles[1][j], fontsize=10, pad=6)
        ax.set_xscale("log")
        ax.set_xlim(R_MIN, R_MAX)
        ymin = math.floor(d["diff_mid"].min() * 5) / 5  # 向下取 0.2 的倍数
        ax.set_ylim(ymin, 0.4)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(x) for x in xt])
        ax.set_yticks(np.arange(ymin, 0.4001, 0.2))
        ax.set_ylabel("Δ烈度")
        ax.set_xlabel("震中距 R (km)")
        ax.legend(frameon=False, loc="best", fontsize=8)

    for ax in axes.flat:
        ax.grid(True, which="major", alpha=0.3, linestyle="--")
        ax.grid(True, which="minor", alpha=0.12, linestyle=":")
        ax.tick_params(which="both", labelsize=9)

    fig.suptitle(
        f"{REGION}  Ms={MS}  烈度衰减对比：GB18306 直接烈度 vs PGA/PGV→GB17742  "
        f"(调整 = {adjust})",
        fontsize=11,
        y=1.05,
    )

    os.makedirs("Test_output", exist_ok=True)
    png_path = os.path.join(
        BASE, f"./Test_output/GB18306_Intensity_Compare_{REGION}_Ms{MS}.png"
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    # ---------- 终端摘要 ----------
    print(f"分区: {REGION}   Ms: {MS}   距离: {R_MIN:.0f}~{R_MAX:.0f} km")
    print("-" * 78)
    print(
        f"{'R(km)':>8} | {'轴':>2} | {'I_GB18306':>9} | {'I_GB17742':>9} | {'Δ':>7}"
    )
    for R_q in [1, 10, 50, 100, 200, 400]:
        idx = int(np.argmin(np.abs(DIST - R_q)))
        for axis in AXES:
            d = data[axis]
            print(
                f"{DIST[idx]:8.1f} | {axis:>2} | "
                f"{d['I1_mid'][idx]:9.2f} | {d['I2_mid'][idx]:9.2f} | "
                f"{d['diff_mid'][idx]:+7.2f}"
            )
    print("-" * 78)
    print("输出文件:")
    print("  图 :", png_path)
    print("  CSV:", csv_path)


if __name__ == "__main__":
    main()
