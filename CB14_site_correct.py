# -*- coding: utf-8 -*-
"""
CB14场地矫正--日本区域
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pynga import CB14
import matplotlib.pyplot as plt


def _parse_CB14_site_correct_factor_all_period(
    Vs30, PGAr, Vref, Region_name, Use_Basin=False, Z25_simul=None, Z25_real=None
):
    """
            基于CB14衰减关系场地项进行日本场地非线性/线性响应矫正   无盆地效应
            输入：
            Vs30  场地实际Vs30  （m/s）
            PGAr  模拟得到的场地模拟PGA  （gal）
            Vref   模拟场地Vs30  (m/s)
            Use_Basin   是否使用盆地项
            Z25   z2.5深度  km

            输出： CB14_site_coorected  格式：DataFrame
            -1 代表PGA   -2代表PGV
            示例：
         period  Site_corrected
    0     -1.00        1.858300
    1     -2.00        2.183600
    0      0.01        1.867400
    1      0.02        1.810200
    2      0.03        1.728900
    ..      ...             ...
    995    9.96        1.333844
    996    9.97        1.333658
    997    9.98        1.333472
    998    9.99        1.333286
    999   10.00        1.333100

    """
    # # # 测试
    # PGAr = 400
    # Vs30 = 1100
    # Vref = 1100
    #####################################################333################333################333
    # CB14初始化
    CB14_1 = CB14.CB14_nga()
    # CB14周期点
    periods = [float(key) for key in list(CB14_1.Coefs.keys())]
    # 数据预存储
    Sa_site_coorected = [0] * len(periods)
    for idx, Ti in enumerate(periods):
        # print(Ti)
        # 实际Vs30 场地矫正
        fsite_real = CB14_1.site_function_CB14(
            PGAr / 981,
            Vs30,
            float(Ti),
            Region_name,
            Use_Basin,
            Z25=Z25_real,
        )

        # 模拟Vs30 场地矫正
        fsite_simul = CB14_1.site_function_CB14(
            PGAr / 981,
            Vref,
            float(Ti),
            Region_name,
            Use_Basin,
            Z25=Z25_simul,
        )

        # 场地矫正比值  实际/模拟
        Sa_site_coorected[idx] = round(np.exp(fsite_real - fsite_simul), 4)

    # 差值周期区间 扩展到0.01-10s
    T100 = pd.DataFrame(np.arange(0.01, 10.01, 0.01).reshape(-1, 1), columns=["period"])

    # 提取边界值作为外插填充值  0.033s   8.0s
    left_val = Sa_site_coorected[0]  # 左边界最近值
    right_val = Sa_site_coorected[-3]  # 右边界最近值（切片到-3）

    # 创建插值函数
    f = interp1d(
        periods[:-2],
        Sa_site_coorected[:-2],
        kind="linear",
        bounds_error=False,  # 允许外插
        fill_value=(left_val, right_val),
    )  # 左右外插用边界值

    # 计算插值结果
    Sa_site_coorected_interp = pd.DataFrame(f(T100), columns=["Site_corrected"])
    # 合并为表格
    Sa_site_coorected_interp0 = pd.concat([T100, Sa_site_coorected_interp], axis=1)
    # PGA + PGV的 结果
    PGA_PGV_site_coorected = pd.DataFrame(
        {
            "period": [-1, -2],
            "Site_corrected": [Sa_site_coorected[-2], Sa_site_coorected[-1]],
        }
    )
    # 整体合并
    CB14_site_coorected = pd.concat([PGA_PGV_site_coorected, Sa_site_coorected_interp0], axis=0)

    CB14_site_coorected["Freq"] = 1 / CB14_site_coorected["period"]

    return CB14_site_coorected


def me():
    pass


if __name__ == "__main__":

    PGAr = 216  # 216.8  # 参考场地PGAr  A1100
    Vs30 = 366  # 500  # 实际场地Vs30
    Vref = 1100  # 参考
    Region_name = "Japan"
    Use_Basin = True
    Z25_simul = 0  # km
    Z25_real = 0.5

    # CB14初始化
    CB14_1 = CB14.CB14_nga()
    if Z25_real == None:
        Z25_real = CB14_1.cal_Z25_CB14(Vs30, Region_name=Region_name)
        print(f"Z25_real={Z25_real:.2f}km")

    CB14_site_coorected = _parse_CB14_site_correct_factor_all_period(
        Vs30, PGAr, Vref, Region_name, Use_Basin, Z25_simul, Z25_real
    )
    #### 绘图
    # =========================================================
    # 1. 提取 period 和修正系数
    # =========================================================
    period = np.asarray(CB14_site_coorected["period"], dtype=float)
    site_factor = np.asarray(CB14_site_coorected["Site_corrected"], dtype=float)

    # PGA、PGV 的实际系数，直接从 period = -1 和 -2 读取
    PGA_factor = site_factor[period == -1][0]
    PGV_factor = site_factor[period == -2][0]

    print(f"PGA site correction factor = {PGA_factor:.4f}")
    print(f"PGV site correction factor = {PGV_factor:.4f}")

    # 只画 T >= 0.1 s 的周期点
    mask_period = period >= 0.095
    period_plot = period[mask_period]
    factor_plot = site_factor[mask_period]

    # 按周期排序，避免曲线乱连
    sort_idx = np.argsort(period_plot)
    period_plot = period_plot[sort_idx]
    factor_plot = factor_plot[sort_idx]

    # =========================================================
    # 2. PGA、PGV 的显示位置
    # 注意：这只是为了在 log 坐标轴上显示两个孤立点
    # 不代表 PGA、PGV 真实周期就是 0.05 s 和 0.08 s
    # =========================================================
    x_PGA_plot = 0.06
    x_PGV_plot = 0.08

    # =========================================================
    # 3. 创建图形
    # =========================================================
    fig, ax1 = plt.subplots(1, 1, figsize=(15 / 2.54, 8 / 2.54))

    # 正常周期修正系数曲线，只画 0.1–10 s
    ax1.loglog(period_plot, factor_plot, "b-", linewidth=1.8, label="CB14 site correction factor")

    # PGA、PGV 作为孤立点画出来
    ax1.scatter(
        x_PGA_plot,
        PGA_factor,
        color="m",
        s=55,
        zorder=5,
        # label=f"PGA = {PGA_factor:.2f}"
    )

    ax1.scatter(
        x_PGV_plot,
        PGV_factor,
        color="c",
        s=55,
        zorder=5,
        # label=f"PGV = {PGV_factor:.2f}"
    )

    # y = 1 参考线
    ax1.axhline(1, color="k", linestyle="--", alpha=1, linewidth=1.3)

    # 灰色填充区域：这里只填充 0.1–1 s，不覆盖 PGA、PGV
    # ax1.axvspan(0.1, 0.11, color="gray", alpha=0.25)

    # PGA、PGV 竖线
    ax1.axvline(x=x_PGA_plot, color="m", linestyle=":", alpha=0.8, linewidth=1.3)
    ax1.axvline(x=x_PGV_plot, color="c", linestyle=":", alpha=0.8, linewidth=1.3)

    # =========================================================
    # 4. 标注 PGA 和 PGV
    # =========================================================
    ax1.set_xlim(0.05, 10)
    ax1.set_ylim(0.1, 10)

    y_min, y_max = ax1.get_ylim()

    # log 坐标下，文字位置建议用乘法比例，不要用线性加法
    text_y1 = y_min * (y_max / y_min) ** 0.04
    text_y2 = y_min * (y_max / y_min) ** 0.25

    ax1.text(
        x_PGA_plot,
        text_y1,
        f"PGA\n{PGA_factor:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="m",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="m", alpha=0.85),
    )

    ax1.text(
        x_PGV_plot,
        text_y2,
        f"PGV\n{PGV_factor:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="c",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="c", alpha=0.85),
    )

    # =========================================================
    # 5. 坐标轴设置
    # =========================================================
    x_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    x_ticks_labels = ["0.1", "0.2", "0.5", "1", "2", "5", "10"]

    y_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    y_ticks_labels = [str(y) for y in y_ticks]

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks_labels)

    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks_labels)

    ax1.set_xlabel("Period (s)")
    ax1.set_ylabel("Site Correction Factor CB14")

    ax1.grid(True, which="both", ls=":", lw=0.5, alpha=1)

    ax1.set_title(
        f"CB14--PGAr={PGAr:.2f} gal  "
        f"Z25_simul={Z25_simul:.2f} km  "
        f"Z25_real={Z25_real:.2f} km\n"
        f"Vs30_real={Vs30} m/s  Vs30_simul={Vref} m/s"
    )

    ax1.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.show()
