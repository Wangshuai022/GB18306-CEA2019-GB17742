# -*- coding: utf-8 -*-
"""
CEA2019 地震动参数衰减关系 class（GB 18306-2015 区划图的更新版）
================================================================
功能：由 区域 + 轴向 + 周期点 + 震级 + 震中距，计算地震动参数
（PGA、PGV、PSA 反应谱）的中值及 ±1σ 区间；也可由地震动参数反算震中距。

周期点约定（本文件内部）：
    -1  → PGA（峰值加速度，单位 gal）
    -2  → PGV（峰值速度，单位 cm/s）
    0.04 ~ 6  → PSA(T)（周期 T 秒的反应谱加速度，单位 gal）
    其他数值周期：在既有的周期点之间做线性插值取系数。

系数来源：Excel 文件《GB长短轴衰减关系系数--区划2019.xlsx》
（必须与本文件放在同一目录）。每个区域 × 轴向一个 sheet，
表名格式：{区域}区{轴向}，如 "青藏区长轴"。
每行一列系数：T(s)、A1、B1、A2、B2、C、D、E、σ。

衰减公式（长轴/短轴同形）：
    lg(Y) = A + B * M - C * lg(R + D * exp(E * M))
    Y = 10**lg(Y)
其中 A、B 按震级分段：M < 6.5 用 A1/B1，M ≥ 6.5 用 A2/B2；
C、D、E、σ 不分段；σ 为对数标准差，±1σ = Y/10^σ ~ Y*10^σ。

反算公式：
    R = 10**((A + B*M - lg(Y)) / C) - D * exp(E * M)

用法示例：
    cal = CEA2019(region="青藏", axis="长轴")   # region 不带"区"字
    Y, lo, up = cal.calculate(T=-1, magnitude=7.5, R=100)   # PGA
    R, r_lo, r_up = cal.invert_R(T=0.3, magnitude=7.5, Y=100)

注意：周期边界（<0.04 s 或 >6 s）的外推规则由上层程序（CEA2019_pre.py）
负责把关；本 class 内部对超界周期会取边界值。
"""

import pandas as pd
import numpy as np
from bisect import bisect
import os
from matplotlib import rcParams

# 让中文提示在 Windows 命令行里正常显示
try:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# 设置中文字体和样式
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class CEA2019:
    """
    地震动参数计算器（PGA / PGV / PSA），支持动态读取 Excel 系数文件

    用法：
        cal = CEA2019(region="青藏", axis="长轴")
        cal.calculate(T=-1, magnitude=7.5, R=100)   # 正算：震中距 → 地震动参数
        cal.invert_R(T=0.3, magnitude=7.5, Y=100)   # 反算：地震动参数 → 震中距

    注意：region 不带"区"字（内部拼 sheet 名时会自动加"区"）。
    """

    # Excel 系数文件名（必须与本文件同一目录）
    _EXCEL_NAME = "GB长短轴衰减关系系数--区划2019.xlsx"

    def __init__(self, region, axis):
        # 动态获取本文件所在目录，拼出 Excel 完整路径（换机器也能找到）
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.excel_path = os.path.join(module_dir, self._EXCEL_NAME)
        # 读取所有 sheet 名称，用于校验"区域+轴向"是否合法
        self.all_sheets = pd.ExcelFile(self.excel_path).sheet_names
        # 验证文件存在性（找不到直接报错，避免后面莫名失败）
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"系数文件未找到：{self.excel_path}")
        # sheet 名格式：{region}区{axis}，如"青藏区长轴"
        sheet_name = f"{region}区{axis}"
        if sheet_name not in self.all_sheets:
            raise ValueError(
                f"未找到表格: {sheet_name}，可用表格: {self.all_sheets}"
            )
        # 读取该区域、该轴向的整张系数表
        self.df = pd.read_excel(self.excel_path, sheet_name=sheet_name)

    def _get_coefficients(self, T):
        """
        获取指定周期点的系数行（支持在既有周期点之间线性插值）

        参数：
            T  -1（PGA）/ -2（PGV）/ 数值周期（如 0.3、1、6）
        返回：
            系数行（pandas Series）：A1、B1、A2、B2、C、D、E、σ

        说明：
            - T == -1 / -2 直接返回 Excel 里 PGA / PGV 那一行；
            - 数值周期：用 bisect 找插入位置，在相邻两个已知周期之间
              按周期线性插值所有数值系数；
            - 边界：小于最小周期取第一行，大于最大周期（6s）取最后一行
              （更严格的外推规则由 CEA2019_pre.py 负责，这里只保证能取到系数）。
        """

        df = self.df
        Ts = df["T(s)"].values

        # 处理 PGA / PGV 特殊值（Excel 里是两行字符串行）
        if T == -1:
            return df[df["T(s)"] == "PGA"].iloc[0]
        elif T == -2:
            return df[df["T(s)"] == "PGV"].iloc[0]

        # 数值周期：只在数值周期行里做二分/插值
        # （T(s) 列前两行是字符串 "PGA"/"PGV"，不能参与数值比较）
        num_df = df[pd.to_numeric(df["T(s)"], errors="coerce").notna()]
        Ts = num_df["T(s)"].astype(float).values
        T = float(T)
        idx = bisect(Ts, T)

        # 边界处理
        if idx == 0:
            # 小于最小值直接取最小值（最接近的周期）
            return num_df.iloc[0]
        elif idx == len(Ts):
            # 大于最大值（6s）直接取 6s 那一行
            return num_df.iloc[-1]
        else:
            # 线性插值：T 落在 T_low 和 T_high 之间
            T_low = Ts[idx - 1]
            T_high = Ts[idx]
            weight = (T - T_low) / (T_high - T_low)

            row_low = num_df.iloc[idx - 1]
            row_high = num_df.iloc[idx]

            # 对所有数值列做线性插值，T(s) 列写回真实周期
            interpolated = row_low.copy()
            for col in df.columns:
                if col == "T(s)":
                    interpolated[col] = T
                elif pd.api.types.is_numeric_dtype(df[col]):
                    interpolated[col] = (
                        row_low[col] + (row_high[col] - row_low[col]) * weight
                    )
            return interpolated

    def calculate(self, T, magnitude, R):
        """
        正算：由 (周期 T, 震级 magnitude, 震中距 R) 求地震动参数中值及 ±1σ

        参数：
            T         周期点：-1=PGA、-2=PGV、数值=PSA(T) 秒
            magnitude 面波震级 Ms
            R         震中距（km），R >= 0
        返回：
            (Y, lower, upper)
            Y      中值（PGA/PSA 单位 gal，PGV 单位 cm/s）
            lower  下限 = Y / 10^σ
            upper  上限 = Y * 10^σ

        公式：lg(Y) = A + B*M - C*lg(R + D*exp(E*M))
        """
        try:
            # 1. 获取该周期点的系数行
            coeff = self._get_coefficients(T)

            # 2. 提取系数（σ 是希腊字母列名）
            A1 = coeff["A1"]
            B1 = coeff["B1"]
            A2 = coeff["A2"]
            B2 = coeff["B2"]
            C = coeff["C"]
            D = coeff["D"]
            E = coeff["E"]
            sigma = coeff["σ"]

            # 3. 按震级分段选择 A、B（M < 6.5 用小震级段）
            if magnitude < 6.5:
                A = A1
                B = B1
            else:
                A = A2
                B = B2

            # 4. 等效近场距离项：D*exp(E*M)，防止 R→0 时对数发散
            R_log = np.log10(R + D * np.exp(E * magnitude))

            # 5. lg(Y) = A + B*M - C*lg(R + D*exp(E*M))
            lg_Y = A + B * magnitude - C * R_log

            # 6. 中值及 ±1σ 区间
            Y = 10**lg_Y
            lower = 10 ** (lg_Y - sigma)
            upper = 10 ** (lg_Y + sigma)

            return (Y, lower, upper)

        except Exception as e:
            print(f"计算失败: {str(e)}")
            return (None, None, None)

    def invert_R(self, T, magnitude, Y):
        """
        反算：由 (周期 T, 震级 magnitude, 地震动参数 Y) 求震中距 R 及 ±1σ

        参数：
            T         周期点：-1=PGA、-2=PGV、数值=PSA(T) 秒
            magnitude 面波震级 Ms
            Y         地震动参数值（PGA/PSA 单位 gal，PGV 单位 cm/s），Y > 0
        返回：
            (R_median, R_lower, R_upper)，单位 km

        公式推导：
            lg(Y) = A + B*M - C*lg(R + D*exp(E*M))
            => lg(R + D*exp(E*M)) = (A + B*M - lg(Y)) / C
            => R = 10**((A + B*M - lg(Y)) / C) - D*exp(E*M)

        ±1σ 区间：Y 更大 → 震中距更小；Y 更小 → 震中距更大。
        距离最小截断到 0.01 km。
        """
        try:
            # 1. 获取该周期点的系数行
            coeff = self._get_coefficients(T)

            # 2. 提取系数
            A1 = coeff["A1"]
            B1 = coeff["B1"]
            A2 = coeff["A2"]
            B2 = coeff["B2"]
            C = coeff["C"]
            D = coeff["D"]
            E = coeff["E"]
            sigma = coeff["σ"]

            # 3. 按震级分段选择 A、B
            if magnitude < 6.5:
                A = A1
                B = B1
            else:
                A = A2
                B = B2

            # 4. lg(Y)
            lg_Y = np.log10(Y)

            # 5. 等效近场距离项：D*exp(E*M)
            exp_term = D * np.exp(E * magnitude)

            # 6. 反解中值 R
            R_median = 10 ** ((A + B * magnitude - lg_Y) / C) - exp_term

            # 7. 置信区间
            # 下界: lg_Y + sigma（Y 更大 → R 更小）
            lg_Y_upper = lg_Y + sigma
            R_lower = 10 ** ((A + B * magnitude - lg_Y_upper) / C) - exp_term

            # 上界: lg_Y - sigma（Y 更小 → R 更大）
            lg_Y_lower = lg_Y - sigma
            R_upper = 10 ** ((A + B * magnitude - lg_Y_lower) / C) - exp_term

            # 确保距离不为负
            R_median = max(R_median, 0.01)
            R_lower = max(R_lower, 0.01)
            R_upper = max(R_upper, 0.01)

            return (R_median, R_lower, R_upper)

        except Exception as e:
            print(f"反解R失败: {str(e)}")
            return (None, None, None)


if __name__ == "__main__":
    # ==================== CEA2019 自检示例 ====================
    # 直接运行本文件即可看到下面的测试结果，覆盖：正算、反算、
    # 正反互验、周期插值、长短轴对比、震级分段。

    # ---------- 1) 正算：不同周期点的地震动参数（中值 ±1σ）----------
    cal_long = CEA2019(region="新疆", axis="长轴")
    cal_short = CEA2019(region="新疆", axis="短轴")

    # %%
    print("=== 正算：青藏区长轴，Ms=7.5，R=100 km ===")
    for T, name in [
        (-1, "PGA"),
        (-2, "PGV"),
        (0.3, "PSA(0.3s)"),
        (6, "PSA(6s)"),
    ]:
        Y, lo, up = cal_long.calculate(T, 7.5, 100)
        unit = "cm/s" if T == -2 else "gal"
        print(f"{name:10s} = {Y:9.3f} {unit}  (±1σ: {lo:.3f} ~ {up:.3f})")

    # ---------- 2) 反算：由地震动参数求震中距 ----------
    print("\n=== 反算：PSA(1s) = 100 gal → 震中距 ===")
    R, r_lo, r_up = cal_long.invert_R(1.0, 7.5, 100.0)
    print(f"R = {R:.2f} km  (±1σ: {r_lo:.2f} ~ {r_up:.2f})")

    # ---------- 3) 正反互验：算出的 Y 反算回去应得到 R ----------
    print("\n=== 正反互验（PGA，Ms=7.5，R=83.6 km）===")
    Y, _, _ = cal_long.calculate(-1, 7.5, 83.6)
    R_back, _, _ = cal_long.invert_R(-1, 7.5, Y)
    print(f"正算 Y = {Y:.3f} gal → 反算 R = {R_back:.2f} km（应 ≈ 83.6）")

    # ---------- 4) 周期插值：0.45s 不在表里，在 0.4 和 0.5 之间插值 ----------
    print("\n=== 周期插值：PSA(0.45s)（0.4 与 0.5 之间线性插值系数）===")
    Y45, lo45, up45 = cal_long.calculate(0.45, 7.5, 100)
    print(f"PSA(0.45s) = {Y45:.3f} gal  (±1σ: {lo45:.3f} ~ {up45:.3f})")

    # ---------- 5) 长轴 vs 短轴（短轴衰减更快，同距离值更小）----------
    print("\n=== 长轴 vs 短轴（PGA，Ms=7.5，R=50 km）===")

    Yl, _, _ = cal_long.calculate(1, 7.5, 58.48)
    Ys, _, _ = cal_short.calculate(1, 7.5, 43.19)
    print(f"长轴 = {Yl:.2f} gal，短轴 = {Ys:.2f} gal")

    # ---------- 6) 震级分段：M<6.5 用 A1/B1，M≥6.5 用 A2/B2 ----------
    print("\n=== 震级分段（PGA，R=100 km）===")
    Y6, _, _ = cal_long.calculate(-1, 6.0, 100)
    Y7, _, _ = cal_long.calculate(-1, 7.0, 100)
    print(f"M=6.0 → {Y6:.2f} gal；M=7.0 → {Y7:.2f} gal")
