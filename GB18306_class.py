# -*- coding: utf-8 -*-
"""
GB 18306-2015 地震动参数衰减关系（俞言祥）
==========================================
本模块把 GB 18306-2015 的两套衰减关系放在同一个文件里，提供两个 class：

    1. GB18306_2015_IntensityCal —— 烈度衰减
        公式：I = A + B*Ms + C*lg(R + R0)
        正算 calculate()   ：由 (Ms, R) 求烈度 I
        反算 invert_R()    ：由 (烈度 I, Ms) 求震中距 R

    2. GB18306_2015_PGA_PGV_GMMs —— PGA / PGV 衰减
        公式：lgY = A + B*Ms + C*lg(R + D*exp(E*Ms))
        正算 calculate()   ：由 (Ms, R) 求 PGA(aE) 和 PGV(vE)
        反算 invert_R()    ：由 (地震动参数 Y, Ms) 求震中距 R
        （注意：Ms > 6.5 时取 A2/B2 系数，否则取 A1/B1）

----------------------------------------------------------------
公共约定
----------------------------------------------------------------
    区域：东部区 / 中部区 / 新疆区 / 青藏区
    轴向：长轴 / 短轴
    单位：
        aE  峰值加速度，gal（即 cm/s²）
        vE  峰值速度，cm/s
        I   烈度，无量纲
        R   震中距，km
    参数来源：
        烈度参数  ：GB 高孟潭宣贯教材 172 页
        PGA/PGV 参数：GB 18306-2015 附录 表4（aE）、表5（vE）

----------------------------------------------------------------
“±1σ”区间约定
----------------------------------------------------------------
    衰减关系给出的是对数域中值 + 对数标准差 σ：
    - calculate() 的 PGA/PGV 返回 (中值, 中值/10^σ, 中值*10^σ)；
    - invert_R() 用 Y±σ（烈度用 I±σ）反算距离的上下界。

----------------------------------------------------------------
用法示例
----------------------------------------------------------------
    from GB18306_2015_GMMs import (
        GB18306_2015_IntensityCal, GB18306_2015_PGA_PGV_GMMs,
    )

    # 1) 烈度：正算（已知震级、震中距 → 烈度）
    cal_i = GB18306_2015_IntensityCal()
    r1 = cal_i.calculate(M=7.5, R=100, region="青藏区", axis_type="长轴")

    # 2) 烈度：反算（已知烈度、震级 → 震中距）
    r2 = cal_i.invert_R(intensity=8.0, M=7.5, region="青藏区", axis_type="长轴")

    # 3) PGA / PGV：正算（返回两个三元组：中值/下限/上限）
    cal_g = GB18306_2015_PGA_PGV_GMMs()
    (aE, aE_lo, aE_up), (vE, vE_lo, vE_up) = \
        cal_g.calculate(7.5, 100, "青藏区", "长轴")

    # 4) PGA / PGV：反算（已知地震动参数、震级 → 震中距）
    R_med, R_lo, R_up = cal_g.invert_R_from_aE(100.0, 7.5, "青藏区", "长轴")
"""

import math

# 让中文提示在 Windows 命令行里正常显示
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class GB18306_2015_IntensityCal:
    """
    烈度衰减计算器（GB 18306-2015）

    功能：
        - calculate()   正算：由 (震级 M, 震中距 R) 求烈度 I（及 ±1σ 区间）
        - invert_R()    反算：由 (烈度 I, 震级 M) 求震中距 R（及 ±1σ 区间）
        - _validate_input()  参数校验（区域、轴向）

    衰减公式：
        I = A + B * M + C * lg(R + R0)
    反解公式（C < 0，烈度越高，对应震中距越小）：
        R = 10**((I - A - B*M) / C) - R0
    标准差 σ：置信区间按 I ± σ（烈度域）计算，再反算成距离域。

    参数表 _PARAMS：
        (区域, 轴向) -> (A, B, C, R0, σ)
        来源：GB 高孟潭宣贯教材 172 页。
    """

    # 参数数据库（包含标准差）
    # 键：(区域, 轴向)，值：(A, B, C, R0, sigma)
    # 来源：GB 高孟潭宣贯教材 172 页
    _PARAMS = {
        ("青藏区", "长轴"): (6.4580, 1.2746, -4.4709, 25, 0.6636),
        ("青藏区", "短轴"): (3.3682, 1.2746, -3.3119, 9, 0.6636),
        ("新疆区", "长轴"): (5.6018, 1.4347, -4.4899, 25, 0.5924),
        ("新疆区", "短轴"): (3.6113, 1.4347, -3.8477, 13, 0.5924),
        ("东部区", "长轴"): (5.7123, 1.3626, -4.2903, 25, 0.5826),
        ("东部区", "短轴"): (3.6588, 1.3626, -3.5406, 13, 0.5826),
        ("中部区", "长轴"): (5.8410, 1.0710, -3.6570, 15, 0.5200),
        ("中部区", "短轴"): (3.9440, 1.0710, -2.8450, 7, 0.5200),
    }

    def calculate(self, M: float, R: float, region: str, axis_type: str) -> dict:
        """
        正算：由 (震级 M, 震中距 R) 求烈度

        参数：
            M          面波震级 Ms
            R          震中距（km），R >= 0
            region     区域：东部区 / 中部区 / 新疆区 / 青藏区
            axis_type  轴向：长轴 / 短轴

        返回：
            dict：
                mean          烈度中值：A + B*M + C*lg(R + R0)
                std           对数标准差 σ
                upper_1sigma  mean + σ（烈度上限）
                lower_1sigma  mean - σ（烈度下限）
                input_params  本次输入参数（便于追溯）
        """
        # 参数验证（区域、轴向非法时抛 ValueError）
        self._validate_input(region, axis_type)

        # 获取参数
        A, B, C, R0, sigma = self._PARAMS[(region, axis_type)]

        # 核心计算：I = A + B*M + C*lg(R + R0)
        mean_val = A + B * M + C * math.log10(R + R0)

        return {
            "mean": mean_val,
            "std": sigma,
            "upper_1sigma": mean_val + sigma,
            "lower_1sigma": mean_val - sigma,
            "input_params": {
                "M": M,
                "R": R,
                "region": region,
                "axis_type": axis_type,
            },
        }

    def invert_R(self, intensity: float, M: float, region: str, axis_type: str) -> dict:
        """
        反算：由 (烈度 I, 震级 M) 求震中距 R

        公式推导：
            I = A + B*M + C*lg(R + R0)
            => (I - A - B*M) / C = lg(R + R0)
            => R = 10**((I - A - B*M) / C) - R0

        ±1σ 区间：
            烈度更大（I+σ）→ 震中距更小（R 下界）；
            烈度更小（I-σ）→ 震中距更大（R 上界）。
            距离最小截断到 0.1 km（防止反算出现负距离）。

        参数：
            intensity  烈度值
            M          面波震级
            region     区域
            axis_type  轴向（长轴 / 短轴）

        返回：
            dict：
                mean          震中距中值（km）
                std           标准差 σ
                upper_1sigma  震中距上限（km）
                lower_1sigma  震中距下限（km）
                input_params  本次输入参数
        """
        # 参数验证
        self._validate_input(region, axis_type)

        # 获取参数
        A, B, C, R0, sigma = self._PARAMS[(region, axis_type)]

        # 计算中值震中距
        log_term = (intensity - A - B * M) / C
        R_mean = 10**log_term - R0

        # 计算置信区间
        # 下界: intensity + sigma（烈度增大 → 震中距减小）
        log_term_lower = ((intensity + sigma) - A - B * M) / C
        R_lower = 10**log_term_lower - R0

        # 上界: intensity - sigma（烈度减小 → 震中距增大）
        log_term_upper = ((intensity - sigma) - A - B * M) / C
        R_upper = 10**log_term_upper - R0

        # 确保距离不为负
        R_mean = max(R_mean, 0.1)
        R_lower = max(R_lower, 0.1)
        R_upper = max(R_upper, 0.1)

        return {
            "mean": R_mean,
            "std": sigma,
            "upper_1sigma": R_upper,
            "lower_1sigma": R_lower,
            "input_params": {
                "intensity": intensity,
                "M": M,
                "region": region,
                "axis_type": axis_type,
            },
        }

    def _validate_input(self, region: str, axis_type: str):
        """输入参数验证：区域 + 轴向必须是 _PARAMS 里的组合，否则报错并列出可用项"""
        if (region, axis_type) not in self._PARAMS:
            available = "\n".join([f"{k[0]} ({k[1]})" for k in self._PARAMS.keys()])
            raise ValueError(f"无效参数！可用组合：\n{available}")


class GB18306_2015_PGA_PGV_GMMs:
    """
    峰值加速度 / 峰值速度衰减计算器（GB 18306-2015，俞言祥）

    功能：
        - calculate()            正算：由 (M, R) 求 PGA(aE)、PGV(vE) 及 ±1σ
        - invert_R()             反算：由地震动参数 Y 求震中距 R
        - invert_R_from_aE()     反算便捷封装（Y 为 aE）
        - invert_R_from_vE()     反算便捷封装（Y 为 vE）
        - _validate_input()      参数校验
        - _get_params()          按震级分段取参数

    衰减公式（长轴、短轴同形）：
        lgY = A + B*M + C*lg(R + D*exp(E*M))
        Y   = 10**lgY
    其中：
        Y      aE（gal）或 vE（cm/s）的中值
        A,B    截距、震级系数（随震级分段：Ms>6.5 用 A2/B2，否则 A1/B1）
        C      距离衰减系数（不分段）
        D,E    距离饱和项参数：D*exp(E*M) 相当于“等效近场距离”
        σ      对数标准差；±1σ = 中值/10^σ ~ 中值*10^σ

    参数表：
        aE_table（表4，峰值加速度，输出单位 gal）
        vE_table（表5，峰值速度，单位 cm/s）
        结构：区域 -> 轴向 -> {A1, B1, A2, B2, C, D, E, sigma}
        来源：GB 18306-2015 附录。
    """

    VALID_REGIONS = ["东部区", "中部区", "新疆区", "青藏区"]
    VALID_AXES = ["长轴", "短轴"]

    def __init__(self):
        self._load_parameters()

    def _load_parameters(self):
        """加载全部区域参数（GB 18306-2015 附录 表4、表5）"""
        # 表4：aE（峰值加速度）衰减参数，输出单位 gal（cm/s²）
        self.aE_table = {
            "东部区": {
                "长轴": {
                    "A1": 1.979,
                    "B1": 0.671,
                    "A2": 3.533,
                    "B2": 0.432,
                    "C": -2.315,
                    "D": 2.088,
                    "E": 0.399,
                    "sigma": 0.236,
                },
                "短轴": {
                    "A1": 1.176,
                    "B1": 0.660,
                    "A2": 2.753,
                    "B2": 0.418,
                    "C": -2.004,
                    "D": 0.944,
                    "E": 0.447,
                    "sigma": 0.236,
                },
            },
            "中部区": {
                "长轴": {
                    "A1": 2.417,
                    "B1": 0.498,
                    "A2": 3.706,
                    "B2": 0.298,
                    "C": -2.079,
                    "D": 2.802,
                    "E": 0.295,
                    "sigma": 0.236,
                },
                "短轴": {
                    "A1": 1.715,
                    "B1": 0.471,
                    "A2": 2.690,
                    "B2": 0.321,
                    "C": -1.723,
                    "D": 1.295,
                    "E": 0.331,
                    "sigma": 0.236,
                },
            },
            "新疆区": {
                "长轴": {
                    "A1": 1.791,
                    "B1": 0.720,
                    "A2": 3.403,
                    "B2": 0.472,
                    "C": -2.389,
                    "D": 1.772,
                    "E": 0.424,
                    "sigma": 0.236,
                },
                "短轴": {
                    "A1": 0.983,
                    "B1": 0.713,
                    "A2": 2.610,
                    "B2": 0.463,
                    "C": -2.118,
                    "D": 0.825,
                    "E": 0.465,
                    "sigma": 0.236,
                },
            },
            "青藏区": {
                "长轴": {
                    "A1": 2.387,
                    "B1": 0.645,
                    "A2": 3.807,
                    "B2": 0.411,
                    "C": -2.416,
                    "D": 2.647,
                    "E": 0.366,
                    "sigma": 0.236,
                },
                "短轴": {
                    "A1": 1.003,
                    "B1": 0.609,
                    "A2": 2.457,
                    "B2": 0.388,
                    "C": -1.854,
                    "D": 0.612,
                    "E": 0.457,
                    "sigma": 0.236,
                },
            },
        }

        # 表5：vE（峰值速度）衰减参数，单位 cm/s
        self.vE_table = {
            "东部区": {
                "长轴": {
                    "A1": -0.363,
                    "B1": 0.791,
                    "A2": 1.437,
                    "B2": 0.513,
                    "C": -2.103,
                    "D": 2.088,
                    "E": 0.399,
                    "sigma": 0.271,
                },
                "短轴": {
                    "A1": -1.147,
                    "B1": 0.788,
                    "A2": 0.712,
                    "B2": 0.502,
                    "C": -1.825,
                    "D": 0.944,
                    "E": 0.447,
                    "sigma": 0.271,
                },
            },
            "中部区": {
                "长轴": {
                    "A1": 0.093,
                    "B1": 0.621,
                    "A2": 1.640,
                    "B2": 0.382,
                    "C": -1.889,
                    "D": 2.802,
                    "E": 0.295,
                    "sigma": 0.271,
                },
                "短轴": {
                    "A1": -0.589,
                    "B1": 0.601,
                    "A2": 0.671,
                    "B2": 0.407,
                    "C": -1.599,
                    "D": 1.295,
                    "E": 0.331,
                    "sigma": 0.271,
                },
            },
            "新疆区": {
                "长轴": {
                    "A1": -0.547,
                    "B1": 0.840,
                    "A2": 1.310,
                    "B2": 0.554,
                    "C": -2.181,
                    "D": 1.772,
                    "E": 0.424,
                    "sigma": 0.271,
                },
                "短轴": {
                    "A1": -1.351,
                    "B1": 0.843,
                    "A2": 0.569,
                    "B2": 0.549,
                    "C": -1.945,
                    "D": 0.825,
                    "E": 0.465,
                    "sigma": 0.271,
                },
            },
            "青藏区": {
                "长轴": {
                    "A1": -0.064,
                    "B1": 0.766,
                    "A2": 1.714,
                    "B2": 0.491,
                    "C": -2.205,
                    "D": 2.647,
                    "E": 0.366,
                    "sigma": 0.271,
                },
                "短轴": {
                    "A1": -1.301,
                    "B1": 0.741,
                    "A2": 0.443,
                    "B2": 0.474,
                    "C": -1.696,
                    "D": 0.612,
                    "E": 0.457,
                    "sigma": 0.271,
                },
            },
        }

    def _validate_input(self, region, axis):
        """输入参数校验：区域和轴向必须合法，否则报错"""
        if region not in self.VALID_REGIONS:
            raise ValueError(f"无效区域 '{region}'，可选区域：{self.VALID_REGIONS}")
        if axis not in self.VALID_AXES:
            raise ValueError(f"无效轴向 '{axis}'，可选轴向：{self.VALID_AXES}")

    def _get_params(self, M, region, axis, param_type):
        """
        按需取回一组衰减参数（含校验与震级分段）

        震级分段：M > 6.5 时取 A2/B2（大震级段），否则取 A1/B1；
        C、D、E、σ 不分段。

        参数：
            param_type  "aE"（用 aE_table）或 "vE"（用 vE_table）
        返回：
            dict：{A, B, C, D, E, sigma}
        """
        self._validate_input(region, axis)

        table = self.aE_table if param_type == "aE" else self.vE_table
        params = table[region][axis]

        # 选择震级分段系数
        suffix = "2" if M > 6.5 else "1"
        return {
            "A": params[f"A{suffix}"],
            "B": params[f"B{suffix}"],
            "C": params["C"],
            "D": params["D"],
            "E": params["E"],
            "sigma": params["sigma"],
        }

    def calculate(self, M, R, region, axis):
        """
        正算：由 (震级 M, 震中距 R) 求 aE、vE 的中值及 ±1σ 区间

        公式（长轴/短轴同形）：
            lgY = A + B*M + C*lg(R + D*exp(E*M))
            Y   = 10**lgY

        参数：
            M      面波震级 Ms
            R      震中距（km），R >= 0
            region 区域：东部区 / 中部区 / 新疆区 / 青藏区
            axis   轴向：长轴 / 短轴

        返回：
            (aE, vE) 两个三元组，每个三元组为 (中值, 下限, 上限)：
                aE 单位 gal（cm/s²），vE 单位 cm/s
                下限 = 中值 / 10^σ，上限 = 中值 * 10^σ
        """
        if R < 0:
            raise ValueError("震中距R必须≥0")

        def _compute(param_type):
            # 取参数（含震级分段）
            params = self._get_params(M, region, axis, param_type)
            # 等效近场距离项：D*exp(E*M)，防止 R→0 时对数发散
            log_term = math.log10(R + params["D"] * math.exp(params["E"] * M))
            # lgY = A + B*M + C*lg(R + D*exp(E*M))
            lgY = params["A"] + params["B"] * M + params["C"] * log_term
            median = 10**lgY                      # 中值
            delta = 10 ** params["sigma"]         # ±1σ 倍数
            return (median, median / delta, median * delta)

        return _compute("aE"), _compute("vE")

    def invert_R(self, Y, M, region, axis, param_type):
        """
        反算：由地震动参数值 Y 求震中距 R（中值及 ±1σ）

        公式推导：
            lgY = A + B*M + C*lg(R + D*exp(E*M))
            => lg(R + D*exp(E*M)) = (lgY - A - B*M) / C
            => R = 10**((lgY - A - B*M) / C) - D*exp(E*M)

        ±1σ 区间：
            Y 更大（Y*10^σ）→ 震中距更小（R 下界）；
            Y 更小（Y/10^σ）→ 震中距更大（R 上界）。
            距离最小截断到 0.01 km。

        参数：
            Y          地震动参数值（aE 或 vE，必须 > 0）
            M          面波震级 Ms
            region     区域
            axis       轴向（长轴 / 短轴）
            param_type "aE" 或 "vE"

        返回：
            (R中值, R下限, R上限)，单位 km
        """
        if Y <= 0:
            raise ValueError("地震动参数Y必须大于0")

        # 1. 获取参数（含震级分段）
        params = self._get_params(M, region, axis, param_type)

        # 2. 等效近场距离项：D*exp(E*M)
        exp_term = params["D"] * math.exp(params["E"] * M)

        # 3. lgY
        lgY = math.log10(Y)

        # 4. lg(R + D*exp(E*M)) = (lgY - A - B*M) / C
        log_term = (lgY - params["A"] - params["B"] * M) / params["C"]

        # 5. 中值 R
        R_median = 10**log_term - exp_term

        # 6. 置信区间
        # 下界: lgY + sigma（Y 更大 → R 更小）
        lgY_upper = lgY + params["sigma"]
        log_term_lower = (lgY_upper - params["A"] - params["B"] * M) / params["C"]
        R_lower = 10**log_term_lower - exp_term

        # 上界: lgY - sigma（Y 更小 → R 更大）
        lgY_lower = lgY - params["sigma"]
        log_term_upper = (lgY_lower - params["A"] - params["B"] * M) / params["C"]
        R_upper = 10**log_term_upper - exp_term

        # 确保距离不为负
        R_median = max(R_median, 0.01)
        R_lower = max(R_lower, 0.01)
        R_upper = max(R_upper, 0.01)

        return R_median, R_lower, R_upper

    def invert_R_from_aE(self, aE, M, region, axis):
        """
        反算便捷封装：由 aE（gal）求震中距 R
        等价于 self.invert_R(aE, M, region, axis, "aE")
        """
        return self.invert_R(aE, M, region, axis, "aE")

    def invert_R_from_vE(self, vE, M, region, axis):
        """
        反算便捷封装：由 vE（cm/s）求震中距 R
        等价于 self.invert_R(vE, M, region, axis, "vE")
        """
        return self.invert_R(vE, M, region, axis, "vE")


def me():
    pass


if __name__ == "__main__":
    # ---------- 自检示例：两个 class 的正算 / 反算 ----------
    cal_i = GB18306_2015_IntensityCal()
    res_i = cal_i.calculate(M=7.5, R=20.85, region="青藏区", axis_type="长轴")
    print("烈度正算：", {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in res_i.items()})
    res_ri = cal_i.invert_R(intensity=8.59, M=7.5, region="青藏区", axis_type="长轴")
    print("烈度反算 R(mean) =", round(res_ri["mean"], 2), "km")

    cal_g = GB18306_2015_PGA_PGV_GMMs()
    (aE, aE_lo, aE_up), (vE, vE_lo, vE_up) = \
        cal_g.calculate(7.5, 26.09, "青藏区", "长轴")
    print(f"PGA/PGV 正算：aE = {aE:.2f} ({aE_lo:.2f}~{aE_up:.2f}) gal, "
          f"vE = {vE:.2f} ({vE_lo:.2f}~{vE_up:.2f}) cm/s")
    R_med, R_lo, R_up = cal_g.invert_R_from_aE(297.27, 7.5, "青藏区", "长轴")
    print(f"PGA 反算 R = {R_med:.2f} ({R_lo:.2f}~{R_up:.2f}) km")
