"""
GB 18306-2015地震动参数衰减关系 俞言祥

给出国标长短轴地震动衰减关系 预测峰值速度、峰值加速度

输入：
M 面波震级
R 震中距 km
区域 ['东部区', '中部区', '新疆区', '青藏区']
主轴 "长轴"  "短轴"

输出：
aE (gal)  峰值加速度
vE (cm/s)  峰值速度

实例：
calc = GB_PGA_PGV_GMMs()
for M, R, region, axis in (7.5, 100, "青藏区", "长轴"):
    aE, vE = calc.calculate(M, R, region, axis)

"""

import math


class GB18306_2015_PGA_PGV_GMMs:
    VALID_REGIONS = ["东部区", "中部区", "新疆区", "青藏区"]
    VALID_AXES = ["长轴", "短轴"]

    def __init__(self):
        self._load_parameters()

    def _load_parameters(self):
        """加载全部区域参数（完整版表4、表5数据）"""
        # 表4：aε衰减参数（单位：g）
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

        # 表5：vε衰减参数（单位：cm/s）
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
        """输入参数校验"""
        if region not in self.VALID_REGIONS:
            raise ValueError(f"无效区域 '{region}'，可选区域：{self.VALID_REGIONS}")
        if axis not in self.VALID_AXES:
            raise ValueError(f"无效轴向 '{axis}'，可选轴向：{self.VALID_AXES}")

    def _get_params(self, M, region, axis, param_type):
        """获取衰减参数（含错误处理）"""
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
        计算地震动参数
        :param M: 面波震级 (Ms)
        :param R: 震中距 (km)，需≥0
        :param region: 区域名称（东部区/中部区/新疆区/青藏区）
        :param axis: 轴向（长轴/短轴）
        :return: (aE中值, aE下限, aE上限), (vE中值, vE下限, vE上限)
        """
        if R < 0:
            raise ValueError("震中距R必须≥0")

        def _compute(param_type):
            params = self._get_params(M, region, axis, param_type)
            log_term = math.log10(R + params["D"] * math.exp(params["E"] * M))
            lgY = params["A"] + params["B"] * M + params["C"] * log_term
            median = 10**lgY  # 注意"10"与"**"之间隐藏的零宽空格
            delta = 10 ** params["sigma"]
            return (median, median / delta, median * delta)

        return _compute("aE"), _compute("vE")

    def invert_R(self, Y, M, region, axis, param_type):
        """
        根据地震动参数反算震中距R
        :param Y: 地震动参数值（aE或vE）
        :param M: 面波震级 (Ms)
        :param region: 区域名称（东部区/中部区/新疆区/青藏区）
        :param axis: 轴向（长轴/短轴）
        :param param_type: 参数类型 ("aE" 或 "vE")
        :return: (R中值, R下限, R上限) 震中距 (km)
        """
        if Y <= 0:
            raise ValueError("地震动参数Y必须大于0")

        # 1. 获取参数
        params = self._get_params(M, region, axis, param_type)

        # 2. 计算指数项
        exp_term = params["D"] * math.exp(params["E"] * M)

        # 3. 计算lgY
        lgY = math.log10(Y)

        # 4. 计算中间项
        log_term = (lgY - params["A"] - params["B"] * M) / params["C"]

        # 5. 计算中值R
        R_median = 10**log_term - exp_term

        # 6. 计算置信区间
        # 下界: lgY + sigma (因为当Y增大时，R减小)
        lgY_upper = lgY + params["sigma"]
        log_term_lower = (lgY_upper - params["A"] - params["B"] * M) / params["C"]
        R_lower = 10**log_term_lower - exp_term

        # 上界: lgY - sigma (因为当Y减小时，R增大)
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
        根据aE反算震中距R
        :param aE: 加速度地震动参数 (gal)
        :param M: 面波震级 (Ms)
        :param region: 区域名称（东部区/中部区/新疆区/青藏区）
        :param axis: 轴向（长轴/短轴）
        :return: (R中值, R下限, R上限) 震中距 (km)
        """
        return self.invert_R(aE, M, region, axis, "aE")

    def invert_R_from_vE(self, vE, M, region, axis):
        """
        根据vE反算震中距R
        :param vE: 速度地震动参数 (cm/s)
        :param M: 面波震级 (Ms)
        :param region: 区域名称（东部区/中部区/新疆区/青藏区）
        :param axis: 轴向（长轴/短轴）
        :return: (R中值, R下限, R上限) 震中距 (km)
        """
        return self.invert_R(vE, M, region, axis, "vE")


# 示例使用
if __name__ == "__main__":
    calc = GB18306_2015_PGA_PGV_GMMs()

    # 测试案例
    test_cases = [
        (7.5, 100, "青藏区", "长轴"),
        (6.0, 30, "新疆区", "短轴"),
        (8.0, 200, "中部区", "长轴"),
        (8.0, 200, "南部区", "长轴"),
    ]

    for M, R, region, axis in test_cases:
        try:
            aE, vE = calc.calculate(M, R, region, axis)
            print(f"\n输入：M={M}, R={R}km, {region}-{axis}")
            print(f"  aE = {aE[0]:.3f}gal ({aE[1]:.3f}~{aE[2]:.3f})")
            print(f"  vE = {vE[0]:.3f}cm/s ({vE[1]:.3f}~{vE[2]:.3f})")
        except ValueError as e:
            print(f"错误：{e}")
