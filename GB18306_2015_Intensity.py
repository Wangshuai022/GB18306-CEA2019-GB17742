import math


class GB18306_2015_IntensityCal:
    """
    烈度计算器（整合版）
    功能：计算地震烈度及其标准差置信区间
    """

    # 参数数据库（包含标准差）
    # 参数来源：GB 高孟潭宣贯教材  172页
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
        主计算方法
        :return: 包含所有计算结果的字典
        """
        # 参数验证
        self._validate_input(region, axis_type)

        # 获取参数
        A, B, C, R0, sigma = self._PARAMS[(region, axis_type)]

        # 核心计算
        mean_val = A + B * M + C * math.log10(R + R0)

        return {
            "mean": mean_val,
            "std": sigma,
            "upper_1sigma": mean_val + sigma,
            "lower_1sigma": mean_val - sigma,
            "input_params": {"M": M, "R": R, "region": region, "axis_type": axis_type},
        }

    def invert_R(self, intensity: float, M: float, region: str, axis_type: str) -> dict:
        """
        反解方法：根据烈度和震级计算震中距
        :param intensity: 烈度值
        :param M: 震级
        :param region: 区域名称
        :param axis_type: 轴向类型（长轴/短轴）
        :return: 包含震中距及其置信区间的字典
        """
        # 参数验证
        self._validate_input(region, axis_type)

        # 获取参数
        A, B, C, R0, sigma = self._PARAMS[(region, axis_type)]

        # 反解公式推导：
        # 原公式: intensity = A + B * M + C * log10(R + R0)
        # => (intensity - A - B * M) / C = log10(R + R0)
        # => R + R0 = 10**((intensity - A - B * M) / C)
        # => R = 10**((intensity - A - B * M) / C) - R0

        # 计算中值震中距
        log_term = (intensity - A - B * M) / C
        R_mean = 10**log_term - R0

        # 计算置信区间
        # 下界: intensity + sigma (烈度增大导致震中距减小)
        log_term_lower = ((intensity + sigma) - A - B * M) / C
        R_lower = 10**log_term_lower - R0

        # 上界: intensity - sigma (烈度减小导致震中距增大)
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
        """输入参数验证"""
        if (region, axis_type) not in self._PARAMS:
            available = "\n".join([f"{k[0]} ({k[1]})" for k in self._PARAMS.keys()])
            raise ValueError(f"无效参数！可用组合：\n{available}")


# 使用示例 ------------------------------------------------------------------------
if __name__ == "__main__":
    # 创建烈度计算器实例
    calculator = GB18306_2015_IntensityCal()

    # 计算案例
    result = calculator.calculate(M=7.5, R=123.5, region="青藏区", axis_type="长轴")

    print("计算结果：")
    print(f"- 均值烈度: {result['mean']:.2f}")
    print(f"- 标准差: ±{result['std']:.4f}")
    print(f"- 置信区间（±1σ）: {result['lower_1sigma']:.2f} ~ {result['upper_1sigma']:.2f}")

    # 示例2: 已知烈度和震级反算震中距
    intensity = result["mean"]  # 使用上一步计算的烈度值
    R_result = calculator.invert_R(intensity, M=7.5, region="青藏区", axis_type="长轴")

    # print(f"已知烈度 I={intensity:.2f}，震级 M={M}，区域={region}，轴向={axis_type}")
    print(
        f"  反算震中距: {R_result['mean']:.2f}km (置信区间（±1σ）: [{R_result['lower_1sigma']:.2f}, {R_result['upper_1sigma']:.2f}]km)"
    )

    # 验证反向计算精度
    print("\n验证计算精度:")
    print(f"  反算 R = {R_result['mean']:.2f}km")

    # 错误输入测试
    try:
        calculator.calculate(7.0, 20, "错误分区", "长轴")
    except ValueError as e:
        print("\n错误处理示例：", e)
