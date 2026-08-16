# -*- coding: utf-8 -*-
"""
GB/T 17742-2020 中国地震烈度表 —— 仪器烈度换算
================================================
把地震动记录换算成烈度，需要输入两个量：
    PGA ：峰值加速度，单位 gal（1 gal = 1 cm/s²）
    PGV ：峰值速度，单位 cm/s

换算公式（国标给出的两个分量公式）：
    加速度分量：I_A = 3.17 * log10(PGA / 100) + 6.59
    速度分量：  I_V = 3 * log10(PGV / 100) + 9.77

怎么合并成最终烈度（国标规则）：
    1) 如果 I_V >= 6 并且 I_A >= 6 ：烈度 = I_V（最多 12 度）
    2) 其他情况                  ：烈度 = (I_A + I_V) / 2（最少 1 度）
    3) 结果保留 1 位小数

注意单位：
    输入必须是 gal（PGA）和 cm/s（PGV）。
    如果手里是 m/s² 或 m/s，先乘 100 再传进来。

用法：
    from GB17742_class import GB17742_2020_Cal_instrument_intensity as Cal
    I = Cal.cal_Intensity(100, 10)                    # 单个点
    I = Cal.cal_Intensity_matrix([100, 200], [10, 20])  # 多个点
    I = Cal.cal_Intensity_matrix_PGV([10, 20])        # 只有 PGV
"""

import numpy as np

# 让中文提示在 Windows 命令行里正常显示
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class GB17742_2020_Cal_instrument_intensity:
    """
    烈度换算器，一共三个方法：
        cal_Intensity(PGA0, PGV0)         单个点：PGA + PGV → 烈度
        cal_Intensity_matrix(PGA0, PGV0)  多个点：PGA + PGV → 烈度（数组也行）
        cal_Intensity_matrix_PGV(PGV0)    只有 PGV 时：PGV → 烈度

    用法：直接拿类名调用，不用创建对象，例如：
        GB17742_2020_Cal_instrument_intensity.cal_Intensity(100, 10)
    """

    def cal_Intensity(PGA0, PGV0):
        """
        单个台站：输入 PGA（gal）、PGV（cm/s），输出烈度（保留 1 位小数）
        """
        # 用 PGA 算一个烈度分量
        I_A = 3.17 * np.log10(PGA0 / 100) + 6.59
        # 用 PGV 算一个烈度分量
        I_V = 3 * np.log10(PGV0 / 100) + 9.77

        # 国标规则：两个分量都 >= 6 度时，用速度分量（最多 12 度）
        if I_V >= 6 and I_A >= 6:
            I0 = min(I_V, 12)
        else:
            # 否则取两个分量的平均
            I0 = (I_V + I_A) / 2
            # 平均结果最低 1 度
            I0 = max(1, I0)

        # 保留 1 位小数
        I10 = round(I0, 1)
        return I10

    def cal_Intensity_matrix(PGA0, PGV0):
        """
        多个台站（数组/矩阵）：输入 PGA（gal）、PGV（cm/s），输出烈度数组
        算法和 cal_Intensity 完全一样，只是用 numpy 一次算完，速度快
        """
        # 转成 numpy 数组（传标量、列表、矩阵都可以）
        PGA0 = np.asarray(PGA0)
        PGV0 = np.asarray(PGV0)

        # 用 PGA 算烈度分量
        I_A = 3.17 * np.log10(PGA0 / 100) + 6.59
        # 用 PGV 算烈度分量
        I_V = 3 * np.log10(PGV0 / 100) + 9.77

        # 国标规则：找出"两个分量都 >= 6 度"的位置
        condition = (I_V >= 6) & (I_A >= 6)

        # 满足条件的位置取 min(I_V, 12)，不满足的位置取平均
        I0 = np.where(
            condition,
            np.minimum(I_V, 12),   # 两个都高：用速度分量，最多 12 度
            (I_V + I_A) / 2,       # 否则：取平均
        )

        # 平均结果最低 1 度
        I0 = np.maximum(I0, 1)

        # 保留 1 位小数
        return np.round(I0, 1)

    def cal_Intensity_matrix_PGV(PGV0):
        """
        只有 PGV（cm/s）时用：PGV → 烈度（保留 1 位小数）
        公式：I = 3 * log10(PGV / 100) + 9.77，范围 1~12 度
        """
        # 转成 numpy 数组
        PGV0 = np.asarray(PGV0)

        # 用 PGV 算烈度
        I_V = 3 * np.log10(PGV0 / 100) + 9.77

        # 最多 12 度
        I0 = np.minimum(I_V, 12)
        # 最少 1 度
        I0 = np.maximum(I0, 1)

        # 保留 1 位小数
        return np.round(I0, 1)


if __name__ == "__main__":
    # 自检：直接运行这个文件，看看输出对不对
    Cal = GB17742_2020_Cal_instrument_intensity
    print("单个点 (PGA=100, PGV=10)：", Cal.cal_Intensity(100, 10))
    print("多个点：", Cal.cal_Intensity_matrix([50, 100, 200], [5, 10, 20]))
    print("只有 PGV：", Cal.cal_Intensity_matrix_PGV([5, 10, 20]))
