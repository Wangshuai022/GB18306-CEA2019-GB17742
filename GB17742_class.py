# 国标烈度转换关系：GB/T 17742-2020 仪器烈度换算（PGA / PGV → 烈度）
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


class GB17742_2020_Cal_instrument_intensity:
    """
    烈度换算器，一共三个方法：
        cal_Intensity(PGA0, PGV0)         单个点：PGA + PGV → 烈度
        cal_Intensity_matrix(PGA0, PGV0)  多个点：PGA + PGV → 烈度（数组也行）
        cal_Intensity_matrix_PGV(PGV0)    只有 PGV 时：PGV → 烈度

    用法：直接拿类名调用，不用创建对象，例如：
        GB17742_2020_Cal_instrument_intensity.cal_Intensity(100, 10)
    """

    @staticmethod
    def cal_Intensity(PGA0, PGV0):
        """把单个台站 PGA 和 PGV 换算为 GB/T 17742 仪器烈度。

        Parameters
        ----------
        PGA0 : float
            正峰值加速度，单位 gal（cm/s²）。
        PGV0 : float
            正峰值速度，单位 cm/s。

        Returns
        -------
        float
            限制在 1--12 度并保留一位小数的仪器烈度。

        Raises
        ------
        ValueError
            PGA 或 PGV 不是正有限数。
        """
        if not np.isfinite(PGA0) or PGA0 <= 0:
            raise ValueError("PGA 必须是正有限数")
        if not np.isfinite(PGV0) or PGV0 <= 0:
            raise ValueError("PGV 必须是正有限数")
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

    @staticmethod
    def cal_Intensity_matrix(PGA0, PGV0):
        """向量化换算多个台站或网格点的仪器烈度。

        ``PGA0``（gal）和 ``PGV0``（cm/s）可为标量、列表或形状可广播的
        ndarray。返回与广播结果同形状的 ndarray，NaN 会原样传播；有限值必须
        为正。算法与 ``cal_Intensity`` 完全相同。
        """
        # 转成 numpy 数组（传标量、列表、矩阵都可以）
        PGA0 = np.asarray(PGA0)
        PGV0 = np.asarray(PGV0)
        if np.any(np.isfinite(PGA0) & (PGA0 <= 0)):
            raise ValueError("PGA 中的有限值必须大于 0")
        if np.any(np.isfinite(PGV0) & (PGV0 <= 0)):
            raise ValueError("PGV 中的有限值必须大于 0")

        # 用 PGA 算烈度分量
        I_A = 3.17 * np.log10(PGA0 / 100) + 6.59
        # 用 PGV 算烈度分量
        I_V = 3 * np.log10(PGV0 / 100) + 9.77

        # 国标规则：找出"两个分量都 >= 6 度"的位置
        condition = (I_V >= 6) & (I_A >= 6)

        # 满足条件的位置取 min(I_V, 12)，不满足的位置取平均
        I0 = np.where(
            condition,
            np.minimum(I_V, 12),  # 两个都高：用速度分量，最多 12 度
            (I_V + I_A) / 2,  # 否则：取平均
        )

        # 平均结果最低 1 度
        I0 = np.maximum(I0, 1)

        # 保留 1 位小数
        return np.round(I0, 1)

    @staticmethod
    def cal_Intensity_matrix_PGV(PGV0):
        """仅用 PGV 分量估计仪器烈度。

        Parameters
        ----------
        PGV0 : array-like
            峰值速度，单位 cm/s；有限值必须为正。

        Returns
        -------
        numpy.ndarray
            ``I=3*log10(PGV/100)+9.77``，限制到 1--12 度并保留一位小数。
        """
        # 转成 numpy 数组
        PGV0 = np.asarray(PGV0)
        if np.any(np.isfinite(PGV0) & (PGV0 <= 0)):
            raise ValueError("PGV 中的有限值必须大于 0")

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
    print("单个点：", Cal.cal_Intensity(195, 17.7))
    print("多个点：", Cal.cal_Intensity_matrix([50, 100, 200], [5, 10, 20]))
    print(
        "只有 PGV：", Cal.cal_Intensity_matrix_PGV([5, 10, 20, 150, 200, 400])
    )
