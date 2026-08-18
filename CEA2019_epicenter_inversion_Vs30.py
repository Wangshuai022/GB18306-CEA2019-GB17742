"""CEA2019 宏观震中反演（CB14 场地修正到 Vs30=500 m/s）。

参与反演和绘图的 PGA、PGV、PSA 观测先按台站实际 Vs30 统一到模型的
500 m/s 参考场地，再调用原 CEA2019 多参数联合宏观震中反演。
"""

from __future__ import annotations

import os

from CEA2019_epicenter_inversion import (
    invert_epicenter_cea2019,
    normalize_periods,
)
from CEA2019_vs_Obs import plot_cea2019_vs_obs
from Vs30_site_correction import (
    DEFAULT_REFERENCE_VS30,
    DEFAULT_VS30_PATH,
    attach_site_audit_to_result,
    correct_observations_to_reference_vs30,
    prepare_site_plot_observations,
)


def invert_epicenter_cea2019_vs30(
    data,
    Ms,
    region,
    hypo,
    strike,
    dip,
    rake,
    invert_GMIMs=(-1, -2),
    plot_GMIMs=None,
    Mw=None,
    eq_type="板间",
    max_dist=200.0,
    extent=400.0,
    dx=0.5,
    dy=0.5,
    shypo=None,
    dhypo=None,
    local_refine=0.1,
    outpath=None,
    plot_path=None,
    true_epi=None,
    fault_lon_mat=None,
    fault_lat_mat=None,
    vs30_path=DEFAULT_VS30_PATH,
    reference_vs30=DEFAULT_REFERENCE_VS30,
    cb14_region="CH",
    use_basin=False,
    vs30_chunksize=1_000_000,
    plot_observations="corrected",
    verbose=True,
):
    """将观测统一到目标 Vs30 后联合反演 CEA2019 最优宏观震中。

    ``invert_GMIMs`` 和 ``plot_GMIMs`` 涉及的 PGA、PGV、PSA 都先使用同一
    台站 A1100 计算 CB14 场地倍率。反演始终使用修正值；绘图可以通过
    ``plot_observations`` 在修正值与原始值之间切换。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        台站观测表。必须包含台站编号、经纬度和所需地震动参数。水平向列
        按 RotD50、H、有效值 RotD50、有效值 H 的顺序选择；PSA 优先 RotD50。
    Ms : float
        CEA2019 衰减关系使用的震级。
    region : str
        CEA2019 分区：``青藏区``、``新疆区``、``东部区`` 或 ``中部区``。
    hypo : tuple(float, float, float)
        初始破裂点 ``(经度, 纬度, 深度km)``。
    strike, dip, rake : float
        断层走向、倾角和滑移角，单位度。
    invert_GMIMs : sequence, default (-1, -2)
        参与反演的周期：-1=PGA、-2=PGV，正数为 PSA 周期（s）。
    plot_GMIMs : sequence or None
        图中显示的参数；None 时与 ``invert_GMIMs`` 相同。
    Mw : float or None, default None
        Leonard2014 定标率使用的矩震级；None 时取 ``Ms``。
    eq_type : str, default "板间"
        Leonard2014/SMD 修正定标率的地震类型。
    max_dist : float, default 200
        参与反演和图中近场分组的最大等效椭圆距离，单位 km。
    extent : float, default 400
        衰减场计算与绘图范围，单位 km。
    dx, dy : float, default 0.5
        自动断层网格目标子断层尺寸，单位 km。
    shypo, dhypo : float or None
        震源在断层面上的沿走向、倾向位置（km）；默认 0 和 0.57W。
    local_refine : float, default 0.1
        最优网格点附近连续精化半宽，单位经纬度；0 表示不精化。
    outpath : str or None
        逐台站反演和场地修正审计表路径；None 表示不导出。
    plot_path : str or None
        预测—观测综合图路径；None 表示不绘图。
    true_epi : tuple(float, float) or None
        已知震中，仅用于位置误差验证。
    fault_lon_mat, fault_lat_mat : array-like or None
        可选二维断层网格；同时提供时跳过自动网格生成。
    vs30_path : str or os.PathLike
        中国 Vs30 网格 CSV 路径。
    reference_vs30 : float, default 500
        CEA2019 模型参考场地 Vs30，单位 m/s。
    cb14_region : str, default "CH"
        传给 ``CB14_site_correct.py`` 的区域代码。
    use_basin : bool, default False
        是否启用 CB14 Z2.5 盆地项；缺少可靠 Z2.5 时保持 False。
    vs30_chunksize : int, default 1000000
        大型 Vs30 CSV 分块读取行数。
    plot_observations : {"corrected", "raw"}, default "corrected"
        ``corrected`` 绘制参考场地观测；``raw`` 仅在图中恢复原始观测。
        该开关不改变震中、chi2、反演残差或导出的反演表。
    verbose : bool, default True
        是否打印处理进度与反演摘要。

    Returns
    -------
    dict
        基础反演结果，并增加逐台站场地修正审计信息、修正后观测表以及
        ``plot_observations``。``epicenter`` 为最优宏观震中经纬度。

    Notes
    -----
    CB14 没有原生系数的正周期由用户的 ``CB14_site_correct.py`` 在线性
    周期轴插值。默认关闭盆地项，因为当前 Vs30 数据不提供实测 Z2.5。
    """
    invert_periods = normalize_periods(invert_GMIMs)
    plot_periods = (
        normalize_periods(plot_GMIMs)
        if plot_GMIMs is not None
        else list(invert_periods)
    )
    periods = list(invert_periods)
    for period in plot_periods:
        if period not in periods:
            periods.append(period)
    corrected, audit = correct_observations_to_reference_vs30(
        data,
        periods,
        vs30_path=vs30_path,
        reference_vs30=reference_vs30,
        region_name=cb14_region,
        use_basin=use_basin,
        chunksize=vs30_chunksize,
        verbose=verbose,
    )
    result = invert_epicenter_cea2019(
        data=corrected,
        Ms=Ms,
        region=region,
        hypo=hypo,
        strike=strike,
        dip=dip,
        rake=rake,
        invert_GMIMs=invert_periods,
        plot_GMIMs=plot_periods,
        Mw=Mw,
        eq_type=eq_type,
        max_dist=max_dist,
        extent=extent,
        dx=dx,
        dy=dy,
        shypo=shypo,
        dhypo=dhypo,
        local_refine=local_refine,
        outpath=None,
        plot_path=None,
        true_epi=true_epi,
        fault_lon_mat=fault_lon_mat,
        fault_lat_mat=fault_lat_mat,
        verbose=verbose,
    )
    result = attach_site_audit_to_result(
        result, audit, corrected, outpath=outpath
    )
    plot_data = prepare_site_plot_observations(corrected, plot_observations)
    result["plot_observations"] = plot_data.attrs["site_plot_observations"]
    if plot_path:
        plot_cea2019_vs_obs(
            data=corrected,
            macro_epicenter=result["epicenter"],
            initial_epicenter=(float(hypo[0]), float(hypo[1])),
            Ms=Ms,
            region=region,
            strike=strike,
            params=result["plot_GMIMs"],
            extent=extent,
            max_dist=max_dist,
            fault_lon_mat=result["mesh"]["lon_mat"],
            fault_lat_mat=result["mesh"]["lat_mat"],
            plot_observations=plot_observations,
            outpath=plot_path,
        )
    return result


if __name__ == "__main__":
    os.makedirs("Test_output", exist_ok=True)
    result = invert_epicenter_cea2019_vs30(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        Ms=6.8,
        Mw=7.0,
        region="青藏区",
        max_dist=400.0,
        hypo=(87.45, 28.50, 10.0),
        strike=187.0,
        dip=49.0,
        rake=-78.0,
        shypo=22,
        invert_GMIMs=[-2],
        plot_GMIMs=[-1, -2, 0.3, 1.0, 3.0, 6.0],
        true_epi=(87.45, 28.50),
        plot_observations="corrected",  # 改为 "raw" 时只切换图中观测值
        outpath="Test_output/CEA2019_epicenter_inversion_Vs30_500_pgv.txt",
        plot_path="Test_output/CEA2019_epicenter_inversion_Vs30_500_pgv.png",
    )
    print(
        "Vs30=500 m/s 最优宏观震中：",
        result["epicenter"],
        "reduced chi2 =",
        f"{result['reduced_chi2']:.3f}",
    )
