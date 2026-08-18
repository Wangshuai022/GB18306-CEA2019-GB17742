"""GB18306 宏观震中反演（CB14 场地修正到 Vs30=500 m/s）。

流程：读取台站实际 Vs30 → 调用 CB14_site_correct.py 反解 A1100 → 将参与
反演及绘图的 PGA/PGV 观测统一到 Vs30=500 m/s → 调用原 GB18306
断层约束宏观震中反演。原始观测、修正系数和修正值均保留在结果审计表中。
"""

from __future__ import annotations

import os

from GB18306_epicenter_inversion import invert_epicenter_gb18306
from GB18306_vs_Obs import plot_gb18306_vs_obs
from Vs30_site_correction import (
    DEFAULT_REFERENCE_VS30,
    DEFAULT_VS30_PATH,
    _columns_for_period,
    _read_observation_table,
    attach_site_audit_to_result,
    correct_observations_to_reference_vs30,
    prepare_site_plot_observations,
)


def _gb_periods_to_correct(data, mode, plot_gmims):
    """读取观测表并汇总反演和绘图需要进行场地修正的 PGA/PGV 周期。"""
    periods = {
        "pga": [-1.0],
        "pgv": [-2.0],
        "pga_pgv": [-1.0, -2.0],
        "intensity": [],
    }.get(mode)
    if periods is None:
        raise ValueError("mode 必须为 pga / pgv / pga_pgv / intensity")
    raw = _read_observation_table(data)
    if plot_gmims is None:
        for period in (-1.0, -2.0):
            if _columns_for_period(raw, period) and period not in periods:
                periods.append(period)
    else:
        for item in plot_gmims:
            if isinstance(item, str):
                if item == "PGA":
                    period = -1.0
                elif item == "PGV":
                    period = -2.0
                elif item == "Intensity":
                    continue
                else:
                    raise ValueError(f"GB18306 不支持绘图参数：{item!r}")
            else:
                value = float(item)
                period = -1.0 if value == 0.0 else value
            if period not in (-1.0, -2.0):
                raise ValueError("GB18306 仅支持 PGA、PGV 和 Intensity")
            if period not in periods:
                periods.append(period)
    return raw, periods


def invert_epicenter_gb18306_vs30(
    data,
    Ms,
    region,
    hypo,
    strike,
    dip,
    rake,
    Mw=None,
    eq_type="板间",
    mode="pga_pgv",
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
    plot_GMIMs=None,
    vs30_path=DEFAULT_VS30_PATH,
    reference_vs30=DEFAULT_REFERENCE_VS30,
    cb14_region="CH",
    use_basin=False,
    vs30_chunksize=1_000_000,
    plot_observations="corrected",
    verbose=True,
):
    """将观测统一到目标 Vs30 后反演 GB18306 最优宏观震中。

    反演始终使用 CB14 修正后的 PGA/PGV。``plot_observations`` 只控制图上的
    观测点：默认与反演一致；也可显示原始场地观测，用于观察场地修正的影响。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        台站观测表。必须包含 ``Sta_ID``、经纬度，以及所选反演参数的观测列。
        水平向列优先级为 RotD50、H、有效值 RotD50、有效值 H。
    Ms : float
        面波震级，供 GB18306 衰减关系使用。
    region : str
        GB18306 分区：``青藏区``、``新疆区``、``东部区`` 或 ``中部区``。
    hypo : tuple(float, float, float)
        初始破裂点 ``(经度, 纬度, 深度km)``，同时用于生成默认断层网格。
    strike, dip, rake : float
        断层走向、倾角和滑移角，单位均为度。
    Mw : float or None, default None
        Leonard2014 定标率使用的矩震级；None 时取 ``Ms``。
    eq_type : str, default "板间"
        Leonard2014/SMD 修正定标率的地震类型。
    mode : {"pga", "pgv", "pga_pgv", "intensity"}
        参与震中反演的参数组合。Vs30 版本不支持纯 ``intensity`` 反演。
    max_dist : float, default 200
        参与反演和图中近场分组的最大等效椭圆距离，单位 km。
    extent : float, default 400
        衰减场计算和绘图的最大距离，单位 km。
    dx, dy : float, default 0.5
        自动断层网格沿走向、倾向的目标子断层尺寸，单位 km。
    shypo, dhypo : float or None
        震源在断层面上的沿走向、倾向位置，单位 km；默认分别为 0 和 0.57W。
    local_refine : float, default 0.1
        最优网格点附近连续精化半宽，单位经纬度；0 表示不精化。
    outpath : str or None
        逐台站反演与场地修正审计表路径；None 表示不导出。
    plot_path : str or None
        预测—观测综合图路径；None 表示不绘图。
    true_epi : tuple(float, float) or None
        已知震中，仅用于计算反演位置误差，不参与优化。
    fault_lon_mat, fault_lat_mat : array-like or None
        可选二维断层网格。两者同时给出时跳过自动 Leonard2014 网格生成。
    plot_GMIMs : sequence or None
        绘图参数，例如 ``[-1, -2, "Intensity"]``；None 时自动选择。
    vs30_path : str or os.PathLike
        中国 Vs30 网格 CSV 路径。
    reference_vs30 : float, default 500
        GB18306 模型参考场地 Vs30，单位 m/s。
    cb14_region : str, default "CH"
        传给 ``CB14_site_correct.py`` 的区域代码。
    use_basin : bool, default False
        是否启用 CB14 Z2.5 盆地项。没有可靠 Z2.5 时应保持 False。
    vs30_chunksize : int, default 1000000
        大型 Vs30 CSV 的分块读取行数。
    plot_observations : {"corrected", "raw"}, default "corrected"
        ``corrected`` 绘制参考场地观测；``raw`` 只在图中恢复原始观测。
        无论选择哪一种，震中反演和 chi2 始终使用参考场地观测。
    verbose : bool, default True
        是否打印 Vs30 查询、断层网格和反演摘要。

    Returns
    -------
    dict
        基础反演结果，并增加 ``site_correction``（逐台站审计表）、
        ``corrected_observations``、``site_reference_vs30``、``site_model``
        和 ``plot_observations``。``epicenter`` 为最优宏观震中经纬度。

    Raises
    ------
    ValueError
        参数组合、绘图模式或观测列不合法，或选择纯烈度反演。
    """
    raw, periods = _gb_periods_to_correct(data, mode, plot_GMIMs)
    if not periods:
        raise ValueError(
            "纯 intensity 不涉及 Vs30 场地修正；请直接使用 "
            "GB18306_epicenter_inversion.py"
        )
    corrected, audit = correct_observations_to_reference_vs30(
        raw,
        periods,
        vs30_path=vs30_path,
        reference_vs30=reference_vs30,
        region_name=cb14_region,
        use_basin=use_basin,
        chunksize=vs30_chunksize,
        verbose=verbose,
    )
    result = invert_epicenter_gb18306(
        data=corrected,
        Ms=Ms,
        region=region,
        hypo=hypo,
        strike=strike,
        dip=dip,
        rake=rake,
        Mw=Mw,
        eq_type=eq_type,
        mode=mode,
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
        plot_GMIMs=plot_GMIMs,
        verbose=verbose,
    )
    result = attach_site_audit_to_result(
        result, audit, corrected, outpath=outpath
    )
    plot_data = prepare_site_plot_observations(corrected, plot_observations)
    result["plot_observations"] = plot_data.attrs["site_plot_observations"]
    if plot_path:
        plot_gb18306_vs_obs(
            data=plot_data,
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
            outpath=plot_path,
        )
    return result


if __name__ == "__main__":
    os.makedirs("Test_output", exist_ok=True)
    result = invert_epicenter_gb18306_vs30(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        Ms=6.8,
        Mw=7.0,
        region="青藏区",
        hypo=(87.45, 28.50, 10.0),
        strike=187.0,
        dip=49.0,
        rake=-78.0,
        mode="pga_pgv",
        plot_GMIMs=[-1, -2, "Intensity"],
        true_epi=(87.45, 28.50),
        plot_observations="corrected",  # 改为 "raw" 时只切换图中观测值
        outpath="Test_output/GB18306_epicenter_inversion_Vs30_500.txt",
        plot_path="Test_output/GB18306_epicenter_inversion_Vs30_500.png",
    )
    print(
        "Vs30=500 m/s 最优宏观震中：",
        result["epicenter"],
        "reduced chi2 =",
        f"{result['reduced_chi2']:.3f}",
    )
