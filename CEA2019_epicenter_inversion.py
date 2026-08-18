"""
CEA2019 震中反演 —— 基于 CEA2019 椭圆衰减，断层网格约束反演范围
================================================================
已知震中周边台站的 PGA / PGV / PSA，用 CEA2019 衰减关系反演"最优震中"。

反演参数（周期点组合，任选若干）：
    -1 或 0   → PGA（gal）
    -2        → PGV（cm/s）
    数值周期  → PSA(T)，如 0.3、1、3、6（cm/s²）
    （CEA2019 无烈度参数）

例如：invert_GMIMs=(-2,) 单参数；(-1,-2) 双参数；(-1,-2,6) 三参数；
      (-1,-2,1,3) 四参数；(1,3) 双参数。
绘图参数 plot_GMIMs 与迭代参数分开，如迭代用 PGA+PGV、
绘图画 [-1,-2,0.3,1,3,6]。
目标函数 = 各周期点对数域残差按各自 σ 归一化的最小二乘 chi2 之和。

断层范围（候选震中）：
    1) 优先外部断层网格 fault_lon_mat / fault_lat_mat（二维矩阵，
       第一行=上缘、最后一行=下缘），提供时直接作为反演范围（跳过 L14）；
    2) 否则用 Leonard2014（SMD 修正版）中位 L/W + 矩形有限断层，
       shypo/dhypo 相对断层上缘（km），默认 None → 0 / 0.57*W；
    3) 反演震中严格限制在断层投影范围内（越界自动回退到网格最优）。

有效范围：长轴距/短轴距 ≤ max_dist（默认 200 km）的台站参与反演。

依赖：
    CEA2019_pre.py、CEA2019_vs_Obs.py、GB18306_epicenter_inversion.py（L14/网格）、
    Leonard2014_fitted_by_SMD_crust.py、mesh_single_rectangular_finite_fault.py

使用案例：
    from CEA2019_epicenter_inversion import invert_epicenter_cea2019
    res = invert_epicenter_cea2019(
        data="台站文件.txt",
        Ms=6.8, region="青藏区",
        hypo=(87.45, 28.5, 10.0),
        strike=187.0, dip=49.0, rake=-78.0,
        invert_GMIMs=(-1, -2),     # 迭代：PGA+PGV
        plot_GMIMs=[-1, -2, 0.3, 1, 3, 6],  # 绘图：PGA/PGV/PSA0.3/1/3/6
        outpath="Test_output/CEA2019_epicenter_inversion_stats.txt",
        plot_path="Test_output/CEA2019_epicenter_inversion.png",
    )
    print(res["epicenter"], res["chi2"])
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from CEA2019_pre import (
    _period_coeffs,
    _region_core,
    period_label,
    validate_periods,
)
from CEA2019_vs_Obs import (
    export_cea2019_vs_obs_txt,
    load_obs_data,
    plot_cea2019_vs_obs,
)
from ellipse_fields import CEA2019EllipseField
from GB18306_epicenter_inversion import fault_mesh_points

# ==================== 周期点参数规范 ====================


def normalize_periods(params):
    """
    输入参数 → 统一周期点列表：
        -1 / 0 / "PGA"  → -1（PGA）
        -2 / "PGV"      → -2（PGV）
        数值周期 / "PSA(T=0.30s)" → 周期（PSA）
    自动去重；CEA2019 无烈度，T > 6s 不支持。
    """
    out = []
    for p in params:
        if isinstance(p, str):
            s = p.strip()
            if s == "PGA":
                item = -1.0
            elif s == "PGV":
                item = -2.0
            elif s == "Intensity":
                raise ValueError("CEA2019 无烈度参数，不支持 'Intensity'")
            elif s.lower().startswith("psa"):
                item = float(s.split("(")[1].split("=")[1].rstrip("s)"))
            else:
                raise ValueError(f"无法识别的参数：{p!r}")
        else:
            t = float(p)
            item = -1.0 if t == 0.0 else t
        if float(item) > 6.0:
            raise ValueError(f"周期 {item:g}s > 6s，CEA2019 不外插，不支持")
        if item not in out:
            out.append(item)
    return validate_periods(out)


# ==================== CEA2019 椭圆场（向量化批预测） ====================


def _period_predict_many(
    T, M, region_core, lon, lat, strike, sta_lon, sta_lat, extent
):
    """
    某周期点 T 在 (n_cand) 个候选震中 × (n_sta) 个台站上的批量预测。
    返回 (value, a_eq, b_eq)，形状 (n_cand, n_sta)；场外为 NaN。
    """
    field = CEA2019EllipseField(region_core, M, extent=extent)
    return field.predict_period_many(T, lon, lat, strike, sta_lon, sta_lat)


def _chi2_batch(
    periods, M, rc, cand_lon, cand_lat, strike, sta, used_mask, extent
):
    """候选震中批量 chi2：各周期点对数残差按 σ 归一化后求和"""
    lg = lambda x: np.log10(np.maximum(x, 1e-9))
    used = used_mask[None, :]
    chi2 = np.zeros(cand_lon.size)
    for T in periods:
        label = period_label(T)
        val, _, _ = _period_predict_many(
            T,
            M,
            rc,
            cand_lon,
            cand_lat,
            strike,
            sta["lon"].values,
            sta["lat"].values,
            extent,
        )
        sigma = _period_coeffs(rc, "长轴", T, M)[3]
        in_field = np.isfinite(val)
        r = (lg(sta[label].values)[None, :] - np.log10(val)) / sigma
        ok = used & in_field
        chi2 = chi2 + np.where(ok, r * r, 0.0).sum(axis=1)
        chi2 = chi2 + 1000.0 * (used & ~in_field).sum(axis=1)
    return chi2


# ==================== 反演主函数 ====================


def invert_epicenter_cea2019(
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
    verbose=True,
):
    """
    CEA2019 震中反演主函数。

    明确输入：
        data     实测数据：文件路径或 DataFrame；观测列由 CEA2019_vs_Obs
                 统一按 RotD50 → H → 有效值 RotD50 → 有效值 H 选择
        Ms       震级（CEA2019 衰减用，M<6.5 与 ≥6.5 分段取系数）
        region   分区："青藏区"/"新疆区"/"东部区"/"中部区"
        hypo     震中 (经度, 纬度, 深度 km)
        strike / dip / rake  走向/倾角/滑移角（rake 判别 SS/RS/NS，供 L14）
        invert_GMIMs  迭代（反演）用的周期点组合，如 (-2,)、(-1,-2)、
                 (-1,-2,6)、(-1,-2,1,3)、(1,3)；缺省 (-1,-2) 即 PGA+PGV
        plot_GMIMs    绘图用的周期点组合（与迭代参数分开），
                 如 [-1,-2,0.3,1,3,6]；缺省 None → 与 invert_GMIMs 相同
        fault_lon_mat / fault_lat_mat  可选外部断层网格（二维矩阵），
                 提供时直接作为反演范围（跳过 L14）
        Mw / eq_type  L14 用（默认 Mw=Ms、板间）
        max_dist  有效范围（长轴距/短轴距 ≤ 该值，km，默认 200）
        extent    衰减场最大距离 km
        shypo / dhypo  相对断层上缘的沿走向/沿倾向位置（km），默认 None
                  → 0 / 0.57*W，可自定义（如 -10 / 15）
        local_refine  最优网格点附近连续精化半宽（°；0 = 仅用网格点）
        outpath / plot_path  统计 txt / 4×N 图输出路径（None = 不导出）
        true_epi  已知震中（仅验证）

    返回 dict：
        epicenter/lon/lat
            最优宏观震中；经纬度单位为度。
        chi2/reduced_chi2/degrees_of_freedom
            使用各参数模型 σ 归一化后的目标函数及自由度修正值。
        n_used/n_sta
            联合反演有效台站数与坐标有效的总台站数。
        table
            逐台站坐标、观测、预测、残差、椭圆距和参与标志。
        mesh
            断层网格、尺寸、震源位置和地表出露调整信息。
        observation_columns
            每个参数实际采用的原始观测列名。
        optimizer_success/optimizer_message/boundary_hit
            连续精化状态和最优点是否位于断层边界。

    注意：联合反演只使用所有 ``invert_GMIMs`` 均为正有限值且位于
    ``max_dist`` 范围内的台站。残差为 log10(观测/预测)，chi2 对每个参数
    按对应 σ 标准化；经纬度是待估的两个自由参数。
    """
    periods = normalize_periods(invert_GMIMs)
    plot_periods = (
        normalize_periods(plot_GMIMs) if plot_GMIMs is not None else periods
    )
    if not periods:
        raise ValueError(
            "invert_GMIMs 不能为空；支持 -1/0(PGA)、-2(PGV)、数值周期(PSA)"
        )
    if len(hypo) != 3:
        raise ValueError("hypo 必须为 (经度, 纬度, 深度km)")
    Mw = float(Mw) if Mw is not None else float(Ms)

    sta = load_obs_data(data, periods)
    lon0, lat0, depth0 = float(hypo[0]), float(hypo[1]), float(hypo[2])
    strike = float(strike) % 360.0
    rc = _region_core(region)

    # ---- 1. 断层网格：优先外部网格，未提供才用 L14 ----
    if (fault_lon_mat is None) != (fault_lat_mat is None):
        raise ValueError(
            "fault_lon_mat 与 fault_lat_mat 必须同时提供或同时省略"
        )
    if fault_lon_mat is not None:
        flon_in = np.asarray(fault_lon_mat, dtype=float)
        flat_in = np.asarray(fault_lat_mat, dtype=float)
        if flon_in.ndim != 2 or flat_in.ndim != 2:
            raise ValueError(
                "fault_lon_mat / fault_lat_mat 必须是二维网格矩阵"
            )
        if flon_in.shape != flat_in.shape:
            raise ValueError("fault_lon_mat 与 fault_lat_mat 形状必须一致")
        mesh = {
            "lon_mat": flon_in,
            "lat_mat": flat_in,
            "depth_mat": None,
            "L_km": None,
            "W_km": None,
            "mechanism": None,
            "fault_type": None,
            "shypo_km": None,
            "dhypo_km": None,
            "source": "external",
        }
        if verbose:
            print(
                f"[断层范围] 使用外部网格：{flon_in.shape[0]} 行 × "
                f"{flon_in.shape[1]} 列（跳过 L14 定标率）"
            )
    else:
        mesh = fault_mesh_points(
            lon0,
            lat0,
            depth0,
            strike,
            float(dip),
            float(rake),
            Mw,
            eq_type=eq_type,
            shypo=shypo,
            dhypo=dhypo,
            dx=dx,
            dy=dy,
            verbose=verbose,
        )
    cand_lon = mesh["lon_mat"].ravel()
    cand_lat = mesh["lat_mat"].ravel()

    # ---- 2. 0~200 km 预筛（以震中投影 + strike 为参考，所有周期点均有效） ----
    labels = [period_label(T) for T in periods]
    observation_mask = np.logical_and.reduce(
        [
            np.isfinite(sta[label].values) & (sta[label].values > 0)
            for label in labels
        ]
    )
    used_mask = observation_mask.copy()
    for T in periods:
        _, a_eq_ref, _ = _period_predict_many(
            T,
            Ms,
            rc,
            np.array([lon0]),
            np.array([lat0]),
            strike,
            sta["lon"].values,
            sta["lat"].values,
            extent,
        )
        used_mask &= np.isfinite(a_eq_ref[0]) & (a_eq_ref[0] <= max_dist)
    if not used_mask.any():
        raise ValueError(
            "CEA2019 反演没有可用台站：请检查所选观测列、正值要求、"
            "max_dist 和 extent"
        )
    if verbose:
        print(
            f"[预筛] 周期点 {[period_label(t) for t in periods]}："
            f"保留 {int(used_mask.sum())}/{len(sta)} 台站"
            f"（长轴距/短轴距 ≤ {max_dist:g} km）"
        )

    # ---- 3. 候选点 chi2（向量化） ----
    chi2 = _chi2_batch(
        periods, Ms, rc, cand_lon, cand_lat, strike, sta, used_mask, extent
    )
    if not np.isfinite(chi2).any():
        raise RuntimeError("所有候选宏观震中的 chi2 均为非有限数")
    j = int(np.nanargmin(chi2))
    grid_best = (float(chi2[j]), float(cand_lon[j]), float(cand_lat[j]))
    best = grid_best

    # 断层投影多边形：反演震中不允许超出断层范围
    flon, flat = mesh["lon_mat"], mesh["lat_mat"]
    poly_lon = np.concatenate(
        [flon[0], flon[:, -1], flon[-1][::-1], flon[:, 0][::-1]]
    )
    poly_lat = np.concatenate(
        [flat[0], flat[:, -1], flat[-1][::-1], flat[:, 0][::-1]]
    )
    from matplotlib.path import Path

    fault_path = Path(np.column_stack([poly_lon, poly_lat]))

    # ---- 4. 局部连续精化（Nelder-Mead，限制在断层范围内） ----
    fell_back = False
    optimizer_success = None
    optimizer_message = "未启用局部精化"
    if local_refine and local_refine > 0:
        from scipy.optimize import minimize

        lon_lo, lon_hi = float(flon.min()), float(flon.max())
        lat_lo, lat_hi = float(flat.min()), float(flat.max())
        refine_lon_lo = max(lon_lo, grid_best[1] - float(local_refine))
        refine_lon_hi = min(lon_hi, grid_best[1] + float(local_refine))
        refine_lat_lo = max(lat_lo, grid_best[2] - float(local_refine))
        refine_lat_hi = min(lat_hi, grid_best[2] + float(local_refine))

        def obj(x):
            lo, la = float(x[0]), float(x[1])
            c = float(
                _chi2_batch(
                    periods,
                    Ms,
                    rc,
                    np.array([lo]),
                    np.array([la]),
                    strike,
                    sta,
                    used_mask,
                    extent,
                )[0]
            )
            pen = 0.0
            if not fault_path.contains_point((lo, la)):
                pen += 1e6
            for v, vmin, vmax in (
                (lo, refine_lon_lo, refine_lon_hi),
                (la, refine_lat_lo, refine_lat_hi),
            ):
                if v < vmin:
                    pen += 1e3 * (vmin - v) ** 2
                elif v > vmax:
                    pen += 1e3 * (v - vmax) ** 2
            return c + pen

        res = minimize(
            obj,
            [best[1], best[2]],
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-10},
        )
        cand_refine = (float(res.fun), float(res.x[0]), float(res.x[1]))
        optimizer_success = bool(res.success and np.isfinite(res.fun))
        optimizer_message = str(res.message)
        inside_fault = fault_path.contains_point(
            (cand_refine[1], cand_refine[2]), radius=1e-12
        )
        inside_refine = (
            refine_lon_lo <= cand_refine[1] <= refine_lon_hi
            and refine_lat_lo <= cand_refine[2] <= refine_lat_hi
        )
        if (
            optimizer_success
            and inside_fault
            and inside_refine
            and cand_refine[0] < grid_best[0]
        ):
            best = cand_refine
        else:
            best = grid_best
            fell_back = True
    lon_opt, lat_opt, chi2_opt = best[1], best[2], best[0]
    if verbose and fell_back:
        print(
            "[精化] 局部优化未得到更优的有效断层内解，已保留网格最优；"
            f"状态：{optimizer_message}"
        )

    # ---- 5. 最终预测 + 逐台站统计表 ----
    lg = lambda x: np.log10(np.maximum(x, 1e-9))
    R = np.sqrt(
        (
            (sta["lon"].values - lon_opt)
            * 111.32
            * math.cos(math.radians(lat_opt))
        )
        ** 2
        + ((sta["lat"].values - lat_opt) * 110.57) ** 2
    )
    table = pd.DataFrame(
        {
            "Sta_ID": sta["Sta_ID"],
            "Sta_longi": np.round(sta["lon"].values, 4),
            "Sta_lati": np.round(sta["lat"].values, 4),
            "R_km": np.round(R, 2),
        }
    )
    rms = {}
    used_final = used_mask.copy()
    final_predictions = {}
    for T in periods:
        label = period_label(T)
        val, a_eq, b_eq = _period_predict_many(
            T,
            Ms,
            rc,
            np.array([lon_opt]),
            np.array([lat_opt]),
            strike,
            sta["lon"].values,
            sta["lat"].values,
            extent,
        )
        val, a_eq, b_eq = val[0], a_eq[0], b_eq[0]
        final_predictions[T] = (val, a_eq, b_eq)
        used_final &= np.isfinite(val)
    if not used_final.any():
        raise RuntimeError("最优宏观震中处没有所有反演参数都有效的台站")

    for T in periods:
        label = period_label(T)
        val, a_eq, b_eq = final_predictions[T]
        res_lg10 = lg(val) - lg(sta[label].values)
        res_ln = res_lg10 * math.log(10.0)
        table[f"{label}_obs"] = np.round(sta[label].values, 4)
        table[f"{label}_pred"] = np.round(val, 4)
        table[f"Repi_long_{label}(km)"] = np.round(a_eq, 2)
        table[f"Repi_short_{label}(km)"] = np.round(b_eq, 2)
        table[f"{label}_res"] = np.round(res_ln, 4)
        table[f"{label}_res_ln"] = np.round(res_ln, 4)
        table[f"{label}_res_lg10"] = np.round(res_lg10, 4)
        rms[f"rms_ln{label}"] = float(
            np.sqrt(np.mean(res_ln[used_final] ** 2))
        )
        rms[f"rms_lg10{label}"] = float(
            np.sqrt(np.mean(res_lg10[used_final] ** 2))
        )
        # 兼容旧调用：旧键仍表示 log10 RMS。
        rms[f"rms_lg{label}"] = rms[f"rms_lg10{label}"]
    table["used"] = used_final
    n_used = int(used_final.sum())
    n_residuals = n_used * len(periods)
    degrees_of_freedom = max(n_residuals - 2, 0)
    reduced_chi2 = (
        float(chi2_opt / degrees_of_freedom)
        if degrees_of_freedom > 0
        else np.nan
    )
    boundary_hit = fault_path.contains_point(
        (lon_opt, lat_opt), radius=1e-8
    ) and not fault_path.contains_point((lon_opt, lat_opt), radius=-1e-8)

    dist_true = (
        _haversine_km(lon_opt, lat_opt, true_epi[0], true_epi[1])
        if true_epi is not None
        else np.nan
    )
    result = {
        "epicenter": (lon_opt, lat_opt),
        "lon": lon_opt,
        "lat": lat_opt,
        "strike": strike,
        "params": periods,
        "invert_GMIMs": periods,
        "plot_GMIMs": plot_periods,
        "param_labels": [period_label(t) for t in periods],
        "chi2": chi2_opt,
        "reduced_chi2": reduced_chi2,
        "degrees_of_freedom": degrees_of_freedom,
        "n_used": n_used,
        "n_sta": len(sta),
        "dist_true_km": dist_true,
        **rms,
        "table": table,
        "mesh": mesh,
        "observation_columns": sta.attrs.get("source_columns", {}),
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "boundary_hit": bool(boundary_hit),
    }
    if verbose:
        print(
            f"\n[CEA2019 反演] 周期点 {result['param_labels']} 最优震中 = "
            f"({lon_opt:.4f}, {lat_opt:.4f})  chi2 = {chi2_opt:.2f}, "
            f"reduced chi2 = {reduced_chi2:.2f}"
            f"（{n_used}/{len(sta)} 台站参与）"
            + (
                f"  距已知震中 {dist_true:.1f} km"
                if np.isfinite(dist_true)
                else ""
            )
        )
    if outpath:
        export_cea2019_vs_obs_txt(
            data=data,
            macro_epicenter=(lon_opt, lat_opt),
            Ms=Ms,
            region=region,
            strike=strike,
            params=plot_periods,
            outpath=outpath,
        )
    if plot_path:
        plot_cea2019_vs_obs(
            data=data,
            macro_epicenter=(lon_opt, lat_opt),
            initial_epicenter=(lon0, lat0),
            Ms=Ms,
            region=region,
            strike=strike,
            params=plot_periods,
            fault_lon_mat=mesh["lon_mat"],
            fault_lat_mat=mesh["lat_mat"],
            outpath=plot_path,
        )
    return result


def _haversine_km(lon1, lat1, lon2, lat2):
    re = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2.0 * re * math.asin(math.sqrt(a))


# ---- 测试 ----
if __name__ == "__main__":
    invert_epicenter_cea2019(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        Ms=6.8,
        Mw=7.0,
        region="青藏区",
        max_dist=400,
        hypo=(87.45, 28.5, 10.0),
        strike=187.0,
        dip=49.0,
        rake=-78.0,
        shypo=10,
        invert_GMIMs=(-1, -2),  # 迭代：PGA + PGV
        plot_GMIMs=[-1, -2, 0.3, 1, 3, 6],  # 绘图：PGA/PGV/PSA0.3/1/3/6
        true_epi=(87.45, 28.5),
        outpath="Test_output/CEA2019_epicenter_inversion_stats_pga_pgv0.txt",
        plot_path="Test_output/CEA2019_epicenter_inversion_pga_pgv0.png",
    )
