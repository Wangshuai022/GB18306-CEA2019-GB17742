"""
GB18306 震中反演 —— 基于 GB18306-2015 椭圆衰减，断层网格约束反演范围
======================================================================
已知震中周边台站的 PGA / PGV / 烈度，用 GB18306 衰减关系反演"最优震中"。

反演线路（mode，目标函数 = 按各自对数标准差 σ 归一化的最小二乘 chi2）：
    pga       : 仅 PGA        （σ = 0.236）
    pgv       : 仅 PGV        （σ = 0.271）
    intensity : 仅烈度        （σ = 0.6636，青藏区）
    pga_pgv   : PGA + PGV 联合

断层范围（候选震中由断层面网格给出）：
    1) 走向 strike、倾角 dip、滑移角 rake 为必需输入；
       rake 自动判别机制：SS（走滑）/ RS（逆冲）/ NS（正断）；
    2) 默认用 Leonard2014（SMD 地壳 104 事件修正版）预测"中位值"
       破裂长度 L、宽度 W；
    3) 由震中经纬度 + 深度 + strike/dip/L/W，调用
       mesh_single_rectangular_finite_fault.build_fault_grid 生成断层面网格点，
       默认 shypo = 0（沿走向相对位置），dhypo = 0.57 * W（沿倾向相对位置）；
    4) 候选震中 = 断层面网格节点的经纬度；
    5) 候选点上批量计算 chi2，取最小者为最优震中，再做局部连续精化；
       反演震中严格限制在断层投影范围内（越界自动回退到网格最优）。

有效范围：GB18306 只适用于 0~200 km（长轴距/短轴距），
反演前以"震中投影 + strike"为参考点预筛台站，超限台站剔除。

依赖（同目录或可导入）：
    GB18306_class.py（衰减模型）
    Leonard2014_fitted_by_SMD_crust.py、mesh_single_rectangular_finite_fault.py
    （均与本文件位于同一目录）

使用案例：
    from GB18306_epicenter_inversion import invert_epicenter_gb18306
    res = invert_epicenter_gb18306(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        Ms=6.8, region="青藏区",
        hypo=(87.378, 28.604, 10.0),        # 震中(经,纬,深度km)
        strike=187.0, dip=49.0, rake=-78.0,  # 走向/倾角/滑移角（必给，定日）
        mode="pga_pgv",
        outpath="GB18306_epicenter_inversion_stats.txt",
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

from ellipse_fields import GB18306EllipseField

try:
    from Leonard2014_fitted_by_SMD_crust import l14_fitted
    from mesh_single_rectangular_finite_fault import build_fault_grid
except ImportError as e:
    raise ImportError(
        "缺少断层工具：Leonard2014_fitted_by_SMD_crust.py 与 "
        "mesh_single_rectangular_finite_fault.py 必须与反演程序位于同一目录。"
        f"原始错误：{e}"
    )

SIGMA_GMM = {"pga": 0.236, "pgv": 0.271}
MODES = ("pga_pgv", "intensity", "pga", "pgv")
MODE_NAMES = {
    "pga_pgv": "PGA+PGV 联合",
    "intensity": "烈度",
    "pga": "仅 PGA",
    "pgv": "仅 PGV",
}


# ==================== 机制判别（rake → SS / RS / NS） ====================


def rake_to_mechanism(rake, verbose=True):
    """
    由滑移角 rake 判别断层机制：
        |rake| ≤ 30° 或 |rake| ≥ 150°  → SS（走滑）
        30° < rake ≤ 150°（含正斜）   → RS（逆冲）
        -150° ≤ rake < -30°（含负斜） → NS（正断）
    返回 (mechanism, fault_type)：fault_type 为 l14_fitted 接受的
    SSF / RF / NF。
    """
    r = float(rake) % 360.0
    if r > 180.0:
        r -= 360.0
    if abs(r) <= 30.0 or abs(r) >= 150.0:
        mech, ft = "SS", "SSF"
    elif r > 30.0:
        mech, ft = "RS", "RF"
    else:
        mech, ft = "NS", "NF"
    if verbose:
        print(f"[机制判别] rake = {rake:g}°  →  {mech}（{ft}）")
    return mech, ft


# ==================== 断层网格（L14 中位值 + 矩形有限断层） ====================


def fault_mesh_points(
    hypo_lon,
    hypo_lat,
    hypo_depth,
    strike,
    dip,
    rake,
    Mw,
    eq_type="板间",
    shypo=None,
    dhypo=None,
    dx=0.5,
    dy=0.5,
    verbose=True,
):
    """
    由震中(经纬度+深度) + 走向/倾角/滑移角，生成断层面网格点。

    流程：
        1) rake → 机制（SS/RS/NS）；
        2) L14（SMD 修正版）预测中位破裂长度 L、宽度 W；
        3) build_fault_grid 生成断层面网格：shypo/dhypo 为相对断层上缘的
           沿走向/沿倾向位置（km），默认 None → shypo=0、dhypo=0.57*W，
           也可自定义（如 shypo=-10、dhypo=15）；
        4) 返回网格经纬度/深度矩阵及尺寸信息。
    """
    mech, fault_type = rake_to_mechanism(rake, verbose=verbose)
    r = l14_fitted(Mw, fault_type, eq_type)
    L_raw, W_raw = float(r["L"]), float(r["W"])
    if verbose:
        print(
            f"[L14 中位值] Mw{Mw:g} {mech}（{eq_type}）"
            f"  L = {L_raw:.2f} km, W = {W_raw:.2f} km（原始预测）"
        )
    # 断层子块默认 0.5×0.5 km：预测尺寸向上取整到 0.5 的整数倍
    # （如 14.2 → 14.5），保证子块数量为整数、子块严格 0.5 km
    L = math.ceil(L_raw / 0.5) * 0.5
    W = math.ceil(W_raw / 0.5) * 0.5
    if verbose:
        print(
            f"[断层尺寸取整] L = {L:.2f} km, W = {W:.2f} km"
            f"（0.5 km 向上取值，子块 {dx:g}×{dy:g} km）"
        )
    if shypo is None:
        shypo = 0.0
    if dhypo is None:
        dhypo = 0.57 * W
        tag = "（默认：0 / 0.57*W）"
    else:
        tag = "（自定义）"
    if verbose:
        print(f"[断层网格] shypo = {shypo:g} km, dhypo = {dhypo:.2f} km {tag}")
    result = build_fault_grid(
        Hypo_longi=float(hypo_lon),
        Hypo_lati=float(hypo_lat),
        Hypo_depth=float(hypo_depth),
        strike=float(strike),
        dip=float(dip),
        Fault_length=L,
        Fault_width=W,
        shypo=float(shypo),
        dhypo=dhypo,
        dx=float(dx),
        dy=float(dy),
    )
    return {
        "lon_mat": np.asarray(result["longitude_matrix"]),
        "lat_mat": np.asarray(result["latitude_matrix"]),
        "depth_mat": np.asarray(result["depth_matrix"]),
        "L_km": L,
        "W_km": W,
        "mechanism": mech,
        "fault_type": fault_type,
        "shypo_km": shypo,
        "dhypo_km": dhypo,
        "n_strike": np.asarray(result["longitude_matrix"]).shape[1],
        "n_dip": np.asarray(result["longitude_matrix"]).shape[0],
    }


# 椭圆场前向模型统一由 ellipse_fields.GB18306EllipseField 提供。


# ==================== 数据读取 ====================


def load_station_data(data, mode="pga_pgv"):
    """读取台站数据；只强制要求当前反演模式真正需要的观测列。"""
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一")
    if isinstance(data, (str, os.PathLike)):
        df = pd.read_csv(str(data), sep="\t", encoding="utf-8")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data 必须是文件路径或 pandas.DataFrame")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"longi": "lon", "lati": "lat"})
    # GB18306 的 PGA/PGV 实为 EPA/EPV：优先 EPA_H/EPV_H，再回退 PGA_H/PGV_H 或 PGA/PGV
    pga_col = next((c for c in ("EPA_H", "PGA_H", "PGA") if c in df.columns), None)
    pgv_col = next((c for c in ("EPV_H", "PGV_H", "PGV") if c in df.columns), None)
    required = {
        "intensity": [("I", "I")],
        "pga": [("pga", pga_col)],
        "pgv": [("pgv", pgv_col)],
        "pga_pgv": [("pga", pga_col), ("pgv", pgv_col)],
    }[mode]
    missing = [name for name, col in required if col is None or col not in df.columns]
    if missing:
        raise ValueError(f"{mode} 反演缺少观测列：{missing}")
    if "Sta_ID" not in df.columns:
        df["Sta_ID"] = [f"S{i + 1}" for i in range(len(df))]
    out = pd.DataFrame(
        {
            "Sta_ID": df["Sta_ID"].astype(str),
            "lon": pd.to_numeric(df["lon"], errors="coerce"),
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "I": (
                pd.to_numeric(df["I"], errors="coerce") if "I" in df.columns else np.nan
            ),
            "pga": (
                pd.to_numeric(df[pga_col], errors="coerce")
                if pga_col is not None
                else np.nan
            ),
            "pgv": (
                pd.to_numeric(df[pgv_col], errors="coerce")
                if pgv_col is not None
                else np.nan
            ),
        }
    )
    out["pga"] = out["pga"].where(out["pga"] > 0)
    out["pgv"] = out["pgv"].where(out["pgv"] > 0)
    out = out.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("没有经纬度有效的观测台站")
    out.attrs["source_columns"] = {"I": "I", "PGA": pga_col, "PGV": pgv_col}
    return out


# ==================== 反演主函数 ====================


def invert_epicenter_gb18306(
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
    verbose=True,
):
    """
    GB18306 震中反演主函数。

    明确输入：
        data     实测数据：文件路径或 DataFrame
                 （PGA/PGV 实为 EPA/EPV，优先用 EPA_H/EPV_H，再回退 PGA_H/PGV_H）
        Ms       面波震级（GB18306 衰减用）
        region   分区："青藏区"/"新疆区"/"东部区"/"中部区"
        hypo     震中 (经度, 纬度, 深度 km) —— 断层网格的破裂起始点
        strike   走向（正北=0°，顺时针）—— 必需输入
        dip      倾角（0~90°）—— 必需输入
        rake     滑移角（-180~180°）—— 必需输入，用于判别 SS/RS/NS
        fault_lon_mat / fault_lat_mat  可选：外部断层网格（二维矩阵，
                   第一行=上缘、最后一行=下缘）。提供时直接作为反演候选
                   范围，**不再用 L14 定标率**；未提供时默认用 L14 中位
                   L/W + build_fault_grid 生成网格。
        Mw       矩震级（L14 定标率用）；缺省取 Ms
        eq_type  构造类型："板间"/"板周"（SMD 修正截距）或 "板内"（论文 SCR 系数）
        mode     线路：pga / pgv / intensity / pga_pgv
        max_dist GB18306 有效范围（长轴距/短轴距 ≤ 该值，km，默认 200）
        extent   衰减场最大距离 km
        dx, dy   断层面网格期望尺寸 km（默认 0.5）
        shypo    沿走向相对断层上缘的位置（km，默认 None → 0）
        dhypo    沿倾向相对断层上缘的位置（km，默认 None → 0.57*W）；
                 可自定义，如 shypo=-10、dhypo=15
        local_refine 最优网格点附近的连续精化半宽（°；0 = 仅用网格点）
        outpath  统计 txt 输出路径（None = 不导出）
        plot_path  4×N 预测-实测图输出路径（None = 不绘图；绘制 PGA/PGV/烈度，
                    含断层投影、初始破裂点与宏观震中）
        plot_GMIMs 绘图参数（与反演 mode 分开），如 (-1,-2,"Intensity")；
                   None 时自动绘制数据中实际存在的全部 GB18306 参数
        true_epi 已知震中（仅验证用）

    返回 dict：epicenter / lon / lat / strike / chi2 / n_used / rms_* /
               table（逐台站统计）/ mesh（断层网格信息）
    """
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，收到：{mode}")
    if len(hypo) != 3:
        raise ValueError("hypo 必须为 (经度, 纬度, 深度km)")
    Mw = float(Mw) if Mw is not None else float(Ms)

    sta = load_station_data(data, mode=mode)
    if plot_GMIMs is None:
        actual_plot_params = []
        if np.isfinite(sta["pga"].values).any():
            actual_plot_params.append(-1)
        if np.isfinite(sta["pgv"].values).any():
            actual_plot_params.append(-2)
        if np.isfinite(sta["I"].values).any():
            actual_plot_params.append("Intensity")
    else:
        actual_plot_params = list(plot_GMIMs)
    lon0, lat0, depth0 = float(hypo[0]), float(hypo[1]), float(hypo[2])
    strike = float(strike) % 360.0

    # ---- 1. 断层网格：优先外部网格，未提供才用 L14 ----
    if (fault_lon_mat is None) != (fault_lat_mat is None):
        raise ValueError("fault_lon_mat 与 fault_lat_mat 必须同时提供或同时省略")
    if fault_lon_mat is not None:
        flon_in = np.asarray(fault_lon_mat, dtype=float)
        flat_in = np.asarray(fault_lat_mat, dtype=float)
        if flon_in.ndim != 2 or flat_in.ndim != 2:
            raise ValueError("fault_lon_mat / fault_lat_mat 必须是二维网格矩阵")
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

    # ---- 2. 前向模型 + 0~200 km 预筛（以震中投影 + strike 为参考） ----
    field = GB18306EllipseField(region, Ms, extent=extent)
    _, _, _, aI_ref, _, aA_ref, _, aV_ref, _ = field.predict(
        lon0, lat0, strike, sta["lon"].values, sta["lat"].values
    )
    observation_mask = {
        "intensity": np.isfinite(sta["I"].values),
        "pga": np.isfinite(sta["pga"].values),
        "pgv": np.isfinite(sta["pgv"].values),
        "pga_pgv": np.isfinite(sta["pga"].values) & np.isfinite(sta["pgv"].values),
    }[mode]
    used_mask = (
        observation_mask
        & {
            "intensity": np.isfinite(aI_ref) & (aI_ref <= max_dist),
            "pga": np.isfinite(aA_ref) & (aA_ref <= max_dist),
            "pgv": np.isfinite(aV_ref) & (aV_ref <= max_dist),
            "pga_pgv": (np.isfinite(aA_ref) & (aA_ref <= max_dist))
            & (np.isfinite(aV_ref) & (aV_ref <= max_dist)),
        }[mode]
    )
    if not used_mask.any():
        raise ValueError(
            f"{mode} 反演没有可用台站：请检查观测列、正值要求、max_dist 和 extent"
        )
    if verbose:
        print(
            f"[预筛] {mode}：保留 {int(used_mask.sum())}/{len(sta)} 台站"
            f"（长轴距/短轴距 ≤ {max_dist:g} km）"
        )

    # ---- 3. 候选点 chi2（向量化） ----
    chi2 = _chi2_batch(field, mode, cand_lon, cand_lat, strike, sta, used_mask)
    if not np.isfinite(chi2).any():
        raise RuntimeError("所有候选宏观震中的 chi2 均为非有限数")
    j = int(np.nanargmin(chi2))
    grid_best = (float(chi2[j]), float(cand_lon[j]), float(cand_lat[j]))
    best = grid_best

    # 断层投影多边形：反演震中不允许超出断层范围
    flon, flat = mesh["lon_mat"], mesh["lat_mat"]
    poly_lon = np.concatenate([flon[0], flon[:, -1], flon[-1][::-1], flon[:, 0][::-1]])
    poly_lat = np.concatenate([flat[0], flat[:, -1], flat[-1][::-1], flat[:, 0][::-1]])
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
                    field,
                    mode,
                    np.array([lo]),
                    np.array([la]),
                    strike,
                    sta,
                    used_mask,
                )[0]
            )
            pen = 0.0
            if not fault_path.contains_point((lo, la)):
                pen += 1e6  # 超出断层范围：重罚
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
            best = grid_best  # 越界回退：只用断层范围内的结果
            fell_back = True

    lon_opt, lat_opt, chi2_opt = best[1], best[2], best[0]
    if verbose and fell_back:
        print(
            "[精化] 局部优化未得到更优的有效断层内解，已保留网格最优；"
            f"状态：{optimizer_message}"
        )

    # ---- 5. 最终预测 + 逐台站统计表 ----
    I, A, V, aI, bI, aA, bA, aV, bV = field.predict(
        lon_opt, lat_opt, strike, sta["lon"].values, sta["lat"].values
    )
    if mode == "intensity":
        in_field = np.isfinite(I)
    elif mode == "pga":
        in_field = np.isfinite(A)
    elif mode == "pgv":
        in_field = np.isfinite(V)
    else:
        in_field = np.isfinite(A) & np.isfinite(V)
    used = used_mask & in_field
    lg = lambda x: np.log10(np.maximum(x, 1e-9))
    # 残差统一定义为“预测−观测”；GMM 主残差采用 ln(Pred/Obs)。
    res_I = I - sta["I"].values
    res_lg10PGA = np.log10(A) - lg(sta["pga"].values)
    res_lg10PGV = np.log10(V) - lg(sta["pgv"].values)
    res_lnPGA = res_lg10PGA * math.log(10.0)
    res_lnPGV = res_lg10PGV * math.log(10.0)
    R = np.sqrt(
        ((sta["lon"].values - lon_opt) * 111.32 * math.cos(math.radians(lat_opt))) ** 2
        + ((sta["lat"].values - lat_opt) * 110.57) ** 2
    )
    table = pd.DataFrame(
        {
            "Sta_ID": sta["Sta_ID"],
            "Sta_longi": np.round(sta["lon"].values, 4),
            "Sta_lati": np.round(sta["lat"].values, 4),
            "R_km": np.round(R, 2),
            "Repi_long_I(km)": np.round(aI, 2),
            "Repi_short_I(km)": np.round(bI, 2),
            "I_obs": np.round(sta["I"].values, 2),
            "I_pred": np.round(I, 2),
            "I_res": np.round(res_I, 3),
            "PGA_obs": np.round(sta["pga"].values, 3),
            "PGA_pred": np.round(A, 3),
            "Repi_long_PGA(km)": np.round(aA, 2),
            "Repi_short_PGA(km)": np.round(bA, 2),
            "PGA_res": np.round(res_lnPGA, 4),
            "PGA_res_ln": np.round(res_lnPGA, 4),
            "PGA_res_lg10": np.round(res_lg10PGA, 4),
            "PGV_obs": np.round(sta["pgv"].values, 3),
            "PGV_pred": np.round(V, 3),
            "Repi_long_PGV(km)": np.round(aV, 2),
            "Repi_short_PGV(km)": np.round(bV, 2),
            "PGV_res": np.round(res_lnPGV, 4),
            "PGV_res_ln": np.round(res_lnPGV, 4),
            "PGV_res_lg10": np.round(res_lg10PGV, 4),
            "used": used,
        }
    )
    n_used = int(used.sum())
    n_residuals = n_used * (2 if mode == "pga_pgv" else 1)
    degrees_of_freedom = max(n_residuals - 2, 0)
    reduced_chi2 = (
        float(chi2_opt / degrees_of_freedom) if degrees_of_freedom > 0 else np.nan
    )
    boundary_hit = fault_path.contains_point(
        (lon_opt, lat_opt), radius=1e-8
    ) and not fault_path.contains_point((lon_opt, lat_opt), radius=-1e-8)

    def masked_rms(values):
        valid = used & np.isfinite(values)
        return float(np.sqrt(np.mean(values[valid] ** 2))) if valid.any() else np.nan

    rms = {
        "rms_lnPGA": masked_rms(res_lnPGA),
        "rms_lnPGV": masked_rms(res_lnPGV),
        "rms_lg10PGA": masked_rms(res_lg10PGA),
        "rms_lg10PGV": masked_rms(res_lg10PGV),
        # 兼容旧调用：旧键仍表示 log10 RMS。
        "rms_lgPGA": masked_rms(res_lg10PGA),
        "rms_lgPGV": masked_rms(res_lg10PGV),
        "rms_I": masked_rms(res_I),
    }
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
        "mode": mode,
        "chi2": chi2_opt,
        "reduced_chi2": reduced_chi2,
        "degrees_of_freedom": degrees_of_freedom,
        "n_used": n_used,
        "n_sta": len(sta),
        "dist_true_km": dist_true,
        **rms,
        "table": table,
        "mesh": mesh,
        "field": field,
        "observation_columns": sta.attrs.get("source_columns", {}),
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "boundary_hit": bool(boundary_hit),
        "plot_GMIMs": actual_plot_params,
    }
    if verbose:
        print(
            f"\n[{MODE_NAMES[mode]}] 最优震中 = ({lon_opt:.4f}, {lat_opt:.4f})"
            f"  chi2 = {chi2_opt:.2f}, reduced chi2 = {reduced_chi2:.2f}"
            f"（{n_used}/{len(sta)} 台站参与）"
            + (f"  距已知震中 {dist_true:.1f} km" if np.isfinite(dist_true) else "")
        )
    if outpath:
        _export_stats(result, outpath)
    if plot_path:
        from GB18306_vs_Obs import plot_gb18306_vs_obs

        plot_gb18306_vs_obs(
            data=data,
            macro_epicenter=(lon_opt, lat_opt),
            initial_epicenter=(lon0, lat0),  # 初始破裂点（发布值）
            Ms=Ms,
            region=region,
            strike=strike,
            params=actual_plot_params,
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
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * re * math.asin(math.sqrt(a))


def _chi2_batch(field, mode, cand_lon, cand_lat, strike, sta, used_mask):
    """候选震中批量 chi2（归一化最小二乘，场外重罚）"""
    I, A, V, _, _, _, _, _, _ = field.predict_many(
        cand_lon, cand_lat, strike, sta["lon"].values, sta["lat"].values
    )
    lg = lambda x: np.log10(np.maximum(x, 1e-9))
    used = used_mask[None, :]
    if mode == "intensity":
        r = (sta["I"].values[None, :] - I) / field.sigma_I
        in_field = np.isfinite(I)
    elif mode == "pga":
        r = (lg(sta["pga"].values)[None, :] - np.log10(A)) / SIGMA_GMM["pga"]
        in_field = np.isfinite(A)
    elif mode == "pgv":
        r = (lg(sta["pgv"].values)[None, :] - np.log10(V)) / SIGMA_GMM["pgv"]
        in_field = np.isfinite(V)
    else:
        rA = (lg(sta["pga"].values)[None, :] - np.log10(A)) / SIGMA_GMM["pga"]
        rV = (lg(sta["pgv"].values)[None, :] - np.log10(V)) / SIGMA_GMM["pgv"]
        r = None
        in_field = np.isfinite(A) & np.isfinite(V)
    if r is None:
        ok = used & in_field
        chi2 = np.where(ok, rA * rA, 0.0).sum(axis=1) + np.where(ok, rV * rV, 0.0).sum(
            axis=1
        )
    else:
        ok = used & in_field
        chi2 = np.where(ok, r * r, 0.0).sum(axis=1)
    chi2 = chi2 + 1000.0 * (used & ~in_field).sum(axis=1)
    return chi2


def _export_stats(result, outpath):
    """逐台站统计 txt（第一行即表头）"""
    tab = result["table"]
    lines = ["\t".join(tab.columns)]
    for row in tab.itertuples(index=False):
        lines.append(
            "\t".join(
                (
                    ""
                    if pd.isna(v)
                    else f"{v:g}" if isinstance(v, (int, float)) else str(v)
                )
                for v in row
            )
        )
    with open(outpath, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存统计表：{os.path.abspath(outpath)}")
    return outpath


# ---- 测试 ----
if __name__ == "__main__":
    invert_epicenter_gb18306(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        Ms=6.8,
        Mw=7.0,
        region="青藏区",
        max_dist=400,
        hypo=(87.45, 28.5, 10.0),
        strike=187,
        dip=49,
        rake=-78,
        shypo=10,
        mode="pga_pgv",  # pga / pgv / intensity / pga_pgv
        true_epi=(87.45, 28.5),
        outpath="Test_output/GB18306_epicenter_inversion_stats_pga_pgv.txt",
        plot_path="Test_output/GB18306_epicenter_inversion_pga_pgv.png",
    )
