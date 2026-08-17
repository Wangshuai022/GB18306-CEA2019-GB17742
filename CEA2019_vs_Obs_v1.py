# -*- coding: utf-8 -*-
"""
CEA2019 预测 vs 实测 —— 基于给定震中的 4×N 综合绘图
====================================================
v1 优化版：与 CEA2019_vs_Obs.py 输出完全一致，主要提速手段：
    1) 去掉 tight_layout（改固定子图间距）——省 ~6 s；
    2) 去掉注释自适应的整图重绘（改数学预留头部空间）——省 ~7 s；
    3) 缩小画布尺寸 + 输出 dpi 可调（默认 150）——渲染/压缩更快。
数值结果（预测、椭圆距、残差）与旧版完全一致。

输入：实测台站文件、初始震中经纬度、震级 Ms、分区、走向 strike；
用 CEA2019 衰减模型预测各台站地震动参数，与实测值对比，输出一张
"4 排 × N 列"图（N = 参数个数）：

    排1（子图1）预测云图（观测点范围内，经纬度区间按公里取齐成近似方形）+
                实测散点，USGS 色标；
    排2（子图2）衰减曲线（所选轴 中值 ±1σ），X 距离为 log 轴，
                范围 1 ~ 最远椭圆距（向上取整到 100 的整数 +100，最小 200）；
                默认用长轴曲线和长轴距，可切短轴（axis="短轴"）；
                超过 200 km 浅灰填充；
    排3（子图3）残差（预测-实测，自然对数），X 距离为线性轴，
                距离上限取最远椭圆距向上取整到 50 的倍数（最小 200 km），
                超过 200 km 浅灰填充；
    排4（子图4）残差分布：全部 / <200 km / ≥200 km 三组，
                每组画 散点 + 半小提琴 + 箱线。

支持的参数（按周期点定义，N = 参数个数）：
    -1 或 0        → PGA（gal）
    -2             → PGV（cm/s）
    0.01 0.1 0.3 1 3 6 等数值周期 → PSA(T=0.01s) 等（cm/s²），
                    标签保留 2 位小数（CEA2019 支持周期间线性插值）
    "Intensity"    → 烈度（特殊，单独表示）

烈度说明：CEA2019 本身只有 PGA/PGV/PSA；这里的"烈度"用 CEA2019 预测的
PGA/PGV 按 GB/T 17742-2020 换算成仪器烈度，再与实测烈度列 I 对比。

台站标记：数据含 Instrument_Type 且同时出现 EI（烈度计）/ HN（强震仪）时，
HN 用圆点、EI 用三角形；未区分（或只有一种）时全部用圆点。

色标（均引用 CEA2019_pre.py 的定义）：
    PGA / PSA  → PGA_LEVELS（PSA 按该周期 PSA/PGA 参考比值缩放）；
    PGV        → PGV_LEVELS（= PGA ÷ 10）；
    烈度       → USGS MMI 十色，1~10 度。

使用案例（直接复制可用）：
    from CEA2019_vs_Obs import plot_cea2019_vs_obs
    plot_cea2019_vs_obs(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        epicenter=(87.378, 28.604), Ms=6.8, region="青藏区", strike=349.0,
        params=(-1, -2, 0.3, 1.0, "Intensity"),
        outpath="CEA2019_vs_Obs.png",
    )
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from CEA2019_pre import (
    USGS_MMI_COLORS,
    PGA_LEVELS,
    PGV_LEVELS,
    _period_levels,
    period_label,
    unit_label,
    fmt_pgv,
    fmt_psa,
    _region_core,
    _period_coeffs,
    _period_curves_sigma,
    _ellipse_radii_for_period,
    predict_period_values,
    km_to_lonlat,
)
from GB17742_class import GB17742_2020_Cal_instrument_intensity as CAL_INT
from stat_violin import (
    apply_style,
    half_violin_box_scatter,
)

# ==================== 参数定义（按周期点） ====================

# 周期点约定：
#   -1 或 0   → PGA（峰值加速度，gal）
#   -2        → PGV（峰值速度，cm/s）
#   数值周期  → PSA(T)，如 0.01、0.1、0.3、1、3、6 s，
#               标签统一保留 2 位小数：PSA(T=0.30s)（cm/s²）
#   "Intensity" → 烈度（特殊：由 CEA2019 预测的 PGA/PGV 按 GB/T 17742 换算）
SUPPORTED_PERIODS = (-1, -2, 0.01, 0.1, 0.3, 1.0, 3.0, 6.0)


def normalize_params(params):
    """
    把用户输入的参数列表规范化为统一列表，支持：
        -1 / 0              → -1（PGA）
        -2                  → -2（PGV）
        0.01 0.1 0.3 1 3 6  → 原周期（PSA）
        字符串 "PGA" / "PGV" / "PSA(T=0.30s)" 同样接受
        "Intensity"         → 烈度（特殊）
    自动去重（如同时传 -1 和 "PGA" 只保留一个）。
    """
    out = []
    for p in params:
        if isinstance(p, str):
            s = p.strip()
            if s == "Intensity":
                item = "Intensity"
            elif s == "PGA":
                item = -1.0
            elif s == "PGV":
                item = -2.0
            elif s.lower().startswith("psa"):
                t = float(s.split("(")[1].split("=")[1].rstrip("s)"))
                item = float(t)
            else:
                raise ValueError(f"无法识别的参数：{p!r}")
        else:
            t = float(p)
            item = -1.0 if t == 0.0 else t
        if item not in out:
            out.append(item)
    return out


def param_info(p):
    """
    规范参数 → 信息 dict：label / kind / T / unit
    label 复用 CEA2019_pre.period_label（PGA / PGV / PSA(T=0.30s)，两位小数）。
    """
    if p == "Intensity":
        return {"label": "烈度", "kind": "intensity", "T": None, "unit": ""}
    t = float(p)
    return {
        "label": period_label(t),
        "kind": "gmm",
        "T": t,
        "unit": unit_label(t),
    }


def obs_col_candidates(info):
    """按参数信息返回实测列候选（水平向 _H 优先，其次 RotD50 / 原始列）"""
    if info["kind"] == "intensity":
        return ["I"]
    t = info["T"]
    if t == -1:
        return ["PGA_H", "PGA"]
    if t == -2:
        return ["PGV_H", "PGV"]
    tag = f"pSa(T={t:.2f}s)"
    return [f"{tag}_H", f"{tag}_RotD50", tag]


# ==================== 数据读取 ====================


def load_obs_data(data, params, param_cols=None):
    """
    读取实测数据（文件路径或 DataFrame），输出：
        Sta_ID / lon / lat / Instrument_Type / <各参数实测列>
    param_cols：参数标签 → 实际列名（缺省按 obs_col_candidates 自动识别）。
    """
    if isinstance(data, (str, os.PathLike)):
        df = pd.read_csv(str(data), sep="\t", encoding="utf-8")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data 必须是文件路径或 pandas.DataFrame")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"longi": "lon", "lati": "lat"})
    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError(
            f"缺少 lon/lat 列，现有列：{list(df.columns)[:20]}..."
        )
    if "Sta_ID" not in df.columns:
        df["Sta_ID"] = [f"S{i + 1}" for i in range(len(df))]

    out = pd.DataFrame(
        {
            "Sta_ID": df["Sta_ID"].astype(str),
            "lon": pd.to_numeric(df["lon"], errors="coerce"),
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "Instrument_Type": (
                df["Instrument_Type"].astype(str)
                if "Instrument_Type" in df.columns
                else ""
            ),
        }
    )
    for p in params:
        info = param_info(p)
        label = info["label"]
        if param_cols is not None and label in param_cols:
            cands = [param_cols[label]]
        else:
            cands = obs_col_candidates(info)
        col = next((c for c in cands if c in df.columns), None)
        if col is None:
            raise ValueError(
                f"参数 {label}（周期 {info['T']}）找不到实测列，候选：{cands}"
            )
        out[label] = pd.to_numeric(df[col], errors="coerce")
    return out.dropna(subset=["lon", "lat"]).reset_index(drop=True)


# ==================== 预测 / 椭圆距 / 残差 ====================


def _predict_one(
    param, lon, lat, strike, region, Ms, sta_lon, sta_lat, extent
):
    """用 CEA2019 预测单个参数在台站（或网格点）处的值"""
    info = param_info(param)
    if info["kind"] == "gmm":
        lab = predict_period_values(
            lon,
            lat,
            strike,
            region,
            Ms,
            [info["T"]],
            sta_lon,
            sta_lat,
            extent=extent,
        )
        return lab[period_label(info["T"])]
    # 烈度：预测 PGA + PGV → GB/T 17742 仪器烈度
    lab = predict_period_values(
        lon, lat, strike, region, Ms, [-1, -2], sta_lon, sta_lat, extent=extent
    )
    return CAL_INT.cal_Intensity_matrix(lab["PGA"], lab["PGV"])


def _a_eq(param, pred, region, Ms, preds):
    """
    台站所在等值椭圆的长轴距 a / 短轴距 b（km）。
    PGA/PGV/PSA 用各自预测值反算；烈度沿用 PGA 的椭圆（烈度由 PGA/PGV 换算）。
    """
    rc = _region_core(region)
    info = param_info(param)
    if info["kind"] == "gmm":
        a, b = _ellipse_radii_for_period(pred, info["T"], Ms, rc)
    else:
        a, b = _ellipse_radii_for_period(preds["PGA"], -1, Ms, rc)
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def _residual(pred, obs, kind):
    """残差 = 预测 - 实测。地震动参数用自然对数 ln(Pred/Obs)；烈度用线性差。"""
    if kind == "gmm":
        return np.log(
            np.asarray(pred, dtype=float) / np.asarray(obs, dtype=float)
        )
    return np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)


def _param_levels(param, Ms, region_core):
    """各参数专用色标分界（PGA/PSA、PGV、烈度各自不同）"""
    info = param_info(param)
    if info["kind"] == "gmm":
        return _period_levels(info["T"], Ms, region_core)
    return np.arange(0.5, 11.5, 1.0)  # 烈度：USGS MMI 十色，1~10 度


def _level_ticks(param, levels):
    """返回 (刻度位置, 刻度文字)：PGA 整数、PGV 按 fmt_pgv、PSA 按 fmt_psa、烈度整数"""
    info = param_info(param)
    if info["kind"] == "intensity":
        pos = np.arange(1, 11)
        return pos, [f"{v:.0f}" for v in pos]
    T = info["T"]
    if T == -2:
        return levels, [fmt_pgv(v) for v in levels]
    if T in (-1, 0):
        return levels, [f"{v:g}" for v in levels]
    return levels, [fmt_psa(v) for v in levels]


def _compute_vs_obs(
    data, epicenter, Ms, region, strike, params, extent, param_cols=None
):
    """
    共享计算（绘图和 txt 导出共用同一套结果）：
    规范化参数 → 台站观测 / CEA2019 预测 / 预测值对应椭圆距(a_eq,b_eq) / 残差。

    返回 dict：
        params  规范化参数列表（-1/-2/周期/Intensity）
        infos   参数 → param_info
        labels  参数标签列表（PGA / PGV / PSA(T=0.30s) / 烈度）
        obs     台站表（Sta_ID/lon/lat/Instrument_Type/<各标签实测列>）
        preds   标签 → 预测值数组
        aeqs    标签 → (a_eq, b_eq) 数组
        ress    标签 → 残差数组（gmm: ln(预测/实测)；烈度: 预测-实测）
        lon0/lat0/strike/rc  归一化后的震中/走向/分区核心名
    """
    params = normalize_params(params)
    if not params:
        raise ValueError(
            "params 不能为空；支持 -1/0(PGA)、-2(PGV)、数值周期(PSA)、'Intensity'"
        )
    infos = {p: param_info(p) for p in params}
    obs = load_obs_data(data, params, param_cols=param_cols)
    lon0, lat0 = float(epicenter[0]), float(epicenter[1])
    strike = strike % 360.0
    rc = _region_core(region)

    preds = {
        infos[p]["label"]: _predict_one(
            p,
            lon0,
            lat0,
            strike,
            region,
            Ms,
            obs["lon"].values,
            obs["lat"].values,
            extent,
        )
        for p in params
    }
    aeqs = {
        infos[p]["label"]: _a_eq(
            p, preds[infos[p]["label"]], region, Ms, preds
        )
        for p in params
    }
    ress = {
        infos[p]["label"]: _residual(
            preds[infos[p]["label"]],
            obs[infos[p]["label"]].values,
            infos[p]["kind"],
        )
        for p in params
    }
    return {
        "params": params,
        "infos": infos,
        "labels": [infos[p]["label"] for p in params],
        "obs": obs,
        "preds": preds,
        "aeqs": aeqs,
        "ress": ress,
        "lon0": lon0,
        "lat0": lat0,
        "strike": strike,
        "rc": rc,
    }


# ==================== 主绘图函数 ====================


def plot_cea2019_vs_obs(
    data,
    epicenter,
    Ms,
    region,
    strike,
    params=(-1, -2, 0.3, 1.0, "Intensity"),
    extent=400.0,
    max_dist=200.0,
    outpath="CEA2019_vs_Obs.png",
    param_cols=None,
    grid_n=80,
    axis="长轴",
    dpi=150,
):
    """
    基于给定震中，用 CEA2019 预测并绘制"预测 vs 实测"4×N 综合图。

    参数：
        data       实测数据：文件路径或 DataFrame
        epicenter  初始震中 (lon, lat)
        Ms         面波震级
        region     分区（"青藏区"等，不带"区"也可以）
        strike     走向（正北=0°，顺时针）
        params     要绘制的参数列表，按周期点：
                   -1/0=PGA，-2=PGV，数值周期=PSA(T=xxx s)，"Intensity"=烈度
        extent     CEA2019 场最大距离 km（默认 400）
        max_dist   GB18306/CEA2019 有效范围上限（默认 200 km），超过浅灰填充
        outpath    输出 PNG 路径
        param_cols 参数名 → 实测列名映射（缺省自动识别）
        grid_n     预测云图网格点数（每边）
        axis       曲线与散点距离轴："长轴"（默认）或 "短轴"，
                   选长轴就用各散点所在椭圆长轴距 a_eq 和长轴曲线；
                   选短轴则全部换成短轴距 b_eq 和短轴曲线。
        dpi        输出 PNG 分辨率（默认 150，可调大保证清晰度）
    """
    if axis not in ("长轴", "短轴"):
        raise ValueError("axis 只能是 '长轴' 或 '短轴'")
    C = _compute_vs_obs(
        data, epicenter, Ms, region, strike, params, extent, param_cols
    )
    params, infos, labels = C["params"], C["infos"], C["labels"]
    obs, preds, aeqs, ress = C["obs"], C["preds"], C["aeqs"], C["ress"]
    lon0, lat0, strike, rc = C["lon0"], C["lat0"], C["strike"], C["rc"]

    # ---- 台站标记：区分 EI（烈度计，三角）/ HN（强震仪，圆）----
    itype = obs["Instrument_Type"].str.upper().values
    has_ei = "EI" in itype
    has_hn = "HN" in itype
    use_markers = has_ei and has_hn

    # ---- 全局样式 + USGS 色标 ----
    apply_style()
    plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
    cmap = ListedColormap(USGS_MMI_COLORS, name="usgs_mmi")
    cmap.set_under(USGS_MMI_COLORS[0])
    cmap.set_over(USGS_MMI_COLORS[-1])

    n = len(params)
    fig, axes = plt.subplots(4, n, figsize=(4.0 * n, 15.5))
    if n == 1:
        axes = axes.reshape(4, 1)

    utm_zone = int((lon0 + 180.0) // 6.0) + 1

    for i, p in enumerate(params):
        info = infos[p]
        label, kind, T = info["label"], info["kind"], info["T"]
        title = f"{label} ({info['unit']})" if info["unit"] else label
        levels = _param_levels(p, Ms, rc)
        norm = BoundaryNorm(levels, ncolors=len(USGS_MMI_COLORS))
        tick_positions, tick_labels = _level_ticks(p, levels)

        pred = preds[label]
        a_eq, b_eq = aeqs[label]
        res = ress[label]
        valid = np.isfinite(pred) & np.isfinite(obs[label].values)

        # ================= 排1：预测云图 + 实测散点 =================
        ax = axes[0, i]
        # 经纬度区间按公里取齐：东西/南北半宽取较大者，保证图幅近似方形
        c_lon = (obs["lon"].min() + obs["lon"].max()) / 2.0
        c_lat = (obs["lat"].min() + obs["lat"].max()) / 2.0
        clon_km = 111.32 * math.cos(math.radians(c_lat))
        dlon_km = (obs["lon"].max() - obs["lon"].min()) * clon_km
        dlat_km = (obs["lat"].max() - obs["lat"].min()) * 110.57
        half_km = max(dlon_km, dlat_km) / 2.0 * 1.25
        half_km = max(half_km, 40.0)
        glon = np.linspace(
            c_lon - half_km / clon_km, c_lon + half_km / clon_km, grid_n
        )
        glat = np.linspace(
            c_lat - half_km / 110.57, c_lat + half_km / 110.57, grid_n
        )
        GLON, GLAT = np.meshgrid(glon, glat)
        pred_grid = _predict_one(
            p,
            lon0,
            lat0,
            strike,
            region,
            Ms,
            GLON.ravel(),
            GLAT.ravel(),
            extent,
        ).reshape(GLON.shape)
        cf = ax.contourf(
            GLON,
            GLAT,
            pred_grid,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        cb = fig.colorbar(
            cf, ax=ax, ticks=tick_positions, pad=0.03, shrink=0.85
        )
        cb.ax.set_yticklabels(tick_labels)
        cb.set_label(title, fontsize=8)

        # 实测散点（同一色标着色；按仪器类型区分形状）
        sel = valid & obs[label].notna().values
        if use_markers:
            for mk, name in (("HN", "强震仪"), ("EI", "烈度计")):
                m = sel & (itype == mk)
                if m.any():
                    ax.scatter(
                        obs["lon"][m],
                        obs["lat"][m],
                        c=obs[label][m],
                        cmap=cmap,
                        norm=norm,
                        marker="o" if mk == "HN" else "^",
                        s=38,
                        edgecolors="k",
                        linewidths=0.4,
                        zorder=6,
                        label=name,
                    )
        else:
            ax.scatter(
                obs["lon"][sel],
                obs["lat"][sel],
                c=obs[label][sel],
                cmap=cmap,
                norm=norm,
                marker="o",
                s=38,
                edgecolors="k",
                linewidths=0.4,
                zorder=6,
            )

        ax.plot(lon0, lat0, "k*", markersize=12, zorder=10)
        sr = math.radians(strike)
        arr_lon, arr_lat = km_to_lonlat(
            lon0, lat0, 80.0 * math.sin(sr), 80.0 * math.cos(sr), utm_zone
        )
        ax.annotate(
            "",
            xy=(arr_lon, arr_lat),
            xytext=(lon0, lat0),
            arrowprops=dict(arrowstyle="->", color="k", lw=1.1),
        )
        ax.set_xlim(glon[0], glon[-1])
        ax.set_ylim(glat[0], glat[-1])
        ax.set_aspect(1.0 / math.cos(math.radians((glat[0] + glat[-1]) / 2.0)))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("经度 (°E)", fontsize=8)
        ax.set_ylabel("纬度 (°N)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        if use_markers:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        # ================= 排2：衰减曲线（log 距离轴）=================
        ax = axes[1, i]
        # 散点距离轴：长轴用 a_eq，短轴用 b_eq
        dist = a_eq if axis == "长轴" else b_eq
        # 距离上限：最远椭圆距向上取整到 100 的整数 +100，最小 200 km
        far = float(np.nanmax(dist[valid])) if valid.any() else 0.0
        a_max = max(200.0, math.ceil(far / 100.0) * 100.0 + 100.0)
        r_scan = np.arange(1.0, a_max + 1.0, 1.0)
        if kind == "gmm":
            lm, ll, lu = _period_curves_sigma(T, Ms, rc, axis, r_scan)
        else:
            pg_m, pg_lo, pg_up = _period_curves_sigma(-1, Ms, rc, axis, r_scan)
            pv_m, pv_lo, pv_up = _period_curves_sigma(-2, Ms, rc, axis, r_scan)
            lm = CAL_INT.cal_Intensity_matrix(pg_m, pv_m)
            ll = CAL_INT.cal_Intensity_matrix(pg_lo, pv_lo)
            lu = CAL_INT.cal_Intensity_matrix(pg_up, pv_up)

        ax.axvspan(max_dist, a_max, color="lightgray", alpha=0.55, zorder=0)
        ax.axvline(max_dist, color="0.4", ls=":", lw=1.0, zorder=1)
        ax.plot(r_scan, lm, color="tab:red", lw=1.5, label=f"{axis}中值")
        ax.fill_between(
            r_scan, ll, lu, color="tab:red", alpha=0.15, label=f"{axis}±1σ"
        )
        ax.set_xscale("log")
        if kind == "gmm":
            ax.set_yscale("log")
        # 实测散点（x = 所在椭圆 长轴距/短轴距）
        sel = valid
        if use_markers:
            for mk, mk_sym in (("HN", "o"), ("EI", "^")):
                m = sel & (itype == mk)
                if m.any():
                    ax.scatter(
                        dist[m],
                        obs[label][m],
                        marker=mk_sym,
                        s=28,
                        facecolor="none",
                        edgecolors="k",
                        linewidths=0.7,
                        zorder=5,
                    )
        else:
            ax.scatter(
                dist[sel],
                obs[label][sel],
                marker="o",
                s=28,
                facecolor="none",
                edgecolors="k",
                linewidths=0.7,
                zorder=5,
            )
        ax.set_xlim(1.0, a_max)
        ax.set_xlabel(f"等效{axis}距 (km)", fontsize=8)
        ax.set_ylabel("预测 / 实测值", fontsize=8)
        ax.set_title(f"{title} 衰减曲线", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

        # ================= 排3：残差（线性距离轴）=================
        ax = axes[2, i]
        # 距离上限：最远椭圆距向上取整到 50 的倍数（最小 200 km），与排2不同
        far3 = float(np.nanmax(dist[valid])) if valid.any() else 0.0
        x_max3 = max(200.0, math.ceil(far3 / 50.0) * 50.0)
        ax.axvspan(max_dist, x_max3, color="lightgray", alpha=0.55, zorder=0)
        ax.axvline(max_dist, color="0.4", ls=":", lw=1.0, zorder=1)
        ax.axhline(0, color="k", lw=1.0, zorder=2)
        if kind == "gmm":
            sigma_ln = _period_coeffs(rc, "长轴", T, Ms)[3] * math.log(10.0)
            ax.axhline(sigma_ln, color="gray", lw=0.8, ls="--")
            ax.axhline(
                -sigma_ln,
                color="gray",
                lw=0.8,
                ls="--",
                label=f"±1σ = ±{sigma_ln:.3f} (ln)",
            )
        if use_markers:
            for mk, mk_sym in (("HN", "o"), ("EI", "^")):
                m = valid & (itype == mk)
                if m.any():
                    ax.scatter(
                        dist[m],
                        res[m],
                        marker=mk_sym,
                        s=26,
                        facecolor="none",
                        edgecolors="k",
                        linewidths=0.7,
                        zorder=5,
                    )
        else:
            ax.scatter(
                dist[valid],
                res[valid],
                marker="o",
                s=26,
                facecolor="none",
                edgecolors="k",
                linewidths=0.7,
                zorder=5,
            )
        ax.set_xlim(0, x_max3)
        ax.set_xlabel(f"等效{axis}距 (km)", fontsize=8)
        ax.set_ylabel("残差（预测-实测）", fontsize=8)
        ax.set_title(f"{title} 残差-距离", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        if kind == "gmm":
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        # ================= 排4：残差分布（三组）=================
        ax = axes[3, i]
        groups = [
            ("全部", res[valid]),
            ("<200 km", res[valid & (dist < max_dist)]),
            ("≥200 km", res[valid & (dist >= max_dist)]),
        ]
        colors_g = ["#1f77b4", "#2ca02c", "#d62728"]
        for xpos, (gname, gdata), gcol in zip(range(3), groups, colors_g):
            half_violin_box_scatter(
                ax, gdata, xpos, gcol, value_fmt="{:.3f}", s=18
            )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["全部", "<200 km", "≥200 km"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel("残差（预测-实测）", fontsize=8)
        ax.set_title(f"{title} 残差分布", fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        # 排4注释自适应：按数据跨度预留头部空间（免整图重绘，提速）
        y0, y1 = ax.get_ylim()
        span = max(y1 - y0, 1e-9)
        ax.set_ylim(y0 - 0.06 * span, y1 + 0.55 * span)

    # ---- 总标题 + 使用案例文本框 ----
    fig.suptitle(
        f"CEA2019 预测 vs 实测  |  震中 ({lon0:.3f}, {lat0:.3f})"
        f"  |  strike {strike:.0f}°  |  Ms {Ms:g}  |  {region}",
        fontsize=14,
    )
    fig.subplots_adjust(
        left=0.05, right=0.975, top=0.93, bottom=0.05,
        wspace=0.38, hspace=0.42,
    )
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print(f"已保存图件：{os.path.abspath(outpath)}")
    return outpath


def _fmt_num(v, nd=6):
    """数值格式化：NaN/非有限值 → 'NaN'；浮点保留 nd 位有效数字"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "NaN"
    if not np.isfinite(v):
        return "NaN"
    return f"{v:.{nd}g}"


def export_cea2019_vs_obs_txt(
    data,
    epicenter,
    Ms,
    region,
    strike,
    params=(-1, -2, 0.3, 1.0, "Intensity"),
    extent=400.0,
    max_dist=200.0,
    outpath="CEA2019_vs_Obs_stats.txt",
    param_cols=None,
):
    """
    输出 1 个大的 txt 统计文档（与绘图共用同一套计算）。
    第一行即表头，可直接 pd.read_csv(outpath, sep='\\t') 读取：
        台站信息（Sta_ID / Sta_longi / Sta_lati / Instrument_Type）
        + 每个参数 5 列：
            <参数>_obs            实测值
            <参数>_pred           预测值
            Repi_long_<参数>(km)  预测值对应等值椭圆长轴距
            Repi_short_<参数>(km) 预测值对应等值椭圆短轴距
            <参数>_res            残差
    残差定义与图一致：PGA/PGV/PSA 用 ln(预测/实测)（自然对数）；
    烈度用 预测 - 实测（线性）。烈度的椭圆沿用 PGA。
    """
    C = _compute_vs_obs(
        data, epicenter, Ms, region, strike, params, extent, param_cols
    )
    obs, labels = C["obs"], C["labels"]
    preds, aeqs, ress = C["preds"], C["aeqs"], C["ress"]

    lines = []
    cols = ["Sta_ID", "Sta_longi", "Sta_lati", "Instrument_Type"]
    for label in labels:
        cols += [
            f"{label}_obs",
            f"{label}_pred",
            f"Repi_long_{label}(km)",
            f"Repi_short_{label}(km)",
            f"{label}_res",
        ]
    lines.append("\t".join(cols))

    for j in range(len(obs)):
        row = [
            str(obs["Sta_ID"][j]),
            f"{obs['lon'][j]:.4f}",
            f"{obs['lat'][j]:.4f}",
            str(obs["Instrument_Type"][j]) or "-",
        ]
        for label in labels:
            a, b = aeqs[label]
            row += [
                _fmt_num(obs[label][j]),
                _fmt_num(preds[label][j]),
                _fmt_num(a[j]),
                _fmt_num(b[j]),
                _fmt_num(ress[label][j]),
            ]
        lines.append("\t".join(row))

    with open(outpath, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存统计表：{os.path.abspath(outpath)}")
    return outpath


# ---- 测试
if __name__ == "__main__":

    plot_cea2019_vs_obs(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        epicenter=(87.45, 28.5),
        Ms=6.8,
        region="青藏",
        strike=187,
        params=(-1, -2, 0.3, 1.0, 3, 6),
        extent=500.0,
        max_dist=200.0,
        outpath="CEA2019_vs_Obs.png",
        grid_n=100,
        axis="短轴",
    )
    export_cea2019_vs_obs_txt(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        epicenter=(87.45, 28.5),
        Ms=6.8,
        region="青藏",
        strike=187,
        params=(-1, -2, 0.3, 1.0, 3, 6),
        extent=500.0,
        max_dist=200.0,
        outpath="CEA2019_vs_Obs_stats.txt",
    )
