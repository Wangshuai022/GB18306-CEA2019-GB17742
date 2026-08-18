"""
CEA2019 预测 vs 实测 —— 基于给定震中的 4×N 综合绘图与统计表
============================================================
输入实测台站、初始震中、震级、分区、走向，用 CEA2019 衰减模型预测各台站
地震动参数，与实测对比，输出一张 "4 排 × N 列"图（N = 参数个数）：

    排1  预测云图（观测点范围内，经纬度按公里取齐成近似方形）+ 实测散点
    排2  衰减曲线（所选轴中值 ±1σ），X 距离 log 轴，1 ~ 最远椭圆距
         （向上取整到 100 的整数 +100，最小 200 km），>200 km 浅灰填充
    排3  残差（预测-实测，PGA/PGV/PSA 用自然对数 ln(Pred/Obs)，
         烈度用线性差），X 距离线性轴，上限向上取整到 50 的倍数（最小 200 km）
    排4  残差分布三组：全部 / <200 km / ≥200 km（散点+半小提琴+箱线）

参数按周期点定义（N 列）：
    -1 或 0   → PGA（gal）
    -2        → PGV（cm/s）
    数值周期  → PSA(T=0.30s) 等（cm/s²），标签保留 2 位小数
    "Intensity" → 烈度（特殊：由 CEA2019 预测的 PGA/PGV 按 GB/T 17742 换算）

台站标记：数据含 Instrument_Type 且同时出现 EI（烈度计）/ HN（强震仪）时，
HN 用圆点、EI 用三角形；未区分则全部圆点。
色标引用 CEA2019_pre.py：PGA/PSA 用 PGA 分界（PSA 按周期缩放），
PGV 用 PGA÷10，烈度用 USGS MMI 十色 1~10 度。

使用案例：
    from CEA2019_vs_Obs import plot_cea2019_vs_obs, export_cea2019_vs_obs_txt
    plot_cea2019_vs_obs(
        data="台站文件.txt", macro_epicenter=(87.378, 28.604),
        Ms=6.8, region="青藏区", strike=349.0,
        params=(-1, -2, 0.3, 1.0, "Intensity"),
        outpath="CEA2019_vs_Obs.png",
    )
    export_cea2019_vs_obs_txt(...)   # 台站统计表（第一行即表头）
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from CEA2019_pre import (
    USGS_MMI_COLORS,
    _ellipse_radii_for_period,
    _period_coeffs,
    _period_curves_sigma,
    _period_levels,
    _region_core,
    fmt_pgv,
    fmt_psa,
    km_to_lonlat,
    period_label,
    predict_period_values,
    unit_label,
)
from ellipse_fields import CEA2019EllipseField
from GB17742_class import GB17742_2020_Cal_instrument_intensity as CAL_INT
from stat_violin import (
    apply_style,
    fit_annotations_inside,
    half_violin_box_scatter,
)

# ==================== 参数规范（按周期点） ====================


def normalize_params(params):
    """
    输入参数 → 统一列表：
        -1 / 0              → -1（PGA）
        -2                  → -2（PGV）
        0.01 0.1 0.3 1 3 6  → 原周期（PSA）
        字符串 "PGA"/"PGV"/"PSA(T=0.30s)" 同样接受
        "Intensity"         → 烈度
    自动去重（如同时传 -1 和 "PGA"）。
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
                item = float(s.split("(")[1].split("=")[1].rstrip("s)"))
            else:
                raise ValueError(f"无法识别的参数：{p!r}")
        else:
            item = -1.0 if float(p) == 0.0 else float(p)
        if item not in out:
            out.append(item)
    return out


def param_info(p):
    """规范参数 → {label, kind, T, unit}；label 复用 period_label（两位小数）"""
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
    """实测列候选：RotD50 → H → 有效值 RotD50 → 有效值 H。"""
    if info["kind"] == "intensity":
        return ["I"]
    if info["T"] == -1:
        return ["PGA_RotD50", "PGA_H", "EPA_RotD50", "EPA_H"]
    if info["T"] == -2:
        return ["PGV_RotD50", "PGV_H", "EPV_RotD50", "EPV_H"]
    tag = f"pSa(T={info['T']:.2f}s)"
    return [f"{tag}_RotD50", f"{tag}_H"]


# ==================== 数据读取 ====================


def load_obs_data(data, params, param_cols=None):
    """读取并规范化 CEA2019 预测—观测对比所需的台站数据。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        制表符分隔文件或内存表。经纬度支持 ``longi/lati``、``lon/lat``。
    params : sequence
        -1=PGA、-2=PGV、正数=PSA 周期（s）、``"Intensity"``=仪器烈度。
    param_cols : dict or None
        可选显式列映射。默认 PGA/PGV 按 RotD50、H、有效值 RotD50、
        有效值 H 选择，PSA 按 RotD50、H 选择。

    Returns
    -------
    pandas.DataFrame
        标准化台站表。非正地震动值转为 NaN，但不会因单个参数缺测删除整行；
        原始列来源保存在 ``attrs["source_columns"]``。

    Raises
    ------
    TypeError
        ``data`` 类型不支持。
    ValueError
        坐标列或请求的观测列缺失，或没有有效台站坐标。
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
    source_columns = {}
    for p in params:
        info = param_info(p)
        label = info["label"]
        cands = (
            [param_cols[label]]
            if param_cols is not None and label in param_cols
            else obs_col_candidates(info)
        )
        col = next((c for c in cands if c in df.columns), None)
        if col is None:
            raise ValueError(
                f"参数 {label}（周期 {info['T']}）找不到实测列，候选：{cands}"
            )
        out[label] = pd.to_numeric(df[col], errors="coerce")
        if info["kind"] == "gmm":
            out[label] = out[label].where(out[label] > 0)
        source_columns[label] = col
    out = out.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("没有经纬度有效的观测台站")
    out.attrs["source_columns"] = source_columns
    return out


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
    预测值所在等值椭圆的长轴距 a / 短轴距 b（km）。
    PGA/PGV/PSA 用各自预测值反算；烈度沿用 PGA 椭圆（烈度由 PGA/PGV 换算）。
    """
    rc = _region_core(region)
    info = param_info(param)
    if info["kind"] == "gmm":
        a, b = _ellipse_radii_for_period(pred, info["T"], Ms, rc)
    else:
        a, b = _ellipse_radii_for_period(preds["PGA"], -1, Ms, rc)
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def _residual(pred, obs, kind):
    """残差 = 预测 - 实测：PGA/PGV/PSA 用 ln(Pred/Obs)；烈度用线性差"""
    if kind == "gmm":
        return np.log(
            np.asarray(pred, dtype=float) / np.asarray(obs, dtype=float)
        )
    return np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)


# ==================== 色标 ====================


def _param_levels(param, Ms, region_core):
    """色标分界：PGA/PSA 用 _period_levels（PSA 按周期缩放），烈度用 MMI 十色 1~10"""
    info = param_info(param)
    if info["kind"] == "gmm":
        return _period_levels(info["T"], Ms, region_core)
    return np.arange(0.5, 11.5, 1.0)


def _level_ticks(param, levels):
    """(刻度位置, 刻度文字)：PGA 整数 / PGV 按 fmt_pgv / PSA 按 fmt_psa / 烈度整数"""
    info = param_info(param)
    if info["kind"] == "intensity":
        pos = np.arange(1, 11)
        return pos, [f"{v:.0f}" for v in pos]
    if info["T"] == -2:
        return levels, [fmt_pgv(v) for v in levels]
    if info["T"] in (-1, 0):
        return levels, [f"{v:g}" for v in levels]
    return levels, [fmt_psa(v) for v in levels]


# ==================== 共享计算（绘图与 txt 共用，保证一致） ====================


def _compute_vs_obs(
    data, epicenter, Ms, region, strike, params, extent, param_cols=None
):
    """集中计算 CEA2019 的观测、预测、椭圆距和残差。

    这是绘图与文本导出的共享实现，保证两种输出使用完全相同的观测列、
    椭圆场和残差定义。``epicenter`` 为宏观震中经纬度，``Ms`` 为模型震级，
    ``strike`` 单位为度，``extent`` 单位为 km。返回字典包含规范周期、观测表、
    各参数预测、长短轴距、残差、共享 ``CEA2019EllipseField`` 和列来源。
    地震动残差为 ``ln(pred/obs)``，烈度残差为 ``pred-obs``。
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

    periods = [infos[p]["T"] for p in params if infos[p]["kind"] == "gmm"]
    if any(infos[p]["kind"] == "intensity" for p in params):
        periods.extend([-1.0, -2.0])
    periods = list(dict.fromkeys(periods))
    field = CEA2019EllipseField(region, Ms, extent=extent)
    raw = field.predict(
        periods,
        lon0,
        lat0,
        strike,
        obs["lon"].values,
        obs["lat"].values,
    )
    preds, aeqs = {}, {}
    for p in params:
        info = infos[p]
        label = info["label"]
        if info["kind"] == "gmm":
            pred, a_eq, b_eq = raw[float(info["T"])]
        else:
            pred = CAL_INT.cal_Intensity_matrix(raw[-1.0][0], raw[-2.0][0])
            a_eq, b_eq = raw[-1.0][1], raw[-1.0][2]
        preds[label] = pred
        aeqs[label] = (a_eq, b_eq)
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
        "field": field,
        "source_columns": obs.attrs.get("source_columns", {}),
    }


# ==================== 主绘图函数 ====================


def plot_cea2019_vs_obs(
    data,
    macro_epicenter,
    Ms,
    region,
    strike,
    params=(-1, -2, 0.3, 1.0, "Intensity"),
    extent=400.0,
    max_dist=200.0,
    outpath="CEA2019_vs_Obs.png",
    param_cols=None,
    grid_n=100,
    axis="长轴",
    initial_epicenter=None,
    fault_lon_mat=None,
    fault_lat_mat=None,
    plot_observations=None,
):
    """绘制 CEA2019 的 4×N 预测—观测综合图。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        台站观测。DataFrame attrs 可携带 Vs30 绘图状态，标题会明确标注。
    macro_epicenter : tuple(float, float)
        用于前向预测的宏观震中经纬度。
    Ms : float
        CEA2019 衰减关系使用的震级。
    region : str
        CEA2019 分区名称。
    strike : float
        椭圆长轴走向，单位度。
    params : sequence
        -1=PGA、-2=PGV、正数=PSA 周期、``"Intensity"``=仪器烈度。
    extent : float, default 400
        椭圆场最大范围，单位 km。
    max_dist : float, default 200
        近场/远场残差分组阈值，单位 km。
    outpath : str or os.PathLike
        PNG 输出路径。
    param_cols : dict or None
        可选显式观测列映射。
    grid_n : int, default 100
        每个方向的绘图网格点数。
    axis : {"长轴", "短轴"}, default "长轴"
        衰减曲线横坐标采用的等效距离。
    initial_epicenter : tuple(float, float) or None
        初始破裂点，仅作地图标记。
    fault_lon_mat, fault_lat_mat : array-like or None
        二维断层网格经纬度，必须同时提供或同时省略。
    plot_observations : {None, "corrected", "raw"}, default None
        场地修正观测的绘图模式。None 表示原样使用 ``data``；``corrected``
        使用参考 Vs30 观测；``raw`` 从 ``*_raw`` 审计列恢复原始场地观测。
        仅改变图中的观测点和绘图残差，不改变任何震中反演结果。

    Returns
    -------
    str or os.PathLike
        原样返回 ``outpath``。四排依次为地图、衰减曲线、残差—距离和
        全部/近场/远场残差分布。

    Raises
    ------
    ValueError
        轴类型非法、断层矩阵未成对提供或请求参数/观测列无效。
    """
    if axis not in ("长轴", "短轴"):
        raise ValueError("axis 只能是 '长轴' 或 '短轴'")
    if (fault_lon_mat is None) != (fault_lat_mat is None):
        raise ValueError(
            "fault_lon_mat 与 fault_lat_mat 必须同时提供或同时省略"
        )
    if plot_observations is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "plot_observations 只能用于包含场地修正审计列的 pandas.DataFrame；"
                "文件路径输入请先调用 correct_observations_to_reference_vs30"
            )
        from Vs30_site_correction import prepare_site_plot_observations

        data = prepare_site_plot_observations(data, plot_observations)
    C = _compute_vs_obs(
        data, macro_epicenter, Ms, region, strike, params, extent, param_cols
    )
    params, infos = C["params"], C["infos"]
    obs, preds, aeqs, ress = C["obs"], C["preds"], C["aeqs"], C["ress"]
    lon0, lat0, strike, rc = C["lon0"], C["lat0"], C["strike"], C["rc"]
    field = C["field"]

    # 台站标记：区分 EI（烈度计，三角）/ HN（强震仪，圆）
    itype = obs["Instrument_Type"].str.upper().values
    use_markers = ("EI" in itype) and ("HN" in itype)

    # 全局样式 + USGS 色标
    apply_style()
    plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]
    cmap = ListedColormap(USGS_MMI_COLORS, name="usgs_mmi")
    cmap.set_under(USGS_MMI_COLORS[0])
    cmap.set_over(USGS_MMI_COLORS[-1])

    n = len(params)
    fig, axes = plt.subplots(4, n, figsize=(4.4 * n, 17))
    if n == 1:
        axes = axes.reshape(4, 1)
    utm_zone = int((lon0 + 180.0) // 6.0) + 1

    # 所有参数共用同一个绘图网格；周期场只计算一次并在各列复用。
    c_lon = (obs["lon"].min() + obs["lon"].max()) / 2.0
    c_lat = (obs["lat"].min() + obs["lat"].max()) / 2.0
    clon_km = 111.32 * math.cos(math.radians(c_lat))
    half_km = (
        max(
            (obs["lon"].max() - obs["lon"].min()) * clon_km,
            (obs["lat"].max() - obs["lat"].min()) * 110.57,
        )
        / 2.0
        * 1.25
    )
    half_km = max(half_km, 40.0)
    glon = np.linspace(
        c_lon - half_km / clon_km, c_lon + half_km / clon_km, grid_n
    )
    glat = np.linspace(
        c_lat - half_km / 110.57, c_lat + half_km / 110.57, grid_n
    )
    GLON, GLAT = np.meshgrid(glon, glat)
    grid_periods = [infos[p]["T"] for p in params if infos[p]["kind"] == "gmm"]
    if any(infos[p]["kind"] == "intensity" for p in params):
        grid_periods.extend([-1.0, -2.0])
    grid_periods = list(dict.fromkeys(grid_periods))
    grid_raw = field.predict(
        grid_periods, lon0, lat0, strike, GLON.ravel(), GLAT.ravel()
    )
    grid_values = {}
    for p in params:
        info = infos[p]
        if info["kind"] == "gmm":
            values = grid_raw[float(info["T"])][0]
        else:
            values = CAL_INT.cal_Intensity_matrix(
                grid_raw[-1.0][0], grid_raw[-2.0][0]
            )
        grid_values[info["label"]] = values.reshape(GLON.shape)

    for i, p in enumerate(params):
        info = infos[p]
        label, kind, T = info["label"], info["kind"], info["T"]
        title = f"{label} ({info['unit']})" if info["unit"] else label
        levels = _param_levels(p, Ms, rc)
        norm = BoundaryNorm(levels, ncolors=len(USGS_MMI_COLORS))
        tick_positions, tick_labels = _level_ticks(p, levels)

        pred, a_eq, b_eq, res = preds[label], *aeqs[label], ress[label]
        valid = np.isfinite(pred) & np.isfinite(obs[label].values)

        # ================= 排1：预测云图 + 实测散点 =================
        ax = axes[0, i]
        pred_grid = grid_values[label]
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

        # 断层投影（可选）：边界 + 上缘/下缘
        if fault_lon_mat is not None and fault_lat_mat is not None:
            flon = np.asarray(fault_lon_mat, dtype=float)
            flat = np.asarray(fault_lat_mat, dtype=float)
            poly_lon = np.concatenate(
                [flon[0], flon[:, -1], flon[-1][::-1], flon[:, 0][::-1]]
            )
            poly_lat = np.concatenate(
                [flat[0], flat[:, -1], flat[-1][::-1], flat[:, 0][::-1]]
            )
            ax.fill(poly_lon, poly_lat, color="0.85", alpha=0.55, zorder=2)

            ax.plot(
                flon[0], flat[0], color="r", lw=2.5, zorder=3, label="断层上缘"
            )
            ax.plot(
                flon[-1],
                flat[-1],
                color="b",
                lw=0.8,
                zorder=3,
                label="断层下缘",
            )
            ax.plot(flon[:, 0], flat[:, 0], color="k", lw=0.8, zorder=3)
            ax.plot(flon[:, -1], flat[:, -1], color="k", lw=0.8, zorder=3)

        sel = valid & obs[label].notna().values
        if use_markers:
            for mk, name, sym in (
                ("HN", "强震仪", "o"),
                ("EI", "烈度计", "^"),
            ):
                m = sel & (itype == mk)
                if m.any():
                    ax.scatter(
                        obs["lon"][m],
                        obs["lat"][m],
                        c=obs[label][m],
                        cmap=cmap,
                        norm=norm,
                        marker=sym,
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

        ax.plot(lon0, lat0, "k*", markersize=9, zorder=10, label="宏观震中")
        if initial_epicenter is not None:
            ax.plot(
                initial_epicenter[0],
                initial_epicenter[1],
                "*",
                markersize=9,
                color="magenta",
                zorder=10,
                label="初始破裂点",
            )
        sr = math.radians(strike)
        arr_lon, arr_lat = km_to_lonlat(
            lon0, lat0, 80.0 * math.sin(sr), 80.0 * math.cos(sr), utm_zone
        )
        ax.annotate(
            "",
            xy=(arr_lon, arr_lat),
            xytext=(lon0, lat0),
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.1},
        )
        ax.set_xlim(glon[0], glon[-1])
        ax.set_ylim(glat[0], glat[-1])
        ax.set_aspect(1.0 / math.cos(math.radians((glat[0] + glat[-1]) / 2.0)))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("经度 (°E)", fontsize=8)
        ax.set_ylabel("纬度 (°N)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

        # ================= 排2：衰减曲线（log 距离轴）=================
        ax = axes[1, i]
        dist = a_eq if axis == "长轴" else b_eq  # 散点距离轴
        far = float(np.nanmax(dist[valid])) if valid.any() else 0.0
        a_max = max(
            200.0, math.ceil(far / 100.0) * 100.0 + 100.0
        )  # 取整到100+100，最小200
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
        if use_markers:
            for mk, sym in (("HN", "o"), ("EI", "^")):
                m = valid & (itype == mk)
                if m.any():
                    ax.scatter(
                        dist[m],
                        obs[label][m],
                        marker=sym,
                        s=28,
                        facecolor="none",
                        edgecolors="k",
                        linewidths=0.7,
                        zorder=5,
                    )
        else:
            ax.scatter(
                dist[valid],
                obs[label][valid],
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
        far3 = float(np.nanmax(dist[valid])) if valid.any() else 0.0
        x_max3 = max(
            200.0, math.ceil(far3 / 50.0) * 50.0
        )  # 取整到50的倍数，最小200
        ax.axvspan(max_dist, x_max3, color="lightgray", alpha=0.55, zorder=0)
        ax.axvline(max_dist, color="0.4", ls=":", lw=1.0, zorder=1)
        ax.axhline(0, color="k", lw=1.0, zorder=2)
        if kind == "gmm":
            sigma_ln = _period_coeffs(rc, "长轴", T, Ms)[3] * math.log(10.0)
            ax.axhline(sigma_ln, color="r", lw=1.3, ls="-.")
            ax.axhline(
                -sigma_ln,
                color="r",
                lw=1.3,
                ls="-.",
                label=f"±1σ = ±{sigma_ln:.3f} (ln)",
            )
        if use_markers:
            for mk, sym in (("HN", "o"), ("EI", "^")):
                m = valid & (itype == mk)
                if m.any():
                    ax.scatter(
                        dist[m],
                        res[m],
                        marker=sym,
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
        ax.axhline(0, color="k", lw=1.0, zorder=0)  # 零线放底层
        for xpos, (_, gdata), gcol in zip(range(3), groups, colors_g):
            half_violin_box_scatter(
                ax, gdata, xpos, gcol, value_fmt="{:.3f}", s=18
            )
        if kind == "gmm":
            ax.axhline(sigma_ln, color="r", lw=1.3, ls="-.")
            ax.axhline(-sigma_ln, color="r", lw=1.3, ls="-.")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["全部", "<200 km", "≥200 km"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel("残差（预测-实测）", fontsize=8)
        ax.set_title(f"{title} 残差分布", fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 排4注释自适应：把 N/μ/m 文本框收进各自子图
    fig.canvas.draw()
    for i in range(n):
        fit_annotations_inside(axes[3, i], fig=fig, draw=False)

    title_parts = [
        "CEA2019 预测 vs 实测",
        f"宏观震中 ({lon0:.3f}, {lat0:.3f})",
        f"strike {strike:.0f}°",
        f"Ms {Ms:g}",
        region,
    ]
    if initial_epicenter is not None:
        title_parts.insert(
            2,
            f"初始破裂点 ({initial_epicenter[0]:.3f}, {initial_epicenter[1]:.3f})",
        )
    if isinstance(data, pd.DataFrame):
        site_plot_mode = data.attrs.get("site_plot_observations")
        if site_plot_mode == "raw":
            title_parts.append(
                "原始场地观测（仅绘图，反演仍使用 Vs30 修正值）"
            )
        elif "site_reference_vs30" in data.attrs:
            site_model = data.attrs.get("site_correction_model", "场地模型")
            site_vs30 = float(data.attrs["site_reference_vs30"])
            title_parts.append(
                f"观测已按 {site_model} 统一至 Vs30={site_vs30:g} m/s"
            )
    fig.suptitle("  |  ".join(title_parts), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图件：{os.path.abspath(outpath)}")
    return outpath


# ==================== 统计表导出 ====================


def _fmt_num(v, nd=6):
    """数值格式化：NaN/非有限值 → 'NaN'；浮点保留 nd 位有效数字"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "NaN"
    return "NaN" if not np.isfinite(v) else f"{v:.{nd}g}"


def export_cea2019_vs_obs_txt(
    data,
    macro_epicenter,
    Ms,
    region,
    strike,
    params=(-1, -2, 0.3, 1.0, "Intensity"),
    extent=400.0,
    max_dist=200.0,
    outpath="CEA2019_vs_Obs_stats.txt",
    param_cols=None,
):
    """导出 CEA2019 逐台站预测—观测统计表。

    Parameters
    ----------
    data, macro_epicenter, Ms, region, strike, params, extent, param_cols
        与 ``plot_cea2019_vs_obs`` 中同名参数含义一致。
    max_dist : float, default 200
        保留为与绘图接口一致的分组阈值；文本表输出全部台站。
    outpath : str or os.PathLike
        制表符分隔文本路径，使用 UTF-8 BOM 编码。

    Returns
    -------
    str or os.PathLike
        原样返回 ``outpath``。表包含台站信息以及每个参数的观测、预测、
        长轴距、短轴距和残差，可用 ``pandas.read_csv(..., sep="\\t")`` 读取。

    Notes
    -----
    PGA/PGV/PSA 残差为 ``ln(预测/实测)``；烈度残差为线性差。
    """
    C = _compute_vs_obs(
        data, macro_epicenter, Ms, region, strike, params, extent, param_cols
    )
    obs, labels = C["obs"], C["labels"]
    preds, aeqs, ress = C["preds"], C["aeqs"], C["ress"]

    cols = ["Sta_ID", "Sta_longi", "Sta_lati", "Instrument_Type"]
    for label in labels:
        cols += [
            f"{label}_obs",
            f"{label}_pred",
            f"Repi_long_{label}(km)",
            f"Repi_short_{label}(km)",
            f"{label}_res",
        ]
    lines = ["\t".join(cols)]
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


# ---- 测试 ----
if __name__ == "__main__":

    os.makedirs("Test_output", exist_ok=True)

    plot_cea2019_vs_obs(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        macro_epicenter=(87.5686, 28.9874),  # 87.5686,28.9874
        Ms=6.8,
        region="青藏",
        strike=187,
        params=(-1, -2, 0.3, 1.0, 3, 6),
        extent=500.0,
        max_dist=200.0,
        outpath="./Test_output/CEA2019_vs_Obs1111.png",
        grid_n=100,
        axis="短轴",
    )
