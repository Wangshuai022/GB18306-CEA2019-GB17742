"""
GB18306 预测 vs 实测 —— 基于给定宏观震中的 4×N 综合绘图与统计表
================================================================
用 GB18306-2015 椭圆衰减预测各台站 PGA / PGV / 烈度，与实测对比，
输出一张 "4 排 × N 列"图（N = 参数个数）：

    排1  预测云图（观测点范围内，经纬度按公里取齐成近似方形）
         + 实测散点（EI 三角 / HN 圆点，未区分则全圆点）
         + 可选断层投影（Fault 矩阵：第一行=上缘，最后一行=下缘）
         + 初始破裂点（发布值）与宏观震中（反演值）分开标记
    排2  衰减曲线（所选轴中值 ±1σ），X 距离 log 轴，
         1 ~ 最远椭圆距（向上取整到 100 的整数 +100，最小 200 km），
         >200 km 浅灰填充
    排3  残差（预测-实测：PGA/PGV 用自然对数 ln(Pred/Obs)，烈度用线性差），
         X 距离线性轴，上限向上取整到 50 的倍数（最小 200 km）
    排4  残差分布三组：全部 / <200 km / ≥200 km（散点+半小提琴+箱线）

参数按周期点定义：
    -1 或 0   → PGA（gal）
    -2        → PGV（cm/s）
    "Intensity" → 烈度（GB18306 烈度衰减直接预测）
    （GB18306 无 PSA，不支持数值周期）

色标沿用 CEA2019_pre.py 的约定：PGA 用 PGA_LEVELS、PGV 用 PGV_LEVELS
（PGA÷10）、烈度用 USGS MMI 十色 1~10 度。

使用案例：
    from GB18306_vs_Obs import plot_gb18306_vs_obs
    plot_gb18306_vs_obs(
        data="台站文件.txt",
        macro_epicenter=(87.612, 28.823),   # 宏观震中（反演得到）
        initial_epicenter=(87.45, 28.5),    # 初始破裂点（发布值）
        Ms=6.8, region="青藏区", strike=349.0,
        params=(-1, -2, "Intensity"),
        fault_lon_mat=lon_mat, fault_lat_mat=lat_mat,  # 可选断层投影
        outpath="GB18306_vs_Obs.png",
    )
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
    PGA_LEVELS,
    PGV_LEVELS,
    USGS_MMI_COLORS,
    fmt_pgv,
    km_to_lonlat,
)
from ellipse_fields import GB18306EllipseField as SharedGB18306EllipseField
from stat_violin import (
    apply_style,
    fit_annotations_inside,
    half_violin_box_scatter,
)

SIGMA_GB18306 = {"pga": 0.236, "pgv": 0.271}


# ==================== 参数规范（PGA / PGV / 烈度） ====================


def normalize_params(params):
    """
    输入参数 → 统一列表：
        -1 / 0 / "PGA"  → -1（PGA）
        -2 / "PGV"      → -2（PGV）
        "Intensity"     → 烈度
    GB18306 无 PSA，数值周期会报错。
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
            else:
                raise ValueError(f"GB18306 不支持参数 {p!r}，仅支持 PGA/PGV/Intensity")
        else:
            t = float(p)
            if t in (-1.0, 0.0):
                item = -1.0
            elif t == -2.0:
                item = -2.0
            else:
                raise ValueError(
                    f"GB18306 无 PSA，不支持周期 {p!r}；仅支持 -1(PGA)/-2(PGV)/Intensity"
                )
        if item not in out:
            out.append(item)
    return out


def param_info(p):
    """规范参数 → {label, kind, T}；kind：gmm（PGA/PGV）或 intensity"""
    if p == "Intensity":
        return {"label": "烈度", "kind": "intensity", "T": None, "unit": ""}
    t = float(p)
    return {
        "label": "PGA" if t == -1 else "PGV",
        "kind": "gmm",
        "T": t,
        "unit": "gal" if t == -1 else "cm/s",
    }


def obs_col_candidates(info):
    """实测列候选：RotD50 → H → 有效值 RotD50 → 有效值 H。"""
    if info["kind"] == "intensity":
        return ["I"]
    if info["T"] == -1:
        return ["PGA_RotD50", "PGA_H", "EPA_RotD50", "EPA_H"]
    return ["PGV_RotD50", "PGV_H", "EPV_RotD50", "EPV_H"]


# ==================== 数据读取 ====================


def load_obs_data(data, params, param_cols=None):
    """读取并规范化 GB18306 预测—观测对比所需的台站数据。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        制表符分隔文件路径或内存数据表。经纬度列可命名为 ``longi/lati``
        或 ``lon/lat``；``Sta_ID`` 和 ``Instrument_Type`` 可省略。
    params : sequence
        规范参数列表：-1=PGA、-2=PGV、``"Intensity"``=烈度。
    param_cols : dict or None
        可选显式列映射，例如 ``{"PGA": "my_pga"}``。未指定时 PGA/PGV
        严格按 RotD50、H、有效值 RotD50、有效值 H 的顺序选择。

    Returns
    -------
    pandas.DataFrame
        标准列 ``Sta_ID/lon/lat/Instrument_Type`` 加各参数观测列。非正
        PGA/PGV 转为 NaN；只删除经纬度无效的行。实际采用的原始列名保存在
        ``result.attrs["source_columns"]``。

    Raises
    ------
    TypeError
        ``data`` 既不是路径也不是 DataFrame。
    ValueError
        经纬度列或请求的观测参数列缺失，或没有有效台站坐标。
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
        raise ValueError(f"缺少 lon/lat 列，现有列：{list(df.columns)[:20]}...")
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
            raise ValueError(f"参数 {label} 找不到实测列，候选：{cands}")
        out[label] = pd.to_numeric(df[col], errors="coerce")
        if info["kind"] == "gmm":
            out[label] = out[label].where(out[label] > 0)
        source_columns[label] = col
    out = out.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("没有经纬度有效的观测台站")
    out.attrs["source_columns"] = source_columns
    return out


# 椭圆场前向模型统一由 ellipse_fields.GB18306EllipseField 提供。


# ==================== 预测 / 椭圆距 / 残差 / 色标 ====================


def predict_one(param, field, lon, lat, strike, sta_lon, sta_lat):
    """GB18306 单参数预测，返回 (预测值, 长轴距, 短轴距)"""
    I, A, V, aI, bI, aA, bA, aV, bV = field.predict(lon, lat, strike, sta_lon, sta_lat)
    if param == "Intensity":
        return I, aI, bI
    if float(param) == -1:
        return A, aA, bA
    return V, aV, bV


def residual(pred, obs, kind):
    """残差 = 预测 - 实测：PGA/PGV 用 ln(Pred/Obs)，烈度用线性差"""
    if kind == "gmm":
        return np.log(np.asarray(pred, dtype=float) / np.asarray(obs, dtype=float))
    return np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)


def param_levels(info):
    """色标分界：PGA → PGA_LEVELS，PGV → PGV_LEVELS，烈度 → MMI 十色 1~10"""
    if info["kind"] == "intensity":
        return np.arange(0.5, 11.5, 1.0)
    return PGA_LEVELS if info["T"] == -1 else PGV_LEVELS


def level_ticks(info, levels):
    """(刻度位置, 刻度文字)"""
    if info["kind"] == "intensity":
        pos = np.arange(1, 11)
        return pos, [f"{v:.0f}" for v in pos]
    if info["T"] == -1:
        return levels, [f"{v:g}" for v in levels]
    return levels, [fmt_pgv(v) for v in levels]


def compute_vs_obs(
    data, epicenter, Ms, region, strike, params, extent, param_cols=None
):
    """在指定宏观震中计算 GB18306 预测、观测、椭圆距和残差。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        台站观测表，格式见 ``load_obs_data``。
    epicenter : tuple(float, float)
        用于前向预测的宏观震中 ``(经度, 纬度)``。
    Ms : float
        GB18306 衰减关系使用的面波震级。
    region : str
        GB18306 分区名称。
    strike : float
        椭圆长轴走向，单位度，函数内部归一化到 [0, 360)。
    params : sequence
        -1=PGA、-2=PGV、``"Intensity"``=烈度。
    extent : float
        椭圆衰减场最大有效距离，单位 km。
    param_cols : dict or None
        可选显式观测列映射。

    Returns
    -------
    dict
        包含规范参数 ``params``、参数元数据 ``infos``、观测表 ``obs``、
        预测字典 ``preds``、长短轴等效距 ``aeqs``、残差 ``ress``、震中、
        走向、共享椭圆场对象以及实际观测列名。PGA/PGV 残差为
        ``ln(pred/obs)``，烈度残差为 ``pred-obs``。
    """
    params = normalize_params(params)
    if not params:
        raise ValueError("params 不能为空；支持 -1/0(PGA)、-2(PGV)、'Intensity'")
    infos = {p: param_info(p) for p in params}
    obs = load_obs_data(data, params, param_cols=param_cols)
    lon0, lat0 = float(epicenter[0]), float(epicenter[1])
    strike = strike % 360.0
    # Pre、观测对比和震中反演统一使用同一解析椭圆场。
    field = SharedGB18306EllipseField(region, Ms, extent=extent)

    raw = field.predict(lon0, lat0, strike, obs["lon"].values, obs["lat"].values)
    raw_by_label = {
        "烈度": (raw[0], raw[3], raw[4]),
        "PGA": (raw[1], raw[5], raw[6]),
        "PGV": (raw[2], raw[7], raw[8]),
    }
    preds, aeqs, ress = {}, {}, {}
    for p in params:
        info = infos[p]
        val, a, b = raw_by_label[info["label"]]
        preds[info["label"]] = val
        aeqs[info["label"]] = (a, b)
        ress[info["label"]] = residual(val, obs[info["label"]].values, info["kind"])
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
        "field": field,
        "source_columns": obs.attrs.get("source_columns", {}),
    }


# ==================== 主绘图函数（4×N） ====================


def plot_gb18306_vs_obs(
    data,
    Ms,
    region,
    strike,
    macro_epicenter,
    initial_epicenter=None,
    params=(-1, -2, "Intensity"),
    extent=400.0,
    max_dist=200.0,
    outpath="GB18306_vs_Obs.png",
    param_cols=None,
    grid_n=100,
    axis="长轴",
    fault_lon_mat=None,
    fault_lat_mat=None,
    plot_observations=None,
    table_outpath=None,
):
    """绘制 GB18306 的 4×N 预测—观测综合图。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        台站观测。若 DataFrame attrs 标记了场地修正/原始绘图模式，图标题
        会明确说明当前观测状态。
    Ms : float
        面波震级。
    region : str
        GB18306 分区名称。
    strike : float
        椭圆长轴走向，单位度。
    macro_epicenter : tuple(float, float)
        用于预测的宏观震中经纬度；可以是反演结果或用户指定值。
    initial_epicenter : tuple(float, float) or None
        初始破裂点，只用于地图标记，不影响预测。
    params : sequence
        绘图参数；GB18306 仅支持 PGA、PGV、烈度。
    extent : float, default 400
        椭圆衰减场范围，单位 km。
    max_dist : float, default 200
        残差图近场/远场分组阈值，单位 km。
    outpath : str or os.PathLike
        PNG 输出路径。
    param_cols : dict or None
        可选显式观测列映射。
    grid_n : int, default 100
        每个方向的绘图网格点数。
    axis : {"长轴", "短轴"}, default "长轴"
        衰减曲线横坐标使用的等效椭圆距离。
    fault_lon_mat, fault_lat_mat : array-like or None
        二维断层网格经纬度；第一/末行作为上下缘。必须同时提供或同时省略。
    plot_observations : {None, "corrected", "raw"}, default None
        场地修正观测的绘图模式。None 表示原样使用 ``data``；``corrected``
        使用 ``Vs30_site_correction.correct_observations_to_reference_vs30``
        生成的参考场地观测；``raw`` 从该表的 ``*_raw`` 列恢复原始场地观测。
        本参数只控制图上的观测点和绘图残差，不进行震中反演。
    table_outpath : str, os.PathLike or None, default None
        配套逐台站TXT路径。None 时自动使用与 ``outpath`` 相同的目录和文件名，
        仅把后缀替换为 ``.txt``。TXT采用UTF-8 BOM和制表符分隔。

    Returns
    -------
    str or os.PathLike
        原样返回 ``outpath``。图的四排依次为地图、衰减曲线、残差—距离和
        全部/近场/远场残差分布；第四排每组标注 N、均值 μ、中位数 m、
        总体标准差 σ 和均方根 RMS。每次出图同时写出配套逐台站TXT。

    Raises
    ------
    ValueError
        ``axis`` 非法，或断层经纬度矩阵没有成对提供。
    """
    if axis not in ("长轴", "短轴"):
        raise ValueError("axis 只能是 '长轴' 或 '短轴'")
    if (fault_lon_mat is None) != (fault_lat_mat is None):
        raise ValueError("fault_lon_mat 与 fault_lat_mat 必须同时提供或同时省略")
    if plot_observations is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "plot_observations 只能用于包含场地修正审计列的 pandas.DataFrame；"
                "文件路径输入请先调用 correct_observations_to_reference_vs30"
            )
        from Vs30_site_correction import prepare_site_plot_observations

        data = prepare_site_plot_observations(data, plot_observations)
    C = compute_vs_obs(
        data, macro_epicenter, Ms, region, strike, params, extent, param_cols
    )
    params, infos = C["params"], C["infos"]
    obs, preds, aeqs, ress = C["obs"], C["preds"], C["aeqs"], C["ress"]
    lon0, lat0, strike, field = C["lon0"], C["lat0"], C["strike"], C["field"]

    itype = obs["Instrument_Type"].str.upper().values
    use_markers = ("EI" in itype) and ("HN" in itype)

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

    # 所有参数共用同一个绘图网格，GB 解析场一次同时返回 I/PGA/PGV。
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
    glon = np.linspace(c_lon - half_km / clon_km, c_lon + half_km / clon_km, grid_n)
    glat = np.linspace(c_lat - half_km / 110.57, c_lat + half_km / 110.57, grid_n)
    GLON, GLAT = np.meshgrid(glon, glat)
    grid_raw = field.predict(lon0, lat0, strike, GLON.ravel(), GLAT.ravel())
    grid_values = {
        "烈度": grid_raw[0].reshape(GLON.shape),
        "PGA": grid_raw[1].reshape(GLON.shape),
        "PGV": grid_raw[2].reshape(GLON.shape),
    }

    for i, p in enumerate(params):
        info = infos[p]
        label, kind, T = info["label"], info["kind"], info["T"]
        title = f"{label} ({info['unit']})" if info["unit"] else label
        levels = param_levels(info)
        norm = BoundaryNorm(levels, ncolors=len(USGS_MMI_COLORS))
        tick_positions, tick_labels = level_ticks(info, levels)

        pred, a_eq, b_eq, res = preds[label], *aeqs[label], ress[label]
        valid = np.isfinite(pred) & np.isfinite(obs[label].values)

        # ================= 排1：预测云图 + 实测散点 + 断层投影 =================
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
        cb = fig.colorbar(cf, ax=ax, ticks=tick_positions, pad=0.03, shrink=0.85)
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
            ax.plot(flon[0], flat[0], color="r", lw=2.5, zorder=3, label="断层上缘")
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
        dist = a_eq if axis == "长轴" else b_eq
        far = float(np.nanmax(dist[valid])) if valid.any() else 0.0
        a_max = max(200.0, math.ceil(far / 100.0) * 100.0 + 100.0)
        r_scan = np.arange(1.0, a_max + 1.0, 1.0)
        if kind == "intensity":
            arr = [field.cal_i.calculate(Ms, float(r), region, axis) for r in r_scan]
            lm = np.array([d["mean"] for d in arr])
            ll = np.array([d["lower_1sigma"] for d in arr])
            lu = np.array([d["upper_1sigma"] for d in arr])
        else:
            gmm = [field.cal_g.calculate(Ms, float(r), region, axis) for r in r_scan]
            tup = [g[0 if T == -1 else 1] for g in gmm]
            lm = np.array([t[0] for t in tup])
            ll = np.array([t[1] for t in tup])
            lu = np.array([t[2] for t in tup])

        ax.axvspan(max_dist, a_max, color="lightgray", alpha=0.55, zorder=0)
        ax.axvline(max_dist, color="0.4", ls=":", lw=1.0, zorder=1)
        ax.plot(r_scan, lm, color="tab:red", lw=1.5, label=f"{axis}中值")
        ax.fill_between(r_scan, ll, lu, color="tab:red", alpha=0.15, label=f"{axis}±1σ")
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
        x_max3 = max(200.0, math.ceil(far3 / 50.0) * 50.0)
        ax.axvspan(max_dist, x_max3, color="lightgray", alpha=0.55, zorder=0)
        ax.axvline(max_dist, color="0.4", ls=":", lw=1.0, zorder=1)
        ax.axhline(0, color="k", lw=1.0, zorder=2)
        if kind == "gmm":
            sigma_ln = SIGMA_GB18306["pga" if T == -1 else "pgv"] * math.log(10.0)
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
            half_violin_box_scatter(ax, gdata, xpos, gcol, value_fmt="{:.3f}", s=18)
        if kind == "gmm":
            ax.axhline(sigma_ln, color="r", lw=1.3, ls="-.")
            ax.axhline(-sigma_ln, color="r", lw=1.3, ls="-.")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["全部", "<200 km", "≥200 km"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel("残差（预测-实测）", fontsize=8)
        ax.set_title(f"{title} 残差分布", fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 排4注释自适应：N/μ/m 文本框收进子图
    fig.canvas.draw()
    for i in range(n):
        fit_annotations_inside(axes[3, i], fig=fig, draw=False)

    title_parts = [
        "GB18306 预测 vs 实测",
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
            title_parts.append("原始场地观测（仅绘图，反演仍使用 Vs30 修正值）")
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
    if table_outpath is None:
        table_outpath = os.path.splitext(os.fspath(outpath))[0] + ".txt"
    export_gb18306_vs_obs_txt(
        data=data,
        Ms=Ms,
        region=region,
        strike=strike,
        macro_epicenter=macro_epicenter,
        params=params,
        extent=extent,
        max_dist=max_dist,
        outpath=table_outpath,
        param_cols=param_cols,
    )
    return outpath


# ==================== 统计表导出 ====================


def _fmt_num(v, nd=6):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "NaN"
    return "NaN" if not np.isfinite(v) else f"{v:.{nd}g}"


def export_gb18306_vs_obs_txt(
    data,
    Ms,
    region,
    strike,
    macro_epicenter,
    params=(-1, -2, "Intensity"),
    extent=400.0,
    max_dist=200.0,
    outpath="GB18306_vs_Obs_stats.txt",
    param_cols=None,
):
    """导出 GB18306 逐台站预测—观测统计表。

    Parameters
    ----------
    data, Ms, region, strike, macro_epicenter, params, extent, param_cols
        与 ``plot_gb18306_vs_obs`` 中同名参数含义一致。
    max_dist : float, default 200
        保留为与绘图接口一致的分组阈值；表中输出全部台站。
    outpath : str or os.PathLike
        制表符分隔文本输出路径，使用 UTF-8 BOM 编码。

    Returns
    -------
    str or os.PathLike
        原样返回 ``outpath``。第一行即表头：台站信息加每个参数的观测、
        预测、长轴距、短轴距和残差五列，可用
        ``pandas.read_csv(outpath, sep="\\t")`` 读取。

    Notes
    -----
    PGA/PGV 残差为 ``ln(预测/实测)``；烈度残差为线性 ``预测-实测``。
    """
    C = compute_vs_obs(
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
    plot_gb18306_vs_obs(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        macro_epicenter=(87.612, 28.823),
        initial_epicenter=(87.45, 28.5),
        Ms=6.8,
        region="青藏区",
        strike=349.0,
        params=(-1, -2, "Intensity"),
        outpath="Test_output/GB18306_vs_Obs.png",
    )
