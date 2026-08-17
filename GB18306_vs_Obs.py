# -*- coding: utf-8 -*-
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

from GB18306_class import GB18306_2015_IntensityCal, GB18306_2015_PGA_PGV_GMMs
from CEA2019_pre import (
    USGS_MMI_COLORS,
    PGA_LEVELS,
    PGV_LEVELS,
    fmt_pgv,
    km_to_lonlat,
)
from stat_violin import (
    apply_style,
    half_violin_box_scatter,
    fit_annotations_inside,
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
                raise ValueError(
                    f"GB18306 不支持参数 {p!r}，仅支持 PGA/PGV/Intensity"
                )
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
    """实测列候选：EPA/EPV 优先（GB18306 的 PGA/PGV 实为 EPA/EPV），
    没有再回退 PGA_H/PGV_H 或 PGA/PGV"""
    if info["kind"] == "intensity":
        return ["I"]
    if info["T"] == -1:
        return ["EPA_H", "PGA_H", "PGA"]
    return ["EPV_H", "PGV_H", "PGV"]


# ==================== 数据读取 ====================


def load_obs_data(data, params, param_cols=None):
    """读取实测数据（路径或 DataFrame）→ Sta_ID/lon/lat/Instrument_Type/各参数实测列"""
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
        cands = (
            [param_cols[label]]
            if param_cols is not None and label in param_cols
            else obs_col_candidates(info)
        )
        col = next((c for c in cands if c in df.columns), None)
        if col is None:
            raise ValueError(f"参数 {label} 找不到实测列，候选：{cands}")
        out[label] = pd.to_numeric(df[col], errors="coerce")
    return out.dropna(subset=["lon", "lat"]).reset_index(drop=True)


# ==================== GB18306 椭圆场前向模型（向量化） ====================


class GB18306EllipseField:
    """
    GB18306 椭圆场：台站相对长轴夹角 θ，在椭圆族 (a(V), b(V)) 上二分求预测值。
    predict_many(lon, lat, strike, sta_lon, sta_lat) → (I, PGA, PGV, aI, bI, aA, bA, aV, bV)
    """

    def __init__(self, region: str, Ms: float, extent: float = 400.0):
        self.region = region
        self.Ms = float(Ms)
        self.extent = float(extent)
        self.cal_i = GB18306_2015_IntensityCal()
        self.cal_g = GB18306_2015_PGA_PGV_GMMs()
        self.cal_i._validate_input(region, "长轴")
        self.cal_g._validate_input(region, "长轴")

        self.IL = self.cal_i._PARAMS[(region, "长轴")]
        self.IS = self.cal_i._PARAMS[(region, "短轴")]
        self.sigma_I = self.IL[4]

        self.gp = {}
        for pt in ("aE", "vE"):
            self.gp[pt] = {
                "long": self.cal_g._get_params(self.Ms, region, "长轴", pt),
                "short": self.cal_g._get_params(self.Ms, region, "短轴", pt),
            }
        self.I_lo = (
            self.IL[0]
            + self.IL[1] * self.Ms
            + self.IL[2] * math.log10(self.extent + self.IL[3])
        )
        self.I_hi = (
            self.IL[0]
            + self.IL[1] * self.Ms
            + self.IL[2] * math.log10(self.IL[3])
        )
        self.lgV = {}
        for pt in ("aE", "vE"):
            pl = self.gp[pt]["long"]
            exp_term = pl["D"] * math.exp(pl["E"] * self.Ms)
            self.lgV[pt] = {
                "lo": pl["A"]
                + pl["B"] * self.Ms
                + pl["C"] * math.log10(self.extent + exp_term),
                "hi": pl["A"]
                + pl["B"] * self.Ms
                + pl["C"] * math.log10(exp_term),
            }

    def _ab_intensity(self, I):
        A, B, C, R0 = self.IL[:4]
        a = 10.0 ** ((I - A - B * self.Ms) / C) - R0
        As, Bs, Cs, R0s = self.IS[:4]
        b = 10.0 ** ((I - As - Bs * self.Ms) / Cs) - R0s
        a = np.maximum(a, 1e-3)
        b = np.maximum(np.minimum(b, a), 1e-3)
        return a, b

    def _ab_value(self, V, pt):
        lg = np.log10(np.maximum(V, 1e-12))
        pl, ps = self.gp[pt]["long"], self.gp[pt]["short"]
        M = self.Ms
        a = 10.0 ** ((lg - pl["A"] - pl["B"] * M) / pl["C"]) - pl[
            "D"
        ] * math.exp(pl["E"] * M)
        b = 10.0 ** ((lg - ps["A"] - ps["B"] * M) / ps["C"]) - ps[
            "D"
        ] * math.exp(ps["E"] * M)
        a = np.maximum(a, 1e-3)
        b = np.maximum(np.minimum(b, a), 1e-3)
        return a, b

    @staticmethod
    def _bisect(R, ct, st, ab_func, lo, hi, iters=60):
        lo = np.full_like(R, lo)
        hi = np.full_like(R, hi)
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            a, b = ab_func(mid)
            f = (R * ct / a) ** 2 + (R * st / b) ** 2 - 1.0
            lo = np.where(f < 0.0, mid, lo)
            hi = np.where(f >= 0.0, mid, hi)
        return (lo + hi) / 2.0

    def _bisect_lgV(self, R, ct, st, pt, iters=60):
        lo = np.full_like(R, self.lgV[pt]["lo"])
        hi = np.full_like(R, self.lgV[pt]["hi"])
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            a, b = self._ab_value(10.0**mid, pt)
            f = (R * ct / a) ** 2 + (R * st / b) ** 2 - 1.0
            lo = np.where(f < 0.0, mid, lo)
            hi = np.where(f >= 0.0, mid, hi)
        return 10.0 ** ((lo + hi) / 2.0)

    def predict_many(self, lon, lat, strike, sta_lon, sta_lat):
        from pyproj import Transformer

        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        sta_lon = np.asarray(sta_lon, dtype=float).ravel()
        sta_lat = np.asarray(sta_lat, dtype=float).ravel()
        n_cand, n_sta = lon.size, sta_lon.size
        I_out = np.full((n_cand, n_sta), np.nan)
        A_out = np.full((n_cand, n_sta), np.nan)
        V_out = np.full((n_cand, n_sta), np.nan)
        aI_out = np.full((n_cand, n_sta), np.nan)
        bI_out = np.full((n_cand, n_sta), np.nan)
        aA_out = np.full((n_cand, n_sta), np.nan)
        bA_out = np.full((n_cand, n_sta), np.nan)
        aV_out = np.full((n_cand, n_sta), np.nan)
        bV_out = np.full((n_cand, n_sta), np.nan)

        groups = {}
        for i in range(n_cand):
            zone = int((lon[i] + 180.0) // 6.0) + 1
            hemi = 326 if lat[i] >= 0 else 327
            groups.setdefault((hemi, zone), []).append(i)
        for (hemi, zone), idx in groups.items():
            fwd = Transformer.from_crs(
                "epsg:4326", f"epsg:{hemi}{zone:02d}", always_xy=True
            )
            ex, ey = np.asarray(fwd.transform(lon[idx], lat[idx]), dtype=float)
            sx, sy = np.asarray(fwd.transform(sta_lon, sta_lat), dtype=float)
            dx = (sx[None, :] - ex[:, None]) / 1000.0
            dy = (sy[None, :] - ey[:, None]) / 1000.0
            R = np.hypot(dx, dy)
            theta = np.arctan2(dy, dx) - math.radians(90.0 - strike)
            ct, st = np.cos(theta), np.sin(theta)

            a0, b0 = self._ab_intensity(self.I_lo)
            r_outer = (
                float(a0)
                * float(b0)
                / np.sqrt((float(b0) * ct) ** 2 + (float(a0) * st) ** 2)
            )
            inside = R <= r_outer
            Ii = self._bisect(
                R, ct, st, self._ab_intensity, self.I_lo, self.I_hi
            )
            aI, bI = self._ab_intensity(Ii)
            I_out[idx, :] = np.where(inside, Ii, np.nan)
            aI_out[idx, :] = np.where(inside, aI, np.nan)
            bI_out[idx, :] = np.where(inside, bI, np.nan)

            for pt, out in (("aE", A_out), ("vE", V_out)):
                a0v, b0v = self._ab_value(10.0 ** self.lgV[pt]["lo"], pt)
                r_outer_v = (
                    float(a0v)
                    * float(b0v)
                    / np.sqrt((float(b0v) * ct) ** 2 + (float(a0v) * st) ** 2)
                )
                inside_v = R <= r_outer_v
                Vi = self._bisect_lgV(R, ct, st, pt)
                aV, bV = self._ab_value(Vi, pt)
                out[idx, :] = np.where(inside_v, Vi, np.nan)
                if pt == "aE":
                    aA_out[idx, :] = np.where(inside_v, aV, np.nan)
                    bA_out[idx, :] = np.where(inside_v, bV, np.nan)
                else:
                    aV_out[idx, :] = np.where(inside_v, aV, np.nan)
                    bV_out[idx, :] = np.where(inside_v, bV, np.nan)
        return (
            I_out,
            A_out,
            V_out,
            aI_out,
            bI_out,
            aA_out,
            bA_out,
            aV_out,
            bV_out,
        )

    def predict(self, lon, lat, strike, sta_lon, sta_lat):
        out = self.predict_many(
            np.atleast_1d(lon), np.atleast_1d(lat), strike, sta_lon, sta_lat
        )
        return tuple(o[0] for o in out)


# ==================== 预测 / 椭圆距 / 残差 / 色标 ====================


def predict_one(param, field, lon, lat, strike, sta_lon, sta_lat):
    """GB18306 单参数预测，返回 (预测值, 长轴距, 短轴距)"""
    I, A, V, aI, bI, aA, bA, aV, bV = field.predict(
        lon, lat, strike, sta_lon, sta_lat
    )
    if param == "Intensity":
        return I, aI, bI
    if float(param) == -1:
        return A, aA, bA
    return V, aV, bV


def residual(pred, obs, kind):
    """残差 = 预测 - 实测：PGA/PGV 用 ln(Pred/Obs)，烈度用线性差"""
    if kind == "gmm":
        return np.log(
            np.asarray(pred, dtype=float) / np.asarray(obs, dtype=float)
        )
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
    """返回 观测/预测/椭圆距/残差 等全部结果（绘图与 txt 共用）"""
    params = normalize_params(params)
    if not params:
        raise ValueError(
            "params 不能为空；支持 -1/0(PGA)、-2(PGV)、'Intensity'"
        )
    infos = {p: param_info(p) for p in params}
    obs = load_obs_data(data, params, param_cols=param_cols)
    lon0, lat0 = float(epicenter[0]), float(epicenter[1])
    strike = strike % 360.0
    field = GB18306EllipseField(region, Ms, extent=extent)

    preds, aeqs, ress = {}, {}, {}
    for p in params:
        info = infos[p]
        val, a, b = predict_one(
            p,
            field,
            lon0,
            lat0,
            strike,
            obs["lon"].values,
            obs["lat"].values,
        )
        preds[info["label"]] = val
        aeqs[info["label"]] = (a, b)
        ress[info["label"]] = residual(
            val, obs[info["label"]].values, info["kind"]
        )
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
):
    """
    基于宏观震中，用 GB18306 预测并绘制"预测 vs 实测"4×N 综合图。

    macro_epicenter：宏观震中（反演/迭代得到的经纬度），必填；
    initial_epicenter：初始破裂点（发布值），仅作标记；
    fault_lon_mat / fault_lat_mat：断层面网格矩阵（第一行=上缘，
        最后一行=下缘），提供时在第一排绘制断层投影。
    """
    if axis not in ("长轴", "短轴"):
        raise ValueError("axis 只能是 '长轴' 或 '短轴'")
    C = compute_vs_obs(
        data, macro_epicenter, Ms, region, strike, params, extent, param_cols
    )
    params, infos, labels = C["params"], C["infos"], C["labels"]
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
        pred_grid = predict_one(
            p, field, lon0, lat0, strike, GLON.ravel(), GLAT.ravel()
        )[0].reshape(GLON.shape)
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
            arrowprops=dict(arrowstyle="->", color="k", lw=1.1),
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
            arr = [
                field.cal_i.calculate(Ms, float(r), region, axis)
                for r in r_scan
            ]
            lm = np.array([d["mean"] for d in arr])
            ll = np.array([d["lower_1sigma"] for d in arr])
            lu = np.array([d["upper_1sigma"] for d in arr])
        else:
            gmm = [
                field.cal_g.calculate(Ms, float(r), region, axis)
                for r in r_scan
            ]
            tup = [g[0 if T == -1 else 1] for g in gmm]
            lm = np.array([t[0] for t in tup])
            ll = np.array([t[1] for t in tup])
            lu = np.array([t[2] for t in tup])

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
        x_max3 = max(200.0, math.ceil(far3 / 50.0) * 50.0)
        ax.axvspan(max_dist, x_max3, color="lightgray", alpha=0.55, zorder=0)
        ax.axvline(max_dist, color="0.4", ls=":", lw=1.0, zorder=1)
        ax.axhline(0, color="k", lw=1.0, zorder=2)
        if kind == "gmm":
            sigma_ln = SIGMA_GB18306["pga" if T == -1 else "pgv"] * math.log(
                10.0
            )
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

    # 排4注释自适应：N/μ/m 文本框收进子图
    try:
        fig.canvas.draw()
        for i in range(n):
            fit_annotations_inside(axes[3, i], fig=fig, draw=False)
    except Exception:
        pass

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
    fig.suptitle("  |  ".join(title_parts), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图件：{os.path.abspath(outpath)}")
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
    """
    输出 1 个大的 txt（与绘图共用同一套计算），第一行即表头：
        台站信息 + 每个参数：<参数>_obs / <参数>_pred /
        Repi_long_<参数>(km) / Repi_short_<参数>(km) / <参数>_res
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
