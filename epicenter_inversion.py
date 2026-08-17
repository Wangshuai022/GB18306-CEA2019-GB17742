# -*- coding: utf-8 -*-
"""
震中反演程序（GB 18306-2015 椭圆衰减模型）
=============================================

已知震中周边若干台站的 PGA / PGV / 烈度，反演"最优震中"（经纬度）。

四种线路（目标函数均为最小二乘，残差按各自对数标准差 σ 归一化）：
    1. pga_pgv   : PGA + PGV 联合反演（对数域残差，按 σ_pga=0.236、σ_pgv=0.271）
    2. intensity : 烈度反演（烈度域残差，按分区 σ_I）
    3. pga       : 仅用 PGA
    4. pgv       : 仅用 PGV

有效范围过滤（0–200 km）
---------------------------
GB 18306 衰减关系只适用于 0–200 km，判定量是台站所在椭圆圈的
"长轴距 / 短轴距"（a_eq / b_eq，即该台站所在等值椭圆的长、短轴半径）。
由于 a_eq / b_eq 依赖震中和走向，反演前先以"参考点 + 中心走向"做一次预筛：
    - 参考点 = --seed-lon/--seed-lat（初始震中），未给时取台站几何中心；
    - 中心走向 = --strike（默认定日地震 349°）；
    - a_eq 或 b_eq 超过 --max-dist（默认 200 km）的台站直接剔除、不参与拟合，
      剔除名单（含 a_eq / b_eq）会在屏幕上打印，并在图中以灰色 × 标出。
结果表中每个台站都带 a_eq / b_eq 与 used 标记。

前向模型
---------
完全复刻 GB18306_Pre.py 的"椭圆场"算法：
    台站相对震中有震中距 R 和相对长轴的夹角 θ（长轴走向 strike，正北=0°顺时针）；
    对某个地震动值 V，长轴距 a(V)、短轴距 b(V) 由 GB 18306 衰减公式反解得到，
    台站落在椭圆族 (a(V), b(V)) 上的条件为：
        (R·cosθ / a)² + (R·sinθ / b)² = 1
    对 V 二分求解即得该台站预测值。本程序用闭合反解式实现，向量化批处理，
    与旧曲线插值版逐点一致（已数值验证）。

反演策略
---------
    粗网格搜索（全台站外包络 + 走向粗扫描）→ 细网格（最优附近）→ Nelder-Mead 局部精化。

依赖：numpy / pandas / scipy / pyproj / matplotlib；GB18306_class.py 须在同一目录。

函数调用（推荐，作为模块 import 使用）
---------------------------------------
    from epicenter_inversion import invert_epicenter, plot_parameter_4panel

    # 输入：实测数据文件、震级、分区、初始震中、线路、走向、容许偏差、反演范围散点
    res = invert_epicenter(
        data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
        Ms=6.8, region="青藏区",
        epicenter=(87.378, 28.604),          # 初始震中
        mode="pga_pgv",                       # pga / pgv / intensity / pga_pgv
        strike=349.0, strike_dev=30.0,        # 走向 + 容许偏差（±30° 窗口）
        fault_longi=flon, fault_lati=flat,    # 反演震中范围散点（向量/矩阵均可）
        max_dist=200.0,                       # GB18306 有效范围（长轴距/短轴距 ≤ 200 km）
    )
    print(res["epicenter"], res["strike"], res["chi2"])

    # 出图：依托最优震中画 PGA 4 联图（云图 / 曲线±1σ / 残差-距离 / 残差小提琴箱线）
    plot_parameter_4panel(res, "PGA", "figure_PGA.png")

命令行用法
-----------
    python epicenter_inversion.py --input xxx.txt --ms 6.8 --region 青藏区 \
        --seed-lon 87.378 --seed-lat 28.604 \
        --strike 349 --strike-range 30 --strike-prior-sigma 10 \
        --fault-file fault_points.txt --modes pga_pgv,intensity,pga,pgv
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from GB18306_class import GB18306_2015_IntensityCal, GB18306_2015_PGA_PGV_GMMs

# 对数标准差（GB 18306-2015 附录）：PGA=0.236，PGV=0.271；烈度 σ 分区不同，取自参数表
SIGMA_GMM = {"pga": 0.236, "pgv": 0.271}

MODES = ("pga_pgv", "intensity", "pga", "pgv")
MODE_NAMES = {
    "pga_pgv": "PGA+PGV 联合",
    "intensity": "烈度",
    "pga": "仅 PGA",
    "pgv": "仅 PGV",
}


def haversine_km(lon1, lat1, lon2, lat2):
    """两经纬度点间大圆距离（km）"""
    r_earth = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r_earth * math.asin(math.sqrt(a))


def read_station_file(path):
    """
    读取台站记录文件（tab 分隔），返回 (df, true_epi, Ms)。

    df 列：Sta_ID / lon / lat / I / pga / pgv
        - 有 _H 列（PGA_H / PGV_H）时优先使用水平向（用户约定都用水平向）；
        - 否则回退到 PGA / PGV。
    true_epi：文件里若带 Hypo_longi / Hypo_lati 则作为"真实震中"参考（用于误差评估）；
    Ms      ：文件里若带 Mag 列则取其中位数（默认震级）。
    """
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    pga_col = "PGA_H" if "PGA_H" in df.columns else ("PGA" if "PGA" in df.columns else None)
    pgv_col = "PGV_H" if "PGV_H" in df.columns else ("PGV" if "PGV" in df.columns else None)
    if pga_col is None or pgv_col is None:
        raise ValueError("文件中找不到 PGA/PGV（或 PGA_H/PGV_H）列！")

    lon_col = "longi" if "longi" in df.columns else "lon"
    lat_col = "lati" if "lati" in df.columns else "lat"
    need = ["Sta_ID", lon_col, lat_col, "I", pga_col, pgv_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列：{missing}")

    out = pd.DataFrame(
        {
            "Sta_ID": df["Sta_ID"].astype(str),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "I": pd.to_numeric(df["I"], errors="coerce"),
            "pga": pd.to_numeric(df[pga_col], errors="coerce"),
            "pgv": pd.to_numeric(df[pgv_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["lon", "lat", "I", "pga", "pgv"]).reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("有效台站记录为 0，请检查数据！")

    true_epi = None
    if "Hypo_longi" in df.columns and "Hypo_lati" in df.columns:
        hl = pd.to_numeric(df["Hypo_longi"], errors="coerce").dropna()
        hb = pd.to_numeric(df["Hypo_lati"], errors="coerce").dropna()
        if len(hl) and len(hb):
            true_epi = (float(hl.iloc[0]), float(hb.iloc[0]))

    ms = None
    if "Mag" in df.columns:
        mag = pd.to_numeric(df["Mag"], errors="coerce").dropna()
        if len(mag):
            ms = float(mag.median())

    return out, true_epi, ms


def load_station_data(data):
    """
    数据输入可以是文件路径或 pandas.DataFrame，统一返回 (df, true_epi, Ms)。

    df 列：Sta_ID / lon / lat / I / pga / pgv（PGA_H/PGV_H 优先，其次 PGA/PGV）。
    """
    if isinstance(data, (str, os.PathLike)):
        return read_station_file(str(data))
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.columns = [str(c).strip() for c in df.columns]
        rename = {"longi": "lon", "lati": "lat",
                  "PGA_H": "pga", "PGA": "pga",
                  "PGV_H": "pgv", "PGV": "pgv"}
        df = df.rename(columns=rename)
        need = ["lon", "lat", "I", "pga", "pgv"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame 缺少列：{missing}，现有列：{list(df.columns)}")
        if "Sta_ID" not in df.columns:
            df["Sta_ID"] = [f"S{i + 1}" for i in range(len(df))]
        out = df[["Sta_ID", "lon", "lat", "I", "pga", "pgv"]].copy()
        for c in ("lon", "lat", "I", "pga", "pgv"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["lon", "lat", "I", "pga", "pgv"]).reset_index(drop=True)
        if len(out) == 0:
            raise ValueError("有效台站记录为 0，请检查数据！")

        true_epi = None
        if "Hypo_longi" in df.columns and "Hypo_lati" in df.columns:
            hl = pd.to_numeric(df["Hypo_longi"], errors="coerce").dropna()
            hb = pd.to_numeric(df["Hypo_lati"], errors="coerce").dropna()
            if len(hl) and len(hb):
                true_epi = (float(hl.iloc[0]), float(hb.iloc[0]))
        ms = None
        if "Mag" in df.columns:
            mag = pd.to_numeric(df["Mag"], errors="coerce").dropna()
            if len(mag):
                ms = float(mag.median())
        return out, true_epi, ms
    raise TypeError("data 必须是文件路径或 pandas.DataFrame")


def parse_fault_points(fault_longi=None, fault_lati=None):
    """
    反演震中范围散点（Fault_longi / Fault_lati），支持：
      - 两个同形状向量 / 矩阵（自动展平，如 meshgrid 结果）；
      - 只传一个 (N,2) 或 (2,N) 经纬度矩阵；
      - 标量单点。
    返回 (lon_pts, lat_pts) 1D 数组；两者都不传时返回 None（改用台站外包络网格）。
    """
    if fault_longi is None and fault_lati is None:
        return None
    if fault_longi is None or fault_lati is None:
        arr = np.asarray(fault_longi if fault_lati is None else fault_lati, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 0], arr[:, 1]
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
            return arr[0], arr[1]
        raise ValueError("只传一个数组时必须是 (N,2) 或 (2,N) 的经纬度矩阵")
    lon = np.asarray(fault_longi, dtype=float).ravel()
    lat = np.asarray(fault_lati, dtype=float).ravel()
    if lon.size != lat.size:
        raise ValueError("Fault_longi 与 Fault_lati 长度不一致")
    return lon, lat


class GB18306EllipseField:
    """
    GB 18306 椭圆场前向模型（向量化，批处理多个候选震中）。

    predict / predict_many 返回 (I, PGA, PGV)，单位：烈度 / gal / cm/s；
    超出场范围（extent 之外）的台站预测值为 NaN。
    """

    def __init__(self, region: str, Ms: float, extent: float = 500.0):
        self.region = region
        self.Ms = float(Ms)
        self.extent = float(extent)
        self.cal_i = GB18306_2015_IntensityCal()
        self.cal_g = GB18306_2015_PGA_PGV_GMMs()
        self.cal_i._validate_input(region, "长轴")
        self.cal_g._validate_input(region, "长轴")

        # 烈度参数：(A, B, C, R0, sigma)
        self.IL = self.cal_i._PARAMS[(region, "长轴")]
        self.IS = self.cal_i._PARAMS[(region, "短轴")]
        self.sigma_I = self.IL[4]

        # PGA / PGV 参数（含 Ms 分段，A2/B2 或 A1/B1）
        self.gp = {}
        for pt in ("aE", "vE"):
            self.gp[pt] = {
                "long": self.cal_g._get_params(self.Ms, region, "长轴", pt),
                "short": self.cal_g._get_params(self.Ms, region, "短轴", pt),
            }

        # 二分上/下界（对应 R→0 的震中值和 R=extent 的最外圈）
        self.I_lo = self.IL[0] + self.IL[1] * self.Ms + self.IL[2] * math.log10(
            self.extent + self.IL[3]
        )
        self.I_hi = self.IL[0] + self.IL[1] * self.Ms + self.IL[2] * math.log10(self.IL[3])
        self.lgV = {}
        for pt in ("aE", "vE"):
            pl = self.gp[pt]["long"]
            exp_term = pl["D"] * math.exp(pl["E"] * self.Ms)
            self.lgV[pt] = {
                "lo": pl["A"] + pl["B"] * self.Ms + pl["C"] * math.log10(self.extent + exp_term),
                "hi": pl["A"] + pl["B"] * self.Ms + pl["C"] * math.log10(exp_term),
            }

    # ---------- 椭圆几何：长轴距 a、短轴距 b ----------
    def _ab_intensity(self, I):
        A, B, C, R0 = self.IL[:4]
        a = 10.0 ** ((I - A - B * self.Ms) / C) - R0
        As, Bs, Cs, R0s = self.IS[:4]
        b = 10.0 ** ((I - As - Bs * self.Ms) / Cs) - R0s
        a = np.maximum(a, 1e-3)
        b = np.maximum(np.minimum(b, a), 1e-3)
        return a, b

    def _ab_value(self, V, pt):
        """由地震动值 V 反解长轴距 a 与短轴距 b（b 不超过 a，退化处画圆）"""
        lg = np.log10(np.maximum(V, 1e-12))
        pl, ps = self.gp[pt]["long"], self.gp[pt]["short"]
        M = self.Ms
        a = 10.0 ** ((lg - pl["A"] - pl["B"] * M) / pl["C"]) - pl["D"] * math.exp(pl["E"] * M)
        b = 10.0 ** ((lg - ps["A"] - ps["B"] * M) / ps["C"]) - ps["D"] * math.exp(ps["E"] * M)
        a = np.maximum(a, 1e-3)
        b = np.maximum(np.minimum(b, a), 1e-3)
        return a, b

    # ---------- 二分求解 ----------
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
            a, b = self._ab_value(10.0 ** mid, pt)
            f = (R * ct / a) ** 2 + (R * st / b) ** 2 - 1.0
            lo = np.where(f < 0.0, mid, lo)
            hi = np.where(f >= 0.0, mid, hi)
        return 10.0 ** ((lo + hi) / 2.0)

    def predict_many(self, lon, lat, strike, sta_lon, sta_lat):
        """
        批量预测：lon/lat 形状 (n_cand,)，sta_lon/sta_lat 形状 (n_sta,)。
        返回 (I, PGA, PGV)，形状 (n_cand, n_sta)。
        """
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

        from pyproj import Transformer

        # 按 UTM 带分组：同一带内一次批量投影
        groups = {}
        for i in range(n_cand):
            zone = int((lon[i] + 180.0) // 6.0) + 1
            hemi = 326 if lat[i] >= 0 else 327
            groups.setdefault((hemi, zone), []).append(i)

        for (hemi, zone), idx in groups.items():
            fwd = Transformer.from_crs("epsg:4326", f"epsg:{hemi}{zone:02d}", always_xy=True)
            ex, ey = np.asarray(fwd.transform(lon[idx], lat[idx]), dtype=float)
            sx, sy = np.asarray(fwd.transform(sta_lon, sta_lat), dtype=float)
            dx = (sx[None, :] - ex[:, None]) / 1000.0
            dy = (sy[None, :] - ey[:, None]) / 1000.0
            R = np.hypot(dx, dy)
            theta = np.arctan2(dy, dx) - math.radians(90.0 - strike)
            ct = np.cos(theta)
            st = np.sin(theta)

            # 烈度：最外圈椭圆（长轴=extent）
            a0, b0 = self._ab_intensity(self.I_lo)
            r_outer = float(a0) * float(b0) / np.sqrt(
                (float(b0) * ct) ** 2 + (float(a0) * st) ** 2
            )
            inside = R <= r_outer
            Ii = self._bisect(R, ct, st, self._ab_intensity, self.I_lo, self.I_hi)
            aI, bI = self._ab_intensity(Ii)
            I_out[idx, :] = np.where(inside, Ii, np.nan)
            aI_out[idx, :] = np.where(inside, aI, np.nan)
            bI_out[idx, :] = np.where(inside, bI, np.nan)

            # PGA / PGV
            for pt, out in (("aE", A_out), ("vE", V_out)):
                a0v, b0v = self._ab_value(10.0 ** self.lgV[pt]["lo"], pt)
                r_outer_v = float(a0v) * float(b0v) / np.sqrt(
                    (float(b0v) * ct) ** 2 + (float(a0v) * st) ** 2
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

        return I_out, A_out, V_out, aI_out, bI_out, aA_out, bA_out, aV_out, bV_out

    def predict(self, lon, lat, strike, sta_lon, sta_lat):
        """单个候选震中的预测（返回 9 个 1D 数组，长度 n_sta）"""
        I, A, V, aI, bI, aA, bA, aV, bV = self.predict_many(
            np.atleast_1d(lon), np.atleast_1d(lat), strike, sta_lon, sta_lat
        )
        return I[0], A[0], V[0], aI[0], bI[0], aA[0], bA[0], aV[0], bV[0]


class EpicenterInverter:
    """震中反演器：网格搜索 + 局部精化 + 结果报表"""

    def __init__(self, field: GB18306EllipseField, sta_df: pd.DataFrame,
                 true_epi=None, pad_deg: float = 1.0, penalty: float = 1000.0,
                 max_dist: float = 200.0, filter_ref=None):
        self.field = field
        self.sta_lon = sta_df["lon"].values.astype(float)
        self.sta_lat = sta_df["lat"].values.astype(float)
        self.sta_id = sta_df["Sta_ID"].values
        self.obs = {
            "I": sta_df["I"].values.astype(float),
            "pga": sta_df["pga"].values.astype(float),
            "pgv": sta_df["pgv"].values.astype(float),
        }
        self.true_epi = true_epi
        self.penalty = penalty
        self.max_dist = float(max_dist)
        self.fixed_used = None
        self.excluded_df = None

        # 0–200 km 预筛：以参考点（初始震中 + 中心走向）计算各台站 a_eq / b_eq，
        # 按线路保留长轴距/短轴距 ≤ max_dist 的台站（b_eq ≤ a_eq，实际以 a_eq 为约束）
        if filter_ref is not None:
            flon, flat, fstrike = filter_ref
            _, _, _, aI, _, aA, _, aV, _ = self.field.predict(
                flon, flat, fstrike, self.sta_lon, self.sta_lat
            )
            okI = np.isfinite(aI) & (aI <= self.max_dist)
            okA = np.isfinite(aA) & (aA <= self.max_dist)
            okV = np.isfinite(aV) & (aV <= self.max_dist)
            self.fixed_used = {
                "intensity": okI,
                "pga": okA,
                "pgv": okV,
                "pga_pgv": okA & okV,
            }
            keep_all = okI & okA & okV
            ex = ~keep_all
            if ex.any():
                self.excluded_df = pd.DataFrame(
                    {
                        "Sta_ID": self.sta_id[ex],
                        "lon": np.round(self.sta_lon[ex], 4),
                        "lat": np.round(self.sta_lat[ex], 4),
                        "a_eq_I": np.round(aI[ex], 2),
                        "a_eq_PGA": np.round(aA[ex], 2),
                        "a_eq_PGV": np.round(aV[ex], 2),
                    }
                )
        else:
            n = self.sta_lon.size
            self.fixed_used = {m: np.ones(n, dtype=bool) for m in MODES}

        # 搜索范围：台站外包络外扩 pad_deg
        self.lon0 = float(self.sta_lon.min() - pad_deg)
        self.lon1 = float(self.sta_lon.max() + pad_deg)
        self.lat0 = float(self.sta_lat.min() - pad_deg)
        self.lat1 = float(self.sta_lat.max() + pad_deg)

    # ---------- 目标函数 ----------
    def chi2_batch(self, mode, lon, lat, strike):
        """向量化：候选震中 (lon, lat) 的归一化卡方（越小越优）"""
        I, A, V, aI, bI, aA, bA, aV, bV = self.field.predict_many(
            lon, lat, strike, self.sta_lon, self.sta_lat
        )
        obs = self.obs
        lg = lambda x: np.log10(np.maximum(x, 1e-9))

        used = self.fixed_used[mode][None, :]
        if mode == "intensity":
            in_field = np.isfinite(I)
        elif mode == "pga":
            in_field = np.isfinite(A)
        elif mode == "pgv":
            in_field = np.isfinite(V)
        else:
            in_field = np.isfinite(A) & np.isfinite(V)
        ok = used & in_field
        if mode == "intensity":
            r = (obs["I"][None, :] - I) / self.field.sigma_I
            chi2 = np.where(ok, r * r, 0.0).sum(axis=1)
        elif mode == "pga":
            r = (lg(obs["pga"])[None, :] - np.log10(A)) / SIGMA_GMM["pga"]
            chi2 = np.where(ok, r * r, 0.0).sum(axis=1)
        elif mode == "pgv":
            r = (lg(obs["pgv"])[None, :] - np.log10(V)) / SIGMA_GMM["pgv"]
            chi2 = np.where(ok, r * r, 0.0).sum(axis=1)
        else:
            rA = (lg(obs["pga"])[None, :] - np.log10(A)) / SIGMA_GMM["pga"]
            rV = (lg(obs["pgv"])[None, :] - np.log10(V)) / SIGMA_GMM["pgv"]
            chi2 = np.where(ok, rA * rA, 0.0).sum(axis=1) + np.where(
                ok, rV * rV, 0.0
            ).sum(axis=1)

        # 场外重罚（预筛保留的台站若落在衰减场外）
        chi2 = chi2 + self.penalty * (used & ~in_field).sum(axis=1)
        return chi2

    def _obj_single(self, mode, strike_fixed):
        """单个候选的目标函数（供 scipy 优化器调用）；越界加二次罚"""

        def obj(x):
            lon, lat = float(x[0]), float(x[1])
            pen = 0.0
            if strike_fixed is not None:
                strike = float(strike_fixed)
            else:
                strike = float(x[2]) % 360.0
                dev = self._circ_dev(strike, self.strike_center)
                if self.strike_prior_sigma > 0:
                    dev_line = self._circ_dev_line(strike, self.strike_center)
                    pen += (dev_line / self.strike_prior_sigma) ** 2
                if abs(dev) > self.strike_range:
                    pen += 1e3 * (abs(dev) - self.strike_range) ** 2
            chi2 = float(self.chi2_batch(mode, np.array([lon]), np.array([lat]), strike)[0])
            for v, vmin, vmax in ((lon, self.lon0, self.lon1), (lat, self.lat0, self.lat1)):
                if v < vmin:
                    pen += 1e3 * (vmin - v) ** 2
                elif v > vmax:
                    pen += 1e3 * (v - vmax) ** 2
            return chi2 + pen

        return obj

    @staticmethod
    def _circ_dev(strike, center):
        """走向相对中心值的环形偏差（±180° 内）"""
        return ((strike - center + 180.0) % 360.0) - 180.0

    @staticmethod
    def _circ_dev_line(strike, center):
        """走向相对中心值的"直线偏差"（长轴无方向性，349° 与 169° 视为同一椭圆）"""
        dev = abs(((strike - center + 180.0) % 360.0) - 180.0)
        return min(dev, 180.0 - dev)

    # ---------- 反演流程 ----------
    def invert(self, mode, strike_center=349.0, strike_range=30.0, strike_step=10.0,
               strike_prior_sigma=10.0, coarse_step=0.1, fine_step=0.02, fine_half=0.5,
               seed=None, cand_lon=None, cand_lat=None, verbose=True):
        from scipy.optimize import minimize

        self.strike_center = float(strike_center)
        self.strike_range = float(strike_range)
        self.strike_prior_sigma = float(strike_prior_sigma)

        if cand_lon is not None and cand_lat is not None:
            # 阶段 1：直接在给定散点（Fault_longi / Fault_lati）上打分
            lon_f = np.asarray(cand_lon, dtype=float).ravel()
            lat_f = np.asarray(cand_lat, dtype=float).ravel()
            if lon_f.size != lat_f.size:
                raise ValueError("cand_lon 与 cand_lat 长度不一致")
            if lon_f.size == 0:
                raise ValueError("反演范围散点为空！")
        else:
            # 阶段 1：台站外包络规则网格
            lon_g = np.arange(self.lon0, self.lon1 + coarse_step / 2.0, coarse_step)
            lat_g = np.arange(self.lat0, self.lat1 + coarse_step / 2.0, coarse_step)
            LON, LAT = np.meshgrid(lon_g, lat_g)
            lon_f = LON.ravel()
            lat_f = LAT.ravel()

        if self.strike_range <= 0.0:
            fixed = True
            strikes = [self.strike_center % 360.0]
        else:
            fixed = False
            offsets = np.arange(
                -self.strike_range, self.strike_range + strike_step / 2.0, strike_step
            )
            strikes = [(self.strike_center + o) % 360.0 for o in offsets]

        # 阶段 1：粗网格（全范围 × 走向扫描）
        best = (np.inf, None, None, None)
        for s in strikes:
            chi2 = self.chi2_batch(mode, lon_f, lat_f, s)
            if not fixed and self.strike_prior_sigma > 0:
                chi2 = chi2 + (self._circ_dev_line(s, self.strike_center) / self.strike_prior_sigma) ** 2
            j = int(np.argmin(chi2))
            if chi2[j] < best[0]:
                best = (float(chi2[j]), float(lon_f[j]), float(lat_f[j]), float(s))

        # 阶段 2：细网格（最优附近，走向取最优及左右相邻档）
        if fixed:
            fine_strikes = [strikes[0]]
        else:
            cand = []
            for k in (-2, -1, 0, 1, 2):
                s = (best[3] + k * strike_step) % 360.0
                if abs(self._circ_dev(s, self.strike_center)) <= self.strike_range + 1e-9:
                    cand.append(round(s, 6))
            fine_strikes = sorted(set(cand))
        flon = np.arange(best[1] - fine_half, best[1] + fine_half + fine_step / 2.0, fine_step)
        flat = np.arange(best[2] - fine_half, best[2] + fine_half + fine_step / 2.0, fine_step)
        FLON, FLAT = np.meshgrid(flon, flat)
        flon_f = FLON.ravel()
        flat_f = FLAT.ravel()
        best2 = best
        for s in fine_strikes:
            chi2 = self.chi2_batch(mode, flon_f, flat_f, s)
            if not fixed and self.strike_prior_sigma > 0:
                chi2 = chi2 + (self._circ_dev_line(s, self.strike_center) / self.strike_prior_sigma) ** 2
            j = int(np.argmin(chi2))
            if chi2[j] < best2[0]:
                best2 = (float(chi2[j]), float(flon_f[j]), float(flat_f[j]), float(s))

        # 阶段 3：Nelder-Mead 局部精化
        obj = self._obj_single(mode, strikes[0] if fixed else None)
        starts = [[best2[1], best2[2]]] if fixed else [[best2[1], best2[2], best2[3]]]
        # 初始震中种子：作为另一个精化起点（不限制搜索范围）
        if seed is not None:
            s0 = [float(seed[0]), float(seed[1])]
            if fixed:
                starts.append(s0)
            else:
                starts.append(s0 + [best2[3]])
        best_nm = None
        for x0 in starts:
            res = minimize(
                obj, x0, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-11},
            )
            if best_nm is None or res.fun < best_nm.fun:
                best_nm = res
        x = best_nm.x
        lon_opt, lat_opt = float(x[0]), float(x[1])
        strike_opt = strikes[0] if fixed else float(x[2]) % 360.0
        chi2_opt = float(best_nm.fun)

        # 最终预测 + 每台站残差表
        I, A, V, aI, bI, aA, bA, aV, bV = self.field.predict(
            lon_opt, lat_opt, strike_opt, self.sta_lon, self.sta_lat
        )
        tab = self.station_table(mode, I, A, V, aI, bI, aA, bA, aV, bV, lon_opt, lat_opt)
        n_used = int(tab["used"].sum())

        rms = {}
        for key, pred in (("pga", A), ("pgv", V), ("intensity", I)):
            if key == "intensity":
                r = self.obs["I"] - pred
                name = "rms_I"
            else:
                r = np.log10(np.maximum(self.obs[key], 1e-9)) - np.log10(pred)
                name = f"rms_lg{key.upper()}"
            rms[name] = float(np.sqrt(np.nanmean(r ** 2))) if np.isfinite(r).any() else np.nan

        dist_true = (
            haversine_km(lon_opt, lat_opt, self.true_epi[0], self.true_epi[1])
            if self.true_epi else np.nan
        )

        if verbose:
            print(
                f"[{MODE_NAMES[mode]}] 最优震中 = ({lon_opt:.4f}, {lat_opt:.4f})"
                f"  走向 = {strike_opt:.1f}°"
                + (f"  距真实震中 {dist_true:.1f} km" if np.isfinite(dist_true) else "")
            )
            print(
                f"    chi2 = {chi2_opt:.2f}（{n_used}/{self.sta_lon.size} 台站参与）"
                f"  RMS_lgPGA = {rms['rms_lgPGA']:.3f}"
                f"  RMS_lgPGV = {rms['rms_lgPGV']:.3f}"
                f"  RMS_I = {rms['rms_I']:.3f}"
            )

        return {
            "mode": mode,
            "lon": lon_opt,
            "lat": lat_opt,
            "strike": strike_opt,
            "chi2": chi2_opt,
            "n_used": n_used,
            "n_sta": self.sta_lon.size,
            "dist_true_km": dist_true,
            **rms,
            "table": tab,
        }

    def station_table(self, mode, I, A, V, aI, bI, aA, bA, aV, bV, lon, lat):
        """每台站的观测 / 预测 / 残差表"""
        lg = lambda x: np.log10(np.maximum(x, 1e-9))
        res_lgPGA = lg(self.obs["pga"]) - np.log10(A)
        res_lgPGV = lg(self.obs["pgv"]) - np.log10(V)
        res_I = self.obs["I"] - I
        in_field = {
            "intensity": np.isfinite(I),
            "pga": np.isfinite(A),
            "pgv": np.isfinite(V),
            "pga_pgv": np.isfinite(A) & np.isfinite(V),
        }[mode]
        used = in_field & self.fixed_used[mode]

        R = np.sqrt(
            ((self.sta_lon - lon) * 111.32 * math.cos(math.radians(lat))) ** 2
            + ((self.sta_lat - lat) * 110.57) ** 2
        )
        return pd.DataFrame(
            {
                "Sta_ID": self.sta_id,
                "lon": np.round(self.sta_lon, 4),
                "lat": np.round(self.sta_lat, 4),
                "R_km": np.round(R, 2),
                "a_eq_I": np.round(aI, 2),
                "b_eq_I": np.round(bI, 2),
                "a_eq_PGA": np.round(aA, 2),
                "b_eq_PGA": np.round(bA, 2),
                "a_eq_PGV": np.round(aV, 2),
                "b_eq_PGV": np.round(bV, 2),
                "I_obs": np.round(self.obs["I"], 2),
                "I_pred": np.round(I, 2),
                "res_I": np.round(res_I, 3),
                "PGA_obs": np.round(self.obs["pga"], 3),
                "PGA_pred": np.round(A, 3),
                "res_lgPGA": np.round(res_lgPGA, 3),
                "PGV_obs": np.round(self.obs["pgv"], 3),
                "PGV_pred": np.round(V, 3),
                "res_lgPGV": np.round(res_lgPGV, 3),
                "used": used,
            }
        )


def invert_epicenter(
    data,
    Ms,
    region,
    epicenter,
    mode="pga_pgv",
    strike=349.0,
    strike_dev=30.0,
    fault_longi=None,
    fault_lati=None,
    strike_prior_sigma=10.0,
    strike_step=10.0,
    max_dist=200.0,
    extent=500.0,
    pad=1.0,
    coarse_step=0.1,
    fine_step=0.02,
    local_refine=0.15,
    true_epi=None,
    verbose=True,
):
    """
    震中反演主函数（可直接被其他程序调用）。

    明确输入：
        data           实测数据：文件路径 或 DataFrame（Sta_ID/lon/lat/I/pga/pgv）
        Ms             震级（面波震级）
        region         分区："青藏区" / "新疆区" / "东部区" / "中部区"
        epicenter      初始震中 (lon, lat)
        mode           迭代参数（线路）："pga" / "pgv" / "intensity" / "pga_pgv"
        strike         走向中心（正北=0°，顺时针；默认定日地震 349°）
        strike_dev     走向容许偏差（中心 ± 该值 内搜索；0 = 固定）
        fault_longi    反演震中范围散点经度（向量/矩阵均可，见 parse_fault_points）
        fault_lati     反演震中范围散点纬度
        strike_prior_sigma  走向高斯先验标准差（°），349° 最可能、越偏越不可能；≤0 关闭
        max_dist       GB18306 有效范围：长轴距/短轴距 ≤ 该值（默认 200 km），超限台站剔除
        true_epi       已知震中（仅用于输出误差评估，不参与打分）

    返回 dict：
        epicenter / lon / lat    最优震中
        strike / chi2 / n_used / n_sta / rms_lgPGA / rms_lgPGV / rms_I / dist_true_km
        table                    逐台站统计表（经纬度/实测/预测/残差/椭圆轴距/used）
        excluded_df              剔除台站（a_eq > max_dist）
        field / inverter         模型与反演器（供出图等后续调用）
    """
    if not (isinstance(Ms, (int, float)) and Ms > 0):
        raise ValueError("必须显式给出震级 Ms（数值）")
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一，收到：{mode}")
    if epicenter is None or len(epicenter) != 2:
        raise ValueError("必须给出初始震中 epicenter=(lon, lat)")

    sta_df, file_epi, file_ms = load_station_data(data)
    if true_epi is None:
        true_epi = file_epi
    epi = (float(epicenter[0]), float(epicenter[1]))

    # 0–200 km 预筛参考点 = 初始震中 + 中心走向
    filter_ref = (epi[0], epi[1], float(strike))
    field = GB18306EllipseField(region, float(Ms), extent=float(extent))
    inv = EpicenterInverter(
        field, sta_df, true_epi=true_epi, pad_deg=float(pad),
        max_dist=float(max_dist), filter_ref=filter_ref,
    )
    pts = parse_fault_points(fault_longi, fault_lati)
    if verbose:
        n_keep = int(inv.fixed_used[mode].sum())
        print(
            f"[{MODE_NAMES[mode]}] 预筛保留 {n_keep}/{len(sta_df)} 台站"
            f"（长轴距/短轴距 ≤ {max_dist:g} km）"
        )
        if pts is not None:
            print(f"   反演范围：Fault 散点 {pts[0].size} 个")
        else:
            print("   反演范围：台站外包络规则网格")

    res = inv.invert(
        mode,
        strike_center=float(strike),
        strike_range=float(strike_dev),
        strike_step=float(strike_step),
        strike_prior_sigma=float(strike_prior_sigma),
        coarse_step=float(coarse_step),
        fine_step=float(fine_step),
        fine_half=float(local_refine) if pts is not None else 0.5,
        seed=epi,
        cand_lon=(pts[0] if pts is not None else None),
        cand_lat=(pts[1] if pts is not None else None),
        verbose=verbose,
    )
    res["epicenter"] = (res["lon"], res["lat"])
    res["data"] = sta_df
    res["field"] = field
    res["inverter"] = inv
    res["true_epi"] = true_epi
    return res


def _parse_modes(text):
    if text.strip().lower() == "all":
        return list(MODES)
    modes = [m.strip().lower() for m in text.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise ValueError(f"未知线路：{m}，可选 {MODES} 或 all")
    return modes


def _param_meta(param):
    """单个参数的元信息：(obs列, pred列, a_eq列, b_eq列, 残差列, σ, 是否对数, 单位, 颜色)"""
    meta = {
        "PGA": ("PGA_obs", "PGA_pred", "a_eq_PGA", "b_eq_PGA", "res_lgPGA",
                SIGMA_GMM["pga"], True, "gal", "#d62728"),
        "PGV": ("PGV_obs", "PGV_pred", "a_eq_PGV", "b_eq_PGV", "res_lgPGV",
                SIGMA_GMM["pgv"], True, "cm/s", "#1f77b4"),
        "Intensity": ("I_obs", "I_pred", "a_eq_I", "b_eq_I", "res_I",
                      None, False, "", "#2ca02c"),
    }
    if param not in meta:
        raise ValueError(f"param 必须是 PGA/PGV/Intensity 之一，收到：{param}")
    return meta[param]


def plot_parameter_4panel(result, param, outpath, true_epi=None):
    """
    依托反演得到的最优震中，绘制单个参数（PGA / PGV / Intensity）的 4 联图：
        子图1  预测云图 + 实测散点（同一色标）
        子图2  实测散点 + 长轴衰减预测曲线（中值 ±1σ）
        子图3  残差（预测-实测）随等效长轴距 a_eq 变化
        子图4  残差散点 + 小提琴 + 箱线统计
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, r"C:\Users\12777\.codex\skills\stat-violin-plot\scripts")
        from stat_violin import apply_style, half_violin_box_scatter
    except Exception:
        from stat_violin import apply_style, half_violin_box_scatter
    apply_style()
    # 修复 CJK 回退：显式字体列表（西文 Times New Roman，中文 Microsoft YaHei）
    plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]

    obs_c, pred_c, ae_c, be_c, res_c, sigma, is_log, unit, color = _param_meta(param)
    field = result["field"]
    table = result["table"]
    used = table["used"].values.astype(bool)
    tab_u = table[used]
    lon0, lat0, strike = result["lon"], result["lat"], result["strike"]
    if true_epi is None:
        true_epi = result.get("true_epi")

    ylab = f"lg10({param}) / {unit}" if (is_log and unit) else (
        f"lg10({param})" if is_log else param
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ============ 子图1：预测云图 + 实测散点（同一色标） ============
    ax = axes[0, 0]
    clon = 111.32 * math.cos(math.radians(lat0))
    if len(tab_u):
        half_km = max(
            (tab_u["lon"].max() - tab_u["lon"].min()) * clon,
            (tab_u["lat"].max() - tab_u["lat"].min()) * 110.57,
        ) / 2.0 + 30.0
    else:
        half_km = 80.0
    half_km = max(half_km, 60.0)
    n = 121
    glon = np.linspace(lon0 - half_km / clon, lon0 + half_km / clon, n)
    glat = np.linspace(lat0 - half_km / 110.57, lat0 + half_km / 110.57, n)
    GLON, GLAT = np.meshgrid(glon, glat)
    Ip, Ap, Vp = field.predict(lon0, lat0, strike, GLON.ravel(), GLAT.ravel())[:3]
    pred_field = {"PGA": Ap, "PGV": Vp, "Intensity": Ip}[param].reshape(GLON.shape)
    field_vals = np.log10(pred_field) if is_log else pred_field
    lo_f = float(np.nanpercentile(field_vals, 2))
    hi_f = float(np.nanpercentile(field_vals, 98))
    if not np.isfinite(lo_f) or lo_f == hi_f:
        lo_f, hi_f = -2.0, 3.0
    lev = np.linspace(lo_f, hi_f, 21)
    cf = ax.contourf(GLON, GLAT, field_vals, levels=lev, cmap="jet_r", extend="both")
    obs_all = np.log10(np.maximum(table[obs_c].values, 1e-12)) if is_log else table[obs_c].values
    ax.scatter(table["lon"], table["lat"], c=obs_all, cmap="jet_r",
               vmin=lo_f, vmax=hi_f, s=42, edgecolors="k", linewidths=0.5, zorder=6)
    ex = result["inverter"].excluded_df
    if ex is not None and len(ex):
        ax.scatter(ex["lon"], ex["lat"], marker="x", s=34, color="gray",
                   linewidths=1.1, zorder=5, label="剔除台站")
    ax.plot(lon0, lat0, marker="*", ms=17, color="black",
            markeredgecolor="white", zorder=8, label="最优震中")
    if true_epi is not None:
        ax.plot(true_epi[0], true_epi[1], marker="^", ms=11, color="blue",
                markeredgecolor="white", zorder=8, label="已知震中")
    cb = fig.colorbar(cf, ax=ax, shrink=0.9)
    cb.set_label(ylab)
    ax.set_title(f"子图1  预测云图 + 实测散点（{param}）")
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_aspect(1.0 / math.cos(math.radians((GLAT.min() + GLAT.max()) / 2.0)))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    # ============ 子图2：实测散点 + 预测衰减曲线（中值 ±1σ，长轴） ============
    ax = axes[0, 1]
    a_max = max(float(tab_u[ae_c].max()) if len(tab_u) else 40.0, 40.0)
    r_curve = np.linspace(1.0, max(a_max * 1.05, 50.0), 300)
    if param == "Intensity":
        ri = [field.cal_i.calculate(field.Ms, float(r), field.region, "长轴") for r in r_curve]
        med = np.array([d["mean"] for d in ri])
        lo_c = np.array([d["lower_1sigma"] for d in ri])
        up = np.array([d["upper_1sigma"] for d in ri])
        sigma_use = field.sigma_I
    else:
        gmm = [field.cal_g.calculate(field.Ms, float(r), field.region, "长轴") for r in r_curve]
        tup = [g[0 if param == "PGA" else 1] for g in gmm]
        med = np.array([t[0] for t in tup])
        lo_c = np.array([t[1] for t in tup])
        up = np.array([t[2] for t in tup])
        sigma_use = SIGMA_GMM["pga" if param == "PGA" else "pgv"]
    yc = np.log10(np.maximum(med, 1e-12)) if is_log else med
    ylo_c = np.log10(np.maximum(lo_c, 1e-12)) if is_log else lo_c
    yup_c = np.log10(np.maximum(up, 1e-12)) if is_log else up
    ax.plot(r_curve, yc, color=color, lw=1.8, label=f"{param} 中值（长轴）")
    ax.fill_between(r_curve, ylo_c, yup_c, color=color, alpha=0.18, label="±1σ")
    obs_y = np.log10(np.maximum(tab_u[obs_c].values, 1e-12)) if is_log else tab_u[obs_c].values
    ax.scatter(tab_u[ae_c], obs_y, s=42, color="k", edgecolors="white",
               linewidths=0.4, zorder=5, label="实测（保留台站）")
    ax.set_xlabel("等效长轴距 a_eq (km)")
    ax.set_ylabel(ylab)
    ax.set_title(f"子图2  实测散点 + 预测曲线 ±1σ（{param}，长轴）")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    # ============ 子图3：残差（预测-实测）随 a_eq 变化 ============
    ax = axes[1, 0]
    res_pm = -tab_u[res_c].values.astype(float)   # 预测 - 实测
    ax.scatter(tab_u[ae_c], res_pm, s=42, color=color, edgecolors="k",
               linewidths=0.4, zorder=5)
    ax.axhline(0, color="k", lw=1.0)
    if sigma_use:
        ax.axhline(sigma_use, color="gray", lw=0.8, ls="--")
        ax.axhline(-sigma_use, color="gray", lw=0.8, ls="--", label=f"±σ = {sigma_use:.3g}")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("等效长轴距 a_eq (km)")
    ax.set_ylabel("残差（预测-实测）")
    ax.set_title(f"子图3  残差随距离变化（{param}）")
    ax.grid(alpha=0.3, linestyle="--")

    # ============ 子图4：残差散点 + 小提琴 + 箱线统计 ============
    ax = axes[1, 1]
    half_violin_box_scatter(ax, res_pm, 0, color, value_fmt="{:.3f}", s=24)
    ax.set_xlim(-0.75, 0.75)
    ax.set_xticks([])
    ax.set_xlabel("残差分布")
    ax.set_ylabel("残差（预测-实测）")
    ax.set_title(f"子图4  残差统计（N={len(res_pm)}）")
    ax.grid(alpha=0.2, linestyle="--", axis="y")

    fig.suptitle(
        f"{param} 4联图 | {MODE_NAMES[result['mode']]} | 最优震中 "
        f"({result['lon']:.4f}, {result['lat']:.4f}) | strike {result['strike']:.1f}°"
        + (
            f" | 距已知震中 {result['dist_true_km']:.1f} km"
            if np.isfinite(result["dist_true_km"]) else ""
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图件：{os.path.abspath(outpath)}")
    return outpath


def export_statistics(result, outpath):
    """逐台站统计 txt：经纬度、实测值、预测值、残差、椭圆轴距等"""
    result["table"].to_csv(outpath, sep="\t", index=False, encoding="utf-8-sig")
    print(f"已保存统计表：{os.path.abspath(outpath)}")
    return outpath


def _read_fault_file(path):
    """读取反演范围散点文件：两列（经度 纬度），支持逗号/空白/tab 分隔，可带表头"""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue  # 表头行
    if not rows:
        raise ValueError(f"Fault 文件无有效数据：{path}")
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1]


def main():
    ap = argparse.ArgumentParser(description="GB18306 椭圆衰减震中反演")
    ap.add_argument("--input", default=None, help="台站记录文件（tab 分隔）")
    ap.add_argument("--modes", default="all", help="线路：pga_pgv,intensity,pga,pgv 或 all")
    ap.add_argument("--region", default="青藏区", help="分区：东部区/中部区/新疆区/青藏区")
    ap.add_argument("--ms", type=float, default=None, help="面波震级（默认取文件 Mag 列中位数）")
    ap.add_argument("--seed-lon", type=float, default=None, help="初始震中经度")
    ap.add_argument("--seed-lat", type=float, default=None, help="初始震中纬度")
    ap.add_argument("--strike", type=float, default=349.0, help="走向中心（默认定日地震 349°）")
    ap.add_argument("--strike-range", type=float, default=30.0, help="走向容许偏差（±值；0=固定）")
    ap.add_argument("--strike-step", type=float, default=10.0, help="走向搜索步长（度）")
    ap.add_argument("--strike-prior-sigma", type=float, default=10.0,
                    help="走向高斯先验σ（°）：中心最可能，≤0 关闭先验")
    ap.add_argument("--fault-longi", type=str, default=None,
                    help="反演范围散点经度，逗号分隔，如 87.0,87.1,87.2")
    ap.add_argument("--fault-lati", type=str, default=None,
                    help="反演范围散点纬度，逗号分隔")
    ap.add_argument("--fault-file", type=str, default=None,
                    help="反演范围散点文件（两列：经度 纬度）")
    ap.add_argument("--max-dist", type=float, default=200.0,
                    help="GB18306 有效范围（长轴距/短轴距 km），超限台站剔除")
    ap.add_argument("--extent", type=float, default=500.0, help="衰减场最大距离 km")
    ap.add_argument("--pad", type=float, default=1.0, help="无 Fault 点时外包络外扩度数")
    ap.add_argument("--local-refine", type=float, default=0.15,
                    help="Fault 最优散点附近细网格半宽（度）；0=仅用给定散点")
    ap.add_argument("--coarse-step", type=float, default=0.1, help="粗网格步长（度）")
    ap.add_argument("--fine-step", type=float, default=0.02, help="细网格步长（度）")
    ap.add_argument("--outdir", default="epicenter_inversion_out", help="输出目录")
    ap.add_argument("--no-fig", action="store_true", help="不输出图件")
    args = ap.parse_args()

    if args.input is None:
        default_input = os.path.join(
            HERE, "20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt"
        )
        args.input = default_input if os.path.exists(default_input) else None
    if args.input is None or not os.path.exists(args.input):
        ap.error(f"找不到输入文件：{args.input}，请用 --input 指定")

    sta_df, true_epi, file_ms = read_station_file(args.input)
    ms = args.ms if args.ms is not None else (file_ms if file_ms is not None else 6.8)
    modes = _parse_modes(args.modes)
    if args.seed_lon is None or args.seed_lat is None:
        if true_epi is not None:
            args.seed_lon, args.seed_lat = true_epi
        else:
            args.seed_lon = float(sta_df["lon"].mean())
            args.seed_lat = float(sta_df["lat"].mean())
    epicenter = (args.seed_lon, args.seed_lat)

    # 反演范围散点
    fault_longi = fault_lati = None
    if args.fault_file:
        fault_longi, fault_lati = _read_fault_file(args.fault_file)
    elif args.fault_longi or args.fault_lati:
        if not (args.fault_longi and args.fault_lati):
            ap.error("--fault-longi 与 --fault-lati 必须同时给出")
        fault_longi = np.array([float(x) for x in args.fault_longi.split(",")])
        fault_lati = np.array([float(x) for x in args.fault_lati.split(",")])
        if fault_longi.size != fault_lati.size:
            ap.error("--fault-longi 与 --fault-lati 数量不一致")

    print("=" * 78)
    print(f"输入文件：{os.path.abspath(args.input)}")
    print(f"台站数：{len(sta_df)}  |  Ms = {ms:g}  |  分区：{args.region}")
    print(f"初始震中：({epicenter[0]:.4f}, {epicenter[1]:.4f})")
    print(
        f"走向：{args.strike % 360:.1f}° ± {args.strike_range:g}°"
        + (f"（先验σ = {args.strike_prior_sigma:g}°）" if args.strike_prior_sigma > 0 else "（无先验）")
    )
    if fault_longi is not None:
        print(f"反演范围：Fault 散点 {fault_longi.size} 个")
    else:
        print("反演范围：台站外包络外扩（--pad 可调；可用 --fault-file 指定散点）")
    print("=" * 78)

    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for mode in modes:
        print(f"\n>>> 线路：{MODE_NAMES[mode]}")
        res = invert_epicenter(
            data=args.input,
            Ms=ms,
            region=args.region,
            epicenter=epicenter,
            mode=mode,
            strike=args.strike,
            strike_dev=args.strike_range,
            fault_longi=fault_longi,
            fault_lati=fault_lati,
            strike_prior_sigma=args.strike_prior_sigma,
            strike_step=args.strike_step,
            max_dist=args.max_dist,
            extent=args.extent,
            pad=args.pad,
            coarse_step=args.coarse_step,
            fine_step=args.fine_step,
            local_refine=args.local_refine,
            true_epi=true_epi,
            verbose=True,
        )
        results.append(res)
        export_statistics(res, os.path.join(args.outdir, f"stats_{mode}.txt"))
        res["table"].to_csv(
            os.path.join(args.outdir, f"epicenter_invert_{mode}.csv"),
            index=False, encoding="utf-8-sig",
        )
        if not args.no_fig:
            for p in ("PGA", "PGV", "Intensity"):
                plot_parameter_4panel(
                    res, p, os.path.join(args.outdir, f"fig_{mode}_{p}.png"),
                    true_epi=true_epi,
                )

    summary_rows = [
        {
            "线路": MODE_NAMES[r["mode"]],
            "mode": r["mode"],
            "经度": round(r["lon"], 4),
            "纬度": round(r["lat"], 4),
            "走向": round(r["strike"], 1),
            "参与台站": f"{r['n_used']}/{r['n_sta']}",
            "chi2": round(r["chi2"], 2),
            "chi2/站": round(r["chi2"] / max(r["n_used"], 1), 2),
            "RMS_lgPGA": round(r["rms_lgPGA"], 3),
            "RMS_lgPGV": round(r["rms_lgPGV"], 3),
            "RMS_I": round(r["rms_I"], 3),
            "距真实震中km": (
                round(r["dist_true_km"], 2) if np.isfinite(r["dist_true_km"]) else np.nan
            ),
        }
        for r in results
    ]
    summary = pd.DataFrame(summary_rows)
    sum_path = os.path.join(args.outdir, "epicenter_invert_summary.csv")
    summary.to_csv(sum_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("反演结果汇总：")
    print(summary.to_string(index=False))
    print("=" * 78)
    print(f"汇总表：{os.path.abspath(sum_path)}")


if __name__ == "__main__":
    main()
