# -*- coding: utf-8 -*-
"""
CEA2019 地震动参数预测（PGA / PGV / PSA 周期点）—— 应用主程序
==============================================================
参照 GB18306_Pre 的思路：输入震中经纬度、走向、分区、面波震级 Ms 和
周期点列表，沿长轴/短轴算衰减曲线 → 椭圆环 → 台站预测 → 综合表/云图。

周期点约定：
    -1            PGA（峰值加速度，gal）
    -2            PGV（峰值速度，cm/s）
    0             PGA（与 -1 等价）
    0 ~ 0.04      PSA：在 PGA(T=0) 与 PSA(0.04s) 之间线性插值
    0.04 ~ 6      PSA(T)：周期 T 秒的反应谱加速度（gal）

周期规则：
    - T > 6 s   ：不进行外插，结果记 NaN；
    - T < 0     ：只允许 -1 / -2 / 0，其他负数报错；
    - 常用点：-1、-2、0.3、1、3、6。

色标规则（云图）：
    - PGA / PSA 共用 PGA 分界 [1,2,5,10,25,50,100,200,400,800,1500]；
    - PGV 分界 = PGA ÷ 10（因为 PGV 数值约为 PGA 的 1/10）；
    - PSA(T) 分界 = PGA 分界 × (PSA(T)/PGA 参考比值)，
      参考比值取 R=10 km 处长轴中值 PSA(T)/PGA，随周期自动缩放。
    - colorbar 刻度：PSA 保留 1 位小数（整数直接显示整数）。

参数命名与单位：
    - PGA           → "PGA"，单位 gal
    - PGV           → "PGV"，单位 cm/s
    - PSA           → "PSA(T=0.30s)" 这种标准写法，单位固定 cm/s²

整合图（plot_period_fields）：
    第一排：云图（每个周期点一列，带 colorbar）
    第二排：衰减曲线（长轴/短轴中值 + ±1σ，X 轴距离 log，Y 轴 log）

依赖文件（同目录）：
    CEA2019_class.py  +  《GB长短轴衰减关系系数--区划2019.xlsx》

常用入口：
    predict_period_values(...)   按周期点预测台站地震动参数
    export_period_table(...)     综合表 TXT（每个周期点：值 + 所在椭圆长短轴）
    plot_period_fields(...)      2×N 整合图（云图 + 衰减曲线）
"""

import math
import os
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 不弹窗口，只保存图片
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# scipy 用于云图插值；没装时云图无法生成（预测/表格不受影响）
try:
    from scipy.interpolate import griddata

    HAVE_GRIDDATA = True
except Exception:
    HAVE_GRIDDATA = False

# 让中文提示在 Windows 命令行里正常显示
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# UTM 投影（经纬度 → 公里坐标），没装时自动退化为简单近似
try:
    from pyproj import Transformer

    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

# 找到 CEA2019_class.py（和本文件同一目录）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from CEA2019_class import CEA2019

# 字体：英文用 Times New Roman，中文自动回退 SimHei
plt.rcParams["font.family"] = [
    "Times New Roman",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]
plt.rcParams["font.size"] = 8.5
plt.rcParams["axes.unicode_minus"] = False

# USGS MMI 色标（I 白 → X+ 深红）
USGS_MMI_COLORS = [
    "#FFFFFF",
    "#BFCCFF",
    "#A0E6FF",
    "#80FFFF",
    "#7AFF93",
    "#FFFF00",
    "#FFC800",
    "#FF9100",
    "#FF0000",
    "#800000",
]

# PGA / PSA 基础分界（gal）；PGV 分界 = ÷10；PSA(T) 分界 = ×比值
PGA_LEVELS = [1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1500]
PGV_LEVELS = [0.1, 0.2, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 150]


# ==================== 周期点工具 ====================


def period_label(T):
    """周期点 → 标准化标签：-1/0 → PGA；-2 → PGV；
    数值 → "PSA(T=0.30s)" 格式（两位小数）"""
    if T in (-1, 0):
        return "PGA"
    if T == -2:
        return "PGV"
    return f"PSA(T={float(T):.2f}s)"


def unit_label(T):
    """单位：PGA → gal；PGV → cm/s；PSA → cm/s²"""
    if T == -2:
        return "cm/s"
    if T in (-1, 0):
        return "gal"
    return "cm/s²"


def validate_periods(periods):
    """
    周期点校验：
        -1 / -2 / 0  合法（PGA / PGV / PGA）；
        0 < T < 0.04 合法（在 PGA 与 PSA(0.04) 之间线性插值）；
        0.04 ~ 6     合法（PSA）；
        T > 6        允许传入，但结果统一记 NaN（不外插）；
        其他负数     报错。
    返回规范化周期列表。
    """
    out = []
    for T in periods:
        if T in (-1, -2, 0):
            out.append(T)
            continue
        Tf = float(T)
        if Tf < 0:
            raise ValueError(
                f"周期 {Tf:g} 不支持，只允许 -1(PGA)、-2(PGV)、"
                f"0(PGA) 和正数周期"
            )
        out.append(Tf)
    return out


def _region_core(region):
    """区域名归一化：'青藏区' 和 '青藏' 都接受（CEA2019 构造时要求不带'区'）"""
    return str(region).replace("区", "")


# ==================== 坐标换算（与 GB18306_Pre 相同） ====================


def km_to_lonlat(lon, lat, east_km, north_km, utm_zone):
    """相对震中的 东向(km)/北向(km) → 经纬度（UTM 投影）"""
    if HAVE_PYPROJ:
        epsg = (
            f"epsg:326{utm_zone:02d}"
            if lat >= 0
            else f"epsg:327{utm_zone:02d}"
        )
        fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
        epi_x, epi_y = fwd.transform(lon, lat)
        inv = Transformer.from_crs(epsg, "epsg:4326", always_xy=True)
        return inv.transform(
            epi_x + np.asarray(east_km) * 1000.0,
            epi_y + np.asarray(north_km) * 1000.0,
        )
    dlat = np.asarray(north_km) / 111.32
    dlon = np.asarray(east_km) / (111.32 * math.cos(math.radians(lat)))
    return lon + dlon, lat + dlat


def lonlat_to_utm(lon, lat, utm_zone):
    """经纬度 → UTM 坐标（米）"""
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    return fwd.transform(lon, lat)


def fmt_pgv(value):
    """标签格式化：整数显示整数，小数保留 1 位，很小的值保留有效数字"""
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def fmt_psa(value):
    """PSA 色标刻度：整数显示整数，否则保留 1 位小数"""
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.1f}"


# ==================== 周期系数与衰减曲线 ====================

# 系数缓存：同一 (区域, 轴向, 周期, 震级) 只读一次 Excel
_COEFF_CACHE = {}
# 计算器实例缓存：同一 (区域, 轴向) 只读一次 Excel（避免逐周期重读文件）
_INST_CACHE = {}


def _get_cal(region_core, axis):
    """取（并缓存）某区域、某轴向的 CEA2019 计算器实例"""
    key = (region_core, axis)
    if key not in _INST_CACHE:
        _INST_CACHE[key] = CEA2019(region_core, axis)
    return _INST_CACHE[key]


def _period_coeffs(region_core, axis, period, M):
    """
    取某周期点、某轴向的系数，并按震级分段选 A/B。

    周期特殊处理：
        -1 或 0  → PGA 系数；
        -2       → PGV 系数；
        0 < T < 0.04 → 在 PGA(T=0) 与 PSA(0.04) 之间线性插值全部系数；
        其他数值 T → 交给 CEA2019 类在已知周期点之间插值。

    返回 (A, B, C, sigma, exp_term)，exp_term = D*exp(E*M)。
    """
    key = (region_core, axis, float(period), M)
    if key in _COEFF_CACHE:
        return _COEFF_CACHE[key]

    cal = _get_cal(region_core, axis)
    T = float(period)
    if T in (-1.0, 0.0):
        row = cal._get_coefficients(-1)  # PGA 行
    elif T == -2.0:
        row = cal._get_coefficients(-2)  # PGV 行
    elif 0.0 < T < 0.04:
        # 0~0.04s：PGA(0s) 与 PSA(0.04s) 之间线性插值
        row_pga = cal._get_coefficients(-1)
        row_04 = cal._get_coefficients(0.04)
        w = T / 0.04
        row = {}
        for col in ("A1", "B1", "A2", "B2", "C", "D", "E", "σ"):
            row[col] = row_pga[col] + (row_04[col] - row_pga[col]) * w
    else:
        row = cal._get_coefficients(T)

    A = row["A1"] if M < 6.5 else row["A2"]
    B = row["B1"] if M < 6.5 else row["B2"]
    C = row["C"]
    sigma = row["σ"]
    exp_term = row["D"] * np.exp(row["E"] * M)
    _COEFF_CACHE[key] = (A, B, C, sigma, exp_term)
    return _COEFF_CACHE[key]


def _period_value(T, M, R, region_core, axis):
    """
    某周期点、某轴向的中值及 ±1σ（用系数公式直接算，支持 0~0.04 插值）。
    返回 (Y, lower, upper)，PGA/PSA 单位 gal，PGV 单位 cm/s。
    """
    A, B, C, sigma, exp_term = _period_coeffs(region_core, axis, T, M)
    lgY = A + B * M - C * np.log10(R + exp_term)
    Y = 10.0**lgY
    return Y, Y / 10.0**sigma, Y * 10.0**sigma


def _period_curves_sigma(T, M, region_core, axis, r_scan):
    """沿某轴向的 中值/±1σ 三条曲线（数组）"""
    med = []
    lo = []
    up = []
    for r in r_scan:
        y, l, u = _period_value(T, M, r, region_core, axis)
        med.append(y)
        lo.append(l)
        up.append(u)
    return np.asarray(med), np.asarray(lo), np.asarray(up)


def calc_axis_curves(period, M, region, r_scan):
    """
    沿长轴、短轴计算该周期点的中值衰减曲线。
    返回 (long_arr, short_arr)。
    """
    rc = _region_core(region)
    long_arr = np.array(
        [_period_value(period, M, r, rc, "长轴")[0] for r in r_scan],
        dtype=float,
    )
    short_arr = np.array(
        [_period_value(period, M, r, rc, "短轴")[0] for r in r_scan],
        dtype=float,
    )
    return long_arr, short_arr


def _ellipse_radii_for_period(Y, period, M, region_core):
    """
    由地震动参数值 Y 解析反算"所在椭圆" (a, b)，与 invert_R 同公式
    （含 0.01 km 最小截断；短轴不超过长轴）。单位 km。
    """
    A_l, B_l, C_l, _, e_l = _period_coeffs(region_core, "长轴", period, M)
    A_s, B_s, C_s, _, e_s = _period_coeffs(region_core, "短轴", period, M)
    Y = np.asarray(Y, dtype=float)
    lg = np.log10(Y)
    a = 10.0 ** ((A_l + B_l * M - lg) / C_l) - e_l
    b = 10.0 ** ((A_s + B_s * M - lg) / C_s) - e_s
    b = np.minimum(b, a)  # 退化保护：短轴不超过长轴
    a = np.maximum(a, 0.01)
    b = np.maximum(b, 0.01)
    return a, b


def _solve_period_values(period, M, region_core, r_scan, R, theta):
    """
    向量化二分：求台站（极坐标 R、相对长轴夹角 theta）在该周期点的值，
    以及所在椭圆的长/短轴距。
    返回 (val, a_eq, b_eq)；场外台站 / T > 6s → NaN。
    """
    nan = np.full_like(R, np.nan)
    if float(period) > 6.0:  # 大于 6s 不外插 → 全部 NaN
        return nan.copy(), nan.copy(), nan.copy()

    long_arr, short_arr = calc_axis_curves(period, M, region_core, r_scan)

    # 最外圈椭圆：长轴距 = extent，短轴距 = 短轴曲线上与 long(extent) 同值的距离
    a_max = float(r_scan[-1])
    v_ext = float(long_arr[-1])
    b_ext = float(np.interp(v_ext, short_arr[::-1], r_scan[::-1]))
    b_ext = min(b_ext, a_max)  # 退化保护

    # 台站在该方向上的最外圈半径，超出者无法预测
    ct = np.cos(theta)
    st = np.sin(theta)
    r_outer = a_max * b_ext / np.sqrt((b_ext * ct) ** 2 + (a_max * st) ** 2)
    inside = np.isfinite(r_outer) & (R <= r_outer)

    A_l, B_l, C_l, _, e_l = _period_coeffs(region_core, "长轴", period, M)
    A_s, B_s, C_s, _, e_s = _period_coeffs(region_core, "短轴", period, M)

    def radii(v):
        lg = np.log10(v)
        a = 10.0 ** ((A_l + B_l * M - lg) / C_l) - e_l
        b = np.minimum(10.0 ** ((A_s + B_s * M - lg) / C_s) - e_s, a)
        a = np.maximum(a, 1e-9)  # 避免除零/负数
        b = np.maximum(b, 1e-9)
        return a, b

    def radius(v):
        a, b = radii(v)
        return a * b / np.sqrt((b * ct) ** 2 + (a * st) ** 2)

    # 值域：最外圈值（小）~ 震中附近长轴值（大）
    v_lo = np.full_like(R, v_ext)
    v_hi = np.full_like(R, float(long_arr[0]))
    for _ in range(60):
        v_mid = (v_lo + v_hi) / 2.0
        f = radius(v_mid) - R
        v_lo = np.where(f > 0.0, v_mid, v_lo)  # 圈还太大 → 值要更大
        v_hi = np.where(f <= 0.0, v_mid, v_hi)  # 圈已太小 → 值要更小
    V = (v_lo + v_hi) / 2.0
    a_eq, b_eq = radii(V)
    mask = inside & np.isfinite(V)
    return (
        np.where(mask, V, np.nan),
        np.where(mask, np.maximum(a_eq, 0.01), np.nan),
        np.where(mask, np.maximum(b_eq, 0.01), np.nan),
    )


# ==================== 台站预测与综合表 ====================


def predict_period_values(
    lon, lat, strike, region, Ms, periods, sta_lon, sta_lat, extent=400
):
    """
    对每个周期点预测台站的地震动参数。

    参数：
        lon, lat   震中经纬度；strike 走向（正北=0°，顺时针）
        region     分区（如"新疆区"，不带"区"也可以）
        Ms         面波震级
        periods    周期点列表，如 [-1, -2, 0.3, 1, 3, 6]
        sta_lon / sta_lat  台站经纬度（标量/数组/矩阵均可）
        extent     场范围半径（km），默认 400

    返回：
        dict：周期标签（PGA/PGV/PSA0.3...）-> 值数组（与输入同形状）
        T > 6s 的周期结果为 NaN；场外台站为 NaN。
    """
    periods = validate_periods(periods)
    rc = _region_core(region)
    strike = strike % 360.0
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    sta_lon = np.asarray(sta_lon, dtype=float)
    sta_lat = np.asarray(sta_lat, dtype=float)
    shape = sta_lon.shape
    fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    sx, sy = np.asarray(fwd.transform(sta_lon.ravel(), sta_lat.ravel()))
    dx = (sx - epi_x) / 1000.0
    dy = (sy - epi_y) / 1000.0
    R = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) - math.radians(90.0 - strike)

    r_scan = np.arange(1.0, extent + 1.0, 1.0)
    out = {}
    for T in periods:
        label = period_label(T)
        v, _, _ = _solve_period_values(T, Ms, rc, r_scan, R, theta)
        out[label] = v.reshape(shape)
    return out


def export_period_table(
    lon,
    lat,
    strike,
    region,
    Ms,
    periods,
    sta_lon,
    sta_lat,
    sta_id=None,
    extent=400,
    output_file=None,
):
    """
    台站综合表：每个周期点输出 3 列（值、所在椭圆长轴距、短轴距），导出 TXT

    表列：
        Sta_ID  Sta_longi  Sta_lati  Repi(km)
        每个周期点（如 -1,-2,0.3,1,3,6）：
            PGA(gal) / PGV(cm/s) / PSA(T=0.30s)(cm/s²) ...
            Repi_long_PGA(km)  Repi_short_PGA(km) ...

    说明：
        - 值先取整到 2 位小数，再用取整后的值解析反算长短轴，表内严格自洽
          （可用 CEA2019.invert_R 验证）；
        - 0 < T < 0.04s 按 PGA(0)~PSA(0.04) 线性插值；
        - T > 6s：值、长短轴全部 NaN（不外插）；
        - 场外台站：NaN。
    """
    periods = validate_periods(periods)
    rc = _region_core(region)
    strike = strike % 360.0
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    sta_lon = np.asarray(sta_lon, dtype=float)
    sta_lat = np.asarray(sta_lat, dtype=float)
    if sta_lon.shape != sta_lat.shape:
        raise ValueError("sta_lon 与 sta_lat 形状必须一致！")

    fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    sx, sy = np.asarray(fwd.transform(sta_lon.ravel(), sta_lat.ravel()))
    dx = (sx - epi_x) / 1000.0
    dy = (sy - epi_y) / 1000.0
    R = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) - math.radians(90.0 - strike)

    r_scan = np.arange(1.0, extent + 1.0, 1.0)

    if sta_id is None:
        sta_id = [f"S{i + 1}" for i in range(R.size)]
    else:
        sta_id = np.asarray(sta_id).ravel()

    df = pd.DataFrame(
        {
            "Sta_ID": sta_id,
            "Sta_longi": np.round(sta_lon.ravel(), 4),
            "Sta_lati": np.round(sta_lat.ravel(), 4),
            "Repi(km)": np.round(R, 2),  # 震中距
        }
    )

    for T in periods:
        label = period_label(T)
        unit = unit_label(T)
        if float(T) > 6.0:
            df[f"{label}({unit})"] = np.full(R.size, np.nan)
            df[f"Repi_long_{label}(km)"] = np.full(R.size, np.nan)
            df[f"Repi_short_{label}(km)"] = np.full(R.size, np.nan)
            continue
        v_raw, _, _ = _solve_period_values(T, Ms, rc, r_scan, R, theta)
        v = np.round(v_raw, 2)  # 先取整
        a_eq, b_eq = _ellipse_radii_for_period(v, T, Ms, rc)  # 再反算椭圆
        df[f"{label}({unit})"] = v
        df[f"Repi_long_{label}(km)"] = np.round(a_eq, 2)
        df[f"Repi_short_{label}(km)"] = np.round(b_eq, 2)

    if output_file is None:
        os.makedirs("Test_output", exist_ok=True)
        output_file = f"./Test_output/CEA2019_pre_table_{rc}区_Ms{Ms:g}.txt"
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8-sig")
    print(f"已导出：{os.path.abspath(output_file)}")
    return df


# ==================== 云图与整合图 ====================


def minor_axis_curve(long_arr, short_arr, r_scan):
    """每个长轴距离 t 对应的短轴距 b(t)（退化处 b = t，画圆）"""
    b = np.interp(long_arr, short_arr[::-1], r_scan[::-1])
    b = np.where(long_arr >= short_arr[0], r_scan[0], b)
    b = np.where(long_arr <= short_arr[-1], np.nan, b)
    b = np.where(~np.isfinite(b) | (b > r_scan), r_scan, b)
    return b


def calc_field_ellipse(long_arr, short_arr, r_scan, angle=0.0):
    """由长短轴曲线生成椭圆环点云 (x_km, y_km, val)"""
    b_t = minor_axis_curve(long_arr, short_arr, r_scan)
    pts = []
    for i, t in enumerate(r_scan):
        r_s = b_t[i]
        if not np.isfinite(r_s):
            continue
        theta = np.linspace(0.0, 2.0 * math.pi, 120)
        x_e = t * np.cos(theta)
        y_e = r_s * np.sin(theta)
        x = x_e * math.cos(angle) - y_e * math.sin(angle)
        y = x_e * math.sin(angle) + y_e * math.cos(angle)
        for j in range(0, len(theta), 2):
            pts.append((x[j], y[j], long_arr[i]))
    return np.array(pts)


def interpolate_to_ll(pts_km, epi_x, epi_y, epsg, extent=400, n=400):
    """点云 griddata 插值成网格场，再转经纬度"""
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)
    xg, yg = np.meshgrid(xs, ys)
    if HAVE_GRIDDATA:
        grid = griddata(
            (pts_km[:, 0], pts_km[:, 1]),
            pts_km[:, 2],
            (xg, yg),
            method="linear",
        )
    else:
        grid = np.full_like(xg, np.nan)
        print("警告：未安装 scipy，无法插值生成云图！")
    dist = np.sqrt(xg**2 + yg**2)
    grid = np.ma.masked_where(dist < 0.5, grid)
    grid = np.ma.masked_where(dist > extent, grid)
    xg_utm = epi_x + xg * 1000.0
    yg_utm = epi_y + yg * 1000.0
    inv = Transformer.from_crs(epsg, "epsg:4326", always_xy=True)
    lons, lats = inv.transform(xg_utm, yg_utm)
    return lons, lats, grid


def _psa_pga_ratio(T, M, region_core, r_ref=10.0):
    """
    PSA(T) / PGA 的参考比值（长轴中值，R = r_ref km）。
    PGA / PGV / 0 返回 None（用各自的固定色标）。
    """
    if T in (-1, -2, 0):
        return None
    pga = _period_value(-1, M, r_ref, region_core, "长轴")[0]
    psa = _period_value(T, M, r_ref, region_core, "长轴")[0]
    if pga and psa and pga > 0:
        return float(psa) / float(pga)
    return 1.0


def _period_levels(T, M, region_core):
    """该周期点的色标分界：
    PGV → PGA÷10；PSA(T) → PGA × (PSA/PGA 参考比值)；PGA/0 → PGA 分界。"""
    if T == -2:
        return PGV_LEVELS
    ratio = _psa_pga_ratio(T, M, region_core)
    if ratio is None:
        return PGA_LEVELS
    return [round(x * ratio, 4) for x in PGA_LEVELS]


def plot_period_fields(
    lon, lat, strike, region, Ms, periods, extent=400, output_file=None
):
    """
    2×N 整合图：
      第一排：云图（每个周期点一列，带 colorbar）
      第二排：衰减曲线（长轴/短轴中值 + ±1σ，X 轴距离 log，Y 轴 log）

    PSA 色标分界按该周期 PSA/PGA 参考比值缩放（类比 PGV = PGA÷10）。
    T > 6s 的周期跳过。
    """
    periods = [T for T in validate_periods(periods) if float(T) <= 6.0]
    if not periods:
        raise ValueError("没有可绘制的周期点（所有周期都 > 6s？）")
    rc = _region_core(region)
    strike = strike % 360.0
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)
    r_scan = np.arange(1.0, extent + 1.0, 1.0)
    angle = math.radians(90.0 - strike)

    cmap = ListedColormap(USGS_MMI_COLORS, name="usgs_mmi")
    cmap.set_under(USGS_MMI_COLORS[0])
    cmap.set_over(USGS_MMI_COLORS[-1])

    cm2in = 1.0 / 2.54
    n = len(periods)
    fig, axs = plt.subplots(2, n, figsize=(12 * n * cm2in, 18 * cm2in))
    if n == 1:
        axs = axs.reshape(2, 1)

    for i, T in enumerate(periods):
        label = period_label(T)
        unit = unit_label(T)
        levels = _period_levels(T, Ms, rc)
        norm = BoundaryNorm(levels, ncolors=len(USGS_MMI_COLORS))

        # 该周期点的长/短轴中值 + ±1σ 曲线
        long_m, long_lo, long_up = _period_curves_sigma(
            T, Ms, rc, "长轴", r_scan
        )
        short_m, short_lo, short_up = _period_curves_sigma(
            T, Ms, rc, "短轴", r_scan
        )

        # ===== 第一排：云图 =====
        pts = calc_field_ellipse(long_m, short_m, r_scan, angle)
        lons, lats, grid = interpolate_to_ll(pts, epi_x, epi_y, epsg, extent)

        ax = axs[0, i]
        cf = ax.contourf(
            lons,
            lats,
            grid,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        cb = fig.colorbar(cf, ax=ax, ticks=levels, pad=0.03, shrink=0.8)
        if T == -2:
            tick_fmt = fmt_pgv
        elif T in (-1, 0):
            tick_fmt = lambda v: f"{v:g}"
        else:
            tick_fmt = fmt_psa  # PSA：1 位小数，整数显示整数
        cb.ax.set_yticklabels([tick_fmt(x) for x in levels])
        cb.set_label(f"{label} ({unit})", fontsize=8)

        ax.plot(lon, lat, "k*", markersize=9, zorder=10)
        sr = math.radians(strike)
        arr_lon, arr_lat = km_to_lonlat(
            lon, lat, extent * math.sin(sr), extent * math.cos(sr), utm_zone
        )
        ax.annotate(
            "",
            xy=(arr_lon, arr_lat),
            xytext=(lon, lat),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

        dlat = extent * 1.05 / 111.32
        dlon = extent * 1.05 / (111.32 * math.cos(math.radians(lat)))
        ax.set_xlim(lon - dlon, lon + dlon)
        ax.set_ylim(lat - dlat, lat + dlat)
        ax.set_aspect(1.0 / math.cos(math.radians(lat)))
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"{label} ({unit})", fontsize=9)

        # ===== 第二排：衰减曲线（长轴/短轴 ±1σ）=====
        ax = axs[1, i]
        ax.plot(r_scan, long_m, color="tab:red", lw=1.4, label="长轴")
        ax.plot(
            r_scan, short_m, color="tab:blue", lw=1.4, ls="--", label="短轴"
        )
        ax.fill_between(
            r_scan,
            long_lo,
            long_up,
            color="tab:red",
            alpha=0.15,
            label="长轴±1σ",
        )
        ax.fill_between(
            r_scan,
            short_lo,
            short_up,
            color="tab:blue",
            alpha=0.15,
            label="短轴±1σ",
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1, extent)
        ax.set_xlabel("R (km)", fontsize=8)
        ax.set_ylabel(label + (f" ({unit})" if unit else ""), fontsize=8)
        ax.set_title(f"{label} 衰减曲线", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=6)

    fig.suptitle(
        f"CEA2019 Ground Motion Fields — {rc}区, "
        f"Ms = {Ms:g}, Strike = {strike:g} deg",
        fontsize=11,
    )
    fig.tight_layout()

    if output_file is None:
        os.makedirs("Test_output", exist_ok=True)
        output_file = f"./Test_output/CEA2019_fields_{rc}区_Ms{Ms:g}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存：{os.path.abspath(output_file)}")
    return fig, axs


def plot_response_spectra_single(
    regions, Ms, R, axis="长轴", periods=(0.01, 6.0), n=300, output_file=None
):
    """
    画反应谱：多个分区在同一震中距 R、震级 Ms 下的 PSA(T) 曲线对比。

    参数：
        regions   分区列表，如 ["青藏区", "新疆区", "东部区", "中部区"]
        Ms        面波震级
        R         震中距（km）
        axis      轴向："长轴" / "短轴"
        periods   周期范围 (Tmin, Tmax)，默认 (0.01, 6.0)
        n         周期采样点数（log 等距，默认 300）
        output_file  图片文件名；不填自动命名

    说明：
        - 周期 0~0.04s 按 PGA(T=0) 与 PSA(0.04s) 线性插值；
        - T > 6s 不外插，最多画到 6s；
        - 纵轴 PSA 单位 cm/s²，横轴周期 log 轴。

    返回：
        fig, ax
    """
    rc_all = [_region_core(r) for r in regions]
    t_min, t_max = periods
    Ts = np.logspace(np.log10(t_min), np.log10(t_max), n)

    cm2in = 1.0 / 2.54

    fig, ax = plt.subplots(figsize=(12 * cm2in, 10 * cm2in))

    for rc in rc_all:
        vals = [_period_value(T, Ms, R, rc, axis)[0] for T in Ts]
        ax.plot(Ts, vals, lw=1.6, label=f"{rc}区")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Period T (s)", fontsize=9)
    ax.set_ylabel("PSA (cm/s²)", fontsize=9)
    ax.set_title(
        f"CEA2019 Response Spectra — {axis}，Ms = {Ms:g}，" f"R = {R:g} km",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if output_file is None:
        os.makedirs("Test_output", exist_ok=True)
        output_file = f"./Test_output/CEA2019_RS_{axis}_Ms{Ms:g}_R{R:g}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存：{os.path.abspath(output_file)}")
    return fig, ax


def plot_response_spectra(
    regions,
    Ms,
    R,
    axes=("长轴", "短轴"),
    periods=(0.01, 6.0),
    n=300,
    output_file=None,
):
    """
    画反应谱：多个分区在同一震中距 R、震级 Ms 下的 PSA(T) 曲线对比。
    1×2 子图：左 = 长轴，右 = 短轴。

    参数：
        regions   分区列表，如 ["青藏区", "新疆区", "东部区", "中部区"]
        Ms        面波震级
        R         震中距（km）
        axes      要画的轴向，默认 ("长轴", "短轴")，每个轴向一个子图；
                  只画一个轴向时传 "长轴" 或 ["长轴"] 即可
        periods   周期范围 (Tmin, Tmax)，默认 (0.01, 6.0)
        n         周期采样点数（log 等距，默认 300）
        output_file  图片文件名；不填自动命名

    说明：
        - 周期 0~0.04s 按 PGA(T=0) 与 PSA(0.04s) 线性插值；
        - T > 6s 不外插，最多画到 6s；
        - 纵轴 PSA 单位 cm/s²，横轴周期 log 轴。

    返回：
        fig, axs  （axs 是一维数组，长度 = len(axes)）
    """
    cm2in = 1.0 / 2.54
    if isinstance(axes, str):
        axes = (axes,)

    rc_all = [_region_core(r) for r in regions]
    t_min, t_max = periods
    Ts = np.logspace(np.log10(t_min), np.log10(t_max), n)

    fig, axs = plt.subplots(
        1, len(axes), figsize=(12 * len(axes) * cm2in, 10 * cm2in), sharey=True
    )
    if len(axes) == 1:
        axs = [axs]

    for ax, axis in zip(axs, axes):
        for rc in rc_all:
            vals = [_period_value(T, Ms, R, rc, axis)[0] for T in Ts]
            ax.plot(Ts, vals, lw=1.6, label=f"{rc}区")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(t_min, t_max)
        ax.set_xlabel("Period T (s)", fontsize=9)

        ax.set_title(
            f"CEA2019 Response Spectra — {axis}，Ms = {Ms:g}，R = {R:g} km",
            fontsize=11,
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axs[0].set_ylabel("PSA (cm/s²)", fontsize=9)

    fig.tight_layout()

    if output_file is None:
        os.makedirs("Test_output", exist_ok=True)
        output_file = f"./Test_output/CEA2019_RS_Ms{Ms:g}_R{R:g}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存：{os.path.abspath(output_file)}")
    return fig, axs


def me():
    pass


if __name__ == "__main__":
    # ---- 常用周期点 -1(PGA) -2(PGV) 0.3 1 3 6 ============
    periods = [-1, -2, 0.3, 1, 3, 6]
    sta_lon = [103.0, 103.5, 102.5]  # 台站点
    sta_lat = [25.0, 25.2, 25.1]
    R_max = 400  # 范围 km

    # ---- ① 台站预测
    res = predict_period_values(
        lon=103,
        lat=25,
        strike=90,
        region="新疆区",
        Ms=7.5,
        periods=periods,
        sta_lon=sta_lon,
        sta_lat=sta_lat,
    )
    for k, v in res.items():
        print(f"{k:8s}", np.round(v, 2))

    # ---- ② 综合表 TXT 导出
    export_period_table(
        lon=103,
        lat=25,
        strike=90,
        region="新疆区",
        Ms=7.5,
        periods=periods,
        sta_lon=sta_lon,
        sta_lat=sta_lat,
    )

    # ---- ③ 2×N 整合图（参数云图 + 衰减曲线）
    plot_period_fields(
        lon=103,
        lat=25,
        strike=90,
        region="新疆区",
        Ms=7.5,
        periods=periods,
        extent=R_max,
    )

    # ---- ④ 反应谱：4 分区、Ms=7.5、R=30 km、周期 0.01~6s
    # 长短轴总图
    plot_response_spectra(
        regions=["青藏区", "新疆区", "东部区", "中部区"],
        Ms=7.5,
        R=10,
        axes=["长轴", "短轴"],
        periods=(0.01, 6.0),
    )

    # 单个出图
    plot_response_spectra_single(
        regions=["青藏区", "新疆区", "东部区", "中部区"],
        Ms=7.5,
        R=10,
        axis="长轴",
        periods=(0.01, 6.0),
    )
