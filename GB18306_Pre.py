# -*- coding: utf-8 -*-
"""
GB18306-2015 烈度圈 + PGA/PGV 衰减场 应用整合包（v2）
======================================================
一个文件搞定 GB18306-2015 的三类输出：
    ① 烈度圈 / 烈度云图（GB18306 衰减关系直接反算）
    ② PGA / PGV 衰减云图 + 台站预测
    ③ 仪器烈度（由 PGA、PGV 按 GB/T 17742-2020 换算）

依赖文件（放在同一文件夹里）：
    GB18306_class.py   两个衰减关系 class（烈度 + PGA/PGV）
    GB17742_class.py   仪器烈度换算（PGA/PGV → 烈度）

整体数据流：
    ① 长轴衰减曲线：1~extent km 每隔 1 km 预测值（烈度 / PGA / PGV）
    ② 短轴反算：对每个长轴值，在短轴曲线上反查"同值距离"作短轴距
    ③ 椭圆环：每个 (长轴距, 短轴距) 画一个椭圆，椭圆上值处处相同
    ④ 云图：extent 个椭圆环 → 点云 → griddata 插值 → 经纬度云图
    ⑤ 仪器烈度：把 PGA、PGV 云图（或台站预测值）按 GB17742 公式换算

两套烈度（容易混淆，先分清）：
    Intensity（GB18306）：由烈度衰减关系 I = A + B*Ms + C*lg(R+R0)
                          反算震中距得到的烈度圈 / 烈度云图；
    Intensity_GB17742   ：由 PGA、PGV 按 GB/T 17742-2020 仪器烈度
                          公式（I_A、I_V 分量组合）换算得到。

整合图（plot_gb18306_all，2×4 子图）：
    第一排（云图，都带 colorbar）：
        烈度云图 | PGA 云图 | PGV 云图 | 仪器烈度云图(GB17742)
    第二排（衰减曲线，横轴距离 1~extent km，log 轴）：
        烈度(1~11) | PGA(log) | PGV(log) | 仪器烈度(1~11)
        每条画长轴、短轴两条中值曲线 + ±1σ 范围带

公共约定：
    - 区域：东部区 / 中部区 / 新疆区 / 青藏区；轴向：长轴 / 短轴
    - 走向 strike：正北=0°，顺时针，东=90°，西=270°，0~360°
    - 经纬度 WGS84，UTM 投影（南北半球自动选 EPSG:326xx/327xx）
    - 单位：aE（gal）、vE（cm/s）、距离（km）、烈度（无量纲）

色标：
    - USGS MMI 十色（I 白 → X+ 深红），所有烈度/云图共用
    - PGA 分界：[1,2,5,10,25,50,100,200,400,800,1500] gal
    - PGV 分界 = PGA 分界 ÷ 10

自动保护：
    - 烈度圈不存在（长短轴 < 1 km）→ 跳过；短轴>长轴 → 按长轴画圆
    - PGA/PGV 场：长短轴曲线相交退化 → 交点后以长轴为准画圆形场

常用入口（4 个主函数）：
    plot_intensity_ellipses(...)   烈度圈图（单图）
    plot_pga_pgv_fields(...)       PGA/PGV 衰减云图（双图）
    plot_gb18306_all(...)          整合图（2×4）
    export_all_table(...)          综合表 TXT（PGA/PGV + 两套烈度）

直接运行本文件执行"演示"；也可以在别的程序里 import 后调用各函数。
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
from matplotlib.ticker import FuncFormatter

# scipy 用于 griddata 插值；没装时云图无法生成（会给出警告）
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

# 精确的经纬度换算工具（UTM 投影）；没装时自动退化为简单近似
try:
    from pyproj import Transformer

    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

# 找到 GB18306_class.py（和本文件在同一个文件夹里），导入两个衰减 class
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from GB18306_class import (
    GB18306_2015_IntensityCal,
    GB18306_2015_PGA_PGV_GMMs,
)

# GB/T 17742-2020 仪器烈度换算（PGA/PGV → 烈度）
from GB17742_class import GB17742_2020_Cal_instrument_intensity

# 全局唯一的两个计算器（烈度 + PGA/PGV）
CALC_INTENSITY = GB18306_2015_IntensityCal()
CALC_GMM = GB18306_2015_PGA_PGV_GMMs()

# 字体：英文用 Times New Roman，中文自动回退 SimHei（标题里的分区名）
plt.rcParams["font.family"] = [
    "Times New Roman",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]
plt.rcParams["font.size"] = 8.5
plt.rcParams["axes.unicode_minus"] = False

# USGS MMI 烈度色标（I 白 → X+ 深红）
USGS_MMI_COLORS = [
    "#FFFFFF",  # I
    "#BFCCFF",  # II
    "#A0E6FF",  # III
    "#80FFFF",  # IV
    "#7AFF93",  # V
    "#FFFF00",  # VI
    "#FFC800",  # VII
    "#FF9100",  # VIII
    "#FF0000",  # IX
    "#800000",  # X+
]

# 罗马数字表：I~IX 对应 0.5~9.5；10 度及以上统一显示 X+
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]

# ---------- 色标与标签约定 ----------
# PGA 分界（gal）：其中 50~100 对应 USGS MMI 6 度区颜色（黄色）
PGA_LEVELS = [1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1500]
# PGV 分界（cm/s）= PGA 分界 ÷ 10
PGV_LEVELS = [0.1, 0.2, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 150]


def to_roman(intensity):
    """阿拉伯数字转罗马数字：4.5→V，8.5→IX；10 度及以上统一显示 X+"""
    idx = int(intensity + 0.5)
    if idx >= 10:
        return "X+"
    idx = max(1, min(9, idx))
    return ROMAN[idx - 1]


def intensity_color(intensity):
    """把烈度换成 USGS MMI 色标的颜色（I 白 → X+ 深红）"""
    idx = int(intensity + 0.5)
    idx = max(1, min(10, idx))
    return USGS_MMI_COLORS[idx - 1]


def intensity_color0(intensity):
    """备用色标：Reds_r（低烈度取浅色端，10 度及以上取最深红端）"""
    idx = int(intensity + 0.5)
    idx = max(1, min(10, idx))
    return plt.cm.Reds_r(1.0 - (idx - 1) / 9.0)


def km_to_lonlat(lon, lat, east_km, north_km, utm_zone):
    """把相对震中的 东向(km)/北向(km) 换算成经纬度（UTM 投影）。
    带号只由经度决定；EPSG 前缀按纬度选择：北半球 326xx，南半球 327xx。"""
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
    # 没有 pyproj 时的简单近似（南北半球通用）
    dlat = np.asarray(north_km) / 111.32
    dlon = np.asarray(east_km) / (111.32 * math.cos(math.radians(lat)))
    return lon + dlon, lat + dlat


def lonlat_to_utm(lon, lat, utm_zone):
    """经纬度 -> UTM 坐标（米）"""
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    return fwd.transform(lon, lat)


def fmt_pgv(value):
    """PGV 标签：整数显示为整数，小数保留 1 位；
    很小的分界值保留有效数字以便区分（如 0.025 不能被抹成 0.0）。"""
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


# ==================== 烈度应用 ====================


def _ellipse_radii(I, region, Ms):
    """向量化：烈度数组 I → (长轴距 a, 短轴距 b)，单位 km。

    对应 GB18306_class 的 invert_R 中值：
        a = 10**((I - A1 - B1*Ms) / C1) - R01
        b = 10**((I - A2 - B2*Ms) / C2) - R02
    短轴大于长轴时按长轴画圆（b = min(b, a)），与绘图规则一致。
    """
    A1, B1, C1, R01, _ = CALC_INTENSITY._PARAMS[(region, "长轴")]
    A2, B2, C2, R02, _ = CALC_INTENSITY._PARAMS[(region, "短轴")]
    a = 10.0 ** ((np.asarray(I) - A1 - B1 * Ms) / C1) - R01
    b = 10.0 ** ((np.asarray(I) - A2 - B2 * Ms) / C2) - R02
    b = np.minimum(b, a)  # 退化保护：短轴>长轴 → 圆
    a = np.maximum(a, 1e-9)  # 避免除零
    b = np.maximum(b, 1e-9)
    return a, b


def _solve_intensity(R, theta, region, Ms, extent=400):
    """
    批量求台站烈度及其所在椭圆的长/短轴距（向量化二分）。

    思路：烈度圈是椭圆族 (a(I), b(I))，I 越高圈越小。
    台站（极坐标 R、相对长轴夹角 theta）落在某圈上的条件是：
        (R·cosθ / a)² + (R·sinθ / b)² = 1
    左边随 I 单调递增，因此对 I 二分即可解出台站烈度。

    返回：
        (I, a_eq, b_eq)
        I     台站烈度（阿拉伯数字，可含小数；场外 NaN）
        a_eq  台站所在烈度圈的长轴距（km）
        b_eq  台站所在烈度圈的短轴距（km）
    """
    R = np.asarray(R, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # 烈度范围：最外圈（沿长轴 R=extent）~ 震中（R→0）
    A1, B1, C1, R01, _ = CALC_INTENSITY._PARAMS[(region, "长轴")]
    I_min = A1 + B1 * Ms + C1 * math.log10(extent + R01)
    I_max = A1 + B1 * Ms + C1 * math.log10(R01)

    # 最外圈判定：台站在该方向上的最外圈半径以内才可解
    a0, b0 = _ellipse_radii(I_min, region, Ms)
    ct = np.cos(theta)
    st = np.sin(theta)
    r_outer = a0 * b0 / np.sqrt((b0 * ct) ** 2 + (a0 * st) ** 2)
    inside = np.isfinite(r_outer) & (R <= r_outer)

    I_lo = np.full_like(R, I_min)
    I_hi = np.full_like(R, I_max)
    for _ in range(60):
        I_mid = (I_lo + I_hi) / 2.0
        a, b = _ellipse_radii(I_mid, region, Ms)
        f = (R * ct / a) ** 2 + (R * st / b) ** 2 - 1.0
        I_lo = np.where(f < 0.0, I_mid, I_lo)  # 台站在圈内 → 烈度更大
        I_hi = np.where(f >= 0.0, I_mid, I_hi)  # 台站在圈外 → 烈度更小
    I = (I_lo + I_hi) / 2.0
    a_eq, b_eq = _ellipse_radii(I, region, Ms)

    mask = inside
    return (
        np.where(mask, I, np.nan),
        np.where(mask, a_eq, np.nan),
        np.where(mask, b_eq, np.nan),
    )


def _intensity_ellipse_data(intensities, region, Ms):
    """
    反算每个烈度圈的长轴/短轴半径（含自动保护），供单图和整合图共用。

    对每个烈度 I：
        长轴距 = invert_R(I, Ms, region, "长轴")["mean"]
        短轴距 = invert_R(I, Ms, region, "短轴")["mean"]
    自动保护：
        - 短轴 > 长轴 → 按长轴画圆（短轴取长轴值）；
        - 长短轴 < 1 km → 该烈度圈不存在，跳过；
        - 全部不存在 → 报错。

    返回：
        [(intensity, major, minor), ...]（按烈度从小到大排序）
    """
    ellipse_data = []
    for intensity in sorted(intensities):
        # invert_R 返回 mean/std/±1σ，这里只取中值 mean
        major = CALC_INTENSITY.invert_R(intensity, Ms, region, "长轴")["mean"]
        minor = CALC_INTENSITY.invert_R(intensity, Ms, region, "短轴")["mean"]
        if minor > major:
            # 自动保护：短轴大于长轴时，统一按长轴绘制圆形
            print(
                f"  提示：烈度 {intensity:g}（{to_roman(intensity)} 度区）"
                f"短轴({minor:.1f} km)大于长轴({major:.1f} km)，"
                f"统一按长轴 {major:.1f} km 绘制圆形"
            )
            minor = major
        if major < 1 or minor < 1:
            # 该烈度圈在当前 Ms 下不存在（缩成点），跳过不画
            print(
                f"  提示：烈度 {intensity:g}（{to_roman(intensity)} 度区）"
                f"超出 Ms={Ms:g} 在 {region} 能达到的最大烈度，跳过不画"
            )
            continue
        ellipse_data.append((intensity, major, minor))
        print(
            f"烈度 {intensity:g}（{to_roman(intensity)} 度区）："
            f"长轴 {major:7.1f} km，短轴 {minor:7.1f} km"
        )
    # 所有烈度区都不存在时，无法画图
    if not ellipse_data:
        raise ValueError(
            "所有烈度区都超出该震级能达到的最大烈度，" "请减小烈度或增大 Ms！"
        )
    return ellipse_data


def _ellipse_lonlat(lon, lat, utm_zone, strike, major, minor, n=361):
    """椭圆公式 + 走向旋转 + 经纬度换算，返回 (经度数组, 纬度数组)。
    供烈度圈单图 / 整合图共用。"""
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    ell_x = major * np.cos(theta)  # 长轴方向
    ell_y = minor * np.sin(theta)  # 短轴方向
    # 走向(自北顺时针) → UTM 旋转角(自东逆时针)
    rot = math.radians(90.0 - strike)
    east = ell_x * math.cos(rot) - ell_y * math.sin(rot)
    north = ell_x * math.sin(rot) + ell_y * math.cos(rot)
    return km_to_lonlat(lon, lat, east, north, utm_zone)


def plot_intensity_ellipses(
    lon, lat, strike, region, intensities, Ms=6.8, output_file=None
):
    """
    画 GB18306-2015 烈度圈（一组同心椭圆，一个烈度一个圈）

    步骤：
        ① 对每个烈度 I，用 CALC_INTENSITY.invert_R 沿"长轴"/"短轴"反算
           震中距（取 mean），得到该烈度圈的长轴、短轴半径；
        ② 自动保护：短轴 > 长轴 → 按长轴画圆；长短轴 < 1 km → 跳过；
        ③ 椭圆公式画点（长轴先朝正东），按走向旋转，再转经纬度；
        ④ 从外圈到内圈逐层 fill 填色（色带 = 各烈度区），
           椭圆边界画黑色虚线，线上标注罗马数字；
        ⑤ 加震中星标、走向箭头，自动定图幅范围。

    参数：
        lon          震中经度（东经为正，度）
        lat          震中纬度（北纬为正，度）
        strike       长轴方位角：正北=0°，顺时针，东=90°，西=270°，0~360°
        region       分区："青藏区" / "新疆区" / "东部区" / "中部区"
        intensities  烈度列表（阿拉伯数字），如 [4.5, 5.5, 6.5, 7.5, 8.5]；
                     图上自动转成罗马数字（V、VI、…、IX；10 度及以上为 X+）
        Ms           面波震级（GB18306 公式里的 M）
        output_file  图片文件名；不填则自动命名

    返回：
        fig, ax  （matplotlib 图形对象，想继续改图或另存都可以用）
    """
    # ---------- 检查参数 ----------
    CALC_INTENSITY._validate_input(region, "长轴")  # 分区写错会在这里报错
    if not intensities:
        raise ValueError("intensities 不能为空！")
    strike = strike % 360.0  # 走向归一到 0~360°
    intensities = sorted(intensities)  # 从小到大：先画外面的大圈

    utm_zone = int((lon + 180.0) // 6.0) + 1  # UTM 分带号（仅由经度决定）

    # ---------- 第 1 步：反算每个烈度的长轴、短轴半径（GB18306 公式）----------
    ellipse_data = _intensity_ellipse_data(intensities, region, Ms)

    # ---------- 画图 ----------
    fig, ax = plt.subplots(figsize=(9, 8))

    for intensity, major, minor in ellipse_data:
        # 第 2~3 步：椭圆公式 + 走向旋转 + 经纬度换算（共用函数）
        ell_lon, ell_lat = _ellipse_lonlat(
            lon, lat, utm_zone, strike, major, minor
        )
        color = intensity_color(intensity)
        # 填充当前椭圆（从小到大逐层向内盖，外圈色带自然露出）
        ax.fill(
            ell_lon,
            ell_lat,
            color=color,
            alpha=0.95,
            linewidth=0,
            zorder=2,
            label=f"Intensity {to_roman(intensity)} "
            f"({major:.0f}x{minor:.0f} km)",
        )
        # 椭圆边界线（黑色虚线）
        ax.plot(ell_lon, ell_lat, color="k", linewidth=1, linestyle="--")
        # 在椭圆线上标注罗马数字（放在 135° 方向，避开走向箭头）
        label_idx = int(round(135.0 / 360.0 * 360))
        ax.text(
            ell_lon[label_idx],
            ell_lat[label_idx],
            to_roman(intensity),
            fontsize=10,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.15", fc="white", ec="k", lw=0.7, alpha=1
            ),
            zorder=11,
        )

    # 震中星标
    ax.plot(
        lon,
        lat,
        "k*",
        markersize=16,
        zorder=10,
        label=f"Epicenter ({lon:.3f}E, {lat:.3f}N)",
    )

    # 走向指示线：从震中指向长轴方向，方便核对旋转方向对不对
    max_major = max(row[1] for row in ellipse_data)
    strike_rad = math.radians(strike)
    arr_east = max_major * math.sin(strike_rad)
    arr_north = max_major * math.cos(strike_rad)
    arr_lon, arr_lat = km_to_lonlat(lon, lat, arr_east, arr_north, utm_zone)
    ax.annotate(
        "",
        xy=(arr_lon, arr_lat),
        xytext=(lon, lat),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
    )
    ax.text(
        arr_lon, arr_lat, f"Strike {strike:g} deg", fontsize=9, color="black"
    )

    # 图的范围按最大的圈（最小的烈度）自动调整，留 15% 边距
    dlat = max_major * 1.15 / 111.32
    dlon = max_major * 1.15 / (111.32 * math.cos(math.radians(lat)))
    ax.set_xlim(lon - dlon, lon + dlon)
    ax.set_ylim(lat - dlat, lat + dlat)

    # aspect = 1/cos(纬度)：让经纬度图上 1° 经度 ≈ 1° 纬度，形状不变形
    ax.set_aspect(1.0 / math.cos(math.radians(lat)))
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(
        f"GB18306-2015 Intensity Ellipses — {region}, "
        f"Ms = {Ms:g}, Strike = {strike:g} deg (from North)"
    )
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if output_file is None:
        output_file = f"intensity_ellipses_{region}_Ms{Ms:g}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存：{os.path.abspath(output_file)}")
    return fig, ax


def export_intensity_table(
    lon,
    lat,
    strike,
    region,
    Ms,
    sta_lon,
    sta_lat,
    sta_id=None,
    extent=400,
    output_file=None,
):
    """
    生成台站烈度预测表并导出为 TXT（pandas DataFrame，tab 分隔）

    表列：
        Sta_ID  Sta_longi  Sta_lati
        Repi(km)          震中距
        Repi_long(km)     台站所在烈度圈的长轴距
        Repi_short(km)    台站所在烈度圈的短轴距
        Intensity         台站烈度（阿拉伯数字，含小数）
        Intensity_roman   台站烈度（罗马数字显示）

    参数：同 plot_intensity_ellipses；
        sta_id      台站编号（可选，默认 S1、S2…；也可传自己的编号列表）
        extent      反解范围半径（km），默认 400
        output_file 输出 txt 文件名；不填自动命名

    返回：
        pandas.DataFrame（同时已保存为 txt）
    """
    CALC_INTENSITY._validate_input(region, "长轴")
    strike = strike % 360.0
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    sta_lon = np.asarray(sta_lon, dtype=float)
    sta_lat = np.asarray(sta_lat, dtype=float)
    if sta_lon.shape != sta_lat.shape:
        raise ValueError("sta_lon 与 sta_lat 形状必须一致！")

    # 批量投影 → 震中距 R、相对长轴的夹角 theta
    fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    sx, sy = np.asarray(fwd.transform(sta_lon.ravel(), sta_lat.ravel()))
    dx = (sx - epi_x) / 1000.0
    dy = (sy - epi_y) / 1000.0
    R = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) - math.radians(90.0 - strike)

    # 求解台站烈度 + 所在烈度圈的长/短轴距
    I, a_eq, b_eq = _solve_intensity(R, theta, region, Ms, extent)

    if sta_id is None:
        sta_id = [f"S{i + 1}" for i in range(R.size)]
    else:
        sta_id = np.asarray(sta_id).ravel()

    df = pd.DataFrame(
        {
            "Sta_ID": sta_id,
            "Sta_longi": np.round(sta_lon.ravel(), 4),
            "Sta_lati": np.round(sta_lat.ravel(), 4),
            "Repi(km)": np.round(R, 2),
            "Repi_long(km)": np.round(a_eq, 2),  # 台站所在烈度圈长轴距
            "Repi_short(km)": np.round(b_eq, 2),  # 台站所在烈度圈短轴距
            "Intensity": np.round(I, 2),
            "Intensity_roman": [
                to_roman(v) if not math.isnan(v) else "NaN" for v in I
            ],
        }
    )

    if output_file is None:
        output_file = f"GB18306_pre_intensity_table_{region}_Ms{Ms:g}.txt"
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8-sig")
    print(f"已导出：{os.path.abspath(output_file)}")
    return df


# ==================== PGA/PGV 应用 ====================


def calc_axis_curves(M, region, r_scan):
    """
    沿长轴、短轴计算 PGA / PGV 衰减曲线（取中值）。

    这是对底层包 GB18306_class.py 的具体调用：对 r_scan 中每个
    距离 r，分别用 "长轴" 和 "短轴" 参数调 calculate(M, r, region, axis)，
    得到 4 条曲线：
        pga_long / pga_short : 沿长轴 / 短轴的 PGA（gal）
        pgv_long / pgv_short : 沿长轴 / 短轴的 PGV（cm/s）

    注意：calculate() 返回 ((aE中值,下限,上限), (vE中值,下限,上限))，
    这里只取中值，即下标 [0]。
    """
    pga_long, pga_short = [], []
    pgv_long, pgv_short = [], []
    for r in r_scan:
        aE, vE = CALC_GMM.calculate(M, r, region, "长轴")
        pga_long.append(aE[0])
        pgv_long.append(vE[0])
        aE, vE = CALC_GMM.calculate(M, r, region, "短轴")
        pga_short.append(aE[0])
        pgv_short.append(vE[0])
    return (
        np.asarray(pga_long),
        np.asarray(pga_short),
        np.asarray(pgv_long),
        np.asarray(pgv_short),
    )


def _gmm_ellipse_radii(Y, Ms, region, param_type):
    """
    PGA（param_type="aE"）或 PGV（param_type="vE"）解析反算"所在椭圆"：
        a = 10**((lgY - A - B*Ms) / C) - D*exp(E*Ms)
    与 GB18306_class 的 invert_R 中值公式完全一致
    （含 0.01 km 最小截断；短轴不超过长轴）。
    返回 (a, b)，单位 km。
    """
    Y = np.asarray(Y, dtype=float)
    p_l = CALC_GMM._get_params(Ms, region, "长轴", param_type)
    p_s = CALC_GMM._get_params(Ms, region, "短轴", param_type)
    lg = np.log10(Y)
    a = 10.0 ** ((lg - p_l["A"] - p_l["B"] * Ms) / p_l["C"]) - p_l[
        "D"
    ] * np.exp(p_l["E"] * Ms)
    b = 10.0 ** ((lg - p_s["A"] - p_s["B"] * Ms) / p_s["C"]) - p_s[
        "D"
    ] * np.exp(p_s["E"] * Ms)
    b = np.minimum(b, a)  # 退化保护：短轴不超过长轴
    a = np.maximum(a, 0.01)  # 与 invert_R 一致的最小截断
    b = np.maximum(b, 0.01)
    return a, b


def calc_field_ellipse(long_arr, short_arr, r_scan, angle=0.0):
    """
    由长短轴衰减曲线生成椭圆场点云 (x_km, y_km, val)。

    核心思路（用户约定）：
        ① 长轴曲线在 r_scan 每个距离 t 上给一个值 V(t) = long(t)；
        ② 短轴距 b(t) 由 minor_axis_curve 反算（短轴曲线上值 = V(t) 的距离）；
        ③ 每个 t 画一个椭圆：长半轴 = t，短半轴 = b(t)，椭圆上值 = V(t)；
        ④ 每个椭圆环取 120 个点（隔点取 60 个），组成点云。

    x / y 是相对震中的公里坐标，已按走向 angle（从 UTM 东向逆时针）旋转；
    val 是该环的 PGA（或 PGV），同一环上处处相同。
    """
    b_t = minor_axis_curve(long_arr, short_arr, r_scan)
    pts = []
    for i, r_long in enumerate(r_scan):
        r_s = b_t[i]
        if not np.isfinite(r_s):
            continue
        val = long_arr[i]
        theta = np.linspace(0.0, 2.0 * math.pi, 120)
        x_e = r_long * np.cos(theta)  # 长轴方向（未旋转）
        y_e = r_s * np.sin(theta)  # 短轴方向（未旋转）
        x = x_e * math.cos(angle) - y_e * math.sin(angle)
        y = x_e * math.sin(angle) + y_e * math.cos(angle)
        for j in range(0, len(theta), 2):
            pts.append((x[j], y[j], val))
    return np.array(pts)


def interpolate_to_ll(pts_km, epi_x, epi_y, epsg, extent=400, n=400):
    """
    把椭圆环点云插值成规则网格场，再投影成经纬度。

    步骤：
        ① 在 ±extent km 的 n×n 网格上，用 griddata(linear) 由点云插值；
        ② 掩膜：震中 0.5 km 内（点云没有极近场点）和 extent 圆外不显示；
        ③ 网格 UTM 坐标（米）→ WGS84 经纬度，供 contourf / contour 使用。
    """
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
        print("警告：未安装 scipy，无法插值生成场图！")

    dist = np.sqrt(xg**2 + yg**2)
    grid = np.ma.masked_where(dist < 0.5, grid)
    grid = np.ma.masked_where(dist > extent, grid)

    xg_utm = epi_x + xg * 1000.0
    yg_utm = epi_y + yg * 1000.0
    inv = Transformer.from_crs(epsg, "epsg:4326", always_xy=True)
    lons, lats = inv.transform(xg_utm, yg_utm)
    return lons, lats, grid


def minor_axis_curve(long_arr, short_arr, r_scan):
    """
    反算每个长轴距离 t 的短轴距离 b(t)（单位 km）。

    做法：值 V = long(t)，在短轴曲线上反查"短轴距离 b 使短轴值 = V"。
    因为 short_arr 是"距离→值"，倒序后变成"值递增→距离"，正好给 np.interp
    用（np.interp 要求 xp 单调递增）。

    退化保护：若 b 反算不出来（超出短轴曲线范围），或 b >= t
    （长短轴曲线相交后"短轴>长轴"），按约定 b = t，即该圈画成圆。
    """
    b = np.interp(long_arr, short_arr[::-1], r_scan[::-1])
    b = np.where(long_arr >= short_arr[0], r_scan[0], b)  # 近场极值
    b = np.where(long_arr <= short_arr[-1], np.nan, b)  # 超出短轴范围
    b = np.where(~np.isfinite(b) | (b > r_scan), r_scan, b)  # 退化→圆
    return b


def field_domain(long_arr, short_arr, r_scan):
    """
    场的最外圈范围（含退化保护）。

    部分分区（如中部区、青藏区大震级）长轴衰减比短轴更快，长短轴曲线
    会在 r 内相交：交点以前 长轴值 > 短轴值（椭圆正常）；交点以后
    长轴值 < 短轴值（反算短轴距会超过长轴距，椭圆退化）。按约定，
    退化处以长轴为准画圆（b = t），因此场仍覆盖到 extent。

    返回：
        (a_max, b_max, r_cross)
        a_max    最外圈长轴距离（km）= extent
        b_max    最外圈短轴距离（km）；无交点时为椭圆短轴，
                 有交点时为 extent（外圈退化为圆形）
        r_cross  长短轴曲线交点距离（km）；范围内未相交时为 inf
    """
    diff = long_arr - short_arr  # 长轴衰减更快，随距离递减
    below = np.where(diff <= 0.0)[0]
    if below.size:
        i = int(below[0])
        if i == 0:
            r_cross = float(r_scan[0])
        else:
            r_cross = float(
                np.interp(0.0, diff[i - 1 : i + 1], r_scan[i - 1 : i + 1])
            )
    else:
        r_cross = float("inf")

    a_max = float(r_scan[-1])
    b_t = minor_axis_curve(long_arr, short_arr, r_scan)
    b_max = float(np.interp(a_max, r_scan, b_t))
    return a_max, b_max, r_cross


def _solve_values(long_arr, short_arr, r_scan, R, theta):
    """
    批量求台站的地震动值，以及台站所在椭圆的长/短轴距离。

    场上每个圈 t：值 V(t) = long(t)，长轴距 = t，短轴距 = b(t)。
    台站 j 在相对长轴的夹角 theta_j 方向上，圈 t 的极径（椭圆极坐标公式）：
        R(t, θ) = a·b / sqrt(b²·cos²θ + a²·sin²θ)
    对固定 θ，R(t, θ) 随 t 单调递增，因此：
        ① 反插值：找最后一个 R(t) ≤ R 的圈，线性插值得到 t0，V0 = long(t0)；
        ② 4 次向量化牛顿迭代（在 V 空间求解 f(V)=0），结果与二分法一致；
        ③ 由 V 反算台站所在椭圆的长轴距 a_eq、短轴距 b_eq。

    返回：
        (val, a_eq, b_eq)
        val   台站处的地震动值（超出场范围 NaN）
        a_eq  台站所在椭圆的长轴距离（km）
        b_eq  台站所在椭圆的短轴距离（km）；退化处 b_eq == a_eq（圆）
    """
    R = np.asarray(R, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # 最外圈椭圆（field_domain 已含退化保护：交点后外圈为圆）
    a_max, b_max, _ = field_domain(long_arr, short_arr, r_scan)
    if not np.isfinite(b_max):
        return np.full_like(R, np.nan)

    # 台站在各方向上的最外圈半径，超出者无法预测（记 NaN）
    ct = np.cos(theta)
    st = np.sin(theta)
    r_outer = a_max * b_max / np.sqrt((b_max * ct) ** 2 + (a_max * st) ** 2)
    inside = np.isfinite(r_outer) & (R <= r_outer)

    # 每圈长短轴：a(t) = t，b(t) = b(t)
    a_t = np.asarray(r_scan, dtype=float)
    b_t = minor_axis_curve(long_arr, short_arr, r_scan)
    valid = np.isfinite(b_t)
    if valid.sum() < 2:
        return np.full_like(R, np.nan)
    a_v = a_t[valid]
    b_v = b_t[valid]

    # R_ring[j, :]：台站 j 在每个圈方向 theta_j 上的椭圆半径（单调递增）
    R_ring = (a_v[None, :] * b_v[None, :]) / np.sqrt(
        (b_v[None, :] * ct[:, None]) ** 2 + (a_v[None, :] * st[:, None]) ** 2
    )

    # 每个台站反插值：找最后一个 R_ring <= R 的圈索引 k，再线性插值
    k = (R_ring <= R[:, None]).sum(axis=1) - 1
    k = np.clip(k, 0, len(a_v) - 2)
    rows = np.arange(len(R))
    r0 = R_ring[rows, k]
    r1 = R_ring[rows, k + 1]
    frac = np.zeros_like(R)
    denom = r1 - r0
    nz = denom > 0
    frac[nz] = (R[nz] - r0[nz]) / denom[nz]
    t0 = a_v[k] + frac * (a_v[k + 1] - a_v[k])

    # 初值：环表反插得到 V0 = long(t0)
    V = np.interp(t0, a_t, long_arr)

    # 向量化牛顿迭代：在 V 空间求解 f(V) = R(t,θ) - R = 0，
    # 与二分使用同一套插值，因此结果与二分严格一致（4 次即可收敛）
    v_lo = float(np.interp(a_max, r_scan, long_arr))  # 最外圈的值（小端）
    v_hi = float(long_arr[0])  # 震中附近的值（大端）
    long_rev = long_arr[::-1]
    short_rev = short_arr[::-1]
    r_scan_rev = r_scan[::-1]

    def radius_from_v(v):
        a = np.interp(v, long_rev, r_scan_rev)
        b = np.interp(v, short_rev, r_scan_rev)
        b = np.minimum(b, a)  # 退化保护：b 不超过 a（交点后按圆形场）
        return a * b / np.sqrt((b * ct) ** 2 + (a * st) ** 2)

    for _ in range(4):
        f = radius_from_v(V) - R
        eps = np.maximum(np.abs(V) * 1e-5, 1e-9)
        fp = (
            radius_from_v(np.clip(V + eps, v_lo, v_hi))
            - radius_from_v(np.clip(V - eps, v_lo, v_hi))
        ) / (2.0 * eps)
        step = np.zeros_like(V)
        nz = np.abs(fp) > 1e-12
        step[nz] = f[nz] / fp[nz]
        V = np.clip(V - step, v_lo, v_hi)

    # 台站所在椭圆：由解出的 V 反算 (a_eq, b_eq)（b 不超过 a，退化即圆）
    a_eq = np.interp(V, long_arr[::-1], r_scan[::-1])
    b_eq = np.minimum(np.interp(V, short_arr[::-1], r_scan[::-1]), a_eq)
    mask = inside
    return (
        np.where(mask, V, np.nan),
        np.where(mask, a_eq, np.nan),
        np.where(mask, b_eq, np.nan),
    )


def predict_pga_pgv(
    lon, lat, strike, region, Ms, sta_lon, sta_lat, extent=400, verbose=True
):
    """
    预测指定台站（经纬度）的 PGA / PGV 中值

    步骤：
        ① 参数校验、走向归一到 0~360°；
        ② UTM 投影：震中 + 全部台站一次批量投影（比逐站建投影快很多）；
        ③ 台站相对震中的 km 偏移 (dx, dy) → 震中距 R、相对长轴夹角 θ；
        ④ _solve_values 批量解出 PGA、PGV（椭圆几何在此不取）；
        ⑤ 逐站打印结果（verbose=True 时）。

    参数：
        lon, lat   震中经纬度（东经/北纬为正，度）
        strike     长轴方位角：正北=0°，顺时针，东=90°，西=270°，0~360°
        region     分区："青藏区" / "新疆区" / "东部区" / "中部区"
        Ms         面波震级
        sta_lon    台站经度（标量 / 向量 / 矩阵均可）
        sta_lat    台站纬度（与 sta_lon 同形状）
        extent     衰减曲线最大距离（km），默认 400
        verbose    是否逐台站打印结果；台站很多时可设 False 提速

    返回：
        pga_pred, pgv_pred  与输入同形状（单位 gal / cm/s）；
        超出场范围的台站为 NaN。
    """
    CALC_GMM._validate_input(region, "长轴")  # 分区写错会在这里报错
    strike = strike % 360.0  # 走向归一到 0~360°
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    r_scan = np.arange(1.0, extent + 1.0, 1.0)  # 1~extent km，间隔 1 km
    pga_long, pga_short, pgv_long, pgv_short = calc_axis_curves(
        Ms, region, r_scan
    )

    sta_lon = np.asarray(sta_lon, dtype=float)
    sta_lat = np.asarray(sta_lat, dtype=float)
    if sta_lon.shape != sta_lat.shape:
        raise ValueError("sta_lon 与 sta_lat 形状必须一致！")
    shape = sta_lon.shape

    # 一次性批量投影所有台站（比逐站建 Transformer 快很多）
    fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    sx, sy = np.asarray(fwd.transform(sta_lon.ravel(), sta_lat.ravel()))
    dx = (sx - epi_x) / 1000.0
    dy = (sy - epi_y) / 1000.0
    R = np.hypot(dx, dy)  # 震中距（km）
    # θ = 台站相对正东的方位角 - 长轴相对正东的方位角（90°-strike）
    theta = np.arctan2(dy, dx) - math.radians(90.0 - strike)

    # 批量向量化求解（PGA 和 PGV 各自一套衰减曲线）
    pga, _, _ = _solve_values(pga_long, pga_short, r_scan, R, theta)
    pgv, _, _ = _solve_values(pgv_long, pgv_short, r_scan, R, theta)

    if verbose:
        for slon, slat, r, p, v in zip(
            sta_lon.ravel(), sta_lat.ravel(), R, pga, pgv
        ):
            out = f"台站 ({slon:.4f}, {slat:.4f})：R = {r:7.1f} km，"
            if math.isnan(p):
                out += "超出场范围，无法预测"
            else:
                out += f"PGA = {p:8.1f} gal，PGV = {v:6.1f} cm/s"
            print(out)

    return pga.reshape(shape), pgv.reshape(shape)


def export_station_table(
    lon,
    lat,
    strike,
    region,
    Ms,
    sta_lon,
    sta_lat,
    sta_id=None,
    extent=400,
    output_file=None,
):
    """
    生成台站预测表并导出为 TXT（通过 pandas DataFrame，tab 分隔）

    表列（txt 用 tab 分隔，utf-8-sig 编码，Excel 可直接打开）：
        Sta_ID  Sta_longi  Sta_lati
        Repi(km)            震中距
        Repi_long(km)       台站所在椭圆的长轴距（椭圆几何按 PGA 曲线求）
        Repi_short(km)      台站所在椭圆的短轴距
        PGA(gal)  PGV(cm/s) 台站预测值

    参数：同 predict_pga_pgv；
        sta_id        台站编号（可选，默认 S1、S2…；也可传自己的编号列表）
        output_file   输出 txt 文件名；不填自动命名

    返回：
        pandas.DataFrame（同时已保存为 txt）
    """

    CALC_GMM._validate_input(region, "长轴")
    strike = strike % 360.0
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    r_scan = np.arange(1.0, extent + 1.0, 1.0)
    pga_long, pga_short, pgv_long, pgv_short = calc_axis_curves(
        Ms, region, r_scan
    )

    sta_lon = np.asarray(sta_lon, dtype=float)
    sta_lat = np.asarray(sta_lat, dtype=float)
    if sta_lon.shape != sta_lat.shape:
        raise ValueError("sta_lon 与 sta_lat 形状必须一致！")

    # 批量投影 → 震中距 R 与相对长轴夹角 θ
    fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    sx, sy = np.asarray(fwd.transform(sta_lon.ravel(), sta_lat.ravel()))
    dx = (sx - epi_x) / 1000.0
    dy = (sy - epi_y) / 1000.0
    R = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) - math.radians(90.0 - strike)

    # 椭圆几何（长轴距/短轴距）按 PGA 曲线求；PGA、PGV 各自取值
    pga, a_eq, b_eq = _solve_values(pga_long, pga_short, r_scan, R, theta)
    pgv, _, _ = _solve_values(pgv_long, pgv_short, r_scan, R, theta)

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
            "Repi_long(km)": np.round(a_eq, 2),  # 台站所在椭圆长轴距
            "Repi_short(km)": np.round(b_eq, 2),  # 台站所在椭圆短轴距
            "PGA(gal)": np.round(pga, 2),
            "PGV(cm/s)": np.round(pgv, 2),
        }
    )

    if output_file is None:
        output_file = f"GB18306_pre_station_table_{region}_Ms{Ms:g}.txt"
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8-sig")
    print(f"已导出：{os.path.abspath(output_file)}")
    return df


def random_stations(lon, lat, strike, region, Ms, extent=400, n=8, seed=42):
    """
    在 1~extent km 场范围内随机生成 n 个台站（保证落在最外圈椭圆内）

    做法：
        ① 先由 field_domain 求最外圈 (a_max, b_max)（含退化保护）；
        ② 均匀采样"震中距 R ∈ [1, a_max] + 方位角 β"，落在椭圆外
           （R > 该方向最外圈半径）的点拒绝重采；
        ③ 批量向量化采样 + 一次性 UTM 投影，速度很快。

    参数：同 predict_pga_pgv；
        n      台站数量
        seed   随机种子（固定可复现）
    返回：
        (sta_lon, sta_lat) 两个 numpy 数组
    """
    CALC_GMM._validate_input(region, "长轴")
    utm_zone = int((lon + 180.0) // 6.0) + 1
    r_scan = np.arange(1.0, extent + 1.0, 1.0)

    # 注意：calc_axis_curves 返回的是 PGA/PGV 预测值（gal / cm/s），不是距离；
    # field_domain 内部用 minor_axis_curve 把"同值"换算成距离(km)。
    pga_long, pga_short, _, _ = calc_axis_curves(Ms, region, r_scan)

    # 最外圈椭圆：长轴距离 a_max（退化时外圈为圆 b_max = a_max）
    a_max, b_max, _ = field_domain(pga_long, pga_short, r_scan)
    if a_max <= 1.5:
        raise ValueError("场范围过小，无法构成完整椭圆场！")

    # 批量采样（拒绝落在椭圆外的点），一次性投影成经纬度
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    inv = Transformer.from_crs(epsg, "epsg:4326", always_xy=True)
    rng = np.random.default_rng(seed)
    sta_lon, sta_lat = [], []
    while len(sta_lon) < n:
        m = max(n - len(sta_lon), 1) * 4  # 一次多采几个，减少循环
        R = rng.uniform(1.0, a_max, m)  # 震中距 1~a_max km
        beta = rng.uniform(0.0, 2.0 * math.pi, m)
        theta = beta - math.radians(90.0 - strike)
        r_max = (
            a_max
            * b_max
            / np.sqrt(
                (b_max * np.cos(theta)) ** 2 + (a_max * np.sin(theta)) ** 2
            )
        )
        ok = R <= r_max  # 只保留椭圆内的点
        R = R[ok]
        beta = beta[ok]
        if R.size == 0:
            continue
        sx = epi_x + R * np.cos(beta) * 1000.0
        sy = epi_y + R * np.sin(beta) * 1000.0
        slon, slat = np.asarray(inv.transform(sx, sy))
        sta_lon.extend(slon.tolist())
        sta_lat.extend(slat.tolist())
    return np.asarray(sta_lon[:n]), np.asarray(sta_lat[:n])


def _pga_pgv_grids(lon, lat, strike, region, Ms, extent=400):
    """
    计算 PGA / PGV 网格场（1~extent km 椭圆环点云 → griddata → 经纬度），
    供 PGA/PGV 双图 / 整合图共用。

    返回：
        (lons, lats, g_pga, g_pgv, utm_zone, epi_x, epi_y, epsg)
        g_pga / g_pgv 为掩膜后的网格场（PGA 单位 gal，PGV 单位 cm/s）
    """
    CALC_GMM._validate_input(region, "长轴")
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    # 第 1 步：长短轴衰减曲线
    r_scan = np.arange(1.0, extent + 1.0, 1.0)  # 1~extent km
    pga_long, pga_short, pgv_long, pgv_short = calc_axis_curves(
        Ms, region, r_scan
    )

    # 第 2 步：椭圆场点云（含走向旋转）
    angle = math.radians(
        90.0 - strike
    )  # 走向(自北顺时针) → UTM 旋转角(自东逆时针)
    pts_pga = calc_field_ellipse(pga_long, pga_short, r_scan, angle)
    pts_pgv = calc_field_ellipse(pgv_long, pgv_short, r_scan, angle)

    # 第 3 步：插值 + 经纬度投影
    lons, lats, g_pga = interpolate_to_ll(pts_pga, epi_x, epi_y, epsg, extent)
    _, _, g_pgv = interpolate_to_ll(pts_pgv, epi_x, epi_y, epsg, extent)
    return lons, lats, g_pga, g_pgv, utm_zone, epi_x, epi_y, epsg


def plot_pga_pgv_fields(
    lon,
    lat,
    strike,
    region,
    Ms,
    output_file=None,
    extent=400,
    sta_lon=None,
    sta_lat=None,
):
    """
    画 GB18306-2015 PGA / PGV 衰减云图（1~extent km，默认 400 km），
    并可选叠加台站预测点

    左图 PGA（gal）、右图 PGV（cm/s），共用 USGS MMI 色标；
    图上含：填色云图、黑色等值线（带数值标注）、震中星标、
    走向箭头、台站圆点（填充色 = 预测值所在区间的色标颜色）。

    参数：
        lon          震中经度（东经为正，度）
        lat          震中纬度（北纬为正，度）
        strike       长轴方位角：正北=0°，顺时针，东=90°，西=270°，0~360°
        region       分区："青藏区" / "新疆区" / "东部区" / "中部区"
        Ms           面波震级（GB18306 公式里的 M）
        output_file  图片文件名；不填则自动命名
        extent       绘图范围半径（km），默认 400
        sta_lon      台站经度（可选，标量/向量/矩阵）
        sta_lat      台站纬度（可选，与 sta_lon 同形状）

    返回：
        fig, axs  （matplotlib 图形对象）
    """
    # ---------- 检查参数 ----------
    CALC_GMM._validate_input(region, "长轴")  # 分区写错会在这里报错
    strike = strike % 360.0  # 走向归一到 0~360°

    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    # ---------- 第 1~3 步：衰减曲线 → 点云 → 插值网格（共用函数）----------
    lons, lats, g_pga, g_pgv, utm_zone, epi_x, epi_y, epsg = _pga_pgv_grids(
        lon, lat, strike, region, Ms, extent
    )

    # ---------- 色标 ----------
    cmap = ListedColormap(USGS_MMI_COLORS, name="usgs_mmi")
    cmap.set_under(USGS_MMI_COLORS[0])  # 低于最小分界 → I 白
    cmap.set_over(USGS_MMI_COLORS[-1])  # 超过最大分界 → X+ 深红
    norm_pga = BoundaryNorm(PGA_LEVELS, ncolors=len(USGS_MMI_COLORS))
    norm_pgv = BoundaryNorm(PGV_LEVELS, ncolors=len(USGS_MMI_COLORS))

    # ---------- 台站预测（可选）----------
    has_sta = sta_lon is not None and sta_lat is not None
    if has_sta:
        # verbose=False：台站很多时不刷屏
        pga_pred, pgv_pred = predict_pga_pgv(
            lon,
            lat,
            strike,
            region,
            Ms,
            sta_lon,
            sta_lat,
            extent,
            verbose=False,
        )
        sta_lon_a = np.asarray(sta_lon, dtype=float)
        sta_lat_a = np.asarray(sta_lat, dtype=float)
    else:
        pga_pred = pgv_pred = None

    # ---------- 画图（1×2：PGA 左，PGV 右）----------
    cm2in = 1.0 / 2.54
    fig, axs = plt.subplots(1, 2, figsize=(20 * cm2in, 9 * cm2in))

    # --- PGA（左）---
    ax = axs[0]
    # contourf：填色云图，颜色按 PGA_LEVELS 分档
    cf = ax.contourf(
        lons,
        lats,
        g_pga,
        levels=PGA_LEVELS,
        cmap=cmap,
        norm=norm_pga,
        extend="both",
    )
    # contour：黑色等值线（分界值位置），并标注数值
    cs = ax.contour(
        lons, lats, g_pga, levels=PGA_LEVELS, colors="k", linewidths=0.5
    )
    ax.clabel(cs, fmt="%.0f", fontsize=6)
    # 色标：刻度 = 分界值
    cb = fig.colorbar(cf, ax=ax, ticks=PGA_LEVELS, pad=0.05, shrink=0.8)
    cb.ax.set_yticklabels([f"{x:g}" for x in PGA_LEVELS])
    cb.set_label("PGA (gal)", fontsize=8)
    ax.set_title("PGA (gal)", fontsize=9)

    # --- PGV（右，分界 = PGA ÷ 10）---
    ax = axs[1]
    cf = ax.contourf(
        lons,
        lats,
        g_pgv,
        levels=PGV_LEVELS,
        cmap=cmap,
        norm=norm_pgv,
        extend="both",
    )
    cs = ax.contour(
        lons, lats, g_pgv, levels=PGV_LEVELS, colors="k", linewidths=0.5
    )
    ax.clabel(cs, fmt=FuncFormatter(lambda v, pos: fmt_pgv(v)), fontsize=6)
    cb = fig.colorbar(cf, ax=ax, ticks=PGV_LEVELS, pad=0.05, shrink=0.8)
    cb.ax.set_yticklabels([fmt_pgv(x) for x in PGV_LEVELS])
    cb.set_label("PGV (cm/s)", fontsize=8)
    ax.set_title("PGV (cm/s)", fontsize=9)

    # ---------- 公共元素（震中、走向箭头、台站、范围、坐标轴）----------
    for i, ax in enumerate(axs):
        # 震中星标
        ax.plot(lon, lat, "k*", markersize=7, zorder=10)

        # 走向指示线：从震中指向长轴方向（长度 = extent）
        max_r = extent
        sr = math.radians(strike)
        arr_lon, arr_lat = km_to_lonlat(
            lon, lat, max_r * math.sin(sr), max_r * math.cos(sr), utm_zone
        )
        ax.annotate(
            "",
            xy=(arr_lon, arr_lat),
            xytext=(lon, lat),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )
        ax.text(
            arr_lon,
            arr_lat,
            f"Strike {strike:g} deg",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
        )

        # 台站：默认圆点，填充色 = 预测值所在区间的色标颜色
        if has_sta:
            vals = pga_pred if i == 0 else pgv_pred
            norm = norm_pga if i == 0 else norm_pgv
            for j, (slon, slat, val) in enumerate(
                zip(sta_lon_a.ravel(), sta_lat_a.ravel(), vals.ravel())
            ):
                if math.isnan(val):
                    continue  # 场外的台站不画
                ax.scatter(
                    slon,
                    slat,
                    marker="o",
                    s=15,
                    facecolor=cmap(norm(val)),
                    edgecolor="k",
                    linewidth=0.5,
                    zorder=12,
                )
                if i == 0:
                    ax.text(
                        slon,
                        slat,
                        f"S{j + 1}",
                        fontsize=6,
                        ha="left",
                        va="bottom",
                        zorder=13,
                    )
            if i == 0:
                ax.scatter(
                    [],
                    [],
                    marker="o",
                    s=15,
                    facecolor="gray",
                    edgecolor="k",
                    linewidth=0.5,
                    label="Station (predicted)",
                )
                ax.legend(loc="lower right", fontsize=7)

        # 图幅范围：以 extent 为准，留 5% 边距
        dlat = extent * 1.05 / 111.32
        dlon = extent * 1.05 / (111.32 * math.cos(math.radians(lat)))
        ax.set_xlim(lon - dlon, lon + dlon)
        ax.set_ylim(lat - dlat, lat + dlat)
        # aspect = 1/cos(纬度)：让经纬度图上 1° 经度 ≈ 1° 纬度，形状不变形
        ax.set_aspect(1.0 / math.cos(math.radians(lat)))
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"GB18306-2015 PGA/PGV Field — {region}, "
        f"Ms = {Ms:g}, Strike = {strike:g} deg (from North)",
        fontsize=10,
    )
    fig.tight_layout()

    if output_file is None:
        output_file = f"PGA_PGV_field_{region}_Ms{Ms:g}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存：{os.path.abspath(output_file)}")
    return fig, axs


def _intensity_curves(region, Ms, r_scan):
    """
    烈度衰减曲线：沿长轴/短轴的值 I(t) 及各自标准差 σ。
    公式 I = A + B*Ms + C*lg(R + R0)，参数来自 CALC_INTENSITY._PARAMS。
    返回 (I_long, I_short, sigma_long, sigma_short)
    """
    A1, B1, C1, R01, s1 = CALC_INTENSITY._PARAMS[(region, "长轴")]
    A2, B2, C2, R02, s2 = CALC_INTENSITY._PARAMS[(region, "短轴")]
    I_long = A1 + B1 * Ms + C1 * np.log10(r_scan + R01)
    I_short = A2 + B2 * Ms + C2 * np.log10(r_scan + R02)
    return I_long, I_short, s1, s2


def _gmm_curves(region, Ms, r_scan, param_type):
    """
    PGA（param_type="aE"）或 PGV（param_type="vE"）沿长轴/短轴的
    中值与 ±1σ（对数域：下限=中值/10^σ，上限=中值*10^σ）。
    返回 (long_med, long_lo, long_up, short_med, short_lo, short_up)
    """
    long_med, long_lo, long_up = [], [], []
    short_med, short_lo, short_up = [], [], []
    for r in r_scan:
        res = CALC_GMM.calculate(Ms, r, region, "长轴")
        tup = res[0] if param_type == "aE" else res[1]
        long_med.append(tup[0])
        long_lo.append(tup[1])
        long_up.append(tup[2])
        res = CALC_GMM.calculate(Ms, r, region, "短轴")
        tup = res[0] if param_type == "aE" else res[1]
        short_med.append(tup[0])
        short_lo.append(tup[1])
        short_up.append(tup[2])
    return (
        np.asarray(long_med),
        np.asarray(long_lo),
        np.asarray(long_up),
        np.asarray(short_med),
        np.asarray(short_lo),
        np.asarray(short_up),
    )


def _intensity_grids(lon, lat, strike, region, Ms, extent=400):
    """
    烈度云图网格场（与 PGA/PGV 同一套出图逻辑）：
    长轴烈度曲线 → 反算短轴距 → 椭圆环点云 → griddata → 经纬度。
    返回 (lons, lats, grid, utm_zone, epi_x, epi_y, epsg)
    """
    CALC_INTENSITY._validate_input(region, "长轴")
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)
    r_scan = np.arange(1.0, extent + 1.0, 1.0)
    I_long, I_short, _, _ = _intensity_curves(region, Ms, r_scan)
    angle = math.radians(90.0 - strike)
    pts = calc_field_ellipse(I_long, I_short, r_scan, angle)
    lons, lats, grid = interpolate_to_ll(pts, epi_x, epi_y, epsg, extent)
    return lons, lats, grid, utm_zone, epi_x, epi_y, epsg


def _cal_gb17742(pga, pgv):
    """
    GB/T 17742-2020 仪器烈度换算（支持数组/矩阵/掩膜数组）

    公式：
        I_A = 3.17 * log10(PGA / 100) + 6.59
        I_V = 3.00 * log10(PGV / 100) + 9.77
    规则：
        I_V >= 6 且 I_A >= 6 → 取 min(I_V, 12)
        否则 → 取 (I_A + I_V) / 2，且 >= 1
    结果保留 1 位小数。
    """
    i_a = 3.17 * np.log10(pga / 100.0) + 6.59
    i_v = 3.00 * np.log10(pgv / 100.0) + 9.77
    both_high = (i_v >= 6.0) & (i_a >= 6.0)
    i0 = np.ma.where(both_high, np.ma.minimum(i_v, 12.0), (i_a + i_v) / 2.0)
    i0 = np.ma.maximum(i0, 1.0)
    return np.ma.round(i0, 1)


# ==================== 整合输出（2×4 整合图 + 综合表）====================


def plot_gb18306_all(
    lon,
    lat,
    strike,
    region,
    Ms,
    intensities,
    extent=400,
    output_file=None,
    sta_lon=None,
    sta_lat=None,
):
    """
    画一张整合图（2×4 子图）：
      第一排（云图，都带 colorbar）：
        烈度云图 | PGA 云图 | PGV 云图 | 仪器烈度云图（GB17742）
      第二排（衰减曲线，距离 1~extent km，长轴/短轴 + ±1σ 范围带）：
        烈度 | PGA | PGV | 仪器烈度（GB17742）
    第 4 列 = 由第 2、3 列的 PGA、PGV 按 GB/T 17742-2020 公式换算的仪器烈度。
    三个云图共用 USGS MMI 色标、同一震中与走向；
    可选叠加台站圆点（各云图按各自预测值所在区间的色标填色）。

    参数：
        lon, lat     震中经纬度（东经/北纬为正，度）
        strike       走向：正北=0°，顺时针，东=90°，西=270°，0~360°
        region       分区："青藏区" / "新疆区" / "东部区" / "中部区"
        Ms           面波震级
        intensities  烈度列表（阿拉伯数字）
        extent       场范围半径（km），默认 400
        output_file  图片文件名；不填则自动命名
        sta_lon, sta_lat  台站经纬度（可选）

    返回：
        fig, axs  （matplotlib 图形对象，axs 形状为 (2, 4)）
    """
    # ---------- 检查参数 ----------
    CALC_INTENSITY._validate_input(region, "长轴")
    CALC_GMM._validate_input(region, "长轴")
    if not intensities:
        raise ValueError("intensities 不能为空！")
    strike = strike % 360.0
    utm_zone = int((lon + 180.0) // 6.0) + 1

    # ---------- 云图数据（共用函数）----------
    lons_i, lats_i, g_I, utm_zone, epi_x, epi_y, epsg = _intensity_grids(
        lon, lat, strike, region, Ms, extent
    )
    lons, lats, g_pga, g_pgv, utm_zone, epi_x, epi_y, epsg = _pga_pgv_grids(
        lon, lat, strike, region, Ms, extent
    )
    # 第 4 列：由 PGA + PGV 网格按 GB17742 换算仪器烈度
    g_I_17742 = _cal_gb17742(g_pga, g_pgv)

    # ---------- 衰减曲线数据（1~extent km）----------
    r_scan = np.arange(1.0, extent + 1.0, 1.0)
    I_long, I_short, sI_l, sI_s = _intensity_curves(region, Ms, r_scan)
    pga_lm, pga_ll, pga_lu, pga_sm, pga_sl, pga_su = _gmm_curves(
        region, Ms, r_scan, "aE"
    )
    pgv_lm, pgv_ll, pgv_lu, pgv_sm, pgv_sl, pgv_su = _gmm_curves(
        region, Ms, r_scan, "vE"
    )

    # ---------- 色标（三个云图共用 USGS MMI）----------
    cmap = ListedColormap(USGS_MMI_COLORS, name="usgs_mmi")
    cmap.set_under(USGS_MMI_COLORS[0])  # 低于最小分界 → I 白
    cmap.set_over(USGS_MMI_COLORS[-1])  # 超过最大分界 → X+ 深红
    INTENSITY_LEVELS = np.arange(1, 12)  # 烈度分界：1~11 度
    norm_I = BoundaryNorm(INTENSITY_LEVELS, ncolors=len(USGS_MMI_COLORS))
    norm_pga = BoundaryNorm(PGA_LEVELS, ncolors=len(USGS_MMI_COLORS))
    norm_pgv = BoundaryNorm(PGV_LEVELS, ncolors=len(USGS_MMI_COLORS))

    # ---------- 台站预测（可选）----------
    has_sta = sta_lon is not None and sta_lat is not None
    if has_sta:
        sta_lon_a = np.asarray(sta_lon, dtype=float)
        sta_lat_a = np.asarray(sta_lat, dtype=float)
        pga_pred, pgv_pred = predict_pga_pgv(
            lon,
            lat,
            strike,
            region,
            Ms,
            sta_lon_a,
            sta_lat_a,
            extent,
            verbose=False,
        )
        # 台站烈度（烈度云图圆点填色用）
        fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
        sx, sy = np.asarray(
            fwd.transform(sta_lon_a.ravel(), sta_lat_a.ravel())
        )
        dx = (sx - epi_x) / 1000.0
        dy = (sy - epi_y) / 1000.0
        R_sta = np.hypot(dx, dy)
        theta_sta = np.arctan2(dy, dx) - math.radians(90.0 - strike)
        I_pred, _, _ = _solve_intensity(R_sta, theta_sta, region, Ms, extent)
        # 台站的 GB17742 仪器烈度（由预测的 PGA/PGV 换算）
        I17742_pred = _cal_gb17742(pga_pred, pgv_pred)
    else:
        pga_pred = pgv_pred = I_pred = I17742_pred = None

    # ---------- 2×4 子图 ----------
    cm2in = 1.0 / 2.54
    fig, axs = plt.subplots(2, 4, figsize=(40 * cm2in, 18 * cm2in))

    # ===== 第一排：云图（烈度 | PGA | PGV | GB17742 仪器烈度）=====
    cloud_cfgs = [
        (
            g_I,
            INTENSITY_LEVELS,
            norm_I,
            "Intensity (烈度云图)",
            "Intensity",
            lambda v: to_roman(v),
        ),
        (
            g_pga,
            PGA_LEVELS,
            norm_pga,
            "PGA (gal)",
            "PGA (gal)",
            lambda v: f"{v:g}",
        ),
        (g_pgv, PGV_LEVELS, norm_pgv, "PGV (cm/s)", "PGV (cm/s)", fmt_pgv),
        (
            g_I_17742,
            INTENSITY_LEVELS,
            norm_I,
            "Intensity (GB17742 仪器烈度)",
            "Intensity (GB17742)",
            lambda v: to_roman(v),
        ),
    ]
    for i, (grid, levels, norm, title, cbar_label, label_fmt) in enumerate(
        cloud_cfgs
    ):
        ax = axs[0, i]
        # 填色云图 + 等值线 + 数值标注
        cf = ax.contourf(
            lons,
            lats,
            grid,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        cs = ax.contour(
            lons, lats, grid, levels=levels, colors="k", linewidths=0.5
        )
        ax.clabel(
            cs, fmt=FuncFormatter(lambda v, pos, f=label_fmt: f(v)), fontsize=6
        )
        cb = fig.colorbar(cf, ax=ax, ticks=levels, pad=0.03, shrink=0.8)
        cb.ax.set_yticklabels([label_fmt(x) for x in levels])
        cb.set_label(cbar_label, fontsize=8)
        ax.set_title(title, fontsize=9)

        # 震中星标 + 走向箭头
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
        ax.text(
            arr_lon,
            arr_lat,
            f"Strike {strike:g} deg",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
        )

        # 台站圆点：按各云图预测值所在区间填色
        if has_sta:
            vals = (
                I_pred
                if i == 0
                else (
                    pga_pred
                    if i == 1
                    else (pgv_pred if i == 2 else I17742_pred)
                )
            )
            for j, (slon, slat, val) in enumerate(
                zip(sta_lon_a.ravel(), sta_lat_a.ravel(), vals.ravel())
            ):
                if math.isnan(val):
                    continue
                face = (
                    intensity_color(val)
                    if (i == 0 or i == 3)
                    else cmap(norm(val))
                )
                ax.scatter(
                    slon,
                    slat,
                    marker="o",
                    s=15,
                    facecolor=face,
                    edgecolor="k",
                    linewidth=0.5,
                    zorder=12,
                )
                if i == 0:
                    ax.text(
                        slon,
                        slat,
                        f"S{j + 1}",
                        fontsize=6,
                        ha="left",
                        va="bottom",
                        zorder=13,
                    )
            if i == 0:
                ax.scatter(
                    [],
                    [],
                    marker="o",
                    s=15,
                    facecolor="gray",
                    edgecolor="k",
                    linewidth=0.5,
                    label="Station (predicted)",
                )
                ax.legend(loc="lower right", fontsize=7)

        # 图幅与坐标
        dlat = extent * 1.05 / 111.32
        dlon = extent * 1.05 / (111.32 * math.cos(math.radians(lat)))
        ax.set_xlim(lon - dlon, lon + dlon)
        ax.set_ylim(lat - dlat, lat + dlat)
        ax.set_aspect(1.0 / math.cos(math.radians(lat)))
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    # ===== 第二排：衰减曲线（长轴/短轴 ±1σ，1~extent km）=====
    # --- 烈度（线性坐标）---
    ax = axs[1, 0]
    ax.plot(r_scan, I_long, color="tab:red", lw=1.4, label="长轴")
    ax.plot(r_scan, I_short, color="tab:blue", lw=1.4, ls="--", label="短轴")
    ax.fill_between(
        r_scan,
        I_long - sI_l,
        I_long + sI_l,
        color="tab:red",
        alpha=0.15,
        label="长轴±1σ",
    )
    ax.fill_between(
        r_scan,
        I_short - sI_s,
        I_short + sI_s,
        color="tab:blue",
        alpha=0.15,
        label="短轴±1σ",
    )
    ax.set_title("Intensity 衰减曲线", fontsize=9)
    ax.set_ylabel("Intensity", fontsize=8)
    ax.set_ylim(1, 11)  # 烈度 Y 轴：1~11 度
    ax.set_yticks(np.arange(1, 11.1, 1))  # 刻度：1~11 的整数

    # --- PGA（对数坐标）---
    ax = axs[1, 1]
    ax.plot(r_scan, pga_lm, color="tab:red", lw=1.4, label="长轴")
    ax.plot(r_scan, pga_sm, color="tab:blue", lw=1.4, ls="--", label="短轴")
    ax.fill_between(
        r_scan, pga_ll, pga_lu, color="tab:red", alpha=0.15, label="长轴±1σ"
    )
    ax.fill_between(
        r_scan, pga_sl, pga_su, color="tab:blue", alpha=0.15, label="短轴±1σ"
    )
    ax.set_yscale("log")
    ax.set_title("PGA 衰减曲线", fontsize=9)
    ax.set_ylabel("PGA (gal)", fontsize=8)
    ax.set_xscale("log")  # 距离 X 轴：log

    # --- PGV（对数坐标）---
    ax = axs[1, 2]
    ax.plot(r_scan, pgv_lm, color="tab:red", lw=1.4, label="长轴")
    ax.plot(r_scan, pgv_sm, color="tab:blue", lw=1.4, ls="--", label="短轴")
    ax.fill_between(
        r_scan, pgv_ll, pgv_lu, color="tab:red", alpha=0.15, label="长轴±1σ"
    )
    ax.fill_between(
        r_scan, pgv_sl, pgv_su, color="tab:blue", alpha=0.15, label="短轴±1σ"
    )
    ax.set_yscale("log")
    ax.set_title("PGV 衰减曲线", fontsize=9)
    ax.set_ylabel("PGV (cm/s)", fontsize=8)
    ax.set_xscale("log")  # 距离 X 轴：log

    # --- GB17742 仪器烈度衰减曲线（由 PGA/PGV 中值与 ±1σ 上下界换算）---
    ax = axs[1, 3]
    i_l = _cal_gb17742(pga_lm, pgv_lm)  # 长轴中值换算
    i_s = _cal_gb17742(pga_sm, pgv_sm)  # 短轴中值换算
    i_l_lo = _cal_gb17742(pga_ll, pgv_ll)  # 长轴 ±1σ 下界换算
    i_l_up = _cal_gb17742(pga_lu, pgv_lu)  # 长轴 ±1σ 上界换算
    i_s_lo = _cal_gb17742(pga_sl, pgv_sl)  # 短轴 ±1σ 下界换算
    i_s_up = _cal_gb17742(pga_su, pgv_su)  # 短轴 ±1σ 上界换算
    ax.plot(r_scan, i_l, color="tab:red", lw=1.4, label="长轴")
    ax.plot(r_scan, i_s, color="tab:blue", lw=1.4, ls="--", label="短轴")
    ax.fill_between(
        r_scan, i_l_lo, i_l_up, color="tab:red", alpha=0.15, label="长轴±1σ"
    )
    ax.fill_between(
        r_scan, i_s_lo, i_s_up, color="tab:blue", alpha=0.15, label="短轴±1σ"
    )
    ax.set_title("GB17742 仪器烈度衰减曲线", fontsize=9)
    ax.set_ylabel("Intensity (GB17742)", fontsize=8)
    ax.set_ylim(1, 11)  # 烈度 Y 轴：1~11 度
    ax.set_yticks(np.arange(1, 11.1, 1))  # 刻度：1~11 的整数

    # 第二排公共元素
    for ax in axs[1, :]:
        ax.set_xscale("log")  # 距离 X 轴：log
        ax.set_xlim(1, extent)
        ax.set_xlabel("R (km)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=6)

    fig.suptitle(
        f"GB18306-2015 Intensity / PGA / PGV — {region}, "
        f"Ms = {Ms:g}, Strike = {strike:g} deg (from North)",
        fontsize=11,
    )
    fig.tight_layout()

    if output_file is None:
        output_file = f"GB18306_all_{region}_Ms{Ms:g}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存：{os.path.abspath(output_file)}")
    return fig, axs


def export_all_table(
    lon,
    lat,
    strike,
    region,
    Ms,
    sta_lon,
    sta_lat,
    sta_id=None,
    extent=400,
    output_file=None,
):
    """
    【核心函数】台站综合预测表：PGA/PGV + 两套烈度 + 所在椭圆，导出 TXT
    =================================================================
    对任意多个台站一次算出并输出一张表（txt，tab 分隔，Excel 可直接打开）：
        Sta_ID  Sta_longi  Sta_lati
        Repi(km)
        三套椭圆长短轴（烈度 / PGA / PGV 各自的"所在椭圆"）：
        Repi_long_Intensity(km)  Repi_short_Intensity(km)
        Repi_long_PGA(km)        Repi_short_PGA(km)
        Repi_long_PGV(km)        Repi_short_PGV(km)
        PGA(gal)  PGV(cm/s)
        Intensity  Intensity_roman
        Intensity_GB17742  Intensity_roman_GB17742

    每个台站的计算流程：
        ① 经纬度 → UTM → 相对震中的 (dx, dy) → 震中距 R、相对长轴夹角 θ；
        ② 在 PGA/PGV 衰减场上解出 PGA、PGV 值；先取整到 2 位小数，
           再用取整后的值按解析公式反算各自所在椭圆的长轴距、短轴距
           （Repi_long_PGA / Repi_short_PGA 等，与 class 的
            invert_R_from_aE / invert_R_from_vE 严格一致）；
        ③ 在烈度圈椭圆族上解出 GB18306 烈度，以及所在烈度圈椭圆的
           长轴距、短轴距（Repi_long_Intensity / Repi_short_Intensity）。
           先对烈度取整到 2 位小数、再用取整后的值反算长短轴，表内
           严格自洽，可用 invert_R(Intensity, Ms, region, "长轴"/"短轴") 验证；
        ④ 仪器烈度：
            Intensity          = GB18306 衰减关系直接反算
                                （I = A + B*Ms + C*lg(R + R0)）；
            Intensity_GB17742  = 由 PGA、PGV 按 GB/T 17742-2020
                                仪器烈度公式换算；
        ⑤ 罗马数字列是对应烈度的显示（10 度及以上统一显示 X+）。

    约定：
        - 单位：PGA（gal）、PGV（cm/s）、距离（km）；
        - 场外台站（超出最外圈椭圆）对应值记 NaN；
        - 输出 txt 为 utf-8-sig 编码，Excel 双击即可打开。

    参数：
        lon, lat     震中经纬度（东经/北纬为正，度）
        strike       走向：正北=0°，顺时针，东=90°，西=270°，0~360°
        region       分区："青藏区" / "新疆区" / "东部区" / "中部区"
        Ms           面波震级
        sta_lon      台站经度（标量/一维数组/二维矩阵均可）
        sta_lat      台站纬度（与 sta_lon 同形状）
        sta_id       台站编号（可选，默认 S1、S2…）
        extent       衰减曲线范围半径（km），默认 400
        output_file  输出 txt 文件名；不填自动命名

    返回：
        pandas.DataFrame（已同时保存为 txt，可继续在脚本里使用）

    用法示例：
        df = export_all_table(
            lon=103.0, lat=25.0, strike=90, region="新疆区", Ms=7.5,
            sta_lon=[103.0, 103.5, 102.5],
            sta_lat=[25.0, 25.2, 25.1],
            output_file="stations.txt",
        )
        print(df.head())
    """
    # ---------- ① 参数校验 ----------
    CALC_INTENSITY._validate_input(region, "长轴")
    CALC_GMM._validate_input(region, "长轴")
    strike = strike % 360.0
    utm_zone = int((lon + 180.0) // 6.0) + 1
    epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    epi_x, epi_y = lonlat_to_utm(lon, lat, utm_zone)

    # ---------- ② 长短轴衰减曲线（1~extent km）----------
    r_scan = np.arange(1.0, extent + 1.0, 1.0)
    pga_long, pga_short, pgv_long, pgv_short = calc_axis_curves(
        Ms, region, r_scan
    )

    # ---------- ③ 台站批量投影 → 震中距 R、相对长轴夹角 theta ----------
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

    # ---------- ④ 解 PGA/PGV 值 + 各自所在椭圆长短轴 ----------
    pga_raw, _, _ = _solve_values(pga_long, pga_short, r_scan, R, theta)
    pgv_raw, _, _ = _solve_values(pgv_long, pgv_short, r_scan, R, theta)
    # 先取整到 2 位小数，再用取整后的值解析反算所在椭圆，
    # 保证表内 Repi_long/Repi_short_PGA/PGV 与 PGA/PGV 严格自洽
    pga = np.round(pga_raw, 2)
    pgv = np.round(pgv_raw, 2)
    a_eq_pga, b_eq_pga = _gmm_ellipse_radii(pga, Ms, region, "aE")
    a_eq_pgv, b_eq_pgv = _gmm_ellipse_radii(pgv, Ms, region, "vE")

    # ---------- ⑤ 两套烈度 ----------
    # 烈度①：GB18306 衰减关系直接反算。先对烈度取整到 2 位小数，
    # 再用它反算所在烈度圈椭圆长短轴，保证表内严格自洽
    I_raw, _, _ = _solve_intensity(R, theta, region, Ms, extent)
    I = np.round(I_raw, 2)
    a_eq_i, b_eq_i = _ellipse_radii(I, region, Ms)
    a_eq_i = np.maximum(a_eq_i, 0.1)  # 与 invert_R 的 0.1 km 截断一致
    b_eq_i = np.maximum(b_eq_i, 0.1)
    # 烈度②：GB/T 17742-2020 仪器烈度（由 PGA/PGV 换算，保留 1 位小数）
    I17742 = GB17742_2020_Cal_instrument_intensity.cal_Intensity_matrix(
        pga, pgv
    )

    # ---------- ⑥ 组表 + 导出 ----------
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
            "Repi_long_Intensity(km)": np.round(a_eq_i, 2),  # 烈度圈椭圆长轴距
            "Repi_short_Intensity(km)": np.round(
                b_eq_i, 2
            ),  # 烈度圈椭圆短轴距
            "Repi_long_PGA(km)": np.round(a_eq_pga, 2),  # PGA 椭圆长轴距
            "Repi_short_PGA(km)": np.round(b_eq_pga, 2),  # PGA 椭圆短轴距
            "Repi_long_PGV(km)": np.round(a_eq_pgv, 2),  # PGV 椭圆长轴距
            "Repi_short_PGV(km)": np.round(b_eq_pgv, 2),  # PGV 椭圆短轴距
            "PGA(gal)": np.round(pga, 2),
            "PGV(cm/s)": np.round(pgv, 2),
            "Intensity": np.round(I, 2),
            "Intensity_roman": [
                to_roman(v) if not math.isnan(v) else "NaN" for v in I
            ],
            "Intensity_GB17742": np.round(I17742, 1),
            "Intensity_roman_GB17742": [
                to_roman(v) if not math.isnan(v) else "NaN" for v in I17742
            ],
        }
    )

    if output_file is None:
        output_file = f"GB18306_pre_all_table_{region}_Ms{Ms:g}.txt"
    df.to_csv(output_file, sep="\t", index=False, encoding="utf-8-sig")
    print(f"已导出：{os.path.abspath(output_file)}")
    return df


def me():
    pass


if __name__ == "__main__":
    # ============ 演示：整合图（2×4）+ 单图 + 综合表 ============
    # 演示参数（可自行修改）
    demo_ms = 7.5  # 面波震级
    lon = 103.0  # 震中经度
    lat = 25.0  # 震中纬度
    strike = 90  # 走向：正北=0°，顺时针
    region = "青藏区"  # 分区
    R_max = 400  # 场范围半径（km）
    demo_intensities = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

    # 1~R_max km 随机台站（固定种子，可复现）
    demo_sta_lon, demo_sta_lat = random_stations(
        lon=lon,
        lat=lat,
        strike=strike,
        region=region,
        Ms=demo_ms,
        extent=R_max,
        n=50,
        seed=42,
    )

    # ① 整合图：2×4（烈度 | PGA | PGV | GB17742 仪器烈度，云图 + 衰减曲线）
    plot_gb18306_all(
        lon=lon,
        lat=lat,
        strike=strike,
        region=region,
        Ms=demo_ms,
        intensities=demo_intensities,
        extent=R_max,
        sta_lon=demo_sta_lon,
        sta_lat=demo_sta_lat,
    )

    # ② 单图：GB18306 烈度圈（填色椭圆）
    plot_intensity_ellipses(
        lon=lon,
        lat=lat,
        strike=strike,
        region=region,
        intensities=demo_intensities,
        Ms=demo_ms,
    )

    # ③ 单图：PGA + PGV 衰减云图 + 台站
    plot_pga_pgv_fields(
        lon=lon,
        lat=lat,
        strike=strike,
        region=region,
        Ms=demo_ms,
        extent=R_max,
        sta_lon=demo_sta_lon,
        sta_lat=demo_sta_lat,
    )

    # ④ 综合表（核心输出）：PGA/PGV + GB18306 烈度 + GB17742 仪器烈度
    #    A100 是返回的 DataFrame，可继续在脚本里使用
    export_all_table(
        lon=lon,
        lat=lat,
        strike=strike,
        region=region,
        Ms=demo_ms,
        sta_lon=demo_sta_lon,
        sta_lat=demo_sta_lat,
    )

    ################################
    # # #测试
    ####################################

    A100 = export_all_table(
        lon=101.224,
        lat=37.791,
        strike=104,
        region="青藏区",
        Ms=6.9,
        sta_lon=[101.3, 101.5, 101.8, 102.1, 102.5],
        sta_lat=[37.8, 37.8, 37.8, 37.8, 37.8],
    )
