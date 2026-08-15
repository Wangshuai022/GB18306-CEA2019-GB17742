# -*- coding: utf-8 -*-
"""
GB18306-2015 烈度圈（椭圆）绘制工具
===================================

功能
----
输入震中经纬度、走向、GB18306 分区、烈度列表和面波震级 Ms，
按 GB18306-2015 烈度衰减公式反算各烈度圈的长轴、短轴半径，
用椭圆公式绘制并按走向旋转，输出一张填色烈度圈图（PNG）。

烈度衰减公式
------------
    I = A + B * Ms + C * lg(R + R0)

对给定的烈度 I 和震级 Ms，分别沿长轴、短轴两个方向反解震中距 R，
即得到该烈度圈的长轴、短轴半径。

坐标与走向约定
--------------
- 经纬度采用 WGS84，绘图前经 UTM 投影换算（保证椭圆形状正确）。
- 投影带号由经度自动计算（6° 分带，1~60）；按纬度自动选择
  北半球（EPSG:326xx）或南半球（EPSG:327xx），南北半球均可使用。
- 走向 = 长轴方位角：正北为 0°，顺时针增加，东 90°，南 180°，
  西 270°，范围 0~360°。

烈度显示约定
------------
- 输入烈度为阿拉伯数字，图上统一显示罗马数字：
  I~IX 对应 0.5~9.5 区间，10 度及以上统一显示为 X+。
- 颜色采用 USGS MMI 色标（I 白 → X+ 深红）。

自动保护
--------
- 若某烈度圈在当前 Ms 下不存在（反算出的长短轴 < 1 km），
  自动跳过不画并给出提示；
- 若短轴大于长轴，按长轴绘制圆形并给出提示。

用法示例
--------
    from plot_intensity_ellipses import plot_intensity_ellipses

    plot_intensity_ellipses(
        lon=102.8,
        lat=25.6,
        strike=135,                 # 走向：正北=0°，顺时针，东=90°，西=270°
        region="青藏区",            # 青藏区 / 新疆区 / 东部区 / 中部区
        intensities=[4.5, 5.5, 6.5, 7.5, 8.5, 9.5],   # 阿拉伯数字
        Ms=6.5,                     # 面波震级
    )

直接运行本文件即使用上述示例画图；也可以在别的程序里 import 后调用。
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")               # 不弹窗口，只保存图片
import matplotlib.pyplot as plt

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

# 找到 GB18306_2015_Intensity.py（和本文件在同一个文件夹里）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from GB18306_2015_Intensity import GB18306_2015_IntensityCal

CALCULATOR = GB18306_2015_IntensityCal()

# 中文字体（标题里要显示分区名）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# USGS MMI 烈度色标（I 白 → X+ 深红）
USGS_MMI_COLORS = [
    "#FFFFFF",   # I
    "#BFCCFF",   # II
    "#A0E6FF",   # III
    "#80FFFF",   # IV
    "#7AFF93",   # V
    "#FFFF00",   # VI
    "#FFC800",   # VII
    "#FF9100",   # VIII
    "#FF0000",   # IX
    "#800000",   # X+
]

# 罗马数字表：I↔0.5~1.5，II↔1.5~2.5，…，IX↔8.5~9.5；10 度及以上统一显示 X+
ROMAN = ["I", "II", "III", "IV", "V", "VI",
         "VII", "VIII", "IX"]


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
        epsg = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
        fwd = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
        epi_x, epi_y = fwd.transform(lon, lat)
        inv = Transformer.from_crs(epsg, "epsg:4326", always_xy=True)
        return inv.transform(epi_x + np.asarray(east_km) * 1000.0,
                             epi_y + np.asarray(north_km) * 1000.0)
    # 没有 pyproj 时的简单近似（南北半球通用）
    dlat = np.asarray(north_km) / 111.32
    dlon = np.asarray(east_km) / (111.32 * math.cos(math.radians(lat)))
    return lon + dlon, lat + dlat


def plot_intensity_ellipses(lon, lat, strike, region, intensities,
                            Ms=6.8, output_file=None):
    """
    画 GB18306-2015 烈度圈（一组同心椭圆，一个烈度一个圈）

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

    说明：
        如果某个烈度区在当前 Ms 下不存在（反算出的长短轴 < 1 km），
        该烈度圈会自动跳过不画；若短轴大于长轴，则统一按长轴绘制圆形，
        并给出提示。
    """
    # ---------- 检查参数 ----------
    CALCULATOR._validate_input(region, "长轴")   # 分区写错会在这里报错
    if not intensities:
        raise ValueError("intensities 不能为空！")
    strike = strike % 360.0                      # 走向归一到 0~360°
    intensities = sorted(intensities)            # 从小到大：先画外面的大圈

    utm_zone = int((lon + 180.0) // 6.0) + 1

    # ---------- 第 1 步：反算每个烈度的长轴、短轴（GB18306 公式）----------
    # 不存在的高烈度区（长短轴缩成点）直接跳过，不画
    ellipse_data = []
    for intensity in intensities:
        major = CALCULATOR.invert_R(intensity, Ms, region, "长轴")["mean"]
        minor = CALCULATOR.invert_R(intensity, Ms, region, "短轴")["mean"]
        if minor > major:
            # 自动保护：短轴大于长轴时，统一按长轴绘制圆形
            print(f"  提示：烈度 {intensity:g}（{to_roman(intensity)} 度区）"
                  f"短轴({minor:.1f} km)大于长轴({major:.1f} km)，"
                  f"统一按长轴 {major:.1f} km 绘制圆形")
            minor = major
        if major < 1 or minor < 1:
            print(f"  提示：烈度 {intensity:g}（{to_roman(intensity)} 度区）"
                  f"超出 Ms={Ms:g} 在 {region} 能达到的最大烈度，跳过不画")
            continue
        ellipse_data.append((intensity, major, minor))
        print(f"烈度 {intensity:g}（{to_roman(intensity)} 度区）："
              f"长轴 {major:7.1f} km，短轴 {minor:7.1f} km")

    # 所有烈度区都不存在时，无法画图
    if not ellipse_data:
        raise ValueError("所有烈度区都超出该震级能达到的最大烈度，"
                         "请减小烈度或增大 Ms！")

    # ---------- 画图 ----------
    fig, ax = plt.subplots(figsize=(9, 8))

    for intensity, major, minor in ellipse_data:
        # 第 2 步：椭圆公式（先让长轴朝正东）
        theta = np.linspace(0.0, 2.0 * math.pi, 361)
        ell_x = major * np.cos(theta)            # 长轴方向
        ell_y = minor * np.sin(theta)            # 短轴方向

        # 第 3 步：按走向旋转（正北=0°，顺时针，东=90°，西=270°）
        rot = math.radians(90.0 - strike)
        east = ell_x * math.cos(rot) - ell_y * math.sin(rot)
        north = ell_x * math.sin(rot) + ell_y * math.cos(rot)

        ell_lon, ell_lat = km_to_lonlat(lon, lat, east, north, utm_zone)
        color = intensity_color(intensity)
        # 填充当前椭圆（从小到大逐层向内盖，外圈色带自然露出）
        ax.fill(ell_lon, ell_lat, color=color, alpha=1,
                linewidth=0, zorder=2,
                label=f"Intensity {to_roman(intensity)} "
                      f"({major:.0f}x{minor:.0f} km)")
        # 椭圆边界线（黑色虚线）
        ax.plot(ell_lon, ell_lat, color="k", linewidth=1, linestyle="--")
        # 在椭圆线上标注罗马数字（放在 135° 方向，避开走向箭头）
        label_idx = int(round(135.0 / 360.0 * (len(theta) - 1)))
        ax.text(ell_lon[label_idx], ell_lat[label_idx],
                to_roman(intensity), fontsize=10, fontweight="bold",
                color="black", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="k", lw=0.7, alpha=1),
                zorder=11)

    # 震中
    ax.plot(lon, lat, "k*", markersize=16, zorder=10,
            label=f"Epicenter ({lon:.3f}E, {lat:.3f}N)")

    # 走向指示线：从震中指向长轴方向，方便核对旋转方向对不对
    max_major = max(row[1] for row in ellipse_data)
    strike_rad = math.radians(strike)
    arr_east = max_major * math.sin(strike_rad)
    arr_north = max_major * math.cos(strike_rad)
    arr_lon, arr_lat = km_to_lonlat(lon, lat, arr_east, arr_north, utm_zone)
    ax.annotate("", xy=(arr_lon, arr_lat), xytext=(lon, lat),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.6))
    ax.text(arr_lon, arr_lat, f"Strike {strike:g} deg",
            fontsize=9, color="black")

    # 图的范围按最大的圈（最小的烈度）自动调整，留 15% 边距
    dlat = max_major * 1.15 / 111.32
    dlon = max_major * 1.15 / (111.32 * math.cos(math.radians(lat)))
    ax.set_xlim(lon - dlon, lon + dlon)
    ax.set_ylim(lat - dlat, lat + dlat)

    ax.set_aspect(1.0 / math.cos(math.radians(lat)))
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(f"GB18306-2015 Intensity Ellipses — {region}, "
                 f"Ms = {Ms:g}, Strike = {strike:g} deg (from North)")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    if output_file is None:
        output_file = f"intensity_ellipses_{region}_Ms{Ms:g}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存：{os.path.abspath(output_file)}")
    return fig, ax


if __name__ == "__main__":
    # 直接运行本文件 = 用下面这组参数画一张示例图（改成你的数据就行）
    plot_intensity_ellipses(
        lon=102.8,
        lat=25.6,
        strike=135,
        region="青藏区",
        intensities=[4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        Ms=5.5,
    )
