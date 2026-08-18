"""由震源位置和矩形断层参数生成规则有限断层网格。

主要入口是 ``build_fault_grid``：输入震源经纬度/深度、走向、倾角、长度、
宽度和期望网格间距，返回节点经纬度/深度矩阵、子断层中心及完整几何元数据。
经纬度单位为度，其余长度和深度统一为 km；深度向下为正。网格第一行是断层
上缘，列方向沿走向，行方向沿倾向。

默认震源在断层面上的位置为 ``shypo=0``、``dhypo=0.57*W``。如果几何上缘
出露地表，当前项目约定会把整张断层面竖直下移，使最浅节点恰好位于 0 km，
并在返回字典中记录平移量和调整后的震源深度。直接运行本文件会执行数值检查
和可视化示例；作为库使用时只需导入 ``build_fault_grid``。
"""

import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# 基本参数
# =============================================================================

# 地球平均半径，单位：km
# 用于将局部水平距离转换为经纬度
EARTH_RADIUS_KM = 6371.0088


# =============================================================================
# 辅助函数 1：正数四舍五入
# =============================================================================
def round_half_up_positive(x):
    """
    对正数执行通常意义下的“四舍五入”。

    注意：
    Python 内置 round() 使用的是“银行家舍入”，例如：
        round(6.5) = 6
        round(7.5) = 8

    这里断层网格数量希望采用通常意义上的四舍五入，因此定义：
        6.4 -> 6
        6.5 -> 7
        6.6 -> 7

    参数
    ----------
    x : float
        待四舍五入的正数。

    返回
    -------
    int
        四舍五入后的整数。
    """
    return int(math.floor(x + 0.5))


# =============================================================================
# 辅助函数 2：局部水平坐标转经纬度
# =============================================================================
def local_offset_to_lonlat(
        origin_lon,
        origin_lat,
        east_offset_km,
        north_offset_km
):
    """
    将相对于某一参考点的东、北方向水平位移转换为经纬度。

    坐标定义：
        East  > 0 ：向东
        North > 0 ：向北

    本函数采用球面地球模型，根据水平距离和方位角计算目标点经纬度。

    对于通常几十至几百 km 量级的有限断层，该方法已经具有较好的精度，
    同时不依赖 pyproj 等额外第三方地理坐标库。

    参数
    ----------
    origin_lon : float
        参考点经度，单位：度。

    origin_lat : float
        参考点纬度，单位：度。

    east_offset_km : float 或 ndarray
        相对于参考点向东的距离，单位：km。

    north_offset_km : float 或 ndarray
        相对于参考点向北的距离，单位：km。

    返回
    -------
    lon : ndarray
        目标点经度，单位：度。

    lat : ndarray
        目标点纬度，单位：度。
    """

    # 转换为 NumPy 数组，便于同时处理标量和矩阵
    east_offset_km = np.asarray(east_offset_km, dtype=float)
    north_offset_km = np.asarray(north_offset_km, dtype=float)

    # ---------------------------------------------------------
    # 1. 计算水平距离
    # ---------------------------------------------------------
    horizontal_distance_km = np.sqrt(
        east_offset_km ** 2 + north_offset_km ** 2
    )

    # ---------------------------------------------------------
    # 2. 计算方位角
    #
    # 地震学和地理学通常定义：
    # 正北 = 0°
    # 正东 = 90°
    # 正南 = 180°
    # 正西 = 270°
    #
    # atan2(East, North) 正好符合这一方位角定义。
    # ---------------------------------------------------------
    bearing_rad = np.arctan2(
        east_offset_km,
        north_offset_km
    )

    # 将参考点经纬度转换成弧度
    lat1 = np.deg2rad(origin_lat)
    lon1 = np.deg2rad(origin_lon)

    # 水平距离转换为地心角
    angular_distance = horizontal_distance_km / EARTH_RADIUS_KM

    # ---------------------------------------------------------
    # 3. 球面正算公式
    # ---------------------------------------------------------
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        +
        np.cos(lat1) * np.sin(angular_distance)
        * np.cos(bearing_rad)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad)
        * np.sin(angular_distance)
        * np.cos(lat1),

        np.cos(angular_distance)
        -
        np.sin(lat1) * np.sin(lat2)
    )

    # 转换为度
    lat = np.rad2deg(lat2)
    lon = np.rad2deg(lon2)

    # 经度统一转换到 [-180°, 180°)
    lon = (lon + 180.0) % 360.0 - 180.0

    return lon, lat


# =============================================================================
# 主函数：建立矩形有限断层网格
# =============================================================================
def build_fault_grid(
        Hypo_longi,
        Hypo_lati,
        Hypo_depth,
        strike,
        dip,
        Fault_length,
        Fault_width,
        shypo=None,
        dhypo=None,
        dx=0.5,
        dy=0.5,
):
    """
    根据震源破裂起始点、断层几何参数以及破裂点在断层上的相对位置，
    建立矩形有限断层网格，并返回每个网格节点的：

        1. 经度矩阵
        2. 纬度矩阵
        3. 深度矩阵

    ========================================================================
    一、输入参数定义
    ========================================================================

    Hypo_longi : float
        破裂起始点经度，单位：度。

    Hypo_lati : float
        破裂起始点纬度，单位：度。

    Hypo_depth : float
        破裂起始点深度，单位：km。
        深度以地表为 0 km，向下为正。

    strike : float
        断层走向角，单位：度。

        定义：
            正北 = 0°
            正东 = 90°
            正南 = 180°
            正西 = 270°

        即从正北方向开始顺时针旋转。

    dip : float
        断层倾角，单位：度。

        0°  = 水平断层
        90° = 垂直断层

    Fault_length : float
        断层沿走向长度，单位：km。

    Fault_width : float
        断层沿倾向宽度，单位：km。

    shypo : float
        破裂起始点相对于“断层上缘中心点”的沿走向位置，单位：km。

        定义：
            上缘中心点 = 0
            沿 strike 方向为正
            沿 strike + 180° 方向为负

        因此理论上：
            -Fault_length / 2 <= shypo <= Fault_length / 2

    dhypo : float
        破裂起始点相对于断层上缘的沿倾向距离，单位：km。

        定义：
            断层上缘 = 0
            沿下倾方向为正

        因此理论上：
            0 <= dhypo <= Fault_width

    dx : float
        用户期望的沿走向子断层尺寸，单位：km。

    dy : float
        用户期望的沿倾向子断层尺寸，单位：km。


    ========================================================================
    二、网格划分规则
    ========================================================================

    沿走向子断层数量：

        Nx = Fault_length / dx

    如果不能整除，则对子断层数量进行通常意义的“四舍五入”。

    例如：

        Fault_length = 37 km
        dx = 5 km

        37 / 5 = 7.4

        因此：
            Nx = 7

    为保证最终断层总长度仍严格等于 37 km，
    实际子断层长度重新计算为：

        actual_dx = 37 / 7 = 5.285714 km


    同理，沿倾向采用：

        actual_dy = Fault_width / Ny


    ========================================================================
    三、输出矩阵方向
    ========================================================================

    输出矩阵大小：

        (Ny + 1, Nx + 1)

    因为输出的是“网格节点坐标”。

    矩阵：

        列方向 ---> 沿走向
        行方向 ---> 沿倾向

    即：

        [上缘左端 -------- 上缘中心 -------- 上缘右端]
               ↓
               ↓  下倾方向
               ↓
        [下缘左端 ------------------------- 下缘右端]


    ========================================================================
    四、地表出露处理
    ========================================================================

    如果根据给定的：

        Hypo_depth
        dip
        dhypo

    计算出的断层顶部深度小于 0 km，则说明断层已经出露地表。

    此时程序将整个断层沿竖直方向向下平移：

        shift = -minimum_depth

    从而保证：

        minimum_depth = 0 km

    注意：

    为保持震源起始点与断层之间的相对位置不变，
    破裂起始点的最终深度也必须同步增加 shift。

    经纬度坐标不发生变化。


    ========================================================================
    五、返回结果
    ========================================================================

    返回字典 result，其中主要包括：

        result["longitude_matrix"]
        result["latitude_matrix"]
        result["depth_matrix"]

    以及网格数、实际网格尺寸、下移距离等辅助信息。
    """

    # =====================================================================
    # 0. 输入参数检查
    # =====================================================================

    if shypo is None:
        shypo = 0.0

    if dhypo is None:
        dhypo = 0.57 * Fault_width

    if Fault_length <= 0:
        raise ValueError("Fault_length 必须大于 0 km。")

    if Fault_width <= 0:
        raise ValueError("Fault_width 必须大于 0 km。")

    if dx <= 0:
        raise ValueError("dx 必须大于 0 km。")

    if dy <= 0:
        raise ValueError("dy 必须大于 0 km。")

    if Hypo_depth < 0:
        raise ValueError("输入的 Hypo_depth 不能小于 0 km。")

    if not (-90.0 <= Hypo_lati <= 90.0):
        raise ValueError("Hypo_lati 必须位于 -90° 到 90° 之间。")

    if not (0.0 <= dip <= 90.0):
        raise ValueError("dip 必须位于 0° 到 90° 之间。")

    # ---------------------------------------------------------
    # 检查 shypo 是否位于断层长度范围内
    # ---------------------------------------------------------
    if shypo < -Fault_length / 2.0 or shypo > Fault_length / 2.0:
        raise ValueError(
            f"shypo = {shypo:.4f} km 超出断层范围。\n"
            f"合法范围应为："
            f"[{-Fault_length / 2.0:.4f}, "
            f"{Fault_length / 2.0:.4f}] km"
        )

    # ---------------------------------------------------------
    # 检查 dhypo 是否位于断层宽度范围内
    # ---------------------------------------------------------
    if dhypo < 0.0 or dhypo > Fault_width:
        raise ValueError(
            f"dhypo = {dhypo:.4f} km 超出断层范围。\n"
            f"合法范围应为：[0, {Fault_width:.4f}] km"
        )

    # 将 strike 统一到 [0°, 360°)
    strike = strike % 360.0

    # 转换为弧度
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)

    # =====================================================================
    # 1. 根据用户给定 dx、dy 计算子断层数量
    # =====================================================================

    # 沿走向子断层数量
    nx_float = Fault_length / dx
    nx = round_half_up_positive(nx_float)

    # 沿倾向子断层数量
    ny_float = Fault_width / dy
    ny = round_half_up_positive(ny_float)

    # 防止极端情况下四舍五入为 0
    nx = max(nx, 1)
    ny = max(ny, 1)

    # =====================================================================
    # 2. 重新计算实际使用的子断层尺寸
    #
    # 目的：
    # 保证最终断层总长度、总宽度严格等于输入的定标率结果。
    # =====================================================================

    actual_dx = Fault_length / nx
    actual_dy = Fault_width / ny

    # =====================================================================
    # 3. 建立沿走向和沿倾向的节点位置
    # =====================================================================

    # ---------------------------------------------------------------------
    # 沿走向坐标
    #
    # 断层上缘中心定义为 s = 0
    #
    # 左端：
    #     s = -L/2
    #
    # 右端：
    #     s = +L/2
    #
    # 正方向为 strike 方向。
    # ---------------------------------------------------------------------
    s_nodes = np.linspace(
        -Fault_length / 2.0,
        Fault_length / 2.0,
        nx + 1
    )

    # ---------------------------------------------------------------------
    # 沿倾向坐标
    #
    # 上缘：
    #     d = 0
    #
    # 下缘：
    #     d = Fault_width
    #
    # 正方向为下倾方向。
    # ---------------------------------------------------------------------
    d_nodes = np.linspace(
        0.0,
        Fault_width,
        ny + 1
    )

    # ---------------------------------------------------------------------
    # 建立二维矩阵
    #
    # S_matrix：
    #     每一个网格节点的沿走向位置
    #
    # D_matrix：
    #     每一个网格节点的沿倾向位置
    #
    # shape：
    #     (ny + 1, nx + 1)
    #
    # 行方向：沿倾向
    # 列方向：沿走向
    # ---------------------------------------------------------------------
    S_matrix, D_matrix = np.meshgrid(
        s_nodes,
        d_nodes
    )

    # =====================================================================
    # 4. 计算每个节点相对于破裂起始点的位置
    # =====================================================================

    # 输入破裂点的断层局部坐标为：
    #
    #     s = shypo
    #     d = dhypo
    #
    # 因此任意网格点相对于破裂点：
    #
    #     Δs = s - shypo
    #     Δd = d - dhypo
    #
    delta_s = S_matrix - shypo
    delta_d = D_matrix - dhypo

    # =====================================================================
    # 5. 将沿走向、沿倾向距离转换为东、北、深度方向距离
    # =====================================================================

    # ---------------------------------------------------------------------
    # 沿走向单位向量
    #
    # strike = 0°：
    #     指向北
    #
    # strike = 90°：
    #     指向东
    #
    # 因此：
    #
    #     East  = Δs * sin(strike)
    #     North = Δs * cos(strike)
    # ---------------------------------------------------------------------

    east_from_strike = delta_s * np.sin(strike_rad)
    north_from_strike = delta_s * np.cos(strike_rad)

    # ---------------------------------------------------------------------
    # 下倾方向定义为：
    #
    #     strike + 90°
    #
    # 即采用右手定则。
    #
    # 沿倾向距离 Δd 在水平方向上的投影为：
    #
    #     Δd_horizontal = Δd * cos(dip)
    #
    # 在深度方向上的投影为：
    #
    #     Δz = Δd * sin(dip)
    #
    # 下倾方向 strike + 90° 对应：
    #
    # East:
    #     cos(strike)
    #
    # North:
    #     -sin(strike)
    #
    # ---------------------------------------------------------------------

    horizontal_down_dip = delta_d * np.cos(dip_rad)

    east_from_dip = (
        horizontal_down_dip
        * np.cos(strike_rad)
    )

    north_from_dip = (
        -horizontal_down_dip
        * np.sin(strike_rad)
    )

    # ---------------------------------------------------------------------
    # 最终相对于震源破裂起始点的水平位移
    # ---------------------------------------------------------------------
    east_offset_km = (
        east_from_strike
        +
        east_from_dip
    )

    north_offset_km = (
        north_from_strike
        +
        north_from_dip
    )

    # =====================================================================
    # 6. 将局部水平坐标转换为经纬度
    # =====================================================================

    longitude_matrix, latitude_matrix = local_offset_to_lonlat(
        Hypo_longi,
        Hypo_lati,
        east_offset_km,
        north_offset_km
    )

    # =====================================================================
    # 7. 计算原始深度矩阵
    # =====================================================================

    # 深度向下为正，因此：
    #
    # z = Hypo_depth + Δd * sin(dip)
    #
    # 当网格点位于破裂起始点上倾方向时：
    #     Δd < 0
    #
    # 深度将减小。
    #
    # 当网格点位于破裂起始点下倾方向时：
    #     Δd > 0
    #
    # 深度将增大。
    #
    raw_depth_matrix = (
        Hypo_depth
        +
        delta_d * np.sin(dip_rad)
    )

    # 原始最浅深度
    minimum_raw_depth = float(np.min(raw_depth_matrix))

    # =====================================================================
    # 8. 判断断层是否出露地表
    # =====================================================================

    if minimum_raw_depth < 0.0:

        # -------------------------------------------------------------
        # 如果最浅深度小于 0，说明部分断层位于地表以上。
        #
        # 整个断层向下平移：
        #
        #     shift = -minimum_raw_depth
        #
        # 使得新的最浅深度恰好等于 0 km。
        # -------------------------------------------------------------
        depth_shift_km = -minimum_raw_depth

        depth_matrix = (
            raw_depth_matrix
            +
            depth_shift_km
        )

        surface_exposed = True

        # -------------------------------------------------------------
        # 破裂起始点也必须同步向下移动，
        # 否则它与断层之间原有的相对位置会发生变化。
        # -------------------------------------------------------------
        final_hypo_depth = (
            Hypo_depth
            +
            depth_shift_km
        )

        warning_message = (
            "\n"
            "============================================================\n"
            "警告：当前断层几何参数导致断层出露地表！\n"
            "============================================================\n"
            f"输入破裂起始点深度       = {Hypo_depth:.6f} km\n"
            f"原始断层最浅深度         = {minimum_raw_depth:.6f} km\n"
            f"断层超出地表距离         = {-minimum_raw_depth:.6f} km\n"
            "\n"
            "为保证所有断层节点深度均不小于 0 km，\n"
            "程序已将整个断层沿竖直方向整体向下平移。\n"
            "\n"
            f"整体向下平移距离         = {depth_shift_km:.6f} km\n"
            f"平移后断层最浅深度       = {np.min(depth_matrix):.6f} km\n"
            f"平移后破裂起始点深度     = {final_hypo_depth:.6f} km\n"
            "============================================================\n"
        )

        warnings.warn(warning_message)

        print(warning_message)

    else:

        # 不需要进行任何平移
        depth_shift_km = 0.0

        depth_matrix = raw_depth_matrix.copy()

        surface_exposed = False

        final_hypo_depth = Hypo_depth

    # =====================================================================
    # 9. 计算断层上缘中心点坐标
    #
    # 上缘中心的局部断层坐标：
    #
    #     s = 0
    #     d = 0
    #
    # 因此相对于破裂点：
    #
    #     Δs = -shypo
    #     Δd = -dhypo
    # =====================================================================

    top_delta_s = -shypo
    top_delta_d = -dhypo

    # 沿走向产生的水平偏移
    top_east_strike = (
        top_delta_s
        * np.sin(strike_rad)
    )

    top_north_strike = (
        top_delta_s
        * np.cos(strike_rad)
    )

    # 沿倾向产生的水平偏移
    top_horizontal_dip = (
        top_delta_d
        * np.cos(dip_rad)
    )

    top_east_dip = (
        top_horizontal_dip
        * np.cos(strike_rad)
    )

    top_north_dip = (
        -top_horizontal_dip
        * np.sin(strike_rad)
    )

    # 上缘中心相对于破裂点的东西、南北偏移
    top_east = (
        top_east_strike
        +
        top_east_dip
    )

    top_north = (
        top_north_strike
        +
        top_north_dip
    )

    # 转成经纬度
    top_center_lon, top_center_lat = local_offset_to_lonlat(
        Hypo_longi,
        Hypo_lati,
        top_east,
        top_north
    )

    # 上缘深度
    top_center_raw_depth = (
        Hypo_depth
        -
        dhypo * np.sin(dip_rad)
    )

    # 加上可能存在的整体下移
    top_center_depth = (
        top_center_raw_depth
        +
        depth_shift_km
    )

    # =====================================================================
    # 10. 输出计算信息
    # =====================================================================

    print("\n")
    print("============================================================")
    print("              有限断层网格生成结果")
    print("============================================================")

    print(f"断层走向 strike              = {strike:.4f}°")
    print(f"断层倾角 dip                 = {dip:.4f}°")

    print("------------------------------------------------------------")

    print(f"断层长度                     = {Fault_length:.6f} km")
    print(f"断层宽度                     = {Fault_width:.6f} km")

    print("------------------------------------------------------------")

    print(f"用户输入 dx                  = {dx:.6f} km")
    print(f"用户输入 dy                  = {dy:.6f} km")

    print(f"Fault_length / dx            = {nx_float:.6f}")
    print(f"Fault_width  / dy            = {ny_float:.6f}")

    print("------------------------------------------------------------")

    print(f"四舍五入后沿走向子块数 Nx    = {nx}")
    print(f"四舍五入后沿倾向子块数 Ny    = {ny}")

    print(f"实际沿走向子块尺寸 actual_dx = {actual_dx:.6f} km")
    print(f"实际沿倾向子块尺寸 actual_dy = {actual_dy:.6f} km")

    print("------------------------------------------------------------")

    print(
        f"网格节点矩阵尺寸             = "
        f"{ny + 1} × {nx + 1}"
    )

    print(
        f"子断层数量                   = "
        f"{ny} × {nx} = {ny * nx}"
    )

    print("------------------------------------------------------------")

    print(
        f"输入破裂点："
        f"Lon={Hypo_longi:.6f}°, "
        f"Lat={Hypo_lati:.6f}°, "
        f"Depth={Hypo_depth:.6f} km"
    )

    print(
        f"破裂点相对位置："
        f"shypo={shypo:.6f} km, "
        f"dhypo={dhypo:.6f} km"
    )

    print("------------------------------------------------------------")

    print(
        f"断层上缘中心："
        f"Lon={float(top_center_lon):.6f}°, "
        f"Lat={float(top_center_lat):.6f}°, "
        f"Depth={top_center_depth:.6f} km"
    )

    print("------------------------------------------------------------")

    if surface_exposed:
        print("地表出露状态                 = 是")
        print(
            f"断层整体下移距离             = "
            f"{depth_shift_km:.6f} km"
        )
        print(
            f"最终破裂起始点深度           = "
            f"{final_hypo_depth:.6f} km"
        )
    else:
        print("地表出露状态                 = 否")
        print("断层整体下移距离             = 0.000000 km")

    print("============================================================")
    print("\n")

    # =====================================================================
    # 11. 将所有结果打包到字典中
    # =====================================================================

    result = {

        # -------------------------------------------------------------
        # 用户最主要需要的三个矩阵
        # -------------------------------------------------------------
        "longitude_matrix": longitude_matrix,
        "latitude_matrix": latitude_matrix,
        "depth_matrix": depth_matrix,

        # -------------------------------------------------------------
        # 原始深度，用于检查是否进行了地表修正
        # -------------------------------------------------------------
        "raw_depth_matrix": raw_depth_matrix,

        # -------------------------------------------------------------
        # 断层局部坐标矩阵
        # -------------------------------------------------------------
        "along_strike_matrix": S_matrix,
        "down_dip_matrix": D_matrix,

        # -------------------------------------------------------------
        # 水平相对坐标
        # -------------------------------------------------------------
        "east_offset_matrix": east_offset_km,
        "north_offset_matrix": north_offset_km,

        # -------------------------------------------------------------
        # 网格信息
        # -------------------------------------------------------------
        "nx": nx,
        "ny": ny,

        "actual_dx": actual_dx,
        "actual_dy": actual_dy,

        # -------------------------------------------------------------
        # 地表出露信息
        # -------------------------------------------------------------
        "surface_exposed": surface_exposed,

        "depth_shift_km": depth_shift_km,

        # -------------------------------------------------------------
        # 破裂起始点最终信息
        # -------------------------------------------------------------
        "original_hypo_depth": Hypo_depth,
        "final_hypo_depth": final_hypo_depth,

        # -------------------------------------------------------------
        # 上缘中心信息
        # -------------------------------------------------------------
        "top_center_lon": float(top_center_lon),
        "top_center_lat": float(top_center_lat),
        "top_center_depth": float(top_center_depth),

        # -------------------------------------------------------------
        # 断层参数
        # -------------------------------------------------------------
        "strike": strike,
        "dip": dip,

        "Fault_length": Fault_length,
        "Fault_width": Fault_width,

        "shypo": shypo,
        "dhypo": dhypo,
    }

    return result


# =============================================================================
# 全局常数
# =============================================================================

# 地球平均半径，单位：km
EARTH_RADIUS_KM = 6371.0088


# =============================================================================
# 辅助函数 1：正数通常意义的四舍五入
# =============================================================================
def round_half_up_positive(x):
    """
    对正数执行通常意义上的四舍五入。

    例如：
        6.4 -> 6
        6.5 -> 7
        6.6 -> 7

    不直接使用 Python 的 round()，因为 Python round() 采用银行家舍入。
    """
    return int(math.floor(x + 0.5))


# =============================================================================
# 辅助函数 2：由经纬度反算相对于参考点的局部 East / North 坐标
# =============================================================================
def lonlat_to_local_offset(
        origin_lon,
        origin_lat,
        lon,
        lat
):
    """
    根据球面地球模型，将经纬度反算为相对于参考点的：

        East  : 向东距离，km
        North : 向北距离，km

    这里故意采用“反算”方法，而不是直接读取主程序中已经计算好的
    east_offset_matrix 和 north_offset_matrix。

    这样可以独立检查：

        局部坐标 -> 经纬度

    这一转换过程本身是否正确。

    参数
    ----------
    origin_lon : float
        参考点经度，单位：度。
        在本程序中使用 Hypocenter 经度。

    origin_lat : float
        参考点纬度，单位：度。
        在本程序中使用 Hypocenter 纬度。

    lon : ndarray
        待转换点经度。

    lat : ndarray
        待转换点纬度。

    返回
    -------
    east_km : ndarray
        向东距离，km。

    north_km : ndarray
        向北距离，km。
    """

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    # 参考点转弧度
    lat1 = np.deg2rad(origin_lat)
    lon1 = np.deg2rad(origin_lon)

    # 目标点转弧度
    lat2 = np.deg2rad(lat)
    lon2 = np.deg2rad(lon)

    # 经度差
    dlon = lon2 - lon1

    # ---------------------------------------------------------------------
    # 1. 计算球面大圆距离
    # ---------------------------------------------------------------------
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    # 防止浮点误差使 a 略微超出 [0, 1]
    a = np.clip(a, 0.0, 1.0)

    angular_distance = (
        2.0
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1.0 - a)
        )
    )

    distance_km = (
        EARTH_RADIUS_KM
        * angular_distance
    )

    # ---------------------------------------------------------------------
    # 2. 计算从参考点指向目标点的初始方位角
    #
    # 方位角定义：
    #
    #     北 = 0°
    #     东 = 90°
    #     南 = 180°
    #     西 = 270°
    # ---------------------------------------------------------------------
    y = (
        np.sin(dlon)
        * np.cos(lat2)
    )

    x = (
        np.cos(lat1) * np.sin(lat2)
        -
        np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.arctan2(y, x)

    # ---------------------------------------------------------------------
    # 3. 转换成局部 East / North
    # ---------------------------------------------------------------------
    east_km = (
        distance_km
        * np.sin(bearing)
    )

    north_km = (
        distance_km
        * np.cos(bearing)
    )

    return east_km, north_km


# =============================================================================
# 辅助函数 3：计算两个角度之间的最小差值
# =============================================================================
def angle_difference_deg(angle1, angle2):
    """
    返回两个角度之间的最小有符号差值，范围 [-180°, 180°)。

    例如：

        angle1 = 359°
        angle2 = 1°

    实际差值应该是 -2°，而不是 358°。
    """

    return (
        (angle1 - angle2 + 180.0)
        % 360.0
        - 180.0
    )


# =============================================================================
# 辅助函数 4：打印数值检查结果
# =============================================================================
def print_numeric_check(
        name,
        calculated,
        expected,
        tolerance,
        unit=""
):
    """
    打印一个数值检查结果。

    如果：

        |calculated - expected| <= tolerance

    则显示 PASS，否则显示 FAIL。
    """

    error = abs(calculated - expected)

    passed = error <= tolerance

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {name:<32s} "
        f"计算值 = {calculated:12.6f} {unit:<4s}   "
        f"理论值 = {expected:12.6f} {unit:<4s}   "
        f"误差 = {error:.3e}"
    )

    return passed


# =============================================================================
# 主函数：可视化 + 几何准确性检查
# =============================================================================
def visualize_and_validate_fault(
        result,
        Hypo_longi,
        Hypo_lati,
        requested_dx,
        requested_dy,
        save_figures=False,
        output_dir="fault_geometry_check",
        show_cell_index=True
):
    """
    对 build_fault_grid() 生成的矩形有限断层进行可视化和数值验证。

    ========================================================================
    输入
    ========================================================================

    result : dict
        build_fault_grid() 返回的 result。

    Hypo_longi : float
        输入的破裂起始点经度，单位：度。

    Hypo_lati : float
        输入的破裂起始点纬度，单位：度。

    requested_dx : float
        用户原始设定的沿走向网格尺寸，单位：km。

    requested_dy : float
        用户原始设定的沿倾向网格尺寸，单位：km。

    save_figures : bool
        是否保存图片。

        False：
            只显示，不保存。

        True：
            自动保存为 PNG，300 dpi。

    output_dir : str
        图片保存目录。

    show_cell_index : bool
        是否在局部断层网格图中标注子断层编号。

        当网格数量太多时，为避免图片过于拥挤，
        程序会自动关闭编号。


    ========================================================================
    输出图
    ========================================================================

    图 1：
        经度-纬度地表投影
        检查走向、倾向方向、上下缘、网格形状。

    图 2：
        East-North-Depth 三维断层
        检查断层三维空间几何和倾斜方向。

    图 3：
        断层局部 s-d 坐标
        检查 shypo、dhypo、Nx、Ny、dx、dy。

    图 4：
        沿倾向深度剖面
        检查 dip、上下缘深度、W*sin(dip)、W*cos(dip)
        以及地表出露后的整体下移。


    ========================================================================
    重要说明
    ========================================================================

    本程序不会仅仅读取主程序中的 East/North 矩阵进行检查。

    它首先根据主程序最终输出的：

        longitude_matrix
        latitude_matrix

    重新反算 East/North。

    因此可以检查：

        局部断层几何
             ↓
        经纬度转换
             ↓
        经纬度反算局部坐标

    整个过程是否自洽。
    """

    # =====================================================================
    # 1. 读取主程序结果
    # =====================================================================

    longitude = np.asarray(
        result["longitude_matrix"],
        dtype=float
    )

    latitude = np.asarray(
        result["latitude_matrix"],
        dtype=float
    )

    depth = np.asarray(
        result["depth_matrix"],
        dtype=float
    )

    raw_depth = np.asarray(
        result["raw_depth_matrix"],
        dtype=float
    )

    S = np.asarray(
        result["along_strike_matrix"],
        dtype=float
    )

    D = np.asarray(
        result["down_dip_matrix"],
        dtype=float
    )

    nx = int(result["nx"])
    ny = int(result["ny"])

    actual_dx = float(result["actual_dx"])
    actual_dy = float(result["actual_dy"])

    strike = float(result["strike"]) % 360.0
    dip = float(result["dip"])

    Fault_length = float(
        result["Fault_length"]
    )

    Fault_width = float(
        result["Fault_width"]
    )

    shypo = float(result["shypo"])
    dhypo = float(result["dhypo"])

    final_hypo_depth = float(
        result["final_hypo_depth"]
    )

    top_center_depth = float(
        result["top_center_depth"]
    )

    top_center_lon = float(
        result["top_center_lon"]
    )

    top_center_lat = float(
        result["top_center_lat"]
    )

    depth_shift_km = float(
        result["depth_shift_km"]
    )

    surface_exposed = bool(
        result["surface_exposed"]
    )

    # 角度转弧度
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)

    # =====================================================================
    # 2. 从“最终经纬度矩阵”独立反算 East / North
    #
    # 这一步非常重要：
    #
    # 不直接相信主程序中的 east_offset_matrix /
    # north_offset_matrix，而是重新从经纬度反推。
    # =====================================================================

    east, north = lonlat_to_local_offset(
        origin_lon=Hypo_longi,
        origin_lat=Hypo_lati,
        lon=longitude,
        lat=latitude
    )

    # =====================================================================
    # 3. 自动数值检查
    # =====================================================================

    print("\n")
    print("=" * 100)
    print("                  有限断层几何准确性自动检查")
    print("=" * 100)

    all_checks = []

    # ---------------------------------------------------------------------
    # 检查 A：矩阵尺寸
    # ---------------------------------------------------------------------
    expected_shape = (
        ny + 1,
        nx + 1
    )

    matrix_shape_pass = (
        longitude.shape == expected_shape
        and latitude.shape == expected_shape
        and depth.shape == expected_shape
    )

    print(
        f"[{'PASS' if matrix_shape_pass else 'FAIL'}] "
        f"矩阵尺寸                         "
        f"实际 = {longitude.shape}，"
        f"理论 = {expected_shape}"
    )

    all_checks.append(matrix_shape_pass)

    # ---------------------------------------------------------------------
    # 检查 B：网格数量的四舍五入
    # ---------------------------------------------------------------------
    expected_nx = max(
        round_half_up_positive(
            Fault_length / requested_dx
        ),
        1
    )

    expected_ny = max(
        round_half_up_positive(
            Fault_width / requested_dy
        ),
        1
    )

    nx_pass = nx == expected_nx
    ny_pass = ny == expected_ny

    print(
        f"[{'PASS' if nx_pass else 'FAIL'}] "
        f"沿走向网格数量 Nx                 "
        f"计算值 = {nx}，理论值 = {expected_nx}"
    )

    print(
        f"[{'PASS' if ny_pass else 'FAIL'}] "
        f"沿倾向网格数量 Ny                 "
        f"计算值 = {ny}，理论值 = {expected_ny}"
    )

    all_checks.extend([
        nx_pass,
        ny_pass
    ])

    # ---------------------------------------------------------------------
    # 检查 C：实际网格尺寸
    # ---------------------------------------------------------------------
    all_checks.append(
        print_numeric_check(
            "实际 dx",
            actual_dx,
            Fault_length / nx,
            1.0e-10,
            "km"
        )
    )

    all_checks.append(
        print_numeric_check(
            "实际 dy",
            actual_dy,
            Fault_width / ny,
            1.0e-10,
            "km"
        )
    )

    # =====================================================================
    # 4. 从最终坐标直接计算网格每条边的三维长度
    # =====================================================================

    # ---------------------------------------------------------------------
    # 沿走向相邻节点差值
    # ---------------------------------------------------------------------
    delta_e_strike = np.diff(
        east,
        axis=1
    )

    delta_n_strike = np.diff(
        north,
        axis=1
    )

    delta_z_strike = np.diff(
        depth,
        axis=1
    )

    # 每一个沿走向网格边的三维长度
    strike_segment_length = np.sqrt(
        delta_e_strike ** 2
        +
        delta_n_strike ** 2
        +
        delta_z_strike ** 2
    )

    # ---------------------------------------------------------------------
    # 沿倾向相邻节点差值
    # ---------------------------------------------------------------------
    delta_e_dip = np.diff(
        east,
        axis=0
    )

    delta_n_dip = np.diff(
        north,
        axis=0
    )

    delta_z_dip = np.diff(
        depth,
        axis=0
    )

    # 每一个沿倾向网格边的三维长度
    dip_segment_length = np.sqrt(
        delta_e_dip ** 2
        +
        delta_n_dip ** 2
        +
        delta_z_dip ** 2
    )

    # 最大网格误差
    max_dx_error = np.max(
        np.abs(
            strike_segment_length
            -
            actual_dx
        )
    )

    max_dy_error = np.max(
        np.abs(
            dip_segment_length
            -
            actual_dy
        )
    )

    dx_grid_pass = (
        max_dx_error <= 1.0e-6
    )

    dy_grid_pass = (
        max_dy_error <= 1.0e-6
    )

    print(
        f"[{'PASS' if dx_grid_pass else 'FAIL'}] "
        f"所有沿走向网格边长度              "
        f"理论 dx = {actual_dx:.6f} km，"
        f"最大误差 = {max_dx_error:.3e} km"
    )

    print(
        f"[{'PASS' if dy_grid_pass else 'FAIL'}] "
        f"所有沿倾向网格边长度              "
        f"理论 dy = {actual_dy:.6f} km，"
        f"最大误差 = {max_dy_error:.3e} km"
    )

    all_checks.extend([
        dx_grid_pass,
        dy_grid_pass
    ])

    # =====================================================================
    # 5. 检查整个断层长度和宽度
    # =====================================================================

    # ---------------------------------------------------------------------
    # 沿走向总向量：
    #
    # 使用断层上缘左端 -> 上缘右端
    # ---------------------------------------------------------------------
    strike_vector = np.array([
        east[0, -1] - east[0, 0],
        north[0, -1] - north[0, 0],
        depth[0, -1] - depth[0, 0]
    ])

    calculated_length = np.linalg.norm(
        strike_vector
    )

    # ---------------------------------------------------------------------
    # 沿倾向总向量：
    #
    # 使用左上角 -> 左下角
    # ---------------------------------------------------------------------
    dip_vector = np.array([
        east[-1, 0] - east[0, 0],
        north[-1, 0] - north[0, 0],
        depth[-1, 0] - depth[0, 0]
    ])

    calculated_width = np.linalg.norm(
        dip_vector
    )

    all_checks.append(
        print_numeric_check(
            "断层总长度",
            calculated_length,
            Fault_length,
            1.0e-6,
            "km"
        )
    )

    all_checks.append(
        print_numeric_check(
            "断层总宽度",
            calculated_width,
            Fault_width,
            1.0e-6,
            "km"
        )
    )

    # =====================================================================
    # 6. 检查 strike
    # =====================================================================

    calculated_strike = (
        np.rad2deg(
            np.arctan2(
                strike_vector[0],
                strike_vector[1]
            )
        )
        + 360.0
    ) % 360.0

    strike_error = abs(
        angle_difference_deg(
            calculated_strike,
            strike
        )
    )

    strike_pass = (
        strike_error <= 1.0e-6
    )

    print(
        f"[{'PASS' if strike_pass else 'FAIL'}] "
        f"断层走向 strike                    "
        f"计算值 = {calculated_strike:.6f}°，"
        f"输入值 = {strike:.6f}°，"
        f"误差 = {strike_error:.3e}°"
    )

    all_checks.append(strike_pass)

    # =====================================================================
    # 7. 检查下倾方向是否等于 strike + 90°
    # =====================================================================

    calculated_dip_azimuth = (
        np.rad2deg(
            np.arctan2(
                dip_vector[0],
                dip_vector[1]
            )
        )
        + 360.0
    ) % 360.0

    theoretical_dip_azimuth = (
        strike + 90.0
    ) % 360.0

    dip_azimuth_error = abs(
        angle_difference_deg(
            calculated_dip_azimuth,
            theoretical_dip_azimuth
        )
    )

    dip_azimuth_pass = (
        dip_azimuth_error <= 1.0e-6
    )

    print(
        f"[{'PASS' if dip_azimuth_pass else 'FAIL'}] "
        f"下倾方位角                         "
        f"计算值 = {calculated_dip_azimuth:.6f}°，"
        f"理论值 = {theoretical_dip_azimuth:.6f}°"
    )

    all_checks.append(
        dip_azimuth_pass
    )

    # =====================================================================
    # 8. 检查 dip
    # =====================================================================

    horizontal_width = np.sqrt(
        dip_vector[0] ** 2
        +
        dip_vector[1] ** 2
    )

    vertical_width = dip_vector[2]

    calculated_dip = np.rad2deg(
        np.arctan2(
            vertical_width,
            horizontal_width
        )
    )

    all_checks.append(
        print_numeric_check(
            "断层倾角 dip",
            calculated_dip,
            dip,
            1.0e-6,
            "deg"
        )
    )

    # =====================================================================
    # 9. 检查 W*cos(dip) 和 W*sin(dip)
    # =====================================================================

    theoretical_horizontal_width = (
        Fault_width
        * np.cos(dip_rad)
    )

    theoretical_vertical_width = (
        Fault_width
        * np.sin(dip_rad)
    )

    all_checks.append(
        print_numeric_check(
            "上下缘水平距离 W*cos(dip)",
            horizontal_width,
            theoretical_horizontal_width,
            1.0e-6,
            "km"
        )
    )

    all_checks.append(
        print_numeric_check(
            "上下缘深度差 W*sin(dip)",
            vertical_width,
            theoretical_vertical_width,
            1.0e-6,
            "km"
        )
    )

    # =====================================================================
    # 10. 从最终坐标反算 shypo / dhypo
    #
    # 这是一个非常重要的检查。
    #
    # 不能只检查 Hypocenter 是否“看起来在断层里面”，
    # 而是重新计算：
    #
    #     Hypocenter 相对于断层上缘中心：
    #
    #         沿走向距离
    #         沿倾向距离
    #
    # 看它们是否真的等于用户输入的 shypo / dhypo。
    # =====================================================================

    # 将上缘中心点经纬度反算为相对于 Hypocenter 的 E / N
    top_center_east, top_center_north = (
        lonlat_to_local_offset(
            origin_lon=Hypo_longi,
            origin_lat=Hypo_lati,
            lon=top_center_lon,
            lat=top_center_lat
        )
    )

    top_center_east = float(
        top_center_east
    )

    top_center_north = float(
        top_center_north
    )

    # 从上缘中心指向 Hypocenter 的三维向量
    hypo_from_top_vector = np.array([
        -top_center_east,
        -top_center_north,
        final_hypo_depth
        -
        top_center_depth
    ])

    # ---------------------------------------------------------------------
    # 沿走向三维单位向量
    # ---------------------------------------------------------------------
    strike_unit_vector = np.array([
        np.sin(strike_rad),
        np.cos(strike_rad),
        0.0
    ])

    # ---------------------------------------------------------------------
    # 沿下倾方向三维单位向量
    #
    # East:
    #     cos(dip)*cos(strike)
    #
    # North:
    #     -cos(dip)*sin(strike)
    #
    # Depth:
    #     sin(dip)
    # ---------------------------------------------------------------------
    dip_unit_vector = np.array([
        np.cos(dip_rad)
        * np.cos(strike_rad),

        -np.cos(dip_rad)
        * np.sin(strike_rad),

        np.sin(dip_rad)
    ])

    # 三维投影得到 shypo / dhypo
    calculated_shypo = np.dot(
        hypo_from_top_vector,
        strike_unit_vector
    )

    calculated_dhypo = np.dot(
        hypo_from_top_vector,
        dip_unit_vector
    )

    all_checks.append(
        print_numeric_check(
            "Hypocenter 沿走向位置 shypo",
            calculated_shypo,
            shypo,
            1.0e-6,
            "km"
        )
    )

    all_checks.append(
        print_numeric_check(
            "Hypocenter 沿倾向位置 dhypo",
            calculated_dhypo,
            dhypo,
            1.0e-6,
            "km"
        )
    )

    # =====================================================================
    # 11. 检查地表出露和平移
    # =====================================================================

    minimum_depth = float(
        np.min(depth)
    )

    depth_nonnegative_pass = (
        minimum_depth >= -1.0e-10
    )

    print(
        f"[{'PASS' if depth_nonnegative_pass else 'FAIL'}] "
        f"最终所有节点深度 >= 0              "
        f"最浅深度 = {minimum_depth:.9f} km"
    )

    all_checks.append(
        depth_nonnegative_pass
    )

    minimum_raw_depth = float(
        np.min(raw_depth)
    )

    if surface_exposed:

        expected_shift = (
            -minimum_raw_depth
        )

        all_checks.append(
            print_numeric_check(
                "地表出露后的整体下移量",
                depth_shift_km,
                expected_shift,
                1.0e-10,
                "km"
            )
        )

        all_checks.append(
            print_numeric_check(
                "平移后最浅深度",
                minimum_depth,
                0.0,
                1.0e-10,
                "km"
            )
        )

    else:

        all_checks.append(
            print_numeric_check(
                "未出露时整体下移量",
                depth_shift_km,
                0.0,
                1.0e-10,
                "km"
            )
        )

    # =====================================================================
    # 12. 检查经纬度转换是否与主程序局部坐标一致
    # =====================================================================

    if (
        "east_offset_matrix" in result
        and
        "north_offset_matrix" in result
    ):

        original_east = np.asarray(
            result["east_offset_matrix"],
            dtype=float
        )

        original_north = np.asarray(
            result["north_offset_matrix"],
            dtype=float
        )

        max_east_error = np.max(
            np.abs(
                east - original_east
            )
        )

        max_north_error = np.max(
            np.abs(
                north - original_north
            )
        )

        coordinate_pass = (
            max_east_error <= 1.0e-6
            and
            max_north_error <= 1.0e-6
        )

        print(
            f"[{'PASS' if coordinate_pass else 'FAIL'}] "
            f"经纬度正算/反算一致性              "
            f"East最大误差={max_east_error:.3e} km，"
            f"North最大误差={max_north_error:.3e} km"
        )

        all_checks.append(
            coordinate_pass
        )

    # =====================================================================
    # 13. 最终检查结果
    # =====================================================================

    print("-" * 100)

    if all(all_checks):

        print(
            "最终结果：全部检查 PASS。"
            " 当前断层几何、网格尺寸、走向、倾角和破裂点相对位置均自洽。"
        )

    else:

        failed_number = (
            len(all_checks)
            -
            sum(all_checks)
        )

        print(
            f"最终结果：存在 {failed_number} 项 FAIL，"
            f"请重点检查对应几何计算。"
        )

    print("=" * 100)
    print("\n")

    # =====================================================================
    # 如果需要保存图片，创建文件夹
    # =====================================================================

    if save_figures:

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

    # #####################################################################
    # 图 1
    #
    # 经度 - 纬度地表投影
    # #####################################################################

    fig1 = plt.figure(
        figsize=(10, 8)
    )

    ax1 = fig1.add_subplot(111)

    # ---------------------------------------------------------------------
    # 绘制所有沿走向网格线
    # ---------------------------------------------------------------------
    for i in range(ny + 1):

        ax1.plot(
            longitude[i, :],
            latitude[i, :],
            "-",
            linewidth=0.8,
            color="0.65"
        )

    # ---------------------------------------------------------------------
    # 绘制所有沿倾向网格线
    # ---------------------------------------------------------------------
    for j in range(nx + 1):

        ax1.plot(
            longitude[:, j],
            latitude[:, j],
            "-",
            linewidth=0.8,
            color="0.65"
        )

    # ---------------------------------------------------------------------
    # 上缘
    # ---------------------------------------------------------------------
    ax1.plot(
        longitude[0, :],
        latitude[0, :],
        "-",
        linewidth=2.8,
        color="red",
        label="Upper edge"
    )

    # ---------------------------------------------------------------------
    # 下缘
    # ---------------------------------------------------------------------
    ax1.plot(
        longitude[-1, :],
        latitude[-1, :],
        "-",
        linewidth=2.8,
        color="blue",
        label="Lower edge"
    )

    # ---------------------------------------------------------------------
    # 左边界
    # ---------------------------------------------------------------------
    ax1.plot(
        longitude[:, 0],
        latitude[:, 0],
        "--",
        linewidth=1.8,
        color="black",
        label="Left boundary"
    )

    # ---------------------------------------------------------------------
    # 右边界
    # ---------------------------------------------------------------------
    ax1.plot(
        longitude[:, -1],
        latitude[:, -1],
        "--",
        linewidth=1.8,
        color="black",
        label="Right boundary"
    )

    # ---------------------------------------------------------------------
    # Hypocenter 的地表投影
    #
    # 注意：
    # Hypocenter 不一定恰好位于某个网格节点上，
    # 因为 shypo / dhypo 不一定是 dx / dy 的整数倍。
    # ---------------------------------------------------------------------
    ax1.scatter(
        Hypo_longi,
        Hypo_lati,
        marker="*",
        s=220,
        color="magenta",
        edgecolor="black",
        zorder=10,
        label="Hypocenter projection"
    )

    # ---------------------------------------------------------------------
    # 断层上缘中心点
    # ---------------------------------------------------------------------
    ax1.scatter(
        top_center_lon,
        top_center_lat,
        marker="o",
        s=80,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="Upper-edge center"
    )

    # ---------------------------------------------------------------------
    # 四个角点编号
    # ---------------------------------------------------------------------
    corner_information = [
        (
            longitude[0, 0],
            latitude[0, 0],
            "UL"
        ),
        (
            longitude[0, -1],
            latitude[0, -1],
            "UR"
        ),
        (
            longitude[-1, 0],
            latitude[-1, 0],
            "LL"
        ),
        (
            longitude[-1, -1],
            latitude[-1, -1],
            "LR"
        )
    ]

    for lon_corner, lat_corner, label in corner_information:

        ax1.annotate(
            label,
            xy=(
                lon_corner,
                lat_corner
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10
        )

    # ---------------------------------------------------------------------
    # 经纬度图纵横比例修正
    #
    # 不能直接使用 axis("equal")。
    #
    # 原因：
    #     1° 纬度 ≈ 111 km
    #
    # 但：
    #     1° 经度 ≈ 111*cos(latitude) km
    #
    # 因此需要根据平均纬度调整比例。
    # ---------------------------------------------------------------------
    mean_latitude = np.mean(
        latitude
    )

    cos_latitude = np.cos(
        np.deg2rad(mean_latitude)
    )

    cos_latitude = max(
        abs(cos_latitude),
        1.0e-6
    )

    ax1.set_aspect(
        1.0 / cos_latitude
    )

    ax1.set_xlabel(
        "Longitude (deg)"
    )

    ax1.set_ylabel(
        "Latitude (deg)"
    )

    ax1.set_title(
        "Fault Surface Projection\n"
        f"Strike={strike:.2f} deg, "
        f"Dip={dip:.2f} deg"
    )

    ax1.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    ax1.legend(
        loc="best"
    )

    fig1.tight_layout()

    if save_figures:

        fig1.savefig(
            output_path
            / "01_fault_surface_projection.png",
            dpi=300,
            bbox_inches="tight"
        )

    # #####################################################################
    # 图 2
    #
    # East - North - Depth 三维断层
    #
    # 这张图是真正以 km 为单位检查三维断层几何。
    # #####################################################################

    fig2 = plt.figure(
        figsize=(10, 8)
    )

    ax2 = fig2.add_subplot(
        111,
        projection="3d"
    )

    # ---------------------------------------------------------------------
    # 所有沿走向网格线
    # ---------------------------------------------------------------------
    for i in range(ny + 1):

        ax2.plot(
            east[i, :],
            north[i, :],
            depth[i, :],
            "-",
            linewidth=0.9,
            color="0.55"
        )

    # ---------------------------------------------------------------------
    # 所有沿倾向网格线
    # ---------------------------------------------------------------------
    for j in range(nx + 1):

        ax2.plot(
            east[:, j],
            north[:, j],
            depth[:, j],
            "-",
            linewidth=0.9,
            color="0.55"
        )

    # 上缘
    ax2.plot(
        east[0, :],
        north[0, :],
        depth[0, :],
        "-",
        linewidth=3.0,
        color="red",
        label="Upper edge"
    )

    # 下缘
    ax2.plot(
        east[-1, :],
        north[-1, :],
        depth[-1, :],
        "-",
        linewidth=3.0,
        color="blue",
        label="Lower edge"
    )

    # Hypocenter
    ax2.scatter(
        0.0,
        0.0,
        final_hypo_depth,
        marker="*",
        s=220,
        color="magenta",
        edgecolor="black",
        label="Hypocenter"
    )

    # ---------------------------------------------------------------------
    # 绘制地表 Depth = 0 平面
    # ---------------------------------------------------------------------
    east_margin = max(
        np.ptp(east) * 0.1,
        1.0
    )

    north_margin = max(
        np.ptp(north) * 0.1,
        1.0
    )

    surface_east = np.array([
        [
            np.min(east) - east_margin,
            np.max(east) + east_margin
        ],
        [
            np.min(east) - east_margin,
            np.max(east) + east_margin
        ]
    ])

    surface_north = np.array([
        [
            np.min(north) - north_margin,
            np.min(north) - north_margin
        ],
        [
            np.max(north) + north_margin,
            np.max(north) + north_margin
        ]
    ])

    surface_depth = np.zeros_like(
        surface_east
    )

    ax2.plot_surface(
        surface_east,
        surface_north,
        surface_depth,
        alpha=0.12,
        color="gray"
    )

    ax2.set_xlabel(
        "East (km)"
    )

    ax2.set_ylabel(
        "North (km)"
    )

    ax2.set_zlabel(
        "Depth (km)"
    )

    ax2.set_title(
        "3-D Fault Geometry in Local ENU Coordinates"
    )

    # 深度向下为正，因此让深度轴反向显示：
    #
    # 地表在上
    # 深部在下
    ax2.invert_zaxis()

    # ---------------------------------------------------------------------
    # 尽量按照真实 km 比例显示三维坐标
    # ---------------------------------------------------------------------
    east_range = max(
        np.ptp(east),
        1.0e-6
    )

    north_range = max(
        np.ptp(north),
        1.0e-6
    )

    depth_range = max(
        np.ptp(depth),
        1.0e-6
    )

    ax2.set_box_aspect(
        (
            east_range,
            north_range,
            depth_range
        )
    )

    ax2.view_init(
        elev=25,
        azim=-60
    )

    ax2.legend(
        loc="best"
    )

    fig2.tight_layout()

    if save_figures:

        fig2.savefig(
            output_path
            / "02_fault_3d_geometry.png",
            dpi=300,
            bbox_inches="tight"
        )

    # #####################################################################
    # 图 3
    #
    # 断层自身局部坐标 s-d
    #
    # 这是检查网格划分和 Hypocenter 相对位置最直接的一张图。
    # #####################################################################

    fig3 = plt.figure(
        figsize=(11, 7)
    )

    ax3 = fig3.add_subplot(111)

    # ---------------------------------------------------------------------
    # 绘制沿走向网格线
    # ---------------------------------------------------------------------
    for i in range(ny + 1):

        ax3.plot(
            S[i, :],
            D[i, :],
            "-",
            color="0.55",
            linewidth=0.9
        )

    # ---------------------------------------------------------------------
    # 绘制沿倾向网格线
    # ---------------------------------------------------------------------
    for j in range(nx + 1):

        ax3.plot(
            S[:, j],
            D[:, j],
            "-",
            color="0.55",
            linewidth=0.9
        )

    # 上缘
    ax3.plot(
        S[0, :],
        D[0, :],
        "-",
        linewidth=3.0,
        color="red",
        label="Upper edge"
    )

    # 下缘
    ax3.plot(
        S[-1, :],
        D[-1, :],
        "-",
        linewidth=3.0,
        color="blue",
        label="Lower edge"
    )

    # ---------------------------------------------------------------------
    # 上缘中心
    # ---------------------------------------------------------------------
    ax3.scatter(
        0.0,
        0.0,
        marker="o",
        s=90,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="Upper-edge center (0, 0)"
    )

    # ---------------------------------------------------------------------
    # Hypocenter
    # ---------------------------------------------------------------------
    ax3.scatter(
        shypo,
        dhypo,
        marker="*",
        s=220,
        color="magenta",
        edgecolor="black",
        zorder=10,
        label=(
            f"Hypocenter "
            f"({shypo:.2f}, {dhypo:.2f}) km"
        )
    )

    # ---------------------------------------------------------------------
    # 如果网格数量不是太大，则在每个子断层中心标记编号：
    #
    #     (iy, ix)
    #
    # iy = 沿倾向编号
    # ix = 沿走向编号
    # ---------------------------------------------------------------------
    if (
        show_cell_index
        and
        nx * ny <= 80
    ):

        s_nodes = S[0, :]
        d_nodes = D[:, 0]

        for iy in range(ny):

            for ix in range(nx):

                cell_center_s = (
                    s_nodes[ix]
                    +
                    s_nodes[ix + 1]
                ) / 2.0

                cell_center_d = (
                    d_nodes[iy]
                    +
                    d_nodes[iy + 1]
                ) / 2.0

                ax3.text(
                    cell_center_s,
                    cell_center_d,
                    f"{iy},{ix}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

    ax3.set_xlabel(
        "Along strike, s (km)"
    )

    ax3.set_ylabel(
        "Down dip, d (km)"
    )

    ax3.set_title(
        "Fault Local Grid\n"
        f"Nx={nx}, Ny={ny}, "
        f"actual dx={actual_dx:.4f} km, "
        f"actual dy={actual_dy:.4f} km"
    )

    # d=0 是上缘，因此让上缘显示在图的顶部
    ax3.invert_yaxis()

    # s 和 d 都是 km，因此可以真正使用 equal
    ax3.set_aspect(
        "equal",
        adjustable="box"
    )

    ax3.grid(
        True,
        linestyle=":",
        alpha=0.5
    )

    ax3.legend(
        loc="best"
    )

    fig3.tight_layout()

    if save_figures:

        fig3.savefig(
            output_path
            / "03_fault_local_grid.png",
            dpi=300,
            bbox_inches="tight"
        )

    # #####################################################################
    # 图 4
    #
    # 沿倾向深度剖面
    #
    # 这里同时画：
    #
    #     1. 最终生成坐标反算得到的结果
    #     2. 理论几何关系
    #
    # 如果两条线重合，说明倾角、水平投影、深度计算是正确的。
    # #####################################################################

    fig4 = plt.figure(
        figsize=(10, 7)
    )

    ax4 = fig4.add_subplot(111)

    # ---------------------------------------------------------------------
    # 对每一行节点取平均。
    #
    # 因为同一行属于同一个 down-dip 位置，
    # 沿走向对称平均以后得到该行的几何中心。
    # ---------------------------------------------------------------------
    center_east_each_row = np.mean(
        east,
        axis=1
    )

    center_north_each_row = np.mean(
        north,
        axis=1
    )

    center_depth_each_row = np.mean(
        depth,
        axis=1
    )

    # ---------------------------------------------------------------------
    # 以下倾上缘为原点，计算各行中心相对于上缘中心的水平位移
    # ---------------------------------------------------------------------
    delta_center_east = (
        center_east_each_row
        -
        center_east_each_row[0]
    )

    delta_center_north = (
        center_north_each_row
        -
        center_north_each_row[0]
    )

    # 下倾方向在水平面的单位向量：
    #
    # East  = cos(strike)
    # North = -sin(strike)
    horizontal_dip_unit_east = (
        np.cos(strike_rad)
    )

    horizontal_dip_unit_north = (
        -np.sin(strike_rad)
    )

    # 投影到下倾水平距离
    generated_horizontal_distance = (
        delta_center_east
        * horizontal_dip_unit_east
        +
        delta_center_north
        * horizontal_dip_unit_north
    )

    # ---------------------------------------------------------------------
    # 理论结果
    # ---------------------------------------------------------------------
    d_nodes = D[:, 0]

    theoretical_horizontal_distance = (
        d_nodes
        * np.cos(dip_rad)
    )

    theoretical_depth = (
        top_center_depth
        +
        d_nodes
        * np.sin(dip_rad)
    )

    # ---------------------------------------------------------------------
    # 最终生成结果
    # ---------------------------------------------------------------------
    ax4.plot(
        generated_horizontal_distance,
        center_depth_each_row,
        "o-",
        linewidth=2.0,
        markersize=6,
        label="Generated fault grid"
    )

    # ---------------------------------------------------------------------
    # 理论结果
    # ---------------------------------------------------------------------
    ax4.plot(
        theoretical_horizontal_distance,
        theoretical_depth,
        "--",
        linewidth=2.0,
        label="Theoretical geometry"
    )

    # 地表
    ax4.axhline(
        y=0.0,
        linestyle="-.",
        linewidth=1.5,
        color="black",
        label="Ground surface"
    )

    # 上缘
    ax4.scatter(
        theoretical_horizontal_distance[0],
        theoretical_depth[0],
        s=100,
        color="red",
        zorder=10,
        label="Upper edge"
    )

    # 下缘
    ax4.scatter(
        theoretical_horizontal_distance[-1],
        theoretical_depth[-1],
        s=100,
        color="blue",
        zorder=10,
        label="Lower edge"
    )

    # ---------------------------------------------------------------------
    # Hypocenter
    #
    # 从上缘开始，dhypo 沿倾向。
    # ---------------------------------------------------------------------
    hypo_horizontal_from_top = (
        dhypo
        * np.cos(dip_rad)
    )

    ax4.scatter(
        hypo_horizontal_from_top,
        final_hypo_depth,
        marker="*",
        s=220,
        color="magenta",
        edgecolor="black",
        zorder=10,
        label="Hypocenter"
    )

    ax4.set_xlabel(
        "Horizontal distance along dip direction (km)"
    )

    ax4.set_ylabel(
        "Depth (km)"
    )

    title_text = (
        "Down-dip Cross Section\n"
        f"W={Fault_width:.3f} km, "
        f"Dip={dip:.3f} deg"
    )

    if surface_exposed:

        title_text += (
            f", downward shift={depth_shift_km:.3f} km"
        )

    ax4.set_title(
        title_text
    )

    # 深度向下为正
    ax4.invert_yaxis()

    ax4.grid(
        True,
        linestyle=":",
        alpha=0.6
    )

    ax4.legend(
        loc="best"
    )

    fig4.tight_layout()

    if save_figures:

        fig4.savefig(
            output_path
            / "04_fault_dip_profile.png",
            dpi=300,
            bbox_inches="tight"
        )

    # =====================================================================
    # 显示所有图片
    # =====================================================================

    plt.show()

    # =====================================================================
    # 返回一些检查结果，方便后续程序继续使用
    # =====================================================================

    validation_result = {

        "all_pass": all(
            all_checks
        ),

        "calculated_length_km":
            calculated_length,

        "calculated_width_km":
            calculated_width,

        "calculated_strike_deg":
            calculated_strike,

        "calculated_dip_deg":
            calculated_dip,

        "calculated_dip_azimuth_deg":
            calculated_dip_azimuth,

        "calculated_shypo_km":
            calculated_shypo,

        "calculated_dhypo_km":
            calculated_dhypo,

        "max_dx_error_km":
            max_dx_error,

        "max_dy_error_km":
            max_dy_error
    }

    return validation_result


def me():
    pass

# =============================================================================
# 示例
# =============================================================================
if __name__ == "__main__":

    # =====================================================================
    # 1. 输入参数
    #
    # 注意：
    # 除经纬度和角度外，其余尺寸单位全部为 km。
    # =====================================================================

    Hypo_longi = 130.0000
    Hypo_lati = 33.0000
    Hypo_depth = 13.0

    strike = 135.0
    dip = 70.0

    Fault_length = 50.0
    Fault_width = 18.0

    shypo = -8.0
    dhypo = 10.0

    # 用户希望的网格尺寸，单位 km
    dx = 0.5
    dy = 0.5

    # =====================================================================
    # 2. 生成断层
    # =====================================================================

    result = build_fault_grid(
        Hypo_longi=Hypo_longi,
        Hypo_lati=Hypo_lati,
        Hypo_depth=Hypo_depth,
        strike=strike,
        dip=dip,
        Fault_length=Fault_length,
        Fault_width=Fault_width,
        shypo=shypo,
        dhypo=dhypo,
        dx=dx,
        dy=dy
    )

    # =====================================================================
    # 3. 可视化并自动检查断层几何
    # =====================================================================

    validation = visualize_and_validate_fault(
        result=result,

        Hypo_longi=Hypo_longi,
        Hypo_lati=Hypo_lati,

        # 注意：
        # 这里传入的是用户最初指定的 dx / dy，
        # 用于检查网格数量的四舍五入是否正确。
        requested_dx=dx,
        requested_dy=dy,

        # True：保存 4 张图片
        # False：只显示，不保存
        save_figures=True,

        output_dir="fault_geometry_check",

        # 子断层数量 <= 80 时，
        # 在局部网格图中显示子断层编号
        show_cell_index=True
    )

    # =====================================================================
    # 4. 最终判断
    # =====================================================================

    if validation["all_pass"]:

        print(
            "\n>>> 断层几何检查全部通过。"
        )

    else:

        print(
            "\n>>> 警告：存在未通过的几何检查项目。"
        )
