"""
不依赖 OpenQuake 的断层距离计算工具。

实现的量：
- Rrup：场点到离散断层网格节点的最小三维笛卡尔距离；
- Rjb：场点到断层面地表投影多边形的最短水平距离；
- Rx：场点到断层上缘折线、垂直于走向的有符号距离；下盘为负、上盘为正；
- azimuth：保留 v2 的兼容量——从断层走向顺时针量到“断层中点 -> 场点”方向的相对方位角；
- source_to_site_angle：NGA-West2 有限源 Source-to-Site Azimuth，范围 [-180, 180]°；
- epi_to_site_angle：从震中（Hypocenter 的地表投影）指向场点的绝对方位角，范围 [0, 360)°；
- epi_to_site_angle_refer_strike：上述方位角相对断层走向的有符号角，范围 [-180, 180]°。

说明
----
1. 方位角测地计算优先使用 ObsPy ``gps2dist_azimuth``（WGS84/GeographicLib 路径）；
   若运行环境没有 ObsPy，则自动回退到模块内置球面方位角算法。
   返回 DataFrame 的兼容封装函数会在函数内部导入 pandas。
2. Rrup 和最近点与 OpenQuake 的离散网格算法保持同一思路。
3. 单断层面的 Rx 按 OpenQuake BaseSurface 的上缘折线算法重写。
4. Rjb 采用“网格节点球面弦长 + 40 km 内正射投影多边形精算”的混合方法；
   对常规矩形网格，边界由网格外缘构成。
5. 多断层面的严格 OpenQuake MultiSurface Rx 使用 GC2 坐标，不等于各分段 Rx 取绝对值最小。
   本模块的多断层封装默认让 Rx 和两个方位角跟随 Rrup 控制面，以保持几何配对；
   如需完全复现旧代码，可使用 association="legacy_independent"。
"""

from __future__ import annotations

# ================================================================
# 版本：single_multi_v4（2026-08-14）
# 单断层入口：_parse_Rrup_Rjb_Rx_Azimuth(...)
# 旧单断层入口：_parse_Rrup_Rjb_Rx(...)；v4 返回距离 + 4 类方位角列
# 多断层入口：_Parse_Rrup_Rjb_Rx_Azimuth_multi_plane(...)
# ================================================================

from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np

try:
    from obspy.geodetics.base import (
        gps2dist_azimuth as _obspy_gps2dist_azimuth,
    )

    _HAS_OBSPY = True
except ImportError:
    _obspy_gps2dist_azimuth = None
    _HAS_OBSPY = False

EARTH_RADIUS_KM = 6371.0


def _to_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _broadcast_sites(site_lon: Any, site_lat: Any, site_depth: Any = 0.0):
    lon, lat, dep = np.broadcast_arrays(
        _to_float_array(site_lon),
        _to_float_array(site_lat),
        _to_float_array(site_depth),
    )
    return lon.ravel(), lat.ravel(), dep.ravel(), lon.shape


def _validate_fault_mesh(
    fault_lon: Any, fault_lat: Any, fault_depth: Any | None = None
):
    lon = _to_float_array(fault_lon)
    lat = _to_float_array(fault_lat)
    if lon.shape != lat.shape:
        raise ValueError("fault_lon 与 fault_lat 的形状必须一致。")
    if lon.ndim != 2:
        raise ValueError(
            "断层网格必须是二维数组，形状为 (沿倾向行数, 沿走向列数)。"
        )
    if min(lon.shape) < 1:
        raise ValueError("断层网格不能为空。")

    if fault_depth is None:
        dep = np.zeros_like(lon)
    else:
        dep = _to_float_array(fault_depth)
        if dep.shape != lon.shape:
            raise ValueError(
                "fault_depth 必须与 fault_lon、fault_lat 形状一致。"
            )

    finite = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(dep)
    if not np.any(finite):
        raise ValueError("断层网格中没有有限坐标。")
    return lon, lat, dep, finite


def geodetic_distance_km(lon1: Any, lat1: Any, lon2: Any, lat2: Any):
    """球面大圆距离，单位 km；输入可广播。"""
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(
        _to_float_array(lon1),
        _to_float_array(lat1),
        _to_float_array(lon2),
        _to_float_array(lat2),
    )
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    hav = (
        np.sin((lat1 - lat2) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.0) ** 2
    )
    hav = np.clip(hav, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))


def azimuth_deg(lon1: Any, lat1: Any, lon2: Any, lat2: Any):
    """从点 1 指向点 2 的初始方位角，北为 0°，顺时针为正，范围 [0, 360)。"""
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(
        _to_float_array(lon1),
        _to_float_array(lat1),
        _to_float_array(lon2),
        _to_float_array(lat2),
    )
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    cos_lat2 = np.cos(lat2)
    true_course = np.degrees(
        np.arctan2(
            np.sin(lon1 - lon2) * cos_lat2,
            np.cos(lat1) * np.sin(lat2)
            - np.sin(lat1) * cos_lat2 * np.cos(lon1 - lon2),
        )
    )
    return (360.0 - true_course) % 360.0


def _normalize_unsigned_angle_deg(
    angle: Any, atol: float = 1.0e-10
) -> np.ndarray:
    """
    将绝对方位角严格规范化到 [0, 360)°。

    主要处理浮点数边界：例如 ObsPy / 测地算法可能给出
    359.999999999997°，物理上等价于 0°，如果直接保存或绘图会显示成 360°。

    规则：
        1. 先对 360° 取模；
        2. 与 0° 或 360° 在 ``atol`` 内等价的值统一设为 0.0°；
        3. 其他值保持在 [0, 360)°。
    """
    angle = _to_float_array(angle)
    wrapped = np.mod(angle, 360.0)

    finite = np.isfinite(wrapped)
    boundary_zero = finite & (
        np.isclose(wrapped, 0.0, atol=atol, rtol=0.0)
        | np.isclose(wrapped, 360.0, atol=atol, rtol=0.0)
    )
    wrapped = np.where(boundary_zero, 0.0, wrapped)
    return wrapped


def geodesic_azimuth_deg(
    lon1: Any, lat1: Any, lon2: Any, lat2: Any
) -> np.ndarray:
    """
    计算从点 1 指向点 2 的初始方位角，范围 [0, 360)°。

    优先使用 ObsPy ``gps2dist_azimuth``，以与用户旧程序保持一致；若当前
    Python 环境未安装 ObsPy，则自动回退到本模块 ``azimuth_deg`` 的球面实现。

    参数支持 NumPy 广播。ObsPy 的 ``gps2dist_azimuth`` 本身以标量为主，
    因此数组输入时在内部逐元素调用；对 NGA-West2 常规场点网格这是稳定的，
    只是速度会比纯 NumPy 球面公式略慢。
    """
    lon1a, lat1a, lon2a, lat2a = np.broadcast_arrays(
        _to_float_array(lon1),
        _to_float_array(lat1),
        _to_float_array(lon2),
        _to_float_array(lat2),
    )

    if not _HAS_OBSPY:
        return _normalize_unsigned_angle_deg(
            azimuth_deg(lon1a, lat1a, lon2a, lat2a)
        )

    flat_lon1 = lon1a.ravel()
    flat_lat1 = lat1a.ravel()
    flat_lon2 = lon2a.ravel()
    flat_lat2 = lat2a.ravel()
    out = np.empty(flat_lon1.size, dtype=float)

    for i in range(flat_lon1.size):
        _, az, _ = _obspy_gps2dist_azimuth(
            float(flat_lat1[i]),
            float(flat_lon1[i]),
            float(flat_lat2[i]),
            float(flat_lon2[i]),
        )
        out[i] = az

    # 统一处理 360° 与 0° 的浮点边界。
    # 例如 359.999999999997° 必须输出为 0.0°，而不是在图上显示为 360°。
    return _normalize_unsigned_angle_deg(out.reshape(lon1a.shape))


def spherical_to_cartesian(lon: Any, lat: Any, depth: Any = 0.0) -> np.ndarray:
    """经纬度和深度转地心直角坐标，单位 km；深度向下为正。"""
    lon, lat, depth = np.broadcast_arrays(
        _to_float_array(lon), _to_float_array(lat), _to_float_array(depth)
    )
    phi = np.radians(lon)
    theta = np.radians(lat)
    radius = EARTH_RADIUS_KM - depth
    cos_lat_r = radius * np.cos(theta)
    return np.stack(
        (
            cos_lat_r * np.cos(phi),
            cos_lat_r * np.sin(phi),
            radius * np.sin(theta),
        ),
        axis=-1,
    )


def _nearest_cartesian(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    max_memory_mb: float = 128.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回每个 target 到 source 点集的最短欧氏距离与 source 索引。"""
    source_xyz = np.asarray(source_xyz, dtype=float).reshape(-1, 3)
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(-1, 3)
    if len(source_xyz) == 0:
        raise ValueError("source_xyz 不能为空。")

    n_source = len(source_xyz)
    # 距离平方矩阵每个元素 8 字节；按目标点分块，避免一次性占用过多内存。
    bytes_budget = max(float(max_memory_mb), 1.0) * 1024.0 * 1024.0
    chunk_size = max(1, int(bytes_budget // (8.0 * n_source)))

    source_norm = np.einsum("ij,ij->i", source_xyz, source_xyz)
    out_dist = np.empty(len(target_xyz), dtype=float)
    out_idx = np.empty(len(target_xyz), dtype=np.int64)

    for start in range(0, len(target_xyz), chunk_size):
        stop = min(start + chunk_size, len(target_xyz))
        block = target_xyz[start:stop]
        block_norm = np.einsum("ij,ij->i", block, block)
        d2 = (
            source_norm[:, None]
            + block_norm[None, :]
            - 2.0 * source_xyz @ block.T
        )
        np.maximum(d2, 0.0, out=d2)
        idx = np.argmin(d2, axis=0)
        out_idx[start:stop] = idx
        out_dist[start:stop] = np.sqrt(d2[idx, np.arange(stop - start)])

    return out_dist, out_idx


def rrup_distance_km(
    fault_lon: Any,
    fault_lat: Any,
    fault_depth: Any,
    site_lon: Any,
    site_lat: Any,
    site_depth: Any = 0.0,
    *,
    return_index: bool = False,
    max_memory_mb: float = 128.0,
):
    """
    计算 Rrup：场点到离散断层网格节点的最小三维距离。

    返回值按场点输入展平为一维数组。return_index=True 时，同时返回最近断层节点在
    有限节点展平数组中的索引。
    """
    flon, flat, fdep, finite = _validate_fault_mesh(
        fault_lon, fault_lat, fault_depth
    )
    slon, slat, sdep, _ = _broadcast_sites(site_lon, site_lat, site_depth)

    fault_xyz = spherical_to_cartesian(
        flon[finite], flat[finite], fdep[finite]
    ).reshape(-1, 3)
    site_xyz = spherical_to_cartesian(slon, slat, sdep).reshape(-1, 3)
    dist, idx = _nearest_cartesian(
        fault_xyz, site_xyz, max_memory_mb=max_memory_mb
    )
    return (dist, idx) if return_index else dist


def closest_fault_mesh_point(
    fault_lon: Any,
    fault_lat: Any,
    fault_depth: Any,
    site_lon: Any,
    site_lat: Any,
    site_depth: Any = 0.0,
    *,
    max_memory_mb: float = 128.0,
):
    """返回每个场点对应的最近断层网格节点 (lon, lat, depth, distance)。"""
    flon, flat, fdep, finite = _validate_fault_mesh(
        fault_lon, fault_lat, fault_depth
    )
    slon, slat, sdep, _ = _broadcast_sites(site_lon, site_lat, site_depth)

    valid_lon = flon[finite]
    valid_lat = flat[finite]
    valid_dep = fdep[finite]
    fault_xyz = spherical_to_cartesian(
        valid_lon, valid_lat, valid_dep
    ).reshape(-1, 3)
    site_xyz = spherical_to_cartesian(slon, slat, sdep).reshape(-1, 3)
    dist, idx = _nearest_cartesian(
        fault_xyz, site_xyz, max_memory_mb=max_memory_mb
    )
    return valid_lon[idx], valid_lat[idx], valid_dep[idx], dist


def _circular_mean_longitude_deg(longitudes: np.ndarray) -> float:
    angles = np.radians(np.asarray(longitudes, dtype=float))
    s = np.nanmean(np.sin(angles))
    c = np.nanmean(np.cos(angles))
    if np.isclose(s, 0.0) and np.isclose(c, 0.0):
        return float(np.nanmean(longitudes))
    return float(np.degrees(np.arctan2(s, c)))


def _orthographic_project(
    lon: Any, lat: Any, center_lon: float, center_lat: float
) -> Tuple[np.ndarray, np.ndarray]:
    """以指定中心作球面正射投影，输出单位 km。"""
    lon = np.radians(_to_float_array(lon))
    lat = np.radians(_to_float_array(lat))
    lon0 = np.radians(center_lon)
    lat0 = np.radians(center_lat)
    dlon = (lon - lon0 + np.pi) % (2.0 * np.pi) - np.pi

    x = EARTH_RADIUS_KM * np.cos(lat) * np.sin(dlon)
    y = EARTH_RADIUS_KM * (
        np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    )
    return x, y


def _surface_boundary(fault_lon: np.ndarray, fault_lat: np.ndarray):
    """按矩形网格外围顺序提取闭合边界：上、右、下、左。"""
    if fault_lon.shape[0] == 1:
        # 退化成折线；Rjb 将按折线距离处理。
        return fault_lon[0].copy(), fault_lat[0].copy()
    if fault_lon.shape[1] == 1:
        return fault_lon[:, 0].copy(), fault_lat[:, 0].copy()

    lon = np.concatenate(
        (
            fault_lon[0, :],
            fault_lon[1:, -1],
            fault_lon[-1, :-1][::-1],
            fault_lon[:-1, 0][::-1],
        )
    )
    lat = np.concatenate(
        (
            fault_lat[0, :],
            fault_lat[1:, -1],
            fault_lat[-1, :-1][::-1],
            fault_lat[:-1, 0][::-1],
        )
    )
    finite = np.isfinite(lon) & np.isfinite(lat)
    return lon[finite], lat[finite]


def _point_to_polyline_distance(
    point_x: np.ndarray,
    point_y: np.ndarray,
    line_x: np.ndarray,
    line_y: np.ndarray,
    *,
    closed: bool,
) -> np.ndarray:
    point_x = np.asarray(point_x, dtype=float).ravel()
    point_y = np.asarray(point_y, dtype=float).ravel()
    line_x = np.asarray(line_x, dtype=float).ravel()
    line_y = np.asarray(line_y, dtype=float).ravel()

    if len(line_x) == 0:
        raise ValueError("边界折线不能为空。")
    if len(line_x) == 1:
        return np.hypot(point_x - line_x[0], point_y - line_y[0])

    if closed:
        x1 = line_x
        y1 = line_y
        x2 = np.roll(line_x, -1)
        y2 = np.roll(line_y, -1)
    else:
        x1, y1 = line_x[:-1], line_y[:-1]
        x2, y2 = line_x[1:], line_y[1:]

    dx = x2 - x1
    dy = y2 - y1
    seg2 = dx * dx + dy * dy
    safe_seg2 = np.where(seg2 > 0.0, seg2, 1.0)

    px = point_x[:, None]
    py = point_y[:, None]
    t = ((px - x1) * dx + (py - y1) * dy) / safe_seg2
    t = np.clip(t, 0.0, 1.0)
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    d2 = (px - nearest_x) ** 2 + (py - nearest_y) ** 2

    zero = seg2 == 0.0
    if np.any(zero):
        d2[:, zero] = (px - x1[zero]) ** 2 + (py - y1[zero]) ** 2
    return np.sqrt(np.min(d2, axis=1))


def _points_in_polygon(
    point_x: np.ndarray,
    point_y: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
) -> np.ndarray:
    """射线法判断点是否位于多边形内部；边界由后续距离容差处理。"""
    x = np.asarray(point_x, dtype=float).ravel()
    y = np.asarray(point_y, dtype=float).ravel()
    px = np.asarray(poly_x, dtype=float).ravel()
    py = np.asarray(poly_y, dtype=float).ravel()
    inside = np.zeros(len(x), dtype=bool)

    j = len(px) - 1
    for i in range(len(px)):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        crosses = (yi > y) != (yj > y)
        x_intersect = (xj - xi) * (y - yi) / (
            (yj - yi) + np.finfo(float).eps
        ) + xi
        inside ^= crosses & (x < x_intersect)
        j = i
    return inside


def rjb_distance_km(
    fault_lon: Any,
    fault_lat: Any,
    site_lon: Any,
    site_lat: Any,
    *,
    boundary_tolerance_km: float = 0.005,
) -> np.ndarray:
    """
    计算 Rjb：场点到断层面地表投影边界多边形的最短水平距离。

    对投影内部及距边界不超过 boundary_tolerance_km 的点返回 0。
    """
    flon, flat, _, _ = _validate_fault_mesh(fault_lon, fault_lat, None)
    slon, slat, _, _ = _broadcast_sites(site_lon, site_lat, 0.0)
    blon, blat = _surface_boundary(flon, flat)

    # 与 OpenQuake Mesh.get_joyner_boore_distance 一致，先计算场点到断层投影
    # 网格节点的最短球面弦长。只有初值小于 40 km 的场点，才进一步用投影
    # 多边形重算；较远场点直接保留节点距离，以减少大规模计算量。
    finite_nodes = np.isfinite(flon) & np.isfinite(flat)
    distance = _min_geodetic_distance_km(
        flon[finite_nodes], flat[finite_nodes], slon, slat
    )
    close = distance < 40.0
    if not np.any(close):
        return distance

    # 投影中心只由断层网格决定，避免场点分布改变同一断层的投影坐标系。
    center_lon = _circular_mean_longitude_deg(flon[finite_nodes])
    center_lat = float(np.nanmean(flat[finite_nodes]))

    bx, by = _orthographic_project(blon, blat, center_lon, center_lat)
    sx, sy = _orthographic_project(
        slon[close], slat[close], center_lon, center_lat
    )

    is_polygon = flon.shape[0] > 1 and flon.shape[1] > 1 and len(bx) >= 3
    refined = _point_to_polyline_distance(sx, sy, bx, by, closed=is_polygon)
    if is_polygon:
        inside = _points_in_polygon(sx, sy, bx, by)
        refined[inside | (refined <= boundary_tolerance_km)] = 0.0
    else:
        refined[refined <= boundary_tolerance_km] = 0.0
    distance[close] = refined
    return distance


def _min_geodetic_distance_km(
    source_lon: Any, source_lat: Any, target_lon: Any, target_lat: Any
) -> np.ndarray:
    source_xyz = spherical_to_cartesian(
        _to_float_array(source_lon).ravel(),
        _to_float_array(source_lat).ravel(),
        0.0,
    ).reshape(-1, 3)
    target_xyz = spherical_to_cartesian(
        _to_float_array(target_lon).ravel(),
        _to_float_array(target_lat).ravel(),
        0.0,
    ).reshape(-1, 3)
    return _nearest_cartesian(source_xyz, target_xyz)[0]


def distance_to_arc_km(
    arc_lon: float,
    arc_lat: float,
    arc_azimuth: float,
    point_lon: Any,
    point_lat: Any,
) -> np.ndarray:
    """场点到无限大圆弧的有符号垂距；弧线右侧为负。"""
    plon, plat, _, _ = _broadcast_sites(point_lon, point_lat, 0.0)
    target_azimuth = azimuth_deg(arc_lon, arc_lat, plon, plat)
    target_distance = geodetic_distance_km(arc_lon, arc_lat, plon, plat)
    angle_from_arc = (target_azimuth - arc_azimuth + 360.0) % 360.0
    argument = np.sin(np.radians(angle_from_arc)) * np.sin(
        target_distance / EARTH_RADIUS_KM
    )
    argument = np.clip(argument, -1.0, 1.0)
    angle = np.arccos(argument)
    return (np.pi / 2.0 - angle) * EARTH_RADIUS_KM


def distance_to_semi_arc_km(
    arc_lon: float,
    arc_lat: float,
    arc_azimuth: float,
    point_lon: Any,
    point_lat: Any,
) -> np.ndarray:
    """场点到从参考点沿给定方位延伸的半无限大圆弧的有符号距离。"""
    plon, plat, _, _ = _broadcast_sites(point_lon, point_lat, 0.0)
    target_azimuth = azimuth_deg(arc_lon, arc_lat, plon, plat)
    delta = np.radians(arc_azimuth - target_azimuth)
    positive_half = np.cos(delta) > 0.0
    lower_left = (~positive_half) & (np.sin(delta) > 0.0)

    distance = np.empty_like(plon)
    if np.any(positive_half):
        distance[positive_half] = distance_to_arc_km(
            arc_lon,
            arc_lat,
            arc_azimuth,
            plon[positive_half],
            plat[positive_half],
        )
    if np.any(~positive_half):
        distance[~positive_half] = geodetic_distance_km(
            arc_lon, arc_lat, plon[~positive_half], plat[~positive_half]
        )
        distance[lower_left] *= -1.0
    return distance


def min_distance_to_segment_km(
    segment_lon: Sequence[float],
    segment_lat: Sequence[float],
    point_lon: Any,
    point_lat: Any,
) -> np.ndarray:
    """场点到有限球面线段的有符号最短距离，线段右侧为正、左侧为负。"""
    seglon = _to_float_array(segment_lon).ravel()
    seglat = _to_float_array(segment_lat).ravel()
    if len(seglon) != 2 or len(seglat) != 2:
        raise ValueError("segment_lon 和 segment_lat 必须各含两个端点。")

    plon, plat, _, _ = _broadcast_sites(point_lon, point_lat, 0.0)
    seg_azimuth = float(
        azimuth_deg(seglon[0], seglat[0], seglon[1], seglat[1])
    )
    az1 = azimuth_deg(seglon[0], seglat[0], plon, plat)
    az2 = azimuth_deg(seglon[1], seglat[1], plon, plat)

    in_band = (np.cos(np.radians(seg_azimuth - az1)) >= 0.0) & (
        np.cos(np.radians(seg_azimuth - az2)) <= 0.0
    )
    left_side = np.sin(np.radians(az1 - seg_azimuth)) < 0.0

    distance = np.empty_like(plon)
    if np.any(in_band):
        distance[in_band] = distance_to_arc_km(
            seglon[0], seglat[0], seg_azimuth, plon[in_band], plat[in_band]
        )
    if np.any(~in_band):
        distance[~in_band] = _min_geodetic_distance_km(
            seglon, seglat, plon[~in_band], plat[~in_band]
        )

    distance = np.abs(distance)
    distance[left_side] *= -1.0
    return distance


def rx_distance_km(
    top_edge_lon: Any,
    top_edge_lat: Any,
    site_lon: Any,
    site_lat: Any,
) -> np.ndarray:
    """
    计算单断层面的 Rx。

    top_edge_lon/top_edge_lat 是按走向顺序排列的断层上缘坐标；下倾侧（上盘）为正。
    """
    tlon = _to_float_array(top_edge_lon).ravel()
    tlat = _to_float_array(top_edge_lat).ravel()
    finite = np.isfinite(tlon) & np.isfinite(tlat)
    tlon, tlat = tlon[finite], tlat[finite]
    if len(tlon) < 2:
        raise ValueError("Rx 计算至少需要两个有限的断层上缘点。")

    slon, slat, _, _ = _broadcast_sites(site_lon, site_lat, 0.0)
    if len(tlon) == 2:
        strike = float(azimuth_deg(tlon[0], tlat[0], tlon[1], tlat[1]))
        return distance_to_arc_km(tlon[0], tlat[0], strike, slon, slat)

    segment_distances = []
    last_segment = len(tlon) - 2
    for i in range(len(tlon) - 1):
        p1_lon, p1_lat = tlon[i], tlat[i]
        p2_lon, p2_lat = tlon[i + 1], tlat[i + 1]

        # OpenQuake 对第一段交换两个端点，再在结果上反号。
        if i == 0:
            p1_lon, p2_lon = p2_lon, p1_lon
            p1_lat, p2_lat = p2_lat, p1_lat

        if i == 0 or i == last_segment:
            segment_azimuth = float(
                azimuth_deg(p1_lon, p1_lat, p2_lon, p2_lat)
            )
            dist = distance_to_semi_arc_km(
                p1_lon, p1_lat, segment_azimuth, slon, slat
            )
        else:
            dist = min_distance_to_segment_km(
                [p1_lon, p2_lon], [p1_lat, p2_lat], slon, slat
            )

        if i == 0:
            dist = -dist
        segment_distances.append(dist)

    all_distances = np.vstack(segment_distances)
    selector = np.argmin(np.abs(all_distances), axis=0)
    result = all_distances[selector, np.arange(all_distances.shape[1])]
    if np.any(~np.isfinite(result)):
        raise ValueError("Rx 计算产生了非有限值，请检查断层上缘坐标。")
    return result


def _great_circle_midpoint(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> Tuple[float, float]:
    """两个地表点之间较短大圆弧的中点。"""
    xyz = spherical_to_cartesian(
        np.array([lon1, lon2]), np.array([lat1, lat2]), 0.0
    )
    direction = xyz[0] + xyz[1]
    norm = np.linalg.norm(direction)
    if norm <= np.finfo(float).eps:
        raise ValueError("两个点接近对跖点，无法唯一确定大圆中点。")
    direction /= norm
    lon = np.degrees(np.arctan2(direction[1], direction[0]))
    lat = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    return float(lon), float(lat)


def fault_middle_point(
    fault_lon: Any, fault_lat: Any, fault_depth: Any | None = None
) -> Tuple[float, float, float]:
    """
    返回矩形网格中点。

    奇数行/列直接取中央节点；偶数时递归求中央两个或四个节点的大圆中点，
    深度始终取算术平均。这与 RectangularMesh.get_middle_point 的定义一致。
    """
    flon, flat, fdep, _ = _validate_fault_mesh(
        fault_lon, fault_lat, fault_depth
    )
    nr, nc = flon.shape
    mid_row = nr // 2

    if nr % 2 == 1:
        mid_col = nc // 2
        if nc % 2 == 1:
            values = (
                flon[mid_row, mid_col],
                flat[mid_row, mid_col],
                fdep[mid_row, mid_col],
            )
            if np.all(np.isfinite(values)):
                return tuple(float(v) for v in values)
            raise ValueError("断层网格中央节点含非有限值。")

        lon1, lon2 = flon[mid_row, mid_col - 1 : mid_col + 1]
        lat1, lat2 = flat[mid_row, mid_col - 1 : mid_col + 1]
        dep1, dep2 = fdep[mid_row, mid_col - 1 : mid_col + 1]
    else:
        p1 = fault_middle_point(
            flon[mid_row - 1 : mid_row],
            flat[mid_row - 1 : mid_row],
            fdep[mid_row - 1 : mid_row],
        )
        p2 = fault_middle_point(
            flon[mid_row : mid_row + 1],
            flat[mid_row : mid_row + 1],
            fdep[mid_row : mid_row + 1],
        )
        lon1, lat1, dep1 = p1
        lon2, lat2, dep2 = p2

    if not np.all(np.isfinite([lon1, lat1, dep1, lon2, lat2, dep2])):
        raise ValueError("用于计算断层中点的中央节点含非有限值。")
    lon, lat = _great_circle_midpoint(lon1, lat1, lon2, lat2)
    depth = (float(dep1) + float(dep2)) / 2.0
    return lon, lat, depth


def _normalized_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("断层网格中存在重合节点，无法确定平均走向。")
    return vectors / norm


def _weighted_top_edge_strike(flon: np.ndarray, flat: np.ndarray) -> float:
    tlon = flon[0].ravel()
    tlat = flat[0].ravel()
    valid_pair = (
        np.isfinite(tlon[:-1])
        & np.isfinite(tlat[:-1])
        & np.isfinite(tlon[1:])
        & np.isfinite(tlat[1:])
    )
    if not np.any(valid_pair):
        raise ValueError("计算走向至少需要一个有效上缘线段。")
    lon1, lat1 = tlon[:-1][valid_pair], tlat[:-1][valid_pair]
    lon2, lat2 = tlon[1:][valid_pair], tlat[1:][valid_pair]
    segment_azimuth = azimuth_deg(lon1, lat1, lon2, lat2)
    weights = geodetic_distance_km(lon1, lat1, lon2, lat2)
    if np.all(weights == 0.0):
        raise ValueError("断层上缘所有线段长度均为零。")
    angle = np.radians(segment_azimuth)
    return float(
        np.degrees(
            np.arctan2(
                np.sum(weights * np.sin(angle)),
                np.sum(weights * np.cos(angle)),
            )
        )
        % 360.0
    )


def fault_strike_deg(
    fault_lon: Any, fault_lat: Any, fault_depth: Any | None = None
) -> float:
    """
    计算 SimpleFaultSurface 风格的平均走向。

    对至少 2×2 的网格，使用顶部一排网格单元的两个三角形面积加权平均走向；
    这对应 SimpleFaultSurface.get_strike 的计算路径。退化网格则使用上缘线段长度
    加权的圆周平均。
    """
    flon, flat, fdep, finite = _validate_fault_mesh(
        fault_lon, fault_lat, fault_depth
    )
    if flon.shape[0] < 2 or flon.shape[1] < 2 or not np.all(finite[:2]):
        return _weighted_top_edge_strike(flon, flat)
    if np.any(fdep[1] < fdep[0]):
        raise ValueError(
            "SimpleFaultSurface 风格走向要求下一行节点不能比上一行更浅。"
        )

    # SimpleFaultSurface 只用顶部一排单元，因为其倾向方向上的走向保持一致。
    points = spherical_to_cartesian(flon[:2], flat[:2], fdep[:2])
    along = points[:, 1:] - points[:, :-1]  # 沿走向 →
    updip = points[:-1] - points[1:]  # 沿上倾 ↑
    diag = points[:-1, 1:] - points[1:, :-1]  # 左下到右上 ↗

    tl_e1 = along[:-1]
    tl_e2 = updip[:, :-1]
    br_e1 = along[1:]
    br_e2 = updip[:, 1:]
    tl_area = 0.5 * np.linalg.norm(np.cross(tl_e1, tl_e2), axis=-1)
    br_area = 0.5 * np.linalg.norm(np.cross(br_e1, br_e2), axis=-1)
    if np.sum(tl_area) + np.sum(br_area) <= np.finfo(float).eps:
        return _weighted_top_edge_strike(flon, flat)

    unit_along = _normalized_vectors(along)
    z_unit = np.array([0.0, 0.0, 1.0])
    norms_west = _normalized_vectors(np.cross(points + z_unit, points))
    norms_north = _normalized_vectors(np.cross(points, norms_west))

    sign_tl = np.sign(
        np.sign(np.sum(unit_along[:-1] * norms_west[:-1, :-1], axis=-1)) + 0.1
    )
    cos_tl = np.clip(
        np.sum(unit_along[:-1] * norms_north[:-1, :-1], axis=-1), -1.0, 1.0
    )
    xx = np.sum(tl_area * cos_tl)
    yy = np.sum(
        tl_area * np.sqrt(np.maximum(0.0, 1.0 - cos_tl * cos_tl)) * sign_tl
    )

    sign_br = np.sign(
        np.sign(np.sum(unit_along[1:] * norms_west[1:, 1:], axis=-1)) + 0.1
    )
    cos_br = np.clip(
        np.sum(unit_along[1:] * norms_north[1:, 1:], axis=-1), -1.0, 1.0
    )
    xx += np.sum(br_area * cos_br)
    yy += np.sum(
        br_area * np.sqrt(np.maximum(0.0, 1.0 - cos_br * cos_br)) * sign_br
    )
    strike = float(np.degrees(np.arctan2(yy, xx)) % 360.0)

    # 复现 get_mean_inclination_and_azimuth 对反向法向的纠正。
    earth_normal = _normalized_vectors(points)
    tl_normal = _normalized_vectors(np.cross(tl_e1, tl_e2))
    br_normal = _normalized_vectors(np.cross(br_e1, br_e2))
    tl_cos = np.clip(
        np.sum(earth_normal[:-1, :-1] * tl_normal, axis=-1), -1.0, 1.0
    )
    br_cos = np.clip(
        np.sum(earth_normal[1:, 1:] * br_normal, axis=-1), -1.0, 1.0
    )
    inc_x = np.sum(tl_area * tl_cos) + np.sum(br_area * br_cos)
    inc_y = np.sum(
        tl_area * np.sqrt(np.maximum(0.0, 1.0 - tl_cos * tl_cos))
    ) + np.sum(br_area * np.sqrt(np.maximum(0.0, 1.0 - br_cos * br_cos)))
    inclination = float(np.degrees(np.arctan2(inc_y, inc_x)))
    if inclination > 90.0:
        strike = (strike + 180.0) % 360.0
    return strike


def relative_azimuth_deg(
    fault_lon: Any,
    fault_lat: Any,
    fault_depth: Any,
    site_lon: Any,
    site_lat: Any,
    *,
    strike_override: float | None = None,
) -> np.ndarray:
    """从断层走向顺时针量到“断层中点 -> 场点”方向的相对方位角。"""
    mid_lon, mid_lat, _ = fault_middle_point(fault_lon, fault_lat, fault_depth)
    strike = (
        fault_strike_deg(fault_lon, fault_lat, fault_depth)
        if strike_override is None
        else float(strike_override)
    )
    slon, slat, _, _ = _broadcast_sites(site_lon, site_lat, 0.0)
    return (azimuth_deg(mid_lon, mid_lat, slon, slat) - strike) % 360.0


def azimuth_of_closest_point_deg(
    fault_lon: Any,
    fault_lat: Any,
    fault_depth: Any,
    site_lon: Any,
    site_lat: Any,
    site_depth: Any = 0.0,
) -> np.ndarray:
    """
    保留 v2 的兼容函数：从每个场点指向其最近断层网格节点的绝对方位角。

    注意
    ----
    这个量不是 NGA-West2 数据库中的 Source-to-Site Azimuth。
    v4 中 NGA-West2 的有限源方位角请使用
    ``nga_west2_source_to_site_azimuth_deg``。
    """
    slon, slat, _, _ = _broadcast_sites(site_lon, site_lat, site_depth)
    close_lon, close_lat, _, _ = closest_fault_mesh_point(
        fault_lon, fault_lat, fault_depth, slon, slat, site_depth
    )
    return azimuth_deg(slon, slat, close_lon, close_lat)


def _unit_sphere_xyz(lon: Any, lat: Any) -> np.ndarray:
    """经纬度转换为单位球面三维向量。"""
    lon, lat = np.broadcast_arrays(_to_float_array(lon), _to_float_array(lat))
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    cos_lat = np.cos(lat_r)
    return np.stack(
        (cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)),
        axis=-1,
    )


def _central_angle_xyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """单位球向量之间的中心角，单位 rad。"""
    dot = np.sum(a * b, axis=-1)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def _closest_points_on_top_edge_sphere(
    top_edge_lon: Any,
    top_edge_lat: Any,
    site_lon: Any,
    site_lat: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    求每个场点在有限断层上缘地表投影折线上的最近点 Xcte。

    每一条相邻上缘节点之间视为较短的大圆弧。对每个场点，先求其到
    该大圆的球面正交投影；若投影落在有限弧段内，则使用投影点；否则
    使用较近的端点。最后在所有上缘弧段中选取真正最近的 Xcte。

    返回
    ----
    closest_lon, closest_lat, segment_index
        最近点经纬度以及控制上缘线段索引。结果按场点展平。
    """
    tlon = _to_float_array(top_edge_lon).ravel()
    tlat = _to_float_array(top_edge_lat).ravel()
    finite = np.isfinite(tlon) & np.isfinite(tlat)
    tlon, tlat = tlon[finite], tlat[finite]
    if len(tlon) < 2:
        raise ValueError(
            "Source-to-Site Azimuth 至少需要两个有限的断层上缘点。"
        )

    slon, slat, _, _ = _broadcast_sites(site_lon, site_lat, 0.0)
    p = _unit_sphere_xyz(slon, slat).reshape(-1, 3)
    a_all = _unit_sphere_xyz(tlon[:-1], tlat[:-1]).reshape(-1, 3)
    b_all = _unit_sphere_xyz(tlon[1:], tlat[1:]).reshape(-1, 3)

    n_site = len(p)
    best_angle = np.full(n_site, np.inf, dtype=float)
    best_xyz = np.empty((n_site, 3), dtype=float)
    best_seg = np.full(n_site, -1, dtype=np.int64)

    tol = 1.0e-10

    for i, (a, b) in enumerate(zip(a_all, b_all)):
        normal = np.cross(a, b)
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= np.finfo(float).eps:
            continue
        normal /= normal_norm

        # 场点在该大圆平面上的正交投影。
        proj = p - np.outer(p @ normal, normal)
        proj_norm = np.linalg.norm(proj, axis=1)
        safe = proj_norm > np.finfo(float).eps
        q = np.empty_like(proj)
        q[safe] = proj[safe] / proj_norm[safe, None]
        q[~safe] = a

        # q 与 -q 都在大圆上，选取离场点更近的那个。
        flip = np.sum(q * p, axis=1) < 0.0
        q[flip] *= -1.0

        ab = float(_central_angle_xyz(a[None, :], b[None, :])[0])
        aq = _central_angle_xyz(np.broadcast_to(a, q.shape), q)
        qb = _central_angle_xyz(q, np.broadcast_to(b, q.shape))

        # 投影点位于有限的 minor arc 上。
        on_segment = np.abs((aq + qb) - ab) <= max(tol, 1.0e-7 * max(ab, 1.0))

        # 若投影不在弧段内，最近点必为两个端点之一。
        pa = _central_angle_xyz(p, np.broadcast_to(a, p.shape))
        pb = _central_angle_xyz(p, np.broadcast_to(b, p.shape))
        use_a = pa <= pb
        endpoint = np.where(use_a[:, None], a, b)
        candidate = np.where(on_segment[:, None], q, endpoint)
        candidate_angle = _central_angle_xyz(p, candidate)

        improve = candidate_angle < best_angle
        best_angle[improve] = candidate_angle[improve]
        best_xyz[improve] = candidate[improve]
        best_seg[improve] = i

    if np.any(best_seg < 0):
        raise ValueError(
            "断层上缘存在退化线段，无法确定 Source-to-Site Azimuth。"
        )

    closest_lon = np.degrees(np.arctan2(best_xyz[:, 1], best_xyz[:, 0]))
    closest_lat = np.degrees(
        np.arctan2(best_xyz[:, 2], np.hypot(best_xyz[:, 0], best_xyz[:, 1]))
    )
    return closest_lon, closest_lat, best_seg


def _wrap_signed_angle_deg(angle: Any) -> np.ndarray:
    """
    把角度统一到 [-180, 180]。

    对恰好 180° 的边界情况保留原始旋转方向：
    原始差角为正时返回 +180°，原始差角为负时返回 -180°。
    这样在 Source-to-Site Azimuth 的断层两端不会把两侧都强制画成 +180°。
    """
    angle = _to_float_array(angle)
    wrapped = (angle + 180.0) % 360.0 - 180.0
    boundary = np.isclose(np.abs(wrapped), 180.0, atol=1.0e-10)
    wrapped = np.where(boundary & (angle > 0.0), 180.0, wrapped)
    return wrapped


def nga_west2_source_to_site_azimuth_deg(
    top_edge_lon: Any,
    top_edge_lat: Any,
    site_lon: Any,
    site_lat: Any,
    *,
    strike_override: float | None = None,
) -> np.ndarray:
    """
    计算 NGA-West2 数据库定义的有限源 Source-to-Site Azimuth，单位度。

    定义
    ----
    对每个场点先求其在断层上缘地表投影上的最近点 Xcte，然后计算
    ``Xcte -> site`` 的方位角相对于 fault strike 的有符号夹角。

    返回范围为 [-180, 180] 度：
    - 下倾侧/上盘一侧通常为正；
    - 上倾侧/下盘一侧通常为负；
    - 当最近点位于上缘内部时，角度趋于 +/-90 度；
    - 超过断层两端后，角度连续向 0 或 +/-180 度变化。

    这对应 PEER NGA-West2 flatfile 中 ``Source to Site Azimuth`` 的有限源定义，
    与 v2 中“场点 -> 最近断层网格节点”的方位角不是同一个量。
    """
    tlon = _to_float_array(top_edge_lon).ravel()
    tlat = _to_float_array(top_edge_lat).ravel()
    finite = np.isfinite(tlon) & np.isfinite(tlat)
    tlon, tlat = tlon[finite], tlat[finite]
    if len(tlon) < 2:
        raise ValueError(
            "Source-to-Site Azimuth 至少需要两个有限的断层上缘点。"
        )

    if strike_override is None:
        strike = _weighted_top_edge_strike(tlon[None, :], tlat[None, :])
    else:
        strike = float(strike_override) % 360.0

    slon, slat, _, _ = _broadcast_sites(site_lon, site_lat, 0.0)
    xlon, xlat, _ = _closest_points_on_top_edge_sphere(tlon, tlat, slon, slat)

    xcte_to_site = geodesic_azimuth_deg(xlon, xlat, slon, slat)
    angle = _wrap_signed_angle_deg(xcte_to_site - strike)

    # 场点恰好落在上缘上时方向本身不唯一；按 0 处理。
    on_edge = geodetic_distance_km(xlon, xlat, slon, slat) <= 1.0e-8
    angle[on_edge] = 0.0
    return angle


def hypocenter_to_site_azimuths_deg(
    hypocenter_lon: float,
    hypocenter_lat: float,
    site_lon: Any,
    site_lat: Any,
    *,
    strike: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算用户旧程序中的另外两个方位角。

    返回
    ----
    epi_to_site_angle : ndarray
        从震中（Hypocenter 地表投影）到场点的绝对方位角，[0, 360) 度。
    epi_to_site_angle_refer_strike : ndarray
        上述方位角减去断层走向并包裹到 [-180, 180] 度。
    """
    slon, slat, _, _ = _broadcast_sites(site_lon, site_lat, 0.0)
    absolute = geodesic_azimuth_deg(
        float(hypocenter_lon), float(hypocenter_lat), slon, slat
    )
    relative = _wrap_signed_angle_deg(absolute - (float(strike) % 360.0))
    return absolute, relative


def compute_plane_distances(
    fault_lon: Any,
    fault_lat: Any,
    fault_depth: Any,
    site_lon: Any,
    site_lat: Any,
    site_depth: Any = 0.0,
    *,
    hypocenter_lon: float | None = None,
    hypocenter_lat: float | None = None,
    strike_override: float | None = None,
):
    """
    一次计算单个断层面的距离和方位角量，返回字典。

    v4 对 ``source_to_site_angle`` 使用 NGA-West2 的有限断层定义；
    同时将所有绝对方位角严格规范化到 [0, 360)，避免浮点误差把正北显示为 360°。
    若同时提供 hypocenter_lon / hypocenter_lat，则额外返回旧程序中的
    ``epi_to_site_angle`` 和 ``epi_to_site_angle_refer_strike``；否则这两项为 NaN。
    """
    flon, flat, fdep, _ = _validate_fault_mesh(
        fault_lon, fault_lat, fault_depth
    )
    slon, slat, sdep, _ = _broadcast_sites(site_lon, site_lat, site_depth)

    strike = (
        fault_strike_deg(flon, flat, fdep)
        if strike_override is None
        else float(strike_override) % 360.0
    )

    source_to_site = nga_west2_source_to_site_azimuth_deg(
        flon[0, :], flat[0, :], slon, slat, strike_override=strike
    )

    if (hypocenter_lon is None) != (hypocenter_lat is None):
        raise ValueError(
            "hypocenter_lon 和 hypocenter_lat 必须同时提供或同时省略。"
        )

    if hypocenter_lon is None:
        epi_to_site = np.full(len(slon), np.nan, dtype=float)
        epi_relative = np.full(len(slon), np.nan, dtype=float)
    else:
        epi_to_site, epi_relative = hypocenter_to_site_azimuths_deg(
            hypocenter_lon,
            hypocenter_lat,
            slon,
            slat,
            strike=strike,
        )

    return {
        "rrup": rrup_distance_km(flon, flat, fdep, slon, slat, sdep),
        "rjb": rjb_distance_km(flon, flat, slon, slat),
        "rx": rx_distance_km(flon[0, :], flat[0, :], slon, slat),
        # 保留 v2 的 azimuth 语义，避免旧调用代码失效。
        "azimuth": relative_azimuth_deg(
            flon, flat, fdep, slon, slat, strike_override=strike
        ),
        "source_to_site_angle": source_to_site,
        "epi_to_site_angle": epi_to_site,
        "epi_to_site_angle_refer_strike": epi_relative,
    }


def _parse_Rrup_Rjb_Rx_Azimuth(
    fault_longi: Any,
    fault_lati: Any,
    fault_depth: Any,
    sta_longi: Any,
    sta_lati: Any,
    *,
    site_depth: Any = 0.0,
    hypocenter_lon: float | None = None,
    hypocenter_lat: float | None = None,
    strike_override: float | None = None,
):
    """
    计算单个断层面对应的一组场点距离和方位角。

    v4 输出列
    --------
    Rrup(km)
        场点到离散断层网格节点的最小三维距离。
    Rjb(km)
        Joyner-Boore 距离。
    Rx(km)
        有符号横断层距离；下盘负、上盘正。
    azimuth
        为保持 v2 兼容，仍表示“断层中点 -> 场点”相对于走向的角度，[0, 360)。
    source_to_site_angle
        NGA-West2 有限源 Source-to-Site Azimuth，[-180, 180]。
    epi_to_site_angle
        震中 -> 场点绝对方位角，[0, 360)。仅在提供 hypocenter_lon/lat 时计算。
    epi_to_site_angle_refer_strike
        震中 -> 场点方位角相对 fault strike 的有符号角，[-180, 180]。
    """
    values = compute_plane_distances(
        fault_longi,
        fault_lati,
        fault_depth,
        sta_longi,
        sta_lati,
        site_depth,
        hypocenter_lon=hypocenter_lon,
        hypocenter_lat=hypocenter_lat,
        strike_override=strike_override,
    )

    import pandas as pd

    return pd.DataFrame(
        {
            "Rrup(km)": values["rrup"],
            "Rjb(km)": values["rjb"],
            "Rx(km)": values["rx"],
            "azimuth": values["azimuth"],
            "source_to_site_angle": values["source_to_site_angle"],
            "epi_to_site_angle": values["epi_to_site_angle"],
            "epi_to_site_angle_refer_strike": values[
                "epi_to_site_angle_refer_strike"
            ],
        }
    )


# 同时提供与现有多断层函数命名风格一致的别名，以及无前导下划线的公开别名。
_Parse_Rrup_Rjb_Rx_Azimuth = _parse_Rrup_Rjb_Rx_Azimuth
parse_rrup_rjb_rx_azimuth = _parse_Rrup_Rjb_Rx_Azimuth

# 向后兼容用户原来的单断层函数名。
_parse_Rrup_Rjb_Rx = _parse_Rrup_Rjb_Rx_Azimuth


def _extract_plane(plane: Any):
    """支持含 .lon/.lat/.depth 的对象、字典，或 (lon, lat, depth) 三元组。"""
    if isinstance(plane, Mapping):
        try:
            return plane["lon"], plane["lat"], plane["depth"]
        except KeyError as exc:
            raise KeyError("断层字典必须包含 lon、lat、depth。") from exc
    if (
        hasattr(plane, "lon")
        and hasattr(plane, "lat")
        and hasattr(plane, "depth")
    ):
        return plane.lon, plane.lat, plane.depth
    if isinstance(plane, Sequence) and len(plane) == 3:
        return plane[0], plane[1], plane[2]
    raise TypeError(
        "每个断层面必须是含 .lon/.lat/.depth 的对象、含 lon/lat/depth 的字典，"
        "或 (lon, lat, depth) 三元组。"
    )


def _take_by_row(matrix: np.ndarray, column_index: np.ndarray) -> np.ndarray:
    return matrix[np.arange(matrix.shape[0]), column_index]


def _Parse_Rrup_Rjb_Rx_Azimuth_multi_plane(
    fault_planes: Iterable[Any],
    lon_grid: Any,
    lat_grid: Any,
    *,
    site_depth: Any = 0.0,
    association: str = "rrup_controlled",
    include_plane_index: bool = False,
    hypocenter_lon: float | None = None,
    hypocenter_lat: float | None = None,
):
    """
    v4 多断层面兼容封装。

    每个断层面分别计算 Rrup/Rjb/Rx 以及 NGA-West2 Source-to-Site Azimuth。
    方向性参数按照 ``association`` 指定的控制面进行配对。

    注意
    ----
    该函数仍不是严格的 NGA-West2 MultiSurface/GC2 实现；对于弯曲多分段断层，
    Rx、Ry 和 Source-to-Site Azimuth 的严格定义需要统一的多段几何坐标系。
    本函数适合把若干独立矩形面作为候选面进行聚合。
    """
    planes = list(fault_planes)
    if len(planes) == 0:
        raise ValueError("fault_planes 不能为空。")

    if (hypocenter_lon is None) != (hypocenter_lat is None):
        raise ValueError(
            "hypocenter_lon 和 hypocenter_lat 必须同时提供或同时省略。"
        )

    slon, slat, sdep, _ = _broadcast_sites(lon_grid, lat_grid, site_depth)
    n_sites = len(slon)
    n_planes = len(planes)

    rrup = np.empty((n_sites, n_planes), dtype=float)
    rjb = np.empty((n_sites, n_planes), dtype=float)
    rx = np.empty((n_sites, n_planes), dtype=float)
    legacy_azimuth = np.empty((n_sites, n_planes), dtype=float)
    source_to_site = np.empty((n_sites, n_planes), dtype=float)
    epi_relative = np.empty((n_sites, n_planes), dtype=float)
    epi_absolute = np.full(n_sites, np.nan, dtype=float)

    if hypocenter_lon is not None:
        # 绝对震中->场点方位角与具体断层面无关，仅计算一次。
        epi_absolute = geodesic_azimuth_deg(
            float(hypocenter_lon), float(hypocenter_lat), slon, slat
        )

    for i, plane in enumerate(planes):
        flon, flat, fdep = _extract_plane(plane)
        values = compute_plane_distances(
            flon,
            flat,
            fdep,
            slon,
            slat,
            sdep,
            hypocenter_lon=hypocenter_lon,
            hypocenter_lat=hypocenter_lat,
        )
        rrup[:, i] = values["rrup"]
        rjb[:, i] = values["rjb"]
        rx[:, i] = values["rx"]
        legacy_azimuth[:, i] = values["azimuth"]
        source_to_site[:, i] = values["source_to_site_angle"]
        epi_relative[:, i] = values["epi_to_site_angle_refer_strike"]

    rrup_index = np.argmin(rrup, axis=1)
    rjb_index = np.argmin(rjb, axis=1)
    rx_index = np.argmin(np.abs(rx), axis=1)

    out_rrup = _take_by_row(rrup, rrup_index)
    out_rjb = _take_by_row(rjb, rjb_index)

    if association == "rrup_controlled":
        direction_index = rrup_index
        out_rx = _take_by_row(rx, direction_index)
        out_azimuth = _take_by_row(legacy_azimuth, direction_index)
        out_source_to_site = _take_by_row(source_to_site, direction_index)
        out_epi_relative = _take_by_row(epi_relative, direction_index)
    elif association == "rx_controlled":
        direction_index = rx_index
        out_rx = _take_by_row(rx, direction_index)
        out_azimuth = _take_by_row(legacy_azimuth, direction_index)
        out_source_to_site = _take_by_row(source_to_site, direction_index)
        out_epi_relative = _take_by_row(epi_relative, direction_index)
    elif association == "legacy_independent":
        direction_index = rx_index
        out_rx = _take_by_row(rx, rx_index)
        out_azimuth = np.min(legacy_azimuth, axis=1)
        # 对有符号 Source-to-Site Azimuth 不能按数值最小聚合；为了避免把
        # 不同断层面的正负方向错误拼接，仍跟随 |Rx| 最小的面。
        out_source_to_site = _take_by_row(source_to_site, rx_index)
        out_epi_relative = _take_by_row(epi_relative, rx_index)
    else:
        raise ValueError(
            "association 只能是 'rrup_controlled'、'rx_controlled' 或 "
            "'legacy_independent'。"
        )

    import pandas as pd

    data = {
        "Rrup(km)": out_rrup,
        "Rjb(km)": out_rjb,
        "Rx(km)": out_rx,
        "azimuth": out_azimuth,
        "source_to_site_angle": out_source_to_site,
        "epi_to_site_angle": epi_absolute,
        "epi_to_site_angle_refer_strike": out_epi_relative,
    }
    if include_plane_index:
        data.update(
            {
                "Rrup_plane_index": rrup_index,
                "Rjb_plane_index": rjb_index,
                "direction_plane_index": direction_index,
            }
        )
    return pd.DataFrame(data)


parse_rrup_rjb_rx_azimuth_multi_plane = _Parse_Rrup_Rjb_Rx_Azimuth_multi_plane


__all__ = [
    "EARTH_RADIUS_KM",
    "geodetic_distance_km",
    "azimuth_deg",
    "geodesic_azimuth_deg",
    "spherical_to_cartesian",
    "rrup_distance_km",
    "closest_fault_mesh_point",
    "rjb_distance_km",
    "distance_to_arc_km",
    "distance_to_semi_arc_km",
    "min_distance_to_segment_km",
    "rx_distance_km",
    "fault_middle_point",
    "fault_strike_deg",
    "relative_azimuth_deg",
    "azimuth_of_closest_point_deg",
    "nga_west2_source_to_site_azimuth_deg",
    "hypocenter_to_site_azimuths_deg",
    "compute_plane_distances",
    "_parse_Rrup_Rjb_Rx_Azimuth",
    "_Parse_Rrup_Rjb_Rx_Azimuth",
    "parse_rrup_rjb_rx_azimuth",
    "_Parse_Rrup_Rjb_Rx_Azimuth_multi_plane",
    "parse_rrup_rjb_rx_azimuth_multi_plane",
]


def me():
    pass
