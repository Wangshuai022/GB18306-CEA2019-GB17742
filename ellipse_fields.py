"""GB18306/CEA2019 共用的解析椭圆前向计算内核。

本模块是 Pre、vs_Obs 和 epicenter_inversion 的统一计算 seam：

* ``GB18306EllipseField`` 同时预测宏观烈度、PGA 和 PGV；
* ``CEA2019EllipseField`` 预测 PGA、PGV 和任意 0~6 s 的 PSA；
* ``predict_many`` 支持多个候选宏观震中与多个台站的笛卡尔积，供反演使用；
* ``predict`` 是单一宏观震中的便捷接口，供 Pre 和观测对比使用。

所有距离单位为 km；PGA/PSA 为 gal，PGV 为 cm/s。走向以正北为
0 度、顺时针为正。超出 ``extent`` 定义的最外椭圆时返回 NaN。
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
from pyproj import Transformer

from CEA2019_class import CEA2019
from GB18306_class import GB18306_2015_IntensityCal, GB18306_2015_PGA_PGV_GMMs


def normalize_cea_region(region: str) -> str:
    """CEA2019 区域名归一化为不带“区”的形式。"""
    return str(region).strip().replace("区", "")


def normalize_cea_period(period: float) -> float:
    """把 0 规范为 PGA(-1)，并校验 CEA2019 支持的周期范围。"""
    period = float(period)
    if period == 0.0:
        return -1.0
    if period in (-1.0, -2.0):
        return period
    if period < 0.0:
        raise ValueError("CEA2019 周期只允许 -1(PGA)、-2(PGV)、0(PGA)或正数")
    if period > 6.0:
        raise ValueError(f"CEA2019 不支持 T={period:g}s；最大周期为 6s")
    return period


@cache
def _cea_calculator(region_core: str, axis: str) -> CEA2019:
    return CEA2019(region_core, axis)


@cache
def cea_period_coeffs(region: str, axis: str, period: float, magnitude: float):
    """返回 CEA2019 选定震级分段后的 ``A,B,C,sigma,exp_term``。"""
    region_core = normalize_cea_region(region)
    if axis not in ("长轴", "短轴"):
        raise ValueError("axis 必须是 '长轴' 或 '短轴'")
    period = normalize_cea_period(period)
    magnitude = float(magnitude)
    cal = _cea_calculator(region_core, axis)

    if 0.0 < period < 0.04:
        row_pga = cal._get_coefficients(-1)
        row_04 = cal._get_coefficients(0.04)
        weight = period / 0.04
        row = {
            name: float(row_pga[name])
            + (float(row_04[name]) - float(row_pga[name])) * weight
            for name in ("A1", "B1", "A2", "B2", "C", "D", "E", "σ")
        }
    else:
        row = cal._get_coefficients(period)

    suffix = "1" if magnitude < 6.5 else "2"
    A = float(row[f"A{suffix}"])
    B = float(row[f"B{suffix}"])
    C = float(row["C"])
    sigma = float(row["σ"])
    exp_term = float(row["D"]) * math.exp(float(row["E"]) * magnitude)
    return A, B, C, sigma, exp_term


def _validate_coordinates(lon, lat, sta_lon, sta_lat):
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    sta_lon = np.asarray(sta_lon, dtype=float).ravel()
    sta_lat = np.asarray(sta_lat, dtype=float).ravel()
    if lon.size == 0 or sta_lon.size == 0:
        raise ValueError("候选宏观震中和台站都不能为空")
    if lon.shape != lat.shape:
        raise ValueError("候选宏观震中的 lon/lat 形状必须一致")
    if sta_lon.shape != sta_lat.shape:
        raise ValueError("台站 sta_lon/sta_lat 形状必须一致")
    if not all(np.isfinite(v).all() for v in (lon, lat, sta_lon, sta_lat)):
        raise ValueError("宏观震中和台站经纬度必须是有限数")
    if ((lon < -180) | (lon > 180)).any() or ((sta_lon < -180) | (sta_lon > 180)).any():
        raise ValueError("经度必须位于 [-180, 180]")
    if ((lat < -90) | (lat > 90)).any() or ((sta_lat < -90) | (sta_lat > 90)).any():
        raise ValueError("纬度必须位于 [-90, 90]")
    return lon, lat, sta_lon, sta_lat


def _geometry_groups(lon, lat, strike, sta_lon, sta_lat):
    """按候选点所在 UTM 分区返回 ``indices, R, cos(theta), sin(theta)``。"""
    groups = {}
    for i in range(lon.size):
        zone = min(60, max(1, int((lon[i] + 180.0) // 6.0) + 1))
        hemisphere = 326 if lat[i] >= 0 else 327
        groups.setdefault((hemisphere, zone), []).append(i)

    angle = math.radians(90.0 - (float(strike) % 360.0))
    for (hemisphere, zone), idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=int)
        transformer = Transformer.from_crs(
            "epsg:4326", f"epsg:{hemisphere}{zone:02d}", always_xy=True
        )
        # list 输入可避免 pyproj 对单元素 ndarray 走标量转换路径的弃用警告。
        ex, ey = np.asarray(
            transformer.transform(lon[idx].tolist(), lat[idx].tolist()), dtype=float
        )
        sx, sy = np.asarray(
            transformer.transform(sta_lon.tolist(), sta_lat.tolist()), dtype=float
        )
        dx = (sx[None, :] - ex[:, None]) / 1000.0
        dy = (sy[None, :] - ey[:, None]) / 1000.0
        R = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx) - angle
        yield idx, R, np.cos(theta), np.sin(theta)


class GB18306EllipseField:
    """GB18306 宏观烈度、PGA、PGV 的统一解析椭圆场。

    Parameters
    ----------
    region : str
        ``东部区/中部区/新疆区/青藏区`` 之一。
    Ms : float
        GB18306 衰减关系使用的面波震级。
    extent : float, default 400
        椭圆场长轴最大距离，单位 km；场外预测返回 NaN。

    Notes
    -----
    同一对象预先缓存长短轴系数，并被 Pre、vs_Obs 和震中反演共同调用，
    避免三套应用各自插值产生差异。PGA 单位 gal，PGV 单位 cm/s。
    """

    def __init__(self, region: str, Ms: float, extent: float = 400.0):
        self.region = str(region)
        self.Ms = float(Ms)
        self.extent = float(extent)
        if not math.isfinite(self.Ms):
            raise ValueError("Ms 必须是有限数")
        if not math.isfinite(self.extent) or self.extent <= 0:
            raise ValueError("extent 必须是正有限数")

        self.cal_i = GB18306_2015_IntensityCal()
        self.cal_g = GB18306_2015_PGA_PGV_GMMs()
        self.cal_i._validate_input(self.region, "长轴")
        self.cal_g._validate_input(self.region, "长轴")

        self.IL = self.cal_i._PARAMS[(self.region, "长轴")]
        self.IS = self.cal_i._PARAMS[(self.region, "短轴")]
        self.sigma_I = float(self.IL[4])
        self.gp = {
            kind: {
                "long": self.cal_g._get_params(self.Ms, self.region, "长轴", kind),
                "short": self.cal_g._get_params(self.Ms, self.region, "短轴", kind),
            }
            for kind in ("aE", "vE")
        }

        self.I_lo = (
            self.IL[0]
            + self.IL[1] * self.Ms
            + self.IL[2] * math.log10(self.extent + self.IL[3])
        )
        self.I_hi = (
            self.IL[0] + self.IL[1] * self.Ms + self.IL[2] * math.log10(self.IL[3])
        )
        self.lg_bounds = {}
        for kind in ("aE", "vE"):
            params = self.gp[kind]["long"]
            exp_term = params["D"] * math.exp(params["E"] * self.Ms)
            self.lg_bounds[kind] = (
                params["A"]
                + params["B"] * self.Ms
                + params["C"] * math.log10(self.extent + exp_term),
                params["A"]
                + params["B"] * self.Ms
                + params["C"] * math.log10(exp_term),
            )

    def _ab_intensity(self, intensity):
        A, B, C, R0 = self.IL[:4]
        a = 10.0 ** ((intensity - A - B * self.Ms) / C) - R0
        A, B, C, R0 = self.IS[:4]
        b = 10.0 ** ((intensity - A - B * self.Ms) / C) - R0
        a = np.maximum(a, 1e-3)
        b = np.maximum(np.minimum(b, a), 1e-3)
        return a, b

    def _ab_value(self, value, kind):
        lg_value = np.log10(np.maximum(value, 1e-12))
        long_params = self.gp[kind]["long"]
        short_params = self.gp[kind]["short"]

        def radius(params):
            return 10.0 ** (
                (lg_value - params["A"] - params["B"] * self.Ms) / params["C"]
            ) - params["D"] * math.exp(params["E"] * self.Ms)

        a = np.maximum(radius(long_params), 1e-3)
        b = np.maximum(np.minimum(radius(short_params), a), 1e-3)
        return a, b

    @staticmethod
    def _bisect(R, ct, st, radii, lo_value, hi_value, log_values=False):
        lo = np.full_like(R, lo_value, dtype=float)
        hi = np.full_like(R, hi_value, dtype=float)
        for _ in range(60):
            mid = (lo + hi) / 2.0
            a, b = radii(10.0**mid if log_values else mid)
            f = (R * ct / a) ** 2 + (R * st / b) ** 2 - 1.0
            lo = np.where(f < 0.0, mid, lo)
            hi = np.where(f >= 0.0, mid, hi)
        result = (lo + hi) / 2.0
        return 10.0**result if log_values else result

    def predict_many(self, lon, lat, strike, sta_lon, sta_lat):
        """预测多个候选震中与多个台站的笛卡尔积。

        ``lon/lat`` 是长度 N 的候选宏观震中，``sta_lon/sta_lat`` 是长度 M
        的台站坐标，``strike`` 单位为度。返回九个形状 ``(N, M)`` 的数组：
        ``I, PGA, PGV, aI, bI, aPGA, bPGA, aPGV, bPGV``；a/b 为预测值所在
        等值椭圆的长短轴距（km）。
        """
        lon, lat, sta_lon, sta_lat = _validate_coordinates(lon, lat, sta_lon, sta_lat)
        shape = (lon.size, sta_lon.size)
        outputs = [np.full(shape, np.nan) for _ in range(9)]
        I_out, A_out, V_out, aI_out, bI_out, aA_out, bA_out, aV_out, bV_out = outputs

        for idx, R, ct, st in _geometry_groups(lon, lat, strike, sta_lon, sta_lat):
            a0, b0 = self._ab_intensity(self.I_lo)
            r_outer = (
                float(a0)
                * float(b0)
                / np.sqrt((float(b0) * ct) ** 2 + (float(a0) * st) ** 2)
            )
            inside = R <= r_outer
            intensity = self._bisect(
                R, ct, st, self._ab_intensity, self.I_lo, self.I_hi
            )
            aI, bI = self._ab_intensity(intensity)
            I_out[idx, :] = np.where(inside, intensity, np.nan)
            aI_out[idx, :] = np.where(inside, aI, np.nan)
            bI_out[idx, :] = np.where(inside, bI, np.nan)

            for kind, value_out, a_out, b_out in (
                ("aE", A_out, aA_out, bA_out),
                ("vE", V_out, aV_out, bV_out),
            ):
                lo, hi = self.lg_bounds[kind]
                a0, b0 = self._ab_value(10.0**lo, kind)
                r_outer = (
                    float(a0)
                    * float(b0)
                    / np.sqrt((float(b0) * ct) ** 2 + (float(a0) * st) ** 2)
                )
                inside = R <= r_outer
                value = self._bisect(
                    R,
                    ct,
                    st,
                    lambda v, current=kind: self._ab_value(v, current),
                    lo,
                    hi,
                    log_values=True,
                )
                a, b = self._ab_value(value, kind)
                value_out[idx, :] = np.where(inside, value, np.nan)
                a_out[idx, :] = np.where(inside, a, np.nan)
                b_out[idx, :] = np.where(inside, b, np.nan)
        return tuple(outputs)

    def predict(self, lon, lat, strike, sta_lon, sta_lat):
        """预测单个宏观震中处的多个台站。

        参数含义与 ``predict_many`` 相同，但 ``lon/lat`` 为标量；返回九个
        长度 M 的一维数组，顺序与 ``predict_many`` 完全一致。
        """
        result = self.predict_many(
            np.atleast_1d(lon), np.atleast_1d(lat), strike, sta_lon, sta_lat
        )
        return tuple(item[0] for item in result)


class CEA2019EllipseField:
    """CEA2019 PGA、PGV、PSA 的统一解析椭圆场。

    Parameters
    ----------
    region : str
        CEA2019 分区，可带或不带末尾“区”。
    Ms : float
        CEA2019 系数分段使用的震级。
    extent : float, default 400
        椭圆场最大长轴距，单位 km；场外返回 NaN。

    Notes
    -----
    -1=PGA、-2=PGV、正数=PSA 周期（0--6 s）。PGA/PSA 单位 gal，PGV
    单位 cm/s；所有 a/b 返回值单位为 km。
    """

    def __init__(self, region: str, Ms: float, extent: float = 400.0):
        self.region = normalize_cea_region(region)
        self.Ms = float(Ms)
        self.extent = float(extent)
        if not math.isfinite(self.Ms):
            raise ValueError("Ms 必须是有限数")
        if not math.isfinite(self.extent) or self.extent <= 0:
            raise ValueError("extent 必须是正有限数")
        # 构造时即验证区域和系数表存在。
        _cea_calculator(self.region, "长轴")
        _cea_calculator(self.region, "短轴")

    def coefficients(self, period, axis="长轴"):
        """返回 ``(A, B, C, sigma, D*exp(E*M))`` 的当前震级系数。"""
        return cea_period_coeffs(self.region, axis, period, self.Ms)

    def sigma(self, period):
        """返回指定周期长轴模型的 log10 标准差 σ。"""
        return self.coefficients(period, "长轴")[3]

    def predict_period_many(self, period, lon, lat, strike, sta_lon, sta_lat):
        """预测一个周期下 N 个候选震中 × M 个台站。

        ``period`` 为 -1(PGA)、-2(PGV) 或 0--6 s PSA 周期；坐标单位为度，
        ``strike`` 以正北为 0°顺时针。返回 ``(value, a_eq, b_eq)`` 三个
        ``(N, M)`` 数组，其中 a/b 单位为 km。
        """
        period = normalize_cea_period(period)
        lon, lat, sta_lon, sta_lat = _validate_coordinates(lon, lat, sta_lon, sta_lat)
        shape = (lon.size, sta_lon.size)
        value_out = np.full(shape, np.nan)
        a_out = np.full(shape, np.nan)
        b_out = np.full(shape, np.nan)

        A_l, B_l, C_l, _, e_l = self.coefficients(period, "长轴")
        A_s, B_s, C_s, _, e_s = self.coefficients(period, "短轴")
        lg_lo = A_l + B_l * self.Ms - C_l * math.log10(self.extent + e_l)
        lg_hi = A_l + B_l * self.Ms - C_l * math.log10(e_l)

        for idx, R, ct, st in _geometry_groups(lon, lat, strike, sta_lon, sta_lat):
            a0 = self.extent
            b0 = min(10.0 ** ((A_s + B_s * self.Ms - lg_lo) / C_s) - e_s, a0)
            b0 = max(b0, 1e-3)
            r_outer = a0 * b0 / np.sqrt((b0 * ct) ** 2 + (a0 * st) ** 2)
            inside = R <= r_outer

            lo = np.full_like(R, lg_lo)
            hi = np.full_like(R, lg_hi)
            for _ in range(60):
                mid = (lo + hi) / 2.0
                a = 10.0 ** ((A_l + B_l * self.Ms - mid) / C_l) - e_l
                b = 10.0 ** ((A_s + B_s * self.Ms - mid) / C_s) - e_s
                a = np.maximum(a, 1e-3)
                b = np.maximum(np.minimum(b, a), 1e-3)
                f = (R * ct / a) ** 2 + (R * st / b) ** 2 - 1.0
                lo = np.where(f < 0.0, mid, lo)
                hi = np.where(f >= 0.0, mid, hi)
            mid = (lo + hi) / 2.0
            a = np.maximum(10.0 ** ((A_l + B_l * self.Ms - mid) / C_l) - e_l, 1e-3)
            b = np.maximum(10.0 ** ((A_s + B_s * self.Ms - mid) / C_s) - e_s, 1e-3)
            b = np.minimum(b, a)
            value_out[idx, :] = np.where(inside, 10.0**mid, np.nan)
            a_out[idx, :] = np.where(inside, a, np.nan)
            b_out[idx, :] = np.where(inside, b, np.nan)
        return value_out, a_out, b_out

    def predict_period(self, period, lon, lat, strike, sta_lon, sta_lat):
        """预测单个宏观震中、单个周期下的全部台站。

        返回三个长度 M 的数组 ``(value, a_eq, b_eq)``。
        """
        result = self.predict_period_many(
            period,
            np.atleast_1d(lon),
            np.atleast_1d(lat),
            strike,
            sta_lon,
            sta_lat,
        )
        return tuple(item[0] for item in result)

    def predict_many(self, periods, lon, lat, strike, sta_lon, sta_lat):
        """一次预测多个周期和多个候选震中。

        返回 ``{规范周期: (value, a_eq, b_eq)}``；每个数组形状均为
        ``(候选震中数, 台站数)``。
        """
        return {
            normalize_cea_period(period): self.predict_period_many(
                period, lon, lat, strike, sta_lon, sta_lat
            )
            for period in periods
        }

    def predict(self, periods, lon, lat, strike, sta_lon, sta_lat):
        """预测单个宏观震中的多个周期。

        返回 ``{规范周期: (value, a_eq, b_eq)}``；每个数组长度为台站数。
        """
        return {
            period: tuple(item[0] for item in result)
            for period, result in self.predict_many(
                periods,
                np.atleast_1d(lon),
                np.atleast_1d(lat),
                strike,
                sta_lon,
                sta_lat,
            ).items()
        }
