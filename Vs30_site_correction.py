"""把实际台站观测统一换算到指定 Vs30 参考场地。

场地修正只调用项目内成熟的 ``CB14_site_correct.py`` 接口，不直接调用其
底层实现。由于 CB14 场地项随 A1100（Vs30=1100 m/s 参考岩石 PGA）非线性
变化，本模块先由实际场地观测 PGA 反解 A1100，再使用同一个 A1100 计算
实际场地相对目标参考场地的 PGA/PGV/PSA 修正系数。

中国 Vs30 CSV 很大（约 1.4 GB），读取时按块筛选台站包围盒，不会整表载入。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from CB14_site_correct import _parse_CB14_site_correct_factor_all_period
except ImportError as exc:  # pragma: no cover - 仅在依赖缺失时触发
    raise ImportError(
        "缺少 CB14_site_correct.py；该文件必须与 Vs30 场地修正程序位于同一目录。"
    ) from exc


DEFAULT_VS30_PATH = (
    r"D:\Ubuntu_share\0.Simulate_plat\Database\8.Vs30_data\China_area_Vs30.csv"
)
DEFAULT_REFERENCE_VS30 = 500.0
CB14_REFERENCE_ROCK_VS30 = 1100.0

def _read_observation_table(data) -> pd.DataFrame:
    """读取反演观测表，同时保留所有原始地震动参数列。"""
    if isinstance(data, (str, os.PathLike)):
        df = pd.read_csv(str(data), sep="\t", encoding="utf-8")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data 必须是制表符文本路径或 pandas.DataFrame")
    df.columns = [str(col).strip() for col in df.columns]
    if "Sta_ID" not in df.columns:
        df["Sta_ID"] = [f"S{i + 1}" for i in range(len(df))]
    lon_col = "longi" if "longi" in df.columns else "lon"
    lat_col = "lati" if "lati" in df.columns else "lat"
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError("观测表必须包含 longi/lati 或 lon/lat")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        raise ValueError("观测表没有经纬度有效的台站")
    df = df.loc[valid].reset_index(drop=True)
    return df


def _coord_columns(df: pd.DataFrame) -> tuple[str, str]:
    lon_col = "longi" if "longi" in df.columns else "lon"
    lat_col = "lati" if "lati" in df.columns else "lat"
    return lon_col, lat_col


def query_station_vs30(
    stations: pd.DataFrame,
    vs30_path: str | os.PathLike = DEFAULT_VS30_PATH,
    *,
    padding_deg: float = 0.1,
    chunksize: int = 1_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """从大型中国 Vs30 网格中查询每个台站的最近邻值。

    Parameters
    ----------
    stations : pandas.DataFrame
        台站表，必须包含 ``longi/lati`` 或 ``lon/lat``；坐标单位为度。
    vs30_path : str or os.PathLike
        CSV 文件路径。文件必须包含 ``longi``、``lati``、``Vs30(m/s)``。
    padding_deg : float, default 0.1
        台站包围盒向外扩展的角度，用来减少大型 CSV 的候选网格范围。
    chunksize : int, default 1000000
        每次读取的 CSV 行数。函数不会把完整的约 1.4 GB 文件载入内存。
    verbose : bool, default True
        是否打印扫描进度、Vs30 范围和最大最近邻距离。

    Returns
    -------
    pandas.DataFrame
        行序与 ``stations`` 一致，包含 ``Vs30(m/s)``、最近网格点经纬度和
        ``Vs30_grid_distance_deg``。最近邻在经纬度平面中通过 cKDTree 求取，
        与原 ``_parse_Sta_Vs30_general`` 的定义一致。

    Raises
    ------
    FileNotFoundError
        Vs30 CSV 路径不存在。
    ValueError
        台站坐标无效，或网格没有覆盖台站包围盒。
    """
    path = Path(vs30_path)
    if not path.is_file():
        raise FileNotFoundError(f"Vs30 数据不存在：{path}")
    lon_col, lat_col = _coord_columns(stations)
    sta_lon = pd.to_numeric(stations[lon_col], errors="coerce").to_numpy(float)
    sta_lat = pd.to_numeric(stations[lat_col], errors="coerce").to_numpy(float)
    if not np.isfinite(sta_lon).all() or not np.isfinite(sta_lat).all():
        raise ValueError("台站经纬度含非有限值")

    pad = max(float(padding_deg), 0.02)
    lon_lo, lon_hi = float(sta_lon.min() - pad), float(sta_lon.max() + pad)
    lat_lo, lat_hi = float(sta_lat.min() - pad), float(sta_lat.max() + pad)
    if verbose:
        print(
            f"[Vs30] 分块读取 {path.name}，筛选范围 "
            f"lon={lon_lo:.3f}~{lon_hi:.3f}, lat={lat_lo:.3f}~{lat_hi:.3f}"
        )

    selected: list[pd.DataFrame] = []
    previous_mean_lat = None
    descending = None
    reached_lat_window = False
    reader = pd.read_csv(
        path,
        usecols=["longi", "lati", "Vs30(m/s)"],
        dtype={"longi": "float32", "lati": "float32", "Vs30(m/s)": "float32"},
        chunksize=max(int(chunksize), 10_000),
        memory_map=True,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        chunk = chunk[np.isfinite(chunk["Vs30(m/s)"]) & (chunk["Vs30(m/s)"] > 0)]
        if chunk.empty:
            continue
        mean_lat = float(chunk["lati"].mean())
        if previous_mean_lat is not None and descending is None:
            descending = mean_lat < previous_mean_lat
        previous_mean_lat = mean_lat
        chunk_min_lat = float(chunk["lati"].min())
        chunk_max_lat = float(chunk["lati"].max())
        if chunk_min_lat <= lat_hi and chunk_max_lat >= lat_lo:
            reached_lat_window = True
            mask = (
                chunk["longi"].between(lon_lo, lon_hi)
                & chunk["lati"].between(lat_lo, lat_hi)
            )
            if mask.any():
                selected.append(chunk.loc[mask].copy())
        if reached_lat_window and descending is True and chunk_max_lat < lat_lo:
            break
        if reached_lat_window and descending is False and chunk_min_lat > lat_hi:
            break
        if verbose and chunk_index % 10 == 0:
            print(f"[Vs30] 已扫描 {chunk_index * chunksize:,} 行……")

    if not selected:
        raise ValueError(
            "Vs30 数据中没有覆盖台站包围盒的有效网格点；请检查数据范围和经纬度"
        )
    grid = pd.concat(selected, ignore_index=True).drop_duplicates(
        subset=["longi", "lati"]
    )
    points = grid[["longi", "lati"]].to_numpy(float)
    tree = cKDTree(points)
    distance, index = tree.query(np.column_stack([sta_lon, sta_lat]), k=1)
    nearest = grid.iloc[np.asarray(index, dtype=int)].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "Vs30(m/s)": nearest["Vs30(m/s)"].to_numpy(float),
            "Vs30_grid_longi": nearest["longi"].to_numpy(float),
            "Vs30_grid_lati": nearest["lati"].to_numpy(float),
            "Vs30_grid_distance_deg": np.asarray(distance, dtype=float),
        }
    )
    if verbose:
        print(
            f"[Vs30] {len(out)} 个台站完成："
            f"{out['Vs30(m/s)'].min():.1f}~{out['Vs30(m/s)'].max():.1f} m/s，"
            f"最大最近邻距离 {out['Vs30_grid_distance_deg'].max():.5f}°"
        )
    return out


def _cb14_factor_table(
    actual_vs30: float,
    reference_pga_gal: float,
    reference_vs30: float,
    region_name: str,
    use_basin: bool,
    z25_actual: float | None = None,
    z25_reference: float | None = None,
) -> pd.DataFrame:
    """只通过用户的 CB14_site_correct.py 获取全周期场地倍率。"""
    table = _parse_CB14_site_correct_factor_all_period(
        Vs30=float(actual_vs30),
        PGAr=float(reference_pga_gal),
        Vref=float(reference_vs30),
        Region_name=region_name,
        Use_Basin=bool(use_basin),
        Z25_simul=z25_reference,
        Z25_real=z25_actual,
    )
    out = table[["period", "Site_corrected"]].copy()
    out["period"] = pd.to_numeric(out["period"], errors="coerce")
    out["Site_corrected"] = pd.to_numeric(
        out["Site_corrected"], errors="coerce"
    )
    out = out.dropna().sort_values("period").reset_index(drop=True)
    if out.empty or (out["Site_corrected"] <= 0).any():
        raise RuntimeError("CB14_site_correct.py 返回了无效场地修正系数")
    return out


def _factor_from_table(table: pd.DataFrame, period: float) -> float:
    period = -1.0 if float(period) == 0.0 else float(period)
    exact = table.loc[np.isclose(table["period"], period), "Site_corrected"]
    if not exact.empty:
        return float(exact.iloc[0])
    if period <= 0:
        raise ValueError(f"CB14_site_correct.py 不支持周期 {period:g}")
    positive = table[table["period"] > 0].sort_values("period")
    if positive.empty:
        raise RuntimeError("CB14_site_correct.py 未返回 PSA 场地修正系数")
    return float(
        np.interp(
            period,
            positive["period"].to_numpy(float),
            positive["Site_corrected"].to_numpy(float),
        )
    )


def solve_cb14_a1100(
    observed_pga_gal: float,
    actual_vs30: float,
    reference_vs30: float = DEFAULT_REFERENCE_VS30,
    *,
    region_name: str = "CH",
    use_basin: bool = False,
    z25_actual: float | None = None,
    z25_reference: float | None = None,
    z25_a1100: float | None = None,
    rtol: float = 1e-5,
    max_iterations: int = 50,
) -> tuple[float, int, pd.DataFrame]:
    """由实际场地观测 PGA 反解 A1100，并返回实际/目标场地倍率表。

    ``CB14_site_correct.py`` 的 ``PGAr`` 实际传入底层 CB14 的 A1100，即
    Vs30=1100 m/s 参考岩石 PGA。先求解

    ``PGA_observed = A1100 * F_PGA(actual_vs30 / 1100, A1100)``，

    再以该 A1100 计算 ``actual_vs30 / reference_vs30`` 的全周期倍率。
    """
    observed = float(observed_pga_gal)
    actual_vs30 = float(actual_vs30)
    reference_vs30 = float(reference_vs30)
    if not np.isfinite(observed) or observed <= 0:
        raise ValueError(f"用于 CB14 非线性反解的 PGA 必须为正值：{observed_pga_gal}")
    if not np.isfinite(actual_vs30) or actual_vs30 <= 0:
        raise ValueError(f"Vs30 必须为正值：{actual_vs30}")
    if not np.isfinite(reference_vs30) or reference_vs30 <= 0:
        raise ValueError(f"参考 Vs30 必须为正值：{reference_vs30}")

    a1100 = observed
    previous_factor = None
    for iteration in range(1, int(max_iterations) + 1):
        actual_over_1100 = _cb14_factor_table(
            actual_vs30,
            a1100,
            CB14_REFERENCE_ROCK_VS30,
            region_name,
            use_basin,
            z25_actual,
            z25_a1100,
        )
        factor = _factor_from_table(actual_over_1100, -1.0)
        updated = observed / factor
        converged = abs(updated - a1100) <= float(rtol) * max(
            abs(updated), 1.0
        )
        factor_stable = previous_factor is not None and factor == previous_factor
        a1100 = float(updated)
        if converged or factor_stable:
            final_table = _cb14_factor_table(
                actual_vs30,
                a1100,
                reference_vs30,
                region_name,
                use_basin,
                z25_actual,
                z25_reference,
            )
            return a1100, iteration, final_table
        previous_factor = factor
    raise RuntimeError(
        "CB14_site_correct.py 的 A1100 反解未收敛："
        f"PGA={observed_pga_gal:g} gal, Vs30={actual_vs30:g} m/s, "
        f"最大迭代次数={max_iterations}"
    )


def iterate_cb14_reference_pga(*args, **kwargs):
    """兼容旧调用；返回值第一项现为严格定义的 A1100。"""
    return solve_cb14_a1100(*args, **kwargs)


def cb14_site_factor_actual_over_reference(
    period: float,
    reference_pga_gal: float,
    actual_vs30: float,
    reference_vs30: float = DEFAULT_REFERENCE_VS30,
    *,
    region_name: str = "CH",
    use_basin: bool = False,
    z25_actual: float | None = None,
    z25_reference: float | None = None,
) -> float:
    """返回指定周期的 CB14“实际场地/参考场地”地震动倍率。

    Parameters
    ----------
    period : float
        -1=PGA，-2=PGV，正数=PSA 周期（s）。
    reference_pga_gal : float
        CB14 的 A1100，即 Vs30=1100 m/s 参考岩石 PGA，单位 gal。
    actual_vs30, reference_vs30 : float
        实际台站和目标参考场地的 Vs30，单位 m/s。
    region_name : str, default "CH"
        CB14 区域代码。
    use_basin : bool, default False
        是否启用 Z2.5 盆地项。
    z25_actual, z25_reference : float or None
        实际场地和目标参考场地的 Z2.5，单位遵循用户 CB14 包（km）。

    Returns
    -------
    float
        正倍率 ``Y_actual / Y_reference``。因此把观测换算到参考场地时应除以
        该倍率，而不是相乘。
    """
    table = _cb14_factor_table(
        actual_vs30,
        reference_pga_gal,
        reference_vs30,
        region_name,
        use_basin,
        z25_actual,
        z25_reference,
    )
    return _factor_from_table(table, period)


def _period_label(period: float) -> str:
    if float(period) == -1.0:
        return "PGA"
    if float(period) == -2.0:
        return "PGV"
    return f"PSA_T{float(period):.2f}s"


def _columns_for_period(df: pd.DataFrame, period: float) -> list[str]:
    period = float(period)
    if period == -1.0:
        candidates = [
            "PGA_RotD50",
            "PGA_H",
            "EPA_RotD50",
            "EPA_H",
        ]
    elif period == -2.0:
        candidates = [
            "PGV_RotD50",
            "PGV_H",
            "EPV_RotD50",
            "EPV_H",
        ]
    else:
        tag = f"pSa(T={period:.2f}s)"
        candidates = [f"{tag}_RotD50", f"{tag}_H"]
    return [col for col in candidates if col in df.columns]


def _pga_driver(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    candidates = [
        "PGA_RotD50",
        "PGA_H",
        "EPA_RotD50",
        "EPA_H",
    ]
    values = np.full(len(df), np.nan)
    sources = np.full(len(df), "", dtype=object)
    for col in candidates:
        if col not in df.columns:
            continue
        current = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        take = ~np.isfinite(values) & np.isfinite(current) & (current > 0)
        values[take] = current[take]
        sources[take] = col
    if not np.isfinite(values).all():
        bad = df.loc[~np.isfinite(values), "Sta_ID"].astype(str).tolist()
        raise ValueError(
            "CB14 场地非线性修正需要每个台站的正 PGA；缺失台站："
            + ", ".join(bad[:10])
        )
    return values, sources


def correct_observations_to_reference_vs30(
    data,
    periods: Iterable[float],
    *,
    vs30_path: str | os.PathLike = DEFAULT_VS30_PATH,
    reference_vs30: float = DEFAULT_REFERENCE_VS30,
    region_name: str = "CH",
    use_basin: bool = False,
    chunksize: int = 1_000_000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把指定 PGA、PGV、PSA 观测统一换算到目标参考 Vs30。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        原始台站观测表。必须含台站编号、经纬度和正 PGA；可直接包含
        ``Vs30(m/s)``，缺失时从 ``vs30_path`` 查询。地震动列严格按
        RotD50、H、有效值 RotD50、有效值 H 的优先级识别。
    periods : iterable of float
        需要修正的参数：-1=PGA，-2=PGV，正数=PSA 周期（s）。
    vs30_path : str or os.PathLike
        中国 Vs30 网格路径，只为缺少有效 Vs30 的台站读取。
    reference_vs30 : float, default 500
        目标参考场地 Vs30，单位 m/s。
    region_name : str, default "CH"
        传给用户 ``CB14_site_correct.py`` 的区域代码。
    use_basin : bool, default False
        是否启用 CB14 Z2.5 盆地项。
    chunksize : int, default 1000000
        Vs30 CSV 分块读取行数。
    verbose : bool, default True
        是否打印查询和修正摘要。

    Returns
    -------
    corrected_data : pandas.DataFrame
        原观测列替换为参考场地值，并新增 ``*_raw`` 和
        ``*_Vs30_<reference>`` 列。attrs 中记录参考 Vs30、CB14 和盆地开关。
    audit_table : pandas.DataFrame
        逐台站审计信息，包括实际/参考 Vs30、PGA 驱动列、反解 A1100、
        迭代次数、各周期倍率、原始值和修正值。

    Notes
    -----
    每个台站先解
    ``PGA_obs = A1100 * F_PGA(Vs30_actual/1100, A1100)``，再以同一个
    A1100 计算所有周期的 ``actual/reference`` 倍率。观测统一采用
    ``Y_reference = Y_observed / F``。

    Raises
    ------
    ValueError
        周期为空、Vs30/PGA 无效，或所需观测列不存在。
    """
    reference_vs30 = float(reference_vs30)
    if not np.isfinite(reference_vs30) or reference_vs30 <= 0:
        raise ValueError("reference_vs30 必须为正值")
    normalized_periods = []
    for period in periods:
        value = -1.0 if float(period) == 0.0 else float(period)
        if value not in normalized_periods:
            normalized_periods.append(value)
    if not normalized_periods:
        raise ValueError("至少指定一个需要场地修正的 PGA/PGV/PSA 参数")

    corrected = _read_observation_table(data)
    existing_vs30 = (
        pd.to_numeric(corrected["Vs30(m/s)"], errors="coerce")
        if "Vs30(m/s)" in corrected.columns
        else pd.Series(np.nan, index=corrected.index, dtype=float)
    )
    need_query = ~(np.isfinite(existing_vs30) & (existing_vs30 > 0))
    grid_info = pd.DataFrame(index=corrected.index)
    if need_query.any():
        queried = query_station_vs30(
            corrected.loc[need_query].reset_index(drop=True),
            vs30_path,
            chunksize=chunksize,
            verbose=verbose,
        )
        existing_vs30.loc[need_query] = queried["Vs30(m/s)"].to_numpy(float)
        for col in queried.columns[1:]:
            grid_info[col] = np.nan
            grid_info.loc[need_query, col] = queried[col].to_numpy(float)
    corrected["Vs30(m/s)"] = existing_vs30.to_numpy(float)
    if not np.isfinite(corrected["Vs30(m/s)"]).all():
        raise ValueError("部分台站未取得有效 Vs30")

    pga_driver, pga_driver_source = _pga_driver(corrected)
    vs30_values = corrected["Vs30(m/s)"].to_numpy(float)
    pga_solutions = [
        solve_cb14_a1100(
            pga,
            vs,
            reference_vs30,
            region_name=region_name,
            use_basin=use_basin,
        )
        for pga, vs in zip(pga_driver, vs30_values)
    ]
    a1100_values = np.array([item[0] for item in pga_solutions], dtype=float)
    iteration_counts = np.array([item[1] for item in pga_solutions], dtype=int)
    factor_tables = [item[2] for item in pga_solutions]
    pga_factors = np.array(
        [_factor_from_table(table, -1.0) for table in factor_tables], dtype=float
    )
    pga_at_reference = pga_driver / pga_factors
    audit = pd.DataFrame(
        {
            "Sta_ID": corrected["Sta_ID"].astype(str),
            "Vs30_actual_mps": vs30_values,
            "Vs30_reference_mps": reference_vs30,
            "CB14_region": region_name,
            "CB14_use_basin": bool(use_basin),
            "PGA_driver_column": pga_driver_source,
            "PGA_driver_raw_gal": pga_driver,
            "CB14_A1100_gal": a1100_values,
            "CB14_A1100_iterations": iteration_counts,
            f"PGA_driver_Vs30_{reference_vs30:g}_gal": pga_at_reference,
        }
    )
    for col in grid_info.columns:
        audit[col] = grid_info[col].to_numpy()

    suffix = f"Vs30_{reference_vs30:g}"
    for period in normalized_periods:
        columns = _columns_for_period(corrected, period)
        if not columns:
            raise ValueError(
                f"周期 {period:g} 没有可修正的水平向观测列；"
                "请检查 PGA/PGV/PSA 列名"
            )
        factors = np.array(
            [_factor_from_table(table, period) for table in factor_tables],
            dtype=float,
        )
        label = _period_label(period)
        audit[f"CB14_factor_{label}_actual_over_reference"] = factors
        for col in columns:
            raw = pd.to_numeric(corrected[col], errors="coerce").to_numpy(float)
            reference_value = raw / factors
            corrected[f"{col}_raw"] = raw
            corrected[f"{col}_{suffix}"] = reference_value
            corrected[col] = reference_value
            audit[f"{col}_raw"] = raw
            audit[f"{col}_{suffix}"] = reference_value

    if verbose:
        print(
            f"[CB14] 已将 {len(corrected)} 个台站的 "
            f"{[_period_label(p) for p in normalized_periods]} 观测统一到 "
            f"Vs30={reference_vs30:g} m/s（盆地项={use_basin}）"
        )
    corrected.attrs["site_reference_vs30"] = reference_vs30
    corrected.attrs["site_correction_model"] = "CB14"
    corrected.attrs["site_use_basin"] = bool(use_basin)
    return corrected, audit


def attach_site_audit_to_result(
    result: dict,
    audit: pd.DataFrame,
    corrected_data: pd.DataFrame,
    *,
    outpath: str | os.PathLike | None = None,
) -> dict:
    """把 Vs30 审计信息合并进基础反演结果。

    Parameters
    ----------
    result : dict
        GB18306/CEA2019 基础反演函数返回值，必须含逐台站 ``table``。
    audit : pandas.DataFrame
        ``correct_observations_to_reference_vs30`` 返回的审计表。
    corrected_data : pandas.DataFrame
        统一到目标 Vs30 的完整观测表。
    outpath : str, os.PathLike or None
        合并后统计表的 TSV 输出路径；None 表示不写文件。

    Returns
    -------
    dict
        原 ``result``（原位补充），新增 ``site_correction``、
        ``corrected_observations``、``site_reference_vs30`` 和 ``site_model``；
        ``table`` 追加所有审计列。

    Raises
    ------
    RuntimeError
        反演台站数与场地修正审计表行数不同。
    """
    result["site_correction"] = audit.copy()
    result["corrected_observations"] = corrected_data.copy()
    result["site_reference_vs30"] = float(audit["Vs30_reference_mps"].iloc[0])
    result["site_model"] = "CB14"
    table = result["table"].reset_index(drop=True).copy()
    audit_values = audit.drop(columns=["Sta_ID"], errors="ignore").reset_index(drop=True)
    if len(table) != len(audit_values):
        raise RuntimeError("反演台站表与场地修正审计表行数不一致")
    result["table"] = pd.concat([table, audit_values], axis=1)
    if outpath is not None:
        path = Path(outpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        result["table"].to_csv(path, sep="\t", index=False, encoding="utf-8")
        print(f"[导出] Vs30 场地修正反演统计：{path}")
    return result


def prepare_observations_for_site_plot(
    data,
    periods: Iterable[float],
    plot_observations: str = "corrected",
    *,
    correction_kwargs: dict | None = None,
) -> pd.DataFrame:
    """为 ``vs_Obs`` 绘图统一准备 corrected/raw 台站观测。

    本函数是文件路径和 DataFrame 的统一入口：原始输入会先调用项目内的
    ``CB14_site_correct.py`` 修正到目标参考场地；已经由
    :func:`correct_observations_to_reference_vs30` 修正过的 DataFrame 会直接
    复用，避免二次场地修正。最后再根据 ``plot_observations`` 返回修正值或
    恢复后的原始值。

    Parameters
    ----------
    data : str, os.PathLike or pandas.DataFrame
        制表符观测文件、原始观测 DataFrame，或已包含场地修正审计列的
        DataFrame。原始输入必须包含有效PGA及经纬度；若没有 ``Vs30(m/s)``，
        使用默认中国Vs30网格查询。
    periods : iterable of float
        需要参与场地修正的参数：-1=PGA、-2=PGV、正数=PSA周期（s）。
        不应传入烈度字符串。
    plot_observations : {"corrected", "raw"}, default "corrected"
        ``corrected`` 返回参考Vs30观测；``raw`` 返回原始场地观测。
    correction_kwargs : dict or None
        可选地传给 :func:`correct_observations_to_reference_vs30`，例如
        ``vs30_path``、``reference_vs30``、``region_name``、``use_basin``、
        ``chunksize`` 和 ``verbose``。

    Returns
    -------
    pandas.DataFrame
        可直接传给GB18306/CEA2019预测—观测计算的绘图数据。attrs记录
        ``site_plot_observations``；corrected模式还保留参考Vs30与CB14信息。

    Raises
    ------
    ValueError
        模式非法、没有可修正参数，或输入是部分修正DataFrame且缺少本次请求
        周期。部分修正表不会被再次修正，以避免PGA/PGV重复除以场地倍率。
    """
    mode = str(plot_observations).strip().lower()
    if mode not in {"corrected", "raw"}:
        raise ValueError(
            "plot_observations 必须为 'corrected' 或 'raw'，"
            f"收到：{plot_observations!r}"
        )

    normalized_periods = []
    for period in periods:
        value = -1.0 if float(period) == 0.0 else float(period)
        if value not in normalized_periods:
            normalized_periods.append(value)
    if not normalized_periods:
        raise ValueError("corrected/raw 场地绘图至少需要一个PGA、PGV或PSA参数")

    kwargs = dict(correction_kwargs or {})
    corrected_data = None
    if isinstance(data, pd.DataFrame) and "site_reference_vs30" in data.attrs:
        requested_reference = float(
            kwargs.get("reference_vs30", data.attrs["site_reference_vs30"])
        )
        existing_reference = float(data.attrs["site_reference_vs30"])
        if not np.isclose(requested_reference, existing_reference):
            raise ValueError(
                "输入DataFrame已经修正到 "
                f"Vs30={existing_reference:g} m/s，不能直接改为 "
                f"Vs30={requested_reference:g} m/s；请改用原始观测"
            )
        missing_periods = []
        for period in normalized_periods:
            columns = _columns_for_period(data, period)
            if not columns or any(f"{col}_raw" not in data.columns for col in columns):
                missing_periods.append(period)
        if missing_periods:
            raise ValueError(
                "输入DataFrame只完成了部分场地修正，缺少周期 "
                f"{missing_periods} 的 *_raw 审计列；请改用原始观测，避免重复修正"
            )
        corrected_data = data

    if corrected_data is None:
        corrected_data, _ = correct_observations_to_reference_vs30(
            data,
            normalized_periods,
            **kwargs,
        )
    return prepare_site_plot_observations(corrected_data, mode)


def prepare_site_plot_observations(
    corrected_data: pd.DataFrame,
    plot_observations: str = "corrected",
) -> pd.DataFrame:
    """生成供预测—观测图使用的数据副本。

    Parameters
    ----------
    corrected_data : pandas.DataFrame
        ``correct_observations_to_reference_vs30`` 返回的数据。被修正的原列
        已替换为参考场地值，同时保留同名 ``*_raw`` 原始观测列。
    plot_observations : {"corrected", "raw"}, default "corrected"
        ``"corrected"`` 绘制统一到参考 Vs30 的观测；``"raw"`` 恢复各
        台站原始场地观测。该选项只改变图上的观测点和残差，不改变反演输入、
        最优震中、chi2 或导出的反演统计表。

    Returns
    -------
    pandas.DataFrame
        独立的数据副本，可安全传给 ``*_vs_Obs`` 绘图函数。返回表的
        ``attrs["site_plot_observations"]`` 记录绘图模式，供标题明确标注。

    Raises
    ------
    ValueError
        ``plot_observations`` 不是 ``"corrected"``/``"raw"``，或输入表不是
        ``correct_observations_to_reference_vs30`` 生成的场地修正数据。
    """
    mode = str(plot_observations).strip().lower()
    if mode not in {"corrected", "raw"}:
        raise ValueError(
            "plot_observations 必须为 'corrected' 或 'raw'，"
            f"收到：{plot_observations!r}"
        )

    plot_data = corrected_data.copy()
    plot_data.attrs = dict(corrected_data.attrs)
    raw_columns = [
        col
        for col in plot_data.columns
        if col.endswith("_raw") and col[: -len("_raw")] in plot_data.columns
    ]
    if "site_reference_vs30" not in plot_data.attrs or not raw_columns:
        raise ValueError(
            "plot_observations 需要 correct_observations_to_reference_vs30 "
            "返回的 DataFrame（必须包含场地修正 attrs 和 *_raw 审计列）"
        )
    plot_data.attrs["site_plot_observations"] = mode
    if mode == "corrected":
        return plot_data

    # 每个被修正的观测列都有一个紧邻保存的 ``列名_raw``。这里只恢复原列，
    # 保留审计列，便于调用者在绘图后继续追溯数值来源。
    for raw_col in raw_columns:
        original_col = raw_col[: -len("_raw")]
        plot_data[original_col] = plot_data[raw_col]

    # 删除“已统一至参考场地”的标题触发属性，改由绘图函数显示原始场地说明。
    plot_data.attrs.pop("site_reference_vs30", None)
    plot_data.attrs.pop("site_correction_model", None)
    plot_data.attrs.pop("site_use_basin", None)
    return plot_data
