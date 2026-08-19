"""预测—观测残差的跨参数组合评估图与配套数据表。

本模块不直接调用任何地震动模型。CEA2019/GB18306 的 ``vs_Obs`` 程序
先计算统一结构的观测、预测、椭圆距和残差，再把结果传给本模块。这样可以
保证给定宏观震中与反演最优宏观震中的评估图使用完全相同的筛选和统计定义。

整张图只有一个坐标轴；每个参数位置按标准统计样式同时绘制左半小提琴、
中央箱线和右侧逐台站抖动散点。小提琴与箱线采用固定纯色，只有散点按照
等效椭圆距离使用 ``Spectral_r`` 填色。
横轴顺序由调用方参数顺序决定，例如 PGA、PGV、PSA(0.10s)...PSA(6.00s)。
TXT 同时包含逐参数统计摘要和绘图所用的逐台站长表，便于用户重新作图。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from stat_violin import apply_style, half_violin_box_scatter

plt.switch_backend("Agg")


DEFAULT_CEA2019_EVALUATION_PARAMS = (
    -1.0,
    -2.0,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    4.00,
    5.00,
    6.00,
)

DEFAULT_GB18306_EVALUATION_PARAMS = (-1.0, -2.0, "Intensity")


def normalize_distance_range(distance_range=None):
    """规范评估图的等效椭圆距离筛选范围。

    Parameters
    ----------
    distance_range : 2-sequence or None
        ``(下限, 上限)``，单位 km。下限包含、上限不包含；任一端可为
        ``None``。例如 ``(None, 200)`` 表示小于 200 km，
        ``(200, None)`` 表示大于等于 200 km。None 表示不按距离筛选。

    Returns
    -------
    tuple(float or None, float or None)
        规范后的距离下限和上限。

    Raises
    ------
    ValueError
        输入不是两个边界、边界为负数，或下限不小于上限。
    """
    if distance_range is None:
        return None, None
    if (
        not isinstance(distance_range, (tuple, list))
        or len(distance_range) != 2
    ):
        raise ValueError("distance_range 必须为 (最小距离, 最大距离) 或 None")
    lower = None if distance_range[0] is None else float(distance_range[0])
    upper = None if distance_range[1] is None else float(distance_range[1])
    if lower is not None and lower < 0:
        raise ValueError("距离下限不能为负数")
    if upper is not None and upper <= 0:
        raise ValueError("距离上限必须为正数")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("距离下限必须小于距离上限")
    return lower, upper


def normalize_station_type(station_type="all"):
    """把中文或英文台站筛选名称规范为 ``all``、``EI`` 或 ``HN``。

    ``EI`` 表示烈度台，``HN`` 表示强震仪；``all`` 同时保留两类及数据中
    其他未知仪器类型。
    """
    if station_type is None:
        return "all"
    key = str(station_type).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "all": "all",
        "both": "all",
        "全部": "all",
        "两者": "all",
        "全部台站": "all",
        "ei": "EI",
        "intensity": "EI",
        "intensity_station": "EI",
        "烈度台": "EI",
        "烈度仪": "EI",
        "hn": "HN",
        "strong_motion": "HN",
        "strong_motion_station": "HN",
        "强震仪": "HN",
        "强震台": "HN",
    }
    if key not in aliases:
        raise ValueError(
            "station_type 只能是 all/EI/HN，或 全部/烈度台/强震仪"
        )
    return aliases[key]


def _parameter_code(param):
    if isinstance(param, str):
        return param
    value = float(param)
    return f"{value:g}"


def _parameter_period(param):
    if isinstance(param, str):
        return np.nan
    value = float(param)
    return value if value > 0 else np.nan


def _station_type_label(station_type):
    return {"all": "全部台站", "EI": "烈度台(EI)", "HN": "强震仪(HN)"}[
        station_type
    ]


def _distance_label(lower, upper):
    if lower is None and upper is None:
        return "全部距离"
    if lower is None:
        return f"< {upper:g} km"
    if upper is None:
        return f">= {lower:g} km"
    return f"{lower:g} <= R < {upper:g} km"


def build_residual_evaluation_tables(
    computation,
    *,
    distance_range=None,
    station_type="all",
    axis="长轴",
):
    """从 ``vs_Obs`` 共享计算结果构建筛选后的长表和统计摘要。

    Parameters
    ----------
    computation : dict
        CEA2019 ``_compute_vs_obs`` 或 GB18306 ``compute_vs_obs`` 的返回值。
        必须包含 ``params/infos/obs/preds/aeqs/ress/source_columns``。
    distance_range : 2-sequence or None
        等效椭圆距离筛选 ``(下限, 上限)``，下限包含、上限不包含。
    station_type : {"all", "EI", "HN"} or Chinese alias
        全部台站、烈度台或强震仪筛选。
    axis : {"长轴", "短轴"}
        距离筛选使用的等效椭圆轴。

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame, dict)
        ``station_table`` 为绘图用逐台站长表；``summary_table`` 含每个参数的
        N、均值、中位数、总体标准差、RMS、最小值、四分位数和最大值；
        metadata 记录筛选条件和观测列来源。
    """
    if axis not in ("长轴", "短轴"):
        raise ValueError("axis 只能是 '长轴' 或 '短轴'")
    lower, upper = normalize_distance_range(distance_range)
    station_type = normalize_station_type(station_type)
    obs = computation["obs"]
    instrument = obs["Instrument_Type"].fillna("").astype(str).str.upper()
    if station_type != "all" and station_type not in set(instrument):
        raise ValueError(f"观测数据中没有 {station_type} 类型台站")

    rows = []
    summary_rows = []
    axis_index = 0 if axis == "长轴" else 1
    source_columns = computation.get("source_columns", {})
    for order, param in enumerate(computation["params"]):
        info = computation["infos"][param]
        label = info["label"]
        distance = np.asarray(
            computation["aeqs"][label][axis_index], dtype=float
        )
        observed = pd.to_numeric(obs[label], errors="coerce").to_numpy(float)
        predicted = np.asarray(computation["preds"][label], dtype=float)
        residual = np.asarray(computation["ress"][label], dtype=float)
        mask = (
            np.isfinite(distance)
            & np.isfinite(observed)
            & np.isfinite(predicted)
            & np.isfinite(residual)
        )
        if station_type != "all":
            mask &= instrument.to_numpy() == station_type
        if lower is not None:
            mask &= distance >= lower
        if upper is not None:
            mask &= distance < upper

        selected = np.flatnonzero(mask)
        for index in selected:
            rows.append(
                {
                    "Parameter_Order": order,
                    "Parameter": label,
                    "Parameter_Code": _parameter_code(param),
                    "Period_s": _parameter_period(param),
                    "Unit": info.get("unit", ""),
                    "Sta_ID": str(obs.iloc[index]["Sta_ID"]),
                    "Instrument_Type": str(instrument.iloc[index]),
                    "lon": float(obs.iloc[index]["lon"]),
                    "lat": float(obs.iloc[index]["lat"]),
                    "Distance_Axis": axis,
                    "Equivalent_Distance_km": float(distance[index]),
                    "Observed": float(observed[index]),
                    "Predicted": float(predicted[index]),
                    "Residual": float(residual[index]),
                    "Observation_Column": source_columns.get(label, ""),
                }
            )

        values = residual[mask]
        n = int(values.size)
        if n:
            q1, q3 = np.quantile(values, [0.25, 0.75])
            stats = {
                "Mean": float(np.mean(values)),
                "Median": float(np.median(values)),
                "Sigma": float(np.std(values, ddof=0)),
                "RMS": float(np.sqrt(np.mean(np.square(values)))),
                "Minimum": float(np.min(values)),
                "Q1": float(q1),
                "Q3": float(q3),
                "Maximum": float(np.max(values)),
            }
        else:
            stats = {
                key: np.nan
                for key in (
                    "Mean",
                    "Median",
                    "Sigma",
                    "RMS",
                    "Minimum",
                    "Q1",
                    "Q3",
                    "Maximum",
                )
            }
        summary_rows.append(
            {
                "Parameter_Order": order,
                "Parameter": label,
                "Parameter_Code": _parameter_code(param),
                "Period_s": _parameter_period(param),
                "N": n,
                **stats,
            }
        )

    station_table = pd.DataFrame(rows)
    summary_table = pd.DataFrame(summary_rows)
    metadata = {
        "axis": axis,
        "distance_min_km": lower,
        "distance_max_km_exclusive": upper,
        "distance_label": _distance_label(lower, upper),
        "station_type": station_type,
        "station_type_label": _station_type_label(station_type),
        "source_columns": source_columns,
    }
    return station_table, summary_table, metadata


def export_residual_evaluation_txt(
    station_table,
    summary_table,
    metadata,
    outpath,
):
    """导出评估图配套 TXT（元数据、统计摘要、逐台站长表）。"""
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        stream.write("# Residual evaluation data\n")
        for key, value in metadata.items():
            text = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, dict)
                else value
            )
            stream.write(f"# {key}: {text}\n")
        stream.write("\n[SUMMARY]\n")
        summary_table.to_csv(
            stream, sep="\t", index=False, na_rep="NaN", float_format="%.6g"
        )
        stream.write("\n[STATION_DATA]\n")
        station_table.to_csv(
            stream, sep="\t", index=False, na_rep="NaN", float_format="%.8g"
        )
    print(f"已保存残差评估数据：{path.resolve()}")
    return str(path)


def _common_y_limits(groups, symmetric_step=None):
    values = [np.asarray(group, dtype=float) for group in groups if len(group)]
    if not values:
        return -1.0, 1.0
    merged = np.concatenate(values)
    merged = merged[np.isfinite(merged)]
    if not len(merged):
        return -1.0, 1.0
    if symmetric_step is not None:
        step = float(symmetric_step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("symmetric_y_step 必须为正数或 None")
        max_abs = float(np.max(np.abs(merged)))
        limit = max(step, np.ceil(max_abs / step) * step)
        return -limit, limit
    lo, hi = float(np.min(merged)), float(np.max(merged))
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), 1.0)
    return lo - 0.08 * span, hi + 0.12 * span


def plot_residual_evaluation_combined(
    computation,
    *,
    outpath,
    model_name,
    residual_label="ln(Predicted / Observed)",
    distance_range=None,
    station_type="all",
    axis="长轴",
    table_outpath=None,
    figsize_cm=None,
    title=None,
    observation_state=None,
    symmetric_y_step=None,
):
    """在单一坐标轴绘制半小提琴—箱线—距离着色散点组合图并输出 TXT。

    Parameters
    ----------
    computation : dict
        两个 ``vs_Obs`` 模型之一的共享计算结果。
    outpath : str or os.PathLike
        PNG 输出路径。
    model_name : str
        图题和 TXT 中记录的模型名称。
    residual_label : str
        单一组合图的纵轴名称。
    distance_range : 2-sequence or None
        ``(下限, 上限)`` km；下限包含、上限不包含。
    station_type : str
        ``all``、``EI``、``HN`` 或对应中文别名。
    axis : {"长轴", "短轴"}
        距离筛选使用的等效椭圆轴。
    table_outpath : str or os.PathLike or None
        配套 TXT；None 时与 PNG 同名，仅替换为 ``.txt``。
    figsize_cm : 2-sequence or None
        图宽和图高，单位 cm。None 时按参数数量自动扩展宽度。
    title : str or None
        可选自定义总标题。
    observation_state : str or None
        可选观测状态说明，例如 ``CB14修正到Vs30=500 m/s``。
    symmetric_y_step : float or None
        对称纵轴边界的取整步长。0.5 表示残差绝对最大值向上取 0.5 的
        整数倍，例如 2.1 得到 ``[-2.5, 2.5]``；None 使用普通自适应范围。

    Returns
    -------
    dict
        包含 ``plot_path/table_path/station_table/summary_table/metadata``。
    """
    if figsize_cm is not None:
        if not isinstance(figsize_cm, (tuple, list)) or len(figsize_cm) != 2:
            raise ValueError("figsize_cm 必须为 (宽cm, 高cm) 或 None")
        if float(figsize_cm[0]) <= 0 or float(figsize_cm[1]) <= 0:
            raise ValueError("figsize_cm 的宽和高必须为正数")

    station_table, summary_table, metadata = build_residual_evaluation_tables(
        computation,
        distance_range=distance_range,
        station_type=station_type,
        axis=axis,
    )
    metadata = {
        "model": model_name,
        "residual_definition": residual_label,
        "observation_state": observation_state or "input observations",
        "scatter_color_by": "Equivalent_Distance_km",
        "scatter_colormap": "Spectral_r",
        "scatter_markers": {
            "HN": "circle (o), no outline",
            "EI": "triangle_up (^), no outline",
            "OTHER": "square (s), no outline",
        },
        "plot_layout": "single_axes_half_violin_box_scatter",
        "violin_fill": "#8FB9E1 solid",
        "box_fill": "#8FB9E1 solid",
        "parameter_annotations": "Median, Mean, Sigma, RMS",
        "sample_count_location": "figure subtitle",
        "symmetric_y_step": symmetric_y_step,
        **metadata,
    }
    labels = summary_table["Parameter"].astype(str).tolist()
    if figsize_cm is None:
        figsize_cm = (max(15.0, 1.5 * len(labels)), 12.0)
    metadata["figure_width_cm"] = float(figsize_cm[0])
    metadata["figure_height_cm"] = float(figsize_cm[1])
    groups = [
        (
            station_table.loc[
                station_table.get("Parameter_Order", pd.Series(dtype=int))
                == index,
                "Residual",
            ].to_numpy(float)
            if not station_table.empty
            else np.array([], dtype=float)
        )
        for index in range(len(labels))
    ]
    positions = np.arange(len(labels), dtype=float)
    distribution_color = "#8FB9E1"
    if not station_table.empty:
        all_distances = station_table["Equivalent_Distance_km"].to_numpy(float)
        distance_min = float(np.nanmin(all_distances))
        distance_max = float(np.nanmax(all_distances))
        if distance_max <= distance_min:
            distance_max = distance_min + 1.0
        distance_norm = Normalize(vmin=distance_min, vmax=distance_max)
    else:
        distance_norm = Normalize(vmin=0.0, vmax=1.0)
    distance_cmap = plt.get_cmap("Spectral_r")

    apply_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    figsize = (float(figsize_cm[0]) / 2.54, float(figsize_cm[1]) / 2.54)
    fig, ax = plt.subplots(figsize=figsize)
    y_limits = _common_y_limits(groups, symmetric_step=symmetric_y_step)
    metadata["y_limits"] = [float(y_limits[0]), float(y_limits[1])]

    for index, (position, values) in enumerate(zip(positions, groups)):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            # 使用 stat-violin-plot 的标准几何：左半小提琴、中央箱线。
            # 自带单色散点关闭，下面在右侧按距离和仪器类型单独绘制。
            half_violin_box_scatter(
                ax,
                values,
                x=position,
                color=distribution_color,
                left_offset=0.15,
                right_offset=0.25,
                violin_scale=0.20,
                box_width=0.11,
                jitter_scale=0.04,
                annotate=False,
                draw_scatter=False,
                showfliers=False,
            )
        else:
            ax.text(
                position,
                0.5,
                "N=0",
                transform=blended_transform_factory(
                    ax.transData, ax.transAxes
                ),
                ha="center",
                va="center",
                fontsize=7,
                color="0.4",
            )

        group_rows = (
            station_table.loc[station_table["Parameter_Order"] == index]
            if not station_table.empty
            else station_table
        )
        if len(group_rows):
            rng = np.random.default_rng(20260819 + index)
            jitter = rng.normal(0.0, 0.04, len(group_rows))
            types = (
                group_rows["Instrument_Type"]
                .astype(str)
                .str.upper()
                .to_numpy()
            )
            residuals = group_rows["Residual"].to_numpy(float)
            distances = group_rows["Equivalent_Distance_km"].to_numpy(float)
            for instrument_type, marker, _legend_label in (
                ("HN", "o", "强震仪 HN"),
                ("EI", "^", "烈度台 EI"),
                ("OTHER", "s", "其他/未知"),
            ):
                mask = (
                    ~np.isin(types, ["HN", "EI"])
                    if instrument_type == "OTHER"
                    else types == instrument_type
                )
                if not mask.any():
                    continue
                ax.scatter(
                    np.full(mask.sum(), position + 0.25) + jitter[mask],
                    residuals[mask],
                    s=12,
                    marker=marker,
                    c=distances[mask],
                    cmap=distance_cmap,
                    norm=distance_norm,
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=0.82,
                    zorder=3,
                )

        row = summary_table.iloc[index]
        annotation_fontsize = 5.2 if len(labels) > 10 else 6.5
        stats_text = (
            f"$m$={row['Median']:.2f}\n"
            f"$\\mu$={row['Mean']:.2f}\n"
            f"$\\sigma$={row['Sigma']:.2f}\n"
            f"RMS={row['RMS']:.2f}"
        )
        ax.text(
            position,
            0.985,
            stats_text,
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="center",
            va="top",
            fontsize=annotation_fontsize,
            linespacing=0.92,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )

    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", zorder=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(y_limits)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Ground-motion parameter")
    ax.set_ylabel(residual_label)
    if not station_table.empty:
        present_types = set(
            station_table["Instrument_Type"].fillna("").astype(str).str.upper()
        )
        marker_specs = []
        if "HN" in present_types:
            marker_specs.append(("o", "强震仪 HN"))
        if "EI" in present_types:
            marker_specs.append(("^", "烈度台 EI"))
        if present_types - {"HN", "EI"}:
            marker_specs.append(("s", "其他/未知"))
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=marker,
                markersize=5,
                markerfacecolor="0.4",
                markeredgecolor="none",
                label=label,
            )
            for marker, label in marker_specs
        ]
        if handles:
            ax.legend(
                handles=handles,
                # 顶部固定用于逐参数 m/μ/σ/RMS，图例不得使用自动位置遮挡统计值。
                loc="lower left",
                frameon=True,
                ncol=min(3, len(handles)),
                fontsize=6.5,
            )
        scalar_mappable = plt.cm.ScalarMappable(
            norm=distance_norm, cmap=distance_cmap
        )
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable, ax=ax, pad=0.015, fraction=0.035
        )
        colorbar.set_label(f"{axis} equivalent distance (km)", fontsize=7.5)

    counts = summary_table["N"].astype(int).tolist()
    if counts and min(counts) == max(counts):
        count_text = f"各参数 N={counts[0]}"
    elif counts:
        count_text = f"各参数 N={min(counts)}–{max(counts)}"
    else:
        count_text = "各参数 N=0"
    filter_text = (
        f"{metadata['station_type_label']}; {axis}等效距 "
        f"{metadata['distance_label']}; {count_text}"
    )
    state_text = observation_state or "输入观测"
    main_title = (
        title or f"{model_name} prediction-observation residual evaluation"
    )
    fig.suptitle(
        f"{main_title}\n{state_text}; {filter_text}",
        fontsize=10,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.970))

    plot_path = Path(outpath)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存残差评估图：{plot_path.resolve()}")

    if table_outpath is None:
        table_outpath = plot_path.with_suffix(".txt")
    table_path = export_residual_evaluation_txt(
        station_table, summary_table, metadata, table_outpath
    )
    return {
        "plot_path": str(plot_path),
        "table_path": str(table_path),
        "station_table": station_table,
        "summary_table": summary_table,
        "metadata": metadata,
    }


__all__ = [
    "DEFAULT_CEA2019_EVALUATION_PARAMS",
    "DEFAULT_GB18306_EVALUATION_PARAMS",
    "build_residual_evaluation_tables",
    "export_residual_evaluation_txt",
    "normalize_distance_range",
    "normalize_station_type",
    "plot_residual_evaluation_combined",
]
