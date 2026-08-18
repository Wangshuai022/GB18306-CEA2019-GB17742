# -*- coding: utf-8 -*-
"""
残差统计图：半小提琴(KDE) + 中心箱线 + 抖动散点（本项目标准样式）
=================================================================
这是从 Codex stat-violin-plot 技能复制到工作目录的绘图包，方便本项目直接使用。
用于画"残差分布"这类统计图：左边 KDE 半小提琴、中间箱线、右边抖动散点，
并在顶部标注 N（样本数）、μ（均值）、m（中位数）、σ（总体标准差）和
RMS（均方根）。

样式规则（固定，不要绕过）：
    - Agg 后端；axes.unicode_minus = False；DPI 300
    - 西文 Times New Roman，中文 Microsoft YaHei（自动回退，消除中文乱码）
    - 颜色按调用方传入，几何参数见 half_violin_box_scatter 的说明

使用案例
--------
1) 独立成图（多组对比）：
    from stat_violin import stat_violin_figure
    stat_violin_figure(
        groups=[[0.1, 0.3, -0.2, ...], [-0.1, 0.2, ...]],   # 每组一个一维数组
        labels=["全部", "<200km", ">=200km"],
        colors=["#1f77b4", "#2ca02c", "#d62728"],
        out_path="residual.png",
        xlabel="分组", ylabel="ln(Pred/Obs)", title="残差分布",
        value_fmt="{:.2f}",
    )

2) 嵌入现有子图（如 CEA2019_vs_Obs.py 的 4×N 大图）：
    from stat_violin import half_violin_box_scatter, apply_style
    apply_style()                       # 先套用全局样式
    ax = fig.add_subplot(...)
    half_violin_box_scatter(ax, data, x=0, color="#1f77b4", value_fmt="{:.2f}")
    # data: 一维残差数组（自动滤掉 NaN）；x: 该组在横轴上的位置

主要参数（half_violin_box_scatter）：
    left_offset  小提琴中心相对组位置的左移量（默认 0.15）
    right_offset 散点相对组位置的右移量（默认 0.25）
    violin_scale 小提琴半宽（默认 0.2）
    box_width    箱线宽（默认 0.11）
    jitter_scale 散点横向抖动幅度（默认 0.04）
    annotate     是否标注 N/μ/m/σ/RMS（默认 True）
    value_fmt    标注数值格式（残差用 {:.2f} 或 {:.3f}）
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def apply_style():
    """套用本项目统计图统一样式。

    本函数修改 Matplotlib 的全局 ``rcParams``，设置 Times New Roman 西文字体、
    Microsoft YaHei 中文回退、ASCII 负号及 300 dpi 输出。无输入参数和返回值；
    若调用方还需要自己的样式，应在调用本函数之后覆盖相应 ``rcParams``。
    """
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei"],  # 显式列表才能触发中文回退
            "font.serif": ["Times New Roman", "Microsoft YaHei"],
            "axes.unicode_minus": False,
            "mathtext.default": "regular",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def half_violin_box_scatter(
    ax,
    data,
    x,
    color,
    left_offset=0.15,
    right_offset=0.25,
    violin_scale=0.2,
    box_width=0.11,
    jitter_scale=0.04,
    annotate=True,
    annotate_y=None,
    value_fmt="{:.2f}",
    alpha_scatter=0.7,
    s=20,
):
    """在已有坐标轴上绘制单组“半小提琴 + 箱线 + 散点”。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        接收图元的目标坐标轴。
    data : array-like
        一维统计样本；NaN 和无穷值会被过滤，过滤后为空则不绘制。
    x : float
        该组的横轴中心位置。
    color : matplotlib color
        小提琴、箱体和散点的颜色。
    left_offset, right_offset : float
        小提琴中心向左、散点中心向右相对 ``x`` 的偏移量。
    violin_scale, box_width, jitter_scale : float
        半小提琴宽度、箱体宽度和散点横向抖动标准差。
    annotate : bool
        是否标注样本量 N、均值 μ、中位数 m、总体标准差 σ 和均方根 RMS。
    annotate_y : float or None
        注释框的 y 坐标；None 时根据样本最大值自动确定。
    value_fmt : str
        μ、m、σ 和 RMS 的 Python 格式串，例如 ``"{:.2f}"``。
    alpha_scatter : float
        散点透明度。
    s : float
        散点面积，单位为 points²。

    Returns
    -------
    None
        图元直接添加到 ``ax``。散点抖动固定使用随机种子 42，保证重复出图一致。
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:  # 空组直接跳过（如全部台站都 <200 km 时">200"组为空）
        return

    lo, hi = float(data.min()), float(data.max())
    span = hi - lo if hi > lo else 1.0
    pad = span * 0.10
    ymin, ymax = lo - pad, hi + pad
    grid = np.linspace(ymin, ymax, 500)

    # ---- 左：KDE 半小提琴（密度越大越宽）----
    violin_center_x = x - left_offset
    if span > 0 and np.std(data) > 0:
        try:
            kde = gaussian_kde(data)
            density = kde.evaluate(grid)
            density_norm = density / density.max()
        except Exception:
            density_norm = None
    else:
        density_norm = None

    if density_norm is not None:
        left_x = violin_center_x - density_norm * violin_scale
        ax.fill_betweenx(
            grid, left_x, violin_center_x,
            facecolor=color, alpha=1.0,
            edgecolor="k", linewidth=1.5,
        )
    else:
        ax.plot(
            [violin_center_x - violin_scale, violin_center_x],
            [data[0], data[0]], color=color, lw=2,
        )

    # ---- 中：箱线（中位数、四分位、须、离群点）----
    ax.boxplot(
        data, positions=[x], widths=box_width, patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor="k", alpha=0.7, linewidth=1.5),
        whiskerprops=dict(color="k", linewidth=1.5),
        capprops=dict(color="k", linewidth=1.5),
        medianprops=dict(color="black", linewidth=2.5),
        flierprops=dict(marker="o", markerfacecolor=color, markersize=4, alpha=0.5),
    )

    # ---- 右：抖动散点（固定随机种子 42，保证每次图一致）----
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, jitter_scale, size=len(data))
    scatter_x = np.full_like(data, x + right_offset) + jitter
    ax.scatter(
        scatter_x, data, color=color, alpha=alpha_scatter,
        s=s, edgecolors="none",
    )

    # ---- 顶部标注 N / μ / m / σ / RMS ----
    if annotate:
        y_text = annotate_y if annotate_y is not None else hi + pad
        n = len(data)
        mu = float(np.mean(data))
        med = float(np.median(data))
        # 这里采用总体标准差（ddof=0），与当前一组有限残差的描述统计一致；
        # RMS 保留残差相对零线的整体幅度，因此通常不等于标准差。
        sigma = float(np.std(data, ddof=0))
        rms = float(np.sqrt(np.mean(np.square(data))))
        text_str = (
            f"$N$ = {n}\n"
            f"$\\mu$ = {value_fmt.format(mu)}\n"
            f"$m$ = {value_fmt.format(med)}\n"
            f"$\\sigma$ = {value_fmt.format(sigma)}\n"
            f"RMS = {value_fmt.format(rms)}"
        )
        ax.text(
            x, y_text, text_str, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5, edgecolor="black"),
        )


def fit_annotations_inside(ax, fig=None, draw=True, margin_px=3.0):
    """扩大 y 轴范围，使坐标轴内的统计注释框不被裁切。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        需要检查文本边界的坐标轴。
    fig : matplotlib.figure.Figure or None
        所属图；None 时使用 ``ax.figure``。
    draw : bool
        True 时先渲染一次以取得文本像素边界。多子图可统一 ``canvas.draw`` 后
        传 False，避免重复渲染。
    margin_px : float
        注释框与坐标区上下边界之间保留的最小像素距离。

    Returns
    -------
    matplotlib.axes.Axes
        调整后的原坐标轴对象。渲染器不可用时保持原范围并安全返回。
    """
    fig = fig if fig is not None else ax.figure
    try:
        if draw:
            fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ab = ax.get_window_extent(renderer)
        for t in ax.texts:
            bb = t.get_window_extent(renderer)
            y0, y1 = ax.get_ylim()
            span = max(y1 - y0, 1e-9)
            ab_h = max(ab.height, 1.0)
            need_top = max(0.0, bb.y1 - (ab.y1 - margin_px)) / ab_h * span
            need_bot = max(0.0, (ab.y0 + margin_px) - bb.y0) / ab_h * span
            if need_top > 0 or need_bot > 0:
                ax.set_ylim(y0 - need_bot * 1.1, y1 + need_top * 1.1)
    except Exception:
        pass
    return ax


def stat_violin_figure(
    groups,
    labels,
    colors,
    out_path,
    xlabel="",
    ylabel="",
    title="",
    suptitle=None,
    figsize=(7, 4),
    ylim=None,
    annotate=True,
    value_fmt="{:.2f}",
    **kwargs,
):
    """生成并保存多组统计量的标准半小提琴对比图。

    Parameters
    ----------
    groups : sequence of array-like
        每组一个一维样本数组。
    labels, colors : sequence
        各组标签和颜色；应与 ``groups`` 长度一致。
    out_path : str or os.PathLike
        输出 PNG 路径。
    xlabel, ylabel, title : str
        坐标轴标签和子图标题。
    suptitle : str or None
        可选总标题。
    figsize : tuple(float, float)
        Matplotlib 图尺寸，单位 inch。
    ylim : tuple(float, float) or None
        y 轴范围；None 时由所有有限样本自动计算并给注释留白。
    annotate : bool
        是否为各组显示 N、均值和中位数。
    value_fmt : str
        注释数值格式。
    **kwargs
        继续传给 :func:`half_violin_box_scatter` 的几何和散点参数。

    Returns
    -------
    str or os.PathLike
        原样返回 ``out_path``。函数保存后会关闭 Figure，适合批量绘图。
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(labels))

    for i, (data, label, color) in enumerate(zip(groups, labels, colors)):
        half_violin_box_scatter(ax, data, positions[i], color,
                                annotate=annotate, value_fmt=value_fmt, **kwargs)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is None:
        all_vals = np.concatenate(
            [
                np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))]
                for g in groups
                if len(g)
            ]
        )
        if len(all_vals):
            lo, hi = float(all_vals.min()), float(all_vals.max())
            span = hi - lo if hi > lo else 1.0
            pad = span * 0.10
            top_extra = span * 0.35 if annotate else 0.0
            ylim = (lo - pad, hi + pad + top_extra)
    if ylim is not None:
        ax.set_ylim(ylim)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fit_annotations_inside(ax, fig=fig, draw=True)  # 自适应：注释框不出图
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # 自检案例：两组随机残差 → 一张对比图
    import os

    rng = np.random.default_rng(7)
    g1 = rng.normal(0.05, 0.3, 30)
    g2 = rng.normal(-0.1, 0.4, 25)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stat_violin_demo.png")
    stat_violin_figure(
        groups=[g1, g2],
        labels=["全部", "≥200 km"],
        colors=["#1f77b4", "#d62728"],
        out_path=out,
        xlabel="分组",
        ylabel="ln(Pred/Obs)",
        title="残差分布案例",
        value_fmt="{:.3f}",
    )
    print(f"自检完成：{out}")
