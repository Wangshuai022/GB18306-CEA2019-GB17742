
# -*- coding: utf-8 -*-
"""
L2014 断层尺寸定标率 —— 修正版（独立可运行，无外部依赖）

============================================================
一、修正原则（基于 SMD 104 个地壳地震有限断层模型重标定）
  1. 保留 Leonard (2014, BSSA 104(6)) 的"自相似"斜率 b：
     - 由 W = C1*L^(2/3) 与 D = C2*sqrt(A) 推导，斜率只依赖 beta=2/3，
       本次全部不动（L 段 5/3、1.0，W 段 2.5 等）。
  2. 只重拟合截距 a（log10 X = (Mw - a)/b），按机制细分 SS/RS/NS：
     - SS: L 两段连续、W 饱和值 19 -> 24.4 km（SMD 最小二乘拟合）
     - RS / NS: L 方形段(<=5.4 km)与自相似段(>5.4 km)在拐点处强制连续
  3. 预测曲线只含拐点、无阶跃（拐点处 a2 = a1 + (b1-b2)*log10(Lc)）。
  4. A 取拟合后 L*W；D 保持论文原公式；板内（SCR）分支沿用论文原系数。

二、修正后残差统计（SMD 104 个事件，ln(obs/pred)）
      SS L: median -0.057, sigma 0.371
      SS W: median +0.048, sigma 0.373
      RS L: median -0.048, sigma 0.424
      RS W: median -0.064, sigma 0.443
      NS L: median -0.014, sigma 0.321
      NS W: median -0.043, sigma 0.333

用法：
    python L14_fitted_standalone.py                # 运行自检示例
    from L14_fitted_standalone import l14_fitted   # 作为模块导入
    r = l14_fitted(6.8, "SSF", "板间")             # -> dict(A,L,W,D)
    L, W, lo, hi = l14_fitted_range(6.8, "SSF", "板间")
"""

import math


# ============================================================
# 修正版截距 a（板间/板周，SMD 标定）
# ============================================================
FITTED_A = {
    "SS_L1": 4.1697,   # SS 长度段1: b=5/3, 3.4 < L <= 45 km
    "SS_L2": 5.2723,   # SS 长度段2: b=1.0,  L > 45 km（宽度饱和段，连续）
    "SS_W": 3.7369,    # SS 宽度:   b=2.5, 饱和值 24.4 km
    "RS_L1": 3.9954,   # RS 长度段1: b=2.0,  L <= 5.4 km（方形段）
    "RS_L2": 4.2393,   # RS 长度段2: b=5/3,  L > 5.4 km（自相似段）
    "RS_W": 3.6335,    # RS 宽度:   b=2.5
    "NS_L1": 3.8025,   # NS 长度段1: b=2.0,  L <= 5.4 km（方形段）
    "NS_L2": 4.0464,   # NS 长度段2: b=5/3,  L > 5.4 km（自相似段）
    "NS_W": 3.4658,    # NS 宽度:   b=2.5
}
SS_W_CAP = 24.4        # 走滑宽度饱和值（km），SMD 最小二乘拟合

# 修正版残差标准差（ln），用于 ±1σ 范围
FITTED_SIGMA = {
    ("SS", "L"): 0.371, ("SS", "W"): 0.373,
    ("RS", "L"): 0.424, ("RS", "W"): 0.443,
    ("NS", "L"): 0.321, ("NS", "W"): 0.333,
    ("All", "L"): 0.383, ("All", "W"): 0.389,
}


# ============================================================
# 论文原版系数（板内 SCR 分支沿用；板间分支用于对照，不参与修正）
# 结构: (b, a, lo, hi) 即 log10 X = (Mw - a)/b，范围针对 X（km 或 m）
# ============================================================
PAPER = {
    "Interplate DS": {
        "L": [(2.0, 4.00, 0.0, 5.4), (5.0 / 3.0, 4.24, 5.4, None)],
        "W": [(2.5, 3.63, 5.4, None)],
        "D": [(2.0, 6.84, 0.0, None)],
    },
    "Interplate SS": {
        "L": [(5.0 / 3.0, 4.17, 3.4, 45.0), (1.0, 5.27, 45.0, None)],
        "W": [(2.5, 3.88, 3.4, 19.0)],
        "D": [(2.0, 6.85, 0.13, None)],
    },
    "SCR DS": {
        "L": [(5.0 / 3.0, 4.32, 2.5, None)],
        "W": [(2.5, 4.14, 2.5, None)],
        "D": [(2.0, 6.46, 0.18, None)],
    },
    "SCR SS": {
        "L": [(5.0 / 3.0, 4.25, 1.6, 70.0), (1.0, 5.44, 70.0, None)],
        "W": [(2.5, 4.22, 1.6, 20.0)],
        "D": [(2.0, 3.71, 0.0, None)],
    },
}


def _pick_rule(rules, value):
    """选择第一条 value 落在 [lo, hi] 范围内的公式；无匹配则就近取端。"""
    for (b, a, lo, hi) in rules:
        ok = True
        if lo is not None and value < lo:
            ok = False
        if hi is not None and value > hi:
            ok = False
        if ok:
            return b, a
    return rules[0][:2] if value < (rules[0][2] or -math.inf) else rules[-1][:2]


def _family(fault_type):
    f = str(fault_type)
    if f in ("SSF", "SS", "走滑"):
        return "SS"
    if f in ("RF", "RS", "RV", "NF", "NS", "逆", "逆冲", "正", "正断"):
        return "DS"
    raise ValueError("unknown fault_type: %r (use SSF/RF/NF 或中文)" % (fault_type,))


def _paper_scaling(Mw, family, eq_type):
    """论文原版 L2014（Table 4），返回 A/L/W/D（km 或 m）。"""
    if eq_type in ("板内",):
        key = "SCR " + family
    else:
        key = "Interplate " + family
    rules = PAPER[key]
    out = {}
    for dim, rs in rules.items():
        if len(rs) == 1:
            b, a, lo, hi = rs[0]
            v = 10 ** ((Mw - a) / b)
            if lo is not None:
                v = max(v, lo)
            if hi is not None:
                v = min(v, hi)
            out[dim] = v
        else:
            b0, a0 = rs[0][0], rs[0][1]
            b, a = _pick_rule(rs, 10 ** ((Mw - a0) / b0))
            out[dim] = 10 ** ((Mw - a) / b)
    out["A"] = out["L"] * out["W"]
    return out


def l14_fitted(Mw, fault_type, eq_type):
    """由 Mw 和断层类型计算修正 Leonard (2014) 中值尺寸。

    Parameters
    ----------
    Mw : float
        矩震级。
    fault_type : str
        走滑 ``SSF/SS/走滑``、逆冲 ``RF/RS/RV/逆/逆冲`` 或正断层
        ``NF/NS/正/正断``。
    eq_type : str
        ``板间/板周`` 使用 SMD 地壳事件拟合截距；``板内`` 使用 Leonard
        (2014) 稳定大陆区原始系数。

    Returns
    -------
    dict
        ``A`` 破裂面积（km²）、``L`` 长度（km）、``W`` 宽度（km）、
        ``D`` 平均位错（m）和系数来源 ``note``。

    Notes
    -----
    走滑宽度按当前项目约定限制为 ``SS_W_CAP``；返回值是中值，不包含随机
    扰动。需要 ±1σ 范围时调用 ``l14_fitted_range``。
    """
    if str(eq_type) in ("板内",):
        fam = _family(fault_type)
        return _paper_scaling(Mw, fam, "板内")

    fam = _family(fault_type)
    if fam == "SS":
        L1 = 10 ** ((Mw - FITTED_A["SS_L1"]) / (5.0 / 3.0))
        L = L1 if L1 <= 45.0 else 10 ** ((Mw - FITTED_A["SS_L2"]) / 1.0)
        W = min(10 ** ((Mw - FITTED_A["SS_W"]) / 2.5), SS_W_CAP)
    else:
        key = "RS" if str(fault_type) in ("RF", "RS", "RV", "逆", "逆冲") \
            else "NS"
        L_small = 10 ** ((Mw - FITTED_A[key + "_L1"]) / 2.0)
        L = L_small if L_small < 5.4 else \
            10 ** ((Mw - FITTED_A[key + "_L2"]) / (5.0 / 3.0))
        W = 10 ** ((Mw - FITTED_A[key + "_W"]) / 2.5)

    A = L * W
    D = 10 ** ((Mw - 6.84) / 2.0)          # 论文 Interplate DS 平均位错
    return {"A": A, "L": L, "W": W, "D": D,
            "note": "L2014 fitted (SMD 104 crustal events; slopes per "
                    "Leonard 2014)"}


def l14_fitted_range(Mw, fault_type, eq_type, dims=("L", "W"),
                     mechanism=None):
    """返回修正 Leonard (2014) 尺寸中值及对数 ±1σ 范围。

    Parameters
    ----------
    Mw, fault_type, eq_type
        与 ``l14_fitted`` 相同。
    dims : sequence of str, default ("L", "W")
        需要输出的尺寸键，可从 ``A/L/W/D`` 中选择有统计量的项目。
    mechanism : {"SS", "RS", "NS"} or None
        残差标准差类别；None 时由 ``fault_type`` 推断。

    Returns
    -------
    dict
        ``{dim: (median, lower, upper, sigma_ln)}``，其中上下界为
        ``median * exp(±sigma_ln)``。
    """
    r = l14_fitted(Mw, fault_type, eq_type)
    mech = mechanism or {"SSF": "SS", "SS": "SS", "走滑": "SS",
                         "RF": "RS", "RS": "RS", "RV": "RS", "逆": "RS",
                         "逆冲": "RS", "NF": "NS", "NS": "NS", "正": "NS",
                         "正断": "NS"}[str(fault_type)]
    out = {}
    for dim in dims:
        sig = FITTED_SIGMA.get((mech, dim),
                               FITTED_SIGMA[("All", dim)])
        med = r[dim]
        out[dim] = (med, med * math.exp(-sig), med * math.exp(sig), sig)
    return out

def me():
    pass

if __name__ == "__main__":
    print("=" * 78)
    print("修正版 L2014（SMD 104 个地壳模型标定）自检")
    print("=" * 78)

    print("\n--- 板间（修正截距）预测表 ---")
    hdr = "Mw    SSF: L/W/A          RF: L/W/A           NF: L/W/A"
    print(hdr)
    for mw in (5.5, 6.0, 6.5, 6.7, 6.8, 7.0, 7.5, 8.0):
        row = ["Mw=%.1f" % mw]
        for ft in ("SSF", "RF", "NF"):
            r = l14_fitted(mw, ft, "板间")
            row.append("%5.1f/%4.1f/%6.0f" % (r["L"], r["W"], r["A"]))
        print("  ".join(row))

    print("\n--- 示例：Mw 6.8 走滑 ±1σ ---")
    rr = l14_fitted_range(6.8, "SS", "板间")
    for dim in ("L", "W"):
        med, lo, hi, sig = rr[dim]
        print("  %s: median=%6.2f km  1σ=[%6.2f, %6.2f]  (σ_ln=%.3f)"
              % (dim, med, lo, hi, sig))

    print("\n--- 板内（SCR，论文原系数）对照 ---")
    for ft in ("SSF", "RF"):
        r = l14_fitted(7.0, ft, "板内")
        print("  Mw7.0 %-3s: L=%6.1f W=%5.1f A=%7.0f" % (ft, r["L"], r["W"],
                                                          r["A"]))

    print("\n自检通过：文件可独立运行，无外部依赖。")
    r = l14_fitted(6.8, "NF", "板内")
