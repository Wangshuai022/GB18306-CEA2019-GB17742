# GB18306 / CEA2019 椭圆衰减模型应用包

本包用于把 GB18306-2015 和 CEA2019 长短轴椭圆衰减关系应用到单次地震。
核心问题是模型需要宏观震中，而有限断层模型通常给出的是初始破裂点；二者可先
近似为同一点，也可以利用观测记录反演最优宏观震中。

当前发布版本：**v1.0**（2026-08-18）。

## V1.0 本轮实际修改清单

下面用 `[功能]` 和 `[文档]` 区分运行逻辑修改与说明补充。没有改动算法的文件
不会再描述成“新增功能”。

| 程序 | V1.0 修改内容 |
|---|---|
| `GB18306_vs_Obs.py` | `[功能]` PGA/PGV 改为 RotD50 四级优先；公共绘图函数新增 `plot_observations`，可对场地修正 DataFrame 显式选择 `corrected/raw`，并在标题标明状态。`[文档]` 补充输入、输出、单位和残差定义。 |
| `CEA2019_vs_Obs.py` | `[功能]` PGA、PGV、PSA 改为 RotD50 优先；公共绘图函数新增 `plot_observations` 并显示场地观测状态。`[文档]` 补充多参数接口说明。 |
| `GB18306_epicenter_inversion.py` | `[功能]` 反演观测列改为 RotD50 四级优先。sigma 加权 chi2、断层约束和返回结构是原有实现，本轮只补充其说明。 |
| `CEA2019_epicenter_inversion.py` | `[功能]` 通过 `CEA2019_vs_Obs.load_obs_data` 继承新的 RotD50 优先级。多参数 chi2 反演算法未改，本轮补充返回结果和有效样本说明。 |
| `Vs30_site_correction.py` | 新增中国 Vs30 大文件分块查询、CB14 非线性 A1100 反解、PGA/PGV/PSA 到参考 Vs30 的统一换算和逐台站审计表。 |
| `GB18306_epicenter_inversion_Vs30.py` | 新增 GB18306 场地修正版反演；反演固定使用 Vs30=500 m/s 观测，绘图可在 `corrected/raw` 间切换。 |
| `CEA2019_epicenter_inversion_Vs30.py` | 新增 CEA2019 多参数场地修正版反演；所有周期共用同一台站 A1100，绘图开关不改变震中和 chi2。 |
| `CB14_site_correct.py`、`pynga/` | 纳入成熟 CB14 场地项及其内部实现依赖，使仓库克隆后可直接执行场地修正；应用程序不直接调用 `pynga`。 |
| `ellipse_fields.py` | `[文档]` 补充既有共享椭圆场的构造参数、数组形状、单位和返回顺序；本轮没有改模型算法。 |
| `GB17742_class.py` | `[文档]` 补充既有仪器烈度换算接口的输入单位、返回值和异常说明；换算公式未改。 |
| `Leonard2014_fitted_by_SMD_crust.py` | `[文档]` 明确既有定标率函数的类型、L/W/A/D 单位及不确定性返回格式；系数未改。 |
| `mesh_single_rectangular_finite_fault.py` | `[文档]` 说明既有 `dhypo=0.57W` 默认值和地表出露整体下移策略；本轮没有改变网格算法。 |
| `fault_distance_azimuth_single_multi.py` | `[文档]` 补充既有 Rrup/Rjb/Rx/方位角入口的坐标单位、符号约定和返回结构；距离算法未改。 |
| `stat_violin.py` | `[文档]` 补充既有半小提琴、箱线、散点和注释自适应接口；绘图算法未改。 |
| `GB18306_Intensity_Compare.py` | `[文档]` 补充既有烈度对比程序的运行方式和输出文件说明；计算未改。 |
| `tests/test_models.py`、`tests/test_vs30_inversion.py`、`tests/test_release_contracts.py` | 增加 RotD50 四级回退、模型正反算、断层网格、CB14/A1100、Vs30 换算、公共函数签名及 corrected/raw 调用链回归测试。 |

本表只列本次有实质修改或新增的程序。原有 `GB18306_class.py`、
`CEA2019_class.py`、`GB18306_Pre.py` 和 `CEA2019_pre.py` 保持模型实现不变，
继续作为两套经验模型的基础类和初始破裂点预测入口。

## 三条工作流

1. **初始破裂点直接预测**：把初始破裂点当作宏观震中，预测台站地震动参数。
2. **初始破裂点预测与观测对比**：直接计算预测值、观测值和残差，检查模型可靠度。
3. **反演宏观震中后对比**：用带模型 sigma 权重的 chi2 在断层投影范围内反演
   最优宏观震中，再以该位置预测并与观测对比。

GB18306 支持宏观烈度、PGA、PGV；CEA2019 支持 PGA、PGV、0~6 s PSA。
CEA2019 图中如选择 `Intensity`，它是由预测 PGA/PGV 按 GB/T 17742-2020
换算的仪器烈度，不是 CEA2019 自身的衰减参数。

## 程序结构

| 文件 | 作用 |
|---|---|
| `GB18306_class.py` | GB18306 烈度、PGA、PGV 正算和距离反算 |
| `CEA2019_class.py` | CEA2019 Excel 系数读取、PGA/PGV/PSA 正反算 |
| `GB17742_class.py` | PGA/PGV 到仪器烈度的换算 |
| `ellipse_fields.py` | 两模型共用的解析椭圆前向计算内核 |
| `Leonard2014_fitted_by_SMD_crust.py` | SMD 地壳事件修正的 L2014 断层长宽定标率 |
| `mesh_single_rectangular_finite_fault.py` | 根据震源位置、走向、倾角和 L/W 生成矩形断层网格 |
| `fault_distance_azimuth_single_multi.py` | 单/多断层 Rrup、Rjb、Rx 和方位角计算 |
| `GB18306_Pre.py` | GB18306 基于初始破裂点预测、绘图和表格 |
| `CEA2019_pre.py` | CEA2019 基于初始破裂点预测、绘图和表格 |
| `GB18306_vs_Obs.py` | GB18306 指定宏观震中的预测—观测残差分析 |
| `CEA2019_vs_Obs.py` | CEA2019 指定宏观震中的预测—观测残差分析 |
| `GB18306_epicenter_inversion.py` | GB18306 最优宏观震中反演，以及指定宏观震中的预测—观测残差分析 |
| `CEA2019_epicenter_inversion.py` | CEA2019 多参数联合宏观震中反演，以及指定宏观震中的预测—观测残差分析 |
| `GB18306_epicenter_inversion_Vs30.py` | 观测经 CB14 统一到 Vs30=500 m/s 后的 GB18306 宏观震中反演与残差分析 |
| `CEA2019_epicenter_inversion_Vs30.py` | 观测经 CB14 统一到 Vs30=500 m/s 后的 CEA2019 多参数联合宏观震中反演与残差分析 |
| `Vs30_site_correction.py` | 中国 Vs30 分块查询、CB14 非线性反解及参考场地换算共用模块 |
| `CB14_site_correct.py` | 用户成熟的 CB14 场地响应倍率接口；应用层统一通过本文件调用 |
| `stat_violin.py` | 预测—观测残差分布共用的半小提琴、箱线和散点绘图模块 |
| `tests/` | 模型、反演、断层工具、观测优先级和 Vs30 修正回归测试 |

Pre、vs_Obs 和 epicenter_inversion 均通过 `ellipse_fields.py` 使用同一解析场，
避免不同应用之间出现近场采样或插值差异。

## 观测列选择

PGA/PGV 使用以下优先级，只接受规范的 `RotD50` 列名：

```text
PGA_RotD50 / PGV_RotD50
→ PGA_H / PGV_H
→ EPA_RotD50 / EPV_RotD50
→ EPA_H / EPV_H
```

CEA2019 的 PSA 优先使用 `pSa(T=xx.xx s)_RotD50`，再回退
`pSa(T=xx.xx s)_H`。
联合反演只使用所有选定参数均为正有限值的台站；单参数反演不会因为其他参数缺测
而删除台站。

残差统一定义为：

```text
PGA/PGV/PSA: ln(预测值 / 观测值)
烈度:        预测值 - 观测值
```

反演目标使用模型原始 `log10 sigma` 归一化的 chi2。返回结果同时包含
`chi2`、`reduced_chi2`、自由度、有效台站数、实际观测列、优化器状态和边界标记。

## 基本调用

### 1. GB18306 初始破裂点预测

```python
from GB18306_Pre import export_all_table

table = export_all_table(
    lon=87.45, lat=28.50, strike=187, region="青藏区", Ms=6.8,
    sta_lon=[87.6, 88.0], sta_lat=[28.7, 29.0],
    output_file="GB18306_prediction.txt",
)
```

### 2. 使用初始破裂点直接做预测—观测对比

```python
from CEA2019_vs_Obs import plot_cea2019_vs_obs

plot_cea2019_vs_obs(
    data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
    macro_epicenter=(87.45, 28.50),
    Ms=6.8, region="青藏区", strike=187,
    params=[-1, -2, 0.3, 1, 3, 6],
    outpath="CEA2019_initial_vs_obs.png",
)
```

### 3. 反演宏观震中并自动绘制残差图

```python
from CEA2019_epicenter_inversion import invert_epicenter_cea2019

result = invert_epicenter_cea2019(
    data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
    Ms=6.8, Mw=7.0, region="青藏区",
    hypo=(87.45, 28.50, 10.0),
    strike=187, dip=49, rake=-78,
    invert_GMIMs=(-1, -2),
    plot_GMIMs=(-1, -2, 0.3, 1, 3, 6),
    outpath="CEA2019_inversion_stats.txt",
    plot_path="CEA2019_inversion_vs_obs.png",
)
print(result["epicenter"], result["reduced_chi2"])
```

若不提供外部 `fault_lon_mat/fault_lat_mat`，反演会直接调用同目录内的 L14 修正
定标率和矩形有限断层网格模块。默认 `shypo=0`、`dhypo=0.57W`；若断层顶部
高于地表，当前网格模块会整体下移断层并同步调整返回的震源深度。

### 4. 将实际台站观测统一到 Vs30=500 m/s 后反演

两个带 `_Vs30` 后缀的程序默认读取：
`D:\Ubuntu_share\0.Simulate_plat\Database\8.Vs30_data\China_area_Vs30.csv`。
大型 CSV 按台站包围盒分块筛选，不会一次性载入内存。场地修正只调用项目内
成熟的 `CB14_site_correct.py`，不直接调用底层 `pynga`。程序先通过该接口由
实际场地观测 PGA 反解 A1100（Vs30=1100 m/s 参考岩石 PGA），再用同一个
A1100 计算实际场地相对 Vs30=500 m/s 的 PGA、PGV 和 PSA 修正倍率：

```text
A1100 = PGA_observed / F_PGA(Vs30_actual / 1100, A1100)
F_T = CB14_site_correct(Vs30_actual / 500, A1100, period=T)
Y_Vs30=500(T) = Y_observed(T) / F_T
```

其中 `T=-1` 直接读取 PGA 倍率，`T=-2` 直接读取 PGV 倍率，正周期读取对应
PSA 倍率；不存在 PGA 与 PGV 之间的周期插值。

默认不启用盆地项，因为当前输入只有 Vs30、没有台站实测 Z2.5。**震中反演、
chi2 目标函数和导出的反演表始终使用修正到 500 m/s 的观测值**。绘图由
`plot_observations` 单独控制：

- `"corrected"`（默认）：图中的实测点、残差和残差分布均使用 Vs30=500 m/s
  的观测值，与经验模型及反演目标完全一致；
- `"raw"`：只把图中的实测点和绘图残差恢复为台站原始场地观测，用于直观看
  场地修正的影响；不会重新反演，也不会改变最优宏观震中、chi2 或导出表。

两种模式都会在图题中明确标注。原始值、修正系数、反解的 A1100、迭代次数和
修正值同时写入导出统计表，便于逐台站复核。

```python
from CEA2019_epicenter_inversion_Vs30 import invert_epicenter_cea2019_vs30

result = invert_epicenter_cea2019_vs30(
    data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
    Ms=6.8, Mw=7.0, region="青藏区",
    hypo=(87.45, 28.50, 10.0),
    strike=187, dip=49, rake=-78,
    invert_GMIMs=(-1, -2),
    plot_GMIMs=(-1, -2, 0.3, 1, 3, 6),
    plot_observations="corrected",  # 默认；改成 "raw" 仅切换图中观测值
    outpath="CEA2019_inversion_Vs30_500.txt",
    plot_path="CEA2019_inversion_Vs30_500.png",
)
```

GB18306 的用法相同：

```python
from GB18306_epicenter_inversion_Vs30 import invert_epicenter_gb18306_vs30

result = invert_epicenter_gb18306_vs30(
    data="20250107_China_Dingri_total_info_Bandpass_0.05_20Hz.txt",
    Ms=6.8, Mw=7.0, region="青藏区",
    hypo=(87.45, 28.50, 10.0),
    strike=187, dip=49, rake=-78,
    mode="pga_pgv",
    plot_GMIMs=(-1, -2, "Intensity"),
    plot_observations="raw",  # 仅图中使用原始场地观测
    plot_path="GB18306_inversion_raw_observations.png",
)
```

## 环境与测试

当前回归环境：`D:\Software\Miniconda\envs\python310\python.exe`，Python 3.10。

```powershell
D:\Software\Miniconda\envs\python310\python.exe -m pip install -r requirements.txt
D:\Software\Miniconda\envs\python310\python.exe -m unittest discover -s tests -v
```

测试覆盖两模型正反算闭合、共享椭圆场的一致性、正确系数、RotD50/H/EPA/EPV
回退优先级、Vs30 修正与绘图开关、缺测参数和 GB/T 17742 调用方式。
