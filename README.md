# GB18306 / CEA2019 椭圆衰减模型应用包

本包用于把 GB18306-2015 和 CEA2019 长短轴椭圆衰减关系应用到单次地震。
核心问题是模型需要宏观震中，而有限断层模型通常给出的是初始破裂点；二者可先
近似为同一点，也可以利用观测记录反演最优宏观震中。

当前发布版本：**v1.0**（2026-08-18）。

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

Pre、vs_Obs 和 epicenter_inversion 均通过 `ellipse_fields.py` 使用同一解析场，
避免不同应用之间出现近场采样或插值差异。

## 观测列选择

PGA/PGV 保持以下优先级：

```text
EPA_H / EPV_H  →  PGA_H / PGV_H  →  PGA / PGV
```

CEA2019 的 PSA 默认优先使用 `pSa(T=xx.xx s)_H`，再回退 RotD50 和无后缀列。
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

## 环境与测试

当前回归环境：`D:\Software\Miniconda\envs\python310\python.exe`，Python 3.10。

```powershell
D:\Software\Miniconda\envs\python310\python.exe -m pip install -r requirements.txt
D:\Software\Miniconda\envs\python310\python.exe -m unittest discover -s tests -v
```

测试覆盖两模型正反算闭合、共享椭圆场的一致性、正确系数、EPA/EPV 回退优先级、
缺测参数和 GB/T 17742 调用方式。
