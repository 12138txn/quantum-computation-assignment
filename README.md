# 量子计算前沿大作业

这个文件夹是课程大作业的完整包，包含 LaTeX 正文、图片、数据和源码。

## 结构

- `main.tex`: 主 LaTeX 文件，建议用 XeLaTeX 编译。
- `figures/`: 正文引用的图片。
- `data/`: 正文表格和图片对应的 CSV 数据。
- `src/`: 当前版本模拟源码和扫描脚本。

## 编译方式

报告使用中文 `ctexart`，推荐 XeLaTeX：

```powershell
xelatex main.tex
xelatex main.tex
```

如果使用 TeXstudio 或 Overleaf，请选择 XeLaTeX 编译器。

## 主要结果文件

冷却模拟：

- `figures/coarse_scan_overview.png`
- `figures/refine_ratio_initial.png`
- `figures/refine_eom_final.png`
- `figures/refine_blue.png`
- `figures/refine_intensity.png`
- `figures/low_ratio_refine.png`
- `figures/time_ratio_0p06.png`
- `figures/time_curve_comparison.png`
- `figures/pe_ratio_0p06_quasisteady.png`
- `data/processed_summary.csv`
- `data/coarse_scan_main.csv`
- `data/local_refine_main_n6.csv`
- `data/local_refine_high_n9.csv`
- `data/time_curve_ratio_0p06.csv`
- `data/time_curve_ratio_0p025.csv`

装载率模拟：

- `figures/loading_detuning_scan.png`
- `figures/loading_time_curves.png`
- `figures/loading_best_state_probabilities.png`
- `figures/loading_scan_cooling_context.png`
- `data/loading_detuning_scan.csv`
- `data/loading_time_curve_best.csv`

## data 文件说明

`data/` 中的 CSV 文件用于保存正文表格和图片背后的数值结果。

| 文件 | 内容 | 对应正文/图片 |
| --- | --- | --- | 
| `coarse_scan_main.csv` | 第一阶段大范围单参数粗扫结果，包含边带强度比、Raman 失谐、蓝失谐和总光强扫描。 | `figures/coarse_scan_overview.png`，用于确定候选参数区域。 |
| `local_refine_main_n6.csv` | 局部精扫主数据，Fock 截断为 `n_fock=6`。 | `figures/refine_ratio_initial.png`、`figures/refine_eom_final.png`、`figures/refine_blue.png`、`figures/refine_intensity.png`。 |
| `local_refine_high_n9.csv` | 对局部精扫候选点进行高截断复核，Fock 截断为 `n_fock=9`。 | 用于确认候选冷却点的稳态声子数和高 Fock 态尾部。 |
| `ratio_low_refine_n6.csv` | 低边带强度比区域补扫，Fock 截断为 `n_fock=6`。 | `figures/low_ratio_refine.png` 的低边带趋势来源之一。 |
| `ratio_low_refine_high_n9.csv` | 低边带强度比候选点的高截断复核。 | 用于确认低边带比参考点的稳态结果。 |
| `processed_summary.csv` | 两个代表性冷却点的汇总表，包含 `n_ss`、`n(5 ms)`、冷却时间常数、散射率和散射效率指标。 | 正文表 `cooling_summary`。 |
| `time_curve_ratio_0p06.csv` | 最终采用冷却点 `I_s/I_c=0.06` 的完整时间演化曲线。 | `figures/time_ratio_0p06.png`、`figures/time_curve_comparison.png`、`figures/pe_ratio_0p06_quasisteady.png`。 |
| `time_curve_ratio_0p025.csv` | 低散射参考点 `I_s/I_c=0.025` 的完整时间演化曲线。 | `figures/time_curve_comparison.png`。 |
| `loading_detuning_scan.csv` | 蓝失谐装载率扫描，包含冷却稳态、碰撞通道 `beta1/beta2` 和最终 `P0/P1/P2`。 | `figures/loading_detuning_scan.png`、`figures/loading_scan_cooling_context.png`、正文表 `loading_summary`。 |
| `loading_time_curve_best.csv` | 装载率最优平台代表点的 `P0(t), P1(t), P2(t)` 时间曲线。 | `figures/loading_best_state_probabilities.png`。 |
## 源码

- `src/4f_test_d1_fixed.py`: 主模型，包含 D1 线冷却、碰撞阻塞和装载率方程。
- `src/run_local_refine_scan.py`: 冷却局部精扫。
- `src/run_best_time_evolution_diag.py`: 最优冷却点完整时间演化。
- `src/run_loading_detuning_scan.py`: 蓝失谐装载率扫描。
- `src/run_cooling_efficiency_scan.py`: 早期冷却效率扫描。

正文附录列出了装载率扫描脚本和核心函数片段；完整源码文件也随包保存。
