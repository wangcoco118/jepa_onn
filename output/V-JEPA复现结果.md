# V-JEPA IntPhys Dev 结果汇总

## Split 说明

| 结果来源 | Split | 说明 |
|---|---|---|
| 本地复现 | IntPhys public Dev | 你本地使用的是 /data/linux/wkx/IntPhys/data/dev，有公开标签并计算了完整指标。 |
| 论文逐属性 accuracy | IntPhys public Dev | 论文说明主评测使用 Dev 子集；你引用的 85.7%、83.7%、86.3% 属于这部分。 |
| 论文 Table S4/S5 | IntPhys private Test | 论文表题明确写的是 IntPhys test set；该测试集标签不公开。 |
| 官方结果包 intphys/performance.csv | Public Dev / regular evaluation（根据 README 推断） | 官方 README 区分了有指标的 intuitive_physics regular evaluation 和不直接输出指标的 intphys_test private-test evaluation。因此该结果包不是 Table S4/S5 的 private Test 结果。 |

## 1. 本地复现结果（IntPhys public Dev）

模型：plain V-JEPA ViT-L/16，checkpoint：vitl16.pth.tar  
数据：IntPhys Dev，O1/O2/O3，frame skip = 2  
结果文件：vjepa_vitl16_r0.csv

本地 CSV 的原始指标字段为：

~~~text
Relative Accuracy (avg)
Relative Accuracy (max)
Absolute Accuracy (max)
Classifier threhshold
Best Absolute Accuracy (max)
Best Classifier threhshold
AUPRC (avg)
AUPRC (max)
AUROC (avg)
AUROC (max)
~~~



### 本地 Filtered 原始指标

| 原始指标 | O1 | O2 | O3 | 宏平均 |
|---|---:|---:|---:|---:|
| Relative Accuracy (avg) | 75.00000% | 83.33333% | 85.00000% | 81.11111% |
| Relative Accuracy (max) | 68.33334% | 61.66667% | 58.33333% | 62.77778% |
| Absolute Accuracy (max) | 50.00000% | 48.33333% | 49.16667% | 49.16667% |
| Classifier threhshold | 0.56509 | 0.56730 | 0.57123 | 0.56787 |
| Best Absolute Accuracy (max) | 60.83334% | 59.16666% | 58.33333% | 59.44444% |
| Best Classifier threhshold | 0.51324 | 0.53758 | 0.51097 | 0.52060 |
| AUPRC (avg) | 0.59376 | 0.55390 | 0.57137 | 0.57301 |
| AUPRC (max) | 0.65439 | 0.61951 | 0.61972 | 0.63121 |
| AUROC (avg) | 0.54306 | 0.52806 | 0.54694 | 0.53935 |
| AUROC (max) | 0.57875 | 0.56111 | 0.55458 | 0.56481 |

### 本地不同 context 的原始结果

这里的指标原始名称仍然是 Relative Accuracy (avg)，没有转换。

| Block | Context 2 | Context 4 | Context 6 | Context 8 | Context 10 | Filtered |
|---|---:|---:|---:|---:|---:|---:|
| O1 | 45.00000% | 73.33334% | 75.00000% | 75.00000% | 75.00000% | 75.00000% |
| O2 | 21.66667% | 70.00000% | 76.66666% | 73.33334% | 81.66666% | 83.33333% |
| O3 | 33.33334% | 86.66666% | 88.33334% | 86.66666% | 81.66666% | 85.00000% |

当前 default_intphys.yaml 的默认设置是：

~~~yaml
context_lengths: [2, 4, 6, 8, 10]
frame_steps: 2
~~~

因此没有指定 context 时，程序会评估五种 context，而不是只使用一个历史帧长度。

## 2. 官方论文结果（Dev 与 private Test 分开）


### 论文 Dev 的逐属性 accuracy（public Dev）


| 论文原始指标 | O1 Object Permanence | O2 Shape Constancy | O3 Continuity |
|---|---:|---:|---:|
| Per-property accuracy，mean ± SD，n=5 | 85.7% ± 7.6% | 83.7% ± 7.8% | 86.3% ± 6.2% |


### 论文附录中的原始 error 指标（private Test）

下面保留论文 Table S4 和 Table S5 的原始形式。这里的数据是 IntPhys private test

| 数据范围 | 论文原始指标 | Surprise | O1 | O2 | O3 |
|---|---|---|---:|---:|---:|
| Private test，All | Pairwise error rate | Avg | 1.4% | 3.1% | 2.0% |
| Private test，All | Pairwise error rate | Max | 0.20% | 21.9% | 23.8% |
| Private test，All | Single-video classification error rate (1-AUROC) | Avg | 41.5% | 42.7% | 41.5% |
| Private test，All | Single-video classification error rate (1-AUROC) | Max | 40.0% | 41.8% | 41.6% |

论文还分别报告了 Visible 和 Occluded 条件：

| 论文原始指标 | Surprise | 条件 | O1 | O2 | O3 |
|---|---|---|---:|---:|---:|
| Pairwise error rate | Avg | Visible | 0.9% | 2.5% | 0.7% |
| Pairwise error rate | Avg | Occluded | 1.8% | 3.5% | 3.3% |
| Pairwise error rate | Max | Visible | 5.2% | 8.8% | 5.9% |
| Pairwise error rate | Max | Occluded | 35.4% | 35.0% | 41.5% |
| Single-video classification error rate (1-AUROC) | Avg | Visible | 33.4% | 37.0% | 34.4% |
| Single-video classification error rate (1-AUROC) | Avg | Occluded | 41.7% | 42.5% | 38.8% |
| Single-video classification error rate (1-AUROC) | Max | Visible | 25.5% | 29.9% | 26.0% |
| Single-video classification error rate (1-AUROC) | Max | Occluded | 47.8% | 47.8% | 49.0% |


## 3. 下载结果包中的结果（public Dev / regular evaluation）

结果包：official_data_intphys.tar.gz  
使用的结果文件：data_intphys/vit-l-rope-howto/intphys/performance.csv  
模型：vit-l-rope-howto，带 RoPE 的 ViT-L HowTo 模型，不是当前的 vitl16.pth.tar。  
Split：public Dev / regular evaluation（根据官方 README 推断；CSV 本身没有单独的 split 列）。

官方 CSV 与本地 CSV 使用同一套原始字段：

~~~text
Relative Accuracy (avg)
Relative Accuracy (max)
Absolute Accuracy (max)
Classifier threhshold
Best Absolute Accuracy (max)
Best Classifier threhshold
AUPRC (avg)
AUPRC (max)
AUROC (avg)
AUROC (max)
~~~

### 官方结果包 Filtered 原始指标

| 原始指标 | O1 | O2 | O3 | 宏平均 |
|---|---:|---:|---:|---:|
| Relative Accuracy (avg) | 91.66667% | 88.33334% | 90.00000% | 90.00000% |
| Relative Accuracy (max) | 68.33334% | 56.66667% | 70.00000% | 65.00000% |
| Absolute Accuracy (max) | 49.16667% | 48.33333% | 49.16667% | 48.88889% |
| Classifier threhshold | 0.60386 | 0.60914 | 0.61520 | 0.60940 |
| Best Absolute Accuracy (max) | 61.66667% | 59.16666% | 65.83334% | 62.22222% |
| Best Classifier threhshold | 0.54428 | 0.57511 | 0.57606 | 0.56515 |
| AUPRC (avg) | 0.64354 | 0.61732 | 0.64038 | 0.63375 |
| AUPRC (max) | 0.67061 | 0.61338 | 0.67217 | 0.65205 |
| AUROC (avg) | 0.59694 | 0.57111 | 0.60000 | 0.58935 |
| AUROC (max) | 0.59194 | 0.51986 | 0.60444 | 0.57208 |

### 官方结果包不同 context 的原始指标

| 原始指标 | Block | Context 2 | Context 4 | Context 6 | Context 8 | Context 10 | Filtered |
|---|---|---:|---:|---:|---:|---:|---:|
| Relative Accuracy (avg) | O1 | 93.33334% | 95.00000% | 93.33334% | 93.33334% | 91.66667% | 91.66667% |
| Relative Accuracy (avg) | O2 | 86.66666% | 88.33334% | 76.66666% | 80.00000% | 88.33334% | 88.33334% |
| Relative Accuracy (avg) | O3 | 91.66667% | 90.00000% | 88.33334% | 90.00000% | 90.00000% | 90.00000% |
| Relative Accuracy (max) | O1 | 60.00000% | 61.66667% | 56.66667% | 76.66666% | 63.33333% | 68.33334% |
| Relative Accuracy (max) | O2 | 40.00000% | 58.33333% | 53.33334% | 58.33333% | 63.33333% | 56.66667% |
| Relative Accuracy (max) | O3 | 53.33334% | 56.66667% | 61.66667% | 60.00000% | 55.00000% | 70.00000% |

## 4. 加入光学神经网络后的本地结果

结果目录：

~~~text
/data/linux/wkx/IntPhys/references_code/jepa_onn/output/optical_best_20260825_175014
~~~

结果文件：

~~~text
intphys_dev_eval_gpu0/intphys-vjepa_vitl16_fsonn_tdm/vjepa_vitl16_fsonn_tdm_r0.csv
~~~

模型结构为 V-JEPA ViT-L/16，Encoder 和 Target Encoder 使用官方 checkpoint；Predictor 的 12 个 QKV 全部替换为 FSONN-TDM 光学 QKV。当前配置为 4 层 SLM、3 个 25 mm 层间距、band-limited ASM、signed phase 输入编码。训练使用 800 个 Train 视频、100 个内部验证视频、batch size 25、50 轮。

需要注意：本次最终 Dev 评估加载的是 optical_best.pt，对应第 47 轮最佳验证损失 0.537753；不是第 50 轮的 optical_best.final.pt。

### 光学模型 Filtered 原始指标

| 原始指标 | O1 | O2 | O3 | 宏平均 |
|---|---:|---:|---:|---:|
| Relative Accuracy (avg) | 83.33333% | 98.33334% | 96.66666% | 92.77778% |
| Relative Accuracy (max) | 63.33333% | 66.66667% | 68.33333% | 66.11111% |
| Absolute Accuracy (max) | 49.16667% | 49.16667% | 49.16667% | 49.16667% |
| Best Absolute Accuracy (max) | 62.50000% | 58.33333% | 61.66667% | 60.83333% |
| AUPRC (avg) | 0.57747 | 0.55670 | 0.61196 | 0.58204 |
| AUPRC (max) | 0.66644 | 0.61082 | 0.65349 | 0.64358 |
| AUROC (avg) | 0.55694 | 0.54333 | 0.56306 | 0.55444 |
| AUROC (max) | 0.58806 | 0.57139 | 0.56944 | 0.57630 |

### Context length = 10

| Block | Optical RA(avg) | Official local RA(avg) | Optical RA(max) | Official local RA(max) | Optical AUROC(max) | Official local AUROC(max) |
|---|---:|---:|---:|---:|---:|---:|
| O1 | 75.00% | 75.00% | 75.00% | 60.00% | 0.60903 | 0.61347 |
| O2 | 91.67% | 81.67% | 75.00% | 41.67% | 0.59625 | 0.53181 |
| O3 | 88.33% | 81.67% | 75.00% | 58.33% | 0.57639 | 0.56431 |

## 5. 电子 Predictor 对照实验

结果目录：

~~~text
/data/linux/wkx/IntPhys/references_code/jepa_onn/output/electronic_control
~~~

该实验从官方 V-JEPA checkpoint 加载原始电子 Encoder、Target Encoder 和 Predictor，冻结 Encoder 与 Target Encoder，只对完整电子 Predictor 继续训练 10 轮。训练配置为 800 个 Train 视频、100 个内部验证视频、batch size 25、learning rate 0.0001。

训练损失变化：

| Epoch | Train JEPA loss | Val JEPA loss |
|---|---:|---:|
| 1 | 0.525049 | 0.535178 |
| 5 | 0.487597 | 0.520694 |
| 10 | 0.473506 | 0.510293 |

验证损失从 0.535178 降至 0.510293，下降约 4.65%，第 10 轮为最佳轮次。

### 电子对照的已完成评估结果

电子评估没有完整结束。CSV 目前只包含 O1 的 frame skip 2、4、6，尚未产生 O2/O3 或 O1 的 frame skip 8、10、Filtered 完整结果。因此下面只报告 O1 已完成部分：

| Block | Frame skip | Context | RA(avg) | RA(max) | AUROC(avg) | AUROC(max) |
|---|---:|---|---:|---:|---:|---:|
| O1 | 2 | Filtered | 81.66666% | 78.33334% | 0.55417 | 0.58806 |
| O1 | 4 | Filtered | 86.66666% | 86.66666% | 0.57889 | 0.59028 |
| O1 | 6 | Filtered | 88.33334% | 88.33334% | 0.58222 | 0.58222 |

因此，电子对照训练本身是完成的，但电子对照的 Dev 测试结果目前是不完整的，不能用于完整的 O1/O2/O3 总体排名。

## 6. 指标定义与两套计算口径

本项目同时保留两类指标，但本节只解释它们的口径差异：

- JEPA 普通指标：来自 `compute_metrics()`，将四视频组拆成两组 possible/impossible 后进行两两比较。
- 官方 IntPhys 指标：来自 `compute_official_metrics()`，保留完整的 2 possible + 2 impossible 四视频组。

普通 `Absolute Accuracy(max)` 是基于 possible 视频最大 surprise 的约 90% 分位点进行阈值分类；`Best Absolute Accuracy(max)` 是扫描候选阈值后的最高分类准确率。官方绝对指标不是阈值准确率，而是：

~~~text
plausibility = -surprise
official_RA = 1 - LR
official_LA = 1 - AUC
~~~

其中 RA 越高越好，LA 是绝对错误率，越低越好。AUC 越高越好。

## 7.JEPA 指标与官方指标对比

### 实验对象与实验目的

固定条件：

- checkpoint：本地 `vitl16.pth.tar`
- 数据：IntPhys public Dev
- Block：O1/O2/O3
- frame skip：2
- Context：2、4、6、8、10
- Filtered：从五个 Context 中选择最低 surprise 后计算

两个结果目录分别是：

~~~text
JEPA 普通指标：
/data/linux/wkx/IntPhys/references_code/references/checkpoints/vjepa/intuitive_physics/intphys-vjepa_vitl16

官方指标补充结果：
/data/linux/wkx/IntPhys/references_code/references/checkpoints/vjepa/intuitive_physics/intphys-vjepa_vitl16_official_20260817_141505
~~~

第二个目录不是另一个模型，而是同一本地 V-JEPA 结果追加了官方四视频组字段。因此本实验是：

~~~text
固定模型、checkpoint、数据和 surprise，只更换指标计算公式。
~~~

### JEPA 普通口径：Filtered

| 指标 | O1 | O2 | O3 | 宏平均 |
|---|---:|---:|---:|---:|
| RA(avg) | 75.00% | 83.33% | 85.00% | 81.11% |
| RA(max) | 68.33% | 61.67% | 58.33% | 62.78% |
| Absolute Accuracy(max) | 50.00% | 48.33% | 49.17% | 49.17% |
| Best Absolute Accuracy(max) | 60.83% | 59.17% | 58.33% | 59.44% |
| AUPRC(avg) | 0.59376 | 0.55390 | 0.57137 | 0.57301 |
| AUPRC(max) | 0.65439 | 0.61951 | 0.61972 | 0.63121 |
| AUROC(avg) | 0.54306 | 0.52806 | 0.54694 | 0.53935 |
| AUROC(max) | 0.57875 | 0.56111 | 0.55458 | 0.56481 |

### 官方 IntPhys 口径：Filtered

| 指标 | O1 | O2 | O3 | 宏平均 |
|---|---:|---:|---:|---:|
| official mean RA | 100.00% | 86.67% | 96.67% | 94.44% |
| official mean AUC | 0.54306 | 0.52806 | 0.54694 | 0.53935 |
| official mean LA | 0.45694 | 0.47194 | 0.45306 | 0.46065 |
| official max RA | 86.67% | 86.67% | 83.33% | 85.56% |
| official max AUC | 0.57875 | 0.56111 | 0.55458 | 0.56481 |
| official max LA | 0.42125 | 0.43889 | 0.44542 | 0.43519 |

### 更换口径后的直接变化

| Block | JEPA RA(avg) | 官方 mean RA | 变化 |
|---|---:|---:|---:|
| O1 | 75.00% | 100.00% | +25.00 个百分点 |
| O2 | 83.33% | 86.67% | +3.34 个百分点 |
| O3 | 85.00% | 96.67% | +11.67 个百分点 |

| Block | JEPA RA(max) | 官方 max RA | 变化 |
|---|---:|---:|---:|
| O1 | 68.33% | 86.67% | +18.34 个百分点 |
| O2 | 61.67% | 86.67% | +25.00 个百分点 |
| O3 | 58.33% | 83.33% | +25.00 个百分点 |


~~~text
JEPA AUROC(avg) = 官方 mean AUC
JEPA AUROC(max) = 官方 max AUC
~~~

### 本实验结论

这是一项**同一模型的评测口径对照实验**，不是光学模型实验，也不是电子 Predictor 训练实验。它说明：

1. 同一个本地 V-JEPA，在 JEPA 普通口径和官方口径下会得到不同的 RA；
2. 官方四视频组 RA 在本数据上普遍高于 JEPA 两两配对 RA；
3. AUC 在两套口径下保持一致；
4. JEPA 的 Absolute Accuracy 不能直接当作官方 LA；
5. 后续与 IntPhys 官方结果比较时，应优先使用 `official_RA`、`official_AUC` 和 `official_LA`，而不是把两套 RA 混在同一张表里。
