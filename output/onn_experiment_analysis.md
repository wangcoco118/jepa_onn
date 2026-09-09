# JEPA_ONN 实验结果与官方 V-JEPA 结果对比

> 本文只汇总 O1。Dev 指标来自本地公开 Dev 评测；Test 官方指标来自已有官网返回结果。没有官网返回值的 Test 实验不自行填充。

## 一、指标说明

- **LR（Relative Error，相对错误率）**：越低越好。
- **RA（Relative Accuracy，相对正确率）**：RA=1-LR，越高越好。
- **LA（Absolute Error，绝对错误率）**：越低越好。
- **AUC（Area Under ROC Curve，ROC 曲线下面积）**：AUC=1-LA，越高越好。
- Test 中的 `relative/absolute` 是官方服务器返回的 O1 指标；本地 `per_movie_scores.csv` 中的 surprise/plausibility 不能替代官方指标。

## 二、当前实验参数

| 实验                 | 预测器                      | 是否反馈         | 反馈层           | 是否记忆      | 训练 Batch |    训练轮数 |
| -------------------- | --------------------------- | ---------------- | ---------------- | ------------- | ---------: | ----------: |
| Transformer baseline | Transformer Predictor       | 否               | —               | 不适用        |         25 |          50 |
| 单层反馈 SLM4        | 光学 ONN Predictor          | 是               | SLM4             | 否            |         25 |          50 |
| 多层反馈 + 记忆      | 光学 ONN Predictor          | 是               | SLM3、SLM4、SLM5 | 是，alpha=0.8 |         25 |          25 |
| 多层反馈，无记忆     | 光学 ONN Predictor          | 是               | SLM3、SLM4、SLM5 | 否            |         25 |          25 |
| 单层反馈 SLM2        | 光学 ONN Predictor          | 是               | SLM2             | 否            |         25 |          30 |
| 反馈关闭（置零）     | 光学 ONN Predictor          | 否，反馈值全为 0 | —               | 否            |         25 |          25 |
| 电子控制基线         | 旧版电子控制（optical_qkv） | 非当前 ONN 反馈  | 不适用           | 不适用        |          1 | 配置记录 50 |

## 三、当前实验 Dev 指标

以下指标均来自本地 Dev 评测，数值使用百分比显示。

- **LR（Relative Error，相对错误率）**：越低越好。
- **RA（Relative Accuracy，相对正确率）**：越高越好。
- **LA（Absolute Error，绝对错误率）**：越低越好。
- **AUC（Area Under ROC Curve，ROC 曲线下面积）**：越高越好。

### Dev 平均聚合

| 实验                 | LR↓ 相对错误率 | RA↑ 相对正确率 | LA↓ 绝对错误率 | AUC↑ ROC面积 |
| -------------------- | --------------: | --------------: | --------------: | ------------: |
| Transformer baseline |         0.0000% |       100.0000% |        45.8060% |      54.1940% |
| 单层反馈 SLM4        |         0.0000% |       100.0000% |        46.0000% |      54.0000% |
| 多层反馈 + 记忆      |         0.0000% |       100.0000% |        46.0280% |      53.9720% |
| 多层反馈，无记忆     |         0.0000% |       100.0000% |        45.6110% |      54.3890% |
| 单层反馈 SLM2        |         0.0000% |       100.0000% |        45.7220% |      54.2780% |
| 反馈关闭（置零）     |         0.0000% |       100.0000% |        45.6110% |      54.3890% |

### Dev 最大聚合

| 实验                 | LR↓ 相对错误率 | RA↑ 相对正确率 | LA↓ 绝对错误率 | AUC↑ ROC面积 |
| -------------------- | --------------: | --------------: | --------------: | ------------: |
| Transformer baseline |         0.0000% |       100.0000% |        42.8060% |      57.1940% |
| 单层反馈 SLM4        |         0.0000% |       100.0000% |        43.1110% |      56.8890% |
| 多层反馈 + 记忆      |         3.3330% |        96.6670% |        43.6110% |      56.3890% |
| 多层反馈，无记忆     |         0.0000% |       100.0000% |        43.1250% |      56.8750% |
| 单层反馈 SLM2        |         0.0000% |       100.0000% |        42.9860% |      57.0140% |
| 反馈关闭（置零）     |         0.0000% |       100.0000% |        42.7080% |      57.2920% |

## 四、当前实验 Test 官方指标

以下是已经获得官网返回值的 O1 Test 结果。没有官网返回值的实验不填充数值。

| 实验                 | 聚合方式 | 官方 relative↓ 相对错误率 | 官方 absolute↓ 绝对错误率 |
| -------------------- | -------- | -------------------------: | -------------------------: |
| Transformer baseline | average  |                    0.6481% |                   47.1493% |
| 反馈关闭（置零）     | average  |                    0.6481% |                   47.1493% |
| 单层反馈 SLM4        | average  |                    0.5556% |                   47.0887% |
| 单层反馈 SLM4        | maximum  |                   16.0185% |                   43.8279% |

## 五、V-JEPA 官方参考结果

以下数据均来自 `V-JEPA复现结果.md`。不同来源的模型、数据划分或指标口径可能不同，因此只能作为参考，不能直接当作完全同条件的排名。

| 来源                   | 数据范围 / 模型                        |      O1 相对结果 |                     O1 绝对结果 | 备注                                               |
| ---------------------- | -------------------------------------- | ---------------: | ------------------------------: | -------------------------------------------------- |
| V-JEPA 论文 Dev        | Public Dev，逐属性 accuracy            |    85.7% ± 7.6% |                              — | 论文中的 per-property accuracy，不等同于当前 LR/RA |
| V-JEPA 论文 Test       | Private Test，pairwise error，平均     |             1.4% |                              — | 标签隐藏，论文报告的官方参考值                     |
| V-JEPA 论文 Test       | Private Test，pairwise error，最大     |            0.20% |                              — | 与 average 聚合口径不同                            |
| V-JEPA 论文 Test       | Private Test，single-video error，平均 |               — |                           41.5% | 绝对错误口径为 1-AUROC                             |
| V-JEPA 论文 Test       | Private Test，single-video error，最大 |               — |                           40.0% | 绝对错误口径为 1-AUROC                             |
| 官方结果包             | Public Dev，vit-l-rope-howto           | RA(avg)=91.6667% | Absolute Accuracy(max)=49.1667% | 不是当前`vitl16.pth.tar` 模型                    |
| 本地 plain V-JEPA 复现 | Public Dev，ViT-L/16                   | RA(avg)=75.0000% | Absolute Accuracy(max)=50.0000% | 当前本地复现参考                                   |

官方结果包在 O1 上的其他重要指标为：`RA(max)=68.3333%`、`AUROC(avg)=0.59694`、`AUROC(max)=0.59194`。

## 六、O1 Test 的直接对照

| 实验                 | 聚合方式 |       官方 relative |            官方 absolute | 与已知参考的关系                                                     |
| -------------------- | -------- | ------------------: | -----------------------: | -------------------------------------------------------------------- |
| Transformer baseline | average  |             0.6481% |                 47.1493% | 当前 Transformer 基线                                                |
| 反馈关闭             | average  |             0.6481% |                 47.1493% | 与 baseline 的已知结果相同                                           |
| 单层反馈 SLM4        | average  |             0.5556% |                 47.0887% | 相对指标较 baseline 低 0.0926 个百分点，绝对指标略低 0.0605 个百分点 |
| 单层反馈 SLM4        | maximum  |            16.0185% |                 43.8279% | absolute 变好，但 relative 明显变差                                  |
| V-JEPA 论文参考      | average  | 1.4% pairwise error | 41.5% single-video error | 不同模型和官方 private Test 参考，不是严格同条件实验                 |

![1788328695527](image/onn_experiment_analysis/1788328695527.png)

![1788328732939](image/onn_experiment_analysis/1788328732939.png)

## 七、结论

1. 在目前已经获得官网返回值的实验中，**SLM4 的 average 聚合最稳定**：relative=0.5556%，absolute=47.0887%，两项都略优于 Transformer baseline。
2. SLM4 的 maximum 聚合表现出明显取舍：absolute 降至 43.8279%，但 relative 升至 16.0185%。因此不能只根据单个指标宣布 maximum 更好。
3. 本地 Dev 中，反馈关闭模型的最大 LA=42.7080%，是当前 ONN 组中较好的结果；但它的官方 Test average 与 baseline 相同，因此尚不能说明关闭反馈更优。
4. 多层无记忆版本的 Dev LA 略优于多层记忆版本：平均 LA 为 45.6110%，而记忆版本为 46.0280%。这只是 Dev 现象，仍需要官方 Test 结果确认。
5. 与 V-JEPA 论文参考相比，当前 SLM4 的 relative error 低于论文的 1.4% 参考值，但 absolute error 高于论文的 41.5% 参考值，说明相对排序和绝对区分能力并未同时改善。
6. `electronic_control` 属于旧版 `optical_qkv` 路径，数据和评测设置不同，不应与当前 ONN Predictor 直接合并排名。

**总体判断：** 当前 ONN 反馈方向是有效的，SLM4 average 已经显示出小幅、可重复解释的 Test 改善；但绝对指标仍是主要短板，后续应重点优化 surprise 的绝对尺度和 possible/impossible 的全局分离能力，而不是继续单纯追求 maximum 聚合。
