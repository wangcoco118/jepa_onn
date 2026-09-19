# 9.17 结果可视化说明

1. 00_experiment_dashboard.png：总体仪表盘。
2. 01_full_test_heatmap.png：完整 Test 热图。
3. 04_parameter_vs_performance.png：参数量与性能权衡；不同 predictor 类型用不同颜色和 marker 区分，图例放在图下方。copy-last 参数量为 0，仅在 CSV/Markdown 中列出。
4. 05_feedback_vs_no_feedback.png：2×3 柱状图，按 1/3/5 层分别直接比较 no-feedback、feedback interpolate 和 feedback linear 的 O1/O2/O3 结果，不使用场景平均值。
5. 06_layer_comparison.png：四图合并预览，每个面板有独立图例。
6. 06a_relative_no_feedback.png：relative error，无反馈。
7. 06b_relative_feedback_linear.png：relative error，linear feedback。
8. 06c_absolute_no_feedback.png：absolute error，无反馈。
9. 06d_absolute_feedback_linear.png：absolute error，linear feedback。
10. 07_output_head_comparison.png：interpolate 与 linear 输出头比较。
11. 08_direct384_vs_interpolate.png：比较 direct384 与 384→1024 恢复输出头；SLM3 direct384 缺少官方 scores.txt 时显示 N/A。

结果分析文档中的四组比较为：层数、反馈、维度恢复、Transformer/copy-last 基础对比。
O1、O2、O3 分别对应 Object permanence、Shape constancy、Continuity；error rate 越低越好。
