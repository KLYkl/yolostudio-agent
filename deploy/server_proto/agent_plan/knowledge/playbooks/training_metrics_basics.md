# 训练指标解释基础

训练后优先分清楚：你现在看到的是训练过程信号，还是足够支撑结论的验证指标。

- 只有 loss：可以观察过程，但不能替代 precision / recall / mAP
- precision 高、recall 低：更像漏检偏多、覆盖不足或阈值偏保守
- precision 低、recall 高：更像误检偏多、类别边界不清或阈值偏激进
- mAP 整体低：先做错检/漏检分析，不要直接同时改很多参数

参考：
- Ultralytics Performance Metrics Deep Dive: https://docs.ultralytics.com/guides/yolo-performance-metrics/
- Ultralytics Model Training Tips: https://docs.ultralytics.com/guides/model-training-tips/
