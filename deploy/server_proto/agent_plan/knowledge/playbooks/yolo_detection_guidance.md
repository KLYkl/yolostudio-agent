# YOLO Detection 调参与流程建议

当前第一阶段只对 YOLO + detection 落具体适配。策略是：

1. 先走 readiness / prepare 流程，确认 data.yaml 和训练结构
2. 再进入 start_training / status / stop
3. 对结果判断时，把 prediction 当辅助证据，不替代训练指标

YOLO 常见参数里，第一阶段优先关注：

- epochs：给足观察窗口，但不要脱离指标盲目拉长
- batch：优先适配显存，不要为了速度盲目增大
- imgsz：若主要问题是小目标漏检，可谨慎评估更高 imgsz
- patience：是训练控制参数，不是替代数据分析的捷径

参考：
- Ultralytics Train Mode: https://docs.ultralytics.com/modes/train/
- Ultralytics Hyperparameter Tuning Guide: https://docs.ultralytics.com/guides/hyperparameter-tuning/
- Ultralytics Model Training Tips: https://docs.ultralytics.com/guides/model-training-tips/
