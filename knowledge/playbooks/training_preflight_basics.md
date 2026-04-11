# 训练前基础检查

第一阶段重点不是调参，而是确认数据是否足够可靠。优先看：

- 标签覆盖是否明显不足
- 是否存在坏图、重复图、结构问题
- 数据是否已经达到 readiness，可直接进入训练

如果当前 readiness 还没通过，或者数据质量风险已经很明显，先修数据再训练，通常比直接调参更有效。

参考：
- Ultralytics Detect Datasets: https://docs.ultralytics.com/datasets/detect/
- Ultralytics Model Training Tips: https://docs.ultralytics.com/guides/model-training-tips/
