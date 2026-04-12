# 本地真实权重 / 视频预测验证（2026-04-11）

## 目标

验证第二主线在接入真实本地权重池与真实视频池时，测试方法是否能稳定覆盖：

1. 权重与视频素材盘点
2. 本机 YOLO 推理环境探测
3. 使用真实视频素材进行 Mock 预测链路验证
4. 在环境允许时执行真实推理

## 素材盘点

### 权重样本
- `zq-4-06-qcar.pt` (5504211 bytes)
- `zq-4-3.pt` (5503827 bytes)
- `zq-4-2.pt` (5491859 bytes)

### 视频样本
- `fyb2026-03-06_094015_491.mp4` (1477643 bytes)
- `fyb2026-03-06_094125_133.mp4` (1730558 bytes)
- `zyb_2026-03-03_125605_456.mp4` (1839444 bytes)

## 本机推理环境探测

- python: `D:\Anaconda\envs\yolo\python.exe`
- runtime_ok: `False`
- summary: 本机 YOLO 预测环境不可用：WinError 10106（_overlapped / winsock 提供程序异常）

## Mock 预测链路验证

- ok: `True`
- summary: 真实视频素材 Mock 验证通过：目录扫描、视频读取、报告生成、摘要汇总链路正常
- assessment: `5/5 -> 1.0`

## 真实推理验证

- blocked: `True`
- summary: 跳过真实推理：本机 YOLO 预测环境当前不可用

## 结论

- 当前测试方法已经升级到“真实素材 + 环境探测 + Mock 链路 + 有条件真实推理”的四段式。
- 这轮真正阻塞真实推理的不是 Agent 代码，而是本机 YOLO 运行环境在导入 `torch/ultralytics` 时失败。
