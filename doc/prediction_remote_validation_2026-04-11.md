# Prediction 远端真实验证（2026-04-11）

## 完成结果

已在远端服务器 `yolostudio` 上完成一轮 **真实视频 prediction 验证**，并把结果拉回本地：

- 本地结果文件：
  - `C:\workspace\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media_output.json`

## 验证输入

### 远端素材目录

- 权重：
  - `/data/prediction_real_media_stage/weights/zq-4-06-qcar.pt`
  - `/data/prediction_real_media_stage/weights/zq-4-2.pt`
  - `/data/prediction_real_media_stage/weights/zq-4-3.pt`
- 视频：
  - `/data/prediction_real_media_stage/videos/fyb2026-03-06_094015_491.mp4`
  - `/data/prediction_real_media_stage/videos/fyb2026-03-06_094125_133.mp4`
  - `/data/prediction_real_media_stage/videos/zyb_2026-03-03_125605_456.mp4`

### 远端执行环境

- conda root：
  - `/opt/conda`
- 已发现环境：
  - `base`
  - `yolodo`
  - `yolostudio-agent-server`
- 本次实际使用：
  - `yolodo`

## 关键结果

### prediction 结果

- 选用权重：
  - `/data/prediction_real_media_stage/weights/zq-4-06-qcar.pt`
- 实际处理视频数：
  - `2`
- 总帧数：
  - `24`
- 有检测帧：
  - `13`
- 总检测框：
  - `15`
- 主要类别：
  - `two_wheeler=15`

### 远端输出目录

- 汇总目录：
  - `/tmp/prediction_real_media_output/codex_roundtrip/videos`
- 报告文件：
  - `/tmp/prediction_real_media_output/codex_roundtrip/videos/video_prediction_report.json`

## 结论

到这一步可以确认：

1. `predict_videos` 已经不只是本地工具级完成，而是**完成了远端真实执行**
2. `summarize_prediction_results` 已经能基于远端真实产物给出结构化结果
3. 预测链路已经从“本地可用”推进到“远端可验证”

## 当前剩余问题

仍需继续补的不是“能不能跑”，而是：

1. 把这轮远端验证收成固定回归基线
2. 再补图片 prediction 的远端真实验证
3. 继续做结构整理，控制 `agent_client.py` / `data_tools.py` / `predict_service.py` 的膨胀
