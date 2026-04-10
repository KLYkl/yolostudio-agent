# 主线回归矩阵测试报告（2026-04-11）

- 运行批次: `20260411_001240`
- Case 数量: `17`
- 检查项通过: `48/64`
- 总体得分: `0.75`

## 1. 总结

- 未满分 case: `tool_unknown_fail_fast, agent_ollama_standard_no_train, agent_ollama_complex_chain, agent_ollama_dirty_summary, agent_ollama_health_grounded, agent_ollama_duplicate_grounded, agent_deepseek_dirty_summary, agent_deepseek_health_grounded`
- 说明：本报告把“执行层是否完成”与“解释层是否 grounded”混合纳入同一轮回归，用来观察当前版本的主线能力边界。

## 2. Case 结果

### tool_standard_root_scan
- 预期: 标准 root 被正确解析到 images/labels，scan 返回结构化统计
- 得分: 4/4 (1.0)

### tool_unknown_fail_fast
- 预期: 未知目录在 resolve_root 阶段提前失败，不继续 split/generate_yaml
- 得分: 2/3 (0.667)

### tool_nonstandard_prepare
- 预期: pics/ann 非标准目录被解析并准备到可训练状态
- 得分: 4/4 (1.0)

### tool_dirty_health
- 预期: 大数据脏数据集健康检查能返回重复组、格式问题与风险等级
- 得分: 4/4 (1.0)

### tool_dirty_readiness
- 预期: readiness 显式暴露 dirty dataset 风险与缺少 data_yaml 的 blocker
- 得分: 4/4 (1.0)

### agent_ollama_standard_no_train
- 预期: 遵守只检查不训练约束，并给出 grounded 结论
- 得分: 2/4 (0.5)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status`
- 回复摘要: Agent 未能生成有效回复。 可输入 /context 查看当前状态，或换一种方式描述需求。

### agent_ollama_complex_chain
- 预期: 复杂训练意图应收敛到 prepare -> start_training 两段式流程
- 得分: 1/4 (0.25)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status, dataset_manager.prepare_dataset`

### agent_ollama_dirty_summary
- 预期: 脏数据总结尽量 grounded，能说出缺失标签风险与真实类名/来源
- 得分: 0/1 (0.0)

### agent_ollama_health_grounded
- 预期: 健康检查应命中专用工具，并以 grounded 方式总结，不编造修改动作
- 得分: 0/4 (0.0)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status`
- 回复摘要: Agent 未能生成有效回复。 已调用工具: detect_corrupted_images, detect_duplicates 可输入 /context 查看当前状态，或换一种方式描述需求。

### agent_ollama_duplicate_grounded
- 预期: 重复检测应命中专用工具，并基于样例路径做 grounded 总结
- 得分: 1/4 (0.25)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status, detect_duplicates`
- 回复摘要: 请问您要检查哪个数据集的重复图片？请提供数据集的完整路径，我将使用 `detect_duplicates` 工具为您检查。

### agent_ollama_state_purity
- 预期: fresh session 仅查状态时，不应被旧训练参数污染
- 得分: 4/4 (1.0)
- Provider: `ollama` / `gemma4:e4b`
- 状态摘要: `{"running": false, "model": "", "data_yaml": "", "device": "", "last_status": {"ok": true, "running": false, "log_file": "runs/train_log_1775829436.txt", "device": "1", "requested_device": "auto", "command": ["/home/kly/miniconda3/bin/yolo", "train", "model=yolov8n.pt", "data=/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs=100", "device=1"], "started_at": 1775829436.981328, "yolo_executable": "/home/kly/miniconda3/bin/yolo", "device_policy": "single_idle_gpu", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "yolov8n.pt", "data_yaml": "/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 100, "device": "1", "device_policy": "single_idle_gpu"}, "reattached": false, "registry_path": "runs/active_train_job.json", "pid": 55368, "return_code": -15, "stopped_at": 1775829459.0967886, "stop_reason": "manual_stop", "last_run": {"pid": 55368, "log_file": "runs/train_log_1775829436.txt", "started_at": 1775829436.981328, "device": "1", "requested_device": "auto", "command": ["/home/kly/miniconda3/bin/yolo", "train", "model=yolov8n.pt", "data=/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs=100", "device=1"], "yolo_executable": "/home/kly/miniconda3/bin/yolo", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "yolov8n.pt", "data_yaml": "/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 100, "device": "1", "device_policy": "single_idle_gpu"}, "running": false, "return_code": -15, "forced": false, "stop_reason": "manual_stop", "updated_at": 1775829459.0967886, "stopped_at": 1775829459.0967886}, "elapsed_seconds": 22.12, "latest_metrics": {"ok": true, "metrics": {"epoch": 1, "total_epochs": 100, "gpu_mem": "2.14G", "box_loss": "1.656", "cls_loss": "9.483", "dfl_loss": "1.349"}}, "summary": "当前无训练在跑，最近一次训练已手动停止，return_code=-15", "next_actions": ["当前无训练在跑，可直接 start_training 启动新训练"]}}`

### agent_deepseek_standard_no_train
- 预期: 遵守只检查不训练约束，并给出 grounded 结论
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, training_readiness`
- 回复摘要: **检查结果：可以直接训练**  **原因分析：** 1. **数据完整性**：数据集包含 33 张图片，全部都有对应的标签文件，无缺失标签 2. **标签质量**：标签格式正确，无坐标或类别ID问题 3. **配置完整**：已检测到现有的 `data.yaml` 配置文件 4. **目录结构**：自动解析到标准结构：    - 图片目录：`/home/kly/test_dataset/images`    - 标签目录：`/home/kly/test_dataset/lab

### agent_deepseek_complex_chain
- 预期: 复杂训练意图应收敛到 prepare -> start_training 两段式流程
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, prepare_dataset_for_training`

### agent_deepseek_dirty_summary
- 预期: 脏数据总结尽量 grounded，能说出缺失标签风险与真实类名/来源
- 得分: 3/4 (0.75)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, run_dataset_health_check, validate_dataset`
- 回复摘要: ## 数据集质量分析结果  基于健康检查和标签验证，**最值得注意的 3 个风险**如下：  ### 1. **严重标签缺失问题** ⚠️ - **风险等级：严重（critical）** - 7027 张图片中，**5179 张（73.7%）缺少对应标签文件** - 这意味着超过三分之二的图片没有标注，训练时这些图片将被视为背景/负样本 - **影响**：模型可能无法学习到足够的正样本特征，训练效果会严重受损  ### 2. **大量重复图片** ⚠️ - **风险等级：高（

### agent_deepseek_health_grounded
- 预期: 健康检查应命中专用工具，并以 grounded 方式总结，不编造修改动作
- 得分: 3/4 (0.75)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, run_dataset_health_check`
- 回复摘要: 健康检查完成: 完整性问题 5, 异常尺寸 0, 重复组 83 关键风险: - 发现 5 个文件扩展名与真实格式不匹配 - 发现 83 组重复图片（额外重复文件 83 个） 完整性: 损坏 0 / 零字节 0 / 格式不匹配 5 尺寸: 异常小 0 / 异常大 0 重复图片: 83 组，额外重复文件 83 个 建议: - 建议先处理损坏/异常图片，再继续数据准备或训练 - 可先人工确认重复图片是否需要清理；如需进一步查看，可单独调用 detect_duplicate_imag

### agent_deepseek_duplicate_grounded
- 预期: 重复检测应命中专用工具，并基于样例路径做 grounded 总结
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, detect_duplicate_images`
- 回复摘要: 检测完成: 发现 83 组重复图片，额外重复文件 83 个 示例重复组: - /home/kly/agent_cap_tests/zyb/images/20161228230642.jpg, /home/kly/agent_cap_tests/zyb/images/training_training_image_20161228230642.jpg - /home/kly/agent_cap_tests/zyb/images/20161228230649.jpg, /home

### agent_deepseek_state_purity
- 预期: fresh session 仅查状态时，不应被旧训练参数污染
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 状态摘要: `{"running": false, "model": "", "data_yaml": "", "device": "", "last_status": {"ok": true, "running": false, "log_file": "runs/train_log_1775829436.txt", "device": "1", "requested_device": "auto", "command": ["/home/kly/miniconda3/bin/yolo", "train", "model=yolov8n.pt", "data=/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs=100", "device=1"], "started_at": 1775829436.981328, "yolo_executable": "/home/kly/miniconda3/bin/yolo", "device_policy": "single_idle_gpu", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "yolov8n.pt", "data_yaml": "/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 100, "device": "1", "device_policy": "single_idle_gpu"}, "reattached": false, "registry_path": "runs/active_train_job.json", "pid": 55368, "return_code": -15, "stopped_at": 1775829459.0967886, "stop_reason": "manual_stop", "last_run": {"pid": 55368, "log_file": "runs/train_log_1775829436.txt", "started_at": 1775829436.981328, "device": "1", "requested_device": "auto", "command": ["/home/kly/miniconda3/bin/yolo", "train", "model=yolov8n.pt", "data=/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs=100", "device=1"], "yolo_executable": "/home/kly/miniconda3/bin/yolo", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "yolov8n.pt", "data_yaml": "/home/kly/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 100, "device": "1", "device_policy": "single_idle_gpu"}, "running": false, "return_code": -15, "forced": false, "stop_reason": "manual_stop", "updated_at": 1775829459.0967886, "stopped_at": 1775829459.0967886}, "elapsed_seconds": 22.12, "latest_metrics": {"ok": true, "metrics": {"epoch": 1, "total_epochs": 100, "gpu_mem": "2.14G", "box_loss": "1.656", "cls_loss": "9.483", "dfl_loss": "1.349"}}, "summary": "当前无训练在跑，最近一次训练已手动停止，return_code=-15", "next_actions": ["当前无训练在跑，可直接 start_training 启动新训练"]}}`

## 3. 本轮暴露的问题

- `tool_unknown_fail_fast`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_ollama_standard_no_train`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_ollama_complex_chain`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_ollama_dirty_summary`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_ollama_health_grounded`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_ollama_duplicate_grounded`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_deepseek_dirty_summary`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_deepseek_health_grounded`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
