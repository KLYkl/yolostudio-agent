# 主线回归矩阵测试报告（2026-04-11）

- 运行批次: `20260411_015257`
- Case 数量: `17`
- 检查项通过: `64/67`
- 总体得分: `0.955`

## 1. 总结

- 未满分 case: `tool_unknown_fail_fast, agent_ollama_health_grounded, agent_deepseek_health_grounded`
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
- 得分: 4/4 (1.0)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status, training_readiness`
- 回复摘要: 可以直接训练 当前可用 YAML: /data/test_dataset/data.yaml 当前 auto 设备策略会解析到: 1 建议: - {'description': '可以直接调用 start_training 开始训练', 'tool': 'start_training', 'args_hint': {'data_yaml': '/data/test_dataset/data.yaml'}}

### agent_ollama_complex_chain
- 预期: 复杂训练意图应收敛到 prepare -> start_training 两段式流程
- 得分: 4/4 (1.0)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status, prepare_dataset_for_training`

### agent_ollama_dirty_summary
- 预期: 脏数据总结尽量 grounded，能说出缺失标签风险与真实类名/来源
- 得分: 4/4 (1.0)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status, scan_dataset, validate_dataset, run_dataset_health_check`
- 回复摘要: 未发现标签格式/坐标问题；发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响 最值得注意的风险: - 发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响 - 发现 5 个文件扩展名与真实格式不匹配 - 发现 83 组重复图片（额外重复文件 83 个） 涉及类别: Excavator, bulldozer, piling_machine, two_wheeler 类名来源: /data/agent_cap_tests/

### agent_ollama_health_grounded
- 预期: 健康检查应命中专用工具，并以 grounded 方式总结，不编造修改动作
- 得分: 3/4 (0.75)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status, run_dataset_health_check`
- 回复摘要: 健康检查完成: 完整性问题 5, 异常尺寸 0, 重复组 83 关键风险: - 发现 5 个文件扩展名与真实格式不匹配 - 发现 83 组重复图片（额外重复文件 83 个） 完整性: 损坏 0 / 零字节 0 / 格式不匹配 5 尺寸: 异常小 0 / 异常大 0 重复图片: 83 组，额外重复文件 83 个 建议: - 建议先处理损坏/异常图片，再继续数据准备或训练 - 可先人工确认重复图片是否需要清理；如需进一步查看，可单独调用 detect_duplicate_imag

### agent_ollama_duplicate_grounded
- 预期: 重复检测应命中专用工具，并基于样例路径做 grounded 总结
- 得分: 4/4 (1.0)
- Provider: `ollama` / `gemma4:e4b`
- 实际工具链: `check_training_status, detect_duplicate_images`
- 回复摘要: 检测完成: 发现 83 组重复图片，额外重复文件 83 个 示例重复组: - /data/agent_cap_tests/zyb/images/20161228230642.jpg, /data/agent_cap_tests/zyb/images/training_training_image_20161228230642.jpg - /data/agent_cap_tests/zyb/images/20161228230649.jpg, /home

### agent_ollama_state_purity
- 预期: fresh session 仅查状态时，不应被旧训练参数污染
- 得分: 4/4 (1.0)
- Provider: `ollama` / `gemma4:e4b`
- 状态摘要: `{"running": false, "model": "", "data_yaml": "", "device": "", "last_status": {"ok": true, "running": false, "log_file": "runs/train_log_1775840042.txt", "device": "1", "requested_device": "auto", "command": ["/opt/conda/bin/yolo", "train", "model=/models/yolov8n.pt", "data=/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs=30", "device=1"], "started_at": 1775840042.7257767, "yolo_executable": "/opt/conda/bin/yolo", "device_policy": "single_idle_gpu", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "/models/yolov8n.pt", "data_yaml": "/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 30, "device": "1", "device_policy": "single_idle_gpu"}, "reattached": false, "registry_path": "runs/active_train_job.json", "pid": 56927, "return_code": -15, "stopped_at": 1775840104.2528205, "stop_reason": "manual_stop", "last_run": {"pid": 56927, "log_file": "runs/train_log_1775840042.txt", "started_at": 1775840042.7257767, "device": "1", "requested_device": "auto", "command": ["/opt/conda/bin/yolo", "train", "model=/models/yolov8n.pt", "data=/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs=30", "device=1"], "yolo_executable": "/opt/conda/bin/yolo", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "/models/yolov8n.pt", "data_yaml": "/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 30, "device": "1", "device_policy": "single_idle_gpu"}, "running": false, "return_code": -15, "forced": false, "stop_reason": "manual_stop", "updated_at": 1775840104.2528205, "stopped_at": 1775840104.2528205}, "elapsed_seconds": 61.53, "latest_metrics": {"ok": true, "metrics": {"epoch": 1, "total_epochs": 30, "gpu_mem": "2.15G", "box_loss": "1.512", "cls_loss": "6.059", "dfl_loss": "1.262"}}, "summary": "当前无训练在跑，最近一次训练已手动停止，return_code=-15", "next_actions": ["当前无训练在跑，可直接 start_training 启动新训练"]}}`

### agent_deepseek_standard_no_train
- 预期: 遵守只检查不训练约束，并给出 grounded 结论
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, training_readiness`
- 回复摘要: 可以直接训练 当前可用 YAML: /data/test_dataset/data.yaml 当前 auto 设备策略会解析到: 1 建议: - {'description': '可以直接调用 start_training 开始训练', 'tool': 'start_training', 'args_hint': {'data_yaml': '/data/test_dataset/data.yaml'}}

### agent_deepseek_complex_chain
- 预期: 复杂训练意图应收敛到 prepare -> start_training 两段式流程
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, prepare_dataset_for_training`

### agent_deepseek_dirty_summary
- 预期: 脏数据总结尽量 grounded，能说出缺失标签风险与真实类名/来源
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 实际工具链: `check_training_status, scan_dataset, validate_dataset, run_dataset_health_check`
- 回复摘要: 未发现标签格式/坐标问题；发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响 最值得注意的风险: - 发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响 - 发现 5 个文件扩展名与真实格式不匹配 - 发现 83 组重复图片（额外重复文件 83 个） 涉及类别: Excavator, bulldozer, piling_machine, two_wheeler 类名来源: /data/agent_cap_tests/

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
- 回复摘要: 检测完成: 发现 83 组重复图片，额外重复文件 83 个 示例重复组: - /data/agent_cap_tests/zyb/images/20161228230642.jpg, /data/agent_cap_tests/zyb/images/training_training_image_20161228230642.jpg - /data/agent_cap_tests/zyb/images/20161228230649.jpg, /home

### agent_deepseek_state_purity
- 预期: fresh session 仅查状态时，不应被旧训练参数污染
- 得分: 4/4 (1.0)
- Provider: `deepseek` / `deepseek-chat`
- 状态摘要: `{"running": false, "model": "", "data_yaml": "", "device": "", "last_status": {"ok": true, "running": false, "log_file": "runs/train_log_1775840042.txt", "device": "1", "requested_device": "auto", "command": ["/opt/conda/bin/yolo", "train", "model=/models/yolov8n.pt", "data=/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs=30", "device=1"], "started_at": 1775840042.7257767, "yolo_executable": "/opt/conda/bin/yolo", "device_policy": "single_idle_gpu", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "/models/yolov8n.pt", "data_yaml": "/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 30, "device": "1", "device_policy": "single_idle_gpu"}, "reattached": false, "registry_path": "runs/active_train_job.json", "pid": 56927, "return_code": -15, "stopped_at": 1775840104.2528205, "stop_reason": "manual_stop", "last_run": {"pid": 56927, "log_file": "runs/train_log_1775840042.txt", "started_at": 1775840042.7257767, "device": "1", "requested_device": "auto", "command": ["/opt/conda/bin/yolo", "train", "model=/models/yolov8n.pt", "data=/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs=30", "device=1"], "yolo_executable": "/opt/conda/bin/yolo", "argument_sources": {"model": "request_or_agent_input", "data_yaml": "request_or_tool_output", "epochs": "request_or_default", "device": "auto_resolved"}, "resolved_args": {"model": "/models/yolov8n.pt", "data_yaml": "/data/agent_cap_tests/zyb/images_split/data.yaml", "epochs": 30, "device": "1", "device_policy": "single_idle_gpu"}, "running": false, "return_code": -15, "forced": false, "stop_reason": "manual_stop", "updated_at": 1775840104.2528205, "stopped_at": 1775840104.2528205}, "elapsed_seconds": 61.53, "latest_metrics": {"ok": true, "metrics": {"epoch": 1, "total_epochs": 30, "gpu_mem": "2.15G", "box_loss": "1.512", "cls_loss": "6.059", "dfl_loss": "1.262"}}, "summary": "当前无训练在跑，最近一次训练已手动停止，return_code=-15", "next_actions": ["当前无训练在跑，可直接 start_training 启动新训练"]}}`

## 3. 本轮暴露的问题

- `tool_unknown_fail_fast`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_ollama_health_grounded`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
- `agent_deepseek_health_grounded`：需要进一步查看对应 transcript / tools / assessment，判断是执行层、状态层还是解释层问题。
