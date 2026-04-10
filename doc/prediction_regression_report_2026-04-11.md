# 第二主线 Phase 1 预测回归报告（2026-04-11）

- Case 数量: `6`
- 检查项通过: `20/20`
- 总体得分: `1.0`

## 1. 总结

- 当前回归覆盖了：tool 成功/失败路径、预测意图路由、已有预测结果 grounded 总结、旧工具名兼容。
- 未满分 case: `无`

## 2. Case 结果

### tool_predict_success
- 预期: predict_images 能在目录输入下输出结构化统计和工件路径
- 得分: 5/5 (1.0)
- 摘要: {'ok': True, 'summary': '预测完成: 已处理 3 张图片, 有检测 2, 无检测 1，主要类别 bulldozer=2, Excavator=1', 'model': 'fake-model.pt', 'source_path': 'D:\\yolodo2.0\\agent_plan\\agent\\tests\\_tmp_prediction_suite\\images', 'processed_images': 3, 'detected_images': 2, 'empty_images

### tool_predict_missing_model
- 预期: 缺少模型时应明确失败，不允许静默预测
- 得分: 2/2 (1.0)
- 摘要: {'ok': False, 'error': '请提供模型路径或模型名称', 'summary': '预测未启动：缺少模型参数', 'next_actions': ['请显式提供 model，例如 /home/kly/yolov8n.pt']}

### tool_predict_missing_path
- 预期: 输入路径不存在时应 fail fast
- 得分: 2/2 (1.0)
- 摘要: {'ok': False, 'error': '路径不存在: D:\\yolodo2.0\\agent_plan\\agent\\tests\\_tmp_prediction_suite\\missing', 'summary': '预测未启动：输入路径不存在', 'next_actions': ['请确认图片文件或目录路径是否正确']}

### agent_prediction_route
- 预期: 同一句话里同时出现模型路径和图片目录时，应优先把非模型路径当 source_path
- 得分: 5/5 (1.0)
- 摘要: {'status': 'completed', 'message': '预测完成: 已处理 2 张图片, 有检测 1, 无检测 1，主要类别 Excavator=1\n统计: 已处理 2 张 / 有检测 1 / 无检测 1\n主要类别: Excavator=1\n有检测样例:\n- /data/images/a.jpg\n无检测样例:\n- /data/images/b.jpg\n标注结果目录: /tmp/predict/annotated\n预测报告: /tmp/predict/prediction_report

### agent_prediction_summary
- 预期: 已有预测结果时，应优先从 active_prediction 做 grounded 总结
- 得分: 5/5 (1.0)
- 摘要: {'status': 'completed', 'message': '预测完成: 已处理 2 张图片, 有检测 1, 无检测 1，主要类别 Excavator=1\n统计: 已处理 2 张 / 有检测 1 / 无检测 1\n主要类别: Excavator=1\n有检测样例:\n- /data/images/a.jpg\n无检测样例:\n- /data/images/b.jpg\n标注结果目录: /tmp/predict/annotated\n预测报告: /tmp/predict/prediction_report

### tool_alias_catalog
- 预期: 当前预测/数据主线相关旧工具名都应在 chat model 层注册兼容别名
- 得分: 1/1 (1.0)
- 别名工具: `batch_predict_images, dataset_manager.prepare_dataset, detect_corrupted_images, detect_duplicate_images, detect_duplicates, predict_directory, predict_images, predict_images_in_dir, prepare_dataset, prepare_dataset_for_training, run_dataset_health_check`
