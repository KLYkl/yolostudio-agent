from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.grounded_reply_builder import build_grounded_tool_reply


def main() -> None:
    health_text = build_grounded_tool_reply(
        [(
            'run_dataset_health_check',
            {
                'ok': True,
                'summary': '健康检查完成: 完整性问题 5, 异常尺寸 0, 重复组 83',
                'health_overview': {'duplicate_group_count': 83, 'duplicate_extra_files': 83},
                'warnings': ['发现 5 个文件扩展名与真实格式不匹配', '发现 83 组重复图片（额外重复文件 83 个）'],
                'integrity': {'corrupted_count': 0, 'zero_bytes_count': 0, 'format_mismatch_count': 5},
                'size_stats': {'abnormal_small_count': 0, 'abnormal_large_count': 0},
                'duplicate_groups': 83,
                'duplicate_extra_files': 83,
                'action_candidates': [
                    {'reason': '继续查看重复样本', 'action': 'review_duplicates', 'tool': 'detect_duplicate_images'},
                ],
            },
        )],
    )
    assert '健康检查完成' in health_text
    assert '格式不匹配 5' in health_text
    assert '重复图片: 83 组' in health_text
    assert '继续查看重复样本' in health_text

    scan_text = build_grounded_tool_reply(
        [(
            'scan_dataset',
            {
                'ok': True,
                'summary': '扫描完成',
                'scan_overview': {'class_count': 2},
                'warnings': ['标签缺失风险较高'],
                'action_candidates': [{'description': '继续做标签合法性校验', 'tool': 'validate_dataset'}],
            },
        )],
    )
    assert '扫描完成' in scan_text
    assert '类别数: 2' in scan_text
    assert '继续做标签合法性校验' in scan_text

    validate_text = build_grounded_tool_reply(
        [(
            'validate_dataset',
            {
                'ok': True,
                'summary': '校验完成',
                'validation_overview': {'issue_count': 3},
                'warnings': ['发现 3 个格式问题'],
                'action_candidates': [{'description': '先修复标签问题', 'tool': 'validate_dataset'}],
            },
        )],
    )
    assert '校验完成' in validate_text
    assert '问题数: 3' in validate_text
    assert '先修复标签问题' in validate_text

    readiness_text = build_grounded_tool_reply(
        [(
            'training_readiness',
            {
                'ok': True,
                'summary': '当前还不能直接训练: 缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'warnings': ['发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响'],
                'resolved_data_yaml': '/data/dirty/data.yaml',
                'device_overview': {'auto_device': '1'},
                'next_actions': ['建议先确认是否接受当前缺失标签风险，再决定是否直接训练'],
            },
        )],
    )
    assert '先准备数据' in readiness_text
    assert 'prepare_dataset_for_training' not in readiness_text
    assert '当前可用 YAML: /data/dirty/data.yaml' in readiness_text
    assert '自动设备: 1' in readiness_text
    assert '下一步: 可以先准备数据，补齐 data.yaml 和划分产物' not in readiness_text
    assert readiness_text.count('- ') <= 2

    dataset_readiness_text = build_grounded_tool_reply(
        [(
            'dataset_training_readiness',
            {
                'ok': True,
                'summary': '从数据集角度看，这份数据还不能直接训练: 缺少可用的 data.yaml, 训练/验证集还没准备好；但可以先准备数据',
                'blockers': ['缺少可用的 data.yaml', '训练/验证集还没准备好'],
                'next_step_summary': '可以先准备数据，补齐 data.yaml 和划分产物。',
            },
        )],
    )
    assert '从数据集角度看，这份数据还不能直接训练' in dataset_readiness_text
    assert '当前问题:' in dataset_readiness_text
    assert '训练/验证集还没准备好' in dataset_readiness_text
    assert '下一步: 可以先准备数据，补齐 data.yaml 和划分产物。' in dataset_readiness_text
    assert 'GPU' not in dataset_readiness_text
    assert 'prepare_dataset_for_training' not in dataset_readiness_text

    status_text = build_grounded_tool_reply(
        [(
            'check_training_status',
            {
                'ok': True,
                'summary': '当前无训练在跑，最近一次训练已完成，return_code=0，已有可分析指标',
                'status_overview': {
                    'run_state': 'completed',
                    'save_dir': '/tmp/demo',
                    'observation_stage': 'final',
                    'epoch': 4,
                    'total_epochs': 4,
                },
                'progress': {'epoch': 4, 'total_epochs': 4, 'progress_ratio': 1.0},
                'latest_metrics': {
                    'metrics': {
                        'precision': 0.1,
                        'recall': 0.2,
                        'map50': 0.3,
                        'map': 0.4,
                    }
                },
                'facts': ['训练进度 4/4', 'return_code=0'],
                'action_candidates': [
                    {'description': '可继续调用 summarize_training_run 汇总训练结果', 'tool': 'summarize_training_run'},
                ],
            },
        )],
    )
    assert '运行状态: completed' in status_text
    assert '最近指标: precision=0.100, recall=0.200, mAP50=0.300, mAP50-95=0.400' in status_text
    assert '下一步: 可继续调用 summarize_training_run 汇总训练结果' in status_text
    assert '事实:' not in status_text

    running_loss_only_status = build_grounded_tool_reply(
        [(
            'check_training_status',
            {
                'ok': True,
                'summary': '训练进行中 (device=1, pid=1001，epoch 1/100，当前仍属早期观察)',
                'run_state': 'running',
                'observation_stage': 'early',
                'progress': {'epoch': 1, 'total_epochs': 100, 'progress_ratio': 0.01},
                'latest_train_metrics': {
                    'gpu_mem': '2.17G',
                    'box_loss': 2.62,
                    'cls_loss': 4.852,
                    'dfl_loss': 2.334,
                },
                'latest_metrics': {
                    'metrics': {
                        'epoch': 1,
                        'total_epochs': 100,
                        'gpu_mem': '2.17G',
                        'box_loss': 2.62,
                        'cls_loss': 4.852,
                        'dfl_loss': 2.334,
                    },
                },
                'analysis_ready': False,
                'minimum_facts_ready': True,
                'signals': ['loss_only_metrics', 'missing_eval_metrics'],
                'action_candidates': [
                    {'description': '可继续调用 check_training_status 观察训练进度', 'tool': 'check_training_status'},
                ],
            },
        )],
    )
    assert 'GPU 显存: 2.17G' in running_loss_only_status
    assert '最近评估指标: 暂无（等待验证阶段产出）' in running_loss_only_status
    assert '当前仅有训练损失: box=2.62, cls=4.852, dfl=2.334' in running_loss_only_status
    assert '当前不足: 缺少稳定评估指标' in running_loss_only_status

    prepare_text = build_grounded_tool_reply(
        [(
            'prepare_dataset_for_training',
            {
                'ok': True,
                'summary': '数据集已准备到可训练状态，但存在数据质量风险',
                'prepare_overview': {'data_yaml': '/data/dirty/images_split/data.yaml'},
                'warnings': ['发现 5179 张图片缺少标签（占比 73.7%），训练结果可能受到明显影响'],
                'action_candidates': [{'description': '如要训练，可直接复用上面的 data_yaml 调用 start_training', 'tool': 'start_training'}],
            },
        )],
    )
    assert '数据集已准备到可训练状态' in prepare_text
    assert '已准备好的 YAML: /data/dirty/images_split/data.yaml' in prepare_text
    assert '如要训练，可直接复用上面的 data_yaml' in prepare_text

    env_text = build_grounded_tool_reply(
        [(
            'list_training_environments',
            {
                'ok': True,
                'summary': '训练环境查询完成',
                'environment_overview': {
                    'environment_count': 2,
                    'default_environment_name': 'yolodo',
                },
                'action_candidates': [
                    {'description': '继续做训练预检', 'tool': 'training_preflight'},
                ],
            },
        )],
    )
    assert '默认训练环境: yolodo' in env_text
    assert '环境数量: 2' in env_text
    assert '继续做训练预检' in env_text

    preflight_text = build_grounded_tool_reply(
        [(
            'training_preflight',
            {
                'ok': True,
                'summary': '训练预检完成',
                'preflight_overview': {
                    'training_environment_name': 'yolodo',
                    'model': 'yolov8n.pt',
                    'data_yaml': '/data/dirty/images_split/data.yaml',
                    'batch': 16,
                    'imgsz': 640,
                },
                'action_candidates': [
                    {'description': '现在可以开始训练', 'tool': 'start_training'},
                ],
            },
        )],
    )
    assert '训练环境: yolodo' in preflight_text
    assert '模型: yolov8n.pt' in preflight_text
    assert '数据 YAML: /data/dirty/images_split/data.yaml' in preflight_text
    assert '批大小: 16' in preflight_text
    assert '输入尺寸: 640' in preflight_text
    assert '现在可以开始训练' in preflight_text

    predict_text = build_grounded_tool_reply(
        [(
            'predict_images',
            {
                'ok': True,
                'summary': '预测完成: 已处理 3 张图片, 有检测 2, 无检测 1，主要类别 Excavator=1, bulldozer=2',
                'prediction_overview': {
                    'mode': 'images',
                    'processed_images': 3,
                    'detected_images': 2,
                    'empty_images': 1,
                    'annotated_dir': '/data/predict/out/annotated',
                    'report_path': '/data/predict/out/prediction_report.json',
                },
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/predict/a.jpg', '/data/predict/c.jpg'],
                'empty_samples': ['/data/predict/b.jpg'],
                'action_candidates': [{'description': '可查看标注结果目录: /data/predict/out/annotated', 'tool': 'inspect_prediction_outputs'}],
            },
        )],
    )
    assert '预测完成' in predict_text
    assert '已处理 3 张' in predict_text
    assert '主要类别: Excavator=1，bulldozer=2' in predict_text
    assert '/data/predict/a.jpg' in predict_text
    assert '标注结果目录: /data/predict/out/annotated' in predict_text
    assert '预测报告: /data/predict/out/prediction_report.json' in predict_text

    predict_video_text = build_grounded_tool_reply(
        [(
            'predict_videos',
            {
                'ok': True,
                'summary': '视频预测完成: 已处理 1 个视频, 总帧数 3, 有检测帧 2, 总检测框 3，主要类别 Excavator=1, bulldozer=2',
                'prediction_overview': {
                    'mode': 'videos',
                    'processed_videos': 1,
                    'total_frames': 3,
                    'detected_frames': 2,
                    'total_detections': 3,
                    'output_dir': '/data/videos/out',
                    'report_path': '/data/videos/out/video_prediction_report.json',
                },
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/videos/a.mp4'],
                'empty_samples': [],
                'action_candidates': [{'description': '可查看视频预测输出目录: /data/videos/out', 'tool': 'inspect_prediction_outputs'}],
            },
        )],
    )
    assert '视频预测完成' in predict_video_text
    assert '总帧数 3' in predict_video_text
    assert '主要类别: Excavator=1，bulldozer=2' in predict_video_text
    assert '视频预测输出目录: /data/videos/out' in predict_video_text

    summary_text = build_grounded_tool_reply(
        [(
            'summarize_prediction_results',
            {
                'ok': True,
                'summary': '预测结果摘要: 已处理 3 张图片, 有检测 2, 无检测 1, 总检测框 3，主要类别 Excavator=1, bulldozer=2',
                'prediction_summary_overview': {
                    'mode': 'images',
                    'processed_images': 3,
                    'detected_images': 2,
                    'empty_images': 1,
                    'total_detections': 3,
                    'annotated_dir': '/data/predict/out/annotated',
                    'report_path': '/data/predict/out/prediction_report.json',
                },
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/predict/a.jpg', '/data/predict/c.jpg'],
                'empty_samples': ['/data/predict/b.jpg'],
                'action_candidates': [{'description': '可查看标注结果目录: /data/predict/out/annotated', 'tool': 'inspect_prediction_outputs'}],
            },
        )],
    )
    assert '预测结果摘要' in summary_text
    assert '总检测框 3' in summary_text
    assert '主要类别: Excavator=1，bulldozer=2' in summary_text
    assert '预测报告: /data/predict/out/prediction_report.json' in summary_text

    video_summary_text = build_grounded_tool_reply(
        [(
            'summarize_prediction_results',
            {
                'ok': True,
                'summary': '视频预测结果摘要: 已处理 1 个视频, 总帧数 3, 有检测帧 2, 总检测框 3，主要类别 Excavator=1, bulldozer=2',
                'prediction_summary_overview': {
                    'mode': 'videos',
                    'processed_videos': 1,
                    'total_frames': 3,
                    'detected_frames': 2,
                    'total_detections': 3,
                    'report_path': '/data/videos/out/video_prediction_report.json',
                },
                'class_counts': {'Excavator': 1, 'bulldozer': 2},
                'detected_samples': ['/data/videos/a.mp4'],
                'empty_samples': [],
                'action_candidates': [{'description': '可查看预测报告: /data/videos/out/video_prediction_report.json', 'tool': 'export_prediction_report'}],
            },
        )],
    )
    assert '视频预测结果摘要' in video_summary_text
    assert '有检测帧 2' in video_summary_text
    assert '预测报告: /data/videos/out/video_prediction_report.json' in video_summary_text

    remote_predict_text = build_grounded_tool_reply(
        [(
            'remote_prediction_pipeline',
            {
                'ok': True,
                'summary': '远端预测闭环完成',
                'pipeline_overview': {
                    'target_label': 'yolostudio',
                    'remote_root': '/tmp/remote_predict',
                    'remote_output_dir': '/tmp/remote_predict/out',
                    'local_result_root': '/local/predict_out',
                    'source_kind': 'images',
                },
                'execution_overview': {
                    'upload_ok': True,
                    'predict_ok': True,
                    'download_ok': True,
                    'predict_tool_name': 'predict_images',
                },
                'action_candidates': [
                    {'description': '可继续查看本机结果目录: /local/predict_out', 'tool': 'inspect_prediction_outputs'},
                ],
            },
        )],
    )
    assert '目标服务器: yolostudio' in remote_predict_text
    assert '远端预测目录: /tmp/remote_predict/out' in remote_predict_text
    assert '本机回传目录: /local/predict_out' in remote_predict_text
    assert '预测执行: predict_images' in remote_predict_text
    assert '可继续查看本机结果目录: /local/predict_out' in remote_predict_text

    remote_train_text = build_grounded_tool_reply(
        [(
            'remote_training_pipeline',
            {
                'ok': True,
                'summary': '远端训练闭环完成',
                'pipeline_overview': {
                    'target_label': 'yolostudio',
                    'remote_root': '/tmp/remote_train',
                    'remote_dataset_path': '/tmp/remote_train/dataset',
                    'remote_result_path': '/tmp/remote_train/runs/data-yolov8n',
                    'local_result_root': '/local/train_out',
                },
                'execution_overview': {
                    'upload_ok': True,
                    'readiness_ok': True,
                    'preflight_ok': True,
                    'start_ok': True,
                    'download_ok': True,
                    'final_run_state': 'completed',
                },
                'action_candidates': [
                    {'description': '可继续查看训练总结或下一步建议', 'tool': 'summarize_training_run'},
                ],
            },
        )],
    )
    assert '目标服务器: yolostudio' in remote_train_text
    assert '远端训练目录: /tmp/remote_train/runs/data-yolov8n' in remote_train_text
    assert '本机回传目录: /local/train_out' in remote_train_text
    assert '最终状态: completed' in remote_train_text
    assert '可继续查看训练总结或下一步建议' in remote_train_text

    preview_extract_text = build_grounded_tool_reply(
        [(
            'preview_extract_images',
            {
                'ok': True,
                'summary': '图片抽取预览完成',
                'extract_preview_overview': {
                    'available_images': 10,
                    'planned_extract_count': 3,
                    'workflow_ready_path': '/data/extract/preview',
                },
                'action_candidates': [{'description': '继续执行图片抽取', 'tool': 'extract_images'}],
            },
        )],
    )
    assert '可用 10 张 / 计划抽取 3 张' in preview_extract_text
    assert '后续可复用目录: /data/extract/preview' in preview_extract_text
    assert '继续执行图片抽取' in preview_extract_text

    extract_images_text = build_grounded_tool_reply(
        [(
            'extract_images',
            {
                'ok': True,
                'summary': '图片抽取完成',
                'extract_overview': {
                    'extracted': 3,
                    'labels_copied': 3,
                    'conflict_count': 0,
                    'output_dir': '/data/extract/out',
                    'workflow_ready_path': '/data/extract/out',
                },
                'action_candidates': [{'description': '继续做 scan_dataset', 'tool': 'scan_dataset'}],
            },
        )],
    )
    assert '已抽取 3 张 / 复制标签 3 / 冲突 0' in extract_images_text
    assert '输出目录: /data/extract/out' in extract_images_text
    assert '可继续接主链的目录: /data/extract/out' in extract_images_text
    assert '继续做 scan_dataset' in extract_images_text

    remote_profiles_text = build_grounded_tool_reply(
        [(
            'list_remote_profiles',
            {
                'ok': True,
                'summary': '远端 profile 1 个 / SSH alias 1 个。 默认 profile: lab。',
                'profile_overview': {
                    'profile_count': 1,
                    'ssh_alias_count': 1,
                    'default_profile': 'lab',
                    'profiles_path': '/tmp/profiles.json',
                },
                'profiles': [{'name': 'lab', 'remote_root': '/srv/agent_stage', 'is_default': True}],
                'ssh_aliases': [{'name': 'alias-demo', 'hostname': 'demo-host', 'port': '22'}],
                'action_candidates': [{'description': '继续上传本地产物到远端', 'tool': 'upload_assets_to_remote'}],
            },
        )],
    )
    assert 'lab (默认) / /srv/agent_stage' in remote_profiles_text
    assert 'alias-demo -> demo-host:22' in remote_profiles_text
    assert '配置文件: /tmp/profiles.json' in remote_profiles_text
    assert '继续上传本地产物到远端' in remote_profiles_text

    camera_text = build_grounded_tool_reply(
        [(
            'scan_cameras',
            {
                'ok': True,
                'summary': '摄像头扫描完成',
                'cameras': [{'id': 0, 'name': 'USB Camera'}],
                'action_candidates': [
                    {'description': '直接启动摄像头实时预测', 'tool': 'start_camera_prediction'},
                ],
            },
        )],
    )
    assert 'id=0 / USB Camera' in camera_text
    assert '直接启动摄像头实时预测' in camera_text

    realtime_status_text = build_grounded_tool_reply(
        [(
            'check_realtime_prediction_status',
            {
                'ok': True,
                'summary': '实时预测状态已更新',
                'session_id': 'rt-1',
                'source_type': 'camera',
                'status': 'running',
                'processed_frames': 12,
                'detected_frames': 5,
                'total_detections': 8,
                'class_counts': {'helmet': 6, 'person': 2},
                'action_candidates': [
                    {'description': '继续查看实时预测详细输出', 'tool': 'inspect_prediction_outputs'},
                ],
            },
        )],
    )
    assert '会话 ID: rt-1' in realtime_status_text
    assert '主要类别: helmet=6，person=2' in realtime_status_text
    assert '继续查看实时预测详细输出' in realtime_status_text

    upload_text = build_grounded_tool_reply(
        [(
            'upload_assets_to_remote',
            {
                'ok': True,
                'summary': '远端上传完成',
                'transfer_overview': {
                    'target_label': 'lab',
                    'profile_name': 'lab',
                    'remote_root': '/srv/agent_stage',
                    'file_count': 2,
                    'verified_file_count': 2,
                    'skipped_file_count': 1,
                    'transferred_bytes': 12,
                    'skipped_bytes': 3,
                    'total_bytes': 15,
                },
                'transfer_strategy_summary': 'SCP 直传 2 个，哈希校验(sha256)',
                'file_results_preview': [{'relative_path': 'best.pt', 'mode': 'scp', 'size_bytes': 12}],
                'action_candidates': [{'description': '继续在远端目录复用这些产物', 'tool': 'download_assets_from_remote'}],
            },
        )],
    )
    assert '远端 profile: lab' in upload_text
    assert '远端目录: /srv/agent_stage' in upload_text
    assert '文件统计: 总计 2 / 已校验 2 / 复用 1' in upload_text
    assert '继续在远端目录复用这些产物' in upload_text

    list_runs_text = build_grounded_tool_reply(
        [(
            'list_training_runs',
            {
                'ok': True,
                'summary': '训练历史查询完成',
                'runs': [
                    {
                        'run_id': 'run-a',
                        'run_state': 'completed',
                        'observation_stage': 'final',
                        'progress': {'epoch': 10, 'total_epochs': 10},
                    }
                ],
                'action_candidates': [
                    {'description': '查看最佳训练记录', 'tool': 'select_best_training_run'},
                ],
            },
        )],
    )
    assert 'run-a: completed / 最终状态，进度 10/10' in list_runs_text
    assert '查看最佳训练记录' in list_runs_text

    inspect_run_text = build_grounded_tool_reply(
        [(
            'inspect_training_run',
            {
                'ok': True,
                'summary': '训练记录详情已就绪',
                'selected_run_id': 'run-a',
                'run_state': 'completed',
                'observation_stage': 'final',
                'progress': {'epoch': 10, 'total_epochs': 10, 'progress_ratio': 1.0},
                'action_candidates': [
                    {'description': '继续生成训练结果分析', 'tool': 'analyze_training_outcome'},
                ],
            },
        )],
    )
    assert '训练记录: run-a' in inspect_run_text
    assert '继续生成训练结果分析' in inspect_run_text

    compare_runs_text = build_grounded_tool_reply(
        [(
            'compare_training_runs',
            {
                'ok': True,
                'summary': '训练记录对比已完成',
                'left_run_id': 'run-a',
                'right_run_id': 'run-b',
                'highlights': ['mAP50 提升 +0.0300'],
                'action_candidates': [
                    {'description': '继续查看更优训练记录', 'tool': 'select_best_training_run'},
                ],
            },
        )],
    )
    assert '对比对象: run-a vs run-b' in compare_runs_text
    assert '继续查看更优训练记录' in compare_runs_text

    run_summary_text = build_grounded_tool_reply(
        [(
            'summarize_training_run',
            {
                'ok': True,
                'summary': '训练结果汇总完成',
                'summary_overview': {
                    'run_state': 'completed',
                    'save_dir': '/tmp/demo',
                    'observation_stage': 'final',
                    'epoch': 10,
                    'total_epochs': 10,
                },
                'action_candidates': [
                    {'description': '分析训练结果', 'tool': 'analyze_training_outcome'},
                ],
            },
        )],
    )
    assert '运行状态: completed' in run_summary_text
    assert '结果目录: /tmp/demo' in run_summary_text
    assert '训练进度: 10/10' in run_summary_text
    assert '分析训练结果' in run_summary_text

    knowledge_text = build_grounded_tool_reply(
        [(
            'retrieve_training_knowledge',
            {
                'ok': True,
                'summary': '知识检索完成',
                'matched_rule_overview': [{'id': 'rule-a', 'interpretation': '优先补齐标签'}],
                'playbook_overview': [{'title': 'playbook-a', 'path': '/docs/playbook-a.md'}],
                'retrieval_overview': {'stage': 'training', 'signal_count': 2},
                'action_candidates': [
                    {'description': '继续分析训练结果', 'tool': 'analyze_training_outcome'},
                ],
            },
        )],
    )
    assert 'rule-a: 优先补齐标签' in knowledge_text
    assert '来源: stage=training，signal_count=2' in knowledge_text
    assert 'playbook-a: /docs/playbook-a.md' in knowledge_text
    assert '继续分析训练结果' in knowledge_text

    recommendation_text = build_grounded_tool_reply(
        [(
            'recommend_next_training_step',
            {
                'ok': True,
                'summary': '下一步建议生成完成',
                'recommendation_overview': {
                    'recommended_action': 'fix_data_quality',
                    'confidence': 'high',
                },
                'action_candidates': [
                    {'description': '先补齐缺失标签', 'tool': 'prepare_dataset_for_training'},
                ],
            },
        )],
    )
    assert '优先动作: fix_data_quality' in recommendation_text
    assert '来源: recommended_action=fix_data_quality，confidence=high' in recommendation_text
    assert '先补齐缺失标签' in recommendation_text

    dup_text = build_grounded_tool_reply(
        [(
            'detect_duplicate_images',
            {
                'ok': True,
                'summary': '检测完成: 发现 2 组重复图片，额外重复文件 2 个',
                'groups': [
                    {
                        'paths': ['/data/a.jpg', '/data/a_copy.jpg'],
                    },
                    {
                        'paths': ['/data/b.jpg', '/data/b_copy.jpg'],
                    },
                ],
                'next_actions': ['建议人工确认 sample groups 中的文件是否应合并或清理'],
            },
        )],
    )
    assert '检测完成: 发现 2 组重复图片' in dup_text
    assert '/data/a.jpg, /data/a_copy.jpg' in dup_text
    assert '建议人工确认' in dup_text

    loop_start_text = build_grounded_tool_reply(
        [(
            'start_training_loop',
            {
                'ok': True,
                'summary': '环训练已启动：helmet-loop（loop_id=loop-123）',
                'loop_id': 'loop-123',
                'loop_name': 'helmet-loop',
                'managed_level': 'full_auto',
                'boundaries': {'max_rounds': 5, 'target_metric': 'map50', 'target_metric_value': 0.8},
                'next_round_plan': {'round_index': 1},
            },
        )],
    )
    assert '环训练已启动' in loop_start_text
    assert 'Loop ID: loop-123' in loop_start_text
    assert '托管级别: full_auto' in loop_start_text

    loop_status_text = build_grounded_tool_reply(
        [(
            'check_training_loop_status',
            {
                'ok': True,
                'summary': '第 2 轮训练已完成，准备下一轮',
                'loop_name': 'helmet-loop',
                'status': 'awaiting_review',
                'current_round_index': 2,
                'max_rounds': 5,
                'best_round_index': 2,
                'best_target_metric': 0.71,
                'knowledge_gate_status': {
                    'outcome': 'awaiting_review',
                    'action_label': '先做误差分析',
                    'summary': '本轮建议“先做误差分析”，已停在轮间闸门等待审阅。',
                    'matched_rule_ids': ['generic_next_low_map_error_analysis'],
                },
                'latest_round_card': {
                    'round_index': 2,
                    'status': 'completed',
                    'knowledge_gate': {
                        'action': 'run_error_analysis',
                        'action_label': '先做误差分析',
                        'category': 'analysis_review',
                        'outcome': 'awaiting_review',
                        'outcome_label': '等待审阅',
                        'decision_type': 'await_review',
                        'decision_reason': '知识策略建议先人工分析，已停在轮间闸门等待确认',
                        'matched_rule_ids': ['generic_next_low_map_error_analysis'],
                        'user_summary': '本轮建议“先做误差分析”，已停在轮间闸门等待审阅。',
                    },
                    'vs_previous': {'highlights': ['mAP50提升 +0.0300']},
                    'next_plan': {'change_set': [{'field': 'epochs', 'old': 50, 'new': 60}]},
                },
                'final_summary': {
                    'termination_detail': '连续 2 轮提升不足，已停止自动续跑',
                    'knowledge_gate_overview': {
                        'count': 2,
                        'last_outcome': 'awaiting_review',
                        'last_outcome_label': '等待审阅',
                        'last_summary': '本轮建议“先做误差分析”，已停在轮间闸门等待审阅。',
                    },
                    'last_knowledge_gate': {
                        'round_index': 2,
                        'action': 'run_error_analysis',
                        'action_label': '先做误差分析',
                        'category': 'analysis_review',
                        'outcome': 'awaiting_review',
                        'outcome_label': '等待审阅',
                    },
                },
            },
        )],
    )
    assert 'helmet-loop' in loop_status_text
    assert '当前最佳轮: 第 2 轮，指标 0.7100' in loop_status_text
    assert '闸门结论: 等待审阅 / 先做误差分析' in loop_status_text
    assert '结论说明: 本轮建议“先做误差分析”，已停在轮间闸门等待审阅。' in loop_status_text
    assert '命中规则: generic_next_low_map_error_analysis' in loop_status_text
    assert 'mAP50提升 +0.0300' in loop_status_text
    assert 'epochs: 50 -> 60' in loop_status_text
    assert '闸门总览: 共 2 次 / 最后一次为 等待审阅' in loop_status_text
    assert '最后知识结论: 第 2 轮 / 等待审阅 / 先做误差分析' in loop_status_text

    print('grounded tool reply ok')


if __name__ == '__main__':
    main()
