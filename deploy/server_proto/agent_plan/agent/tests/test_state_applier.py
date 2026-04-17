from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_applier import (
    CRITICAL_STATEFUL_TOOLS,
    STATEFUL_TOOL_PROJECTORS,
    apply_tool_result_to_state,
)


def main() -> None:
    state = SessionState(session_id='state-applier-smoke')
    assert CRITICAL_STATEFUL_TOOLS.issubset(set(STATEFUL_TOOL_PROJECTORS)), STATEFUL_TOOL_PROJECTORS.keys()

    apply_tool_result_to_state(
        state,
        'scan_dataset',
        {
            'ok': True,
            'summary': '数据集扫描完成',
            'dataset_root': '/tmp/dataset',
            'resolved_img_dir': '/tmp/dataset/images',
            'resolved_label_dir': '/tmp/dataset/labels',
            'detected_data_yaml': '/tmp/dataset/data.yaml',
            'detected_classes_txt': '/tmp/dataset/classes.txt',
            'total_images': 120,
            'labeled_images': 118,
            'missing_labels': 2,
            'missing_label_images': 2,
            'missing_label_ratio': 0.0167,
            'empty_labels': 1,
            'risk_level': 'low',
            'scan_overview': {'image_count': 120, 'class_count': 3},
            'warnings': ['发现 2 张图片缺少标签（占比 1.7%），训练结果可能受到明显影响'],
            'classes': ['car', 'bus', 'truck'],
            'class_stats': {'car': 52, 'bus': 41, 'truck': 25},
            'top_classes': [{'class_name': 'car', 'count': 52}, {'class_name': 'bus', 'count': 41}],
            'class_name_source': 'classes_txt',
            'data_yaml_candidates': ['/tmp/dataset/data.yaml'],
            'classes_txt_candidates': ['/tmp/dataset/classes.txt'],
            'next_actions': ['可继续 validate_dataset 做标签合法性校验'],
            'action_candidates': [{'tool': 'run_dataset_health_check', 'description': '继续做健康检查'}],
        },
    )
    assert state.active_dataset.last_scan['scan_overview']['class_count'] == 3
    assert state.active_dataset.last_scan['action_candidates'][0]['tool'] == 'run_dataset_health_check'
    assert state.active_dataset.last_scan['detected_data_yaml'] == '/tmp/dataset/data.yaml'
    assert state.active_dataset.last_scan['detected_classes_txt'] == '/tmp/dataset/classes.txt'
    assert state.active_dataset.last_scan['class_stats']['truck'] == 25
    assert state.active_dataset.last_scan['least_class'] == {'name': 'truck', 'count': 25}
    assert state.active_dataset.last_scan['most_class'] == {'name': 'car', 'count': 52}
    assert state.active_dataset.last_scan['missing_label_ratio'] == 0.0167

    apply_tool_result_to_state(
        state,
        'run_dataset_health_check',
        {
            'ok': True,
            'summary': '数据集健康检查完成',
            'dataset_root': '/tmp/dataset',
            'resolved_img_dir': '/tmp/dataset/images',
            'risk_level': 'medium',
            'issue_count': 4,
            'duplicate_groups': 2,
            'health_overview': {'duplicate_groups': 2, 'corrupt_images': 0},
            'action_candidates': [{'tool': 'detect_duplicate_images', 'description': '查看重复图片详情'}],
        },
    )
    assert state.active_dataset.last_health_check['health_overview']['duplicate_groups'] == 2
    assert state.active_dataset.last_health_check['action_candidates'][0]['tool'] == 'detect_duplicate_images'

    apply_tool_result_to_state(
        state,
        'preview_extract_images',
        {
            'ok': True,
            'summary': '图片抽取预览完成',
            'planned_extract_count': 12,
            'output_dir': '/tmp/extract_preview',
            'extract_preview_overview': {'planned_extract_count': 12},
            'action_candidates': [{'tool': 'extract_images', 'description': '执行抽取'}],
        },
        {'source_path': '/data/src_images'},
    )
    assert state.active_dataset.last_extract_preview['source_path'] == '/data/src_images'
    assert state.active_dataset.last_extract_preview['extract_preview_overview']['planned_extract_count'] == 12

    apply_tool_result_to_state(
        state,
        'extract_images',
        {
            'ok': True,
            'summary': '图片抽取完成',
            'extracted': 8,
            'labels_copied': 8,
            'output_dir': '/tmp/extract/out',
            'workflow_ready_path': '/tmp/extract',
            'output_img_dir': '/tmp/extract/images',
            'output_label_dir': '/tmp/extract/labels',
            'extract_overview': {'extracted': 8, 'workflow_ready': True},
            'action_candidates': [{'tool': 'scan_dataset', 'description': '继续做数据扫描'}],
        },
        {'source_path': '/data/src'},
    )
    assert state.active_dataset.dataset_root == '/tmp/extract'
    assert state.active_dataset.img_dir == '/tmp/extract/images'
    assert state.active_dataset.label_dir == '/tmp/extract/labels'
    assert state.active_dataset.last_extract_result['extracted'] == 8
    assert state.active_dataset.last_extract_result['extract_overview']['workflow_ready'] is True
    assert state.active_dataset.last_extract_result['action_candidates'][0]['tool'] == 'scan_dataset'

    apply_tool_result_to_state(
        state,
        'scan_videos',
        {
            'ok': True,
            'summary': '视频扫描完成',
            'total_videos': 3,
            'video_scan_overview': {'total_videos': 3},
            'action_candidates': [{'tool': 'extract_video_frames', 'description': '继续抽帧'}],
        },
        {'source_path': '/data/videos'},
    )
    assert state.active_dataset.last_video_scan['source_path'] == '/data/videos'
    assert state.active_dataset.last_video_scan['video_scan_overview']['total_videos'] == 3

    apply_tool_result_to_state(
        state,
        'start_image_prediction',
        {
            'ok': True,
            'summary': '图片目录较大，已转为后台预测会话：共 320 张图片（session_id=image-predict-1234abcd）',
            'session_id': 'image-predict-1234abcd',
            'status': 'running',
            'run_state': 'running',
            'running': True,
            'started_in_background': True,
            'source_path': '/data/images-large',
            'model': '/models/large.pt',
            'total_images': 320,
            'processed_images': 0,
            'detected_images': 0,
            'empty_images': 0,
            'output_dir': '/tmp/predict-large',
            'report_path': '',
            'prediction_session_overview': {'session_id': 'image-predict-1234abcd', 'status': 'running', 'total_images': 320},
            'action_candidates': [{'tool': 'check_image_prediction_status', 'description': '查看后台进度'}],
        },
        {'source_path': '/data/images-large', 'model': '/models/large.pt'},
    )
    assert state.active_prediction.image_prediction_session_id == 'image-predict-1234abcd'
    assert state.active_prediction.image_prediction_status == 'running'
    assert state.active_prediction.last_image_prediction_status['prediction_session_overview']['total_images'] == 320

    apply_tool_result_to_state(
        state,
        'check_image_prediction_status',
        {
            'ok': True,
            'summary': '后台图片预测运行中: 已处理 64/320 张图片, 有检测 40, 无检测 24',
            'session_id': 'image-predict-1234abcd',
            'status': 'running',
            'run_state': 'running',
            'running': True,
            'source_path': '/data/images-large',
            'model': '/models/large.pt',
            'total_images': 320,
            'processed_images': 64,
            'detected_images': 40,
            'empty_images': 24,
            'total_detections': 88,
            'class_counts': {'excavator': 60, 'truck': 28},
            'output_dir': '/tmp/predict-large',
            'report_path': '',
            'prediction_session_overview': {'session_id': 'image-predict-1234abcd', 'status': 'running', 'processed_images': 64},
            'action_candidates': [{'tool': 'stop_image_prediction', 'description': '停止后台预测'}],
        },
    )
    assert state.active_prediction.last_image_prediction_status['processed_images'] == 64
    assert state.active_prediction.last_image_prediction_status['class_counts']['excavator'] == 60

    apply_tool_result_to_state(
        state,
        'predict_videos',
        {
            'ok': True,
            'summary': '视频预测完成',
            'source_path': '/data/videos',
            'model': '/models/a.pt',
            'output_dir': '/tmp/predict_videos',
            'report_path': '/tmp/predict_videos/video_prediction_report.json',
            'processed_videos': 2,
            'total_frames': 12,
            'detected_frames': 3,
            'total_detections': 4,
            'class_counts': {'bulldozer': 4},
            'warnings': [],
            'detected_samples': ['/data/videos/a.mp4'],
            'empty_samples': ['/data/videos/b.mp4'],
        },
    )
    assert state.active_prediction.source_path == '/data/videos'
    assert state.active_prediction.model == '/models/a.pt'
    assert state.active_prediction.report_path.endswith('video_prediction_report.json')
    assert state.active_prediction.last_result['mode'] == 'videos'

    apply_tool_result_to_state(
        state,
        'stop_image_prediction',
        {
            'ok': True,
            'summary': '后台图片预测已停止: 已处理 120/320 张图片, 有检测 75, 无检测 45',
            'session_id': 'image-predict-1234abcd',
            'status': 'stopped',
            'run_state': 'stopped',
            'running': False,
            'source_path': '/data/images-large',
            'model': '/models/large.pt',
            'total_images': 320,
            'processed_images': 120,
            'detected_images': 75,
            'empty_images': 45,
            'total_detections': 160,
            'class_counts': {'excavator': 100, 'truck': 60},
            'warnings': ['预测已在后台会话中被手动停止，结果为部分产物'],
            'output_dir': '/tmp/predict-large',
            'report_path': '/tmp/predict-large/prediction_report.json',
            'prediction_session_overview': {'session_id': 'image-predict-1234abcd', 'status': 'stopped', 'processed_images': 120},
            'action_candidates': [{'tool': 'inspect_prediction_outputs', 'description': '查看预测产物'}],
        },
    )
    assert state.active_prediction.image_prediction_status == 'stopped'
    assert state.active_prediction.last_image_prediction_status['report_path'] == '/tmp/predict-large/prediction_report.json'
    assert state.active_prediction.last_result['processed_images'] == 120
    assert state.active_prediction.last_result['mode'] == 'images'

    apply_tool_result_to_state(
        state,
        'inspect_prediction_outputs',
        {
            'ok': True,
            'summary': '预测输出检查完成',
            'mode': 'images',
            'prediction_output_overview': {'artifact_root_count': 3},
            'action_candidates': [{'tool': 'export_prediction_report', 'description': '导出报告'}],
            'output_dir': '/tmp/predict-out',
            'report_path': '/tmp/predict-out/prediction_report.json',
            'artifact_roots': ['/tmp/predict-out', '/tmp/predict-out/annotated', '/tmp/predict-out/labels_yolo'],
            'path_list_files': {},
        },
    )
    assert state.active_prediction.last_inspection['prediction_output_overview']['artifact_root_count'] == 3
    assert state.active_prediction.last_inspection['action_candidates'][0]['tool'] == 'export_prediction_report'

    apply_tool_result_to_state(
        state,
        'export_prediction_report',
        {
            'ok': True,
            'summary': '预测报告导出完成',
            'mode': 'images',
            'export_overview': {'export_format': 'markdown'},
            'action_candidates': [{'tool': 'inspect_prediction_outputs', 'description': '继续查看输出'}],
            'output_dir': '/tmp/predict-out',
            'report_path': '/tmp/predict-out/prediction_report.json',
            'export_path': '/tmp/predict-out/prediction_summary.md',
            'export_format': 'markdown',
        },
    )
    assert state.active_prediction.last_export['export_overview']['export_format'] == 'markdown'
    assert state.active_prediction.last_export['action_candidates'][0]['tool'] == 'inspect_prediction_outputs'

    apply_tool_result_to_state(
        state,
        'export_prediction_path_lists',
        {
            'ok': True,
            'summary': '预测路径清单导出完成',
            'mode': 'images',
            'path_list_overview': {'detected_count': 2, 'empty_count': 1, 'failed_count': 0},
            'action_candidates': [{'tool': 'organize_prediction_results', 'description': '继续整理结果'}],
            'output_dir': '/tmp/predict-out',
            'report_path': '/tmp/predict-out/prediction_report.json',
            'export_dir': '/tmp/predict-out/path_lists',
            'detected_items_path': '/tmp/predict-out/path_lists/detected_items.txt',
            'empty_items_path': '/tmp/predict-out/path_lists/empty_items.txt',
            'failed_items_path': '/tmp/predict-out/path_lists/failed_items.txt',
            'detected_count': 2,
            'empty_count': 1,
            'failed_count': 0,
        },
    )
    assert state.active_prediction.last_path_lists['path_list_overview']['empty_count'] == 1
    assert state.active_prediction.last_path_lists['action_candidates'][0]['tool'] == 'organize_prediction_results'
    assert state.active_prediction.last_path_lists['report_path'] == '/tmp/predict-out/prediction_report.json'

    apply_tool_result_to_state(
        state,
        'organize_prediction_results',
        {
            'ok': True,
            'summary': '预测结果整理完成',
            'mode': 'images',
            'organization_overview': {'bucket_count': 2},
            'action_candidates': [{'tool': 'export_prediction_path_lists', 'description': '导出路径清单'}],
            'source_output_dir': '/tmp/predict-out',
            'source_report_path': '/tmp/predict-out/prediction_report.json',
            'destination_dir': '/tmp/predict-out/organized_by_class',
            'organize_by': 'by_class',
            'artifact_preference': 'auto',
            'copied_items': 2,
            'bucket_stats': {'Excavator': 1, 'bulldozer': 1},
            'sample_outputs': ['/tmp/predict-out/organized_by_class/Excavator/a.jpg'],
        },
    )
    assert state.active_prediction.last_organized_result['organization_overview']['bucket_count'] == 2
    assert state.active_prediction.last_organized_result['action_candidates'][0]['tool'] == 'export_prediction_path_lists'

    apply_tool_result_to_state(
        state,
        'scan_cameras',
        {
            'ok': True,
            'summary': '发现 2 个摄像头',
            'camera_count': 2,
            'cameras': [{'camera_id': 0}, {'camera_id': 1}],
            'camera_overview': {'camera_count': 2},
            'action_candidates': [{'tool': 'start_camera_prediction', 'description': '启动摄像头预测'}],
        },
    )
    assert state.active_prediction.last_realtime_status['camera_overview']['camera_count'] == 2
    assert state.active_prediction.last_realtime_status['action_candidates'][0]['tool'] == 'start_camera_prediction'

    apply_tool_result_to_state(
        state,
        'check_realtime_prediction_status',
        {
            'ok': True,
            'summary': '实时预测进行中',
            'session_id': 'rt-1',
            'source_type': 'camera',
            'source_label': 'camera:0',
            'status': 'running',
            'processed_frames': 25,
            'detected_frames': 8,
            'total_detections': 12,
            'class_counts': {'excavator': 12},
            'output_dir': '/tmp/realtime',
            'report_path': '/tmp/realtime/status.json',
            'realtime_status_overview': {'processed_frames': 25},
            'action_candidates': [{'tool': 'stop_realtime_prediction', 'description': '停止实时预测'}],
        },
    )
    assert state.active_prediction.last_realtime_status['realtime_status_overview']['processed_frames'] == 25
    assert state.active_prediction.last_realtime_status['action_candidates'][0]['tool'] == 'stop_realtime_prediction'

    apply_tool_result_to_state(
        state,
        'check_training_status',
        {
            'ok': True,
            'running': False,
            'device': '1',
            'pid': 4321,
            'log_file': '/tmp/train.log',
            'started_at': 123.4,
            'resolved_args': {'model': '/tmp/yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'batch': 16, 'imgsz': 960, 'optimizer': 'AdamW', 'freeze': 12, 'resume': True, 'lr0': 0.004, 'patience': 20, 'workers': 4, 'amp': False},
            'command': ['yolo', 'train', 'model=/tmp/yolov8n.pt', 'data=/tmp/data.yaml', 'device=1', 'batch=16', 'imgsz=960', 'optimizer=AdamW', 'freeze=12', 'resume=True', 'lr0=0.004', 'patience=20', 'workers=4', 'amp=False'],
            'summary': '当前没有在训练',
            'run_state': 'completed',
            'progress': {'epoch': 5, 'total_epochs': 5, 'progress_ratio': 1.0},
            'latest_metrics': {'ok': True, 'metrics': {'precision': 0.8}},
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'signals': ['training_completed'],
            'facts': ['precision=0.800'],
        },
    )
    tr = state.active_training
    assert tr.running is False
    assert tr.model == '/tmp/yolov8n.pt'
    assert tr.data_yaml == '/tmp/data.yaml'
    assert tr.device == '1'
    assert tr.batch == 16
    assert tr.imgsz == 960
    assert tr.optimizer == 'AdamW'
    assert tr.freeze == 12
    assert tr.resume is True
    assert tr.lr0 == 0.004
    assert tr.patience == 20
    assert tr.workers == 4
    assert tr.amp is False
    assert tr.pid is None
    assert tr.log_file == '/tmp/train.log'
    assert tr.started_at == 123.4
    assert 'command' not in tr.last_status

    apply_tool_result_to_state(
        state,
        'summarize_training_run',
        {
            'ok': True,
            'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
            'run_state': 'completed',
            'log_file': '/tmp/train.log',
            'progress': {'epoch': 5, 'total_epochs': 5, 'progress_ratio': 1.0},
            'latest_metrics': {'ok': True, 'metrics': {'precision': 0.8}},
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'signals': ['training_completed'],
            'facts': ['precision=0.800'],
        },
    )
    assert state.active_training.last_summary['run_state'] == 'completed'
    assert state.active_training.training_run_summary['run_state'] == 'completed'
    assert 'selected_run_id' not in state.active_training.last_summary

    apply_tool_result_to_state(
        state,
        'training_readiness',
        {
            'ok': True,
            'summary': '训练前检查完成',
            'ready': True,
            'risk_level': 'low',
            'warnings': [],
            'blockers': [],
            'resolved_data_yaml': '/tmp/data.yaml',
            'resolved_img_dir': '/tmp/images',
            'resolved_label_dir': '/tmp/labels',
        },
    )
    assert state.active_dataset.last_readiness['ready'] is True
    assert state.active_dataset.data_yaml == '/tmp/data.yaml'

    apply_tool_result_to_state(
        state,
        'prepare_dataset_for_training',
        {
            'ok': True,
            'summary': '数据准备完成',
            'dataset_root': '/tmp/prepared',
            'img_dir': '/tmp/prepared/images',
            'label_dir': '/tmp/prepared/labels',
            'data_yaml': '/tmp/prepared/data.yaml',
            'ready': True,
            'steps_completed': [
                {
                    'step': 'scan',
                    'ok': True,
                    'summary': '扫描完成: 共 90 张图片，类别 2 个',
                    'total_images': 90,
                    'labeled_images': 84,
                    'missing_labels': 6,
                    'missing_label_images': 6,
                    'missing_label_ratio': 0.0667,
                    'empty_labels': 1,
                    'risk_level': 'medium',
                    'classes': ['epidural', 'subdural'],
                    'class_stats': {'epidural': 66, 'subdural': 24},
                    'top_classes': [{'class_name': 'epidural', 'count': 66}],
                    'warnings': ['发现 6 张图片缺少标签（占比 6.7%），训练结果可能受到明显影响'],
                    'detected_data_yaml': '/tmp/prepared/data.yaml',
                    'detected_classes_txt': '/tmp/prepared/classes.txt',
                    'class_name_source': 'detected_classes_txt',
                },
            ],
        },
    )
    assert state.active_dataset.last_scan['class_stats']['epidural'] == 66
    assert state.active_dataset.last_scan['least_class'] == {'name': 'subdural', 'count': 24}
    assert state.active_dataset.last_scan['most_class'] == {'name': 'epidural', 'count': 66}
    assert state.active_dataset.last_scan['detected_classes_txt'] == '/tmp/prepared/classes.txt'

    apply_tool_result_to_state(
        state,
        'list_training_environments',
        {
            'ok': True,
            'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
            'environments': [{'name': 'yolodo'}, {'name': 'yolo'}],
            'default_environment': {'name': 'yolodo'},
        },
    )
    assert state.active_training.last_environment_probe['default_environment']['name'] == 'yolodo'
    assert state.active_training.training_environment == 'yolodo'
    assert state.active_training.last_environment_probe['environments'][0]['name'] == 'yolodo'
    assert 'summary' in state.active_training.last_environment_probe
    assert 'profiles' not in state.active_training.last_environment_probe

    apply_tool_result_to_state(
        state,
        'training_preflight',
        {
            'ok': True,
            'summary': '训练预检通过',
            'resolved_args': {'model': '/tmp/yolov8n.pt', 'data_yaml': '/tmp/data.yaml', 'epochs': 5, 'device': '1', 'training_environment': 'yolodo', 'project': '/runs/ablation', 'name': 'exp-blue', 'batch': 8, 'imgsz': 1280, 'fraction': 0.5, 'classes': [1, 3], 'single_cls': False, 'optimizer': 'SGD', 'freeze': 4, 'resume': False, 'lr0': 0.01, 'patience': 8, 'workers': 2, 'amp': True},
            'training_environment': {'name': 'yolodo'},
            'ready_to_start': True,
        },
    )
    assert state.active_training.last_preflight['ready_to_start'] is True
    assert state.active_training.model == '/tmp/yolov8n.pt'
    assert state.active_training.training_environment == 'yolodo'
    assert state.active_training.project == '/runs/ablation'
    assert state.active_training.run_name == 'exp-blue'
    assert state.active_training.batch == 8
    assert state.active_training.imgsz == 1280
    assert state.active_training.fraction == 0.5
    assert state.active_training.classes == [1, 3]
    assert state.active_training.single_cls is False
    assert state.active_training.optimizer == 'SGD'
    assert state.active_training.freeze == 4
    assert state.active_training.resume is False
    assert state.active_training.lr0 == 0.01
    assert state.active_training.patience == 8
    assert state.active_training.workers == 2
    assert state.active_training.amp is True
    assert state.active_training.last_preflight['training_environment']['name'] == 'yolodo'
    assert 'resolved_args' in state.active_training.last_preflight
    assert 'training_environment' in state.active_training.last_preflight
    assert 'command' not in state.active_training.last_preflight

    apply_tool_result_to_state(
        state,
        'start_training',
        {
            'ok': True,
            'summary': '训练启动成功',
            'pid': 9999,
            'device': '0',
            'log_file': '/tmp/train-start.log',
            'started_at': 456.7,
            'resolved_args': {'model': '/tmp/yolov8m.pt', 'data_yaml': '/tmp/data2.yaml', 'epochs': 10, 'training_environment': 'yolodo'},
            'training_environment': {'name': 'yolodo', 'display_name': 'YOLODO', 'gpu_available': True},
            'command': ['yolo', 'train', 'model=/tmp/yolov8m.pt'],
        },
    )
    assert state.active_training.last_start_result['resolved_args']['epochs'] == 10
    assert state.active_training.last_start_result['training_environment']['name'] == 'yolodo'
    assert 'command' not in state.active_training.last_start_result

    apply_tool_result_to_state(
        state,
        'check_training_loop_status',
        {
            'ok': True,
            'summary': '环训练进行中',
            'loop_id': 'loop-123',
            'loop_name': 'data-yolov8n',
            'status': 'running_round',
            'managed_level': 'conservative_auto',
            'max_rounds': 3,
            'current_round_index': 1,
            'completed_rounds': 0,
            'recorded_rounds': 1,
            'latest_round_card': {'round_index': 1, 'status': 'running'},
            'rounds': [{'round_index': 1}],
        },
    )
    assert state.active_training.active_loop_id == 'loop-123'
    assert state.active_training.last_loop_status['max_rounds'] == 3
    assert 'rounds' not in state.active_training.last_loop_status

    apply_tool_result_to_state(
        state,
        'start_training_loop',
        {
            'ok': True,
            'summary': '环训练已启动：loop-demo（loop_id=loop-123）',
            'loop_id': 'loop-123',
            'loop_name': 'loop-demo',
            'status': 'queued',
            'managed_level': 'conservative_auto',
            'boundaries': {'max_rounds': 3},
            'next_round_plan': {'round_index': 1},
        },
        {
            'model': '/tmp/loop-yolov8n.pt',
            'data_yaml': '/tmp/loop-data.yaml',
            'device': 'auto',
            'training_environment': 'yolodo',
            'epochs': 12,
            'batch': 6,
        },
    )
    assert state.active_training.model == '/tmp/loop-yolov8n.pt'
    assert state.active_training.data_yaml == '/tmp/data2.yaml'
    assert state.active_training.training_environment == 'yolodo'
    assert state.active_training.device == 'auto'
    assert state.active_training.batch == 6
    assert state.active_training.active_loop_name == 'loop-demo'
    assert state.active_training.active_loop_request['data_yaml'] == '/tmp/loop-data.yaml'
    assert state.active_training.active_loop_request['model'] == '/tmp/loop-yolov8n.pt'

    apply_tool_result_to_state(
        state,
        'inspect_training_loop',
        {
            'ok': True,
            'summary': '环训练详情',
            'loop_id': 'loop-123',
            'loop_name': 'data-yolov8n',
            'status': 'running_round',
            'managed_level': 'conservative_auto',
            'max_rounds': 3,
            'current_round_index': 1,
            'completed_rounds': 0,
            'recorded_rounds': 1,
            'rounds': [{'round_index': 1}],
            'experience_timeline': [{'round_index': 1, 'decision': 'continue'}],
        },
    )
    assert state.active_training.last_loop_detail['rounds'][0]['round_index'] == 1
    assert state.active_training.last_loop_detail['experience_timeline'][0]['decision'] == 'continue'

    apply_tool_result_to_state(
        state,
        'list_training_runs',
        {
            'ok': True,
            'summary': '找到 2 条最近训练记录',
            'runs': [
                {'run_id': 'train_log_1', 'run_state': 'stopped'},
                {'run_id': 'train_log_2', 'run_state': 'completed'},
            ],
        },
    )
    assert state.active_training.recent_runs[0]['run_id'] == 'train_log_1'

    apply_tool_result_to_state(
        state,
        'inspect_training_run',
        {
            'ok': True,
            'summary': '训练记录详情已就绪',
            'selected_run_id': 'train_log_2',
            'run_state': 'completed',
            'observation_stage': 'final',
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'progress': {'epoch': 5, 'total_epochs': 5, 'progress_ratio': 1.0},
            'facts': ['precision=0.800'],
        },
    )
    assert state.active_training.last_run_inspection['selected_run_id'] == 'train_log_2'

    apply_tool_result_to_state(
        state,
        'compare_training_runs',
        {
            'ok': True,
            'summary': '训练对比完成',
            'left_run_id': 'train_log_2',
            'right_run_id': 'train_log_1',
            'metric_deltas': {'precision': {'left': 0.8, 'right': 0.6, 'delta': 0.2}},
            'highlights': ['precision提升 +0.2000'],
        },
    )
    assert state.active_training.last_run_comparison['left_run_id'] == 'train_log_2'

    apply_tool_result_to_state(
        state,
        'select_best_training_run',
        {
            'ok': True,
            'summary': '最佳训练记录: train_log_2',
            'best_run_id': 'train_log_2',
            'best_run': {'run_id': 'train_log_2', 'run_state': 'completed'},
        },
    )
    assert state.active_training.best_run_selection['best_run_id'] == 'train_log_2'

    apply_tool_result_to_state(
        state,
        'retrieve_training_knowledge',
        {
            'ok': True,
            'summary': '检索到训练知识',
            'topic': 'optimizer',
            'stage': 'after_metrics',
            'model_family': 'yolov8',
            'matched_rule_ids': ['optimizer_rule'],
            'retrieval_overview': {'rule_count': 1},
            'action_candidates': [{'tool': 'recommend_next_training_step', 'description': '给出下一步建议'}],
        },
    )
    assert state.active_knowledge.last_retrieval['retrieval_overview']['rule_count'] == 1
    assert state.active_knowledge.last_retrieval['action_candidates'][0]['tool'] == 'recommend_next_training_step'

    apply_tool_result_to_state(
        state,
        'recommend_next_training_step',
        {
            'ok': True,
            'summary': '下一步建议: 继续短周期观察。',
            'recommended_action': 'continue_observing',
            'matched_rule_ids': ['generic_next_continue_observing'],
            'signals': ['training_running'],
            'recommendation_overview': {'recommended_action': 'continue_observing'},
            'action_candidates': [{'tool': 'check_training_status', 'description': '继续观察训练状态'}],
        },
    )
    assert state.active_knowledge.last_recommendation['recommended_action'] == 'continue_observing'
    assert state.active_knowledge.last_recommendation['recommendation_overview']['recommended_action'] == 'continue_observing'
    assert state.active_knowledge.last_recommendation['action_candidates'][0]['tool'] == 'check_training_status'

    apply_tool_result_to_state(
        state,
        'list_remote_profiles',
        {
            'ok': True,
            'summary': '发现 1 个远端配置',
            'profiles_path': '/tmp/remote_profiles.json',
            'default_profile': 'lab',
            'profiles': [{'name': 'lab', 'target_label': 'lab-host', 'remote_root': '/srv/lab'}],
            'ssh_aliases': ['lab'],
            'profile_overview': {'profile_count': 1},
            'action_candidates': [{'tool': 'upload_assets_to_remote', 'description': '上传产物到远端'}],
        },
    )
    assert state.active_remote_transfer.last_profile_listing['profile_overview']['profile_count'] == 1
    assert state.active_remote_transfer.last_profile_listing['action_candidates'][0]['tool'] == 'upload_assets_to_remote'

    apply_tool_result_to_state(
        state,
        'upload_assets_to_remote',
        {
            'ok': True,
            'summary': '远端上传完成',
            'target_label': 'lab-host',
            'profile_name': 'lab',
            'remote_root': '/srv/lab',
            'uploaded_count': 2,
            'uploaded_items': ['/srv/lab/a', '/srv/lab/b'],
            'transfer_overview': {'uploaded_count': 2},
            'action_candidates': [{'tool': 'download_assets_from_remote', 'description': '下载远端结果'}],
        },
    )
    assert state.active_remote_transfer.last_upload['transfer_overview']['uploaded_count'] == 2
    assert state.active_remote_transfer.last_upload['action_candidates'][0]['tool'] == 'download_assets_from_remote'

    print('state applier ok')


if __name__ == '__main__':
    main()
