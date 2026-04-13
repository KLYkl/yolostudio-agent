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
from yolostudio_agent.agent.client.state_applier import apply_tool_result_to_state


def main() -> None:
    state = SessionState(session_id='state-applier-smoke')

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
        },
        {'source_path': '/data/src'},
    )
    assert state.active_dataset.dataset_root == '/tmp/extract'
    assert state.active_dataset.img_dir == '/tmp/extract/images'
    assert state.active_dataset.label_dir == '/tmp/extract/labels'
    assert state.active_dataset.last_extract_result['extracted'] == 8

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
        'recommend_next_training_step',
        {
            'ok': True,
            'summary': '下一步建议: 继续短周期观察。',
            'recommended_action': 'continue_observing',
            'matched_rule_ids': ['generic_next_continue_observing'],
            'signals': ['training_running'],
        },
    )
    assert state.active_knowledge.last_recommendation['recommended_action'] == 'continue_observing'
    print('state applier ok')


if __name__ == '__main__':
    main()
