from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.client.session_state import SessionState
from agent_plan.agent.client.state_applier import apply_tool_result_to_state


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
            'command': ['yolo', 'train', 'model=/tmp/yolov8n.pt', 'data=/tmp/data.yaml', 'device=1'],
            'summary': '当前没有在训练',
        },
    )
    tr = state.active_training
    assert tr.running is False
    assert tr.model == ''
    assert tr.data_yaml == ''
    assert tr.device == ''
    assert tr.pid is None
    assert tr.log_file == ''
    assert tr.started_at is None
    print('state applier ok')


if __name__ == '__main__':
    main()
