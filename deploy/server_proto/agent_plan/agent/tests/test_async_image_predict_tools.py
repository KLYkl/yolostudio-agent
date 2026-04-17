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

from yolostudio_agent.agent.server.tools import predict_tools


def main() -> None:
    original_start = predict_tools.service.start_image_prediction
    original_check = predict_tools.service.check_image_prediction_status
    original_stop = predict_tools.service.stop_image_prediction

    try:
        predict_tools.service.start_image_prediction = lambda **kwargs: {
            'ok': True,
            'summary': '图片目录较大，已转为后台预测会话：共 320 张图片（session_id=image-predict-1234abcd）',
            'session_id': 'image-predict-1234abcd',
            'status': 'running',
            'run_state': 'running',
            'running': True,
            'source_path': kwargs['source_path'],
            'model': kwargs['model'],
            'total_images': 320,
            'processed_images': 0,
            'detected_images': 0,
            'empty_images': 0,
            'output_dir': '/tmp/predict-large',
            'report_path': '',
            'next_actions': [],
        }
        started = predict_tools.start_image_prediction(source_path='/data/images-large', model='demo.pt')
        assert started['ok'] is True, started
        assert started['prediction_session_overview']['total_images'] == 320, started
        assert 'check_image_prediction_status' in started['next_actions'][0], started
        assert started['action_candidates'], started

        predict_tools.service.check_image_prediction_status = lambda session_id='': {
            'ok': True,
            'summary': '后台图片预测运行中: 已处理 64/320 张图片, 有检测 40, 无检测 24',
            'session_id': session_id or 'image-predict-1234abcd',
            'status': 'running',
            'run_state': 'running',
            'running': True,
            'source_path': '/data/images-large',
            'model': 'demo.pt',
            'total_images': 320,
            'processed_images': 64,
            'detected_images': 40,
            'empty_images': 24,
            'total_detections': 88,
            'class_counts': {'excavator': 60, 'truck': 28},
            'output_dir': '/tmp/predict-large',
            'report_path': '',
            'next_actions': [],
        }
        running = predict_tools.check_image_prediction_status('image-predict-1234abcd')
        assert running['ok'] is True, running
        assert running['prediction_session_overview']['processed_images'] == 64, running
        assert 'stop_image_prediction' in running['next_actions'][0], running

        predict_tools.service.stop_image_prediction = lambda session_id='': {
            'ok': True,
            'summary': '后台图片预测已停止: 已处理 120/320 张图片, 有检测 75, 无检测 45',
            'session_id': session_id or 'image-predict-1234abcd',
            'status': 'stopped',
            'run_state': 'stopped',
            'running': False,
            'source_path': '/data/images-large',
            'model': 'demo.pt',
            'total_images': 320,
            'processed_images': 120,
            'detected_images': 75,
            'empty_images': 45,
            'total_detections': 160,
            'class_counts': {'excavator': 100, 'truck': 60},
            'output_dir': '/tmp/predict-large',
            'report_path': '/tmp/predict-large/prediction_report.json',
            'next_actions': [],
        }
        stopped = predict_tools.stop_image_prediction('image-predict-1234abcd')
        assert stopped['ok'] is True, stopped
        assert stopped['prediction_session_overview']['status'] == 'stopped', stopped
        assert 'prediction_report.json' in stopped['next_actions'][0], stopped

        print('async image predict tools ok')
    finally:
        predict_tools.service.start_image_prediction = original_start
        predict_tools.service.check_image_prediction_status = original_check
        predict_tools.service.stop_image_prediction = original_stop


if __name__ == '__main__':
    main()
