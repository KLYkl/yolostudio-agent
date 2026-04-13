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
    original_scan_cameras = predict_tools.service.scan_cameras
    original_scan_screens = predict_tools.service.scan_screens
    original_test_rtsp_stream = predict_tools.service.test_rtsp_stream
    original_start_camera_prediction = predict_tools.service.start_camera_prediction
    original_start_rtsp_prediction = predict_tools.service.start_rtsp_prediction
    original_start_screen_prediction = predict_tools.service.start_screen_prediction
    original_check_realtime_prediction_status = predict_tools.service.check_realtime_prediction_status
    original_stop_realtime_prediction = predict_tools.service.stop_realtime_prediction

    try:
        predict_tools.service.scan_cameras = lambda max_devices=5: {
            'ok': True,
            'summary': f'摄像头扫描完成: 发现 1 个可用摄像头（前 {max_devices} 个槽位）',
            'camera_count': 1,
            'cameras': [{'id': 0, 'name': '摄像头 0'}],
            'next_actions': [],
        }
        camera_scan = predict_tools.scan_cameras(3)
        assert camera_scan['ok'] is True, camera_scan
        assert 'start_camera_prediction' in camera_scan['next_actions'][0], camera_scan

        predict_tools.service.scan_screens = lambda: {
            'ok': True,
            'summary': '屏幕扫描完成: 发现 1 个可用屏幕',
            'screen_count': 1,
            'screens': [{'id': 1, 'name': '屏幕 1 (1920x1080)', 'width': 1920, 'height': 1080}],
            'next_actions': [],
        }
        screen_scan = predict_tools.scan_screens()
        assert screen_scan['ok'] is True, screen_scan
        assert 'start_screen_prediction' in screen_scan['next_actions'][0], screen_scan

        predict_tools.service.test_rtsp_stream = lambda rtsp_url, timeout_ms=5000: {
            'ok': True,
            'summary': 'RTSP 流测试通过：当前地址可连接并能读取视频帧',
            'rtsp_url': rtsp_url,
            'next_actions': [],
        }
        rtsp_probe = predict_tools.test_rtsp_stream('rtsp://demo/live', timeout_ms=1200)
        assert rtsp_probe['ok'] is True, rtsp_probe
        assert 'start_rtsp_prediction' in rtsp_probe['next_actions'][0], rtsp_probe

        predict_tools.service.start_camera_prediction = lambda **kwargs: {
            'ok': True,
            'summary': '实时预测已启动: camera 源 camera:0',
            'session_id': 'realtime-camera-12345678',
            'source_type': 'camera',
            'source_label': 'camera:0',
            'output_dir': '/tmp/realtime-camera',
            'next_actions': [],
        }
        camera_start = predict_tools.start_camera_prediction(model='demo.pt', camera_id=0)
        assert camera_start['ok'] is True, camera_start
        assert 'check_realtime_prediction_status' in camera_start['next_actions'][0], camera_start

        predict_tools.service.start_rtsp_prediction = lambda **kwargs: {
            'ok': True,
            'summary': '实时预测已启动: rtsp 源 rtsp://demo/live',
            'session_id': 'realtime-rtsp-12345678',
            'source_type': 'rtsp',
            'source_label': kwargs['rtsp_url'],
            'output_dir': '/tmp/realtime-rtsp',
            'next_actions': [],
        }
        rtsp_start = predict_tools.start_rtsp_prediction(model='demo.pt', rtsp_url='rtsp://demo/live')
        assert rtsp_start['ok'] is True, rtsp_start
        assert 'check_realtime_prediction_status' in rtsp_start['next_actions'][0], rtsp_start

        predict_tools.service.start_screen_prediction = lambda **kwargs: {
            'ok': True,
            'summary': '实时预测已启动: screen 源 screen:1',
            'session_id': 'realtime-screen-12345678',
            'source_type': 'screen',
            'source_label': 'screen:1',
            'output_dir': '/tmp/realtime-screen',
            'next_actions': [],
        }
        screen_start = predict_tools.start_screen_prediction(model='demo.pt', screen_id=1)
        assert screen_start['ok'] is True, screen_start
        assert 'check_realtime_prediction_status' in screen_start['next_actions'][0], screen_start

        predict_tools.service.check_realtime_prediction_status = lambda session_id='': {
            'ok': True,
            'summary': '实时预测运行中: 已处理 8 帧, 有检测 8 帧, 总检测 16',
            'session_id': session_id or 'realtime-camera-12345678',
            'source_type': 'camera',
            'source_label': 'camera:0',
            'status': 'running',
            'processed_frames': 8,
            'detected_frames': 8,
            'total_detections': 16,
            'class_counts': {'excavator': 16},
            'output_dir': '/tmp/realtime-camera',
            'report_path': '',
            'next_actions': [],
            'running': True,
        }
        running_status = predict_tools.check_realtime_prediction_status('realtime-camera-12345678')
        assert running_status['ok'] is True, running_status
        assert 'stop_realtime_prediction' in running_status['next_actions'][0], running_status

        predict_tools.service.stop_realtime_prediction = lambda session_id='': {
            'ok': True,
            'summary': '实时预测已停止: 已处理 8 帧，检测到 16 个目标',
            'session_id': session_id or 'realtime-camera-12345678',
            'source_type': 'camera',
            'source_label': 'camera:0',
            'status': 'stopped',
            'processed_frames': 8,
            'detected_frames': 8,
            'total_detections': 16,
            'class_counts': {'excavator': 16},
            'output_dir': '/tmp/realtime-camera',
            'report_path': '/tmp/realtime-camera/realtime_prediction_report.json',
            'next_actions': [],
            'running': False,
        }
        stopped = predict_tools.stop_realtime_prediction('realtime-camera-12345678')
        assert stopped['ok'] is True, stopped
        assert 'realtime_prediction_report.json' in stopped['next_actions'][0], stopped

        print('realtime predict tools ok')
    finally:
        predict_tools.service.scan_cameras = original_scan_cameras
        predict_tools.service.scan_screens = original_scan_screens
        predict_tools.service.test_rtsp_stream = original_test_rtsp_stream
        predict_tools.service.start_camera_prediction = original_start_camera_prediction
        predict_tools.service.start_rtsp_prediction = original_start_rtsp_prediction
        predict_tools.service.start_screen_prediction = original_start_screen_prediction
        predict_tools.service.check_realtime_prediction_status = original_check_realtime_prediction_status
        predict_tools.service.stop_realtime_prediction = original_stop_realtime_prediction


if __name__ == '__main__':
    main()
