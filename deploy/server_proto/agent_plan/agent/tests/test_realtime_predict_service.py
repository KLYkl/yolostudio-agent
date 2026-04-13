from __future__ import annotations

import shutil
import sys
import time
import types
import json
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services import realtime_predict_service as realtime_module


TMP_ROOT = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_realtime_predict_service')


class _FakeCapture:
    def __init__(self, limit: int = 1000) -> None:
        self._limit = limit
        self._count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, object | None]:
        if self._count >= self._limit:
            return False, None
        self._count += 1
        return True, {'frame_index': self._count}

    def release(self) -> None:
        self.released = True


def _wait_for_session(
    service: realtime_module.RealtimePredictService,
    session_id: str,
    predicate,
    *,
    timeout: float = 2.0,
) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = service.check_realtime_prediction_status(session_id=session_id)
        if predicate(last):
            return last
        time.sleep(0.01)
    raise AssertionError(f'session {session_id} did not reach expected state, last={last}')


def main() -> None:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    original_cv2 = realtime_module.cv2
    original_mss = realtime_module.mss
    realtime_module.cv2 = object()
    realtime_module.mss = object()

    try:
        service = realtime_module.RealtimePredictService(TMP_ROOT)
        service._scan_cameras = lambda max_devices: [{'id': 0, 'name': '摄像头 0'}]
        service._scan_screens = lambda: [{'id': 1, 'name': '屏幕 1 (1920x1080)', 'width': 1920, 'height': 1080, 'left': 0, 'top': 0}]
        service._test_rtsp = lambda rtsp_url, timeout_ms: (True, None)
        service._load_model = lambda model: {'model': model}
        service._sleep_ms = lambda _frame_interval_ms: time.sleep(0.002)

        def _fake_process_frame(self, session, model, frame) -> None:
            del model, frame
            session['processed_frames'] = int(session.get('processed_frames') or 0) + 1
            session['last_frame_at'] = time.time()
            session['detected_frames'] = int(session.get('detected_frames') or 0) + 1
            session['total_detections'] = int(session.get('total_detections') or 0) + 2
            session['class_counts'] = {'excavator': int(session['processed_frames']) * 2}

        service._process_frame = types.MethodType(_fake_process_frame, service)

        camera_scan = service.scan_cameras(max_devices=3)
        assert camera_scan['ok'] is True, camera_scan
        assert camera_scan['camera_count'] == 1, camera_scan

        screen_scan = service.scan_screens()
        assert screen_scan['ok'] is True, screen_scan
        assert screen_scan['screen_count'] == 1, screen_scan

        rtsp_probe = service.test_rtsp_stream(rtsp_url='rtsp://demo/live', timeout_ms=1200)
        assert rtsp_probe['ok'] is True, rtsp_probe

        active_capture = _FakeCapture(limit=200)
        service._open_capture = lambda source: active_capture
        started = service.start_camera_prediction(model='demo.pt', camera_id=0, max_frames=0)
        assert started['ok'] is True, started
        session_id = started['session_id']

        running_status = _wait_for_session(
            service,
            session_id,
            lambda status: status.get('ok') and status.get('processed_frames', 0) >= 2 and status.get('running') is True,
        )
        assert running_status['source_type'] == 'camera', running_status

        blocked = service.start_rtsp_prediction(model='demo.pt', rtsp_url='rtsp://demo/live')
        assert blocked['ok'] is False, blocked
        assert '请先停止当前会话' in blocked['summary'], blocked

        stopped = service.stop_realtime_prediction(session_id=session_id)
        assert stopped['ok'] is True, stopped
        assert stopped['status'] in {'stopped', 'completed'}, stopped
        assert stopped['processed_frames'] >= 2, stopped
        assert Path(stopped['report_path']).exists(), stopped
        stopped_report = json.loads(Path(stopped['report_path']).read_text(encoding='utf-8'))
        assert stopped_report['report_path'] == str(Path(stopped['report_path']).resolve()), stopped_report
        assert active_capture.released is True

        service._open_capture = lambda source: _FakeCapture(limit=2)
        rtsp_started = service.start_rtsp_prediction(model='demo.pt', rtsp_url='rtsp://demo/live', max_frames=2)
        assert rtsp_started['ok'] is True, rtsp_started
        rtsp_status = _wait_for_session(
            service,
            rtsp_started['session_id'],
            lambda status: status.get('ok') and status.get('status') == 'completed',
        )
        assert rtsp_status['processed_frames'] == 2, rtsp_status
        assert Path(rtsp_status['report_path']).exists(), rtsp_status
        rtsp_report = json.loads(Path(rtsp_status['report_path']).read_text(encoding='utf-8'))
        assert rtsp_report['report_path'] == str(Path(rtsp_status['report_path']).resolve()), rtsp_report

        service._get_screen_monitor = lambda screen_id: {'id': screen_id, 'width': 1920, 'height': 1080, 'left': 0, 'top': 0}
        service._grab_screen_frame = lambda monitor: {'monitor_id': monitor['id']}
        screen_started = service.start_screen_prediction(model='demo.pt', screen_id=1, max_frames=3)
        assert screen_started['ok'] is True, screen_started
        screen_status = _wait_for_session(
            service,
            screen_started['session_id'],
            lambda status: status.get('ok') and status.get('status') == 'completed',
        )
        assert screen_status['source_type'] == 'screen', screen_status
        assert screen_status['processed_frames'] == 3, screen_status
        assert Path(screen_status['report_path']).exists(), screen_status
        screen_report = json.loads(Path(screen_status['report_path']).read_text(encoding='utf-8'))
        assert screen_report['report_path'] == str(Path(screen_status['report_path']).resolve()), screen_report

        waiting_rtsp = {
            'status': 'running',
            'source_type': 'rtsp',
            'capture_opened_at': time.time(),
            'processed_frames': 0,
            'detected_frames': 0,
            'total_detections': 0,
            'output_dir': '/tmp/realtime-rtsp',
            'report_path': '',
            'running': True,
        }
        assert '等待首帧 / 关键帧' in service._status_summary(waiting_rtsp)
        assert any('GOP / keyframe interval' in item for item in service._status_next_actions(waiting_rtsp))

        print('realtime predict service ok')
    finally:
        realtime_module.cv2 = original_cv2
        realtime_module.mss = original_mss
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
