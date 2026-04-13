from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    mss = None

from PIL import Image

from yolostudio_agent.agent.server.services.prediction_runtime_helpers import run_batch_inference
from yolostudio_agent.agent.server.services.realtime_device_helpers import DeviceScanner


class RealtimePredictService:
    def __init__(self, output_root: Path) -> None:
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._active_session_id: str = ''
        self._latest_session_id: str = ''

    def scan_cameras(self, *, max_devices: int = 5) -> dict[str, Any]:
        if cv2 is None:
            return {
                'ok': False,
                'error': '当前环境缺少 cv2，无法扫描摄像头',
                'summary': '摄像头扫描失败：缺少 OpenCV 运行时',
                'next_actions': ['请在当前环境安装 opencv-python，再重试摄像头扫描'],
            }
        cameras = self._scan_cameras(max_devices=max_devices)
        summary = f'摄像头扫描完成: 发现 {len(cameras)} 个可用摄像头'
        next_actions = ['如需开始实时预测，可调用 start_camera_prediction']
        if not cameras:
            next_actions = ['当前未发现可用摄像头；请确认设备连接和系统权限']
        return {
            'ok': True,
            'summary': summary,
            'camera_count': len(cameras),
            'cameras': cameras,
            'next_actions': next_actions,
        }

    def scan_screens(self) -> dict[str, Any]:
        if mss is None:
            return {
                'ok': False,
                'error': '当前环境缺少 mss，无法扫描屏幕',
                'summary': '屏幕扫描失败：缺少 mss 运行时',
                'next_actions': ['请在当前环境安装 mss，再重试屏幕扫描'],
            }
        screens = self._scan_screens()
        summary = f'屏幕扫描完成: 发现 {len(screens)} 个可用屏幕'
        next_actions = ['如需开始屏幕预测，可调用 start_screen_prediction']
        if not screens:
            next_actions = ['当前未发现可用屏幕；请确认运行环境是否支持屏幕采集']
        return {
            'ok': True,
            'summary': summary,
            'screen_count': len(screens),
            'screens': screens,
            'next_actions': next_actions,
        }

    def test_rtsp_stream(self, *, rtsp_url: str, timeout_ms: int = 5000) -> dict[str, Any]:
        if cv2 is None:
            return {
                'ok': False,
                'error': '当前环境缺少 cv2，无法测试 RTSP',
                'summary': 'RTSP 流测试失败：缺少 OpenCV 运行时',
                'next_actions': ['请在当前环境安装 opencv-python，再重试 RTSP 测试'],
            }
        ok, error = self._test_rtsp(rtsp_url=rtsp_url, timeout_ms=timeout_ms)
        if ok:
            return {
                'ok': True,
                'summary': 'RTSP 流测试通过：当前地址可连接并能读取视频帧',
                'rtsp_url': rtsp_url,
                'next_actions': ['如需开始实时预测，可调用 start_rtsp_prediction'],
            }
        return {
            'ok': False,
            'error': error or '无法连接 RTSP 流',
            'summary': 'RTSP 流测试失败：当前地址不可用',
            'rtsp_url': rtsp_url,
            'next_actions': ['请确认 RTSP 地址、账号密码和网络连通性'],
        }

    def start_camera_prediction(
        self,
        *,
        model: str,
        camera_id: int = 0,
        conf: float = 0.25,
        iou: float = 0.45,
        output_dir: str = '',
        frame_interval_ms: int = 100,
        max_frames: int = 0,
    ) -> dict[str, Any]:
        return self._start_stream_session(
            model=model,
            source_type='camera',
            source_value=camera_id,
            source_label=f'camera:{camera_id}',
            conf=conf,
            iou=iou,
            output_dir=output_dir,
            frame_interval_ms=frame_interval_ms,
            max_frames=max_frames,
        )

    def start_rtsp_prediction(
        self,
        *,
        model: str,
        rtsp_url: str,
        conf: float = 0.25,
        iou: float = 0.45,
        output_dir: str = '',
        frame_interval_ms: int = 100,
        max_frames: int = 0,
    ) -> dict[str, Any]:
        if not str(rtsp_url or '').strip():
            return {
                'ok': False,
                'error': 'rtsp_url 不能为空',
                'summary': 'RTSP 实时预测未启动：缺少流地址',
                'next_actions': ['请提供可用的 rtsp_url'],
            }
        return self._start_stream_session(
            model=model,
            source_type='rtsp',
            source_value=rtsp_url,
            source_label=rtsp_url,
            conf=conf,
            iou=iou,
            output_dir=output_dir,
            frame_interval_ms=frame_interval_ms,
            max_frames=max_frames,
        )

    def start_screen_prediction(
        self,
        *,
        model: str,
        screen_id: int = 1,
        conf: float = 0.25,
        iou: float = 0.45,
        output_dir: str = '',
        frame_interval_ms: int = 100,
        max_frames: int = 0,
    ) -> dict[str, Any]:
        return self._start_stream_session(
            model=model,
            source_type='screen',
            source_value=screen_id,
            source_label=f'screen:{screen_id}',
            conf=conf,
            iou=iou,
            output_dir=output_dir,
            frame_interval_ms=frame_interval_ms,
            max_frames=max_frames,
        )

    def check_realtime_prediction_status(self, *, session_id: str = '') -> dict[str, Any]:
        session = self._resolve_session(session_id)
        if session is None:
            return {
                'ok': False,
                'error': '当前没有可查询的实时预测会话',
                'summary': '实时预测状态查询失败：当前没有会话',
                'next_actions': ['请先启动摄像头 / RTSP / 屏幕实时预测'],
            }
        snapshot = self._snapshot(session)
        return {
            'ok': True,
            **snapshot,
            'summary': self._status_summary(snapshot),
            'next_actions': self._status_next_actions(snapshot),
        }

    def stop_realtime_prediction(self, *, session_id: str = '') -> dict[str, Any]:
        session = self._resolve_session(session_id)
        if session is None:
            return {
                'ok': False,
                'error': '当前没有可停止的实时预测会话',
                'summary': '停止实时预测失败：当前没有运行中的会话',
                'next_actions': ['请先启动摄像头 / RTSP / 屏幕实时预测'],
            }
        session['stop_event'].set()
        thread = session.get('thread')
        if isinstance(thread, threading.Thread):
            thread.join(timeout=5.0)
        snapshot = self._snapshot(session)
        return {
            'ok': True,
            **snapshot,
            'summary': f'实时预测已停止: 已处理 {snapshot.get("processed_frames", 0)} 帧，检测到 {snapshot.get("total_detections", 0)} 个目标',
            'next_actions': self._status_next_actions(snapshot),
        }

    def _start_stream_session(
        self,
        *,
        model: str,
        source_type: str,
        source_value: Any,
        source_label: str,
        conf: float,
        iou: float,
        output_dir: str,
        frame_interval_ms: int,
        max_frames: int,
    ) -> dict[str, Any]:
        if cv2 is None:
            return {
                'ok': False,
                'error': '当前环境缺少 cv2，无法启动实时预测',
                'summary': '实时预测未启动：缺少 OpenCV 运行时',
                'next_actions': ['请在当前环境安装 opencv-python，再重试实时预测'],
            }
        if source_type == 'screen' and mss is None:
            return {
                'ok': False,
                'error': '当前环境缺少 mss，无法启动屏幕预测',
                'summary': '屏幕预测未启动：缺少 mss 运行时',
                'next_actions': ['请在当前环境安装 mss，再重试屏幕预测'],
            }
        if not str(model or '').strip():
            return {
                'ok': False,
                'error': '请提供模型路径或模型名称',
                'summary': '实时预测未启动：缺少模型参数',
                'next_actions': ['请显式提供 model，例如 /models/yolov8n.pt'],
            }

        with self._lock:
            active = self._sessions.get(self._active_session_id) if self._active_session_id else None
            if active and active.get('running'):
                return {
                    'ok': False,
                    'error': '当前已有实时预测会话在运行',
                    'summary': '实时预测未启动：请先停止当前会话',
                    'active_session_id': str(active.get('session_id') or ''),
                    'next_actions': ['先执行 check_realtime_prediction_status / stop_realtime_prediction，再启动新的实时预测'],
                }

        session_id = f'realtime-{source_type}-{uuid.uuid4().hex[:8]}'
        resolved_output_dir = self._resolve_output_dir(source_type=source_type, output_dir=output_dir)
        session: dict[str, Any] = {
            'session_id': session_id,
            'source_type': source_type,
            'source_label': source_label,
            'source_value': source_value,
            'running': True,
            'status': 'running',
            'started_at': time.time(),
            'ended_at': None,
            'processed_frames': 0,
            'detected_frames': 0,
            'total_detections': 0,
            'class_counts': {},
            'output_dir': str(resolved_output_dir.resolve()),
            'report_path': '',
            'error': '',
            'conf': conf,
            'iou': iou,
            'frame_interval_ms': max(1, int(frame_interval_ms)),
            'max_frames': max(0, int(max_frames)),
            'stop_event': threading.Event(),
            'capture_opened_at': None,
            'last_frame_at': None,
            'model': model,
        }
        thread = threading.Thread(target=self._run_session, args=(session,), daemon=True)
        session['thread'] = thread

        with self._lock:
            self._sessions[session_id] = session
            self._active_session_id = session_id
            self._latest_session_id = session_id
        thread.start()
        return {
            'ok': True,
            'summary': f'实时预测已启动: {source_type} 源 {source_label}',
            'session_id': session_id,
            'source_type': source_type,
            'source_label': source_label,
            'output_dir': str(resolved_output_dir.resolve()),
            'next_actions': [
                '可继续调用 check_realtime_prediction_status 查看实时进度',
                '如需结束，可调用 stop_realtime_prediction',
            ],
        }

    def _run_session(self, session: dict[str, Any]) -> None:
        terminal_status = str(session.get('status') or 'running')
        try:
            model = self._load_model(str(session['model']))
            if session['source_type'] == 'screen':
                self._run_screen_loop(session, model)
            else:
                self._run_capture_loop(session, model)
            if session.get('status') == 'running':
                terminal_status = 'completed' if not session['stop_event'].is_set() else 'stopped'
        except Exception as exc:
            terminal_status = 'error'
            session['error'] = str(exc)
        finally:
            session['ended_at'] = time.time()
            report_path = self._write_report(session)
            session['report_path'] = str(report_path.resolve())
            session['status'] = terminal_status
            session['running'] = False
            with self._lock:
                if self._active_session_id == session.get('session_id'):
                    self._active_session_id = ''

    def _run_capture_loop(self, session: dict[str, Any], model: Any) -> None:
        source_value = session['source_value']
        capture = self._open_capture(source_value)
        if capture is None or not capture.isOpened():
            raise RuntimeError(f'无法打开视频源: {source_value}')
        session['capture_opened_at'] = time.time()
        try:
            while not session['stop_event'].is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                self._process_frame(session, model, frame)
                if session['max_frames'] and int(session['processed_frames']) >= int(session['max_frames']):
                    break
                self._sleep_ms(int(session['frame_interval_ms']))
        finally:
            capture.release()

    def _run_screen_loop(self, session: dict[str, Any], model: Any) -> None:
        screen_id = int(session['source_value'])
        monitor = self._get_screen_monitor(screen_id)
        if monitor is None:
            raise RuntimeError(f'找不到可用屏幕: {screen_id}')
        while not session['stop_event'].is_set():
            frame = self._grab_screen_frame(monitor)
            self._process_frame(session, model, frame)
            if session['max_frames'] and int(session['processed_frames']) >= int(session['max_frames']):
                break
            self._sleep_ms(int(session['frame_interval_ms']))

    def _process_frame(self, session: dict[str, Any], model: Any, frame: Any) -> None:
        if cv2 is None:
            raise RuntimeError('当前环境缺少 cv2')
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = run_batch_inference(model, [pil], conf=float(session['conf']), iou=float(session['iou']))[0]
        session['processed_frames'] = int(session.get('processed_frames') or 0) + 1
        session['last_frame_at'] = time.time()
        if detections:
            session['detected_frames'] = int(session.get('detected_frames') or 0) + 1
            session['total_detections'] = int(session.get('total_detections') or 0) + len(detections)
            class_counter = Counter(session.get('class_counts') or {})
            for det in detections:
                class_counter[str(det.get('class_name', 'unknown'))] += 1
            session['class_counts'] = dict(sorted(class_counter.items(), key=lambda item: (-item[1], item[0])))

    def _snapshot(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            'session_id': str(session.get('session_id') or ''),
            'source_type': str(session.get('source_type') or ''),
            'source_label': str(session.get('source_label') or ''),
            'running': bool(session.get('running')),
            'status': str(session.get('status') or ''),
            'started_at': session.get('started_at'),
            'ended_at': session.get('ended_at'),
            'capture_opened_at': session.get('capture_opened_at'),
            'last_frame_at': session.get('last_frame_at'),
            'processed_frames': int(session.get('processed_frames') or 0),
            'detected_frames': int(session.get('detected_frames') or 0),
            'total_detections': int(session.get('total_detections') or 0),
            'class_counts': dict(session.get('class_counts') or {}),
            'output_dir': str(session.get('output_dir') or ''),
            'report_path': str(session.get('report_path') or ''),
            'error': str(session.get('error') or ''),
        }

    def _status_summary(self, snapshot: dict[str, Any]) -> str:
        status = str(snapshot.get('status') or '')
        if status == 'running':
            if int(snapshot.get('processed_frames') or 0) <= 0 and snapshot.get('capture_opened_at'):
                if str(snapshot.get('source_type') or '') == 'rtsp':
                    return '实时预测已连接 RTSP 流，正在等待首帧 / 关键帧'
                return '实时预测已连接输入源，正在等待首帧'
            return (
                f"实时预测运行中: 已处理 {snapshot.get('processed_frames', 0)} 帧, "
                f"有检测 {snapshot.get('detected_frames', 0)} 帧, 总检测 {snapshot.get('total_detections', 0)}"
            )
        if status == 'error':
            return f"实时预测异常结束: {snapshot.get('error') or 'unknown error'}"
        return (
            f"实时预测已结束: 已处理 {snapshot.get('processed_frames', 0)} 帧, "
            f"有检测 {snapshot.get('detected_frames', 0)} 帧, 总检测 {snapshot.get('total_detections', 0)}"
        )

    def _status_next_actions(self, snapshot: dict[str, Any]) -> list[str]:
        next_actions: list[str] = []
        if snapshot.get('report_path'):
            next_actions.append(f"可查看实时预测报告: {snapshot.get('report_path')}")
        if snapshot.get('output_dir'):
            next_actions.append(f"可查看实时预测输出目录: {snapshot.get('output_dir')}")
        if snapshot.get('running') and int(snapshot.get('processed_frames') or 0) <= 0 and snapshot.get('capture_opened_at'):
            if str(snapshot.get('source_type') or '') == 'rtsp':
                next_actions.append('若长时间仍无帧推进，请检查 RTSP 源是否需要等待关键帧，或把 GOP / keyframe interval 调小')
            else:
                next_actions.append('若长时间仍无帧推进，请检查输入源是否持续输出画面')
        if snapshot.get('running'):
            next_actions.append('如需结束，可调用 stop_realtime_prediction')
        return next_actions or ['当前可重新启动新的实时预测会话']

    def _write_report(self, session: dict[str, Any]) -> Path:
        output_dir = Path(str(session['output_dir']))
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'realtime_prediction_report.json'
        payload = self._snapshot(session)
        payload['report_path'] = str(report_path.resolve())
        payload['generated_at'] = time.time()
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return report_path

    def _resolve_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            if str(session_id).strip():
                return self._sessions.get(session_id)
            if self._active_session_id and self._active_session_id in self._sessions:
                return self._sessions[self._active_session_id]
            if self._latest_session_id and self._latest_session_id in self._sessions:
                return self._sessions[self._latest_session_id]
            return None

    def _resolve_output_dir(self, *, source_type: str, output_dir: str) -> Path:
        if str(output_dir).strip():
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        path = self._output_root / f'realtime_{source_type}_{timestamp}'
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scan_cameras(self, *, max_devices: int) -> list[dict[str, Any]]:
        return DeviceScanner.scan_cameras(max_devices=max_devices)

    def _scan_screens(self) -> list[dict[str, Any]]:
        return DeviceScanner.scan_screens()

    def _test_rtsp(self, *, rtsp_url: str, timeout_ms: int) -> tuple[bool, str | None]:
        return DeviceScanner.test_rtsp(rtsp_url, timeout_ms=timeout_ms)

    def _open_capture(self, source: Any) -> Any:
        if cv2 is None:
            return None
        return cv2.VideoCapture(source)

    def _get_screen_monitor(self, screen_id: int) -> dict[str, Any] | None:
        screens = self._scan_screens()
        for screen in screens:
            if int(screen.get('id', 0)) == int(screen_id):
                return screen
        return None

    def _grab_screen_frame(self, monitor: dict[str, Any]) -> Any:
        if mss is None or np is None or cv2 is None:
            raise RuntimeError('当前环境不支持屏幕采集')
        with mss.mss() as sct:
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def _load_model(self, model: str) -> Any:
        from yolostudio_agent.agent.server.services.prediction_runtime_helpers import load_prediction_model

        return load_prediction_model(model)

    @staticmethod
    def _sleep_ms(frame_interval_ms: int) -> None:
        time.sleep(max(frame_interval_ms, 1) / 1000.0)
