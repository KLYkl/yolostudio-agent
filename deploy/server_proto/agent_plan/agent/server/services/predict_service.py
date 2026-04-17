from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from yolostudio_agent.agent.server.services.prediction_image_helpers import predict_images_batch
from yolostudio_agent.agent.server.services.prediction_report_helpers import (
    export_prediction_path_lists as _export_prediction_path_lists,
    export_prediction_report as _export_prediction_report,
    inspect_prediction_outputs as _inspect_prediction_outputs,
    organize_prediction_results as _organize_prediction_results,
    summarize_prediction_report,
)
from yolostudio_agent.agent.server.services.prediction_runtime_helpers import (
    draw_detections,
    load_prediction_model,
    pil_to_bgr,
    read_image,
    run_batch_inference,
)
from yolostudio_agent.agent.server.services.prediction_video_batch_helpers import predict_videos_batch
from yolostudio_agent.agent.server.services.prediction_video_helpers import predict_single_video
from yolostudio_agent.agent.server.services.realtime_predict_service import RealtimePredictService

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency for video prediction
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency for video prediction
    np = None

from PIL import Image

from utils.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from utils.file_utils import discover_files, get_unique_dir

try:
    from utils.label_writer import write_yolo_txt_from_xyxy as _write_yolo_txt_from_xyxy
except (ImportError, ModuleNotFoundError):
    def _write_yolo_txt_from_xyxy(
        txt_path: Path,
        detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with txt_path.open('w', encoding='utf-8') as handle:
            for detection in detections:
                class_id = int(detection.get('class_id', 0))
                x1, y1, x2, y2 = [float(value) for value in detection.get('xyxy', [0.0, 0.0, 0.0, 0.0])]
                xc = (x1 + x2) / 2 / frame_width
                yc = (y1 + y2) / 2 / frame_height
                bw = (x2 - x1) / frame_width
                bh = (y2 - y1) / frame_height
                handle.write(f'{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')

def _missing_video_runtime() -> dict[str, Any]:
    missing: list[str] = []
    if cv2 is None:
        missing.append('cv2')
    if np is None:
        missing.append('numpy')
    modules = ', '.join(missing) if missing else '视频推理依赖'
    return {
        'ok': False,
        'error': f'当前运行环境缺少视频处理依赖: {modules}',
        'summary': '视频预测未启动：缺少视频处理依赖',
        'next_actions': ['请在当前运行环境安装 opencv-python 和 numpy，或切换到已具备视频依赖的环境'],
    }


class PredictService:
    def __init__(self, output_root: str | Path | None = None) -> None:
        root = Path(output_root) if output_root else Path(
            os.getenv('YOLOSTUDIO_PREDICT_OUTPUT_ROOT', 'runs/predict')
        )
        self._output_root = root
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._realtime = RealtimePredictService(self._output_root)
        self._image_lock = threading.Lock()
        self._image_sessions: dict[str, dict[str, Any]] = {}
        self._active_image_session_id: str = ''
        self._latest_image_session_id: str = ''
        try:
            self._background_image_threshold = max(0, int(str(os.getenv('YOLOSTUDIO_ASYNC_IMAGE_PREDICT_THRESHOLD', '200') or '200').strip()))
        except Exception:
            self._background_image_threshold = 200

    def _prepare_image_prediction_request(
        self,
        *,
        model: str,
        source_path: str,
        max_images: int,
        output_dir: str,
    ) -> dict[str, Any]:
        if not model.strip():
            return {
                'ok': False,
                'error': '请提供模型路径或模型名称',
                'summary': '预测未启动：缺少模型参数',
                'next_actions': ['请显式提供 model，例如 /models/yolov8n.pt 或你自己的权重文件'],
            }

        source = Path(source_path)
        if not source.exists():
            return {
                'ok': False,
                'error': f'路径不存在: {source_path}',
                'summary': '预测未启动：输入路径不存在',
                'next_actions': ['请确认图片文件或目录路径是否正确'],
            }

        image_paths = discover_files(source, IMAGE_EXTENSIONS)
        if not image_paths:
            return {
                'ok': False,
                'error': '未找到可预测的图片文件',
                'summary': '预测未启动：当前路径下没有可用图片',
                'next_actions': ['请提供单张图片路径，或包含图片的目录路径'],
            }

        truncated = False
        if max_images > 0 and len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
            truncated = True

        return {
            'ok': True,
            'source': source,
            'image_paths': image_paths,
            'truncated': truncated,
            'resolved_output_dir': self._resolve_output_dir(source, output_dir),
        }

    def _should_background_image_prediction(self, *, source: Path, image_count: int) -> bool:
        return source.is_dir() and self._background_image_threshold > 0 and image_count > self._background_image_threshold

    def _image_session_snapshot(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            'session_id': str(session.get('session_id') or ''),
            'source_path': str(session.get('source_path') or ''),
            'model': str(session.get('model') or ''),
            'running': bool(session.get('running')),
            'status': str(session.get('status') or ''),
            'run_state': str(session.get('run_state') or session.get('status') or ''),
            'started_at': session.get('started_at'),
            'ended_at': session.get('ended_at'),
            'total_images': int(session.get('total_images') or 0),
            'processed_images': int(session.get('processed_images') or 0),
            'detected_images': int(session.get('detected_images') or 0),
            'empty_images': int(session.get('empty_images') or 0),
            'total_detections': int(session.get('total_detections') or 0),
            'class_counts': dict(session.get('class_counts') or {}),
            'output_dir': str(session.get('output_dir') or ''),
            'report_path': str(session.get('report_path') or ''),
            'warnings': list(session.get('warnings') or []),
            'last_image_path': str(session.get('last_image_path') or ''),
            'error': str(session.get('error') or ''),
        }

    def _image_status_summary(self, snapshot: dict[str, Any]) -> str:
        status = str(snapshot.get('status') or '')
        processed = int(snapshot.get('processed_images') or 0)
        total = int(snapshot.get('total_images') or 0)
        detected = int(snapshot.get('detected_images') or 0)
        empty = int(snapshot.get('empty_images') or 0)
        if status == 'running':
            return f'后台图片预测运行中: 已处理 {processed}/{total} 张图片, 有检测 {detected}, 无检测 {empty}'
        if status == 'stopping':
            return f'后台图片预测正在停止: 已处理 {processed}/{total} 张图片, 有检测 {detected}, 无检测 {empty}'
        if status == 'stopped':
            return f'后台图片预测已停止: 已处理 {processed}/{total} 张图片, 有检测 {detected}, 无检测 {empty}'
        if status == 'error':
            return f"后台图片预测异常结束: {snapshot.get('error') or 'unknown error'}"
        return f'后台图片预测已完成: 已处理 {processed}/{total} 张图片, 有检测 {detected}, 无检测 {empty}'

    def _image_status_next_actions(self, snapshot: dict[str, Any]) -> list[str]:
        actions: list[str] = []
        if snapshot.get('report_path'):
            actions.append(f"可查看预测报告: {snapshot.get('report_path')}")
        if snapshot.get('output_dir'):
            actions.append(f"可查看预测输出目录: {snapshot.get('output_dir')}")
        if snapshot.get('running'):
            if str(snapshot.get('status') or '') == 'stopping':
                actions.append('停止请求已发出，可稍后再次调用 check_image_prediction_status 查看最终状态')
            else:
                actions.append('如需结束，可调用 stop_image_prediction')
        return actions or ['可调整参数后重新启动图片预测']

    def _resolve_image_session(self, session_id: str) -> dict[str, Any] | None:
        with self._image_lock:
            if str(session_id).strip():
                return self._image_sessions.get(session_id)
            if self._active_image_session_id and self._active_image_session_id in self._image_sessions:
                return self._image_sessions[self._active_image_session_id]
            if self._latest_image_session_id and self._latest_image_session_id in self._image_sessions:
                return self._image_sessions[self._latest_image_session_id]
            return None

    def predict_images(
        self,
        *,
        model: str,
        source_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        output_dir: str = '',
        save_annotated: bool = True,
        save_labels: bool = False,
        save_original: bool = False,
        generate_report: bool = True,
        max_images: int = 0,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        prepared = self._prepare_image_prediction_request(
            model=model,
            source_path=source_path,
            max_images=max_images,
            output_dir=output_dir,
        )
        if not prepared.get('ok'):
            return prepared
        source = prepared['source']
        image_paths = list(prepared['image_paths'])
        truncated = bool(prepared['truncated'])

        if self._should_background_image_prediction(source=source, image_count=len(image_paths)):
            return self.start_image_prediction(
                source_path=source_path,
                model=model,
                conf=conf,
                iou=iou,
                output_dir=output_dir,
                save_annotated=save_annotated,
                save_labels=save_labels,
                save_original=save_original,
                generate_report=generate_report,
                max_images=max_images,
                batch_size=batch_size,
            )

        try:
            predictor = self._load_model(model)
        except Exception as exc:
            return {
                'ok': False,
                'error': f'加载预测模型失败: {exc}',
                'error_type': exc.__class__.__name__,
                'summary': '预测未启动：模型加载失败',
                'next_actions': ['请确认模型路径是否正确，且当前环境已安装 ultralytics'],
            }

        resolved_output_dir = prepared['resolved_output_dir']
        result = predict_images_batch(
            predictor,
            image_paths=image_paths,
            source=source,
            conf=conf,
            iou=iou,
            resolved_output_dir=resolved_output_dir,
            save_annotated=save_annotated,
            save_labels=save_labels,
            save_original=save_original,
            generate_report=generate_report,
            batch_size=batch_size,
            truncated=truncated,
            draw_detections_fn=self._draw_detections,
            read_image_fn=self._read_image,
            run_batch_inference_fn=self._run_batch_inference,
            write_yolo_txt_fn=_write_yolo_txt_from_xyxy,
        )
        if result.get('ok'):
            result['model'] = model
            result['source_path'] = str(source.resolve())
            if result.get('report_path'):
                report_path = Path(str(result['report_path']))
                try:
                    payload = json.loads(report_path.read_text(encoding='utf-8'))
                    payload['model'] = model
                    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception:
                    pass
        return result

    def start_image_prediction(
        self,
        *,
        model: str,
        source_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        output_dir: str = '',
        save_annotated: bool = True,
        save_labels: bool = False,
        save_original: bool = False,
        generate_report: bool = True,
        max_images: int = 0,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        prepared = self._prepare_image_prediction_request(
            model=model,
            source_path=source_path,
            max_images=max_images,
            output_dir=output_dir,
        )
        if not prepared.get('ok'):
            return prepared

        with self._image_lock:
            active = self._image_sessions.get(self._active_image_session_id) if self._active_image_session_id else None
            if active and active.get('running'):
                active_snapshot = self._image_session_snapshot(active)
                return {
                    'ok': False,
                    'error': '当前已有后台图片预测会话在运行',
                    'summary': '后台图片预测未启动：请先停止当前会话',
                    'active_session_id': active_snapshot.get('session_id'),
                    'next_actions': ['先执行 check_image_prediction_status / stop_image_prediction，再启动新的后台图片预测'],
                }

        try:
            predictor = self._load_model(model)
        except Exception as exc:
            return {
                'ok': False,
                'error': f'加载预测模型失败: {exc}',
                'error_type': exc.__class__.__name__,
                'summary': '后台图片预测未启动：模型加载失败',
                'next_actions': ['请确认模型路径是否正确，且当前环境已安装 ultralytics'],
            }

        source = prepared['source']
        image_paths = list(prepared['image_paths'])
        session_id = f'image-predict-{uuid.uuid4().hex[:8]}'
        resolved_output_dir = prepared['resolved_output_dir']
        session: dict[str, Any] = {
            'session_id': session_id,
            'source_path': str(source.resolve()),
            'model': model,
            'running': True,
            'status': 'running',
            'run_state': 'running',
            'started_at': time.time(),
            'ended_at': None,
            'total_images': len(image_paths),
            'processed_images': 0,
            'detected_images': 0,
            'empty_images': 0,
            'total_detections': 0,
            'class_counts': {},
            'output_dir': str(resolved_output_dir.resolve()),
            'report_path': '',
            'warnings': [],
            'last_image_path': '',
            'error': '',
            'stop_event': threading.Event(),
            'thread': None,
            'predictor': predictor,
            'image_paths': image_paths,
            'source': source,
            'conf': conf,
            'iou': iou,
            'save_annotated': save_annotated,
            'save_labels': save_labels,
            'save_original': save_original,
            'generate_report': generate_report,
            'batch_size': batch_size,
            'truncated': bool(prepared['truncated']),
        }
        thread = threading.Thread(target=self._run_image_prediction_session, args=(session,), daemon=True)
        session['thread'] = thread

        with self._image_lock:
            self._image_sessions[session_id] = session
            self._active_image_session_id = session_id
            self._latest_image_session_id = session_id
        thread.start()

        snapshot = self._image_session_snapshot(session)
        return {
            'ok': True,
            **snapshot,
            'started_in_background': True,
            'summary': f'图片目录较大，已转为后台预测会话：共 {snapshot["total_images"]} 张图片（session_id={session_id}）',
            'prediction_session_overview': {
                'session_id': session_id,
                'status': 'running',
                'total_images': snapshot['total_images'],
                'processed_images': 0,
            },
            'next_actions': [
                '可继续调用 check_image_prediction_status 查看后台进度',
                '如需结束，可调用 stop_image_prediction',
            ],
        }

    def check_image_prediction_status(self, *, session_id: str = '') -> dict[str, Any]:
        session = self._resolve_image_session(session_id)
        if session is None:
            return {
                'ok': False,
                'error': '当前没有可查询的后台图片预测会话',
                'summary': '后台图片预测状态查询失败：当前没有会话',
                'next_actions': ['请先启动后台图片预测，或直接执行小规模同步 predict_images'],
            }
        snapshot = self._image_session_snapshot(session)
        return {
            'ok': True,
            **snapshot,
            'summary': self._image_status_summary(snapshot),
            'prediction_session_overview': {
                'session_id': snapshot['session_id'],
                'status': snapshot['status'],
                'total_images': snapshot['total_images'],
                'processed_images': snapshot['processed_images'],
                'detected_images': snapshot['detected_images'],
            },
            'next_actions': self._image_status_next_actions(snapshot),
        }

    def stop_image_prediction(self, *, session_id: str = '') -> dict[str, Any]:
        session = self._resolve_image_session(session_id)
        if session is None:
            return {
                'ok': False,
                'error': '当前没有可停止的后台图片预测会话',
                'summary': '停止后台图片预测失败：当前没有运行中的会话',
                'next_actions': ['请先启动后台图片预测'],
            }
        session['stop_event'].set()
        thread = session.get('thread')
        if isinstance(thread, threading.Thread):
            thread.join(timeout=5.0)
            if thread.is_alive():
                session['status'] = 'stopping'
                session['run_state'] = 'stopping'
        snapshot = self._image_session_snapshot(session)
        return {
            'ok': True,
            **snapshot,
            'summary': self._image_status_summary(snapshot),
            'prediction_session_overview': {
                'session_id': snapshot['session_id'],
                'status': snapshot['status'],
                'total_images': snapshot['total_images'],
                'processed_images': snapshot['processed_images'],
                'detected_images': snapshot['detected_images'],
            },
            'next_actions': self._image_status_next_actions(snapshot),
        }

    def _run_image_prediction_session(self, session: dict[str, Any]) -> None:
        terminal_status = 'completed'

        def _progress(payload: dict[str, Any]) -> None:
            session['processed_images'] = int(payload.get('processed_images') or 0)
            session['detected_images'] = int(payload.get('detected_images') or 0)
            session['empty_images'] = int(payload.get('empty_images') or 0)
            session['total_detections'] = int(payload.get('total_detections') or 0)
            session['class_counts'] = dict(payload.get('class_counts') or {})
            session['last_image_path'] = str(payload.get('last_image_path') or '')

        try:
            result = predict_images_batch(
                session['predictor'],
                image_paths=list(session['image_paths']),
                source=session['source'],
                conf=float(session['conf']),
                iou=float(session['iou']),
                resolved_output_dir=Path(str(session['output_dir'])),
                save_annotated=bool(session['save_annotated']),
                save_labels=bool(session['save_labels']),
                save_original=bool(session['save_original']),
                generate_report=bool(session['generate_report']),
                batch_size=max(1, int(session.get('batch_size') or 8)),
                truncated=bool(session.get('truncated')),
                draw_detections_fn=self._draw_detections,
                read_image_fn=self._read_image,
                run_batch_inference_fn=self._run_batch_inference,
                write_yolo_txt_fn=_write_yolo_txt_from_xyxy,
                progress_callback=_progress,
                should_stop=lambda: bool(session['stop_event'].is_set()),
            )
            session['processed_images'] = int(result.get('processed_images') or session.get('processed_images') or 0)
            session['detected_images'] = int(result.get('detected_images') or session.get('detected_images') or 0)
            session['empty_images'] = int(result.get('empty_images') or session.get('empty_images') or 0)
            session['total_detections'] = int(sum((result.get('class_counts') or {}).values()))
            session['class_counts'] = dict(result.get('class_counts') or session.get('class_counts') or {})
            session['output_dir'] = str(result.get('output_dir') or session.get('output_dir') or '')
            session['report_path'] = str(result.get('report_path') or session.get('report_path') or '')
            session['warnings'] = list(result.get('warnings') or [])
            session['last_image_path'] = str((result.get('image_results') or [{}])[-1].get('image_path') or session.get('last_image_path') or '')
            if result.get('ok'):
                session['status'] = str(result.get('run_state') or 'completed')
                session['run_state'] = str(result.get('run_state') or 'completed')
            else:
                terminal_status = 'error'
                session['status'] = 'error'
                session['run_state'] = 'error'
                session['error'] = str(result.get('error') or '图片预测失败')
            if result.get('ok') and result.get('stopped'):
                terminal_status = 'stopped'
        except Exception as exc:
            terminal_status = 'error'
            session['status'] = 'error'
            session['run_state'] = 'error'
            session['error'] = str(exc)
        finally:
            if terminal_status != 'error':
                session['status'] = str(session.get('status') or terminal_status)
                session['run_state'] = str(session.get('run_state') or session.get('status') or terminal_status)
            session['running'] = False
            session['ended_at'] = time.time()
            with self._image_lock:
                if self._active_image_session_id == session.get('session_id'):
                    self._active_image_session_id = ''
            session.pop('predictor', None)
            session.pop('image_paths', None)
            session.pop('source', None)


    def summarize_prediction_results(self, *, report_path: str = '', output_dir: str = '') -> dict[str, Any]:
        return summarize_prediction_report(report_path=report_path, output_dir=output_dir)

    def inspect_prediction_outputs(self, *, report_path: str = '', output_dir: str = '') -> dict[str, Any]:
        return _inspect_prediction_outputs(report_path=report_path, output_dir=output_dir)

    def export_prediction_report(
        self,
        *,
        report_path: str = '',
        output_dir: str = '',
        export_path: str = '',
        export_format: str = 'markdown',
    ) -> dict[str, Any]:
        return _export_prediction_report(
            report_path=report_path,
            output_dir=output_dir,
            export_path=export_path,
            export_format=export_format,
        )

    def export_prediction_path_lists(
        self,
        *,
        report_path: str = '',
        output_dir: str = '',
        export_dir: str = '',
    ) -> dict[str, Any]:
        return _export_prediction_path_lists(
            report_path=report_path,
            output_dir=output_dir,
            export_dir=export_dir,
        )

    def organize_prediction_results(
        self,
        *,
        report_path: str = '',
        output_dir: str = '',
        destination_dir: str = '',
        organize_by: str = 'detected_only',
        include_empty: bool = False,
        artifact_preference: str = 'auto',
    ) -> dict[str, Any]:
        return _organize_prediction_results(
            report_path=report_path,
            output_dir=output_dir,
            destination_dir=destination_dir,
            organize_by=organize_by,
            include_empty=include_empty,
            artifact_preference=artifact_preference,
        )

    def scan_cameras(self, *, max_devices: int = 5) -> dict[str, Any]:
        return self._realtime.scan_cameras(max_devices=max_devices)

    def scan_screens(self) -> dict[str, Any]:
        return self._realtime.scan_screens()

    def test_rtsp_stream(self, *, rtsp_url: str, timeout_ms: int = 5000) -> dict[str, Any]:
        return self._realtime.test_rtsp_stream(rtsp_url=rtsp_url, timeout_ms=timeout_ms)

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
        return self._realtime.start_camera_prediction(
            model=model,
            camera_id=camera_id,
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
        return self._realtime.start_rtsp_prediction(
            model=model,
            rtsp_url=rtsp_url,
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
        return self._realtime.start_screen_prediction(
            model=model,
            screen_id=screen_id,
            conf=conf,
            iou=iou,
            output_dir=output_dir,
            frame_interval_ms=frame_interval_ms,
            max_frames=max_frames,
        )

    def check_realtime_prediction_status(self, *, session_id: str = '') -> dict[str, Any]:
        return self._realtime.check_realtime_prediction_status(session_id=session_id)

    def stop_realtime_prediction(self, *, session_id: str = '') -> dict[str, Any]:
        return self._realtime.stop_realtime_prediction(session_id=session_id)


    def predict_videos(
        self,
        *,
        model: str,
        source_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        output_dir: str = '',
        save_video: bool = True,
        save_keyframes_annotated: bool = True,
        save_keyframes_raw: bool = False,
        generate_report: bool = True,
        max_videos: int = 0,
        max_frames: int = 0,
    ) -> dict[str, Any]:
        if cv2 is None or np is None:
            return _missing_video_runtime()

        if not model.strip():
            return {
                'ok': False,
                'error': '请提供模型路径或模型名称',
                'summary': '视频预测未启动：缺少模型参数',
                'next_actions': ['请显式提供 model，例如 /models/yolov8n.pt 或你自己的权重文件'],
            }

        source = Path(source_path)
        if not source.exists():
            return {
                'ok': False,
                'error': f'路径不存在: {source_path}',
                'summary': '视频预测未启动：输入路径不存在',
                'next_actions': ['请确认视频文件或目录路径是否正确'],
            }

        video_paths = discover_files(source, VIDEO_EXTENSIONS)
        if not video_paths:
            return {
                'ok': False,
                'error': '未找到可预测的视频文件',
                'summary': '视频预测未启动：当前路径下没有可用视频',
                'next_actions': ['请提供单个视频文件路径，或包含视频的目录路径'],
            }

        truncated = False
        if max_videos > 0 and len(video_paths) > max_videos:
            video_paths = video_paths[:max_videos]
            truncated = True

        try:
            predictor = self._load_model(model)
        except Exception as exc:
            return {
                'ok': False,
                'error': f'加载预测模型失败: {exc}',
                'error_type': exc.__class__.__name__,
                'summary': '视频预测未启动：模型加载失败',
                'next_actions': ['请确认模型路径是否正确，且当前环境已安装 ultralytics'],
            }

        resolved_output_dir = self._resolve_output_dir(source, output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        result = predict_videos_batch(
            predictor,
            source=source,
            video_paths=video_paths,
            conf=conf,
            iou=iou,
            resolved_output_dir=resolved_output_dir,
            save_video=save_video,
            save_keyframes_annotated=save_keyframes_annotated,
            save_keyframes_raw=save_keyframes_raw,
            generate_report=generate_report,
            max_frames=max_frames,
            truncated=truncated,
            predict_single_video_fn=self._predict_single_video,
        )
        if result.get('ok'):
            result['model'] = model
            result['source_path'] = str(source.resolve())
            if result.get('report_path'):
                report_path = Path(str(result['report_path']))
                try:
                    payload = json.loads(report_path.read_text(encoding='utf-8'))
                    payload['model'] = model
                    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception:
                    pass
        return result

    @staticmethod
    def _load_model(model: str) -> Any:
        return load_prediction_model(model)

    @staticmethod
    def _run_batch_inference(model: Any, frames: list[Image.Image], *, conf: float, iou: float) -> list[list[dict[str, Any]]]:
        return run_batch_inference(model, frames, conf=conf, iou=iou)

    @staticmethod
    def _draw_detections(frame: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
        return draw_detections(frame, detections)

    def _predict_single_video(
        self,
        predictor: Any,
        *,
        video_path: Path,
        output_root: Path,
        conf: float,
        iou: float,
        save_video: bool,
        save_keyframes_annotated: bool,
        save_keyframes_raw: bool,
        max_frames: int,
    ) -> dict[str, Any]:
        return predict_single_video(
            predictor,
            video_path=video_path,
            output_root=output_root,
            conf=conf,
            iou=iou,
            save_video=save_video,
            save_keyframes_annotated=save_keyframes_annotated,
            save_keyframes_raw=save_keyframes_raw,
            max_frames=max_frames,
            cv2_module=cv2,
            image_fromarray=Image.fromarray,
            run_batch_inference_fn=self._run_batch_inference,
            draw_detections_fn=self._draw_detections,
            pil_to_bgr_fn=self._pil_to_bgr,
        )

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> Any:
        return pil_to_bgr(image, np_module=np, cv2_module=cv2)

    @staticmethod
    def _read_image(path: Path) -> Image.Image | None:
        return read_image(path)

    def _resolve_output_dir(self, source: Path, output_dir: str) -> Path:
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = source.stem if source.is_file() else source.name
        return get_unique_dir(self._output_root / f'{stem}_predict_{timestamp}')
