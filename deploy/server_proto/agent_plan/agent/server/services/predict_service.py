from __future__ import annotations

import json
import os
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

        resolved_output_dir = self._resolve_output_dir(source, output_dir)
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
