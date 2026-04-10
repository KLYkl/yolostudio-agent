from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from utils.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from utils.file_utils import discover_files, get_unique_dir
from utils.label_writer import write_yolo_txt_from_xyxy

_DETECTION_COLORS: list[tuple[int, int, int]] = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
]


class PredictService:
    def __init__(self, output_root: str | Path | None = None) -> None:
        root = Path(output_root) if output_root else Path(
            os.getenv('YOLOSTUDIO_PREDICT_OUTPUT_ROOT', 'runs/predict')
        )
        self._output_root = root
        self._output_root.mkdir(parents=True, exist_ok=True)

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
                'next_actions': ['请显式提供 model，例如 /home/kly/yolov8n.pt'],
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
        annotated_dir = resolved_output_dir / 'annotated'
        labels_dir = resolved_output_dir / 'labels_yolo'
        originals_dir = resolved_output_dir / 'originals'
        if save_annotated:
            annotated_dir.mkdir(parents=True, exist_ok=True)
        if save_labels:
            labels_dir.mkdir(parents=True, exist_ok=True)
        if save_original:
            originals_dir.mkdir(parents=True, exist_ok=True)

        class_counter: Counter[str] = Counter()
        processed_images = 0
        detected_images = 0
        empty_images = 0
        failed_reads: list[dict[str, str]] = []
        detected_samples: list[str] = []
        empty_samples: list[str] = []
        image_results: list[dict[str, Any]] = []
        batch_size = max(1, batch_size)

        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start: batch_start + batch_size]
            valid_paths: list[Path] = []
            valid_frames: list[Image.Image] = []
            for image_path in batch_paths:
                frame = self._read_image(image_path)
                if frame is None:
                    failed_reads.append({
                        'path': str(image_path.resolve()),
                        'reason': '无法读取图片（可能已损坏或格式不受支持）',
                    })
                    continue
                valid_paths.append(image_path)
                valid_frames.append(frame)

            if not valid_frames:
                continue

            batch_detections = self._run_batch_inference(
                predictor,
                valid_frames,
                conf=conf,
                iou=iou,
            )
            for image_path, frame, detections in zip(valid_paths, valid_frames, batch_detections):
                processed_images += 1
                if detections:
                    detected_images += 1
                    if len(detected_samples) < 3:
                        detected_samples.append(str(image_path.resolve()))
                else:
                    empty_images += 1
                    if len(empty_samples) < 3:
                        empty_samples.append(str(image_path.resolve()))

                for det in detections:
                    class_counter[str(det.get('class_name', 'unknown'))] += 1

                artifact_paths: dict[str, str] = {}
                if save_annotated:
                    annotated_path = annotated_dir / image_path.name
                    annotated = self._draw_detections(frame, detections)
                    annotated.save(annotated_path)
                    artifact_paths['annotated'] = str(annotated_path.resolve())
                if save_labels:
                    label_path = labels_dir / f'{image_path.stem}.txt'
                    write_yolo_txt_from_xyxy(label_path, detections, int(frame.size[0]), int(frame.size[1]))
                    artifact_paths['label_yolo'] = str(label_path.resolve())
                if save_original:
                    original_path = originals_dir / image_path.name
                    shutil.copy2(image_path, original_path)
                    artifact_paths['original_copy'] = str(original_path.resolve())

                image_results.append({
                    'image_path': str(image_path.resolve()),
                    'detections': len(detections),
                    'classes': sorted({str(det.get('class_name', 'unknown')) for det in detections}),
                    'artifact_paths': artifact_paths,
                })

        if processed_images <= 0:
            return {
                'ok': False,
                'error': '未成功读取任何图片，预测未执行',
                'summary': '预测未启动：没有可处理的有效图片',
                'failed_reads': failed_reads,
                'next_actions': ['请先做 run_dataset_health_check，确认图片是否损坏或格式异常'],
            }

        report_path = ''
        if generate_report:
            report_path = str((resolved_output_dir / 'prediction_report.json').resolve())
            Path(report_path).write_text(
                json.dumps(
                    {
                        'generated_at': datetime.now().isoformat(),
                        'model': model,
                        'source_path': str(source.resolve()),
                        'processed_images': processed_images,
                        'detected_images': detected_images,
                        'empty_images': empty_images,
                        'failed_reads': failed_reads,
                        'class_counts': dict(class_counter),
                        'output_dir': str(resolved_output_dir.resolve()),
                        'annotated_dir': str(annotated_dir.resolve()) if save_annotated else '',
                        'labels_dir': str(labels_dir.resolve()) if save_labels else '',
                        'originals_dir': str(originals_dir.resolve()) if save_original else '',
                        'image_results': image_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )

        warnings: list[str] = []
        if failed_reads:
            warnings.append(f'有 {len(failed_reads)} 张图片读取失败')
        if truncated:
            warnings.append(f'已按 max_images 限制，仅处理前 {len(image_paths)} 张图片')
        if detected_images == 0:
            warnings.append('当前未检测到任何目标，可考虑更换模型或调低 conf')

        class_counts = dict(sorted(class_counter.items(), key=lambda item: (-item[1], item[0])))
        top_classes = [f'{name}={count}' for name, count in list(class_counts.items())[:4]]
        summary = f'预测完成: 已处理 {processed_images} 张图片, 有检测 {detected_images}, 无检测 {empty_images}'
        if top_classes:
            summary += f"，主要类别 {', '.join(top_classes)}"

        next_actions: list[str] = []
        if detected_images and save_annotated:
            next_actions.append(f'可查看标注结果目录: {annotated_dir.resolve()}')
        if save_labels:
            next_actions.append(f'可复用 YOLO 标签目录: {labels_dir.resolve()}')
        if report_path:
            next_actions.append(f'可查看预测报告: {report_path}')
        if empty_images:
            next_actions.append('若无检测图片较多，可尝试降低 conf 或更换模型再测')
        if not next_actions:
            next_actions.append('可调整 conf / iou 后重新预测，对比不同模型结果')

        return {
            'ok': True,
            'summary': summary,
            'model': model,
            'source_path': str(source.resolve()),
            'processed_images': processed_images,
            'detected_images': detected_images,
            'empty_images': empty_images,
            'failed_reads': failed_reads,
            'class_counts': class_counts,
            'detected_samples': detected_samples,
            'empty_samples': empty_samples,
            'output_dir': str(resolved_output_dir.resolve()),
            'annotated_dir': str(annotated_dir.resolve()) if save_annotated else '',
            'labels_dir': str(labels_dir.resolve()) if save_labels else '',
            'originals_dir': str(originals_dir.resolve()) if save_original else '',
            'report_path': report_path,
            'save_annotated': save_annotated,
            'save_labels': save_labels,
            'save_original': save_original,
            'generate_report': generate_report,
            'warnings': warnings,
            'next_actions': next_actions,
        }

    def summarize_prediction_results(self, *, report_path: str = '', output_dir: str = '') -> dict[str, Any]:
        resolved_report = Path(report_path) if report_path else Path(output_dir) / 'prediction_report.json'
        if not str(resolved_report).strip() or not resolved_report.exists():
            return {
                'ok': False,
                'error': '未找到 prediction_report.json',
                'summary': '预测结果汇总失败：找不到报告文件',
                'next_actions': ['请提供 report_path，或传入包含 prediction_report.json 的 output_dir'],
            }
        try:
            payload = json.loads(resolved_report.read_text(encoding='utf-8'))
        except Exception as exc:
            return {
                'ok': False,
                'error': f'解析预测报告失败: {exc}',
                'error_type': exc.__class__.__name__,
                'summary': '预测结果汇总失败：报告文件不可解析',
                'next_actions': ['请确认 prediction_report.json 是否为有效 UTF-8 JSON 文件'],
            }

        image_results = payload.get('image_results') or []
        video_results = payload.get('video_results') or []
        failed_reads = payload.get('failed_reads') or []
        class_counts = dict(sorted((payload.get('class_counts') or {}).items(), key=lambda item: (-item[1], item[0])))

        if video_results:
            processed_videos = int(payload.get('processed_videos') or len(video_results))
            total_frames = int(payload.get('total_frames') or sum(int(item.get('processed_frames') or 0) for item in video_results))
            detected_frames = int(payload.get('detected_frames') or sum(int(item.get('detected_frames') or 0) for item in video_results))
            total_detections = int(payload.get('total_detections') or sum(int(item.get('total_detections') or 0) for item in video_results))
            detected_samples = [str(item.get('video_path')) for item in video_results if int(item.get('detected_frames') or 0) > 0][:3]
            empty_samples = [str(item.get('video_path')) for item in video_results if int(item.get('detected_frames') or 0) <= 0][:3]
            warnings: list[str] = []
            failed_videos = payload.get('failed_videos') or []
            if failed_videos:
                warnings.append(f'有 {len(failed_videos)} 个视频读取或处理失败')
            if processed_videos and detected_frames <= 0:
                warnings.append('当前报告里没有检测到任何目标帧')
            if not class_counts:
                warnings.append('当前报告里没有有效类别统计')

            top_classes = [f'{name}={count}' for name, count in list(class_counts.items())[:4]]
            summary = f'视频预测结果摘要: 已处理 {processed_videos} 个视频, 总帧数 {total_frames}, 有检测帧 {detected_frames}, 总检测框 {total_detections}'
            if top_classes:
                summary += f"，主要类别 {', '.join(top_classes)}"

            next_actions: list[str] = [f'可查看预测报告: {resolved_report.resolve()}']
            return {
                'ok': True,
                'summary': summary,
                'mode': 'videos',
                'report_path': str(resolved_report.resolve()),
                'output_dir': str(payload.get('output_dir') or resolved_report.parent),
                'model': str(payload.get('model') or ''),
                'source_path': str(payload.get('source_path') or ''),
                'processed_videos': processed_videos,
                'total_frames': total_frames,
                'detected_frames': detected_frames,
                'total_detections': total_detections,
                'failed_reads': failed_videos,
                'class_counts': class_counts,
                'detected_samples': detected_samples,
                'empty_samples': empty_samples,
                'warnings': warnings,
                'next_actions': next_actions,
            }

        processed_images = int(payload.get('processed_images') or len(image_results))
        detected_images = int(payload.get('detected_images') or sum(1 for item in image_results if int(item.get('detections') or 0) > 0))
        empty_images = int(payload.get('empty_images') or max(processed_images - detected_images, 0))
        total_detections = sum(int(item.get('detections') or 0) for item in image_results)
        detected_samples = [str(item.get('image_path')) for item in image_results if int(item.get('detections') or 0) > 0][:3]
        empty_samples = [str(item.get('image_path')) for item in image_results if int(item.get('detections') or 0) <= 0][:3]
        warnings: list[str] = []
        if failed_reads:
            warnings.append(f'有 {len(failed_reads)} 张图片读取失败')
        if processed_images and empty_images / processed_images >= 0.8:
            warnings.append('无检测图片占比较高，建议复核模型或调低 conf 后复测')
        if not class_counts:
            warnings.append('当前报告里没有有效类别统计')

        top_classes = [f'{name}={count}' for name, count in list(class_counts.items())[:4]]
        summary = f'预测结果摘要: 已处理 {processed_images} 张图片, 有检测 {detected_images}, 无检测 {empty_images}, 总检测框 {total_detections}'
        if top_classes:
            summary += f"，主要类别 {', '.join(top_classes)}"

        next_actions: list[str] = []
        if payload.get('annotated_dir'):
            next_actions.append(f"可查看标注结果目录: {payload.get('annotated_dir')}")
        if payload.get('labels_dir'):
            next_actions.append(f"可复用 YOLO 标签目录: {payload.get('labels_dir')}")
        next_actions.append(f'可查看预测报告: {resolved_report.resolve()}')

        return {
            'ok': True,
            'summary': summary,
            'mode': 'images',
            'report_path': str(resolved_report.resolve()),
            'output_dir': str(payload.get('output_dir') or resolved_report.parent),
            'annotated_dir': str(payload.get('annotated_dir') or ''),
            'labels_dir': str(payload.get('labels_dir') or ''),
            'originals_dir': str(payload.get('originals_dir') or ''),
            'model': str(payload.get('model') or ''),
            'source_path': str(payload.get('source_path') or ''),
            'processed_images': processed_images,
            'detected_images': detected_images,
            'empty_images': empty_images,
            'failed_reads': failed_reads,
            'class_counts': class_counts,
            'total_detections': total_detections,
            'detected_samples': detected_samples,
            'empty_samples': empty_samples,
            'warnings': warnings,
            'next_actions': next_actions,
        }

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
        if not model.strip():
            return {
                'ok': False,
                'error': '请提供模型路径或模型名称',
                'summary': '视频预测未启动：缺少模型参数',
                'next_actions': ['请显式提供 model，例如 /home/kly/yolov8n.pt'],
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

        total_frames = 0
        detected_frames = 0
        total_detections = 0
        class_counter: Counter[str] = Counter()
        failed_videos: list[dict[str, str]] = []
        detected_samples: list[str] = []
        empty_samples: list[str] = []
        video_results: list[dict[str, Any]] = []

        for video_path in video_paths:
            stats = self._predict_single_video(
                predictor,
                video_path=video_path,
                output_root=resolved_output_dir,
                conf=conf,
                iou=iou,
                save_video=save_video,
                save_keyframes_annotated=save_keyframes_annotated,
                save_keyframes_raw=save_keyframes_raw,
                max_frames=max_frames,
            )
            if not stats.get('ok'):
                failed_videos.append({
                    'path': str(video_path.resolve()),
                    'reason': str(stats.get('error') or 'unknown'),
                })
                continue

            total_frames += int(stats.get('processed_frames') or 0)
            detected_frames += int(stats.get('detected_frames') or 0)
            total_detections += int(stats.get('total_detections') or 0)
            for name, count in (stats.get('class_counts') or {}).items():
                class_counter[str(name)] += int(count)
            if stats.get('detected_frames'):
                detected_samples.append(str(video_path.resolve()))
            else:
                empty_samples.append(str(video_path.resolve()))
            video_results.append(stats)

        processed_videos = len(video_results)
        if processed_videos <= 0:
            return {
                'ok': False,
                'error': '未成功处理任何视频',
                'summary': '视频预测未启动：没有可处理的有效视频',
                'failed_videos': failed_videos,
                'next_actions': ['请先确认视频文件完整可读，并检查模型路径'],
            }

        report_path = ''
        if generate_report:
            report_path = str((resolved_output_dir / 'video_prediction_report.json').resolve())
            Path(report_path).write_text(
                json.dumps(
                    {
                        'generated_at': datetime.now().isoformat(),
                        'model': model,
                        'source_path': str(source.resolve()),
                        'processed_videos': processed_videos,
                        'total_frames': total_frames,
                        'detected_frames': detected_frames,
                        'total_detections': total_detections,
                        'failed_videos': failed_videos,
                        'class_counts': dict(class_counter),
                        'output_dir': str(resolved_output_dir.resolve()),
                        'video_results': video_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )

        warnings: list[str] = []
        if failed_videos:
            warnings.append(f'有 {len(failed_videos)} 个视频读取或处理失败')
        if truncated:
            warnings.append(f'已按 max_videos 限制，仅处理前 {len(video_paths)} 个视频')
        if detected_frames == 0:
            warnings.append('当前未检测到任何目标帧，可考虑更换模型或调低 conf')

        class_counts = dict(sorted(class_counter.items(), key=lambda item: (-item[1], item[0])))
        top_classes = [f'{name}={count}' for name, count in list(class_counts.items())[:4]]
        summary = f'视频预测完成: 已处理 {processed_videos} 个视频, 总帧数 {total_frames}, 有检测帧 {detected_frames}, 总检测框 {total_detections}'
        if top_classes:
            summary += f"，主要类别 {', '.join(top_classes)}"

        next_actions: list[str] = []
        next_actions.append(f'可查看视频预测输出目录: {resolved_output_dir.resolve()}')
        if report_path:
            next_actions.append(f'可查看视频预测报告: {report_path}')
        if detected_frames == 0:
            next_actions.append('若视频全部无检测，可调低 conf 或更换模型再测')

        return {
            'ok': True,
            'summary': summary,
            'model': model,
            'source_path': str(source.resolve()),
            'processed_videos': processed_videos,
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'total_detections': total_detections,
            'failed_videos': failed_videos,
            'class_counts': class_counts,
            'detected_samples': detected_samples[:3],
            'empty_samples': empty_samples[:3],
            'output_dir': str(resolved_output_dir.resolve()),
            'report_path': report_path,
            'save_video': save_video,
            'save_keyframes_annotated': save_keyframes_annotated,
            'save_keyframes_raw': save_keyframes_raw,
            'generate_report': generate_report,
            'warnings': warnings,
            'next_actions': next_actions,
            'video_results': video_results,
        }

    @staticmethod
    def _load_model(model: str) -> Any:
        from ultralytics import YOLO

        return YOLO(model)

    @staticmethod
    def _run_batch_inference(model: Any, frames: list[Image.Image], *, conf: float, iou: float) -> list[list[dict[str, Any]]]:
        results = model(frames, conf=conf, iou=iou, half=True, verbose=False)
        batch: list[list[dict[str, Any]]] = []
        for index, result in enumerate(results):
            frame = frames[index]
            width, height = frame.size
            detections: list[dict[str, Any]] = []
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                batch.append(detections)
                continue
            xyxy = getattr(boxes, 'xyxy', None)
            confs = getattr(boxes, 'conf', None)
            classes = getattr(boxes, 'cls', None)
            if xyxy is None or confs is None or classes is None:
                batch.append(detections)
                continue
            coords = xyxy.cpu().tolist()
            scores = confs.cpu().tolist()
            class_ids = classes.cpu().tolist()
            names = getattr(model, 'names', {})
            for coord, score, class_id in zip(coords, scores, class_ids):
                cid = int(class_id)
                x1, y1, x2, y2 = coord
                detections.append({
                    'class_id': cid,
                    'class_name': str(names.get(cid, cid)),
                    'confidence': float(score),
                    'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                    'bbox': [
                        (x1 + x2) / 2 / width,
                        (y1 + y2) / 2 / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ],
                })
            batch.append(detections)
        return batch

    @staticmethod
    def _draw_detections(frame: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
        image = frame.copy()
        draw = ImageDraw.Draw(image)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
            class_id = int(det.get('class_id', 0))
            color = _DETECTION_COLORS[class_id % len(_DETECTION_COLORS)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            label = f"{det.get('class_name', 'unknown')} {float(det.get('confidence', 0.0)):.2f}"
            text_top = max(0, y1 - 14)
            draw.rectangle((x1, text_top, x1 + max(30, len(label) * 7), y1), fill=color)
            draw.text((x1 + 2, text_top + 1), label, fill=(255, 255, 255))
        return image

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
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return {'ok': False, 'error': '无法打开视频'}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        video_dir = output_root / video_path.stem
        video_dir.mkdir(parents=True, exist_ok=True)

        annotated_dir = video_dir / 'keyframes' / 'annotated'
        raw_dir = video_dir / 'keyframes' / 'raw'
        if save_keyframes_annotated:
            annotated_dir.mkdir(parents=True, exist_ok=True)
        if save_keyframes_raw:
            raw_dir.mkdir(parents=True, exist_ok=True)

        writer: cv2.VideoWriter | None = None
        video_output_path = ''
        if save_video and width > 0 and height > 0:
            video_output_path = str((video_dir / f'{video_path.stem}_result.mp4').resolve())
            writer = cv2.VideoWriter(
                video_output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                float(fps or 30.0),
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                writer = None
                video_output_path = ''

        processed_frames = 0
        detected_frames = 0
        total_detections = 0
        class_counter: Counter[str] = Counter()
        frame_results: list[dict[str, Any]] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and processed_frames >= max_frames:
                break
            processed_frames += 1

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = self._run_batch_inference(predictor, [pil], conf=conf, iou=iou)[0]
            if detections:
                detected_frames += 1
                total_detections += len(detections)
                for det in detections:
                    class_counter[str(det.get('class_name', 'unknown'))] += 1
                if save_keyframes_annotated:
                    annotated_path = annotated_dir / f'frame_{processed_frames:06d}.jpg'
                    self._draw_detections(pil, detections).save(annotated_path)
                if save_keyframes_raw:
                    raw_path = raw_dir / f'frame_{processed_frames:06d}.jpg'
                    cv2.imwrite(str(raw_path), frame)
            if writer is not None:
                annotated_frame = self._pil_to_bgr(self._draw_detections(pil, detections))
                writer.write(annotated_frame)
            frame_results.append({
                'frame_index': processed_frames,
                'detections': len(detections),
                'classes': sorted({str(det.get('class_name', 'unknown')) for det in detections}),
            })

        cap.release()
        if writer is not None:
            writer.release()

        return {
            'ok': True,
            'video_path': str(video_path.resolve()),
            'output_dir': str(video_dir.resolve()),
            'annotated_video': video_output_path,
            'annotated_keyframes_dir': str(annotated_dir.resolve()) if save_keyframes_annotated else '',
            'raw_keyframes_dir': str(raw_dir.resolve()) if save_keyframes_raw else '',
            'processed_frames': processed_frames,
            'detected_frames': detected_frames,
            'total_detections': total_detections,
            'class_counts': dict(sorted(class_counter.items(), key=lambda item: (-item[1], item[0]))),
            'frame_results': frame_results[:50],
        }

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> Any:
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _read_image(path: Path) -> Image.Image | None:
        try:
            with Image.open(path) as image:
                return image.convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError):
            return None

    def _resolve_output_dir(self, source: Path, output_dir: str) -> Path:
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = source.stem if source.is_file() else source.name
        return get_unique_dir(self._output_root / f'{stem}_predict_{timestamp}')
