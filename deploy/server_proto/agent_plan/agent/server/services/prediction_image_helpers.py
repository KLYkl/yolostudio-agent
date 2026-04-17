from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def predict_images_batch(
    predictor: Any,
    *,
    image_paths: list[Path],
    source: Path,
    conf: float,
    iou: float,
    resolved_output_dir: Path,
    save_annotated: bool,
    save_labels: bool,
    save_original: bool,
    generate_report: bool,
    batch_size: int,
    truncated: bool,
    draw_detections_fn: Callable[[Any, list[dict[str, Any]]], Any],
    read_image_fn: Callable[[Path], Any | None],
    run_batch_inference_fn: Callable[..., list[list[dict[str, Any]]]],
    write_yolo_txt_fn: Callable[[Path, list[dict[str, Any]], int, int], None],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
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
    stopped = False

    for batch_start in range(0, len(image_paths), batch_size):
        if should_stop and should_stop():
            stopped = True
            break
        batch_paths = image_paths[batch_start: batch_start + batch_size]
        valid_paths: list[Path] = []
        valid_frames: list[Any] = []
        for image_path in batch_paths:
            if should_stop and should_stop():
                stopped = True
                break
            frame = read_image_fn(image_path)
            if frame is None:
                failed_reads.append({
                    'path': str(image_path.resolve()),
                    'reason': '无法读取图片（可能已损坏或格式不受支持）',
                })
                continue
            valid_paths.append(image_path)
            valid_frames.append(frame)

        if stopped:
            break
        if not valid_frames:
            continue

        batch_detections = run_batch_inference_fn(
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
                annotated = draw_detections_fn(frame, detections)
                annotated.save(annotated_path)
                artifact_paths['annotated'] = str(annotated_path.resolve())
            if save_labels:
                label_path = labels_dir / f'{image_path.stem}.txt'
                write_yolo_txt_fn(label_path, detections, int(frame.size[0]), int(frame.size[1]))
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
            if progress_callback is not None:
                progress_callback({
                    'processed_images': processed_images,
                    'detected_images': detected_images,
                    'empty_images': empty_images,
                    'total_detections': sum(class_counter.values()),
                    'class_counts': dict(sorted(class_counter.items(), key=lambda item: (-item[1], item[0]))),
                    'last_image_path': str(image_path.resolve()),
                })

    if processed_images <= 0:
        if stopped:
            return {
                'ok': True,
                'stopped': True,
                'run_state': 'stopped',
                'summary': '预测已停止：尚未处理任何图片',
                'failed_reads': failed_reads,
                'next_actions': ['可调整参数后重新启动图片预测'],
                'processed_images': 0,
                'detected_images': 0,
                'empty_images': 0,
                'class_counts': {},
                'detected_samples': [],
                'empty_samples': [],
                'output_dir': str(resolved_output_dir.resolve()),
                'annotated_dir': str(annotated_dir.resolve()) if save_annotated else '',
                'labels_dir': str(labels_dir.resolve()) if save_labels else '',
                'originals_dir': str(originals_dir.resolve()) if save_original else '',
                'report_path': '',
                'save_annotated': save_annotated,
                'save_labels': save_labels,
                'save_original': save_original,
                'generate_report': generate_report,
                'warnings': ['预测在处理首张图片前被停止'],
                'image_results': [],
            }
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
                    'model': '',
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
    if stopped:
        warnings.append('预测已在后台会话中被手动停止，结果为部分产物')
    if detected_images == 0:
        warnings.append('当前未检测到任何目标，可考虑更换模型或调低 conf')

    class_counts = dict(sorted(class_counter.items(), key=lambda item: (-item[1], item[0])))
    top_classes = [f'{name}={count}' for name, count in list(class_counts.items())[:4]]
    summary_prefix = '预测已停止' if stopped else '预测完成'
    summary = f'{summary_prefix}: 已处理 {processed_images} 张图片, 有检测 {detected_images}, 无检测 {empty_images}'
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
        'stopped': stopped,
        'run_state': 'stopped' if stopped else 'completed',
        'summary': summary,
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
        'image_results': image_results,
    }
