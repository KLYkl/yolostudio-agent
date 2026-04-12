from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_prediction_report(*, report_path: str = '', output_dir: str = '') -> dict[str, Any]:
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

    video_results = payload.get('video_results') or []
    if video_results:
        return _summarize_video_payload(payload, resolved_report)
    return _summarize_image_payload(payload, resolved_report)


def _summarize_video_payload(payload: dict[str, Any], resolved_report: Path) -> dict[str, Any]:
    video_results = payload.get('video_results') or []
    class_counts = dict(sorted((payload.get('class_counts') or {}).items(), key=lambda item: (-item[1], item[0])))
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
        'next_actions': [f'可查看预测报告: {resolved_report.resolve()}'],
    }


def _summarize_image_payload(payload: dict[str, Any], resolved_report: Path) -> dict[str, Any]:
    image_results = payload.get('image_results') or []
    failed_reads = payload.get('failed_reads') or []
    class_counts = dict(sorted((payload.get('class_counts') or {}).items(), key=lambda item: (-item[1], item[0])))
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
