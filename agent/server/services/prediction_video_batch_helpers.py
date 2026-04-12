from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def predict_videos_batch(
    predictor: Any,
    *,
    source: Path,
    video_paths: list[Path],
    conf: float,
    iou: float,
    resolved_output_dir: Path,
    save_video: bool,
    save_keyframes_annotated: bool,
    save_keyframes_raw: bool,
    generate_report: bool,
    max_frames: int,
    truncated: bool,
    predict_single_video_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    total_frames = 0
    detected_frames = 0
    total_detections = 0
    class_counter: Counter[str] = Counter()
    failed_videos: list[dict[str, str]] = []
    detected_samples: list[str] = []
    empty_samples: list[str] = []
    video_results: list[dict[str, Any]] = []

    for video_path in video_paths:
        stats = predict_single_video_fn(
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
                    'model': '',
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

    next_actions: list[str] = [f'可查看视频预测输出目录: {resolved_output_dir.resolve()}']
    if report_path:
        next_actions.append(f'可查看视频预测报告: {report_path}')
    if detected_frames == 0:
        next_actions.append('若视频全部无检测，可调低 conf 或更换模型再测')

    return {
        'ok': True,
        'summary': summary,
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
