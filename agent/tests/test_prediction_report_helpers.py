from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.prediction_report_helpers import summarize_prediction_report

TMP_ROOT = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_prediction_report_helper_test')


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        image_report = TMP_ROOT / 'prediction_report.json'
        image_report.write_text(json.dumps({
            'output_dir': 'D:/out/predict',
            'annotated_dir': 'D:/out/predict/annotated',
            'labels_dir': 'D:/out/predict/labels_yolo',
            'model': 'demo.pt',
            'source_path': 'D:/data/images',
            'processed_images': 3,
            'detected_images': 2,
            'empty_images': 1,
            'failed_reads': [{'path': 'D:/broken.jpg', 'reason': 'bad'}],
            'class_counts': {'bulldozer': 2, 'Excavator': 1},
            'image_results': [
                {'image_path': 'D:/data/images/a.jpg', 'detections': 1},
                {'image_path': 'D:/data/images/b.jpg', 'detections': 0},
                {'image_path': 'D:/data/images/c.jpg', 'detections': 2},
            ],
        }, ensure_ascii=False), encoding='utf-8')
        image_summary = summarize_prediction_report(report_path=str(image_report))
        assert image_summary['ok'] is True, image_summary
        assert image_summary['mode'] == 'images', image_summary
        assert image_summary['processed_images'] == 3, image_summary
        assert image_summary['detected_images'] == 2, image_summary
        assert image_summary['total_detections'] == 3, image_summary
        assert image_summary['class_counts']['bulldozer'] == 2, image_summary
        assert image_summary['next_actions'][0] == '可查看标注结果目录: D:/out/predict/annotated', image_summary

        videos_dir = TMP_ROOT / 'videos'
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_report = videos_dir / 'prediction_report.json'
        video_report.write_text(json.dumps({
            'output_dir': 'D:/out/videos',
            'model': 'demo.pt',
            'source_path': 'D:/data/videos',
            'processed_videos': 2,
            'total_frames': 24,
            'detected_frames': 13,
            'total_detections': 15,
            'failed_videos': [{'path': 'D:/data/videos/bad.mp4', 'reason': 'open failed'}],
            'class_counts': {'two_wheeler': 15},
            'video_results': [
                {'video_path': 'D:/data/videos/a.mp4', 'processed_frames': 12, 'detected_frames': 10, 'total_detections': 11},
                {'video_path': 'D:/data/videos/b.mp4', 'processed_frames': 12, 'detected_frames': 3, 'total_detections': 4},
            ],
        }, ensure_ascii=False), encoding='utf-8')
        video_summary = summarize_prediction_report(output_dir=str(videos_dir))
        assert video_summary['ok'] is True, video_summary
        assert video_summary['mode'] == 'videos', video_summary
        assert video_summary['processed_videos'] == 2, video_summary
        assert video_summary['total_frames'] == 24, video_summary
        assert video_summary['detected_frames'] == 13, video_summary
        assert video_summary['total_detections'] == 15, video_summary
        assert video_summary['class_counts']['two_wheeler'] == 15, video_summary
        assert video_summary['warnings'][0].startswith('有 1 个视频读取或处理失败'), video_summary

        missing_summary = summarize_prediction_report(output_dir=str(TMP_ROOT / 'missing'))
        assert missing_summary['ok'] is False, missing_summary
        assert '找不到报告文件' in missing_summary['summary'], missing_summary

        print('prediction report helpers ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
