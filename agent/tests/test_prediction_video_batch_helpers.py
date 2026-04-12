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

from yolostudio_agent.agent.server.services.prediction_video_batch_helpers import predict_videos_batch

TMP_ROOT = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_prediction_video_batch_helper_test')


def _predict_single_video(_predictor, *, video_path: Path, output_root: Path, **_kwargs):
    if video_path.stem == 'broken':
        return {'ok': False, 'error': '无法打开视频'}
    base_dir = output_root / video_path.stem
    base_dir.mkdir(parents=True, exist_ok=True)
    if video_path.stem == 'empty':
        return {
            'ok': True,
            'video_path': str(video_path.resolve()),
            'output_dir': str(base_dir.resolve()),
            'processed_frames': 5,
            'detected_frames': 0,
            'total_detections': 0,
            'class_counts': {},
            'frame_results': [],
        }
    return {
        'ok': True,
        'video_path': str(video_path.resolve()),
        'output_dir': str(base_dir.resolve()),
        'processed_frames': 7,
        'detected_frames': 3,
        'total_detections': 4,
        'class_counts': {'Excavator': 1, 'bulldozer': 3},
        'frame_results': [],
    }


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    source_dir = TMP_ROOT / 'videos'
    output_dir = TMP_ROOT / 'out'
    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        good = source_dir / 'good.mp4'
        empty = source_dir / 'empty.mp4'
        broken = source_dir / 'broken.mp4'
        for path in (good, empty, broken):
            path.write_bytes(b'video')

        result = predict_videos_batch(
            object(),
            source=source_dir,
            video_paths=[good, empty, broken],
            conf=0.25,
            iou=0.45,
            resolved_output_dir=output_dir,
            save_video=True,
            save_keyframes_annotated=True,
            save_keyframes_raw=False,
            generate_report=True,
            max_frames=0,
            truncated=True,
            predict_single_video_fn=_predict_single_video,
        )

        assert result['ok'] is True, result
        assert result['processed_videos'] == 2, result
        assert result['total_frames'] == 12, result
        assert result['detected_frames'] == 3, result
        assert result['total_detections'] == 4, result
        assert result['class_counts']['bulldozer'] == 3, result
        assert len(result['failed_videos']) == 1, result
        assert '已按 max_videos 限制' in ' '.join(result['warnings']), result
        report = Path(result['report_path'])
        assert report.exists(), result
        payload = json.loads(report.read_text(encoding='utf-8'))
        assert payload['processed_videos'] == 2, payload
        assert payload['class_counts']['bulldozer'] == 3, payload
        print('prediction video batch helpers ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
