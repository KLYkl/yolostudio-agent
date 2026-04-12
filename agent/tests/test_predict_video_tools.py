from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.tools import predict_tools


TMP_ROOT = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_predict_video_tools')


def _make_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5.0, (32, 32))
    if not writer.isOpened():
        raise RuntimeError('无法创建测试视频')
    try:
        for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)):
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[:, :] = color
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)

    source_dir = TMP_ROOT / 'videos'
    output_dir = TMP_ROOT / 'predict_out'
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        _make_video(source_dir / 'sample.avi')

        original_load = predict_tools.service._load_model
        original_run = predict_tools.service._run_batch_inference
        original_draw = predict_tools.service._draw_detections

        predict_tools.service._load_model = lambda model: object()

        def _fake_run(_model, frames, *, conf: float, iou: float):
            outputs = []
            for frame in frames:
                mean = np.array(frame).mean(axis=(0, 1))
                if mean[0] > mean[1] and mean[0] > mean[2]:
                    outputs.append([
                        {'class_id': 0, 'class_name': 'Excavator', 'confidence': 0.9, 'xyxy': [1, 1, 20, 20]},
                    ])
                elif mean[1] > mean[0] and mean[1] > mean[2]:
                    outputs.append([])
                else:
                    outputs.append([
                        {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.85, 'xyxy': [3, 3, 24, 24]},
                        {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.8, 'xyxy': [5, 5, 28, 28]},
                    ])
            return outputs

        predict_tools.service._run_batch_inference = _fake_run
        predict_tools.service._draw_detections = lambda frame, detections: frame

        result = predict_tools.predict_videos(
            source_path=str(source_dir),
            model='fake-model.pt',
            output_dir=str(output_dir),
            save_video=False,
            save_keyframes_annotated=True,
            save_keyframes_raw=True,
            generate_report=True,
        )
        assert result['ok'] is True, result
        assert result['processed_videos'] == 1, result
        assert result['total_frames'] == 3, result
        assert result['detected_frames'] == 2, result
        assert result['total_detections'] == 3, result
        assert result['class_counts']['bulldozer'] == 2, result
        assert result['class_counts']['Excavator'] == 1, result
        assert Path(result['report_path']).exists(), result
        video_dir = Path(result['output_dir']) / 'sample'
        assert (video_dir / 'keyframes' / 'annotated').exists(), result
        assert (video_dir / 'keyframes' / 'raw').exists(), result

        summary = predict_tools.summarize_prediction_results(report_path=result['report_path'])
        assert summary['ok'] is True, summary
        assert summary['mode'] == 'videos', summary
        assert summary['processed_videos'] == 1, summary
        assert summary['total_frames'] == 3, summary
        assert summary['detected_frames'] == 2, summary
        assert summary['total_detections'] == 3, summary
        assert summary['class_counts']['bulldozer'] == 2, summary

        missing_model = predict_tools.predict_videos(source_path=str(source_dir), model='')
        assert missing_model['ok'] is False, missing_model
        assert '缺少模型参数' in missing_model['summary'], missing_model

        missing_path = predict_tools.predict_videos(source_path=str(TMP_ROOT / 'missing'), model='fake-model.pt')
        assert missing_path['ok'] is False, missing_path
        assert '输入路径不存在' in missing_path['summary'], missing_path

        print('predict video tools ok')
    finally:
        predict_tools.service._load_model = original_load
        predict_tools.service._run_batch_inference = original_run
        predict_tools.service._draw_detections = original_draw
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT)


if __name__ == '__main__':
    main()
