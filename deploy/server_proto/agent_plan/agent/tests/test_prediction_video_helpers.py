from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.prediction_video_helpers import predict_single_video

TMP_ROOT = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_prediction_video_helper_test')


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


def _run_batch_inference(_predictor, frames, *, conf: float, iou: float):
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


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    source_dir = TMP_ROOT / 'videos'
    output_dir = TMP_ROOT / 'predict_out'
    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        video_path = source_dir / 'sample.avi'
        _make_video(video_path)
        result = predict_single_video(
            object(),
            video_path=video_path,
            output_root=output_dir,
            conf=0.25,
            iou=0.45,
            save_video=False,
            save_keyframes_annotated=True,
            save_keyframes_raw=True,
            max_frames=0,
            cv2_module=cv2,
            image_fromarray=Image.fromarray,
            run_batch_inference_fn=_run_batch_inference,
            draw_detections_fn=lambda frame, detections: frame,
            pil_to_bgr_fn=lambda image: cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
        )
        assert result['ok'] is True, result
        assert result['processed_frames'] == 3, result
        assert result['detected_frames'] == 2, result
        assert result['total_detections'] == 3, result
        assert result['class_counts']['bulldozer'] == 2, result
        assert result['class_counts']['Excavator'] == 1, result
        video_dir = Path(result['output_dir'])
        assert (video_dir / 'keyframes' / 'annotated').exists(), result
        assert (video_dir / 'keyframes' / 'raw').exists(), result
        assert len(result['frame_results']) == 3, result
        print('prediction video helpers ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
