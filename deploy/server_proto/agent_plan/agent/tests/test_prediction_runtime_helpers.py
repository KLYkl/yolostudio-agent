from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.prediction_runtime_helpers import (
    draw_detections,
    pil_to_bgr,
    read_image,
    run_batch_inference,
)

TMP_ROOT = Path('D:/yolodo2.0/agent_plan/.tmp_prediction_runtime_helper_test')


class _TensorLike:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _TensorLike(xyxy)
        self.conf = _TensorLike(conf)
        self.cls = _TensorLike(cls)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    names = {0: 'Excavator', 1: 'bulldozer'}

    def __call__(self, frames, conf: float, iou: float, half: bool, verbose: bool):
        assert conf == 0.25
        assert iou == 0.45
        assert half is True
        assert verbose is False
        return [
            _Result(_Boxes([[1, 1, 10, 10]], [0.9], [0])) ,
            _Result(_Boxes([[2, 2, 20, 20], [4, 4, 24, 24]], [0.8, 0.7], [1, 1])),
        ]


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        image_path = TMP_ROOT / 'sample.jpg'
        Image.new('RGB', (32, 32), (120, 30, 40)).save(image_path, format='JPEG')
        broken_path = TMP_ROOT / 'broken.jpg'
        broken_path.write_text('not-an-image', encoding='utf-8')

        image = read_image(image_path)
        assert image is not None and image.size == (32, 32)
        assert read_image(broken_path) is None

        frames = [Image.new('RGB', (32, 32), (255, 0, 0)), Image.new('RGB', (64, 64), (0, 255, 0))]
        detections = run_batch_inference(_FakeModel(), frames, conf=0.25, iou=0.45)
        assert len(detections) == 2
        assert detections[0][0]['class_name'] == 'Excavator'
        assert len(detections[1]) == 2

        drawn = draw_detections(frames[0], detections[0])
        assert drawn.size == frames[0].size
        bgr = pil_to_bgr(drawn, np_module=np, cv2_module=__import__('cv2'))
        assert tuple(bgr.shape[:2]) == (32, 32)

        print('prediction runtime helpers ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
