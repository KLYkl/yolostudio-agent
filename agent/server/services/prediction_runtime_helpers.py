from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError

_DETECTION_COLORS: list[tuple[int, int, int]] = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
]


def load_prediction_model(model: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model)


def run_batch_inference(model: Any, frames: list[Image.Image], *, conf: float, iou: float) -> list[list[dict[str, Any]]]:
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


def draw_detections(frame: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
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


def pil_to_bgr(image: Image.Image, *, np_module: Any, cv2_module: Any) -> Any:
    return cv2_module.cvtColor(np_module.array(image), cv2_module.COLOR_RGB2BGR)


def read_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            return image.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        return None
