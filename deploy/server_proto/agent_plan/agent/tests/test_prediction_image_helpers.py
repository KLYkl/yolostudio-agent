from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.prediction_image_helpers import predict_images_batch

TMP_ROOT = Path('D:/yolodo2.0/agent_plan/.tmp_prediction_image_helper_test')


def _make_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path, format='JPEG')


def _draw(frame, detections):
    return frame


def _read(path: Path):
    try:
        with Image.open(path) as img:
            return img.convert('RGB')
    except Exception:
        return None


def _run(_predictor, frames, *, conf: float, iou: float):
    outputs = []
    for frame in frames:
        width = int(frame.size[0])
        if width == 32:
            outputs.append([{'class_id': 0, 'class_name': 'Excavator', 'confidence': 0.9, 'xyxy': [1,1,20,20]}])
        elif width == 48:
            outputs.append([])
        else:
            outputs.append([
                {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.8, 'xyxy': [2,2,24,24]},
                {'class_id': 1, 'class_name': 'bulldozer', 'confidence': 0.7, 'xyxy': [5,5,28,28]},
            ])
    return outputs


def _write(path: Path, detections, w: int, h: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(len(detections)), encoding='utf-8')


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    source_dir = TMP_ROOT / 'images'
    output_dir = TMP_ROOT / 'out'
    source_dir.mkdir(parents=True, exist_ok=True)
    try:
        _make_image(source_dir / 'a.jpg', (32, 32), (255, 0, 0))
        _make_image(source_dir / 'b.jpg', (48, 48), (0, 255, 0))
        _make_image(source_dir / 'c.jpg', (64, 64), (0, 0, 255))
        (source_dir / 'broken.jpg').write_text('bad', encoding='utf-8')
        image_paths = [source_dir / 'a.jpg', source_dir / 'b.jpg', source_dir / 'c.jpg', source_dir / 'broken.jpg']
        result = predict_images_batch(
            object(),
            image_paths=image_paths,
            source=source_dir,
            conf=0.25,
            iou=0.45,
            resolved_output_dir=output_dir,
            save_annotated=True,
            save_labels=True,
            save_original=True,
            generate_report=True,
            batch_size=2,
            truncated=False,
            draw_detections_fn=_draw,
            read_image_fn=_read,
            run_batch_inference_fn=_run,
            write_yolo_txt_fn=_write,
        )
        assert result['ok'] is True, result
        assert result['processed_images'] == 3, result
        assert result['detected_images'] == 2, result
        assert result['empty_images'] == 1, result
        assert len(result['failed_reads']) == 1, result
        assert result['class_counts']['bulldozer'] == 2, result
        assert result['class_counts']['Excavator'] == 1, result
        assert Path(result['annotated_dir']).exists(), result
        assert Path(result['labels_dir']).exists(), result
        assert Path(result['originals_dir']).exists(), result
        report = Path(result['report_path'])
        assert report.exists(), result
        payload = json.loads(report.read_text(encoding='utf-8'))
        assert payload['processed_images'] == 3
        assert payload['class_counts']['bulldozer'] == 2
        print('prediction image helpers ok')
    finally:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
