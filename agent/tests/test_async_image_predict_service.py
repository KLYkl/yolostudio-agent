from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from PIL import Image

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.predict_service import PredictService


TMP_ROOT = Path('C:/workspace/yolodo2.0/agent_plan/.tmp_async_image_predict_service')


def _make_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path, format='JPEG')


def _wait_for_status(service: PredictService, session_id: str, predicate, *, timeout: float = 4.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = service.check_image_prediction_status(session_id=session_id)
        if predicate(last):
            return last
        time.sleep(0.02)
    raise AssertionError(f'image prediction session {session_id} did not reach expected state, last={last}')


def main() -> None:
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    source_dir = TMP_ROOT / 'images'
    output_root = TMP_ROOT / 'predict_out'
    output_root.mkdir(parents=True, exist_ok=True)
    for idx, size in enumerate((32, 40, 48, 56, 64, 72), start=1):
        _make_image(source_dir / f'{idx}.jpg', (size, size), (idx * 20 % 255, 50, 100))

    service = PredictService(output_root)
    service._background_image_threshold = 2
    original_load = service._load_model
    original_run = service._run_batch_inference
    original_draw = service._draw_detections
    try:
        service._load_model = lambda model: {'model': model}

        def _fake_run(_model, frames, *, conf: float, iou: float):
            del conf, iou
            time.sleep(0.05)
            outputs = []
            for frame in frames:
                width = int(frame.size[0])
                if width % 3 == 0:
                    outputs.append([])
                else:
                    outputs.append([
                        {'class_id': 0, 'class_name': 'Excavator', 'confidence': 0.91, 'xyxy': [1, 1, width - 2, width - 2]},
                    ])
            return outputs

        service._run_batch_inference = _fake_run
        service._draw_detections = lambda frame, detections: frame

        started = service.predict_images(
            source_path=str(source_dir),
            model='fake-model.pt',
            output_dir=str(output_root / 'auto-background'),
        )
        assert started['ok'] is True, started
        assert started['started_in_background'] is True, started
        session_id = started['session_id']

        completed = _wait_for_status(
            service,
            session_id,
            lambda status: status.get('ok') and status.get('status') == 'completed',
        )
        assert completed['processed_images'] == 6, completed
        assert completed['total_images'] == 6, completed
        assert completed['detected_images'] >= 1, completed
        assert Path(completed['report_path']).exists(), completed

        stop_started = service.start_image_prediction(
            source_path=str(source_dir),
            model='fake-model.pt',
            output_dir=str(output_root / 'manual-background'),
            batch_size=1,
        )
        assert stop_started['ok'] is True, stop_started
        stop_session_id = stop_started['session_id']
        _wait_for_status(
            service,
            stop_session_id,
            lambda status: status.get('ok') and status.get('processed_images', 0) >= 1,
        )
        stopped = service.stop_image_prediction(session_id=stop_session_id)
        assert stopped['ok'] is True, stopped
        assert stopped['status'] in {'stopped', 'completed'}, stopped
        assert stopped['processed_images'] >= 1, stopped
        assert Path(stopped['report_path']).exists(), stopped

        print('async image predict service ok')
    finally:
        service._load_model = original_load
        service._run_batch_inference = original_run
        service._draw_detections = original_draw
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
