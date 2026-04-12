from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable


def predict_single_video(
    predictor: Any,
    *,
    video_path: Path,
    output_root: Path,
    conf: float,
    iou: float,
    save_video: bool,
    save_keyframes_annotated: bool,
    save_keyframes_raw: bool,
    max_frames: int,
    cv2_module: Any,
    image_fromarray: Callable[[Any], Any],
    run_batch_inference_fn: Callable[..., list[list[dict[str, Any]]]],
    draw_detections_fn: Callable[[Any, list[dict[str, Any]]], Any],
    pil_to_bgr_fn: Callable[[Any], Any],
) -> dict[str, Any]:
    cap = cv2_module.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return {'ok': False, 'error': '无法打开视频'}

    fps = cap.get(cv2_module.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2_module.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2_module.CAP_PROP_FRAME_HEIGHT)) or 0
    video_dir = output_root / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = video_dir / 'keyframes' / 'annotated'
    raw_dir = video_dir / 'keyframes' / 'raw'
    if save_keyframes_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    if save_keyframes_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    writer: Any | None = None
    video_output_path = ''
    if save_video and width > 0 and height > 0:
        video_output_path = str((video_dir / f'{video_path.stem}_result.mp4').resolve())
        writer = cv2_module.VideoWriter(
            video_output_path,
            cv2_module.VideoWriter_fourcc(*'mp4v'),
            float(fps or 30.0),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            writer = None
            video_output_path = ''

    processed_frames = 0
    detected_frames = 0
    total_detections = 0
    class_counter: Counter[str] = Counter()
    frame_results: list[dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames > 0 and processed_frames >= max_frames:
            break
        processed_frames += 1

        pil = image_fromarray(cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB))
        detections = run_batch_inference_fn(predictor, [pil], conf=conf, iou=iou)[0]
        if detections:
            detected_frames += 1
            total_detections += len(detections)
            for det in detections:
                class_counter[str(det.get('class_name', 'unknown'))] += 1
            if save_keyframes_annotated:
                annotated_path = annotated_dir / f'frame_{processed_frames:06d}.jpg'
                draw_detections_fn(pil, detections).save(annotated_path)
            if save_keyframes_raw:
                raw_path = raw_dir / f'frame_{processed_frames:06d}.jpg'
                cv2_module.imwrite(str(raw_path), frame)
        if writer is not None:
            annotated_frame = pil_to_bgr_fn(draw_detections_fn(pil, detections))
            writer.write(annotated_frame)
        frame_results.append({
            'frame_index': processed_frames,
            'detections': len(detections),
            'classes': sorted({str(det.get('class_name', 'unknown')) for det in detections}),
        })

    cap.release()
    if writer is not None:
        writer.release()

    return {
        'ok': True,
        'video_path': str(video_path.resolve()),
        'output_dir': str(video_dir.resolve()),
        'annotated_video': video_output_path,
        'annotated_keyframes_dir': str(annotated_dir.resolve()) if save_keyframes_annotated else '',
        'raw_keyframes_dir': str(raw_dir.resolve()) if save_keyframes_raw else '',
        'processed_frames': processed_frames,
        'detected_frames': detected_frames,
        'total_detections': total_detections,
        'class_counts': dict(sorted(class_counter.items(), key=lambda item: (-item[1], item[0]))),
        'frame_results': frame_results[:50],
    }
