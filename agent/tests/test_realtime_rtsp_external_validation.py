from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.predict_service import PredictService


def _snapshot_for_log(result: dict[str, Any]) -> dict[str, Any]:
    return {
        'timestamp': time.time(),
        'ok': bool(result.get('ok')),
        'status': str(result.get('status') or ''),
        'running': bool(result.get('running')),
        'processed_frames': int(result.get('processed_frames') or 0),
        'detected_frames': int(result.get('detected_frames') or 0),
        'total_detections': int(result.get('total_detections') or 0),
        'capture_opened_at': result.get('capture_opened_at'),
        'last_frame_at': result.get('last_frame_at'),
        'error': str(result.get('error') or ''),
        'summary': str(result.get('summary') or ''),
    }


def _assess(
    probe_result: dict[str, Any],
    start_result: dict[str, Any],
    final_result: dict[str, Any],
) -> tuple[str, str]:
    if not probe_result.get('ok'):
        return 'probe_failed', 'RTSP 探测未通过'
    if not start_result.get('ok'):
        return 'start_failed', 'RTSP 实时预测未启动'
    if not final_result.get('ok'):
        return 'status_failed', 'RTSP 实时预测状态获取失败'
    if str(final_result.get('status') or '') == 'error':
        return 'session_error', 'RTSP 会话异常结束'
    if int(final_result.get('processed_frames') or 0) > 0:
        return 'frames_progressed', 'RTSP 会话已推进到实际处理帧'
    if final_result.get('capture_opened_at'):
        return 'waiting_for_first_keyframe', 'RTSP 会话已连上输入源，但仍在等待首帧 / 关键帧'
    return 'no_frames_observed', 'RTSP 会话未观察到帧推进'


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate an external RTSP source with realtime prediction service.')
    parser.add_argument('--rtsp-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--timeout-ms', type=int, default=5000)
    parser.add_argument('--frame-interval-ms', type=int, default=120)
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--wait-seconds', type=float, default=20.0)
    parser.add_argument('--poll-interval-seconds', type=float, default=0.5)
    parser.add_argument('--keep-running', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    service = PredictService(output_root=output_dir)

    probe_result = service.test_rtsp_stream(rtsp_url=args.rtsp_url, timeout_ms=args.timeout_ms)
    start_result: dict[str, Any] = {}
    final_result: dict[str, Any] = {}
    stop_result: dict[str, Any] = {}
    status_history: list[dict[str, Any]] = []

    if probe_result.get('ok'):
        start_result = service.start_rtsp_prediction(
            model=args.model,
            rtsp_url=args.rtsp_url,
            frame_interval_ms=args.frame_interval_ms,
            max_frames=args.max_frames,
        )

    if start_result.get('ok'):
        session_id = str(start_result.get('session_id') or '')
        deadline = time.time() + max(float(args.wait_seconds), 1.0)
        while time.time() < deadline:
            final_result = service.check_realtime_prediction_status(session_id=session_id)
            status_history.append(_snapshot_for_log(final_result))
            if str(final_result.get('status') or '') in {'completed', 'stopped', 'error'} and not final_result.get('running'):
                break
            if int(final_result.get('processed_frames') or 0) >= max(1, int(args.max_frames or 0)):
                break
            time.sleep(max(float(args.poll_interval_seconds), 0.1))

        if not args.keep_running:
            stop_result = service.stop_realtime_prediction(session_id=session_id)
            if stop_result.get('ok'):
                final_result = stop_result
                status_history.append(_snapshot_for_log(stop_result))

    assessment, assessment_summary = _assess(probe_result, start_result, final_result)
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'rtsp_url': args.rtsp_url,
        'model': args.model,
        'output_dir': str(output_dir.resolve()),
        'timeout_ms': int(args.timeout_ms),
        'frame_interval_ms': int(args.frame_interval_ms),
        'max_frames': int(args.max_frames),
        'wait_seconds': float(args.wait_seconds),
        'poll_interval_seconds': float(args.poll_interval_seconds),
        'keep_running': bool(args.keep_running),
        'probe_result': probe_result,
        'start_result': start_result,
        'final_result': final_result,
        'stop_result': stop_result,
        'status_history': status_history,
        'assessment': assessment,
        'assessment_summary': assessment_summary,
    }
    report_path = output_dir / 'external_rtsp_validation.json'
    payload['report_path'] = str(report_path.resolve())
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report_path)


if __name__ == '__main__':
    main()
