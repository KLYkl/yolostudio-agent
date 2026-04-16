from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_projectors.common import _realtime_snapshot


def apply_realtime_prediction_tool_result(
    session_state: SessionState,
    tool_name: str,
    result: dict[str, Any],
    tool_args: dict[str, Any],
) -> None:
    pred = session_state.active_prediction
    if tool_name == 'scan_cameras' and result.get('ok'):
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='camera_overview',
            extra_keys=('camera_count', 'cameras'),
        )
    elif tool_name == 'scan_screens' and result.get('ok'):
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='screen_overview',
            extra_keys=('screen_count', 'screens'),
        )
    elif tool_name == 'test_rtsp_stream':
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='stream_test_overview',
            extra_keys=('rtsp_url', 'ok', 'error'),
        )
    elif tool_name in {'start_camera_prediction', 'start_rtsp_prediction', 'start_screen_prediction'} and result.get('ok'):
        pred.model = str(tool_args.get('model') or pred.model)
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.realtime_session_id = str(result.get('session_id') or pred.realtime_session_id)
        pred.realtime_source_type = str(result.get('source_type') or pred.realtime_source_type)
        pred.realtime_source_label = str(result.get('source_label') or pred.realtime_source_label)
        pred.realtime_status = 'running'
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='realtime_session_overview',
            extra_keys=('session_id', 'source_type', 'source_label', 'output_dir', 'status'),
        )
        pred.last_realtime_status['status'] = 'running'
    elif tool_name == 'check_realtime_prediction_status' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.realtime_session_id = str(result.get('session_id') or pred.realtime_session_id)
        pred.realtime_source_type = str(result.get('source_type') or pred.realtime_source_type)
        pred.realtime_source_label = str(result.get('source_label') or pred.realtime_source_label)
        pred.realtime_status = str(result.get('status') or pred.realtime_status)
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='realtime_status_overview',
            extra_keys=(
                'session_id',
                'source_type',
                'source_label',
                'status',
                'processed_frames',
                'detected_frames',
                'total_detections',
                'class_counts',
                'output_dir',
                'report_path',
                'error',
            ),
        )
    elif tool_name == 'stop_realtime_prediction' and result.get('ok'):
        pred.output_dir = str(result.get('output_dir') or pred.output_dir)
        pred.report_path = str(result.get('report_path') or pred.report_path)
        pred.realtime_session_id = str(result.get('session_id') or pred.realtime_session_id)
        pred.realtime_source_type = str(result.get('source_type') or pred.realtime_source_type)
        pred.realtime_source_label = str(result.get('source_label') or pred.realtime_source_label)
        pred.realtime_status = str(result.get('status') or 'stopped')
        pred.last_realtime_status = _realtime_snapshot(
            result,
            overview_key='realtime_status_overview',
            extra_keys=(
                'session_id',
                'source_type',
                'source_label',
                'status',
                'processed_frames',
                'detected_frames',
                'total_detections',
                'class_counts',
                'output_dir',
                'report_path',
                'error',
            ),
        )
