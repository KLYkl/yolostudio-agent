from __future__ import annotations

from typing import Callable

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_projectors.dataset import apply_dataset_tool_result
from yolostudio_agent.agent.client.state_projectors.knowledge import apply_knowledge_tool_result
from yolostudio_agent.agent.client.state_projectors.prediction import apply_prediction_tool_result
from yolostudio_agent.agent.client.state_projectors.realtime import apply_realtime_prediction_tool_result
from yolostudio_agent.agent.client.state_projectors.remote import apply_remote_transfer_tool_result
from yolostudio_agent.agent.client.state_projectors.training import apply_training_tool_result


StateProjector = Callable[[SessionState, str, dict, dict], None]


STATEFUL_TOOL_PROJECTORS: dict[str, StateProjector] = {
    'scan_dataset': apply_dataset_tool_result,
    'validate_dataset': apply_dataset_tool_result,
    'run_dataset_health_check': apply_dataset_tool_result,
    'detect_duplicate_images': apply_dataset_tool_result,
    'preview_extract_images': apply_dataset_tool_result,
    'extract_images': apply_dataset_tool_result,
    'scan_videos': apply_dataset_tool_result,
    'extract_video_frames': apply_dataset_tool_result,
    'split_dataset': apply_dataset_tool_result,
    'generate_yaml': apply_dataset_tool_result,
    'training_readiness': apply_dataset_tool_result,
    'dataset_training_readiness': apply_dataset_tool_result,
    'prepare_dataset_for_training': apply_dataset_tool_result,
    'start_training': apply_training_tool_result,
    'list_training_environments': apply_training_tool_result,
    'training_preflight': apply_training_tool_result,
    'list_training_runs': apply_training_tool_result,
    'inspect_training_run': apply_training_tool_result,
    'compare_training_runs': apply_training_tool_result,
    'select_best_training_run': apply_training_tool_result,
    'start_training_loop': apply_training_tool_result,
    'list_training_loops': apply_training_tool_result,
    'check_training_loop_status': apply_training_tool_result,
    'inspect_training_loop': apply_training_tool_result,
    'pause_training_loop': apply_training_tool_result,
    'resume_training_loop': apply_training_tool_result,
    'stop_training_loop': apply_training_tool_result,
    'check_training_status': apply_training_tool_result,
    'summarize_training_run': apply_training_tool_result,
    'stop_training': apply_training_tool_result,
    'predict_images': apply_prediction_tool_result,
    'start_image_prediction': apply_prediction_tool_result,
    'check_image_prediction_status': apply_prediction_tool_result,
    'stop_image_prediction': apply_prediction_tool_result,
    'predict_videos': apply_prediction_tool_result,
    'summarize_prediction_results': apply_prediction_tool_result,
    'inspect_prediction_outputs': apply_prediction_tool_result,
    'export_prediction_report': apply_prediction_tool_result,
    'export_prediction_path_lists': apply_prediction_tool_result,
    'organize_prediction_results': apply_prediction_tool_result,
    'scan_cameras': apply_realtime_prediction_tool_result,
    'scan_screens': apply_realtime_prediction_tool_result,
    'test_rtsp_stream': apply_realtime_prediction_tool_result,
    'start_camera_prediction': apply_realtime_prediction_tool_result,
    'start_rtsp_prediction': apply_realtime_prediction_tool_result,
    'start_screen_prediction': apply_realtime_prediction_tool_result,
    'check_realtime_prediction_status': apply_realtime_prediction_tool_result,
    'stop_realtime_prediction': apply_realtime_prediction_tool_result,
    'retrieve_training_knowledge': apply_knowledge_tool_result,
    'analyze_training_outcome': apply_knowledge_tool_result,
    'recommend_next_training_step': apply_knowledge_tool_result,
    'list_remote_profiles': apply_remote_transfer_tool_result,
    'upload_assets_to_remote': apply_remote_transfer_tool_result,
    'download_assets_from_remote': apply_remote_transfer_tool_result,
}


CRITICAL_STATEFUL_TOOLS = {
    'prepare_dataset_for_training',
    'start_training',
    'start_training_loop',
    'check_training_status',
    'stop_training',
    'predict_images',
    'start_image_prediction',
    'upload_assets_to_remote',
}


def apply_tool_result_to_state(
    session_state: SessionState,
    tool_name: str,
    result: dict,
    tool_args: dict | None = None,
) -> None:
    projector = STATEFUL_TOOL_PROJECTORS.get(str(tool_name or '').strip())
    if projector is None:
        return
    projector(session_state, str(tool_name or '').strip(), result, dict(tool_args or {}))
