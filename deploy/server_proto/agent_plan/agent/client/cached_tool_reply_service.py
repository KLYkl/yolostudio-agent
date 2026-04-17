from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState

CACHED_TOOL_CONTEXT_KEY = 'cached_tool_context'


def build_cached_tool_context_payload(state: SessionState) -> dict[str, Any] | None:
    dataset = state.active_dataset
    training = state.active_training
    remote_transfer = state.active_remote_transfer
    prediction = state.active_prediction
    knowledge = state.active_knowledge
    payload: dict[str, Any] = {}

    if dataset.last_health_check:
        payload['run_dataset_health_check'] = dict(dataset.last_health_check)
    if dataset.last_duplicate_check:
        payload['detect_duplicate_images'] = dict(dataset.last_duplicate_check)
    if dataset.last_extract_preview:
        payload['preview_extract_images'] = dict(dataset.last_extract_preview)
    if dataset.last_extract_result:
        payload['extract_images'] = dict(dataset.last_extract_result)
    if dataset.last_video_scan:
        payload['scan_videos'] = dict(dataset.last_video_scan)
    if dataset.last_frame_extract:
        payload['extract_video_frames'] = dict(dataset.last_frame_extract)

    summary_payload = dict(prediction.last_summary or {})
    if not summary_payload:
        last_result = dict(prediction.last_result or {})
        if any(
            key in last_result
            for key in ('summary_overview', 'action_candidates', 'total_detections')
        ):
            summary_payload = last_result
    if summary_payload:
        payload['summarize_prediction_results'] = summary_payload
    if prediction.last_inspection:
        payload['inspect_prediction_outputs'] = dict(prediction.last_inspection)
    if prediction.last_export:
        payload['export_prediction_report'] = dict(prediction.last_export)
    if prediction.last_path_lists:
        payload['export_prediction_path_lists'] = dict(prediction.last_path_lists)
    if prediction.last_organized_result:
        payload['organize_prediction_results'] = dict(prediction.last_organized_result)
    image_status = str(
        (prediction.last_image_prediction_status or {}).get('status')
        or prediction.image_prediction_status
        or ''
    ).strip().lower()
    if prediction.last_image_prediction_status and image_status not in {'running', 'queued', 'pending', 'stopping'}:
        payload['check_image_prediction_status'] = dict(prediction.last_image_prediction_status)
    realtime_status = str(
        (prediction.last_realtime_status or {}).get('status')
        or prediction.realtime_status
        or ''
    ).strip().lower()
    if prediction.last_realtime_status and realtime_status not in {'running', 'starting', 'pending', 'stopping'}:
        payload['check_realtime_prediction_status'] = dict(prediction.last_realtime_status)

    training_run_state = str(
        (training.last_status or {}).get('run_state')
        or ((training.last_status or {}).get('status_overview') or {}).get('run_state')
        or ''
    ).strip().lower()
    training_summary = dict(training.training_run_summary or training.last_summary or {})
    if training_summary:
        payload['summarize_training_run'] = training_summary
    if training.last_status and not training.running and training_run_state != 'running':
        payload['check_training_status'] = dict(training.last_status)
    if training.recent_runs:
        payload['list_training_runs'] = {
            'ok': True,
            'summary': '训练历史查询完成',
            'runs': list(training.recent_runs),
        }
    if training.last_run_inspection:
        payload['inspect_training_run'] = dict(training.last_run_inspection)
    if training.last_run_comparison:
        payload['compare_training_runs'] = dict(training.last_run_comparison)
    if training.best_run_selection:
        payload['select_best_training_run'] = dict(training.best_run_selection)
    if training.recent_loops:
        payload['list_training_loops'] = {
            'ok': True,
            'summary': '环训练列表已就绪',
            'loops': list(training.recent_loops),
        }
    if training.last_loop_status:
        payload['check_training_loop_status'] = dict(training.last_loop_status)
    if training.last_loop_detail or training.last_loop_status:
        payload['inspect_training_loop'] = dict(training.last_loop_detail or training.last_loop_status)

    if knowledge.last_retrieval:
        payload['retrieve_training_knowledge'] = dict(knowledge.last_retrieval)
    if knowledge.last_analysis:
        payload['analyze_training_outcome'] = dict(knowledge.last_analysis)
    if knowledge.last_recommendation:
        payload['recommend_next_training_step'] = dict(knowledge.last_recommendation)

    if remote_transfer.last_profile_listing:
        payload['list_remote_profiles'] = dict(remote_transfer.last_profile_listing)
    if remote_transfer.last_upload:
        payload['upload_assets_to_remote'] = dict(remote_transfer.last_upload)
    if remote_transfer.last_download:
        payload['download_assets_from_remote'] = dict(remote_transfer.last_download)
    if training.last_remote_roundtrip:
        payload['remote_training_pipeline'] = dict(training.last_remote_roundtrip)
    if prediction.last_remote_roundtrip:
        payload['remote_prediction_pipeline'] = dict(prediction.last_remote_roundtrip)

    return payload or None


def extract_cached_tool_context_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    payload = state.get(CACHED_TOOL_CONTEXT_KEY)
    if isinstance(payload, dict):
        return dict(payload)
    return None
