from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.tool_adapter import canonical_tool_name

CACHED_TOOL_SNAPSHOT_PREFIX = 'CACHED_TOOL_SNAPSHOT='


def build_cached_tool_snapshot_payload(state: SessionState) -> dict[str, Any] | None:
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


def build_cached_tool_snapshot_message(state: SessionState) -> str | None:
    payload = build_cached_tool_snapshot_payload(state)
    if not payload:
        return None
    return f'{CACHED_TOOL_SNAPSHOT_PREFIX}{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}'


def extract_cached_tool_snapshot(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        content = getattr(message, 'content', '')
        if not isinstance(content, str) or not content.startswith(CACHED_TOOL_SNAPSHOT_PREFIX):
            continue
        raw = content[len(CACHED_TOOL_SNAPSHOT_PREFIX):].strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
        return None
    return None


def _last_pending_tool_call(messages: list[Any]) -> tuple[str, dict[str, Any]] | None:
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or not getattr(message, 'tool_calls', None):
            continue
        tool_call = message.tool_calls[0]
        return canonical_tool_name(tool_call.get('name') or ''), dict(tool_call.get('args') or {})
    return None


def _args_empty(args: dict[str, Any]) -> bool:
    for value in args.values():
        if value not in (None, '', [], {}, ()):
            return False
    return True


def _match_scalar_targets(
    payload: dict[str, Any],
    args: dict[str, Any],
    *,
    target_keys: tuple[str, ...],
) -> bool:
    for key in target_keys:
        expected = str(args.get(key) or '').strip()
        if not expected:
            continue
        actual = str(payload.get(key) or '').strip()
        if not actual or actual != expected:
            return False
    return True


def _match_run_inspection(payload: dict[str, Any], args: dict[str, Any]) -> bool:
    requested_run_id = str(args.get('run_id') or '').strip()
    if not requested_run_id:
        return True
    selected_run_id = str(payload.get('selected_run_id') or payload.get('run_id') or '').strip()
    return not selected_run_id or selected_run_id == requested_run_id


def _match_run_comparison(payload: dict[str, Any], args: dict[str, Any]) -> bool:
    expected_left = str(args.get('left_run_id') or '').strip()
    expected_right = str(args.get('right_run_id') or '').strip()
    if not expected_left and not expected_right:
        return True
    cached_left = str(payload.get('left_run_id') or '').strip()
    cached_right = str(payload.get('right_run_id') or '').strip()
    left_ok = not expected_left or not cached_left or cached_left == expected_left
    right_ok = not expected_right or not cached_right or cached_right == expected_right
    return left_ok and right_ok


def _match_loop_payload(payload: dict[str, Any], args: dict[str, Any]) -> bool:
    requested_loop_id = str(args.get('loop_id') or '').strip()
    if not requested_loop_id:
        return True
    cached_loop_id = str(payload.get('loop_id') or '').strip()
    return not cached_loop_id or cached_loop_id == requested_loop_id


def _normalize_signal_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _match_knowledge_payload(payload: dict[str, Any], args: dict[str, Any]) -> bool:
    for key in ('topic', 'stage', 'model_family', 'task_type'):
        expected = str(args.get(key) or '').strip()
        if not expected:
            continue
        actual = str(payload.get(key) or '').strip()
        if actual and actual != expected:
            return False
    expected_signals = _normalize_signal_list(args.get('signals'))
    if not expected_signals:
        return True
    cached_signals = _normalize_signal_list(payload.get('signals'))
    return not cached_signals or cached_signals == expected_signals


def resolve_cached_tool_reply(messages: list[Any]) -> tuple[str, dict[str, Any]] | None:
    snapshot = extract_cached_tool_snapshot(messages)
    if not snapshot:
        return None
    pending_tool_call = _last_pending_tool_call(messages)
    if not pending_tool_call:
        return None
    tool_name, args = pending_tool_call
    payload = snapshot.get(tool_name)
    if not isinstance(payload, dict):
        return None

    if tool_name in {
        'run_dataset_health_check',
        'detect_duplicate_images',
        'summarize_training_run',
        'list_training_runs',
        'select_best_training_run',
        'list_training_loops',
        'list_remote_profiles',
        'upload_assets_to_remote',
        'download_assets_from_remote',
        'remote_training_pipeline',
        'remote_prediction_pipeline',
        'check_training_status',
        'analyze_training_outcome',
        'recommend_next_training_step',
    }:
        return (tool_name, payload) if _args_empty(args) else None
    if tool_name in {'preview_extract_images', 'extract_images'}:
        return (
            tool_name,
            payload,
        ) if _match_scalar_targets(payload, args, target_keys=('source_path', 'output_dir')) else None
    if tool_name == 'scan_videos':
        return (tool_name, payload) if _match_scalar_targets(payload, args, target_keys=('source_path',)) else None
    if tool_name == 'extract_video_frames':
        return (
            tool_name,
            payload,
        ) if _match_scalar_targets(payload, args, target_keys=('source_path', 'output_dir')) else None
    if tool_name == 'inspect_training_run':
        return (tool_name, payload) if _match_run_inspection(payload, args) else None
    if tool_name == 'compare_training_runs':
        return (tool_name, payload) if _match_run_comparison(payload, args) else None
    if tool_name in {'check_training_loop_status', 'inspect_training_loop'}:
        return (tool_name, payload) if _match_loop_payload(payload, args) else None
    if tool_name == 'retrieve_training_knowledge':
        return (tool_name, payload) if _match_knowledge_payload(payload, args) else None
    if tool_name in {
        'summarize_prediction_results',
        'inspect_prediction_outputs',
        'export_prediction_report',
        'export_prediction_path_lists',
        'organize_prediction_results',
    }:
        return (
            tool_name,
            payload,
        ) if _match_scalar_targets(payload, args, target_keys=('report_path', 'output_dir', 'export_path', 'export_dir', 'destination_dir')) else None
    if tool_name == 'check_image_prediction_status':
        return (
            tool_name,
            payload,
        ) if _match_scalar_targets(payload, args, target_keys=('session_id', 'image_prediction_session_id')) else None
    if tool_name == 'check_realtime_prediction_status':
        return (
            tool_name,
            payload,
        ) if _match_scalar_targets(payload, args, target_keys=('session_id', 'realtime_session_id')) else None
    return None
