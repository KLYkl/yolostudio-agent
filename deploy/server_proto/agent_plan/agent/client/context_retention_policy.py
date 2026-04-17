from __future__ import annotations

from dataclasses import dataclass

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.session_state import SessionState


FOLLOWUP_MARKERS = (
    '刚才',
    '上次',
    '那个',
    '这次',
    '现在',
    '当前',
    '详细一点',
    '再详细',
    '再展开',
    '再概括',
    '再解释',
    '总结',
    '摘要',
    '情况',
    '状态',
    '结果',
    '详情',
    '信息',
    'why',
    'summary',
    'status',
    'details',
)


@dataclass(frozen=True)
class ContextRetentionDecision:
    reuse_history: bool
    reason: str


def _has_cached_followup_context(state: SessionState) -> bool:
    dataset = state.active_dataset
    training = state.active_training
    prediction = state.active_prediction
    knowledge = state.active_knowledge
    remote = state.active_remote_transfer
    return any(
        (
            dataset.last_scan,
            dataset.last_validate,
            dataset.last_health_check,
            dataset.last_duplicate_check,
            dataset.last_extract_preview,
            dataset.last_extract_result,
            dataset.last_video_scan,
            dataset.last_frame_extract,
            training.last_status,
            training.last_summary,
            training.training_run_summary,
            training.last_run_inspection,
            training.last_run_comparison,
            training.best_run_selection,
            training.recent_runs,
            training.last_loop_status,
            training.last_loop_detail,
            training.recent_loops,
            prediction.last_result,
            prediction.last_summary,
            prediction.last_inspection,
            prediction.last_export,
            prediction.last_path_lists,
            prediction.last_organized_result,
            prediction.last_realtime_status,
            prediction.last_remote_roundtrip,
            knowledge.last_retrieval,
            knowledge.last_analysis,
            knowledge.last_recommendation,
            remote.last_profile_listing,
            remote.last_upload,
            remote.last_download,
        )
    )


def _has_active_workflow_context(state: SessionState) -> bool:
    training = state.active_training
    prediction = state.active_prediction
    return any(
        (
            str(training.workflow_state or '').strip().lower() not in {'', 'idle'},
            str(training.loop_workflow_state or '').strip().lower() not in {'', 'loop_idle'},
            training.running,
            str(training.active_loop_id or '').strip(),
            str(prediction.image_prediction_status or '').strip().lower() in {'queued', 'running', 'stopping'},
            str(prediction.realtime_status or '').strip().lower() in {'starting', 'running', 'stopping'},
        )
    )


def _has_new_task_targets(user_text: str) -> bool:
    text = str(user_text or '')
    return any(
        (
            intent_parsing.extract_all_paths_from_text(text),
            intent_parsing.extract_model_from_text(text),
            intent_parsing.extract_remote_server_from_text(text),
            intent_parsing.extract_remote_root_from_text(text),
            intent_parsing.extract_rtsp_url_from_text(text),
            intent_parsing.extract_realtime_session_id_from_text(text),
        )
    )


def _looks_like_followup_prompt(user_text: str) -> bool:
    text = str(user_text or '').strip()
    lowered = text.lower()
    if not text:
        return False
    if any(marker in text or marker in lowered for marker in FOLLOWUP_MARKERS):
        return True
    return len(text) <= 24 and ('?' in text or '？' in text)


def build_context_retention_decision(
    *,
    state: SessionState,
    user_text: str,
    explicitly_references_previous_context: bool,
) -> ContextRetentionDecision:
    if explicitly_references_previous_context:
        return ContextRetentionDecision(True, 'explicit_reference')
    if str(state.pending_confirmation.tool_name or '').strip():
        return ContextRetentionDecision(True, 'pending_confirmation')
    if state.active_training.training_plan_draft:
        return ContextRetentionDecision(True, 'training_plan_draft')
    if _has_active_workflow_context(state):
        return ContextRetentionDecision(True, 'active_workflow')
    if _has_cached_followup_context(state) and not _has_new_task_targets(user_text) and _looks_like_followup_prompt(user_text):
        return ContextRetentionDecision(True, 'cached_followup_context')
    return ContextRetentionDecision(False, 'strip_ephemeral_context')
