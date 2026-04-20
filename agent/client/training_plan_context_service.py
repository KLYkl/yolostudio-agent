from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState

TRAINING_PLAN_CONTEXT_KEY = 'training_plan_context'


def build_training_plan_context_from_draft(draft: dict[str, Any] | None) -> dict[str, Any] | None:
    draft = dict(draft or {})
    if not draft:
        return None

    payload = {
        'stage': str(draft.get('stage') or ''),
        'status': str(draft.get('status') or ''),
        'dataset_path': str(draft.get('dataset_path') or ''),
        'execution_mode': str(draft.get('execution_mode') or ''),
        'execution_backend': str(draft.get('execution_backend') or ''),
        'training_environment': str(draft.get('training_environment') or ''),
        'advanced_details_requested': bool(draft.get('advanced_details_requested')),
        'reasoning_summary': str(draft.get('reasoning_summary') or ''),
        'data_summary': str(draft.get('data_summary') or ''),
        'preflight_summary': str(draft.get('preflight_summary') or ''),
        'next_step_tool': str(draft.get('next_step_tool') or ''),
        'next_step_args': dict(draft.get('next_step_args') or {}),
        'planned_training_args': dict(draft.get('planned_training_args') or {}),
        'command_preview': list(draft.get('command_preview') or []),
        'blockers': [str(item).strip() for item in (draft.get('blockers') or []) if str(item).strip()],
        'warnings': [str(item).strip() for item in (draft.get('warnings') or []) if str(item).strip()],
        'risks': [str(item).strip() for item in (draft.get('risks') or []) if str(item).strip()],
    }
    return payload


def build_training_plan_context_payload(state: SessionState) -> dict[str, Any] | None:
    return build_training_plan_context_from_draft(dict(state.active_training.training_plan_draft or {}))


def extract_training_plan_context_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    payload = state.get(TRAINING_PLAN_CONTEXT_KEY)
    if isinstance(payload, dict):
        return dict(payload)
    return None
