from __future__ import annotations

from typing import Any

TRAINING_PLAN_CONTEXT_KEY = 'training_plan_context'

_TRAINING_PLAN_METADATA_KEYS = (
    'source_intent',
    'planner_user_request',
    'planner_decision_source',
    'planner_decision',
    'planner_output',
    'planner_observed_tools',
)


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
    for key in _TRAINING_PLAN_METADATA_KEYS:
        value = draft.get(key)
        if value in (None, '', [], {}):
            continue
        payload[key] = list(value) if isinstance(value, list) else value
    return payload


def build_training_plan_draft_from_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    context = dict(context or {})
    if not context:
        return None

    draft = {
        'stage': str(context.get('stage') or ''),
        'status': str(context.get('status') or ''),
        'dataset_path': str(context.get('dataset_path') or ''),
        'execution_mode': str(context.get('execution_mode') or ''),
        'execution_backend': str(context.get('execution_backend') or ''),
        'training_environment': str(context.get('training_environment') or ''),
        'advanced_details_requested': bool(context.get('advanced_details_requested')),
        'reasoning_summary': str(context.get('reasoning_summary') or ''),
        'data_summary': str(context.get('data_summary') or ''),
        'preflight_summary': str(context.get('preflight_summary') or ''),
        'next_step_tool': str(context.get('next_step_tool') or ''),
        'next_step_args': dict(context.get('next_step_args') or {}),
        'planned_training_args': dict(context.get('planned_training_args') or {}),
        'command_preview': list(context.get('command_preview') or []),
        'blockers': [str(item).strip() for item in (context.get('blockers') or []) if str(item).strip()],
        'warnings': [str(item).strip() for item in (context.get('warnings') or []) if str(item).strip()],
        'risks': [str(item).strip() for item in (context.get('risks') or []) if str(item).strip()],
    }
    for key in _TRAINING_PLAN_METADATA_KEYS:
        value = context.get(key)
        if value in (None, '', [], {}):
            continue
        draft[key] = list(value) if isinstance(value, list) else value
    if not any(draft.get(key) for key in (
        'dataset_path',
        'execution_mode',
        'reasoning_summary',
        'data_summary',
        'preflight_summary',
        'planned_training_args',
        'command_preview',
        'warnings',
        'risks',
        'blockers',
        'next_step_tool',
        *_TRAINING_PLAN_METADATA_KEYS,
    )):
        return None
    return draft


def build_training_plan_context_payload(source: Any) -> dict[str, Any] | None:
    if isinstance(source, dict):
        return build_training_plan_context_from_draft(source)
    active_training = getattr(source, 'active_training', None)
    draft = getattr(active_training, 'training_plan_draft', None)
    if isinstance(draft, dict):
        return build_training_plan_context_from_draft(draft)
    return None


def extract_training_plan_context_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    payload = state.get(TRAINING_PLAN_CONTEXT_KEY)
    if isinstance(payload, dict):
        return dict(payload)
    return None
