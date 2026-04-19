from __future__ import annotations

from typing import Any, Mapping

try:
    from langgraph.types import Command, interrupt
except Exception:
    from langgraph.types import Command

    def interrupt(value: Any) -> Any:
        return value

from yolostudio_agent.agent.client.training_schemas import (
    PendingTurnIntent,
    coerce_training_plan,
    merge_training_plan_edits,
    update_plan_after_prepare,
)


def _state_value(state: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(key, default)
    return default


def _normalize_decision(value: Any) -> PendingTurnIntent:
    if isinstance(value, PendingTurnIntent):
        return value
    if isinstance(value, Mapping):
        return PendingTurnIntent.model_validate(dict(value))
    return PendingTurnIntent(action='unclear', reason='')


def training_confirmation_node(state: Mapping[str, Any]) -> Any:
    plan = coerce_training_plan(_state_value(state, 'training_plan', {}))
    phase = str(_state_value(state, 'training_phase', 'prepare') or 'prepare').strip().lower() or 'prepare'
    suspended_plan = _state_value(state, 'suspended_training_plan')
    payload = {
        'type': 'training_confirmation',
        'phase': phase,
        'plan': plan.model_dump(),
        'suspended_training_plan': dict(suspended_plan or {}) if isinstance(suspended_plan, Mapping) else None,
    }
    decision = _normalize_decision(interrupt(payload))
    action = str(decision.action or 'unclear').strip().lower()

    if action == 'approve':
        return Command(goto='execute_prepare' if phase == 'prepare' else 'execute_training')

    if action == 'reject':
        return Command(update={
            'training_plan': None,
            'training_phase': None,
            'suspended_training_plan': None,
            'pending_new_task': '',
            'training_status_reply': '',
            'status': 'cancelled',
        })

    if action == 'edit':
        updated = merge_training_plan_edits(plan, decision.edits)
        return Command(update={
            'training_plan': updated.model_dump(),
            'training_phase': phase,
            'training_status_reply': '',
        }, goto='training_confirmation')

    if action == 'new_task':
        return Command(update={
            'suspended_training_plan': plan.model_dump(),
            'pending_new_task': str(decision.reason or '').strip(),
            'training_phase': phase,
            'training_status_reply': '',
        }, goto='route_new_task')

    if action == 'status':
        return Command(update={
            'training_plan': plan.model_dump(),
            'training_phase': phase,
            'training_status_reply': str(decision.reason or '').strip(),
        }, goto='answer_training_status')

    return Command(update={
        'training_plan': plan.model_dump(),
        'training_phase': phase,
    }, goto='training_confirmation')


def post_prepare_node(state: Mapping[str, Any]) -> Command:
    plan = coerce_training_plan(_state_value(state, 'training_plan', {}))
    updated = update_plan_after_prepare(
        plan,
        prepare_result=_state_value(state, 'prepare_result', {}),
        readiness=_state_value(state, 'training_readiness', {}),
    )
    return Command(update={
        'training_plan': updated.model_dump(),
        'training_phase': 'start',
        'training_status_reply': '',
    }, goto='training_confirmation')


def answer_training_status_node(state: Mapping[str, Any]) -> Command:
    plan = coerce_training_plan(_state_value(state, 'training_plan', {}))
    phase = str(_state_value(state, 'training_phase', 'prepare') or 'prepare').strip().lower() or 'prepare'
    reply = str(_state_value(state, 'training_status_reply', '') or '').strip()
    return Command(update={
        'training_plan': plan.model_dump(),
        'training_phase': phase,
        'training_status_reply': reply,
    }, goto='training_confirmation')
