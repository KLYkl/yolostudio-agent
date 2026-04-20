from typing import Any, Mapping

try:
    from langchain_core.runnables import RunnableConfig
except Exception:
    RunnableConfig = Any  # type: ignore[misc,assignment]

try:
    from langchain_core.runnables.config import set_config_context
except Exception:
    from contextlib import contextmanager

    @contextmanager
    def set_config_context(config: Any):
        del config
        yield None

try:
    from langgraph.types import Command, interrupt
except Exception:
    class Command:  # type: ignore[override]
        def __init__(self, *, update: Any = None, goto: str | None = None, resume: Any = None, graph: Any = None):
            self.update = update
            self.goto = goto
            self.resume = resume
            self.graph = graph

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


def training_confirmation_node(state: Mapping[str, Any], config: RunnableConfig | None = None) -> Any:
    plan = coerce_training_plan(_state_value(state, 'training_plan', {}))
    phase = str(_state_value(state, 'training_phase', 'prepare') or 'prepare').strip().lower() or 'prepare'
    suspended_plan = _state_value(state, 'suspended_training_plan')
    next_step_tool = str(_state_value(state, 'training_next_step_tool', '') or '').strip()
    next_step_args = dict(_state_value(state, 'training_next_step_args', {}) or {})
    execution_mode = str(_state_value(state, 'training_execution_mode', '') or '').strip()
    status_reply = str(_state_value(state, 'training_status_reply', '') or '').strip()
    payload = {
        'type': 'training_confirmation',
        'phase': phase,
        'execution_mode': execution_mode,
        'plan': plan.model_dump(),
        'next_step_tool': next_step_tool,
        'next_step_args': next_step_args,
        'status_reply': status_reply,
        'suspended_training_plan': dict(suspended_plan or {}) if isinstance(suspended_plan, Mapping) else None,
    }
    with set_config_context(config or {}):
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
        updated_args = dict(next_step_args)
        if phase == 'prepare':
            if updated.dataset_path:
                updated_args['dataset_path'] = updated.dataset_path
        else:
            updated_args = updated.model_dump()
            if next_step_tool == 'start_training_loop' and 'epochs_per_round' in updated_args and 'epochs' not in updated_args:
                updated_args['epochs'] = updated_args['epochs_per_round']
        return Command(update={
            'training_plan': updated.model_dump(),
            'training_phase': phase,
            'training_next_step_tool': next_step_tool,
            'training_next_step_args': updated_args,
            'training_execution_mode': execution_mode,
            'training_status_reply': '',
        }, goto='refresh_training_start_after_edit' if phase == 'start' else 'training_confirmation')

    if action == 'new_task':
        return Command(update={
            'suspended_training_plan': {
                'plan': plan.model_dump(),
                'phase': phase,
                'execution_mode': execution_mode,
                'next_step_tool': next_step_tool,
                'next_step_args': next_step_args,
            },
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
        readiness=_state_value(state, 'training_preflight', {}) or _state_value(state, 'training_readiness', {}),
    )
    resolved_args = dict(_state_value(state, 'training_preflight', {}) or {}).get('resolved_args') or updated.model_dump()
    next_tool_name = 'start_training_loop' if getattr(updated, 'mode', 'train') == 'loop' else 'start_training'
    if next_tool_name == 'start_training_loop' and 'epochs_per_round' in resolved_args and 'epochs' not in resolved_args:
        resolved_args['epochs'] = resolved_args['epochs_per_round']
    return Command(update={
        'training_plan': updated.model_dump(),
        'training_phase': 'start',
        'training_next_step_tool': next_tool_name,
        'training_next_step_args': resolved_args,
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
