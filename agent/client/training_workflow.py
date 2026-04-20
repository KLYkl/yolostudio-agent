from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from yolostudio_agent.agent.client.session_state import SessionState

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


class TrainingWorkflowState(StrEnum):
    IDLE = 'idle'
    READINESS_CHECKED = 'readiness_checked'
    PREPARED = 'prepared'
    PREFLIGHT_READY = 'preflight_ready'
    PENDING_CONFIRMATION = 'pending_confirmation'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STOPPED = 'stopped'


class TrainingLoopWorkflowState(StrEnum):
    LOOP_IDLE = 'loop_idle'
    LOOP_RUNNING = 'loop_running'
    LOOP_AWAITING_REVIEW = 'loop_awaiting_review'
    LOOP_PAUSED = 'loop_paused'
    LOOP_STOPPED = 'loop_stopped'
    LOOP_COMPLETED = 'loop_completed'


TRAINING_PENDING_TOOLS = {'prepare_dataset_for_training', 'start_training', 'remote_training_pipeline'}
RUNNING_RUN_STATES = {'running', 'in_progress', 'active'}
FAILED_RUN_STATES = {'failed', 'error', 'crashed'}
STOPPED_RUN_STATES = {'stopped', 'cancelled', 'terminated', 'aborted'}
COMPLETED_RUN_STATES = {'completed', 'finished', 'succeeded', 'success'}

RUNNING_LOOP_STATES = {'running', 'in_progress', 'active', 'queued'}
AWAITING_REVIEW_LOOP_STATES = {'awaiting_review', 'review_required', 'manual_review'}
PAUSED_LOOP_STATES = {'paused'}
STOPPED_LOOP_STATES = {'stopped', 'cancelled', 'terminated', 'aborted'}
COMPLETED_LOOP_STATES = {'completed', 'finished', 'succeeded', 'success'}


@dataclass(slots=True)
class WorkflowSyncResult:
    training_state: str
    loop_state: str
    training_changed: bool
    loop_changed: bool


def _normalized(value: object) -> str:
    return str(value or '').strip().lower()


def _first_non_empty(*values: object) -> str:
    for value in values:
        normalized = _normalized(value)
        if normalized:
            return normalized
    return ''


def _pending_tool_name(pending_confirmation: dict[str, Any] | None, session_state: SessionState) -> str:
    del session_state
    if pending_confirmation is not None:
        return _first_non_empty(
            pending_confirmation.get('tool_name'),
            pending_confirmation.get('name'),
        )
    return ''


def derive_training_workflow_state(
    session_state: SessionState,
    *,
    pending_confirmation: dict[str, Any] | None = None,
) -> str:
    ds = session_state.active_dataset
    tr = session_state.active_training
    pending_tool = _pending_tool_name(pending_confirmation, session_state)
    run_state = _first_non_empty(
        tr.last_status.get('run_state'),
        tr.training_run_summary.get('run_state'),
        tr.last_summary.get('run_state'),
        tr.last_start_result.get('run_state'),
    )

    if pending_tool in TRAINING_PENDING_TOOLS:
        return TrainingWorkflowState.PENDING_CONFIRMATION.value
    if tr.running or run_state in RUNNING_RUN_STATES:
        return TrainingWorkflowState.RUNNING.value
    if run_state in FAILED_RUN_STATES or bool(tr.last_start_result.get('error')):
        return TrainingWorkflowState.FAILED.value
    if run_state in STOPPED_RUN_STATES:
        return TrainingWorkflowState.STOPPED.value
    if run_state in COMPLETED_RUN_STATES:
        return TrainingWorkflowState.COMPLETED.value
    if tr.last_preflight.get('ready_to_start'):
        return TrainingWorkflowState.PREFLIGHT_READY.value
    if (ds.last_readiness.get('ready') is True and _normalized(ds.data_yaml or tr.data_yaml)) or bool(
        ds.last_split or tr.data_yaml
    ):
        return TrainingWorkflowState.PREPARED.value
    if bool(ds.last_readiness or tr.last_environment_probe or tr.last_preflight):
        return TrainingWorkflowState.READINESS_CHECKED.value
    return TrainingWorkflowState.IDLE.value


def derive_training_loop_workflow_state(session_state: SessionState) -> str:
    tr = session_state.active_training
    loop_status = _first_non_empty(
        tr.active_loop_status,
        tr.last_loop_status.get('status'),
        tr.last_loop_detail.get('status'),
    )
    if loop_status in AWAITING_REVIEW_LOOP_STATES:
        return TrainingLoopWorkflowState.LOOP_AWAITING_REVIEW.value
    if loop_status in PAUSED_LOOP_STATES:
        return TrainingLoopWorkflowState.LOOP_PAUSED.value
    if loop_status in STOPPED_LOOP_STATES:
        return TrainingLoopWorkflowState.LOOP_STOPPED.value
    if loop_status in COMPLETED_LOOP_STATES:
        return TrainingLoopWorkflowState.LOOP_COMPLETED.value
    if loop_status in RUNNING_LOOP_STATES or _normalized(tr.active_loop_id):
        return TrainingLoopWorkflowState.LOOP_RUNNING.value
    return TrainingLoopWorkflowState.LOOP_IDLE.value


def sync_training_workflow_state(
    session_state: SessionState,
    *,
    pending_confirmation: dict[str, Any] | None = None,
    append_event: Callable[[str, dict[str, Any]], None] | None = None,
    reason: str = '',
) -> WorkflowSyncResult:
    tr = session_state.active_training
    previous_training_state = _normalized(tr.workflow_state) or TrainingWorkflowState.IDLE.value
    previous_loop_state = _normalized(tr.loop_workflow_state) or TrainingLoopWorkflowState.LOOP_IDLE.value
    next_training_state = derive_training_workflow_state(
        session_state,
        pending_confirmation=pending_confirmation,
    )
    next_loop_state = derive_training_loop_workflow_state(session_state)
    training_changed = previous_training_state != next_training_state
    loop_changed = previous_loop_state != next_loop_state

    tr.workflow_state = next_training_state
    tr.loop_workflow_state = next_loop_state

    if append_event is not None and training_changed:
        append_event(
            'training_workflow_state_changed',
            {
                'previous_state': previous_training_state,
                'state': next_training_state,
                'reason': reason,
                'run_state': _first_non_empty(
                    tr.last_status.get('run_state'),
                    tr.training_run_summary.get('run_state'),
                    tr.last_summary.get('run_state'),
                ),
                'pending_tool': _pending_tool_name(pending_confirmation, session_state),
            },
        )
    if append_event is not None and loop_changed:
        append_event(
            'training_loop_workflow_state_changed',
            {
                'previous_state': previous_loop_state,
                'state': next_loop_state,
                'reason': reason,
                'loop_id': str(tr.active_loop_id or ''),
                'loop_status': _first_non_empty(
                    tr.active_loop_status,
                    tr.last_loop_status.get('status'),
                    tr.last_loop_detail.get('status'),
                ),
            },
        )

    return WorkflowSyncResult(
        training_state=next_training_state,
        loop_state=next_loop_state,
        training_changed=training_changed,
        loop_changed=loop_changed,
    )
