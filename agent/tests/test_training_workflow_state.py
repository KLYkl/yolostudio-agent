from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.training_workflow import (
    TrainingLoopWorkflowState,
    TrainingWorkflowState,
    sync_training_workflow_state,
)


def _event_recorder(events: list[tuple[str, dict]]) -> Callable[[str, dict], None]:
    def _append(event_type: str, payload: dict) -> None:
        events.append((event_type, dict(payload)))

    return _append


def main() -> None:
    state = SessionState(session_id='workflow-state-smoke')
    events: list[tuple[str, dict]] = []
    append_event = _event_recorder(events)

    initial = sync_training_workflow_state(state, append_event=append_event, reason='initial')
    assert initial.training_state == TrainingWorkflowState.IDLE.value
    assert initial.loop_state == TrainingLoopWorkflowState.LOOP_IDLE.value
    assert not initial.training_changed
    assert not initial.loop_changed
    assert not events

    state.active_dataset.last_readiness = {'ready': False, 'summary': '还缺 data.yaml'}
    readiness = sync_training_workflow_state(state, append_event=append_event, reason='readiness_checked')
    assert readiness.training_state == TrainingWorkflowState.READINESS_CHECKED.value
    assert events[-1][0] == 'training_workflow_state_changed'
    assert events[-1][1]['previous_state'] == TrainingWorkflowState.IDLE.value
    assert events[-1][1]['state'] == TrainingWorkflowState.READINESS_CHECKED.value

    state.active_dataset.last_readiness = {'ready': True, 'summary': '可以训练'}
    state.active_dataset.data_yaml = '/data/dataset/data.yaml'
    prepared = sync_training_workflow_state(state, append_event=append_event, reason='prepared')
    assert prepared.training_state == TrainingWorkflowState.PREPARED.value

    state.active_training.last_preflight = {'ready_to_start': True, 'summary': '环境可用'}
    preflight = sync_training_workflow_state(state, append_event=append_event, reason='preflight_ready')
    assert preflight.training_state == TrainingWorkflowState.PREFLIGHT_READY.value

    pending = sync_training_workflow_state(
        state,
        pending_confirmation={'tool_name': 'start_training'},
        append_event=append_event,
        reason='pending_confirmation',
    )
    assert pending.training_state == TrainingWorkflowState.PENDING_CONFIRMATION.value
    assert events[-1][1]['pending_tool'] == 'start_training'

    remote_pending = sync_training_workflow_state(
        state,
        pending_confirmation={'tool_name': 'remote_training_pipeline'},
        append_event=append_event,
        reason='remote_pending_confirmation',
    )
    assert remote_pending.training_state == TrainingWorkflowState.PENDING_CONFIRMATION.value
    assert state.active_training.workflow_state == TrainingWorkflowState.PENDING_CONFIRMATION.value

    runtime_cleared = sync_training_workflow_state(
        state,
        pending_confirmation={},
        append_event=append_event,
        reason='runtime_pending_cleared',
    )
    assert runtime_cleared.training_state == TrainingWorkflowState.PREFLIGHT_READY.value
    assert state.active_training.workflow_state == TrainingWorkflowState.PREFLIGHT_READY.value

    state.active_training.running = True
    running = sync_training_workflow_state(state, append_event=append_event, reason='running')
    assert running.training_state == TrainingWorkflowState.RUNNING.value

    state.active_training.running = False
    state.active_training.last_status = {'run_state': 'completed'}
    completed = sync_training_workflow_state(state, append_event=append_event, reason='completed')
    assert completed.training_state == TrainingWorkflowState.COMPLETED.value

    state.active_training.last_status = {'run_state': 'failed'}
    failed = sync_training_workflow_state(state, append_event=append_event, reason='failed')
    assert failed.training_state == TrainingWorkflowState.FAILED.value

    state.active_training.last_status = {'run_state': 'stopped'}
    stopped = sync_training_workflow_state(state, append_event=append_event, reason='stopped')
    assert stopped.training_state == TrainingWorkflowState.STOPPED.value

    state.active_training.active_loop_id = 'loop-001'
    state.active_training.active_loop_status = 'running'
    loop_running = sync_training_workflow_state(state, append_event=append_event, reason='loop_running')
    assert loop_running.loop_state == TrainingLoopWorkflowState.LOOP_RUNNING.value

    state.active_training.active_loop_status = 'awaiting_review'
    loop_review = sync_training_workflow_state(state, append_event=append_event, reason='loop_review')
    assert loop_review.loop_state == TrainingLoopWorkflowState.LOOP_AWAITING_REVIEW.value

    state.active_training.active_loop_status = 'paused'
    loop_paused = sync_training_workflow_state(state, append_event=append_event, reason='loop_paused')
    assert loop_paused.loop_state == TrainingLoopWorkflowState.LOOP_PAUSED.value

    state.active_training.active_loop_status = 'stopped'
    loop_stopped = sync_training_workflow_state(state, append_event=append_event, reason='loop_stopped')
    assert loop_stopped.loop_state == TrainingLoopWorkflowState.LOOP_STOPPED.value

    state.active_training.active_loop_status = 'completed'
    loop_completed = sync_training_workflow_state(state, append_event=append_event, reason='loop_completed')
    assert loop_completed.loop_state == TrainingLoopWorkflowState.LOOP_COMPLETED.value

    loop_events = [payload for event_type, payload in events if event_type == 'training_loop_workflow_state_changed']
    assert loop_events
    assert loop_events[-1]['state'] == TrainingLoopWorkflowState.LOOP_COMPLETED.value
    print('training workflow state ok')


if __name__ == '__main__':
    main()
