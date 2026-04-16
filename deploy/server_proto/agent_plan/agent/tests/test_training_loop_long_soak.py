from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.training_loop_service import TrainingLoopService
from yolostudio_agent.agent.tests.test_training_loop_service import (
    _FakeKnowledgeService,
    _FakeTrainService,
    _wait_for_status,
)
from yolostudio_agent.agent.tests.training_loop_soak_support import run_training_loop_soak


WORK = Path(__file__).resolve().parent / '_tmp_training_loop_long_soak'


def _run_hundred_round_soak() -> None:
    scripted_rounds = []
    for index in range(100):
        scripted_rounds.append({
            'metrics': {
                'map50': round(0.50 + index * 0.0001, 4),
                'precision': round(0.70 + index * 0.0001, 4),
                'recall': round(0.60 + index * 0.0001, 4),
            },
            'recommended_action': 'continue_observing',
        })
    train_service = _FakeTrainService(scripted_rounds, running_polls=1)
    service = TrainingLoopService(
        state_dir=WORK / 'hundred_rounds',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        poll_interval=0.005,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=10,
        managed_level='full_auto',
        max_rounds=100,
        min_improvement=0.0,
        no_improvement_rounds=100,
        allowed_tuning_params=[],
    )
    assert result['ok'] is True, result
    assert result['boundaries']['max_rounds'] == 100, result
    assert result['boundaries']['allowed_tuning_params'] == [], result
    final_state = _wait_for_status(service, result['loop_id'], {'completed'}, timeout=60.0)
    assert final_state['stop_reason'] == 'max_rounds_reached', final_state
    assert final_state['final_summary']['round_count'] == 100, final_state
    assert len(final_state['rounds']) == 100, final_state
    assert final_state['best_round_index'] == 100, final_state
    detail = service.inspect_loop(result['loop_id'])
    assert len(detail['round_cards']) == 100, detail
    assert detail['latest_round_card']['round_index'] == 100, detail


def _run_pause_resume_idempotence() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.51}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.52}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.53}, 'recommended_action': 'continue_observing'},
    ], running_polls=2)
    service = TrainingLoopService(
        state_dir=WORK / 'pause_resume_idempotence',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        poll_interval=0.01,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=20,
        managed_level='review',
        max_rounds=3,
    )
    assert result['ok'] is True, result
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'}, timeout=5.0)
    assert waiting['current_round_index'] == 1, waiting

    pause_once = service.pause_loop(result['loop_id'])
    pause_twice = service.pause_loop(result['loop_id'])
    assert pause_once['ok'] is True, pause_once
    assert pause_twice['ok'] is True, pause_twice

    resume_once = service.resume_loop(result['loop_id'])
    resume_twice = service.resume_loop(result['loop_id'])
    assert resume_once['ok'] is True, resume_once
    assert resume_twice['ok'] is True, resume_twice

    second_wait = _wait_for_status(service, result['loop_id'], {'awaiting_review', 'completed'}, timeout=5.0)
    if second_wait['status'] == 'awaiting_review':
        resume_final = service.resume_loop(result['loop_id'])
        assert resume_final['ok'] is True, resume_final
    final_state = _wait_for_status(service, result['loop_id'], {'completed'}, timeout=5.0)
    assert final_state['final_summary']['round_count'] == 3, final_state


def _run_support_payload_smoke() -> None:
    output_path = WORK / 'support_payload' / 'soak.json'
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.41}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.42}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.43}, 'recommended_action': 'continue_observing'},
    ], running_polls=1)
    payload = run_training_loop_soak(
        output_path=output_path,
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=8,
        loop_name='support-smoke',
        managed_level='full_auto',
        max_rounds=3,
        min_improvement=0.0,
        no_improvement_rounds=3,
        allowed_tuning_params=[],
        state_dir=WORK / 'support_payload' / 'state',
        loop_poll_interval=0.005,
        watch_poll_interval=0.01,
        timeout=5.0,
        train_service=train_service,
    )
    assert payload['ok'] is True, payload
    assert payload['final_status'] == 'completed', payload
    assert payload['round_count'] == 3, payload
    assert payload['stop_reason'] == 'max_rounds_reached', payload
    assert payload['output_path'] == str(output_path), payload
    assert output_path.exists(), payload


def _run_support_review_gate_smoke() -> None:
    output_path = WORK / 'support_review_gate' / 'soak.json'
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.19, 'precision': 0.41, 'recall': 0.48}, 'recommended_action': 'run_error_analysis'},
    ], running_polls=1)
    payload = run_training_loop_soak(
        output_path=output_path,
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=8,
        loop_name='support-review-gate',
        managed_level='conservative_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=2,
        allowed_tuning_params=[],
        state_dir=WORK / 'support_review_gate' / 'state',
        loop_poll_interval=0.005,
        watch_poll_interval=0.01,
        wait_mode='review_or_terminal',
        timeout=5.0,
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
    )
    assert payload['ok'] is True, payload
    assert payload['final_status'] == 'awaiting_review', payload
    assert payload['round_count'] == 1, payload
    assert payload['final_state']['next_round_plan']['round_index'] == 2, payload
    assert payload['final_state']['rounds'][0]['decision']['decision_type'] == 'await_review', payload
    assert payload['output_path'] == str(output_path), payload
    assert output_path.exists(), payload


def _run_support_review_resume_smoke() -> None:
    output_path = WORK / 'support_review_resume' / 'soak.json'
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.19, 'precision': 0.41, 'recall': 0.48}, 'recommended_action': 'run_error_analysis'},
        {'metrics': {'map50': 0.20, 'precision': 0.42, 'recall': 0.49}, 'recommended_action': 'run_error_analysis'},
    ], running_polls=1)
    payload = run_training_loop_soak(
        output_path=output_path,
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=8,
        loop_name='support-review-resume',
        managed_level='conservative_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=2,
        allowed_tuning_params=[],
        state_dir=WORK / 'support_review_resume' / 'state',
        loop_poll_interval=0.005,
        watch_poll_interval=0.01,
        wait_mode='terminal',
        auto_resume_reviews=1,
        recreate_service_on_review_resume=True,
        timeout=5.0,
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
    )
    assert payload['ok'] is True, payload
    assert payload['final_status'] == 'completed', payload
    assert payload['stop_reason'] == 'max_rounds_reached', payload
    assert len(payload['observed_states']) >= 2, payload
    assert payload['observed_states'][0]['status'] == 'awaiting_review', payload
    assert payload['review_resumes'][0]['ok'] is True, payload
    assert payload['review_resumes'][0]['status'] == 'queued', payload


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        _run_hundred_round_soak()
        _run_pause_resume_idempotence()
        _run_support_payload_smoke()
        _run_support_review_gate_smoke()
        _run_support_review_resume_smoke()
        print('training loop long soak ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
