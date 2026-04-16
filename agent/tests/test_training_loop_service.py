from __future__ import annotations

import copy
import shutil
import sys
import time
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.knowledge_service import KnowledgeService
from yolostudio_agent.agent.server.services.training_loop_service import TrainingLoopService


WORK = Path(__file__).resolve().parent / '_tmp_training_loop_service'


class _FakeTrainService:
    def __init__(self, scripted_rounds: list[dict], *, running_polls: int = 1) -> None:
        self._scripted_rounds = [copy.deepcopy(item) for item in scripted_rounds]
        self._running_polls = max(1, int(running_polls))
        self.started_args: list[dict] = []
        self._current: dict | None = None
        self._results_by_log: dict[str, dict] = {}

    def training_preflight(self, **kwargs):
        return {
            'ok': True,
            'ready_to_start': True,
            'summary': '训练预检通过',
            'resolved_args': copy.deepcopy(kwargs),
            'blockers': [],
            'warnings': [],
        }

    def start(self, **kwargs):
        round_index = len(self.started_args) + 1
        if round_index > len(self._scripted_rounds):
            return {'ok': False, 'error': '没有更多脚本轮次可供启动'}
        log_file = f'fake_round_{round_index}.log'
        started_at = time.time()
        self.started_args.append(copy.deepcopy(kwargs))
        scripted = copy.deepcopy(self._scripted_rounds[round_index - 1])
        summary = {
            'ok': True,
            'summary': scripted.get('summary') or f'round {round_index} done',
            'run_state': scripted.get('run_state', 'completed'),
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'model_family': 'yolo',
            'task_type': 'detection',
            'status_source': 'fake',
            'started_at': started_at,
            'stopped_at': started_at + 1,
            'updated_at': started_at + 1,
            'pid': 1000 + round_index,
            'device': '0',
            'training_environment': {'name': 'fake-env', 'display_name': 'fake-env'},
            'log_file': log_file,
            'model': kwargs.get('model'),
            'data_yaml': kwargs.get('data_yaml'),
            'epochs': kwargs.get('epochs'),
            'resolved_args': copy.deepcopy(kwargs),
            'progress': {
                'epoch': kwargs.get('epochs'),
                'total_epochs': kwargs.get('epochs'),
                'progress_ratio': 1.0,
            },
            'metrics': copy.deepcopy(scripted.get('metrics') or {}),
            'signals': ['training_completed'],
            'facts': [f'round={round_index}'],
            'latest_metrics': {
                'ok': True,
                'metrics': copy.deepcopy(scripted.get('metrics') or {}),
            },
            'save_dir': f'/tmp/fake_round_{round_index}',
            'next_actions': ['fake next action'],
            'error_lines': copy.deepcopy(scripted.get('error_lines') or []),
            'scripted_action': scripted.get('recommended_action', 'continue_observing'),
        }
        self._results_by_log[log_file] = summary
        self._current = {
            'remaining': self._running_polls,
            'log_file': log_file,
            'pid': 1000 + round_index,
            'epochs': kwargs.get('epochs'),
            'metrics': copy.deepcopy(scripted.get('metrics') or {}),
        }
        return {
            'ok': True,
            'message': 'fake training started',
            'pid': 1000 + round_index,
            'device': '0',
            'requested_device': kwargs.get('device', 'auto'),
            'log_file': log_file,
            'argument_sources': {},
            'command': ['yolo', 'train'],
            'resolved_args': {**copy.deepcopy(kwargs), 'device_policy': 'single_idle_gpu'},
            'yolo_executable': 'yolo',
            'training_environment': {'name': 'fake-env', 'display_name': 'fake-env'},
            'started_at': started_at,
            'registry_path': str(WORK / 'fake_registry.json'),
            'reattached': False,
        }

    def status(self):
        if self._current:
            if self._current['remaining'] > 0:
                self._current['remaining'] -= 1
                return {
                    'ok': True,
                    'running': True,
                    'summary': 'fake training running',
                    'pid': self._current['pid'],
                    'log_file': self._current['log_file'],
                    'progress': {
                        'epoch': max(1, self._current['epochs'] // 2),
                        'total_epochs': self._current['epochs'],
                        'progress_ratio': 0.5,
                    },
                    'latest_metrics': {
                        'ok': True,
                        'metrics': None,
                    },
                }
            self._current = None
        return {
            'ok': True,
            'running': False,
            'summary': 'fake training idle',
        }

    def inspect_training_run(self, run_id: str = ''):
        return copy.deepcopy(self._results_by_log[run_id])

    def compare_training_runs(self, left_run_id: str = '', right_run_id: str = ''):
        left = self._results_by_log[left_run_id]
        right = self._results_by_log[right_run_id]
        left_map = float((left.get('metrics') or {}).get('map50') or 0.0)
        right_map = float((right.get('metrics') or {}).get('map50') or 0.0)
        return {
            'ok': True,
            'summary': '训练对比完成',
            'left_run_id': left_run_id,
            'right_run_id': right_run_id,
            'metric_deltas': {
                'map50': {
                    'left': left_map,
                    'right': right_map,
                    'delta': round(left_map - right_map, 4),
                }
            },
            'highlights': [f'mAP50变化 {left_map - right_map:+.4f}'],
            'next_actions': ['fake compare next'],
        }

    def stop(self):
        if self._current:
            self._current['remaining'] = 0
        return {'ok': True, 'message': 'fake stop ok'}


class _FakeKnowledgeService:
    def analyze_training_outcome(self, *, metrics=None, **kwargs):
        del kwargs
        payload = dict(metrics or {})
        action = payload.get('scripted_action', 'continue_observing')
        return {
            'ok': True,
            'summary': f'分析完成: {action}',
            'assessment': action,
            'interpretation': f'interpretation={action}',
            'recommendation': f'recommendation={action}',
            'signals': payload.get('signals') or [],
            'facts': payload.get('facts') or [],
        }

    def recommend_next_training_step(self, *, status=None, **kwargs):
        del kwargs
        payload = dict(status or {})
        action = payload.get('scripted_action', 'continue_observing')
        return {
            'ok': True,
            'summary': f'建议完成: {action}',
            'recommended_action': action,
            'recommendation': f'recommendation={action}',
            'why': f'why={action}',
            'signals': payload.get('signals') or [],
            'basis': payload.get('facts') or [],
        }


class _FakeLoopLlmResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLoopLlm:
    def __init__(self, *, round_review_response: str | None = None, next_round_plan_response: str | None = None) -> None:
        self.prompts: list[str] = []
        self._round_review_response = round_review_response or (
            '{"review_summary":"LLM 认为当前轮还有提升空间，但更该先稳住学习率。","recommended_action":"continue_observing","why":"最近一轮提升幅度变小，下一轮应该优先看收敛稳定性。","carry_forward":["上一轮 mAP50 有提升，但幅度已经收窄。"],"blockers":[],"next_focus":"收敛稳定性","confidence":0.86}'
        )
        self._next_round_plan_response = next_round_plan_response or (
            '{"plan_summary":"下一轮建议小幅下调学习率并保留当前训练结构。","reason":"指标还在涨，但斜率已经放缓。","suggested_param_updates":[{"field":"lr0","value":0.005,"reason":"先稳住收敛。"},{"field":"epochs","value":38,"reason":"给模型一点继续收敛的窗口。"}]}'
        )

    def invoke(self, prompt: str):
        self.prompts.append(str(prompt))
        text = str(prompt)
        if '"task": "round_review"' in text:
            return _FakeLoopLlmResponse(self._round_review_response)
        if '"task": "next_round_plan"' in text:
            return _FakeLoopLlmResponse(self._next_round_plan_response)
        return _FakeLoopLlmResponse('{}')


def _wait_for_status(service: TrainingLoopService, loop_id: str, expected: set[str], timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = service.inspect_loop(loop_id)
        if str(last.get('status') or '') in expected:
            return last
        time.sleep(0.02)
    raise AssertionError(f'timeout waiting for {expected}, last={last}')


def _run_default_loop_llm_is_not_built_in_service() -> None:
    service = TrainingLoopService(
        state_dir=WORK / 'default_loop_llm_none',
        train_service=_FakeTrainService([
            {'metrics': {'map50': 0.50}, 'recommended_action': 'continue_observing'},
        ]),
        knowledge_service=_FakeKnowledgeService(),
        poll_interval=0.02,
    )
    assert service.loop_llm is None, service.loop_llm


def _run_full_auto_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.50, 'precision': 0.70, 'recall': 0.60}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.56, 'precision': 0.73, 'recall': 0.63}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.561, 'precision': 0.74, 'recall': 0.64}, 'recommended_action': 'continue_observing'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'full_auto',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=50,
        batch=16,
        managed_level='full_auto',
        max_rounds=5,
        min_improvement=0.01,
        no_improvement_rounds=1,
    )
    assert result['ok'] is True
    final_state = _wait_for_status(service, result['loop_id'], {'completed'})
    assert final_state['stop_reason'] == 'no_improvement'
    assert len(final_state['rounds']) == 3
    assert final_state['best_round_index'] == 3
    assert final_state['final_summary']['best_round_index'] == 3
    assert final_state['latest_round_card']['round_index'] == 3
    assert train_service.started_args[1]['epochs'] > train_service.started_args[0]['epochs']
    assert train_service.started_args[2]['epochs'] > train_service.started_args[1]['epochs']


def _run_review_mode_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.42, 'precision': 0.61, 'recall': 0.55}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.47, 'precision': 0.64, 'recall': 0.58}, 'recommended_action': 'continue_observing'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'review_mode',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=40,
        managed_level='review',
        max_rounds=2,
        no_improvement_rounds=5,
    )
    assert result['ok'] is True
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'})
    assert len(waiting['rounds']) == 1
    assert waiting['next_round_plan']['round_index'] == 2
    resumed = service.resume_loop(result['loop_id'])
    assert resumed['ok'] is True
    final_state = _wait_for_status(service, result['loop_id'], {'completed'})
    assert final_state['stop_reason'] == 'max_rounds_reached'
    assert len(final_state['rounds']) == 2
    assert final_state['final_summary']['round_count'] == 2


def _run_real_knowledge_short_window_continue() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'precision': 0.002, 'recall': 0.667, 'map50': 0.006, 'map': 0.002}, 'recommended_action': 'continue_observing'},
        {'metrics': {'precision': 0.003, 'recall': 0.670, 'map50': 0.007, 'map': 0.002}, 'recommended_action': 'continue_observing'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'real_knowledge_short_window',
        train_service=train_service,
        knowledge_service=KnowledgeService(project_root=Path(__file__).resolve().parents[2]),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=1,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=2,
        allowed_tuning_params=[],
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'completed'}, timeout=10.0)
    assert final_state['stop_reason'] == 'max_rounds_reached', final_state
    assert final_state['final_summary']['round_count'] == 2, final_state
    assert final_state['rounds'][0]['recommendation']['recommended_action'] == 'continue_observing', final_state


def _run_conservative_review_gate_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.19, 'precision': 0.41, 'recall': 0.48}, 'recommended_action': 'run_error_analysis'},
        {'metrics': {'map50': 0.20, 'precision': 0.42, 'recall': 0.49}, 'recommended_action': 'run_error_analysis'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'conservative_review_gate',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='conservative_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=[],
    )
    assert result['ok'] is True, result
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'})
    assert waiting['rounds'][0]['recommendation']['recommended_action'] == 'run_error_analysis', waiting
    assert waiting['rounds'][0]['decision']['decision_type'] == 'await_review', waiting
    assert waiting['next_round_plan']['round_index'] == 2, waiting
    knowledge_gate = waiting['latest_round_card']['knowledge_gate']
    assert knowledge_gate['action'] == 'run_error_analysis', waiting
    assert knowledge_gate['action_label'] == '先做误差分析', waiting
    assert knowledge_gate['category'] == 'analysis_review', waiting
    assert knowledge_gate['outcome'] == 'awaiting_review', waiting
    assert knowledge_gate['decision_type'] == 'await_review', waiting
    assert waiting['knowledge_gate_status']['outcome'] == 'awaiting_review', waiting
    resumed = service.resume_loop(result['loop_id'])
    assert resumed['ok'] is True, resumed
    final_state = _wait_for_status(service, result['loop_id'], {'completed'})
    assert final_state['stop_reason'] == 'max_rounds_reached', final_state
    assert final_state['final_summary']['round_count'] == 2, final_state


def _run_full_auto_review_action_continue() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.19, 'precision': 0.41, 'recall': 0.48}, 'recommended_action': 'run_error_analysis'},
        {'metrics': {'map50': 0.20, 'precision': 0.42, 'recall': 0.49}, 'recommended_action': 'run_error_analysis'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'full_auto_review_action',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=[],
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'completed'})
    assert final_state['stop_reason'] == 'max_rounds_reached', final_state
    assert final_state['final_summary']['round_count'] == 2, final_state
    assert final_state['rounds'][0]['recommendation']['recommended_action'] == 'run_error_analysis', final_state
    assert final_state['rounds'][0]['decision']['decision_type'] == 'auto_continue', final_state
    first_gate = final_state['round_cards'][0]['knowledge_gate']
    assert first_gate['action'] == 'run_error_analysis', final_state
    assert first_gate['decision_type'] == 'auto_continue', final_state
    assert first_gate['outcome'] == 'auto_continue', final_state
    assert final_state['final_summary']['last_knowledge_gate']['action'] == 'run_error_analysis', final_state
    assert final_state['final_summary']['knowledge_gate_overview']['count'] == 2, final_state


def _run_fix_data_quality_still_stops() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.08, 'precision': 0.20, 'recall': 0.17}, 'recommended_action': 'fix_data_quality'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'fix_data_quality_stop',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=3,
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'stopped'})
    assert final_state['stop_reason'] == 'fix_data_quality', final_state
    assert final_state['final_summary']['round_count'] == 1, final_state


def _run_real_knowledge_long_window_review_gate() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'precision': 0.41, 'recall': 0.48, 'map50': 0.19, 'map': 0.08}, 'recommended_action': 'run_error_analysis'},
        {'metrics': {'precision': 0.42, 'recall': 0.49, 'map50': 0.20, 'map': 0.09}, 'recommended_action': 'run_error_analysis'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'real_knowledge_long_window_review_gate',
        train_service=train_service,
        knowledge_service=KnowledgeService(project_root=Path(__file__).resolve().parents[2]),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=4,
        managed_level='conservative_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=[],
    )
    assert result['ok'] is True, result
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'})
    assert waiting['rounds'][0]['recommendation']['recommended_action'] == 'run_error_analysis', waiting
    assert waiting['rounds'][0]['decision']['decision_type'] == 'await_review', waiting
    assert waiting['next_round_plan']['round_index'] == 2, waiting
    assert waiting['latest_round_card']['knowledge_gate']['decision_type'] == 'await_review', waiting
    assert waiting['knowledge_gate_status']['summary'], waiting


def _run_real_knowledge_long_window_full_auto_continue() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'precision': 0.41, 'recall': 0.48, 'map50': 0.19, 'map': 0.08}, 'recommended_action': 'run_error_analysis'},
        {'metrics': {'precision': 0.42, 'recall': 0.49, 'map50': 0.20, 'map': 0.09}, 'recommended_action': 'run_error_analysis'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'real_knowledge_long_window_full_auto',
        train_service=train_service,
        knowledge_service=KnowledgeService(project_root=Path(__file__).resolve().parents[2]),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=4,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=[],
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'completed'}, timeout=10.0)
    assert final_state['stop_reason'] == 'max_rounds_reached', final_state
    assert final_state['final_summary']['round_count'] == 2, final_state
    assert final_state['rounds'][0]['recommendation']['recommended_action'] == 'run_error_analysis', final_state
    assert final_state['rounds'][0]['decision']['decision_type'] == 'auto_continue', final_state
    assert final_state['final_summary']['knowledge_gate_rounds'][0]['decision_type'] == 'auto_continue', final_state
    assert final_state['final_summary']['knowledge_gate_overview']['last_outcome'] == 'stopped', final_state


def _run_round_memory_payload_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.47, 'precision': 0.66, 'recall': 0.60}, 'recommended_action': 'continue_observing'},
    ])
    service = TrainingLoopService(
        state_dir=WORK / 'round_memory_payload',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'completed'})
    first_round = final_state['rounds'][0]
    second_round = final_state['rounds'][1]
    assert first_round['round_review']['recommended_action'] == 'continue_observing', final_state
    assert first_round['planner_output']['decision_type'] == 'auto_continue', final_state
    assert first_round['round_memory']['next_focus'] == '继续观察', final_state
    assert first_round['experience_context'] == {}, final_state
    assert second_round['experience_context']['recent_round_memory'][0]['round_index'] == 1, final_state
    assert second_round['planner_input']['recent_round_memory'][0]['round_index'] == 1, final_state
    assert final_state['round_cards'][0]['round_review']['recommended_action'] == 'continue_observing', final_state
    assert final_state['round_cards'][0]['planner_output']['decision_type'] == 'auto_continue', final_state
    assert final_state['latest_round_memory']['round_index'] == 2, final_state
    assert final_state['final_summary']['experience_timeline'][0]['round_index'] == 1, final_state
    assert final_state['final_summary']['last_round_memory']['round_index'] == 2, final_state


def _run_running_status_counts_only_finished_rounds() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
    ], running_polls=20)
    service = TrainingLoopService(
        state_dir=WORK / 'running_status_counts',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
    )
    assert result['ok'] is True, result
    running_state = _wait_for_status(service, result['loop_id'], {'running_round'})
    assert running_state['current_round_index'] == 1, running_state
    assert running_state['completed_rounds'] == 0, running_state
    assert running_state['recorded_rounds'] == 1, running_state
    assert running_state['current_training_status']['running'] is True, running_state


def _run_default_loop_epochs_is_small() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
    ], running_polls=20)
    service = TrainingLoopService(
        state_dir=WORK / 'default_loop_epochs',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=False,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
    )
    assert result['ok'] is True, result
    running_state = _wait_for_status(service, result['loop_id'], {'running_round'})
    current_round = running_state['current_round']
    assert current_round['training_args']['epochs'] == 10, running_state
    assert current_round['effective_args']['epochs'] == 10, running_state
    assert train_service.started_args[0]['epochs'] == 10, train_service.started_args


def _run_llm_guided_planner_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
    ])
    loop_llm = _FakeLoopLlm()
    service = TrainingLoopService(
        state_dir=WORK / 'llm_guided_planner',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=loop_llm,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        lr0=0.01,
        managed_level='review',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=['lr0', 'epochs'],
    )
    assert result['ok'] is True, result
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'})
    first_round = waiting['rounds'][0]
    next_plan = waiting['next_round_plan']
    assert first_round['round_review']['review_source'] == 'llm', waiting
    assert first_round['round_review']['next_focus'] == '收敛稳定性', waiting
    assert next_plan['planner_source'] == 'llm', waiting
    assert next_plan['planner_decision_type'] == '', waiting
    assert next_plan['training_args']['lr0'] == 0.005, waiting
    assert next_plan['training_args']['epochs'] == 38, waiting
    assert waiting['latest_planner_output']['planner_source'] == 'llm', waiting
    assert len(loop_llm.prompts) >= 2, loop_llm.prompts


def _run_llm_guided_planner_review_decision_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.45, 'precision': 0.64, 'recall': 0.58}, 'recommended_action': 'continue_observing'},
    ])
    loop_llm = _FakeLoopLlm(
        next_round_plan_response='{"plan_summary":"先别自动续训，下一轮建议进人工审阅。","reason":"虽然还可以继续，但这轮信号值得人工看一眼。","suggested_decision_type":"await_review","suggested_decision_reason":"规划器建议先人工确认下一轮。","suggested_param_updates":[{"field":"epochs","value":34,"reason":"只小幅增加训练窗口。"}]}'
    )
    service = TrainingLoopService(
        state_dir=WORK / 'llm_guided_planner_review_decision',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=loop_llm,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=['epochs'],
    )
    assert result['ok'] is True, result
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'})
    assert waiting['rounds'][0]['decision']['decision_type'] == 'await_review', waiting
    assert waiting['next_round_plan']['planner_source'] == 'llm', waiting
    assert waiting['next_round_plan']['planner_decision_type'] == 'await_review', waiting
    assert waiting['next_round_plan']['planner_decision_reason'] == '规划器建议先人工确认下一轮。', waiting
    assert waiting['next_round_plan']['training_args']['epochs'] == 34, waiting
    assert waiting['latest_planner_output']['decision_type'] == 'await_review', waiting
    assert len(train_service.started_args) == 1, train_service.started_args


def _run_llm_guided_planner_stop_decision_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.45, 'precision': 0.64, 'recall': 0.58}, 'recommended_action': 'continue_observing'},
    ])
    loop_llm = _FakeLoopLlm(
        next_round_plan_response='{"plan_summary":"建议先停止，不再自动进入下一轮。","reason":"当前轮已经给出足够信号，继续自动推进收益有限。","suggested_decision_type":"stop","suggested_decision_reason":"规划器建议在这一轮后先停下。","suggested_param_updates":[]}'
    )
    service = TrainingLoopService(
        state_dir=WORK / 'llm_guided_planner_stop_decision',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=loop_llm,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=3,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=['epochs'],
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'stopped'})
    assert final_state['stop_reason'] == 'planner_stop', final_state
    assert final_state['final_summary']['round_count'] == 1, final_state
    assert final_state['latest_planner_output']['decision_type'] == 'stop', final_state
    assert len(train_service.started_args) == 1, train_service.started_args


def _run_llm_guided_planner_stop_overrides_review_gate_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.45, 'precision': 0.64, 'recall': 0.58}, 'recommended_action': 'continue_observing'},
    ])
    loop_llm = _FakeLoopLlm(
        next_round_plan_response='{"plan_summary":"建议先停止，不再进入审阅等待。","reason":"当前信号已经足够，没必要再挂在 review gate 上。","suggested_decision_type":"stop","suggested_decision_reason":"规划器建议在这一轮后先停下。","suggested_param_updates":[]}'
    )
    service = TrainingLoopService(
        state_dir=WORK / 'llm_guided_planner_stop_overrides_review_gate',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=loop_llm,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='review',
        max_rounds=3,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=['epochs'],
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'stopped'})
    assert final_state['stop_reason'] == 'planner_stop', final_state
    assert final_state['latest_planner_output']['decision_type'] == 'stop', final_state
    assert final_state['final_summary']['round_count'] == 1, final_state
    assert len(train_service.started_args) == 1, train_service.started_args


def _run_llm_guided_planner_continue_after_plateau_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.50, 'precision': 0.70, 'recall': 0.60}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.56, 'precision': 0.73, 'recall': 0.63}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.561, 'precision': 0.74, 'recall': 0.64}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.562, 'precision': 0.75, 'recall': 0.65}, 'recommended_action': 'continue_observing'},
    ])
    loop_llm = _FakeLoopLlm(
        next_round_plan_response='{"plan_summary":"虽然提升变慢，但建议再给一轮确认趋势。","reason":"当前平台期还不够长，先再观察一轮更稳妥。","suggested_decision_type":"auto_continue","suggested_decision_reason":"规划器建议在平台期再续跑一轮确认趋势。","suggested_param_updates":[{"field":"epochs","value":52,"reason":"只小幅增加训练窗口。"}]}'
    )
    service = TrainingLoopService(
        state_dir=WORK / 'llm_guided_planner_continue_after_plateau',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=loop_llm,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=50,
        managed_level='full_auto',
        max_rounds=4,
        min_improvement=0.01,
        no_improvement_rounds=1,
        allowed_tuning_params=['epochs'],
    )
    assert result['ok'] is True, result
    final_state = _wait_for_status(service, result['loop_id'], {'completed'})
    assert final_state['stop_reason'] == 'max_rounds_reached', final_state
    assert final_state['final_summary']['round_count'] == 4, final_state
    assert final_state['rounds'][2]['decision']['decision_type'] == 'auto_continue', final_state
    assert final_state['rounds'][2]['decision']['reason'] == '规划器建议在平台期再续跑一轮确认趋势。', final_state
    assert final_state['rounds'][2]['planner_output']['planner_source'] == 'llm', final_state
    assert final_state['rounds'][2]['planner_output']['decision_type'] == 'auto_continue', final_state
    assert final_state['rounds'][2]['round_memory']['decision_reason'] == '规划器建议在平台期再续跑一轮确认趋势。', final_state
    assert len(train_service.started_args) == 4, train_service.started_args


def _run_llm_guided_planner_keeps_current_params_without_updates_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.45, 'precision': 0.64, 'recall': 0.58}, 'recommended_action': 'continue_observing'},
    ])
    loop_llm = _FakeLoopLlm(
        next_round_plan_response='{"plan_summary":"下一轮先保持当前参数继续观察。","reason":"虽然可以继续，但当前没有足够依据改动参数。","suggested_decision_type":"await_review","suggested_decision_reason":"先给人工看一眼当前趋势。","suggested_param_strategy":"keep_current","suggested_param_updates":[]}'
    )
    service = TrainingLoopService(
        state_dir=WORK / 'llm_guided_planner_keep_current_params',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=loop_llm,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=['epochs'],
    )
    assert result['ok'] is True, result
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'})
    assert waiting['next_round_plan']['planner_source'] == 'llm', waiting
    assert waiting['next_round_plan']['planner_decision_type'] == 'await_review', waiting
    assert waiting['next_round_plan']['training_args']['epochs'] == 30, waiting
    assert waiting['next_round_plan']['change_set'] == [
        {'field': 'name', 'old': 'data-yolov8n-r1', 'new': 'data-yolov8n-r2'}
    ], waiting
    assert len(train_service.started_args) == 1, train_service.started_args


def _run_llm_guided_planner_can_explicitly_apply_heuristic_params_smoke() -> None:
    train_service = _FakeTrainService([
        {'metrics': {'map50': 0.44, 'precision': 0.63, 'recall': 0.57}, 'recommended_action': 'continue_observing'},
        {'metrics': {'map50': 0.45, 'precision': 0.64, 'recall': 0.58}, 'recommended_action': 'continue_observing'},
    ])
    loop_llm = _FakeLoopLlm(
        next_round_plan_response='{"plan_summary":"下一轮先沿用系统建议的小步续跑。","reason":"当前暂无更好的改参证据，先沿用保守 heuristic。","suggested_decision_type":"await_review","suggested_decision_reason":"先给人工看一眼 heuristic 方案。","suggested_param_strategy":"apply_heuristic","suggested_param_updates":[]}'
    )
    service = TrainingLoopService(
        state_dir=WORK / 'llm_guided_planner_apply_heuristic_params',
        train_service=train_service,
        knowledge_service=_FakeKnowledgeService(),
        loop_llm=loop_llm,
        poll_interval=0.02,
    )
    result = service.start_loop(
        model='yolov8n.pt',
        data_yaml='data.yaml',
        epochs=30,
        managed_level='full_auto',
        max_rounds=2,
        min_improvement=0.0,
        no_improvement_rounds=5,
        allowed_tuning_params=['epochs'],
    )
    assert result['ok'] is True, result
    waiting = _wait_for_status(service, result['loop_id'], {'awaiting_review'})
    assert waiting['next_round_plan']['planner_source'] == 'llm', waiting
    assert waiting['next_round_plan']['training_args']['epochs'] == 40, waiting
    assert {'field': 'epochs', 'old': 30, 'new': 40} in waiting['next_round_plan']['change_set'], waiting
    assert len(train_service.started_args) == 1, train_service.started_args


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        _run_default_loop_llm_is_not_built_in_service()
        _run_full_auto_smoke()
        _run_review_mode_smoke()
        _run_real_knowledge_short_window_continue()
        _run_conservative_review_gate_smoke()
        _run_full_auto_review_action_continue()
        _run_fix_data_quality_still_stops()
        _run_real_knowledge_long_window_review_gate()
        _run_real_knowledge_long_window_full_auto_continue()
        _run_round_memory_payload_smoke()
        _run_running_status_counts_only_finished_rounds()
        _run_default_loop_epochs_is_small()
        _run_llm_guided_planner_smoke()
        _run_llm_guided_planner_review_decision_smoke()
        _run_llm_guided_planner_stop_decision_smoke()
        _run_llm_guided_planner_stop_overrides_review_gate_smoke()
        _run_llm_guided_planner_continue_after_plateau_smoke()
        _run_llm_guided_planner_keeps_current_params_without_updates_smoke()
        _run_llm_guided_planner_can_explicitly_apply_heuristic_params_smoke()
        print('training loop service smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
