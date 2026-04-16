from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services.knowledge_service import KnowledgeService
from yolostudio_agent.agent.server.services.train_service import TrainService
from yolostudio_agent.agent.server.services.training_loop_service import TrainingLoopService

_TERMINAL_STATES = {'completed', 'failed', 'stopped'}
_WAIT_MODE_TARGETS = {
    'terminal': _TERMINAL_STATES,
    'review_or_terminal': _TERMINAL_STATES | {'awaiting_review'},
}


class ForcedDecisionKnowledgeService:
    def __init__(self, action: str = 'continue_observing') -> None:
        self.action = str(action or 'continue_observing').strip() or 'continue_observing'

    def analyze_training_outcome(self, *, metrics: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        payload = dict(metrics or {})
        return {
            'ok': True,
            'summary': f'分析完成: {self.action}',
            'assessment': self.action,
            'interpretation': f'forced_interpretation={self.action}',
            'recommendation': f'forced_recommendation={self.action}',
            'signals': list(payload.get('signals') or []),
            'facts': list(payload.get('facts') or []),
        }

    def recommend_next_training_step(self, *, status: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        payload = dict(status or {})
        return {
            'ok': True,
            'summary': f'建议完成: {self.action}',
            'recommended_action': self.action,
            'recommendation': f'forced_recommendation={self.action}',
            'why': f'forced_why={self.action}',
            'signals': list(payload.get('signals') or []),
            'basis': list(payload.get('facts') or []),
        }


def parse_allowed_tuning_params(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or '').strip()
    if not text:
        return None
    if text.lower() in {'none', 'null', '[]'}:
        return []
    return [item.strip() for item in text.replace(';', ',').split(',') if item.strip()]


def wait_for_loop_state(
    service: TrainingLoopService,
    loop_id: str,
    *,
    wait_mode: str = 'terminal',
    poll_interval: float = 5.0,
    timeout: float | None = None,
) -> dict[str, Any]:
    normalized_wait_mode = str(wait_mode or 'terminal').strip().lower() or 'terminal'
    target_states = _WAIT_MODE_TARGETS.get(normalized_wait_mode)
    if not target_states:
        raise ValueError(f'unsupported wait_mode: {wait_mode}')
    interval = max(0.1, float(poll_interval))
    deadline = time.time() + float(timeout) if timeout and float(timeout) > 0 else None
    last: dict[str, Any] = {}
    while True:
        last = service.inspect_loop(loop_id)
        status = str(last.get('status') or '').strip().lower()
        if status in target_states:
            return last
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(
                f'等待环训练状态超时（loop_id={loop_id}, wait_mode={normalized_wait_mode}, last_status={status or "unknown"}）'
            )
        time.sleep(interval)


def write_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_path


def run_training_loop_soak(
    *,
    output_path: str | Path | None,
    model: str,
    data_yaml: str,
    epochs: int = 1,
    device: str = '0',
    training_environment: str = '',
    project: str = '',
    name: str = '',
    batch: int | None = None,
    imgsz: int | None = None,
    fraction: float | None = None,
    classes: list[int] | str | None = None,
    single_cls: bool | None = None,
    optimizer: str = '',
    freeze: int | None = None,
    resume: bool | None = None,
    lr0: float | None = None,
    patience: int | None = None,
    workers: int | None = None,
    amp: bool | None = None,
    loop_name: str = '',
    managed_level: str = 'full_auto',
    max_rounds: int = 20,
    target_metric: str = 'map50',
    target_metric_value: float | None = None,
    min_improvement: float = 0.0,
    no_improvement_rounds: int = 999,
    max_failures: int = 2,
    allowed_tuning_params: list[str] | None = None,
    auto_handle_oom: bool = False,
    include_case_sources: bool = False,
    include_test_sources: bool = False,
    max_imgsz: int = 1536,
    min_batch: int = 1,
    knowledge_mode: str = 'forced',
    forced_action: str = 'continue_observing',
    state_dir: str | Path | None = None,
    loop_poll_interval: float = 5.0,
    watch_poll_interval: float = 5.0,
    wait_mode: str = 'terminal',
    auto_resume_reviews: int = 0,
    recreate_service_on_review_resume: bool = False,
    timeout: float | None = None,
    train_service: Any | None = None,
    knowledge_service: Any | None = None,
) -> dict[str, Any]:
    normalized_knowledge_mode = str(knowledge_mode or 'forced').strip().lower() or 'forced'
    if knowledge_service is None:
        if normalized_knowledge_mode == 'real':
            knowledge_service = KnowledgeService()
        else:
            normalized_knowledge_mode = 'forced'
            knowledge_service = ForcedDecisionKnowledgeService(action=forced_action)

    service = TrainingLoopService(
        state_dir=state_dir,
        train_service=train_service or TrainService(),
        knowledge_service=knowledge_service,
        poll_interval=loop_poll_interval,
    )
    payload: dict[str, Any] = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'ok': False,
        'knowledge_mode': normalized_knowledge_mode,
        'forced_action': forced_action if normalized_knowledge_mode == 'forced' else None,
        'config': {
            'model': model,
            'data_yaml': data_yaml,
            'epochs': int(epochs),
            'device': device,
            'training_environment': training_environment,
            'project': project,
            'name': name,
            'batch': batch,
            'imgsz': imgsz,
            'fraction': fraction,
            'classes': classes,
            'single_cls': single_cls,
            'optimizer': optimizer,
            'freeze': freeze,
            'resume': resume,
            'lr0': lr0,
            'patience': patience,
            'workers': workers,
            'amp': amp,
            'loop_name': loop_name,
            'managed_level': managed_level,
            'max_rounds': int(max_rounds),
            'target_metric': target_metric,
            'target_metric_value': target_metric_value,
            'min_improvement': float(min_improvement),
            'no_improvement_rounds': int(no_improvement_rounds),
            'max_failures': int(max_failures),
            'allowed_tuning_params': list(allowed_tuning_params or []),
            'auto_handle_oom': bool(auto_handle_oom),
            'include_case_sources': bool(include_case_sources),
            'include_test_sources': bool(include_test_sources),
            'max_imgsz': int(max_imgsz),
            'min_batch': int(min_batch),
            'state_dir': str(state_dir) if state_dir else '',
            'loop_poll_interval': float(loop_poll_interval),
            'watch_poll_interval': float(watch_poll_interval),
            'wait_mode': str(wait_mode or 'terminal'),
            'auto_resume_reviews': max(0, int(auto_resume_reviews)),
            'recreate_service_on_review_resume': bool(recreate_service_on_review_resume),
            'timeout': None if timeout is None else float(timeout),
        },
    }
    start_result = service.start_loop(
        model=model,
        data_yaml=data_yaml,
        epochs=epochs,
        device=device,
        training_environment=training_environment,
        project=project,
        name=name,
        batch=batch,
        imgsz=imgsz,
        fraction=fraction,
        classes=classes,
        single_cls=single_cls,
        optimizer=optimizer,
        freeze=freeze,
        resume=resume,
        lr0=lr0,
        patience=patience,
        workers=workers,
        amp=amp,
        loop_name=loop_name,
        managed_level=managed_level,
        max_rounds=max_rounds,
        target_metric=target_metric,
        target_metric_value=target_metric_value,
        min_improvement=min_improvement,
        no_improvement_rounds=no_improvement_rounds,
        max_failures=max_failures,
        allowed_tuning_params=allowed_tuning_params,
        auto_handle_oom=auto_handle_oom,
        include_case_sources=include_case_sources,
        include_test_sources=include_test_sources,
        max_imgsz=max_imgsz,
        min_batch=min_batch,
    )
    payload['start_result'] = start_result
    payload['loop_id'] = start_result.get('loop_id')
    payload['loop_name'] = start_result.get('loop_name')
    if not start_result.get('ok'):
        payload['error'] = start_result.get('error') or start_result.get('summary') or '环训练启动失败'
        if output_path:
            payload['output_path'] = str(write_payload(output_path, payload))
        return payload

    loop_id = str(start_result.get('loop_id') or '')
    remaining_review_resumes = max(0, int(auto_resume_reviews))
    observed_states: list[dict[str, Any]] = []
    review_resumes: list[dict[str, Any]] = []
    while True:
        current_wait_mode = 'review_or_terminal' if remaining_review_resumes > 0 else wait_mode
        final_state = wait_for_loop_state(
            service,
            loop_id,
            wait_mode=current_wait_mode,
            poll_interval=watch_poll_interval,
            timeout=timeout,
        )
        observed_states.append({
            'status': final_state.get('status'),
            'summary': final_state.get('summary'),
            'current_round_index': final_state.get('current_round_index'),
            'latest_round_card': final_state.get('latest_round_card'),
        })
        if str(final_state.get('status') or '').strip().lower() != 'awaiting_review' or remaining_review_resumes <= 0:
            break
        if recreate_service_on_review_resume:
            service = TrainingLoopService(
                state_dir=state_dir,
                train_service=train_service or TrainService(),
                knowledge_service=knowledge_service,
                poll_interval=loop_poll_interval,
            )
        resume_result = service.resume_loop(loop_id)
        review_resumes.append({
            'resume_index': len(review_resumes) + 1,
            'ok': bool(resume_result.get('ok')),
            'summary': resume_result.get('summary'),
            'status': resume_result.get('status'),
        })
        if not resume_result.get('ok'):
            payload['error'] = resume_result.get('error') or resume_result.get('summary') or '环训练恢复失败'
            payload['observed_states'] = observed_states
            payload['review_resumes'] = review_resumes
            if output_path:
                payload['output_path'] = str(write_payload(output_path, payload))
            return payload
        remaining_review_resumes -= 1
    payload['ok'] = True
    payload['observed_states'] = observed_states
    payload['review_resumes'] = review_resumes
    payload['final_state'] = final_state
    payload['final_status'] = final_state.get('status')
    payload['summary'] = final_state.get('summary')
    payload['stop_reason'] = final_state.get('stop_reason')
    payload['termination_reason'] = final_state.get('termination_reason')
    payload['termination_detail'] = final_state.get('termination_detail')
    payload['best_round_index'] = final_state.get('best_round_index')
    payload['round_count'] = len(final_state.get('rounds') or [])
    payload['latest_round_card'] = final_state.get('latest_round_card')
    payload['final_summary'] = final_state.get('final_summary')
    if output_path:
        payload['output_path'] = str(write_payload(output_path, payload))
    return payload
