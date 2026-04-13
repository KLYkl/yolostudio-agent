from __future__ import annotations

from pathlib import Path
from typing import Any


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _normalize_status_source(status: dict[str, Any]) -> str:
    if status.get('running'):
        return 'active'
    if status.get('last_run'):
        return 'last_run'
    if status.get('log_file'):
        return 'log_only'
    return 'unavailable'


def _normalize_run_state(status: dict[str, Any], parsed_log: dict[str, Any]) -> str:
    if status.get('running'):
        return 'running'
    if not status.get('log_file') and status.get('return_code') is None:
        return 'unavailable'
    if status.get('stop_reason') == 'manual_stop':
        return 'stopped'
    if parsed_log.get('run_state_hint') == 'failed':
        return 'failed'
    if parsed_log.get('run_state_hint') == 'completed' or status.get('return_code') == 0:
        return 'completed'
    if status.get('return_code') is not None and status.get('return_code') != 0:
        return 'failed'
    if parsed_log.get('run_state_hint') == 'in_progress':
        return 'running'
    return 'stopped'


def _resolve_save_dir(status: dict[str, Any], parsed_log: dict[str, Any]) -> str:
    save_dir = str(parsed_log.get('save_dir') or '').strip()
    if save_dir:
        return save_dir
    resolved_args = status.get('resolved_args') or {}
    project = str(resolved_args.get('project') or '').strip()
    name = str(resolved_args.get('name') or '').strip()
    if project and name:
        return str(Path(project) / name)
    return ''


def _derive_observation_stage(run_state: str, progress: dict[str, Any]) -> str:
    if run_state in {'completed', 'stopped', 'failed'}:
        return 'final'

    epoch = progress.get('epoch')
    total_epochs = progress.get('total_epochs')
    progress_ratio = progress.get('progress_ratio')
    if progress_ratio is None:
        progress_ratio = progress.get('epoch_fraction')

    try:
        epoch_value = int(epoch) if epoch is not None else None
    except Exception:
        epoch_value = None

    try:
        ratio_value = float(progress_ratio) if progress_ratio is not None else None
    except Exception:
        ratio_value = None

    if epoch_value is None:
        return 'early'
    if epoch_value < 3:
        return 'early'
    if ratio_value is not None and ratio_value >= 0.8:
        return 'late'
    return 'mid'


def build_training_facts(status: dict[str, Any], parsed_log: dict[str, Any] | None) -> dict[str, Any]:
    parsed = dict(parsed_log or {})
    metrics = dict(parsed.get('metrics') or ((status.get('latest_metrics') or {}).get('metrics') or {}))
    progress = dict(parsed.get('progress') or {})
    signals = list(parsed.get('signals') or [])
    parsed_facts = [str(item).strip() for item in (parsed.get('facts') or []) if str(item).strip()]
    facts = list(parsed_facts)
    run_state = _normalize_run_state(status, parsed)
    status_source = _normalize_status_source(status)
    save_dir = _resolve_save_dir(status, parsed)

    if run_state == 'running':
        signals.append('training_running')
    elif run_state == 'completed':
        signals.append('training_completed')
    elif run_state == 'stopped':
        signals.append('training_stopped')
    elif run_state == 'failed':
        signals.append('training_failed')
    else:
        signals.append('training_unavailable')

    has_eval_metrics = bool(parsed.get('has_eval_metrics'))
    has_loss_metrics = bool(parsed.get('has_loss_metrics'))
    analysis_ready = bool(has_eval_metrics)
    minimum_facts_ready = bool(has_eval_metrics or has_loss_metrics or parsed_facts or parsed.get('error_lines'))

    epoch = progress.get('epoch')
    total_epochs = progress.get('total_epochs')
    progress_ratio = progress.get('progress_ratio')
    if progress_ratio is None:
        progress_ratio = progress.get('epoch_fraction')
    if progress_ratio is None and epoch is not None and total_epochs:
        try:
            progress_ratio = float(epoch) / float(total_epochs)
        except Exception:
            progress_ratio = None

    latest_metrics = {
        'ok': bool(metrics),
        'metrics': metrics or None,
    }
    progress_payload = {
        'epoch': epoch,
        'total_epochs': total_epochs,
        'progress_ratio': progress_ratio,
        'epoch_fraction': progress_ratio,
        'completed_epochs': progress.get('completed_epochs'),
    }
    observation_stage = _derive_observation_stage(run_state, progress_payload)

    if status.get('return_code') is not None:
        facts.append(f"return_code={status.get('return_code')}")
    if status.get('stop_reason'):
        facts.append(f"stop_reason={status.get('stop_reason')}")
    if status.get('device'):
        facts.append(f"device={status.get('device')}")
    if save_dir:
        facts.append(f"save_dir={save_dir}")

    next_actions: list[str] = []
    if run_state == 'running':
        if observation_stage == 'early':
            next_actions = ['当前仍属早期观察，建议继续训练到至少更多 epoch 后再下结论', '可继续调用 check_training_status 或 summarize_training_run 观察进度']
        elif observation_stage == 'late':
            next_actions = ['当前已接近结束，可做阶段性分析，但仍不要当成最终结论', '如需阶段性解释，可继续调用 analyze_training_outcome']
        else:
            next_actions = ['当前可做阶段性分析，但仍建议结合后续 epoch 继续观察', '可继续调用 check_training_status 或 summarize_training_run 观察进度']
    elif run_state == 'failed':
        next_actions = ['优先查看训练日志中的报错信息', '确认环境、模型文件和数据配置后再决定是否重试']
    elif analysis_ready:
        next_actions = ['可继续调用 analyze_training_outcome 解释训练效果', '如需行动建议，可调用 recommend_next_training_step']
    elif has_loss_metrics:
        next_actions = ['当前只有训练损失，建议继续训练或确认验证集/评估指标是否有产出', '补齐 precision/recall/mAP 后再做效果判断']
    elif minimum_facts_ready:
        next_actions = ['当前训练事实仍不完整，建议继续补齐指标或日志', '暂时不要只凭局部现象做大幅调参']
    elif run_state in {'completed', 'stopped'}:
        next_actions = ['当前已有训练记录，但缺少可分析日志或指标', '优先确认 log_file 与结果目录是否仍然可读取']
    else:
        next_actions = ['当前没有可分析的训练事实，先检查是否真的启动过训练', '必要时重新运行 check_training_status']

    return {
        'run_state': run_state,
        'status_source': status_source,
        'model_family': 'yolo',
        'task_type': 'detection',
        'metrics': metrics,
        'signals': _dedupe(signals),
        'facts': _dedupe(facts),
        'progress': progress_payload,
        'observation_stage': observation_stage,
        'latest_metrics': latest_metrics,
        'analysis_ready': analysis_ready,
        'minimum_facts_ready': minimum_facts_ready,
        'has_eval_metrics': has_eval_metrics,
        'has_loss_metrics': has_loss_metrics,
        'latest_eval_metrics': parsed.get('latest_eval_metrics') or None,
        'latest_train_metrics': parsed.get('latest_train_metrics') or None,
        'error_lines': parsed.get('error_lines') or [],
        'save_dir': save_dir or None,
        'next_actions': next_actions,
    }


def summarize_training_run(status: dict[str, Any], parsed_log: dict[str, Any] | None) -> dict[str, Any]:
    facts = build_training_facts(status, parsed_log)
    run_state = facts['run_state']
    observation_stage = facts['observation_stage']
    if run_state == 'completed' and facts['analysis_ready']:
        summary = '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。'
    elif run_state == 'completed' and facts['minimum_facts_ready']:
        summary = '训练结果汇总: 最近一次训练已完成，但当前只有部分可读事实，暂时不能下可靠结论。'
    elif run_state == 'completed':
        summary = '训练结果汇总: 最近一次训练已完成，但当前缺少可分析日志或指标。'
    elif run_state == 'running' and facts['analysis_ready']:
        if observation_stage == 'late':
            summary = '训练结果汇总: 训练仍在运行，当前已接近结束，可做阶段性分析。'
        elif observation_stage == 'mid':
            summary = '训练结果汇总: 训练仍在运行，当前已进入中段观察，可做阶段性分析。'
        else:
            summary = '训练结果汇总: 训练仍在运行，虽已出现指标，但当前仍属早期观察。'
    elif run_state == 'running':
        if observation_stage == 'late':
            summary = '训练结果汇总: 训练仍在运行，当前已接近结束，但事实仍不完整。'
        elif observation_stage == 'mid':
            summary = '训练结果汇总: 训练仍在运行，当前已进入中段观察，但事实仍不完整。'
        else:
            summary = '训练结果汇总: 训练仍在运行，但当前还只有早期或不完整事实。'
    elif run_state == 'failed' and facts['minimum_facts_ready']:
        summary = '训练结果汇总: 最近一次训练未正常完成，但仍有部分可读事实，应先排查错误。'
    elif run_state == 'failed':
        summary = '训练结果汇总: 最近一次训练未正常完成，且当前缺少可读日志事实。'
    elif run_state == 'stopped' and facts['minimum_facts_ready']:
        summary = '训练结果汇总: 最近一次训练已停止，当前只有部分训练事实。'
    elif run_state == 'stopped':
        summary = '训练结果汇总: 最近一次训练已停止，但当前缺少可分析日志或指标。'
    else:
        summary = '训练结果汇总: 当前没有可用训练结果。'

    return {
        'ok': True,
        'summary': summary,
        'run_state': run_state,
        'status_source': facts['status_source'],
        'model_family': facts['model_family'],
        'task_type': facts['task_type'],
        'metrics': facts['metrics'],
        'signals': facts['signals'],
        'facts': facts['facts'],
        'progress': facts['progress'],
        'observation_stage': observation_stage,
        'latest_metrics': facts['latest_metrics'],
        'analysis_ready': facts['analysis_ready'],
        'minimum_facts_ready': facts['minimum_facts_ready'],
        'has_eval_metrics': facts['has_eval_metrics'],
        'has_loss_metrics': facts['has_loss_metrics'],
        'latest_eval_metrics': facts['latest_eval_metrics'],
        'latest_train_metrics': facts['latest_train_metrics'],
        'error_lines': facts['error_lines'],
        'save_dir': facts['save_dir'],
        'log_file': status.get('log_file'),
        'return_code': status.get('return_code'),
        'stop_reason': status.get('stop_reason'),
        'next_actions': facts['next_actions'],
    }
