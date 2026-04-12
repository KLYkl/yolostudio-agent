from __future__ import annotations

import re
from pathlib import Path
from typing import Any

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_FLOAT = r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?'
EPOCH_LINE = re.compile(
    r'^\s*(?P<epoch>\d+)\/(?P<total>\d+)\s+'
    r'(?P<gpu_mem>\d+(?:\.\d+)?[GM])\s+'
    r'(?P<box_loss>' + _FLOAT + r')\s+'
    r'(?P<cls_loss>' + _FLOAT + r')\s+'
    r'(?P<dfl_loss>' + _FLOAT + r')'
)
EVAL_LINE = re.compile(
    r'^all\s+'
    r'(?P<images>\d+)\s+'
    r'(?P<instances>\d+)\s+'
    r'(?P<precision>' + _FLOAT + r')\s+'
    r'(?P<recall>' + _FLOAT + r')\s+'
    r'(?P<map50>' + _FLOAT + r')\s+'
    r'(?P<map>' + _FLOAT + r')\s*$'
)
COMPLETED_LINE = re.compile(r'^(?P<epochs>\d+)\s+epochs completed(?:\s+in\s+(?P<hours>' + _FLOAT + r')\s+hours\.)?')
RESULTS_SAVED_LINE = re.compile(r'^Results saved to\s+(?P<save_dir>.+)$')
VALIDATING_LINE = re.compile(r'^Validating\s+(?P<weights>.+?)\.\.\.$')


def _sanitize_line(raw_line: str) -> str:
    return ANSI_ESCAPE.sub('', raw_line).replace('\r', '').strip()


def _maybe_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None or value == '':
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_training_log(log_file: str | Path) -> dict[str, Any]:
    path = Path(log_file)
    if not path.exists():
        return {'ok': False, 'error': 'log file not found'}

    latest_train_metrics: dict[str, Any] | None = None
    latest_eval_metrics: dict[str, Any] | None = None
    current_epoch: int | None = None
    current_total_epochs: int | None = None
    observed_epochs: set[int] = set()
    completed_epochs: int | None = None
    completed_hours: float | None = None
    save_dir: str = ''
    validating_weights: str = ''
    has_traceback = False
    error_lines: list[str] = []

    for raw_line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = _sanitize_line(raw_line)
        if not line:
            continue

        completed_match = COMPLETED_LINE.match(line)
        if completed_match:
            completed_epochs = _maybe_int(completed_match.group('epochs'))
            completed_hours = _maybe_float(completed_match.group('hours'))
            continue

        saved_match = RESULTS_SAVED_LINE.match(line)
        if saved_match:
            save_dir = saved_match.group('save_dir').strip()
            continue

        validating_match = VALIDATING_LINE.match(line)
        if validating_match:
            validating_weights = validating_match.group('weights').strip()
            continue

        if line.startswith('Traceback (most recent call last):'):
            has_traceback = True
        if has_traceback and len(error_lines) < 8:
            error_lines.append(line)
        elif any(token in line for token in ('RuntimeError:', 'AssertionError:', 'ValueError:', 'Error:', 'Exception:')) and len(error_lines) < 8:
            error_lines.append(line)

        train_match = EPOCH_LINE.match(line)
        if train_match:
            current_epoch = int(train_match.group('epoch'))
            current_total_epochs = int(train_match.group('total'))
            observed_epochs.add(current_epoch)
            latest_train_metrics = {
                'epoch': current_epoch,
                'total_epochs': current_total_epochs,
                'gpu_mem': train_match.group('gpu_mem'),
                'box_loss': _maybe_float(train_match.group('box_loss')),
                'cls_loss': _maybe_float(train_match.group('cls_loss')),
                'dfl_loss': _maybe_float(train_match.group('dfl_loss')),
            }
            continue

        eval_match = EVAL_LINE.match(line)
        if eval_match:
            latest_eval_metrics = {
                'epoch': current_epoch,
                'total_epochs': current_total_epochs,
                'images': int(eval_match.group('images')),
                'instances': int(eval_match.group('instances')),
                'precision': _maybe_float(eval_match.group('precision')),
                'recall': _maybe_float(eval_match.group('recall')),
                'map50': _maybe_float(eval_match.group('map50')),
                'map': _maybe_float(eval_match.group('map')),
                'mAP50': _maybe_float(eval_match.group('map50')),
                'mAP50-95': _maybe_float(eval_match.group('map')),
            }
            continue

    merged_metrics: dict[str, Any] = {}
    if latest_train_metrics:
        merged_metrics.update(latest_train_metrics)
    if latest_eval_metrics:
        merged_metrics.update(latest_eval_metrics)

    has_eval_metrics = bool(latest_eval_metrics and latest_eval_metrics.get('precision') is not None)
    has_loss_metrics = bool(latest_train_metrics and any(latest_train_metrics.get(key) is not None for key in ('box_loss', 'cls_loss', 'dfl_loss')))
    epoch = _maybe_int(merged_metrics.get('epoch'))
    total_epochs = _maybe_int(merged_metrics.get('total_epochs'))
    progress_ratio = (float(epoch) / float(total_epochs)) if epoch and total_epochs else None

    signals: list[str] = []
    facts: list[str] = []
    if epoch is not None and total_epochs is not None:
        facts.append(f'训练进度 {epoch}/{total_epochs}')
        if progress_ratio is not None and progress_ratio < 0.2:
            signals.append('early_training_observation')
    if has_loss_metrics and not has_eval_metrics:
        signals.append('loss_only_metrics')
        signals.append('insufficient_eval_metrics')
        losses = []
        for name in ('box_loss', 'cls_loss', 'dfl_loss'):
            value = merged_metrics.get(name)
            if value is not None:
                losses.append(f'{name}={value:.3f}' if isinstance(value, float) else f'{name}={value}')
        if losses:
            facts.append('仅有训练损失: ' + ', '.join(losses))
    if has_eval_metrics:
        facts.append(
            '评估指标: '
            f"precision={merged_metrics.get('precision', 0):.3f}, "
            f"recall={merged_metrics.get('recall', 0):.3f}, "
            f"mAP50={merged_metrics.get('map50', 0):.3f}, "
            f"mAP50-95={merged_metrics.get('map', 0):.3f}"
        )
    if not has_loss_metrics and not has_eval_metrics:
        signals.append('metrics_missing')
    if not has_eval_metrics and has_loss_metrics:
        signals.append('missing_eval_metrics')
    if has_traceback or error_lines:
        signals.append('training_log_failed')
    if completed_epochs is not None:
        facts.append(f'日志显示训练完成 {completed_epochs} 轮')
    if completed_hours is not None:
        facts.append(f'完成耗时约 {completed_hours:.3f} 小时')
    if save_dir:
        facts.append(f'结果目录: {save_dir}')
    if validating_weights:
        facts.append(f'验证权重: {validating_weights}')

    return {
        'ok': True,
        'metrics': merged_metrics or None,
        'latest_train_metrics': latest_train_metrics,
        'latest_eval_metrics': latest_eval_metrics,
        'has_loss_metrics': has_loss_metrics,
        'has_eval_metrics': has_eval_metrics,
        'observed_epochs': sorted(observed_epochs),
        'progress': {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'progress_ratio': progress_ratio,
            'epoch_fraction': progress_ratio,
            'completed_epochs': completed_epochs,
        },
        'signals': list(dict.fromkeys(signals)),
        'facts': facts,
        'run_state_hint': 'failed' if (has_traceback or error_lines) else ('completed' if completed_epochs is not None else ('in_progress' if epoch is not None else 'unavailable')),
        'save_dir': save_dir,
        'validating_weights': validating_weights,
        'error_lines': error_lines,
    }


def parse_latest_metrics(log_file: str | Path) -> dict[str, Any]:
    parsed = parse_training_log(log_file)
    if not parsed.get('ok'):
        return parsed
    return {'ok': True, 'metrics': parsed.get('metrics')}
