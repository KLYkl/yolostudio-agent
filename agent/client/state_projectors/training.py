from __future__ import annotations

from typing import Any

from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.state_projectors.common import (
    _apply_resolved_training_args,
    _apply_training_command_overrides,
    _training_environment_probe_snapshot,
    _training_loop_request_snapshot,
    _training_loop_status_snapshot,
    _training_preflight_snapshot,
    _training_run_summary_snapshot,
    _training_start_snapshot,
    _training_status_snapshot,
)


def _training_run_weight_path(result: dict[str, Any]) -> str:
    summary_overview = dict(result.get('summary_overview') or {})
    return str(
        result.get('best_weight_path')
        or result.get('weights_path')
        or result.get('weight_path')
        or summary_overview.get('best_weight_path')
        or summary_overview.get('weights_path')
        or ''
    ).strip()


def apply_training_tool_result(
    session_state: SessionState,
    tool_name: str,
    result: dict[str, Any],
    tool_args: dict[str, Any],
) -> None:
    ds = session_state.active_dataset
    tr = session_state.active_training
    if tool_name == 'start_training' and result.get('ok'):
        tr.running = True
        resolved_args = result.get('resolved_args') or {}
        tr.device = result.get('device', '')
        _apply_resolved_training_args(
            tr,
            ds,
            resolved_args=resolved_args,
            training_environment=result.get('training_environment') or {},
            tool_args=tool_args,
            prefer_tool_args_for_start=True,
            force_assign=True,
        )
        tr.pid = result.get('pid')
        tr.log_file = result.get('log_file', '')
        tr.started_at = result.get('started_at')
        tr.last_start_result = _training_start_snapshot(result)
        tr.last_summary = {}
        tr.training_run_summary = {}
    elif tool_name == 'list_training_environments' and result.get('ok'):
        tr.last_environment_probe = _training_environment_probe_snapshot(result)
        default_environment = tr.last_environment_probe.get('default_environment') or {}
        if default_environment:
            tr.training_environment = str(default_environment.get('display_name') or default_environment.get('name') or tr.training_environment)
    elif tool_name == 'training_preflight' and result.get('ok'):
        tr.last_preflight = _training_preflight_snapshot(result)
        _apply_resolved_training_args(
            tr,
            ds,
            resolved_args=result.get('resolved_args') or {},
            training_environment=result.get('training_environment') or {},
        )
    elif tool_name == 'list_training_runs' and result.get('ok'):
        tr.recent_runs = list(result.get('runs') or [])
    elif tool_name == 'inspect_training_run' and result.get('ok'):
        tr.last_run_inspection = result
        if result.get('selected_run_id'):
            matched = next(
                (
                    item for item in tr.recent_runs
                    if str(item.get('run_id') or '') == str(result.get('selected_run_id') or '')
                ),
                None,
            )
            if matched is None:
                tr.recent_runs = [result, *tr.recent_runs[:9]]
        inspected_run_id = str(result.get('selected_run_id') or result.get('run_id') or '').strip()
        inspected_weight_path = _training_run_weight_path(result)
        if inspected_run_id and inspected_weight_path and tr.best_run_selection:
            best_selection = dict(tr.best_run_selection or {})
            best_run = dict(best_selection.get('best_run') or {})
            best_run_id = str(best_run.get('run_id') or best_selection.get('best_run_id') or '').strip()
            if best_run_id and best_run_id == inspected_run_id:
                best_run['run_id'] = inspected_run_id
                best_run['best_weight_path'] = inspected_weight_path
                best_selection['best_run'] = best_run
                best_selection['best_weight_path'] = inspected_weight_path
                best_selection['resolved_weight_path'] = inspected_weight_path
                tr.best_run_selection = best_selection
    elif tool_name == 'compare_training_runs' and result.get('ok'):
        tr.last_run_comparison = result
    elif tool_name == 'select_best_training_run' and result.get('ok'):
        tr.best_run_selection = result
    elif tool_name == 'start_training_loop' and result.get('ok'):
        _apply_resolved_training_args(
            tr,
            ds,
            resolved_args=tool_args or {},
            training_environment={},
            tool_args=tool_args,
            prefer_tool_args_for_start=True,
            assign_data_yaml=False,
        )
        tr.device = str((tool_args or {}).get('device') or tr.device)
        tr.active_loop_request = _training_loop_request_snapshot(tool_args)
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_name = str(result.get('loop_name') or tr.active_loop_name)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
        tr.last_loop_detail = {}
    elif tool_name == 'list_training_loops' and result.get('ok'):
        tr.recent_loops = list(result.get('loops') or [])
        if result.get('active_loop_id'):
            tr.active_loop_id = str(result.get('active_loop_id') or tr.active_loop_id)
    elif tool_name == 'check_training_loop_status' and result.get('ok'):
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_name = str(result.get('loop_name') or tr.active_loop_name)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
    elif tool_name == 'inspect_training_loop' and result.get('ok'):
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_name = str(result.get('loop_name') or tr.active_loop_name)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
        tr.last_loop_detail = _training_loop_status_snapshot(result, include_detail=True)
    elif tool_name in {'pause_training_loop', 'resume_training_loop', 'stop_training_loop'} and result.get('ok'):
        tr.active_loop_id = str(result.get('loop_id') or tr.active_loop_id)
        tr.active_loop_status = str(result.get('status') or tr.active_loop_status)
        tr.last_loop_status = _training_loop_status_snapshot(result)
    elif tool_name == 'check_training_status':
        tr.last_status = _training_status_snapshot(result)
        is_running = bool(result.get('running'))
        tr.running = is_running
        _apply_resolved_training_args(
            tr,
            ds,
            resolved_args=result.get('resolved_args') or {},
            training_environment=result.get('training_environment') or {},
        )
        tr.device = str(result.get('device') or tr.device)
        tr.log_file = str(result.get('log_file') or tr.log_file)
        tr.started_at = result.get('started_at', tr.started_at)
        _apply_training_command_overrides(tr, ds, result.get('command') or [])
        tr.pid = result.get('pid', tr.pid) if is_running else None
    elif tool_name == 'summarize_training_run' and result.get('ok'):
        summary_snapshot = _training_run_summary_snapshot(result)
        tr.last_summary = summary_snapshot
        tr.training_run_summary = summary_snapshot
        tr.running = str(result.get('run_state') or '').strip().lower() == 'running'
        if result.get('log_file'):
            tr.log_file = str(result.get('log_file'))
        if result.get('latest_metrics'):
            tr.last_status = {
                **tr.last_status,
                'run_state': result.get('run_state'),
                'progress': result.get('progress'),
                'latest_metrics': result.get('latest_metrics'),
                'analysis_ready': result.get('analysis_ready'),
                'minimum_facts_ready': result.get('minimum_facts_ready'),
                'signals': result.get('signals'),
                'facts': result.get('facts'),
            }
    elif tool_name == 'stop_training' and result.get('ok'):
        tr.running = False
        tr.pid = None
        tr.log_file = ''
        tr.started_at = None
        tr.last_status = _training_status_snapshot(result)
