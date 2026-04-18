from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from yolostudio_agent.agent.client.training_plan_service import (
    build_training_preflight_tool_args,
    resolve_training_start_args,
)

DirectToolInvoker = Callable[..., Awaitable[dict[str, Any]]]
TrainingArgsCollector = Callable[..., dict[str, Any]]
TrainingPlanDraftBuilder = Callable[..., dict[str, Any]]
PrepareFollowupMessageRenderer = Callable[[dict[str, Any], dict[str, Any]], Awaitable[str]]
SleepInvoker = Callable[[float], Awaitable[Any]]


async def run_post_prepare_training_start_flow(
    *,
    user_text: str,
    dataset_path: str,
    readiness: dict[str, Any] | None,
    synthetic_followup: dict[str, Any] | None,
    prepare_parsed: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_prepare_followup_message: PrepareFollowupMessageRenderer,
) -> dict[str, Any]:
    followup_args = dict((synthetic_followup or {}).get('args') or {})
    prepare_parsed = dict(prepare_parsed or {})
    preflight_args = build_training_preflight_tool_args(followup_args)
    preflight = await direct_tool('training_preflight', **preflight_args)
    resolved_followup_args = resolve_training_start_args(followup_args, preflight)
    draft = build_training_plan_draft_fn(
        user_text=user_text,
        dataset_path=dataset_path,
        readiness=readiness or {},
        preflight=preflight,
        next_tool_name='start_training' if preflight.get('ready_to_start') else '',
        next_tool_args=resolved_followup_args if preflight.get('ready_to_start') else {},
        planned_training_args=resolved_followup_args,
    )
    if not preflight.get('ready_to_start'):
        reply = await render_prepare_followup_message(prepare_parsed, preflight)
        return {
            'preflight': preflight,
            'draft': draft,
            'followup_action': {
                'action': 'save_draft_and_reply',
                'draft': draft,
                'reply': reply,
                'status': 'error',
            },
        }
    return {
        'preflight': preflight,
        'draft': draft,
        'followup_action': {
            'action': 'save_draft_and_handoff',
            'draft': draft,
            'reply': '',
        },
    }


async def run_remote_training_start_flow(
    *,
    pipeline_args: dict[str, Any] | None,
    resolved_inputs: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
) -> dict[str, Any]:
    pipeline_args = dict(pipeline_args or {})
    resolved_inputs = dict(resolved_inputs or {})
    dataset_path = str(resolved_inputs.get('dataset_path') or '')
    model_path = str(resolved_inputs.get('model_path') or '')
    readiness = await direct_tool('training_readiness', img_dir=dataset_path)
    if not readiness.get('ok'):
        return {
            'ok': False,
            'stage': 'readiness',
            'dataset_path': dataset_path,
            'model_path': model_path,
            'readiness': readiness,
            'prepare': {},
            'preflight': {},
            'start': {},
        }

    prepare_result: dict[str, Any] = {}
    data_yaml = str(readiness.get('resolved_data_yaml') or '').strip()
    if not readiness.get('ready'):
        prepare_args: dict[str, Any] = {'dataset_path': dataset_path}
        if pipeline_args.get('force_split'):
            prepare_args['force_split'] = True
        prepare_result = await direct_tool('prepare_dataset_for_training', **prepare_args)
        if not prepare_result.get('ok') or not prepare_result.get('ready'):
            return {
                'ok': False,
                'stage': 'prepare',
                'dataset_path': dataset_path,
                'model_path': model_path,
                'readiness': readiness,
                'prepare': prepare_result,
                'preflight': {},
                'start': {},
            }
        data_yaml = str(prepare_result.get('data_yaml') or '').strip()

    requested_args = collect_requested_training_args(
        str(pipeline_args.get('user_text') or ''),
        data_yaml=data_yaml,
    )
    requested_args['model'] = model_path
    requested_args['data_yaml'] = data_yaml
    requested_args['device'] = str(requested_args.get('device') or 'auto')
    requested_args['epochs'] = int(requested_args.get('epochs', 100))

    preflight_args = build_training_preflight_tool_args(requested_args)
    preflight = await direct_tool('training_preflight', **preflight_args)
    if not preflight.get('ok') or not preflight.get('ready_to_start'):
        return {
            'ok': False,
            'stage': 'preflight',
            'dataset_path': dataset_path,
            'model_path': model_path,
            'readiness': readiness,
            'prepare': prepare_result,
            'preflight': preflight,
            'start': {},
        }

    resolved_args = resolve_training_start_args(requested_args, preflight)
    start_result = await direct_tool('start_training', **resolved_args)
    return {
        'ok': bool(start_result.get('ok')),
        'stage': 'completed' if start_result.get('ok') else 'start',
        'dataset_path': dataset_path,
        'model_path': model_path,
        'readiness': readiness,
        'prepare': prepare_result,
        'preflight': preflight,
        'start': start_result,
    }


async def wait_for_remote_training_terminal_state(
    *,
    direct_tool: DirectToolInvoker,
    poll_interval_seconds: int = 15,
    max_wait_seconds: int = 7200,
    sleep: SleepInvoker = asyncio.sleep,
) -> dict[str, Any]:
    started = time.monotonic()
    status_checks: list[dict[str, Any]] = []
    interval = max(0, int(poll_interval_seconds))
    wait_limit = max(1, int(max_wait_seconds))

    while True:
        status_result = await direct_tool('check_training_status')
        status_checks.append({
            'summary': status_result.get('summary'),
            'running': status_result.get('running'),
            'run_state': status_result.get('run_state'),
            'save_dir': status_result.get('save_dir'),
            'log_file': status_result.get('log_file'),
        })
        if not status_result.get('ok'):
            return {
                'ok': False,
                'message': '训练已启动，但轮询训练状态失败；未执行自动回传。',
                'status_result': status_result,
                'status_checks': status_checks,
            }

        run_state = str(status_result.get('run_state') or '').strip().lower()
        if not status_result.get('running') and run_state not in {'', 'running'}:
            summary_result = await direct_tool('summarize_training_run')
            inspect_result = await direct_tool('inspect_training_run')
            return {
                'ok': True,
                'status_result': status_result,
                'summary_result': summary_result,
                'inspect_result': inspect_result,
                'status_checks': status_checks,
            }

        if (time.monotonic() - started) >= wait_limit:
            return {
                'ok': False,
                'timed_out': True,
                'message': f'训练已启动，但在等待窗口 {wait_limit}s 内仍未结束；未执行自动回传。',
                'status_result': status_result,
                'status_checks': status_checks,
            }

        if interval > 0:
            await sleep(interval)
