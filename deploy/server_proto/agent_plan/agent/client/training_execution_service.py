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
TrainingWaitInvoker = Callable[..., Awaitable[dict[str, Any]]]
RemoteResultPathResolver = Callable[..., str]


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


async def run_remote_training_pipeline_flow(
    *,
    pipeline_args: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    resolve_training_remote_inputs: Callable[[dict[str, Any]], dict[str, Any]],
    collect_requested_training_args: TrainingArgsCollector,
    wait_for_remote_training_terminal_state: TrainingWaitInvoker,
    resolve_remote_training_result_path: RemoteResultPathResolver,
) -> dict[str, Any]:
    pipeline_args = dict(pipeline_args or {})
    upload_args = dict(pipeline_args.get('upload_args') or {})
    upload_result = await direct_tool('upload_assets_to_remote', **upload_args)
    if not upload_result.get('ok'):
        return {
            'stage': 'upload',
            'upload': upload_result,
            'resolved_inputs': {},
            'start_flow': {},
            'wait': {},
            'download': {},
            'pipeline_result': {},
        }

    resolved_inputs = dict(resolve_training_remote_inputs(upload_result) or {})
    if not resolved_inputs.get('ok'):
        return {
            'stage': 'resolve',
            'upload': upload_result,
            'resolved_inputs': resolved_inputs,
            'start_flow': {},
            'wait': {},
            'download': {},
            'pipeline_result': {},
        }

    start_flow = await run_remote_training_start_flow(
        pipeline_args=pipeline_args,
        resolved_inputs=resolved_inputs,
        direct_tool=direct_tool,
        collect_requested_training_args=collect_requested_training_args,
    )
    start_result = dict(start_flow.get('start') or {})
    if not start_flow.get('ok'):
        return {
            'stage': 'start',
            'upload': upload_result,
            'resolved_inputs': resolved_inputs,
            'start_flow': start_flow,
            'wait': {},
            'download': {},
            'pipeline_result': {},
        }

    wait_result: dict[str, Any] = {}
    final_status: dict[str, Any] = {}
    final_summary: dict[str, Any] = {}
    final_inspection: dict[str, Any] = {}
    download_result: dict[str, Any] = {}
    remote_result_path = ''

    if start_result.get('ok') and pipeline_args.get('wait_for_completion'):
        poll_interval = pipeline_args.get('poll_interval_seconds', 15)
        max_wait = pipeline_args.get('max_wait_seconds', 7200)
        wait_result = await wait_for_remote_training_terminal_state(
            poll_interval_seconds=int(15 if poll_interval is None else poll_interval),
            max_wait_seconds=int(7200 if max_wait is None else max_wait),
        )
        final_status = dict(wait_result.get('status_result') or {})
        final_summary = dict(wait_result.get('summary_result') or {})
        final_inspection = dict(wait_result.get('inspect_result') or {})
        remote_result_path = str(resolve_remote_training_result_path(
            start_result=start_result,
            status_result=final_status,
            summary_result=final_summary,
            inspection_result=final_inspection,
        ) or '')
        if wait_result.get('ok') and pipeline_args.get('download_after_completion'):
            if remote_result_path:
                download_args = {
                    'remote_paths': [remote_result_path],
                    'server': upload_args.get('server', ''),
                    'profile': upload_args.get('profile', ''),
                    'host': upload_args.get('host', ''),
                    'username': upload_args.get('username', ''),
                    'port': upload_args.get('port', 0),
                    'local_root': pipeline_args.get('local_result_root', ''),
                    'recursive': True,
                }
                download_result = await direct_tool('download_assets_from_remote', **download_args)
            else:
                download_result = {
                    'ok': False,
                    'summary': '训练已结束，但当前无法解析远端结果目录，未执行自动回传。',
                    'error': 'missing_remote_result_path',
                }

    readiness = dict(start_flow.get('readiness') or {})
    prepare_result = dict(start_flow.get('prepare') or {})
    preflight = dict(start_flow.get('preflight') or {})
    dataset_path = str(start_flow.get('dataset_path') or resolved_inputs.get('dataset_path') or '')
    model_path = str(start_flow.get('model_path') or resolved_inputs.get('model_path') or '')
    final_run_state = str(
        (final_summary.get('run_state') if isinstance(final_summary, dict) else '')
        or (final_status.get('run_state') if isinstance(final_status, dict) else '')
        or ''
    ).strip().lower()
    wait_required = bool(pipeline_args.get('wait_for_completion'))
    wait_ok = True
    if wait_required:
        wait_ok = bool(wait_result.get('ok')) and final_run_state == 'completed'
    download_required = bool(pipeline_args.get('download_after_completion'))
    download_ok = (not download_required) or bool(download_result.get('ok'))

    pipeline_result = {
        'ok': start_result.get('ok') is True and wait_ok and download_ok,
        'upload': upload_result,
        'readiness': readiness,
        'prepare': prepare_result,
        'preflight': preflight,
        'start': start_result,
        'wait': wait_result,
        'final_status': final_status,
        'final_summary': final_summary,
        'final_inspection': final_inspection,
        'download': download_result,
        'remote_dataset_path': dataset_path,
        'remote_model_path': model_path,
        'remote_result_path': remote_result_path,
        'local_result_root': str((download_result or {}).get('local_root') or pipeline_args.get('local_result_root') or ''),
        'wait_for_completion': wait_required,
        'download_after_completion': download_required,
        'final_run_state': final_run_state,
    }
    pipeline_result['pipeline_overview'] = {
        'target_label': str(upload_result.get('target_label') or upload_args.get('server') or '').strip(),
        'remote_root': str(upload_result.get('remote_root') or upload_args.get('remote_root') or '').strip(),
        'remote_dataset_path': pipeline_result['remote_dataset_path'],
        'remote_model_path': pipeline_result['remote_model_path'],
        'remote_result_path': pipeline_result['remote_result_path'],
        'local_result_root': pipeline_result['local_result_root'],
    }
    pipeline_result['execution_overview'] = {
        'upload_ok': bool(upload_result.get('ok')),
        'readiness_ok': bool(readiness.get('ok')),
        'prepare_ok': bool((not prepare_result) or prepare_result.get('ok')),
        'preflight_ok': bool(preflight.get('ok')),
        'start_ok': bool(start_result.get('ok')),
        'wait_ok': bool(wait_ok),
        'download_ok': bool(download_ok),
        'final_run_state': final_run_state,
    }
    action_candidates: list[dict[str, Any]] = []
    if pipeline_result['local_result_root']:
        action_candidates.append({
            'tool': 'summarize_training_run',
            'description': f"可继续查看本机训练产物目录: {pipeline_result['local_result_root']}",
        })
    elif pipeline_result['remote_result_path']:
        action_candidates.append({
            'tool': 'download_assets_from_remote',
            'description': f"如需回传，可继续下载远端训练目录: {pipeline_result['remote_result_path']}",
        })
    if final_run_state == 'completed':
        action_candidates.append({
            'tool': 'summarize_training_run',
            'description': '可继续查看训练总结或下一步建议',
        })
    if action_candidates:
        pipeline_result['action_candidates'] = action_candidates[:4]

    return {
        'stage': 'completed',
        'upload': upload_result,
        'resolved_inputs': resolved_inputs,
        'start_flow': start_flow,
        'wait': wait_result,
        'download': download_result,
        'pipeline_result': pipeline_result,
    }
