from __future__ import annotations

from typing import Any, Awaitable, Callable

from yolostudio_agent.agent.client.training_contracts import (
    TrainingPlanFollowupAction,
    TrainingRecoveryBootstrap,
)
from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.training_plan_service import (
    build_training_preflight_tool_args,
    resolve_training_start_args,
)
from yolostudio_agent.agent.client.training_request_service import (
    DatasetPathExtractor,
    LocalPathExistenceChecker,
    PrepareOnlyRequestChecker,
    ToolResultMessageRenderer,
    TrainingArgsCollector,
    run_prepare_only_flow,
)

DirectToolInvoker = Callable[..., Awaitable[dict[str, Any]]]
TrainingPlanDraftBuilder = Callable[..., dict[str, Any]]
TrainingPlanMessageRenderer = Callable[[dict[str, Any], bool], Awaitable[str]]


def build_training_recovery_base_args(session_state: SessionState) -> dict[str, Any]:
    active_training = session_state.active_training
    base_args = dict((active_training.last_start_result or {}).get('resolved_args') or {})
    if not str(base_args.get('model') or '').strip():
        base_args['model'] = str(active_training.model or '').strip()
    if not str(base_args.get('data_yaml') or '').strip():
        base_args['data_yaml'] = str(active_training.data_yaml or session_state.active_dataset.data_yaml or '').strip()
    if not str(base_args.get('training_environment') or '').strip() and str(active_training.training_environment or '').strip():
        base_args['training_environment'] = str(active_training.training_environment).strip()
    return base_args


def resolve_training_recovery_bootstrap(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    latest_dataset_path: str,
    explicit_run_ids: list[str] | None,
    requested_execute: bool,
    wants_repeat_prepare: bool,
    wants_retry_last_plan: bool,
    wants_resume_recent_training: bool,
    wants_analysis_only: bool,
) -> TrainingRecoveryBootstrap | None:
    explicit_run_ids = list(explicit_run_ids or [])
    active_training = session_state.active_training

    if active_training.running and explicit_run_ids and wants_resume_recent_training:
        return {
            'reply': '当前已有活动训练；如果你想恢复或切换到另一个历史 run，请先停止当前训练，再明确要恢复的 run。',
            'defer_to_graph': False,
            'proceed': False,
        }

    running_hot_update_intent = (
        active_training.running
        and not latest_dataset_path
        and not explicit_run_ids
        and not any(token in user_text for token in ('预测', '推理', '识别', '视频', '图片'))
        and not any(token in user_text for token in ('第几轮', '训练到哪了', '训练到第几轮', '跑到第几轮', '训练状态', '训练进度', '当前进度', '还在训练吗', '还在跑吗', '现在状态'))
        and (
            any(token in normalized_text for token in ('batch', 'device', 'epochs', 'optimizer', 'freeze', 'lr0', 'resume', 'imgsz', 'fraction', 'classes', 'single_cls'))
            or any(token in user_text for token in ('轮数', '轮', '优化器', '冻结', '学习率', '环境', 'GPU', '显卡'))
        )
    )
    if running_hot_update_intent:
        return {
            'reply': (
                '当前训练还在运行，不能直接热更新 batch、轮数、优化器或设备等核心参数。'
                '如果要改参数，请先停止当前训练，再生成新的训练计划。'
            ),
            'defer_to_graph': False,
            'proceed': False,
        }

    if requested_execute and active_training.running:
        return {
            'reply': '当前训练已经在运行；如果要新开训练，请先停止当前训练，或明确给出新的数据集和模型。',
            'defer_to_graph': False,
            'proceed': False,
        }

    readiness = dict(session_state.active_dataset.last_readiness or {})
    data_yaml = str(session_state.active_dataset.data_yaml or '').strip()
    if wants_repeat_prepare and readiness.get('ready') and data_yaml:
        return {
            'reply': f'当前数据集已经准备完成：{data_yaml}；不需要重复 prepare。你可以直接继续训练或重新规划。',
            'defer_to_graph': False,
            'proceed': False,
        }

    if not (wants_retry_last_plan or wants_resume_recent_training):
        return None

    if wants_analysis_only:
        if active_training.training_run_summary or active_training.last_summary or active_training.last_status:
            return {
                'reply': '',
                'defer_to_graph': True,
                'proceed': False,
            }
        return None

    base_args = build_training_recovery_base_args(session_state)
    dataset_path = str(session_state.active_dataset.dataset_root or session_state.active_dataset.img_dir or '').strip()
    if not dataset_path and str(base_args.get('data_yaml') or '').strip():
        dataset_path = str(session_state.active_dataset.dataset_root or session_state.active_dataset.img_dir or '').strip()
    if not str(base_args.get('model') or '').strip() or not str(base_args.get('data_yaml') or '').strip():
        return {
            'reply': '当前缺少足够的历史训练参数，暂时不能直接恢复这次训练计划；请先明确数据集和模型。',
            'defer_to_graph': False,
            'proceed': False,
        }

    run_state = str(
        (active_training.training_run_summary or {}).get('run_state')
        or (active_training.last_summary or {}).get('run_state')
        or (active_training.last_status or {}).get('run_state')
        or ''
    ).strip().lower()
    if wants_resume_recent_training:
        if run_state != 'stopped':
            return {
                'reply': '当前只有已停止的训练才适合按最近状态继续；失败或已完成的训练更适合按原计划重试或重新规划。',
                'defer_to_graph': False,
                'proceed': False,
            }
        base_args['resume'] = True
    else:
        base_args['resume'] = False

    return {
        'reply': '',
        'defer_to_graph': False,
        'proceed': True,
        'base_args': base_args,
        'dataset_path': dataset_path,
    }


def resolve_training_recovery_followup_action(
    *,
    bootstrap: TrainingRecoveryBootstrap | None,
    plan_result: dict[str, Any] | None = None,
) -> TrainingPlanFollowupAction:
    bootstrap = dict(bootstrap or {})
    if not bootstrap:
        return {'action': 'none'}
    if bootstrap.get('defer_to_graph'):
        return {'action': 'defer_to_graph'}
    if not bootstrap.get('proceed'):
        return {'action': 'reply', 'reply': str(bootstrap.get('reply') or '')}
    if plan_result is None:
        return {'action': 'build_plan'}
    plan_result = dict(plan_result or {})
    action = 'save_draft_and_reply'
    if plan_result.get('defer_to_graph'):
        action = 'save_draft_and_handoff'
    return {
        'action': action,
        'draft': dict(plan_result.get('draft') or {}),
        'reply': str(plan_result.get('reply') or ''),
    }


async def run_training_recovery_bootstrap_flow(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    latest_dataset_path: str,
    explicit_run_ids: list[str] | None,
    requested_execute: bool,
    wants_repeat_prepare: bool,
    wants_retry_last_plan: bool,
    wants_resume_recent_training: bool,
    wants_analysis_only: bool,
    direct_tool: DirectToolInvoker,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> TrainingPlanFollowupAction:
    bootstrap = resolve_training_recovery_bootstrap(
        session_state=session_state,
        user_text=user_text,
        normalized_text=normalized_text,
        latest_dataset_path=latest_dataset_path,
        explicit_run_ids=explicit_run_ids,
        requested_execute=requested_execute,
        wants_repeat_prepare=wants_repeat_prepare,
        wants_retry_last_plan=wants_retry_last_plan,
        wants_resume_recent_training=wants_resume_recent_training,
        wants_analysis_only=wants_analysis_only,
    )
    followup_action = resolve_training_recovery_followup_action(bootstrap=bootstrap)
    if str(followup_action.get('action') or '').strip() != 'build_plan':
        return followup_action
    bootstrap = dict(bootstrap or {})
    plan_result = await run_training_recovery_entrypoint(
        session_state=session_state,
        user_text=user_text,
        dataset_path=str(bootstrap.get('dataset_path') or ''),
        base_args=dict(bootstrap.get('base_args') or {}),
        direct_tool=direct_tool,
        build_training_plan_draft_fn=build_training_plan_draft_fn,
        render_training_plan_message=render_training_plan_message,
    )
    return resolve_training_recovery_followup_action(
        bootstrap=bootstrap,
        plan_result=plan_result,
    )


async def run_training_plan_bootstrap_flow(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    latest_dataset_path: str,
    explicit_run_ids: list[str] | None,
    requested_execute: bool,
    wants_repeat_prepare: bool,
    wants_retry_last_plan: bool,
    wants_resume_recent_training: bool,
    wants_analysis_only: bool,
    looks_like_prepare_only_request: PrepareOnlyRequestChecker,
    extract_dataset_path: DatasetPathExtractor,
    local_path_exists: LocalPathExistenceChecker,
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_tool_result_message: ToolResultMessageRenderer,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> TrainingPlanFollowupAction | None:
    prepare_only_followup = await run_prepare_only_flow(
        user_text=user_text,
        looks_like_prepare_only_request=looks_like_prepare_only_request,
        extract_dataset_path=extract_dataset_path,
        local_path_exists=local_path_exists,
        direct_tool=direct_tool,
        collect_requested_training_args=collect_requested_training_args,
        build_training_plan_draft_fn=build_training_plan_draft_fn,
        render_tool_result_message=render_tool_result_message,
    )
    if prepare_only_followup is not None:
        result = dict(prepare_only_followup)
        if str(result.get('action') or '').strip() == 'save_draft_and_handoff':
            result['handoff_mode'] = 'defer'
        return result

    recovery_followup = await run_training_recovery_bootstrap_flow(
        session_state=session_state,
        user_text=user_text,
        normalized_text=normalized_text,
        latest_dataset_path=latest_dataset_path,
        explicit_run_ids=explicit_run_ids,
        requested_execute=requested_execute,
        wants_repeat_prepare=wants_repeat_prepare,
        wants_retry_last_plan=wants_retry_last_plan,
        wants_resume_recent_training=wants_resume_recent_training,
        wants_analysis_only=wants_analysis_only,
        direct_tool=direct_tool,
        build_training_plan_draft_fn=build_training_plan_draft_fn,
        render_training_plan_message=render_training_plan_message,
    )
    result = dict(recovery_followup or {})
    if str(result.get('action') or '').strip() == 'save_draft_and_handoff':
        result['handoff_mode'] = 'handoff'
    return result


async def run_training_recovery_orchestration(
    *,
    user_text: str,
    dataset_path: str,
    readiness: dict[str, Any] | None,
    base_args: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> dict[str, Any]:
    readiness = dict(readiness or {})
    base_args = dict(base_args or {})
    preflight_args = build_training_preflight_tool_args(base_args)
    preflight = await direct_tool('training_preflight', **preflight_args)
    next_args = resolve_training_start_args(base_args, preflight)
    ready_to_start = bool(preflight.get('ready_to_start'))
    draft = build_training_plan_draft_fn(
        user_text=user_text,
        dataset_path=dataset_path,
        readiness=readiness,
        preflight=preflight,
        next_tool_name='start_training' if ready_to_start else '',
        next_tool_args=next_args if ready_to_start else {},
        planned_training_args=next_args,
    )
    return {
        'draft': draft,
        'reply': await render_training_plan_message(draft, pending=ready_to_start),
        'defer_to_graph': ready_to_start,
    }


async def run_training_recovery_entrypoint(
    *,
    session_state: SessionState,
    user_text: str,
    dataset_path: str,
    base_args: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> dict[str, Any]:
    readiness = dict(session_state.active_dataset.last_readiness or {})
    if dataset_path and not readiness:
        readiness = await direct_tool('training_readiness', img_dir=dataset_path)
    if not (session_state.active_training.last_environment_probe or {}).get('environments'):
        await direct_tool('list_training_environments')
    return await run_training_recovery_orchestration(
        user_text=user_text,
        dataset_path=dataset_path,
        readiness=readiness,
        base_args=base_args,
        direct_tool=direct_tool,
        build_training_plan_draft_fn=build_training_plan_draft_fn,
        render_training_plan_message=render_training_plan_message,
    )
