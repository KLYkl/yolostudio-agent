from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.training_contracts import (
    PrepareOnlyRequestContext,
    PrepareOnlyResult,
    TrainingPlanFollowupAction,
    TrainingRequestContext,
    TrainingRequestGuard,
    TrainingRequestResult,
)
from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.training_plan_service import run_training_request_orchestration

DirectToolInvoker = Callable[..., Awaitable[dict[str, Any]]]
TrainingArgsCollector = Callable[..., dict[str, Any]]
TrainingDiscussionChecker = Callable[[str], bool]
TrainingExecutionBackendExtractor = Callable[[str], str]
TrainingPlanDraftBuilder = Callable[..., dict[str, Any]]
TrainingPlanMessageRenderer = Callable[[dict[str, Any], bool], Awaitable[str]]
ToolResultMessageRenderer = Callable[[str, dict[str, Any]], Awaitable[str]]
ModelPathExtractor = Callable[[str], str]
PrepareOnlyRequestChecker = Callable[[str], bool]
DatasetPathExtractor = Callable[[str], str]
LocalPathExistenceChecker = Callable[[str], bool]


async def run_prepare_only_flow(
    *,
    user_text: str,
    looks_like_prepare_only_request: PrepareOnlyRequestChecker,
    extract_dataset_path: DatasetPathExtractor,
    local_path_exists: LocalPathExistenceChecker,
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_tool_result_message: ToolResultMessageRenderer,
) -> TrainingPlanFollowupAction | None:
    request_context = resolve_prepare_only_request_context(
        user_text=user_text,
        looks_like_prepare_only_request=looks_like_prepare_only_request,
        extract_dataset_path=extract_dataset_path,
    )
    if not request_context.get('matches'):
        return None
    dataset_path = str(request_context.get('dataset_path') or '').strip()
    local_path_result = resolve_prepare_only_local_path_result(
        dataset_path=dataset_path,
        local_path_exists=local_path_exists,
    )
    if local_path_result is not None:
        return resolve_prepare_only_followup_action(result=local_path_result)
    result = await run_prepare_only_entrypoint(
        user_text=user_text,
        dataset_path=dataset_path,
        direct_tool=direct_tool,
        collect_requested_training_args=collect_requested_training_args,
        build_training_plan_draft_fn=build_training_plan_draft_fn,
        render_tool_result_message=render_tool_result_message,
    )
    return resolve_prepare_only_followup_action(result=result)


async def run_prepare_only_entrypoint(
    *,
    user_text: str,
    dataset_path: str,
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_tool_result_message: ToolResultMessageRenderer,
) -> PrepareOnlyResult:
    readiness = await direct_tool('dataset_training_readiness', img_dir=dataset_path)
    if not readiness.get('ok'):
        reply = await render_tool_result_message('dataset_training_readiness', readiness)
        if not reply:
            reply = str(readiness.get('error') or '数据准备前检查失败')
        return {
            'status': 'error',
            'reply': reply,
            'draft': None,
            'clear_draft': False,
            'defer_to_graph': False,
        }

    if readiness.get('ready') and str(readiness.get('resolved_data_yaml') or '').strip():
        data_yaml = str(readiness.get('resolved_data_yaml') or '').strip()
        return {
            'status': 'completed',
            'reply': f'当前数据已经可训练，现成 data.yaml: {data_yaml}。如果你只是想准备数据，这一步已经完成。',
            'draft': None,
            'clear_draft': True,
            'defer_to_graph': False,
        }

    resolved_img_dir = str(readiness.get('resolved_img_dir') or '').strip()
    resolved_label_dir = str(readiness.get('resolved_label_dir') or '').strip()
    if not resolved_img_dir or not resolved_label_dir:
        return {
            'status': 'completed',
            'reply': (
                f'我还没核实到可用的数据集结构：{dataset_path}。'
                '当前没有确认到 images/ 和 labels/ 目录，请检查路径是否写对。'
            ),
            'draft': None,
            'clear_draft': True,
            'defer_to_graph': False,
        }

    prepare_args: dict[str, Any] = {'dataset_path': dataset_path}
    classes_txt = str(intent_parsing.extract_classes_txt_from_text(user_text) or '').strip()
    if classes_txt:
        prepare_args['classes_txt'] = classes_txt
    if any(token in user_text for token in ('按默认比例', '默认比例', '先划分', '划分训练集', '划分数据集', 'split')):
        prepare_args['force_split'] = True

    planned_training_args = collect_requested_training_args(
        user_text,
        data_yaml=None,
    )
    draft = build_training_plan_draft_fn(
        user_text=user_text,
        dataset_path=dataset_path,
        readiness=readiness,
        preflight={},
        next_tool_name='prepare_dataset_for_training',
        next_tool_args=dict(prepare_args),
        planned_training_args=planned_training_args,
    )
    draft['execution_mode'] = 'prepare_only'
    return {
        'status': 'completed',
        'reply': '',
        'draft': draft,
        'clear_draft': False,
        'defer_to_graph': True,
    }


def resolve_prepare_only_followup_action(
    *,
    result: PrepareOnlyResult | None,
) -> TrainingPlanFollowupAction:
    result = dict(result or {})
    if not result:
        return {'action': 'none'}
    draft = dict(result.get('draft') or {})
    if result.get('defer_to_graph'):
        return {
            'action': 'save_draft_and_handoff',
            'draft': draft,
            'reply': str(result.get('reply') or ''),
        }
    if draft:
        return {
            'action': 'save_draft_and_reply',
            'draft': draft,
            'reply': str(result.get('reply') or ''),
            'status': str(result.get('status') or 'completed'),
        }
    if result.get('clear_draft'):
        return {
            'action': 'clear_draft_and_reply',
            'reply': str(result.get('reply') or ''),
            'status': str(result.get('status') or 'completed'),
        }
    return {
        'action': 'reply',
        'reply': str(result.get('reply') or ''),
        'status': str(result.get('status') or 'completed'),
    }


def resolve_prepare_only_request_context(
    *,
    user_text: str,
    looks_like_prepare_only_request: PrepareOnlyRequestChecker,
    extract_dataset_path: DatasetPathExtractor,
) -> PrepareOnlyRequestContext:
    if not looks_like_prepare_only_request(user_text):
        return {'matches': False, 'dataset_path': ''}
    dataset_path = str(extract_dataset_path(user_text) or '').strip()
    if not dataset_path:
        return {'matches': False, 'dataset_path': ''}
    return {'matches': True, 'dataset_path': dataset_path}


def resolve_prepare_only_local_path_result(
    *,
    dataset_path: str,
    local_path_exists: LocalPathExistenceChecker,
) -> PrepareOnlyResult | None:
    dataset_path = str(dataset_path or '').strip()
    if not dataset_path:
        return None
    candidate = Path(dataset_path).expanduser()
    if candidate.is_absolute() and not local_path_exists(dataset_path):
        return {
            'status': 'completed',
            'reply': f'我还没核实到这个路径存在：{dataset_path}。请先检查路径是否写对。',
            'draft': None,
            'clear_draft': True,
            'defer_to_graph': False,
        }
    return None


def resolve_training_request_entrypoint_guard(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    dataset_path: str,
    wants_train: bool,
    wants_predict: bool,
    no_train: bool,
    readiness_only_query: bool,
    wants_training_outcome_analysis: bool,
    wants_next_step_guidance: bool,
    wants_training_knowledge: bool,
    wants_training_revision: bool,
    wants_stop_training: bool,
    blocks_training_start: bool,
    explicit_run_ids: list[str] | None,
    extract_model_from_text: ModelPathExtractor,
) -> TrainingRequestGuard:
    explicit_run_ids = list(explicit_run_ids or [])
    active_training = session_state.active_training
    if (
        active_training.running
        and wants_train
        and wants_training_revision
        and not wants_stop_training
        and not any(token in user_text for token in ('新数据', '新数据集', '另一个数据集', '换数据集', '改数据集'))
        and not wants_predict
        and not explicit_run_ids
        and not any(token in user_text for token in ('数据', '数据集', 'dataset', 'img_dir', 'label_dir', '换成', '改成', '改用', '现在用'))
        and 'resume' not in normalized_text
    ):
        return {
            'reply': (
                '当前训练还在运行，不能直接热更新 batch、轮数、优化器或设备等核心参数。'
                '如果要改参数，请先停止当前训练，再生成新的训练计划。'
            ),
            'draft': None,
            'defer_to_graph': False,
            'proceed': False,
        }

    if (
        wants_train
        and not dataset_path
        and not no_train
        and not readiness_only_query
        and not wants_training_outcome_analysis
        and not wants_next_step_guidance
        and not wants_training_knowledge
        and not blocks_training_start
    ):
        requested_model = extract_model_from_text(user_text)
        missing_fields = ['数据集路径']
        if not requested_model:
            missing_fields.append('预训练权重/模型')
        lines = ['当前还不能开始训练：']
        for field in missing_fields:
            lines.append(f'- 缺少{field}')
        lines.append('请先补充最少必要信息；我至少需要数据集目录，训练时还需要可用的预训练权重/模型。')
        return {
            'reply': '\n'.join(lines),
            'draft': None,
            'defer_to_graph': False,
            'proceed': False,
        }

    if not (
        dataset_path
        and wants_train
        and not no_train
        and not readiness_only_query
        and not wants_training_outcome_analysis
        and not wants_next_step_guidance
        and not wants_training_knowledge
        and not blocks_training_start
    ):
        return {
            'reply': '',
            'draft': None,
            'defer_to_graph': False,
            'proceed': False,
            'return_none': True,
        }

    return {
        'reply': '',
        'draft': None,
        'defer_to_graph': False,
        'proceed': True,
    }


async def prepare_training_request_context(
    *,
    session_state: SessionState,
    user_text: str,
    dataset_path: str,
    frame_followup_path: str,
    current_training_plan_context: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
) -> TrainingRequestContext:
    active_training = session_state.active_training
    readiness = await direct_tool('training_readiness', img_dir=dataset_path)
    await direct_tool('list_training_environments')
    requested_args = collect_requested_training_args(
        user_text,
        data_yaml=str(readiness.get('resolved_data_yaml') or session_state.active_dataset.data_yaml or ''),
    )
    if not str(requested_args.get('model') or '').strip():
        current_training_plan_context = dict(current_training_plan_context or {})
        current_plan_args = dict(current_training_plan_context.get('planned_training_args') or {})
        draft_model = str(current_plan_args.get('model') or '').strip()
        preserved_model = ''
        if frame_followup_path:
            preserved_model = draft_model or str(active_training.model or '').strip()
        elif any(token in user_text for token in ('继续', '刚才', '上次', '恢复')):
            preserved_model = draft_model or str(active_training.model or '').strip()
        if preserved_model:
            requested_args['model'] = preserved_model
    return {
        'readiness': readiness,
        'requested_args': requested_args,
    }


async def run_training_request_entrypoint(
    *,
    session_state: SessionState,
    user_text: str,
    normalized_text: str,
    dataset_path: str,
    frame_followup_path: str,
    wants_train: bool,
    wants_predict: bool,
    no_train: bool,
    readiness_only_query: bool,
    wants_training_outcome_analysis: bool,
    wants_next_step_guidance: bool,
    wants_training_knowledge: bool,
    wants_training_revision: bool,
    wants_stop_training: bool,
    blocks_training_start: bool,
    explicit_run_ids: list[str] | None,
    wants_split: bool,
    current_training_plan_context: dict[str, Any] | None,
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
    is_training_discussion_only: TrainingDiscussionChecker,
    extract_training_execution_backend: TrainingExecutionBackendExtractor,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> TrainingRequestResult | None:
    guard = resolve_training_request_entrypoint_guard(
        session_state=session_state,
        user_text=user_text,
        normalized_text=normalized_text,
        dataset_path=dataset_path,
        wants_train=wants_train,
        wants_predict=wants_predict,
        no_train=no_train,
        readiness_only_query=readiness_only_query,
        wants_training_outcome_analysis=wants_training_outcome_analysis,
        wants_next_step_guidance=wants_next_step_guidance,
        wants_training_knowledge=wants_training_knowledge,
        wants_training_revision=wants_training_revision,
        wants_stop_training=wants_stop_training,
        blocks_training_start=blocks_training_start,
        explicit_run_ids=explicit_run_ids,
        extract_model_from_text=intent_parsing.extract_model_from_text,
    )
    if not guard.get('proceed'):
        if guard.get('return_none'):
            return None
        return {
            'reply': str(guard.get('reply') or '').strip(),
            'draft': dict(guard.get('draft') or {}) or None,
            'defer_to_graph': bool(guard.get('defer_to_graph')),
        }

    prepared_context = await prepare_training_request_context(
        session_state=session_state,
        user_text=user_text,
        dataset_path=dataset_path,
        frame_followup_path=frame_followup_path,
        current_training_plan_context=current_training_plan_context,
        direct_tool=direct_tool,
        collect_requested_training_args=collect_requested_training_args,
    )
    readiness = dict(prepared_context.get('readiness') or {})
    requested_args = dict(prepared_context.get('requested_args') or {})
    plan_result = await run_training_request_orchestration(
        user_text=user_text,
        dataset_path=dataset_path,
        readiness=readiness,
        requested_args=requested_args,
        wants_split=wants_split,
        discussion_only=is_training_discussion_only(user_text),
        execution_backend=extract_training_execution_backend(user_text),
        direct_tool=direct_tool,
        build_training_plan_draft_fn=build_training_plan_draft_fn,
        render_training_plan_message=render_training_plan_message,
    )
    return {
        'reply': str(plan_result.get('reply') or '').strip(),
        'draft': dict(plan_result.get('draft') or {}),
        'defer_to_graph': bool(plan_result.get('defer_to_graph')),
    }
