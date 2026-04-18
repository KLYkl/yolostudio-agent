from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.session_state import SessionState


LoopDataYamlResolver = Callable[..., str]
LoopPrepareArgsBuilder = Callable[[str, str], dict[str, Any]]
LoopFactCompactor = Callable[[str, dict[str, Any]], dict[str, Any]]
EventAppender = Callable[[str, dict[str, Any]], None]
RendererTextInvoker = Callable[..., Awaitable[str]]
TrainingPlanDraftRenderer = Callable[..., str]
DirectToolInvoker = Callable[..., Awaitable[dict[str, Any]]]
DraftSaver = Callable[[dict[str, Any]], None]
AssistantMessageAppender = Callable[[str], None]
GraphHandoffInvoker = Callable[[str, str], Awaitable[dict[str, Any]]]
TrainingPlanDraftBuilder = Callable[..., dict[str, Any]]
TrainingPlanMessageRenderer = Callable[[dict[str, Any], bool], Awaitable[str]]
ToolResultMessageRenderer = Callable[[str, dict[str, Any]], Awaitable[str]]
TrainingArgsCollector = Callable[..., dict[str, Any]]
TrainingDiscussionChecker = Callable[[str], bool]
TrainingExecutionBackendExtractor = Callable[[str], str]
TrainingAdvancedDetailsChecker = Callable[[str], bool]
ModelPathExtractor = Callable[[str], str]
PrepareOnlyRequestChecker = Callable[[str], bool]
DatasetPathExtractor = Callable[[str], str]
LocalPathExistenceChecker = Callable[[str], bool]

TRAINING_PREFLIGHT_STRING_FIELDS = (
    'training_environment',
    'project',
    'name',
    'optimizer',
)
TRAINING_PREFLIGHT_OPTIONAL_FIELDS = (
    'batch',
    'imgsz',
    'fraction',
    'classes',
    'single_cls',
    'freeze',
    'resume',
    'lr0',
    'patience',
    'workers',
    'amp',
)


async def _render_orchestration_result(
    draft: dict[str, Any],
    *,
    pending: bool,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> dict[str, Any]:
    return {
        'draft': draft,
        'reply': await render_training_plan_message(draft, pending=pending),
        'defer_to_graph': pending,
    }


def build_training_preflight_tool_args(
    planned_args: dict[str, Any] | None,
    *,
    fallback_model: str = '',
    fallback_data_yaml: str = '',
) -> dict[str, Any]:
    planned_args = dict(planned_args or {})
    payload = {
        'model': str(planned_args.get('model') or fallback_model or ''),
        'data_yaml': str(planned_args.get('data_yaml') or fallback_data_yaml or ''),
        'epochs': int(planned_args.get('epochs', 100)),
        'device': str(planned_args.get('device', 'auto') or 'auto'),
    }
    for field in TRAINING_PREFLIGHT_STRING_FIELDS:
        payload[field] = str(planned_args.get(field) or '')
    for field in TRAINING_PREFLIGHT_OPTIONAL_FIELDS:
        payload[field] = planned_args.get(field)
    return payload


def resolve_training_start_args(
    planned_args: dict[str, Any] | None,
    preflight: dict[str, Any] | None,
    *,
    fallback_model: str = '',
    fallback_data_yaml: str = '',
) -> dict[str, Any]:
    planned_args = dict(planned_args or {})
    resolved_args = dict((preflight or {}).get('resolved_args') or {})
    payload = {
        'model': str(resolved_args.get('model') or planned_args.get('model') or fallback_model or ''),
        'data_yaml': str(resolved_args.get('data_yaml') or planned_args.get('data_yaml') or fallback_data_yaml or ''),
        'epochs': int(resolved_args.get('epochs') or planned_args.get('epochs', 100)),
        'device': str(resolved_args.get('device') or planned_args.get('device') or 'auto'),
    }
    for field in TRAINING_PREFLIGHT_STRING_FIELDS:
        payload[field] = str(resolved_args.get(field) or planned_args.get(field) or '')
    for field in TRAINING_PREFLIGHT_OPTIONAL_FIELDS:
        payload[field] = resolved_args.get(field, planned_args.get(field))
    return payload


async def run_training_request_orchestration(
    *,
    user_text: str,
    dataset_path: str,
    readiness: dict[str, Any] | None,
    requested_args: dict[str, Any] | None,
    wants_split: bool,
    discussion_only: bool,
    execution_backend: str,
    direct_tool: DirectToolInvoker,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> dict[str, Any]:
    readiness = dict(readiness or {})
    requested_args = dict(requested_args or {})
    requested_model = str(requested_args.get('model') or '').strip()

    if not requested_model:
        draft = build_training_plan_draft_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            next_tool_name='',
            next_tool_args={},
            planned_training_args=requested_args,
        )
        blockers = list(draft.get('blockers') or [])
        blockers.insert(0, '当前缺少预训练权重/模型，先补模型后再确认训练')
        draft['blockers'] = blockers
        return await _render_orchestration_result(
            draft,
            pending=False,
            render_training_plan_message=render_training_plan_message,
        )

    if execution_backend != 'standard_yolo':
        draft = build_training_plan_draft_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            next_tool_name='',
            next_tool_args={},
            planned_training_args=requested_args,
        )
        return await _render_orchestration_result(
            draft,
            pending=False,
            render_training_plan_message=render_training_plan_message,
        )

    can_direct_train = bool(readiness.get('ready')) and bool(readiness.get('resolved_data_yaml'))
    if can_direct_train:
        resolved_data_yaml = str(readiness.get('resolved_data_yaml') or '')
        preflight_args = build_training_preflight_tool_args(
            requested_args,
            fallback_model=requested_model,
            fallback_data_yaml=resolved_data_yaml,
        )
        preflight = await direct_tool('training_preflight', **preflight_args)
        next_args = resolve_training_start_args(
            requested_args,
            preflight,
            fallback_model=requested_model,
            fallback_data_yaml=resolved_data_yaml,
        )
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
        return await _render_orchestration_result(
            draft,
            pending=bool(ready_to_start and not discussion_only),
            render_training_plan_message=render_training_plan_message,
        )

    if readiness.get('preparable'):
        next_args: dict[str, Any] = {'dataset_path': dataset_path}
        if wants_split:
            next_args['force_split'] = True
        explicit_classes_txt = str(requested_args.get('classes_txt') or '').strip()
        if explicit_classes_txt:
            next_args['classes_txt'] = explicit_classes_txt
        draft = build_training_plan_draft_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight={},
            next_tool_name='prepare_dataset_for_training',
            next_tool_args=next_args,
            planned_training_args=requested_args,
        )
        return await _render_orchestration_result(
            draft,
            pending=not discussion_only,
            render_training_plan_message=render_training_plan_message,
        )

    draft = build_training_plan_draft_fn(
        user_text=user_text,
        dataset_path=dataset_path,
        readiness=readiness,
        preflight={},
        next_tool_name='',
        next_tool_args={},
        planned_training_args=requested_args,
    )
    return await _render_orchestration_result(
        draft,
        pending=False,
        render_training_plan_message=render_training_plan_message,
    )


async def run_prepare_only_entrypoint(
    *,
    user_text: str,
    dataset_path: str,
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_tool_result_message: ToolResultMessageRenderer,
) -> dict[str, Any]:
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


def resolve_prepare_only_request_context(
    *,
    user_text: str,
    looks_like_prepare_only_request: PrepareOnlyRequestChecker,
    extract_dataset_path: DatasetPathExtractor,
) -> dict[str, Any]:
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
) -> dict[str, Any] | None:
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


def resolve_training_plan_dialogue_context(
    *,
    user_text: str,
    explicit_run_ids: list[str] | None,
    is_training_discussion_only: TrainingDiscussionChecker,
) -> dict[str, Any]:
    explicit_run_ids = list(explicit_run_ids or [])
    normalized = user_text.lower()
    is_loop_dialogue = any(token in user_text for token in ('环训练', '循环训练', '循环训', '循环跑', '自动复训', '自动续训')) or any(
        token in normalized for token in ('training loop', 'loop training', 'auto retrain', 'auto training loop')
    )
    discussion_only_hint = is_training_discussion_only(user_text) or any(token in user_text for token in ('不执行', '不要执行', '暂不执行', '先不执行'))
    training_readiness_question = any(
        token in user_text
        for token in (
            '能不能直接训练',
            '能否直接训练',
            '可不可以直接训练',
            '可以直接训练吗',
            '是否可以直接训练',
            '是否能直接训练',
            '还能不能直接训练',
            '可否直接训练',
            '能直接训练吗',
        )
    )
    contradictory_train_intent = (
        not training_readiness_question
        and any(token in user_text for token in ('不要训练', '先不要训练', '不训练了'))
        and any(token in user_text for token in ('开始训练', '启动训练', '直接训练', '直接开始训练', '开训', '执行'))
    )
    requested_execute = (
        any(token in user_text for token in ('执行', '开始吧', '就这样', '确认', '可以开始', '开训', '启动吧', '直接训练', '直接开始训练'))
        or normalized.strip() in {'y', 'yes'}
    ) and not discussion_only_hint and not contradictory_train_intent
    if any(token in user_text for token in ('为什么', '原因', '依据', '怎么看')):
        requested_execute = False
    wants_repeat_prepare = any(token in user_text for token in ('再 prepare 一次', '再准备一次', '重新 prepare 一次', '重新准备一次', '再做一次准备', '重新准备一遍'))
    wants_retry_last_plan = any(token in user_text for token in ('按原计划重试一次', '按原计划重试', '重试刚才那次训练', '重试上次训练', '按刚才的计划再来一次', '按原计划再来一次'))
    wants_resume_recent_training = (
        any(
            token in user_text
            for token in (
                '从最近状态继续训练',
                '从最近状态继续',
                '从最近状态恢复训练',
                '恢复刚才训练',
                '接着上次训练',
                '恢复上次训练',
                'resume 上次训练',
                'resume 刚才训练',
                'resume 另一个 run',
                'resume run',
                '继续另一个 run',
            )
        )
        or (
            'resume' in normalized
            and (
                bool(explicit_run_ids)
                or any(token in user_text for token in ('上次', '刚才', '最近', '继续', '恢复', '另一个', '历史', 'run'))
            )
        )
    )
    wants_analysis_only = any(token in user_text for token in ('只分析', '只看结果', '不要接着训', '不要继续训', '不要继续训练'))
    latest_dataset_path = intent_parsing.extract_dataset_path_from_text(user_text)
    all_paths = intent_parsing.extract_all_paths_from_text(user_text)
    project_path_hint = intent_parsing.extract_project_from_text(user_text)
    custom_script_hint = intent_parsing.extract_custom_training_script_from_text(user_text)
    dataset_candidates = [
        item for item in all_paths
        if not intent_parsing.looks_like_model_path(item)
        and item != project_path_hint
        and item != custom_script_hint
    ]
    if dataset_candidates and any(token in user_text for token in ('换成', '现在用', '改成', '改用')):
        latest_dataset_path = dataset_candidates[-1]
    if (
        latest_dataset_path
        and project_path_hint
        and latest_dataset_path == project_path_hint
        and not any(token in user_text for token in ('数据', 'dataset', 'img_dir', 'label_dir'))
    ):
        latest_dataset_path = ''
    if (
        latest_dataset_path
        and custom_script_hint
        and latest_dataset_path == custom_script_hint
        and not any(token in user_text for token in ('数据', 'dataset', 'img_dir', 'label_dir'))
    ):
        latest_dataset_path = ''
    dataset_path_revision_requested = bool(latest_dataset_path) and (
        any(token in user_text for token in ('数据', '数据集', 'dataset', 'img_dir', 'label_dir'))
        or any(token in user_text for token in ('现在用', '改用', '换成', '改成'))
        or (
            any(token in user_text for token in ('训练', '开训', '开始训练', '继续训练'))
            and not any(token in user_text for token in ('预测', '推理', '识别', '抽帧', '提帧', '视频', '图片'))
        )
    )
    return {
        'normalized': normalized,
        'is_loop_dialogue': is_loop_dialogue,
        'discussion_only_hint': discussion_only_hint,
        'training_readiness_question': training_readiness_question,
        'contradictory_train_intent': contradictory_train_intent,
        'requested_execute': requested_execute,
        'wants_repeat_prepare': wants_repeat_prepare,
        'wants_retry_last_plan': wants_retry_last_plan,
        'wants_resume_recent_training': wants_resume_recent_training,
        'wants_analysis_only': wants_analysis_only,
        'latest_dataset_path': latest_dataset_path,
        'dataset_path_revision_requested': dataset_path_revision_requested,
    }


def resolve_training_plan_dialogue_flags(
    *,
    user_text: str,
    normalized_text: str,
    draft: dict[str, Any] | None,
    pending: dict[str, Any] | None,
    requested_execute: bool,
    clear_fields: list[str] | None,
    wants_retry_last_plan: bool,
    wants_resume_recent_training: bool,
    dataset_path_revision_requested: bool,
    custom_training_script_requested: bool,
) -> dict[str, Any]:
    draft = dict(draft or {})
    pending = dict(pending or {})
    clear_fields = list(clear_fields or [])

    has_revision = any(
        token in normalized_text or token in user_text
        for token in (
            'batch', 'imgsz', 'device', 'epochs', '轮', '轮数', '优化器', 'optimizer', '冻结', 'freeze', 'resume',
            'lr0', '学习率', 'patience', '早停', 'workers', '线程数', 'amp', '混合精度',
            '模型', '权重', 'project', '输出目录', 'name', '实验名', '运行名',
            'fraction', '全量数据', '抽样', 'classes', '类别', 'single_cls', '单类别',
            '环境', '为什么', '原因', '依据', '先只做准备', '只做准备', '标准 yolo', '自定义脚本', 'trainer',
            '高级参数', '高级配置', '展开参数', '详细参数',
            '划分', '自动划分', '不划分', '不要划分', '默认比例',
        )
    ) or wants_retry_last_plan or wants_resume_recent_training or custom_training_script_requested or dataset_path_revision_requested

    switching_prepare_only_to_train = bool(
        pending
        and pending.get('name') == 'prepare_dataset_for_training'
        and str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only'
        and any(token in user_text for token in ('直接训练', '开始训练', '启动训练', '执行训练', '那就训练', '那你训练'))
    )
    has_revision = has_revision or switching_prepare_only_to_train

    wants_cancel_plan = (
        any(token in user_text for token in ('取消', '算了', '先不做', '不用了', '先别开始训练', '先不要开始训练', '先别开训', '先不要开训', '先别开始', '先不要开始'))
        and not clear_fields
        and not requested_execute
        and not has_revision
        and not any(token in user_text for token in ('取消了', '已经取消', '刚才'))
    )
    wants_skip_training_recheck = (
        not pending
        and any(token in user_text for token in ('不要训练', '先不要训练', '不训练了'))
        and any(token in user_text for token in ('重新检查', '检查一下', '能不能直接训练', '是否能直接训练'))
    )
    wants_pause_confirmation = bool(pending) and any(token in user_text for token in ('等等', '等一下', '先等等', '先等下', '稍等', '先稍等'))
    wants_prepare_output_explanation = (
        bool(pending)
        and pending.get('name') == 'prepare_dataset_for_training'
        and any(token in user_text for token in ('data_yaml', 'yaml', '产物路径', '输出路径', '会生成到哪里'))
    )
    wants_plan_preview = (
        any(token in user_text for token in ('先别执行', '先不要执行', '先别启动', '先不要启动', '先讨论', '先看看计划', '先给我计划', '想先看计划', '记一下我想先看计划'))
        and not has_revision
        and not requested_execute
    )
    wants_original_plan = (
        bool(draft)
        and not has_revision
        and not requested_execute
        and any(token in user_text for token in ('最开始那套呢', '最开始那个计划呢', '第一版计划呢', '最早那套呢', '最开始那版呢'))
    )
    wants_continue_plan = (
        bool(draft)
        and not has_revision
        and not requested_execute
        and any(token in user_text for token in ('训练计划继续', '继续刚才训练计划', '继续刚才的训练计划', '继续刚才那个训练计划', '刚才训练计划继续'))
    )
    wants_prepare_only = any(token in user_text for token in ('只做准备', '只准备', '先准备不要训练'))
    wants_disable_split = any(token in user_text for token in ('不要自动划分', '不要划分', '不划分'))

    return {
        'has_revision': has_revision,
        'switching_prepare_only_to_train': switching_prepare_only_to_train,
        'wants_cancel_plan': wants_cancel_plan,
        'wants_skip_training_recheck': wants_skip_training_recheck,
        'wants_pause_confirmation': wants_pause_confirmation,
        'wants_prepare_output_explanation': wants_prepare_output_explanation,
        'wants_plan_preview': wants_plan_preview,
        'wants_original_plan': wants_original_plan,
        'wants_continue_plan': wants_continue_plan,
        'wants_prepare_only': wants_prepare_only,
        'wants_disable_split': wants_disable_split,
    }


def resolve_training_plan_dialogue_existing_action(
    *,
    draft: dict[str, Any] | None,
    pending: dict[str, Any] | None,
    flag_context: dict[str, Any] | None,
    requested_execute: bool,
    wants_repeat_prepare: bool,
    readiness: dict[str, Any] | None,
    data_yaml: str,
) -> dict[str, Any]:
    draft = dict(draft or {})
    pending = dict(pending or {})
    flag_context = dict(flag_context or {})
    readiness = dict(readiness or {})
    has_pending = bool(pending)
    has_revision = bool(flag_context.get('has_revision'))

    if flag_context.get('wants_cancel_plan'):
        return {'action': 'cancel_pending' if has_pending else 'cancel_draft'}
    if flag_context.get('wants_skip_training_recheck'):
        return {'action': 'clear_and_recheck'}
    if flag_context.get('wants_pause_confirmation'):
        return {'action': 'confirmation_message'}
    if flag_context.get('wants_prepare_output_explanation'):
        dataset_path = str((pending.get('args') or {}).get('dataset_path') or draft.get('dataset_path') or '').strip()
        expected_yaml = f'{dataset_path.rstrip("/")}/data.yaml' if dataset_path else '准备输出目录中的 data.yaml'
        dataset_label = dataset_path or '<当前数据集>'
        return {
            'action': 'reply_with_pending',
            'reply': (
                f'如果继续 prepare，我会基于数据集 {dataset_label} 生成可训练产物；'
                f'预期会产出可用的 data_yaml（通常是 {expected_yaml}），完成后我会把真实路径写回状态。'
            ),
        }
    if wants_repeat_prepare and readiness.get('ready') and data_yaml:
        if has_pending:
            return {'action': 'confirmation_message'}
        return {
            'action': 'reply',
            'reply': f'当前数据集已经准备完成：{data_yaml}；不需要重复 prepare。你可以直接继续训练或重新规划。',
        }
    if flag_context.get('wants_plan_preview'):
        return {'action': 'render_plan'}
    if flag_context.get('wants_original_plan'):
        return {
            'action': 'render_original_plan',
            'preamble': (
                '当前只保留最新训练计划草案；最开始那套已经被后续修订覆盖。'
                '如果要回退，请直接说明要恢复的数据集、模型或关键参数。'
            ),
        }
    if flag_context.get('wants_continue_plan'):
        return {'action': 'render_plan'}
    if requested_execute and not has_revision:
        if has_pending:
            return {'action': 'approve_pending'}
        next_tool_name = str(draft.get('next_step_tool') or '').strip()
        if next_tool_name:
            return {'action': 'defer_to_graph'}
        return {'action': 'render_draft'}
    if not has_revision:
        return {'action': 'noop'}
    return {'action': 'proceed_revision'}


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
) -> dict[str, Any] | None:
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
) -> dict[str, Any]:
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
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
) -> dict[str, Any]:
    active_training = session_state.active_training
    readiness = await direct_tool('training_readiness', img_dir=dataset_path)
    await direct_tool('list_training_environments')
    requested_args = collect_requested_training_args(
        user_text,
        data_yaml=str(readiness.get('resolved_data_yaml') or session_state.active_dataset.data_yaml or ''),
    )
    if not str(requested_args.get('model') or '').strip():
        draft_model = str((((active_training.training_plan_draft or {}).get('planned_training_args') or {}).get('model')) or '').strip()
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
    direct_tool: DirectToolInvoker,
    collect_requested_training_args: TrainingArgsCollector,
    is_training_discussion_only: TrainingDiscussionChecker,
    extract_training_execution_backend: TrainingExecutionBackendExtractor,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
    render_training_plan_message: TrainingPlanMessageRenderer,
) -> dict[str, Any] | None:
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
    return await _render_orchestration_result(
        draft,
        pending=ready_to_start,
        render_training_plan_message=render_training_plan_message,
    )


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


async def prepare_training_revision_context(
    *,
    session_state: SessionState,
    user_text: str,
    draft: dict[str, Any] | None,
    pending: dict[str, Any] | None,
    latest_dataset_path: str,
    clear_fields: list[str] | None,
    switching_prepare_only_to_train: bool,
    wants_prepare_only: bool,
    wants_disable_split: bool,
    collect_requested_training_args: TrainingArgsCollector,
    extract_training_execution_backend: TrainingExecutionBackendExtractor,
    wants_training_advanced_details: TrainingAdvancedDetailsChecker,
    direct_tool: DirectToolInvoker,
) -> dict[str, Any]:
    revised_draft = dict(draft or {})
    pending = dict(pending or {})
    clear_fields = list(clear_fields or [])
    planned_args = dict(revised_draft.get('planned_training_args') or {})
    dataset_path = str(
        latest_dataset_path
        or revised_draft.get('dataset_path')
        or session_state.active_dataset.dataset_root
        or session_state.active_dataset.img_dir
        or ''
    ).strip()
    readiness = dict(session_state.active_dataset.last_readiness or {})
    previous_dataset_path = str(revised_draft.get('dataset_path') or '').strip()
    if latest_dataset_path and dataset_path and dataset_path != previous_dataset_path:
        readiness = await direct_tool('training_readiness', img_dir=dataset_path)
        await direct_tool('list_training_environments')
        resolved_yaml = str(readiness.get('resolved_data_yaml') or '').strip()
        if resolved_yaml:
            planned_args['data_yaml'] = resolved_yaml
        else:
            planned_args.pop('data_yaml', None)
    requested_data_yaml_hint: str | None = str(planned_args.get('data_yaml') or '').strip()
    if not requested_data_yaml_hint:
        requested_data_yaml_hint = None if latest_dataset_path else str(session_state.active_dataset.data_yaml or '').strip()
    requested_args = collect_requested_training_args(
        user_text,
        data_yaml=requested_data_yaml_hint,
    )
    for field in clear_fields:
        planned_args.pop(field, None)
    planned_args.update(
        {
            key: value
            for key, value in requested_args.items()
            if value is not None and value != ''
        }
    )
    execution_backend = extract_training_execution_backend(user_text)
    advanced_requested = wants_training_advanced_details(user_text) or bool(revised_draft.get('advanced_details_requested'))
    if switching_prepare_only_to_train:
        revised_draft['execution_mode'] = 'prepare_then_train'
    if wants_prepare_only:
        revised_draft['execution_mode'] = 'prepare_only'
        revised_draft['next_step_tool'] = 'prepare_dataset_for_training'
    if wants_disable_split:
        next_step_args = dict(revised_draft.get('next_step_args') or {})
        next_step_args.pop('force_split', None)
        revised_draft['next_step_args'] = next_step_args
    next_tool_name = str(revised_draft.get('next_step_tool') or pending.get('name') or '').strip()
    next_tool_args = dict(revised_draft.get('next_step_args') or pending.get('args') or {})
    execution_mode = str(revised_draft.get('execution_mode') or '').strip().lower()
    return {
        'revised_draft': revised_draft,
        'planned_args': planned_args,
        'dataset_path': dataset_path,
        'readiness': readiness,
        'next_tool_name': next_tool_name,
        'next_tool_args': next_tool_args,
        'execution_mode': execution_mode,
        'execution_backend': execution_backend,
        'advanced_requested': advanced_requested,
    }


def resolve_training_revision_followup_action(
    *,
    revised_draft: dict[str, Any] | None,
    pending: dict[str, Any] | None,
    requested_execute: bool,
    wants_retry_last_plan: bool,
    wants_resume_recent_training: bool,
) -> dict[str, Any]:
    revised_draft = dict(revised_draft or {})
    pending = dict(pending or {})
    force_confirmation = wants_retry_last_plan or wants_resume_recent_training
    next_step_tool = str(revised_draft.get('next_step_tool') or '').strip()
    if next_step_tool and (pending or force_confirmation or requested_execute):
        if not pending and (requested_execute or force_confirmation):
            return {'action': 'defer_to_graph'}
        return {'action': 'refresh_confirmation'}
    return {'action': 'render_completed'}


async def build_training_revision_draft(
    *,
    user_text: str,
    dataset_path: str,
    readiness: dict[str, Any] | None,
    planned_args: dict[str, Any] | None,
    next_tool_name: str,
    next_tool_args: dict[str, Any] | None,
    execution_mode: str,
    execution_backend: str,
    advanced_requested: bool,
    direct_tool: DirectToolInvoker,
    build_training_plan_draft_fn: TrainingPlanDraftBuilder,
) -> dict[str, Any]:
    readiness = dict(readiness or {})
    planned_args = dict(planned_args or {})
    next_tool_args = dict(next_tool_args or {})
    execution_mode = str(execution_mode or '').strip().lower()
    next_tool_name = str(next_tool_name or '').strip()

    if execution_backend != 'standard_yolo':
        draft = build_training_plan_draft_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight={},
            next_tool_name='',
            next_tool_args={},
            planned_training_args=planned_args,
        )
        draft['advanced_details_requested'] = advanced_requested
        return draft

    if (
        (next_tool_name == 'start_training' or execution_mode in {'direct_train', 'discussion_only', 'blocked'})
        and readiness.get('ready')
        and planned_args.get('model')
    ):
        preflight_args = build_training_preflight_tool_args(planned_args)
        preflight = await direct_tool('training_preflight', **preflight_args)
        resolved_training_args = resolve_training_start_args(planned_args, preflight)
        draft = build_training_plan_draft_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight=preflight,
            next_tool_name='start_training' if preflight.get('ready_to_start') else '',
            next_tool_args=resolved_training_args if preflight.get('ready_to_start') else {},
            planned_training_args=resolved_training_args,
        )
        draft['advanced_details_requested'] = advanced_requested
        return draft

    if readiness.get('preparable'):
        prepare_args: dict[str, Any] = {'dataset_path': dataset_path}
        if next_tool_args.get('force_split'):
            prepare_args['force_split'] = next_tool_args.get('force_split')
        draft = build_training_plan_draft_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness=readiness,
            preflight={},
            next_tool_name='prepare_dataset_for_training',
            next_tool_args=prepare_args,
            planned_training_args=planned_args,
        )
        draft['advanced_details_requested'] = advanced_requested
        return draft

    draft = build_training_plan_draft_fn(
        user_text=user_text,
        dataset_path=dataset_path,
        readiness=readiness,
        preflight={},
        next_tool_name='',
        next_tool_args={},
        planned_training_args=planned_args,
    )
    draft['advanced_details_requested'] = advanced_requested
    return draft


def build_training_loop_start_fallback_plan(
    *,
    user_text: str,
    dataset_path: str,
    loop_args: dict[str, Any],
    observed_tools: dict[str, dict[str, Any]] | None = None,
    known_training_loop_data_yaml: LoopDataYamlResolver,
    build_loop_prepare_args: LoopPrepareArgsBuilder,
) -> dict[str, Any]:
    observed_tools = dict(observed_tools or {})
    readiness = dict(observed_tools.get('training_readiness') or {})
    prepare_result = dict(observed_tools.get('prepare_dataset_for_training') or {})
    model = str(loop_args.get('model') or '').strip()
    data_yaml = known_training_loop_data_yaml(loop_args, observed_tools, dataset_path=dataset_path)
    if not model:
        return {
            'decision': 'block',
            'reason': '当前还不能开启环训练：缺少预训练权重/模型。请先明确模型，例如 yolov8n.pt。',
            'planner_source': 'fallback',
        }
    if prepare_result.get('ok') and data_yaml:
        next_args = dict(loop_args)
        next_args['model'] = model
        next_args['data_yaml'] = data_yaml
        if not str(next_args.get('managed_level') or '').strip():
            next_args['managed_level'] = 'conservative_auto'
        if next_args.get('max_rounds') in {None, ''}:
            next_args['max_rounds'] = 5
        return {
            'decision': 'start',
            'next_tool': 'start_training_loop',
            'next_args': next_args,
            'reason': '数据已经准备完成，可以直接启动循环训练。',
            'planner_source': 'fallback',
        }
    if data_yaml:
        next_args = dict(loop_args)
        next_args['model'] = model
        next_args['data_yaml'] = data_yaml
        if not str(next_args.get('managed_level') or '').strip():
            next_args['managed_level'] = 'conservative_auto'
        if next_args.get('max_rounds') in {None, ''}:
            next_args['max_rounds'] = 5
        return {
            'decision': 'start',
            'next_tool': 'start_training_loop',
            'next_args': next_args,
            'reason': '当前数据已具备训练条件，可以直接进入循环训练。',
            'planner_source': 'fallback',
        }
    if not readiness:
        if dataset_path:
            return {
                'decision': 'observe',
                'next_tool': 'training_readiness',
                'next_args': {'img_dir': dataset_path},
                'reason': '先读取训练前检查结果，再决定是 prepare 还是 start。',
                'planner_source': 'fallback',
            }
        return {
            'decision': 'block',
            'reason': '当前还不能开启环训练：缺少可用数据路径，无法判断是否需要先 prepare。',
            'planner_source': 'fallback',
        }
    if readiness and not readiness.get('ok', True) and not readiness.get('preparable'):
        blockers = [str(item).strip() for item in (readiness.get('blockers') or []) if str(item).strip()]
        blocker_detail = str(readiness.get('error') or (blockers[0] if blockers else '') or readiness.get('summary') or '').strip()
        return {
            'decision': 'block',
            'reason': f'当前还不能开启环训练：{blocker_detail or "训练前检查失败"}',
            'planner_source': 'fallback',
        }
    if dataset_path and readiness.get('preparable'):
        return {
            'decision': 'prepare',
            'next_tool': 'prepare_dataset_for_training',
            'next_args': build_loop_prepare_args(user_text, dataset_path),
            'reason': '当前数据还不能直接进入循环训练，先准备数据集，再继续启动 loop。',
            'planner_source': 'fallback',
        }
    blockers = [str(item).strip() for item in (readiness.get('blockers') or []) if str(item).strip()]
    blocker_detail = str(readiness.get('error') or (blockers[0] if blockers else '') or readiness.get('summary') or '').strip()
    return {
        'decision': 'block',
        'reason': f'当前还不能开启环训练：{blocker_detail or "缺少可训练的 data_yaml。"}',
        'planner_source': 'fallback',
    }


def build_training_loop_start_draft(
    session_state: SessionState,
    *,
    user_text: str,
    dataset_path: str,
    loop_args: dict[str, Any],
    observed_tools: dict[str, dict[str, Any]] | None,
    plan: dict[str, Any],
    known_training_loop_data_yaml: LoopDataYamlResolver,
) -> dict[str, Any]:
    observed_tools = dict(observed_tools or {})
    readiness = dict(observed_tools.get('training_readiness') or {})
    prepare_result = dict(observed_tools.get('prepare_dataset_for_training') or {})
    latest_summary = str(
        prepare_result.get('summary')
        or readiness.get('summary')
        or session_state.active_dataset.last_readiness.get('summary')
        or ''
    ).strip()
    planned_args = dict(loop_args)
    data_yaml = known_training_loop_data_yaml(planned_args, observed_tools, dataset_path=dataset_path)
    if data_yaml:
        planned_args['data_yaml'] = data_yaml
    next_tool_name = str(plan.get('next_tool') or '').strip()
    previous_draft = dict(session_state.active_training.training_plan_draft or {})
    execution_mode = 'prepare_then_loop' if next_tool_name == 'prepare_dataset_for_training' else 'direct_loop'
    if next_tool_name == 'start_training_loop' and (
        'prepare_dataset_for_training' in observed_tools
        or str(previous_draft.get('execution_mode') or '').strip().lower() == 'prepare_then_loop'
    ):
        execution_mode = 'prepare_then_loop'
    return {
        'source_intent': 'training_loop',
        'execution_mode': execution_mode,
        'execution_backend': 'standard_yolo',
        'dataset_path': dataset_path,
        'data_summary': latest_summary,
        'reasoning_summary': str(plan.get('reason') or '').strip(),
        'planned_training_args': dict(planned_args),
        'planned_loop_args': dict(planned_args),
        'next_step_tool': next_tool_name,
        'next_step_args': dict(plan.get('next_args') or {}),
        'planner_decision_source': str(plan.get('planner_source') or 'fallback'),
        'planner_decision': 'prepare' if next_tool_name == 'prepare_dataset_for_training' else 'start',
        'planner_output': dict(plan.get('planner_payload') or {}),
        'planner_user_request': user_text,
        'planner_observed_tools': list(observed_tools.keys()),
        'editable_fields': ['model', 'epochs', 'batch', 'imgsz', 'device', 'training_environment', 'project', 'name'],
    }


def training_plan_user_facts(draft: dict[str, Any], *, pending: bool) -> dict[str, Any]:
    execution_mode_raw = str(draft.get('execution_mode') or '').strip().lower()
    next_step_tool = str(draft.get('next_step_tool') or '').strip()
    loop_like = 'loop' in execution_mode_raw or next_step_tool == 'start_training_loop'
    args_source = draft.get('planned_loop_args') if loop_like else draft.get('planned_training_args')
    args = dict(args_source or draft.get('planned_training_args') or {})
    next_args = dict(draft.get('next_step_args') or {})
    execution_mode_map = {
        'prepare_then_train': '先准备再训练',
        'prepare_then_loop': '先准备再进入循环训练',
        'direct_train': '直接训练',
        'direct_loop': '直接启动循环训练',
        'prepare_only': '只做准备，暂不启动训练',
        'discussion_only': '先讨论方案，暂不执行',
        'blocked': '当前存在阻塞，先解决问题',
    }
    execution_backend_map = {
        'standard_yolo': '标准 YOLO 训练',
        'custom_script': '自定义训练脚本',
        'custom_trainer': '自定义 Trainer',
    }
    return {
        'pending_confirmation': bool(pending),
        'dataset_path': str(draft.get('dataset_path') or '').strip(),
        'current_judgment': str(draft.get('data_summary') or '').strip(),
        'plan_reason': str(draft.get('reasoning_summary') or '').strip(),
        'execution_mode': execution_mode_map.get(execution_mode_raw, execution_mode_raw),
        'execution_backend': execution_backend_map.get(str(draft.get('execution_backend') or ''), str(draft.get('execution_backend') or '').strip()),
        'training_environment': str(draft.get('training_environment') or '').strip(),
        'model': str(args.get('model') or '').strip(),
        'data_yaml': str(args.get('data_yaml') or '').strip(),
        'classes_txt': str(args.get('classes_txt') or next_args.get('classes_txt') or '').strip(),
        'project': str(args.get('project') or '').strip(),
        'name': str(args.get('name') or '').strip(),
        'epochs': args.get('epochs'),
        'device': str(args.get('device') or '').strip(),
        'loop_requested': loop_like,
        'managed_level': str(args.get('managed_level') or '').strip(),
        'max_rounds': args.get('max_rounds'),
        'next_step': _human_training_step_name(next_step_tool),
        'next_step_tool': next_step_tool,
        'blockers': [str(item).strip() for item in (draft.get('blockers') or []) if str(item).strip()],
        'warnings': [str(item).strip() for item in (draft.get('warnings') or []) if str(item).strip()],
    }


def training_plan_render_error(
    draft: dict[str, Any],
    *,
    pending: bool,
    error: Exception | None = None,
) -> str:
    facts = training_plan_user_facts(draft, pending=pending)
    summary_bits: list[str] = []
    if facts.get('dataset_path'):
        summary_bits.append(f"数据集：{facts['dataset_path']}")
    if facts.get('model'):
        summary_bits.append(f"模型：{facts['model']}")
    if facts.get('classes_txt'):
        summary_bits.append(f"类名文件：{facts['classes_txt']}")
    if facts.get('next_step'):
        summary_bits.append(f"下一步：{facts['next_step']}")
    prefix = '模型这次没有成功生成计划说明。'
    if error:
        prefix = f'{prefix} 我不会再用固定模板冒充模型输出。'
    if summary_bits:
        return f"{prefix} 当前已确认的计划事实：{'；'.join(summary_bits)}。请稍后重试。"
    return f'{prefix} 请稍后重试。'


async def render_training_plan_message(
    *,
    planner_llm: Any,
    draft: dict[str, Any],
    pending: bool,
    render_training_plan_draft: TrainingPlanDraftRenderer,
    invoke_renderer_text: RendererTextInvoker,
) -> str:
    if not draft:
        return ''
    if planner_llm is None:
        return render_training_plan_draft(draft, pending=pending)

    facts = training_plan_user_facts(draft, pending=pending)
    messages = [
        SystemMessage(
            content=(
                '你是 YoloStudio Training Agent 的计划说明器。'
                '请基于已验证事实，用自然中文向用户说明当前训练计划。'
                '不要输出工具名、字段名、JSON、命令、payload、函数名，'
                '也不要使用“训练计划草案：”“原因和说明”“关键风险提示”这类固定模板标题。'
                '像同一个 Agent 在继续对话一样说明，不要每次都套相同句式。'
                '如果是循环训练，请明确说“循环训练”，不要混成普通训练。'
                '优先用 2 到 4 句自然中文：先说当前结论，再解释原因，最后说明下一步。'
                '如果 pending_confirmation=true，请用一句自然中文说明“如果你同意，我就按这个计划执行”。'
                '不要补充未验证事实。'
            )
        ),
        HumanMessage(
            content=(
                '请根据以下已验证事实，直接给用户一段自然中文说明：\n'
                f'{json.dumps(facts, ensure_ascii=False, indent=2)}'
            )
        ),
    ]
    text = await invoke_renderer_text(
        messages=messages,
        failure_event='planner_render_failed',
        failure_payload={
            'dataset_path': facts.get('dataset_path', ''),
            'next_step': facts.get('next_step', ''),
        },
    )
    if text:
        return text
    return training_plan_render_error(draft, pending=pending)


async def run_training_loop_start_orchestration(
    session_state: SessionState,
    *,
    user_text: str,
    thread_id: str,
    dataset_path: str,
    loop_args: dict[str, Any],
    observed_tools: dict[str, dict[str, Any]] | None = None,
    direct_tool: DirectToolInvoker,
    build_training_loop_start_fallback_plan_fn: Callable[..., dict[str, Any]],
    known_training_loop_data_yaml: LoopDataYamlResolver,
    append_event: EventAppender,
    compact_training_loop_start_fact: LoopFactCompactor,
    build_training_loop_start_draft_fn: Callable[..., dict[str, Any]],
    save_training_plan_draft: DraftSaver,
    append_ai_message: AssistantMessageAppender,
    handoff_to_graph: GraphHandoffInvoker,
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = dict(observed_tools or {})
    if not str(loop_args.get('model') or '').strip():
        reply = '当前还不能开启环训练：缺少预训练权重/模型。请先明确模型，例如 yolov8n.pt。'
        append_ai_message(reply)
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    async def _plan_once(step_index: int) -> dict[str, Any]:
        known_data_yaml = known_training_loop_data_yaml(loop_args, observed, dataset_path=dataset_path)
        if known_data_yaml:
            loop_args['data_yaml'] = known_data_yaml
        plan = build_training_loop_start_fallback_plan_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            loop_args=loop_args,
            observed_tools=observed,
        )
        append_event(
            'loop_start_planned',
            {
                'source': 'fallback',
                'decision': plan.get('decision'),
                'next_tool': plan.get('next_tool'),
                'step_index': step_index,
            },
        )
        return plan

    plan = await _plan_once(1)
    if plan.get('decision') == 'block':
        reply = str(plan.get('reason') or '当前还不能开启环训练。').strip()
        append_ai_message(reply)
        return {'status': 'completed', 'message': reply, 'tool_call': None}

    next_tool_name = str(plan.get('next_tool') or '').strip()
    next_tool_args = dict(plan.get('next_args') or {})
    if next_tool_name in {'training_readiness', 'list_training_environments'}:
        observed_result = await direct_tool(next_tool_name, _state_mode='observe', **next_tool_args)
        observed[next_tool_name] = observed_result
        append_event(
            'loop_start_observed_tool',
            {
                'tool': next_tool_name,
                'args': next_tool_args,
                'result': compact_training_loop_start_fact(next_tool_name, observed_result),
                'step_index': 1,
            },
        )
        plan = await _plan_once(2)
        if plan.get('decision') == 'block':
            reply = str(plan.get('reason') or '当前还不能开启环训练。').strip()
            append_ai_message(reply)
            return {'status': 'completed', 'message': reply, 'tool_call': None}
        next_tool_name = str(plan.get('next_tool') or '').strip()
        if next_tool_name in {'training_readiness', 'list_training_environments'}:
            reply = '当前还不能稳定规划下一步；读到的事实没有继续收敛。请换一种方式说明需求，或直接明确 data.yaml / 模型。'
            append_ai_message(reply)
            return {'status': 'completed', 'message': reply, 'tool_call': None}

    draft = build_training_loop_start_draft_fn(
        user_text=user_text,
        dataset_path=dataset_path,
        loop_args=loop_args,
        observed_tools=observed,
        plan=plan,
    )
    save_training_plan_draft(draft)
    return await handoff_to_graph(thread_id, user_text)


def _human_training_step_name(tool_name: str) -> str:
    normalized = str(tool_name or '').strip()
    mapping = {
        'prepare_dataset_for_training': '先准备数据集',
        'start_training': '启动训练',
        'start_training_loop': '启动循环训练',
        'training_preflight': '先做训练预检',
    }
    return mapping.get(normalized, normalized)
