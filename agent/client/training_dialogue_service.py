from __future__ import annotations

from typing import Any, Awaitable, Callable

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.training_plan_service import (
    build_training_preflight_tool_args,
    resolve_training_start_args,
)

DirectToolInvoker = Callable[..., Awaitable[dict[str, Any]]]
TrainingPlanDraftBuilder = Callable[..., dict[str, Any]]
TrainingArgsCollector = Callable[..., dict[str, Any]]
TrainingDiscussionChecker = Callable[[str], bool]
TrainingExecutionBackendExtractor = Callable[[str], str]
TrainingAdvancedDetailsChecker = Callable[[str], bool]


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
