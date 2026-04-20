from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage

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

    readiness_error = str(readiness.get('error') or '').strip()
    readiness_unavailable = (not readiness) or (
        readiness.get('ok') is False
        and any(token in readiness_error.lower() for token in ('未找到工具', 'tool not found', 'unknown tool'))
    )
    if readiness_unavailable and dataset_path and requested_model and not discussion_only:
        fallback_args = dict(requested_args)
        fallback_data_yaml = str(fallback_args.get('data_yaml') or '').strip()
        if fallback_data_yaml:
            fallback_args['data_yaml'] = fallback_data_yaml
            next_tool_name = 'start_training'
            next_tool_args = dict(fallback_args)
        else:
            next_tool_name = 'prepare_dataset_for_training'
            next_tool_args = {'dataset_path': dataset_path}
            if wants_split:
                next_tool_args['force_split'] = True
            explicit_classes_txt = str(fallback_args.get('classes_txt') or '').strip()
            if explicit_classes_txt:
                next_tool_args['classes_txt'] = explicit_classes_txt
        draft = build_training_plan_draft_fn(
            user_text=user_text,
            dataset_path=dataset_path,
            readiness={},
            preflight={},
            next_tool_name=next_tool_name,
            next_tool_args=next_tool_args,
            planned_training_args=fallback_args,
        )
        return await _render_orchestration_result(
            draft,
            pending=True,
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
    entrypoint_result = await run_training_loop_start_entrypoint(
        session_state,
        user_text=user_text,
        dataset_path=dataset_path,
        loop_args=loop_args,
        observed_tools=observed_tools,
        direct_tool=direct_tool,
        build_training_loop_start_fallback_plan_fn=build_training_loop_start_fallback_plan_fn,
        known_training_loop_data_yaml=known_training_loop_data_yaml,
        append_event=append_event,
        compact_training_loop_start_fact=compact_training_loop_start_fact,
        build_training_loop_start_draft_fn=build_training_loop_start_draft_fn,
    )
    draft = dict(entrypoint_result.get('draft') or {})
    if draft:
        save_training_plan_draft(draft)
    reply = str(entrypoint_result.get('reply') or '').strip()
    if reply and not entrypoint_result.get('defer_to_graph'):
        append_ai_message(reply)
        return {'status': 'completed', 'message': reply, 'tool_call': None}
    if entrypoint_result.get('defer_to_graph'):
        return await handoff_to_graph(thread_id, user_text)
    return {'status': 'completed', 'message': reply, 'tool_call': None}


async def run_training_loop_start_entrypoint(
    session_state: SessionState,
    *,
    user_text: str,
    dataset_path: str,
    loop_args: dict[str, Any],
    observed_tools: dict[str, dict[str, Any]] | None = None,
    direct_tool: DirectToolInvoker,
    build_training_loop_start_fallback_plan_fn: Callable[..., dict[str, Any]],
    known_training_loop_data_yaml: LoopDataYamlResolver,
    append_event: EventAppender,
    compact_training_loop_start_fact: LoopFactCompactor,
    build_training_loop_start_draft_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    observed: dict[str, dict[str, Any]] = dict(observed_tools or {})
    if not str(loop_args.get('model') or '').strip():
        reply = '当前还不能开启环训练：缺少预训练权重/模型。请先明确模型，例如 yolov8n.pt。'
        return {'reply': reply, 'draft': None, 'defer_to_graph': False}

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
        return {'reply': reply, 'draft': None, 'defer_to_graph': False}

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
            return {'reply': reply, 'draft': None, 'defer_to_graph': False}
        next_tool_name = str(plan.get('next_tool') or '').strip()
        if next_tool_name in {'training_readiness', 'list_training_environments'}:
            reply = '当前还不能稳定规划下一步；读到的事实没有继续收敛。请换一种方式说明需求，或直接明确 data.yaml / 模型。'
            return {'reply': reply, 'draft': None, 'defer_to_graph': False}

    draft = build_training_loop_start_draft_fn(
        user_text=user_text,
        dataset_path=dataset_path,
        loop_args=loop_args,
        observed_tools=observed,
        plan=plan,
    )
    return {'reply': '', 'draft': draft, 'defer_to_graph': True}


def _human_training_step_name(tool_name: str) -> str:
    normalized = str(tool_name or '').strip()
    mapping = {
        'prepare_dataset_for_training': '先准备数据集',
        'start_training': '启动训练',
        'start_training_loop': '启动循环训练',
        'training_preflight': '先做训练预检',
    }
    return mapping.get(normalized, normalized)
