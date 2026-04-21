from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Mapping

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.runnables import RunnableConfig
except Exception:
    class _BaseMessage:
        def __init__(self, content: Any = '', **kwargs: Any):
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)

    class AIMessage(_BaseMessage):
        pass

    class HumanMessage(_BaseMessage):
        pass

    class SystemMessage(_BaseMessage):
        pass

    class ToolMessage(_BaseMessage):
        pass

    RunnableConfig = Any  # type: ignore[assignment]

try:
    from langgraph.graph import END
except Exception:
    END = '__end__'  # type: ignore[assignment]

try:
    from langgraph.errors import GraphInterrupt
except Exception:
    GraphInterrupt = None  # type: ignore[assignment]

try:
    from langgraph.types import Command, Interrupt, interrupt
except Exception:
    class Command:  # type: ignore[override]
        def __init__(self, *, update: Any = None, goto: str | None = None, resume: Any = None, graph: Any = None):
            self.update = update
            self.goto = goto
            self.resume = resume
            self.graph = graph

    Interrupt = None  # type: ignore[assignment]

    def interrupt(value: Any) -> Any:
        return value

try:
    from langgraph._internal._constants import (
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_SCRATCHPAD,
        CONFIG_KEY_SEND,
        RESUME,
    )
except Exception:
    try:
        from langgraph.constants import (  # type: ignore[attr-defined]
            CONFIG_KEY_CHECKPOINT_NS,
            CONFIG_KEY_SCRATCHPAD,
            CONFIG_KEY_SEND,
            RESUME,
        )
    except Exception:
        CONFIG_KEY_CHECKPOINT_NS = None  # type: ignore[assignment]
        CONFIG_KEY_SCRATCHPAD = None  # type: ignore[assignment]
        CONFIG_KEY_SEND = None  # type: ignore[assignment]
        RESUME = None  # type: ignore[assignment]

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.mainline_route_support import resolve_mainline_dispatch_payload
from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_from_draft,
    extract_training_plan_context_from_state,
)
from yolostudio_agent.agent.client.training_plan_service import (
    build_training_preflight_tool_args,
    normalize_training_device,
    resolve_training_start_args,
    training_plan_render_error,
    training_plan_user_facts,
)
from yolostudio_agent.agent.client.training_request_service import (
    prepare_training_request_context,
    resolve_training_request_entrypoint_guard,
    run_prepare_only_flow,
)
from yolostudio_agent.agent.client.training_schemas import (
    PendingTurnIntent,
    coerce_training_plan,
    merge_training_plan_edits,
    update_plan_after_prepare,
)

ClientGetter = Callable[[], Any | None]
TrainingPlanDraftRenderer = Callable[..., str]
RendererTextInvoker = Callable[..., Awaitable[str]]


def _thread_id_from_config(config: Any) -> str:
    if isinstance(config, dict):
        configurable = dict(config.get('configurable') or {})
        return str(configurable.get('thread_id') or '').strip()
    configurable = dict(getattr(config, 'configurable', {}) or {})
    return str(configurable.get('thread_id') or '').strip()


def _configurable_from_config(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config.get('configurable') or {})
    return dict(getattr(config, 'configurable', {}) or {})


def _interrupt_with_runtime_config(config: Any, payload: Any) -> Any:
    configurable = _configurable_from_config(config)
    if not configurable:
        return interrupt(payload)
    if (
        GraphInterrupt is None
        or Interrupt is None
        or CONFIG_KEY_CHECKPOINT_NS is None
        or CONFIG_KEY_SCRATCHPAD is None
        or CONFIG_KEY_SEND is None
        or RESUME is None
    ):
        return interrupt(payload)

    scratchpad = configurable.get(CONFIG_KEY_SCRATCHPAD)
    send = configurable.get(CONFIG_KEY_SEND)
    checkpoint_ns = configurable.get(CONFIG_KEY_CHECKPOINT_NS)
    interrupt_counter = getattr(scratchpad, 'interrupt_counter', None)
    get_null_resume = getattr(scratchpad, 'get_null_resume', None)
    resume_values = getattr(scratchpad, 'resume', None)
    if (
        scratchpad is None
        or not callable(send)
        or checkpoint_ns is None
        or not callable(interrupt_counter)
        or not callable(get_null_resume)
        or not isinstance(resume_values, list)
    ):
        return interrupt(payload)

    idx = interrupt_counter()
    if idx < len(resume_values):
        send([(RESUME, resume_values)])
        return resume_values[idx]

    null_resume = get_null_resume(True)
    if null_resume is not None:
        if len(resume_values) == idx:
            resume_values.append(null_resume)
        send([(RESUME, resume_values)])
        return resume_values[idx]

    raise GraphInterrupt((Interrupt.from_ns(value=payload, ns=checkpoint_ns),))


def _latest_human_text_from_state(state: Mapping[str, Any]) -> str:
    for message in reversed(list((state or {}).get('messages') or [])):
        if isinstance(message, HumanMessage):
            return str(getattr(message, 'content', '') or '').strip()
        if getattr(message.__class__, '__name__', '') == 'HumanMessage':
            return str(getattr(message, 'content', '') or '').strip()
    return ''


def _tool_result_message(tool_name: str, *, parsed: dict[str, Any], tool_call_id: str = '', status: str = '') -> ToolMessage:
    kwargs: dict[str, Any] = {
        'content': json.dumps(parsed, ensure_ascii=False),
        'name': tool_name,
        'tool_call_id': tool_call_id,
    }
    if status:
        kwargs['status'] = status
    return ToolMessage(**kwargs)


def _training_state_update_from_interrupt_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(payload or {})
    return {
        'training_plan': dict(payload.get('plan') or {}) or None,
        'training_phase': str(payload.get('phase') or '').strip() or None,
        'training_execution_mode': str(payload.get('execution_mode') or '').strip() or None,
        'training_next_step_tool': str(payload.get('next_step_tool') or '').strip() or None,
        'training_next_step_args': dict(payload.get('next_step_args') or {}) or None,
        'suspended_training_plan': dict(payload.get('suspended_training_plan') or {}) or None,
        'pending_new_task': str(payload.get('pending_new_task') or '').strip() or None,
        'training_status_reply': str(payload.get('status_reply') or '').strip() or None,
        'training_plan_context': None,
        'training_entry_request': None,
        'pending_confirmation': None,
    }


def _normalize_decision(value: Any) -> PendingTurnIntent:
    if isinstance(value, PendingTurnIntent):
        return value
    if isinstance(value, Mapping):
        return PendingTurnIntent.model_validate(dict(value))
    return PendingTurnIntent(action='unclear', reason='')


def _update_plan_from_preflight(
    client: Any,
    *,
    plan_payload: dict[str, Any],
    preflight: dict[str, Any],
    next_tool_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated_plan = dict(plan_payload or {})
    resolved_args = dict(preflight.get('resolved_args') or {})
    if next_tool_name == 'start_training_loop' and 'epochs' in resolved_args and 'epochs_per_round' not in resolved_args:
        resolved_args['epochs_per_round'] = resolved_args.get('epochs')
    if next_tool_name == 'start_training_loop':
        resolved_args.pop('epochs', None)
    for key in (
        'model',
        'data_yaml',
        'batch',
        'imgsz',
        'device',
        'training_environment',
        'project',
        'name',
        'fraction',
        'classes',
        'single_cls',
        'optimizer',
        'freeze',
        'resume',
        'lr0',
        'patience',
        'workers',
        'amp',
    ):
        value = resolved_args.get(key)
        if value not in (None, ''):
            updated_plan[key] = value
        elif key in {'fraction', 'classes', 'single_cls', 'freeze', 'resume', 'lr0', 'patience', 'workers', 'amp'}:
            updated_plan[key] = [] if key == 'classes' and value is None else value
    updated_plan['device'] = normalize_training_device(updated_plan.get('device'))
    if str(updated_plan.get('mode') or '').strip().lower() == 'loop':
        for key in ('max_rounds', 'epochs_per_round', 'loop_name'):
            value = resolved_args.get(key)
            if value not in (None, ''):
                updated_plan[key] = value
    else:
        epochs_value = resolved_args.get('epochs')
        if epochs_value not in (None, ''):
            updated_plan['epochs'] = epochs_value
    blockers = [str(item).strip() for item in (preflight.get('blockers') or []) if str(item).strip()]
    warnings = [str(item).strip() for item in (preflight.get('warnings') or []) if str(item).strip()]
    summary = str(preflight.get('summary') or '').strip()
    updated_plan['blockers'] = blockers
    updated_plan['warnings'] = warnings
    updated_plan['readiness_summary'] = summary
    return updated_plan, (resolved_args or dict(plan_payload or {}))


async def _render_orchestration_result(
    draft: dict[str, Any],
    *,
    pending: bool,
    render_training_plan_message: Callable[[dict[str, Any], bool], Awaitable[str]],
) -> dict[str, Any]:
    return {
        'draft': draft,
        'reply': await render_training_plan_message(draft, pending=pending),
        'defer_to_graph': pending,
    }


async def plan_training_request(
    *,
    user_text: str,
    dataset_path: str,
    readiness: dict[str, Any] | None,
    requested_args: dict[str, Any] | None,
    wants_split: bool,
    discussion_only: bool,
    execution_backend: str,
    direct_tool: Callable[..., Awaitable[dict[str, Any]]],
    build_training_plan_draft_fn: Callable[..., dict[str, Any]],
    render_training_plan_message: Callable[[dict[str, Any], bool], Awaitable[str]],
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


def build_training_loop_start_fallback_plan_core(
    *,
    user_text: str,
    dataset_path: str,
    loop_args: dict[str, Any],
    observed_tools: dict[str, dict[str, Any]] | None = None,
    known_training_loop_data_yaml: Callable[..., str],
    build_loop_prepare_args: Callable[[str, str], dict[str, Any]],
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


async def render_training_plan_message_core(
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


async def plan_training_loop_start_request(
    session_state: Any,
    *,
    user_text: str,
    dataset_path: str,
    loop_args: dict[str, Any],
    observed_tools: dict[str, dict[str, Any]] | None = None,
    direct_tool: Callable[..., Awaitable[dict[str, Any]]],
    build_training_loop_start_fallback_plan_fn: Callable[..., dict[str, Any]],
    known_training_loop_data_yaml: Callable[..., str],
    append_event: Callable[[str, dict[str, Any]], None],
    compact_training_loop_start_fact: Callable[[str, dict[str, Any]], dict[str, Any]],
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


async def training_confirmation_node(
    state: Mapping[str, Any],
    config: RunnableConfig | None = None,
) -> Any:
    plan = coerce_training_plan(state.get('training_plan', {}))
    phase = str(state.get('training_phase', 'prepare') or 'prepare').strip().lower() or 'prepare'
    suspended_plan = state.get('suspended_training_plan')
    next_step_tool = str(state.get('training_next_step_tool', '') or '').strip()
    next_step_args = dict(state.get('training_next_step_args', {}) or {})
    execution_mode = str(state.get('training_execution_mode', '') or '').strip()
    status_reply = str(state.get('training_status_reply', '') or '').strip()
    payload = {
        'type': 'training_confirmation',
        'phase': phase,
        'execution_mode': execution_mode,
        'plan': plan.model_dump(),
        'next_step_tool': next_step_tool,
        'next_step_args': next_step_args,
        'status_reply': status_reply,
        'suspended_training_plan': dict(suspended_plan or {}) if isinstance(suspended_plan, Mapping) else None,
    }
    decision = _normalize_decision(_interrupt_with_runtime_config(config, payload))
    action = str(decision.action or 'unclear').strip().lower()

    if action == 'approve':
        return Command(goto='execute_prepare' if phase == 'prepare' else 'execute_training')

    if action == 'reject':
        return Command(update={
            'training_plan': None,
            'training_phase': None,
            'suspended_training_plan': None,
            'pending_new_task': '',
            'training_status_reply': '',
            'training_entry_request': None,
            'status': 'cancelled',
        })

    if action == 'edit':
        updated = merge_training_plan_edits(plan, decision.edits)
        updated_args = dict(next_step_args)
        updated_execution_mode = execution_mode
        raw_edits = {}
        if decision.edits is not None:
            model_dump = getattr(decision.edits, 'model_dump', None)
            if callable(model_dump):
                raw_edits = model_dump(exclude_none=False)
                fields_set = set(
                    getattr(decision.edits, 'model_fields_set', None)
                    or getattr(decision.edits, '__fields_set__', set())
                    or set()
                )
                if fields_set:
                    raw_edits = {key: raw_edits.get(key) for key in fields_set if key in raw_edits}
                else:
                    raw_edits = {
                        key: value
                        for key, value in raw_edits.items()
                        if value is not None
                    }
            else:
                raw_edits = dict(vars(decision.edits))
        if phase == 'prepare':
            if updated.dataset_path:
                updated_args['dataset_path'] = updated.dataset_path
            if raw_edits.get('prepare_only'):
                updated_execution_mode = 'prepare_only'
            if 'force_split' in raw_edits:
                if raw_edits.get('force_split'):
                    updated_args['force_split'] = True
                else:
                    updated_args.pop('force_split', None)
        else:
            updated_args = updated.model_dump()
            if next_step_tool == 'start_training_loop' and 'epochs_per_round' in updated_args and 'epochs' not in updated_args:
                updated_args['epochs'] = updated_args['epochs_per_round']
        approve_after_edit = bool(getattr(decision, 'approve_after_edit', None))
        next_goto = 'refresh_training_start_after_edit' if phase == 'start' else 'training_confirmation'
        if approve_after_edit:
            next_goto = 'execute_prepare' if phase == 'prepare' else 'execute_training'
        return Command(update={
            'training_plan': updated.model_dump(),
            'training_phase': phase,
            'training_next_step_tool': next_step_tool,
            'training_next_step_args': updated_args,
            'training_execution_mode': updated_execution_mode,
            'training_status_reply': '',
        }, goto=next_goto)

    if action == 'new_task':
        return Command(update={
            'suspended_training_plan': {
                'plan': plan.model_dump(),
                'phase': phase,
                'execution_mode': execution_mode,
                'next_step_tool': next_step_tool,
                'next_step_args': next_step_args,
            },
            'pending_new_task': str(decision.reason or '').strip(),
            'training_phase': phase,
            'training_status_reply': '',
        }, goto='route_new_task')

    if action == 'status':
        return Command(update={
            'training_plan': plan.model_dump(),
            'training_phase': phase,
            'training_status_reply': str(decision.reason or '').strip(),
        }, goto='answer_training_status')

    return Command(update={
        'training_plan': plan.model_dump(),
        'training_phase': phase,
    }, goto='training_confirmation')


training_confirmation_node.__annotations__['config'] = RunnableConfig | None


def post_prepare_node(state: Mapping[str, Any]) -> Command:
    plan = coerce_training_plan(state.get('training_plan', {}))
    updated = update_plan_after_prepare(
        plan,
        prepare_result=state.get('prepare_result', {}),
        readiness=state.get('training_preflight', {}) or state.get('training_readiness', {}),
    )
    updated = updated.model_copy(update={'device': normalize_training_device(getattr(updated, 'device', ''))})
    resolved_args = dict(state.get('training_preflight', {}) or {}).get('resolved_args') or updated.model_dump()
    resolved_args['device'] = normalize_training_device(resolved_args.get('device') or getattr(updated, 'device', ''))
    next_tool_name = 'start_training_loop' if getattr(updated, 'mode', 'train') == 'loop' else 'start_training'
    if next_tool_name == 'start_training_loop' and 'epochs_per_round' in resolved_args and 'epochs' not in resolved_args:
        resolved_args['epochs'] = resolved_args['epochs_per_round']
    return Command(update={
        'training_plan': updated.model_dump(),
        'training_phase': 'start',
        'training_next_step_tool': next_tool_name,
        'training_next_step_args': resolved_args,
        'training_status_reply': '',
        'training_entry_request': None,
    }, goto='training_confirmation')


def answer_training_status_node(state: Mapping[str, Any]) -> Command:
    plan = coerce_training_plan(state.get('training_plan', {}))
    phase = str(state.get('training_phase', 'prepare') or 'prepare').strip().lower() or 'prepare'
    reply = str(state.get('training_status_reply', '') or '').strip()
    return Command(update={
        'training_plan': plan.model_dump(),
        'training_phase': phase,
        'training_status_reply': reply,
    }, goto='training_confirmation')


def build_training_workflow_nodes(*, client_getter: ClientGetter) -> dict[str, Any]:
    async def plan_training_node(
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
    ) -> Command:
        client = client_getter()
        if client is None:
            return Command(goto='agent_runtime')
        thread_id = _thread_id_from_config(config)
        entry_request = dict((state or {}).get('training_entry_request') or {})
        context = extract_training_plan_context_from_state(dict(state or {})) or {}
        if context and not entry_request:
            interrupt_payload = client._draft_to_training_confirmation_interrupt(
                context,
                thread_id=thread_id,
            )
            if interrupt_payload:
                update = _training_state_update_from_interrupt_payload(interrupt_payload)
                update['training_entry_request'] = None
                return Command(update=update, goto='training_confirmation')
        latest_user_text = str(entry_request.get('user_text') or _latest_human_text_from_state(state)).strip()
        if not latest_user_text:
            return Command(update={'training_entry_request': None}, goto='agent_runtime')
        training_entrypoint_args = dict(entry_request.get('training_entrypoint_request_args') or {})
        prepare_only_followup = await run_prepare_only_flow(
            user_text=latest_user_text,
            looks_like_prepare_only_request=client._looks_like_prepare_only_request,
            extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
            local_path_exists=lambda path: __import__('pathlib').Path(path).expanduser().exists(),
            direct_tool=client.direct_tool,
            collect_requested_training_args=client._collect_requested_training_args,
            build_training_plan_draft_fn=client._build_training_plan_draft,
            render_tool_result_message=client._render_tool_result_message,
        )
        if prepare_only_followup:
            entrypoint_result = {
                'reply': str(prepare_only_followup.get('reply') or '').strip(),
                'draft': dict(prepare_only_followup.get('draft') or {}),
                'defer_to_graph': str(prepare_only_followup.get('action') or '').strip() == 'save_draft_and_handoff',
            }
        elif bool(training_entrypoint_args.get('wants_training_loop_start')):
            resolved_yaml = client._session_training_data_yaml(
                dataset_path=str(training_entrypoint_args.get('dataset_path') or '').strip(),
            )
            loop_args = client._collect_requested_training_loop_args(
                latest_user_text,
                data_yaml=resolved_yaml if resolved_yaml else None,
            )
            entrypoint_result = await plan_training_loop_start_request(
                client.session_state,
                user_text=latest_user_text,
                dataset_path=str(training_entrypoint_args.get('dataset_path') or '').strip(),
                loop_args=loop_args,
                direct_tool=client.direct_tool,
                build_training_loop_start_fallback_plan_fn=client._build_training_loop_start_fallback_plan,
                known_training_loop_data_yaml=client._known_training_loop_data_yaml,
                append_event=lambda event_type, payload: client.memory.append_event(
                    client.session_state.session_id,
                    event_type,
                    payload,
                ),
                compact_training_loop_start_fact=client._compact_training_loop_start_fact,
                build_training_loop_start_draft_fn=client._build_training_loop_start_draft,
            )
        else:
            mainline_context = client._collect_mainline_context(latest_user_text)
            route_state = await client._resolve_mainline_route_state(latest_user_text, mainline_context)
            dispatch_payload = resolve_mainline_dispatch_payload(
                mainline_context=mainline_context,
                route_state=route_state,
            )
            training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or training_entrypoint_args)
            guard = resolve_training_request_entrypoint_guard(
                session_state=client.session_state,
                user_text=latest_user_text,
                normalized_text=str(training_entrypoint_args.get('normalized_text') or '').lower(),
                dataset_path=str(training_entrypoint_args.get('dataset_path') or '').strip(),
                wants_train=bool(training_entrypoint_args.get('wants_train')),
                wants_predict=bool(training_entrypoint_args.get('wants_predict')),
                no_train=bool(training_entrypoint_args.get('no_train')),
                readiness_only_query=bool(training_entrypoint_args.get('readiness_only_query')),
                wants_training_outcome_analysis=bool(training_entrypoint_args.get('wants_training_outcome_analysis')),
                wants_next_step_guidance=bool(training_entrypoint_args.get('wants_next_step_guidance')),
                wants_training_knowledge=bool(training_entrypoint_args.get('wants_training_knowledge')),
                wants_training_revision=bool(training_entrypoint_args.get('wants_training_revision')),
                wants_stop_training=bool(training_entrypoint_args.get('wants_stop_training')),
                blocks_training_start=bool(training_entrypoint_args.get('blocks_training_start')),
                explicit_run_ids=list(training_entrypoint_args.get('explicit_run_ids') or []),
                extract_model_from_text=intent_parsing.extract_model_from_text,
            )
            if not guard.get('proceed'):
                if guard.get('return_none'):
                    return Command(update={'training_entry_request': None}, goto='agent_runtime')
                entrypoint_result = {
                    'reply': str(guard.get('reply') or '').strip(),
                    'draft': dict(guard.get('draft') or {}) or None,
                    'defer_to_graph': bool(guard.get('defer_to_graph')),
                }
            else:
                prepared_context = await prepare_training_request_context(
                    session_state=client.session_state,
                    user_text=latest_user_text,
                    dataset_path=str(training_entrypoint_args.get('dataset_path') or '').strip(),
                    frame_followup_path=str(training_entrypoint_args.get('frame_followup_path') or ''),
                    current_training_plan_context=context,
                    direct_tool=client.direct_tool,
                    collect_requested_training_args=client._collect_requested_training_args,
                )
                entrypoint_result = await plan_training_request(
                    user_text=latest_user_text,
                    dataset_path=str(training_entrypoint_args.get('dataset_path') or '').strip(),
                    readiness=dict(prepared_context.get('readiness') or {}),
                    requested_args=dict(prepared_context.get('requested_args') or {}),
                    wants_split=bool(training_entrypoint_args.get('wants_split')),
                    discussion_only=client._is_training_discussion_only(latest_user_text),
                    execution_backend=client._extract_training_execution_backend_from_text(latest_user_text),
                    direct_tool=client.direct_tool,
                    build_training_plan_draft_fn=client._build_training_plan_draft,
                    render_training_plan_message=client._render_training_plan_message,
                )
        draft = dict(entrypoint_result.get('draft') or {})
        discussion_only = client._is_training_discussion_only(latest_user_text)
        if draft and discussion_only:
            rendered = str(entrypoint_result.get('reply') or '').strip()
            if not rendered:
                rendered = await client._render_training_plan_message(
                    draft,
                    pending=bool(draft.get('next_step_tool')),
                )
            return Command(
                update={
                    'messages': [AIMessage(content=rendered)] if rendered else [],
                    'training_plan_context': build_training_plan_context_from_draft(draft),
                    'training_entry_request': None,
                },
                goto=END,
            )
        if bool(entrypoint_result.get('defer_to_graph')) and draft:
            interrupt_payload = client._draft_to_training_confirmation_interrupt(
                draft,
                thread_id=thread_id,
            )
            if interrupt_payload:
                update = _training_state_update_from_interrupt_payload(interrupt_payload)
                update['training_entry_request'] = None
                return Command(update=update, goto='training_confirmation')
        reply = str(entrypoint_result.get('reply') or '').strip()
        if reply:
            return Command(
                update={
                    'messages': [AIMessage(content=reply)],
                    'training_plan_context': None,
                    'training_entry_request': None,
                },
                goto=END,
            )
        return Command(update={'training_entry_request': None}, goto='agent_runtime')

    async def refresh_training_start_after_edit(
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
    ) -> Command:
        del config
        client = client_getter()
        if client is None:
            return Command(goto='training_confirmation')
        plan_payload = dict((state or {}).get('training_plan') or {})
        if not plan_payload:
            return Command(goto='training_confirmation')
        next_tool_name = str((state or {}).get('training_next_step_tool') or '').strip()
        if next_tool_name not in {'start_training', 'start_training_loop'}:
            return Command(goto='training_confirmation')
        preflight_args = dict((state or {}).get('training_next_step_args') or plan_payload)
        preflight = await client.direct_tool('training_preflight', **preflight_args)
        updated_plan, resolved_args = _update_plan_from_preflight(
            client,
            plan_payload=plan_payload,
            preflight=preflight,
            next_tool_name=next_tool_name,
        )
        return Command(
            update={
                'training_plan': updated_plan,
                'training_next_step_tool': next_tool_name,
                'training_next_step_args': resolved_args,
                'training_preflight': preflight,
                'training_status_reply': '',
            },
            goto='training_confirmation',
        )

    async def execute_prepare(
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
    ) -> Command:
        del config
        client = client_getter()
        if client is None:
            return Command(goto='training_confirmation')
        plan_payload = dict((state or {}).get('training_plan') or {})
        if not plan_payload:
            return Command(goto='training_confirmation')
        plan = coerce_training_plan(plan_payload)
        next_args = dict((state or {}).get('training_next_step_args') or {'dataset_path': plan.dataset_path})
        next_args = {key: value for key, value in next_args.items() if value is not None}
        prepare_result = await client.direct_tool('prepare_dataset_for_training', **next_args)
        messages: list[Any] = [_tool_result_message('prepare_dataset_for_training', parsed=prepare_result)]
        if not prepare_result.get('ok'):
            failed_plan = dict(plan_payload)
            failed_plan['blockers'] = [str(prepare_result.get('error') or prepare_result.get('summary') or '数据准备失败').strip()]
            failed_plan['prepare_summary'] = str(prepare_result.get('summary') or prepare_result.get('error') or '').strip()
            return Command(
                update={
                    'messages': messages,
                    'training_plan': failed_plan,
                    'training_status_reply': failed_plan['prepare_summary'],
                },
                goto='training_confirmation',
            )
        preflight_args = client._model_dump_compat(plan)
        prepared_yaml = str(prepare_result.get('data_yaml') or prepare_result.get('resolved_data_yaml') or '').strip()
        if prepared_yaml:
            preflight_args['data_yaml'] = prepared_yaml
        if getattr(plan, 'mode', 'train') == 'loop':
            preflight_args.setdefault('epochs', getattr(plan, 'epochs_per_round', None))
        preflight = await client.direct_tool('training_preflight', **preflight_args)
        messages.append(_tool_result_message('training_preflight', parsed=preflight))
        return Command(
            update={
                'messages': messages,
                'prepare_result': prepare_result,
                'training_preflight': preflight,
                'training_status_reply': '',
            },
            goto='post_prepare',
        )

    async def execute_training(
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
    ) -> Command:
        del config
        client = client_getter()
        if client is None:
            return Command(goto='training_confirmation')
        plan_payload = dict((state or {}).get('training_plan') or {})
        if not plan_payload:
            return Command(goto='training_confirmation')
        plan = coerce_training_plan(plan_payload)
        next_tool_name = str((state or {}).get('training_next_step_tool') or '').strip()
        if next_tool_name not in {'start_training', 'start_training_loop'}:
            next_tool_name = 'start_training_loop' if getattr(plan, 'mode', 'train') == 'loop' else 'start_training'
        next_args = dict((state or {}).get('training_next_step_args') or client._model_dump_compat(plan))
        next_args = {key: value for key, value in next_args.items() if value is not None}
        parsed = await client.direct_tool(next_tool_name, **next_args)
        reply = await client._render_tool_result_message(next_tool_name, parsed)
        messages: list[Any] = [_tool_result_message(next_tool_name, parsed=parsed)]
        if reply:
            messages.append(AIMessage(content=reply))
        if parsed.get('ok'):
            return Command(
                update={
                    'messages': messages,
                    'training_plan': None,
                    'training_phase': None,
                    'training_execution_mode': None,
                    'training_next_step_tool': None,
                    'training_next_step_args': None,
                    'suspended_training_plan': None,
                    'pending_new_task': None,
                    'training_status_reply': '',
                    'training_plan_context': None,
                    'training_entry_request': None,
                },
                goto=END,
            )
        failed_plan = dict(plan_payload)
        failed_plan['blockers'] = [str(parsed.get('error') or parsed.get('summary') or '训练启动失败').strip()]
        failed_plan['readiness_summary'] = str(parsed.get('summary') or parsed.get('error') or '').strip()
        return Command(
            update={
                'messages': messages,
                'training_plan': failed_plan,
                'training_status_reply': str(reply or failed_plan['readiness_summary']).strip(),
            },
            goto='training_confirmation',
        )

    async def route_new_task(
        state: Mapping[str, Any],
        config: RunnableConfig | None = None,
    ) -> Command:
        del config
        pending_new_task = str((state or {}).get('pending_new_task') or '').strip()
        if not pending_new_task:
            suspended = dict((state or {}).get('suspended_training_plan') or {})
            if suspended:
                return Command(
                    update={
                        'training_plan': dict(suspended.get('plan') or {}) or None,
                        'training_phase': str(suspended.get('phase') or '').strip() or None,
                        'training_execution_mode': str(suspended.get('execution_mode') or '').strip() or None,
                        'training_next_step_tool': str(suspended.get('next_step_tool') or '').strip() or None,
                        'training_next_step_args': dict(suspended.get('next_step_args') or {}) or None,
                        'suspended_training_plan': None,
                        'training_status_reply': '',
                    },
                    goto='training_confirmation',
                )
            return Command(goto='agent_runtime')
        return Command(
            update={
                'messages': [HumanMessage(content=pending_new_task)],
                'pending_new_task': None,
                'training_status_reply': '',
            },
            goto='agent_runtime',
        )

    plan_training_node.__annotations__['config'] = RunnableConfig | None
    refresh_training_start_after_edit.__annotations__['config'] = RunnableConfig | None
    execute_prepare.__annotations__['config'] = RunnableConfig | None
    execute_training.__annotations__['config'] = RunnableConfig | None
    route_new_task.__annotations__['config'] = RunnableConfig | None

    return {
        'plan_training': plan_training_node,
        'training_confirmation': training_confirmation_node,
        'refresh_training_start_after_edit': refresh_training_start_after_edit,
        'execute_prepare': execute_prepare,
        'post_prepare': post_prepare_node,
        'execute_training': execute_training,
        'answer_training_status': answer_training_status_node,
        'route_new_task': route_new_task,
    }


def install_training_workflow_nodes(workflow: Any, *, client_getter: ClientGetter) -> dict[str, Any]:
    nodes = build_training_workflow_nodes(client_getter=client_getter)
    for name, node in nodes.items():
        workflow.add_node(name, node)
    return nodes
