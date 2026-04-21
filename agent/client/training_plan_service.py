from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    class _BaseMessage:
        def __init__(self, content: Any = '', **kwargs: Any):
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)

    class HumanMessage(_BaseMessage):
        pass

    class SystemMessage(_BaseMessage):
        pass

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


def normalize_training_device(value: Any, *, default: str = 'auto') -> str:
    device = str(value or '').strip()
    return device or default
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
        'device': normalize_training_device(planned_args.get('device')),
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
        'device': normalize_training_device(resolved_args.get('device') or planned_args.get('device')),
    }
    for field in TRAINING_PREFLIGHT_STRING_FIELDS:
        payload[field] = str(resolved_args.get(field) or planned_args.get(field) or '')
    for field in TRAINING_PREFLIGHT_OPTIONAL_FIELDS:
        payload[field] = resolved_args.get(field, planned_args.get(field))
    return payload


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
    planned_args['device'] = normalize_training_device(planned_args.get('device'))
    data_yaml = known_training_loop_data_yaml(planned_args, observed_tools, dataset_path=dataset_path)
    if data_yaml:
        planned_args['data_yaml'] = data_yaml
    next_tool_name = str(plan.get('next_tool') or '').strip()
    execution_mode = 'prepare_then_loop' if next_tool_name == 'prepare_dataset_for_training' else 'direct_loop'
    if next_tool_name == 'start_training_loop' and 'prepare_dataset_for_training' in observed_tools:
        execution_mode = 'prepare_then_loop'
    next_step_args = dict(plan.get('next_args') or {})
    if next_tool_name == 'start_training_loop':
        next_step_args['device'] = normalize_training_device(next_step_args.get('device') or planned_args.get('device'))
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
        'next_step_args': next_step_args,
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
        'device': normalize_training_device(args.get('device')),
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


def _human_training_step_name(tool_name: str) -> str:
    normalized = str(tool_name or '').strip()
    mapping = {
        'prepare_dataset_for_training': '先准备数据集',
        'start_training': '启动训练',
        'start_training_loop': '启动循环训练',
        'training_preflight': '先做训练预检',
    }
    return mapping.get(normalized, normalized)
