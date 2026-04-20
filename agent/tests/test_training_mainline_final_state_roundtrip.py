from __future__ import annotations

import asyncio
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

try:
    import langchain_ollama  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

try:
    import langchain_core.messages  # type: ignore  # noqa: F401
except Exception:
    core_mod = types.ModuleType('langchain_core')
    messages_mod = types.ModuleType('langchain_core.messages')
    tools_mod = types.ModuleType('langchain_core.tools')

    class _BaseMessage:
        def __init__(self, content=''):
            self.content = content

    class _AIMessage(_BaseMessage):
        def __init__(self, content='', tool_calls=None):
            super().__init__(content)
            self.tool_calls = tool_calls or []

    class _HumanMessage(_BaseMessage):
        pass

    class _SystemMessage(_BaseMessage):
        pass

    class _ToolMessage(_BaseMessage):
        def __init__(self, content='', name='', tool_call_id=''):
            super().__init__(content)
            self.name = name
            self.tool_call_id = tool_call_id

    class _BaseTool:
        name = 'fake'
        description = 'fake'
        args_schema = None

    class _StructuredTool(_BaseTool):
        @classmethod
        def from_function(cls, func=None, coroutine=None, name='', description='', args_schema=None, return_direct=False):
            tool = cls()
            tool.func = func
            tool.coroutine = coroutine
            tool.name = name
            tool.description = description
            tool.args_schema = args_schema
            tool.return_direct = return_direct
            return tool

    messages_mod.AIMessage = _AIMessage
    messages_mod.BaseMessage = _BaseMessage
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.SystemMessage = _SystemMessage
    messages_mod.ToolMessage = _ToolMessage
    tools_mod.BaseTool = _BaseTool
    tools_mod.StructuredTool = _StructuredTool
    core_mod.messages = messages_mod
    core_mod.tools = tools_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.messages'] = messages_mod
    sys.modules['langchain_core.tools'] = tools_mod

try:
    import langchain_mcp_adapters.client  # type: ignore  # noqa: F401
except Exception:
    client_mod = types.ModuleType('langchain_mcp_adapters.client')

    class _FakeMCPClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def get_tools(self):
            return []

    parent_mod = types.ModuleType('langchain_mcp_adapters')
    parent_mod.client = client_mod
    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters'] = parent_mod
    sys.modules['langchain_mcp_adapters.client'] = client_mod

try:
    import pydantic  # type: ignore  # noqa: F401
except Exception:
    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, description=''):
        del description
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

try:
    import langgraph.prebuilt  # type: ignore  # noqa: F401
    import langgraph.types  # type: ignore  # noqa: F401
    import langgraph.checkpoint.memory  # type: ignore  # noqa: F401
except Exception:
    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in training mainline final-state smoke')

    class _Command:
        def __init__(self, resume=None):
            self.resume = resume

    class _InMemorySaver:
        def __init__(self, *args, **kwargs):
            self.storage = {}
            self.writes = {}
            self.blobs = {}

    prebuilt_mod.create_react_agent = _fake_create_react_agent
    types_mod.Command = _Command
    checkpoint_mod.InMemorySaver = _InMemorySaver
    sys.modules['langgraph.prebuilt'] = prebuilt_mod
    sys.modules['langgraph.types'] = types_mod
    sys.modules['langgraph.checkpoint.memory'] = checkpoint_mod

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.client.mainline_route_support import resolve_mainline_dispatch_payload
from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_from_draft,
    build_training_plan_context_payload,
)
from yolostudio_agent.agent.client.training_request_service import (
    run_prepare_only_flow,
    run_training_request_entrypoint,
)
from langchain_core.messages import AIMessage, ToolMessage


class _FakeGraph:
    def __init__(self, *, summary_payload: dict[str, Any], analysis_payload: dict[str, Any], recommendation_payload: dict[str, Any]) -> None:
        self.summary_payload = dict(summary_payload)
        self.analysis_payload = dict(analysis_payload)
        self.recommendation_payload = dict(recommendation_payload)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.client = None
        self.plan_context: dict[str, Any] | None = None

    def bind(self, client) -> None:
        self.client = client

    def get_state(self, config):
        del config
        client = getattr(self, 'client', None)
        if client is not None:
            draft = dict(client.session_state.active_training.training_plan_draft or {})
            if draft:
                self.plan_context = build_training_plan_context_from_draft(draft)
            has_draft = bool(draft)
            has_pending = bool(client.session_state.pending_confirmation.tool_name)
            if not has_draft and not has_pending:
                self.plan_context = None
        if not self.plan_context:
            return None
        return types.SimpleNamespace(values={'training_plan_context': dict(self.plan_context)})

    async def ainvoke(self, payload, config=None):
        messages = list(payload['messages'])
        client = getattr(self, 'client', None)
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break

        plan_context = dict(payload.get('training_plan_context') or self.plan_context or {})
        if not plan_context and client is not None:
            mainline_context = client._collect_mainline_context(user_text)
            route_state = await client._resolve_mainline_route_state(user_text, mainline_context)
            dispatch_payload = resolve_mainline_dispatch_payload(
                mainline_context=mainline_context,
                route_state=route_state,
            )
            training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
            if training_entrypoint_args:
                prepare_only_followup = await run_prepare_only_flow(
                    user_text=user_text,
                    looks_like_prepare_only_request=client._looks_like_prepare_only_request,
                    extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
                    local_path_exists=lambda path: Path(path).expanduser().exists(),
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
                else:
                    entrypoint_result = await run_training_request_entrypoint(
                        session_state=client.session_state,
                        user_text=user_text,
                        normalized_text=str(training_entrypoint_args.get('normalized_text') or ''),
                        dataset_path=str(training_entrypoint_args.get('dataset_path') or ''),
                        frame_followup_path=str(training_entrypoint_args.get('frame_followup_path') or ''),
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
                        wants_split=bool(training_entrypoint_args.get('wants_split')),
                        current_training_plan_context=build_training_plan_context_payload(client.session_state),
                        direct_tool=client.direct_tool,
                        collect_requested_training_args=client._collect_requested_training_args,
                        is_training_discussion_only=client._is_training_discussion_only,
                        extract_training_execution_backend=client._extract_training_execution_backend_from_text,
                        build_training_plan_draft_fn=client._build_training_plan_draft,
                        render_training_plan_message=client._render_training_plan_message,
                    )
                draft = dict((entrypoint_result or {}).get('draft') or {})
                reply = str((entrypoint_result or {}).get('reply') or '').strip()
                discussion_only = client._is_training_discussion_only(user_text)
                if draft:
                    client.session_state.active_training.training_plan_draft = dict(draft)
                    self.plan_context = build_training_plan_context_from_draft(draft)
                if draft and discussion_only:
                    rendered = reply or await client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + ([AIMessage(content=rendered)] if rendered else [])}
                if bool((entrypoint_result or {}).get('defer_to_graph')) and draft:
                    plan_context = dict(self.plan_context or {})
                elif reply:
                    return {'messages': messages + [AIMessage(content=reply)]}
                elif draft:
                    rendered = await client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + [AIMessage(content=rendered)]}

        next_tool = str(plan_context.get('next_step_tool') or '').strip()
        next_args = dict(plan_context.get('next_step_args') or {})
        is_execute_turn = any(
            token in user_text
            for token in ('执行', '开始吧', '就这样', '确认', '可以开始', '开训', '启动吧', '直接训练', '直接开始训练')
        ) or str(user_text).strip().lower() in {'y', 'yes'}
        if client is not None and config and next_tool and is_execute_turn:
            thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
            client._set_pending_confirmation(
                thread_id,
                {'name': next_tool, 'args': next_args, 'id': None, 'synthetic': True},
            )
            return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}

        if any(token in user_text for token in ('训练停了吗', '训练结束了吗', '训练完成了吗', '训练跑完了吗', '训练状态', '当前状态', '第几轮')):
            assert client is not None
            self.calls.append(('check_training_status', {}))
            result = await client.direct_tool('check_training_status')
            reply = await client._render_tool_result_message('check_training_status', result)
            if not reply:
                reply = str(result.get('summary') or result.get('error') or '操作已完成')
            tool_call_id = f'call-{len(self.calls)}'
            return {
                'messages': messages + [
                    AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': 'check_training_status', 'args': {}}]),
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), name='check_training_status', tool_call_id=tool_call_id),
                    AIMessage(content=reply),
                ]
            }
        if '这次训练效果怎么样' in user_text:
            tool_plan = [
                ('summarize_training_run', self.summary_payload),
                ('analyze_training_outcome', self.analysis_payload),
            ]
            final_text = (
                f"{self.summary_payload['summary']}\n\n"
                f"{self.analysis_payload['summary']}"
            )
        elif '下一步先补数据还是调参数' in user_text:
            tool_plan = [
                ('summarize_training_run', self.summary_payload),
                ('recommend_next_training_step', self.recommendation_payload),
            ]
            final_text = (
                f"{self.summary_payload['summary']}\n\n"
                f"{self.recommendation_payload['summary']}\n"
                f"建议动作: {self.recommendation_payload['recommended_action']}"
            )
        else:
            raise AssertionError(f'unexpected graph prompt: {user_text}')

        tool_messages: list[Any] = []
        for tool_name, result in tool_plan:
            self.calls.append((tool_name, {}))
            tool_call_id = f'call-{len(self.calls)}'
            tool_messages.extend(
                [
                    AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': tool_name, 'args': {}}]),
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), name=tool_name, tool_call_id=tool_call_id),
                ]
            )
        return {'messages': messages + tool_messages + [AIMessage(content=final_text)]}


WORK = Path(__file__).resolve().parent / '_tmp_training_mainline_final_state_roundtrip'


def _build_status_payload(run_state: str, metrics: dict[str, Any], *, summary: str, facts: list[str], signals: list[str]) -> dict[str, Any]:
    return {
        'ok': True,
        'summary': summary,
        'running': False,
        'run_state': run_state,
        'observation_stage': 'final',
        'progress': {'epoch': 30, 'total_epochs': 30, 'progress_ratio': 1.0},
        'latest_metrics': {'ok': True, 'metrics': {'epoch': 30, 'total_epochs': 30, **metrics}},
        'analysis_ready': True,
        'minimum_facts_ready': True,
        'signals': list(signals),
        'facts': list(facts),
        'next_actions': ['可继续调用 summarize_training_run 查看最终训练事实'],
    }


async def _run_final_state_scenario(*, session_id: str, status_query: str, final_run_state: str, status_summary: str, training_summary: str, analysis_summary: str, recommendation_summary: str, recommended_action: str, metrics: dict[str, Any], facts: list[str], signals: list[str]) -> None:
    scenario_root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(scenario_root))
    status_payload = _build_status_payload(final_run_state, metrics, summary=status_summary, facts=facts, signals=signals)
    summary_payload = {
        'ok': True,
        'summary': training_summary,
        'run_state': final_run_state,
        'model_family': 'yolo',
        'task_type': 'detection',
        'metrics': dict(metrics),
        'signals': list(signals),
        'facts': list(facts),
        'next_actions': ['可继续调用 analyze_training_outcome 解释训练效果', '如需下一步动作建议，可调用 recommend_next_training_step'],
        'analysis_ready': True,
        'minimum_facts_ready': True,
        'observation_stage': 'final',
        'progress': {'epoch': 30, 'total_epochs': 30, 'progress_ratio': 1.0},
    }
    analysis_payload = {
        'ok': True,
        'summary': analysis_summary,
        'assessment': recommended_action,
        'interpretation': '当前已是最终状态，可以基于验证指标做判断。',
        'recommendation': recommendation_summary,
        'matched_rule_ids': ['generic_training_final_observation'],
        'signals': list(signals),
        'facts': list(facts),
        'next_actions': ['依据最终指标决定是补数据还是调参数'],
        'source_summary': {'official': 2, 'workflow': 1},
    }
    recommendation_payload = {
        'ok': True,
        'summary': recommendation_summary,
        'recommended_action': recommended_action,
        'basis': list(facts),
        'why': '当前已经拿到最终状态和可分析指标，可以做下一步动作判断。',
        'matched_rule_ids': ['generic_training_final_observation'],
        'signals': list(signals),
        'next_actions': ['按照建议处理当前结果'],
        'source_summary': {'official': 2, 'workflow': 1},
    }
    graph = _FakeGraph(
        summary_payload=summary_payload,
        analysis_payload=analysis_payload,
        recommendation_payload=recommendation_payload,
    )
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    graph.bind(client)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：当前数据已具备直接训练条件。',
                'dataset_root': '/data/final-ready',
                'resolved_img_dir': '/data/final-ready/images',
                'resolved_label_dir': '/data/final-ready/labels',
                'resolved_data_yaml': '/data/final-ready/data.yaml',
                'ready': True,
                'preparable': False,
                'warnings': [],
                'blockers': [],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
                'environments': [
                    {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
                    {'name': 'base', 'display_name': 'base', 'selected_by_default': False},
                ],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：当前可直接启动训练',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'training_environment': 'yolodo',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动: model=yolov8n.pt, data=/data/final-ready/data.yaml, device=auto',
                'device': kwargs.get('device', 'auto') or 'auto',
                'pid': 9999,
                'log_file': '/runs/train_final_state.txt',
                'started_at': 456.7,
                'resolved_args': dict(kwargs),
            }
        elif tool_name == 'check_training_status':
            result = dict(status_payload)
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        client._record_secondary_event(tool_name, result)
        if tool_name == 'start_training' and result.get('ok'):
            client.session_state.active_training.training_plan_draft = {}
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/final-ready，用 yolov8n.pt 训练 30轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    assert '执行方式: 直接训练' in turn1['message']
    assert [name for name, _ in calls[:3]] == ['training_readiness', 'list_training_environments', 'training_preflight']

    turn2 = await client.chat('那就执行。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'

    turn3 = await client.confirm(turn2['thread_id'], approved=True)
    assert turn3['status'] == 'completed', turn3
    assert '训练已启动' in turn3['message']

    graph_call_count_before_status = len(graph.calls)
    turn4 = await client.chat(status_query)
    assert turn4['status'] == 'completed', turn4
    assert calls[-1][0] == 'check_training_status'
    assert len(graph.calls) == graph_call_count_before_status, graph.calls
    assert status_summary in turn4['message']
    assert '可继续调用 summarize_training_run 查看最终训练事实' in turn4['message']
    assert client.session_state.active_training.last_status.get('run_state') == final_run_state

    turn5 = await client.chat('这次训练效果怎么样？')
    assert turn5['status'] == 'completed', turn5
    assert graph.calls[-2:] == [('summarize_training_run', {}), ('analyze_training_outcome', {})], graph.calls
    assert '训练结果汇总:' in turn5['message']
    assert '训练结果分析:' in turn5['message']
    assert '不能当成最终结论' not in turn5['message']

    turn6 = await client.chat('下一步先补数据还是调参数？')
    assert turn6['status'] == 'completed', turn6
    assert graph.calls[-2:] == [('summarize_training_run', {}), ('recommend_next_training_step', {})], graph.calls
    assert recommended_action in turn6['message'] or recommendation_summary in turn6['message']
    assert client.session_state.active_training.last_summary.get('run_state') == final_run_state


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _run_final_state_scenario(
            session_id='training-mainline-completed',
            status_query='训练跑完了吗？',
            final_run_state='completed',
            status_summary='训练已完成，当前已拿到最终验证指标。',
            training_summary='训练结果汇总: 当前训练已完成，可基于最终验证指标判断效果。',
            analysis_summary='训练结果分析: 当前 precision 尚可，但 recall 和 mAP 仍偏低，更像数据覆盖和标注质量问题。',
            recommendation_summary='下一步建议: 当前更像数据质量问题，建议先补数据和检查标注，再决定是否调参。',
            recommended_action='fix_data_quality',
            metrics={'precision': 0.781, 'recall': 0.412, 'map50': 0.486, 'map': 0.239},
            facts=['epoch=30/30', 'precision=0.781', 'recall=0.412', 'mAP50=0.486', 'mAP50-95=0.239'],
            signals=['training_completed', 'high_precision_low_recall', 'low_map_overall'],
        )
        await _run_final_state_scenario(
            session_id='training-mainline-stopped',
            status_query='训练停了吗？',
            final_run_state='stopped',
            status_summary='训练已停止，当前保留了停止前的最终可读指标。',
            training_summary='训练结果汇总: 本次训练已停止，但已有可分析的最终验证指标。',
            analysis_summary='训练结果分析: 停止前的最终指标显示 precision 偏低、recall 偏高，更像误检偏多。',
            recommendation_summary='下一步建议: 当前先收紧误检，优先检查类别边界和阈值，再考虑继续训练。',
            recommended_action='inspect_false_positives',
            metrics={'precision': 0.432, 'recall': 0.804, 'map50': 0.471, 'map': 0.251},
            facts=['epoch=30/30', 'precision=0.432', 'recall=0.804', 'mAP50=0.471', 'mAP50-95=0.251'],
            signals=['training_stopped', 'low_precision_high_recall'],
        )
        print('training mainline final-state roundtrip ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
