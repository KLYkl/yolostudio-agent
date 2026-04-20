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

    client_mod.MultiServerMCPClient = _FakeMCPClient
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
        raise AssertionError('create_react_agent should not be called in training loop client smoke')

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

import yolostudio_agent.agent.client.agent_client as agent_client_module
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.tests._training_plan_test_support import (
    clear_training_plan_draft,
    current_training_plan_draft,
    set_training_plan_draft,
)
from yolostudio_agent.agent.client.training_plan_context_service import build_training_plan_context_from_draft
from langchain_core.messages import AIMessage, ToolMessage


class _DummyGraph:
    def get_state(self, config):
        return None


class _ScriptedGraph:
    def __init__(self, script) -> None:
        self.client: YoloStudioAgentClient | None = None
        self.script = script
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.plan_context: dict[str, Any] | None = None

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        if not self.plan_context:
            return None
        return types.SimpleNamespace(values={'training_plan_context': dict(self.plan_context)})

    def update_state(self, config, update):
        del config
        if not isinstance(update, dict) or 'training_plan_context' not in update:
            return
        context = update.get('training_plan_context')
        self.plan_context = dict(context) if isinstance(context, dict) and context else None

    async def ainvoke(self, payload, config=None):
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            if 'HumanMessage' not in message.__class__.__name__:
                continue
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        plan_context = dict(payload.get('training_plan_context') or self.plan_context or {})
        next_tool = str(plan_context.get('next_step_tool') or '').strip()
        next_args = dict(plan_context.get('next_step_args') or {})
        execution_mode = str(plan_context.get('execution_mode') or '').strip().lower()
        reasoning_summary = str(plan_context.get('reasoning_summary') or '').strip()
        loop_request_tokens = ('环训练', '循环训练', '循环训', 'loop training', 'training loop', '自动复训', '自动续训')
        loop_control_tokens = ('暂停', '恢复', '终止', '停止', '状态', '详情', '列表', '最近', '查看')
        is_loop_plan_handoff = (
            next_tool in {'prepare_dataset_for_training', 'start_training_loop'}
            and any(token in user_text or token in str(user_text).lower() for token in loop_request_tokens)
            and not any(token in user_text for token in loop_control_tokens)
        )
        is_post_prepare_loop_handoff = (
            next_tool == 'start_training_loop'
            and execution_mode == 'prepare_then_loop'
            and '下一步进入循环训练' in reasoning_summary
        )
        if self.client is not None and config and next_tool and (is_loop_plan_handoff or is_post_prepare_loop_handoff):
            thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
            self.client._set_pending_confirmation(
                thread_id,
                {'name': next_tool, 'args': next_args, 'id': None, 'synthetic': True},
            )
            return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}

        is_loop_start_request = (
            self.client is not None
            and any(token in user_text or token in str(user_text).lower() for token in loop_request_tokens)
            and not any(token in user_text for token in loop_control_tokens)
            and next_tool not in {'prepare_dataset_for_training', 'start_training_loop'}
        )
        if is_loop_start_request:
            dataset_path = agent_client_module.intent_parsing.extract_dataset_path_from_text(user_text)
            loop_args = self.client._collect_requested_training_loop_args(user_text, data_yaml=None)
            entrypoint_result = await self.client._run_training_loop_start_entrypoint(
                user_text=user_text,
                dataset_path=dataset_path,
                loop_args=loop_args,
            )
            draft = dict(entrypoint_result.get('draft') or {})
            reply = str(entrypoint_result.get('reply') or '').strip()
            if draft:
                set_training_plan_draft(self.client, draft)
                self.plan_context = build_training_plan_context_from_draft(draft)
            if entrypoint_result.get('defer_to_graph') and draft and config:
                thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
                next_tool_name = str(draft.get('next_step_tool') or '').strip()
                next_tool_args = dict(draft.get('next_step_args') or {})
                self.client._set_pending_confirmation(
                    thread_id,
                    {'name': next_tool_name, 'args': next_tool_args, 'id': None, 'synthetic': True},
                )
                return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}
            if reply:
                return {'messages': messages + [AIMessage(content=reply)]}

        tool_name, args, result, final_text = self.script(messages)
        self.calls.append((tool_name, dict(args)))
        tool_call_id = f'call-{len(self.calls)}'
        messages.extend(
            [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': tool_name, 'args': args}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name=tool_name, tool_call_id=tool_call_id),
                AIMessage(content=final_text),
            ]
        )
        return {'messages': messages}


WORK = Path(__file__).resolve().parent / '_tmp_training_loop_client_roundtrip'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-loop-client-roundtrip', memory_root=str(WORK))
        def _graph_script(messages):
            user_text = ''
            for message in reversed(messages):
                content = getattr(message, 'content', '')
                if isinstance(content, str) and content:
                    user_text = content
                    break
            if '环训练状态怎么样' in user_text:
                return (
                    'check_training_loop_status',
                    {'loop_id': 'loop-123'},
                    {
                        'ok': True,
                        'summary': '第 2 轮训练已完成，准备下一轮',
                        'loop_id': 'loop-123',
                        'loop_name': 'helmet-loop',
                        'status': 'awaiting_review',
                        'current_round_index': 2,
                        'max_rounds': 3,
                        'best_round_index': 2,
                        'best_target_metric': 0.68,
                        'latest_round_card': {
                            'round_index': 2,
                            'status': 'completed',
                            'vs_previous': {'highlights': ['mAP50提升 +0.0300']},
                            'next_plan': {'change_set': [{'field': 'epochs', 'old': 30, 'new': 40}]},
                        },
                    },
                    '第 2 轮训练已完成，准备下一轮\n当前最佳轮: 第 2 轮',
                )
            return (
                'pause_training_loop',
                {'loop_id': 'loop-123'},
                {
                    'ok': True,
                    'summary': '已记录暂停请求：当前第 2 轮结束后将停住',
                    'loop_id': 'loop-123',
                    'status': 'awaiting_review',
                },
                '已记录暂停请求：当前第 2 轮结束后将停住',
            )

        graph = _ScriptedGraph(_graph_script)
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        graph.bind(client)
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'training_readiness':
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：当前数据已具备训练条件。',
                    'dataset_root': '/data/loop',
                    'resolved_img_dir': '/data/loop/images',
                    'resolved_label_dir': '/data/loop/labels',
                    'resolved_data_yaml': '/data/loop/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [],
                }
            elif tool_name == 'list_training_environments':
                result = {
                    'ok': True,
                    'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                    'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                    'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                }
            elif tool_name == 'start_training_loop':
                result = {
                    'ok': True,
                    'summary': '环训练已启动：helmet-loop（loop_id=loop-123）',
                    'loop_id': 'loop-123',
                    'loop_name': 'helmet-loop',
                    'status': 'queued',
                    'managed_level': kwargs.get('managed_level', 'conservative_auto'),
                    'boundaries': {
                        'max_rounds': kwargs.get('max_rounds', 5),
                        'target_metric': kwargs.get('target_metric', 'map50'),
                        'target_metric_value': kwargs.get('target_metric_value'),
                    },
                    'next_round_plan': {'round_index': 1, 'change_set': []},
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')

            client._apply_to_state(tool_name, result, kwargs)
            if tool_name in {'start_training', 'start_training_loop'} and result.get('ok'):
                clear_training_plan_draft(client)
            client._record_secondary_event(tool_name, result)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        turn1 = await client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个全托管环训练，最多 3 轮。')
        assert turn1['status'] == 'needs_confirmation', turn1
        assert turn1['tool_call']['name'] == 'start_training_loop'
        assert turn1['tool_call']['args']['model'] == 'yolov8n.pt'
        assert turn1['tool_call']['args']['data_yaml'] == '/data/loop/data.yaml'
        assert turn1['tool_call']['args']['managed_level'] == 'full_auto'
        assert turn1['tool_call']['args']['max_rounds'] == 3

        turn2 = await client.confirm(turn1['thread_id'], approved=True)
        assert turn2['status'] == 'completed', turn2
        assert '环训练已启动' in turn2['message']
        assert client.session_state.active_training.active_loop_id == 'loop-123'
        assert client.session_state.active_training.active_loop_status == 'queued'

        assert await client._try_handle_mainline_intent('环训练状态怎么样？', 'thread-loop-status') is None
        turn3 = await client.chat('环训练状态怎么样？')
        assert turn3['status'] == 'completed', turn3
        assert graph.calls[-1][0] == 'check_training_loop_status'
        assert '当前最佳轮: 第 2 轮' in turn3['message']
        assert client.session_state.active_training.active_loop_status == 'awaiting_review'

        assert await client._try_handle_mainline_intent('这一轮结束后停住', 'thread-loop-pause') is None
        turn4 = await client.chat('这一轮结束后停住')
        assert turn4['status'] == 'completed', turn4
        assert graph.calls[-1] == ('pause_training_loop', {'loop_id': 'loop-123'}), graph.calls
        assert '暂停请求' in turn4['message']

        print('training loop client roundtrip ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
