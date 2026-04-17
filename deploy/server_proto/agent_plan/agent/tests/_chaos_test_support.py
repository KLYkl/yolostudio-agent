from __future__ import annotations

import json
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


def _install_fake_test_dependencies() -> None:
    fake_openai = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_openai

    fake_ollama = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_ollama.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_ollama

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

    client_mod = types.ModuleType('langchain_mcp_adapters.client')

    class _FakeMCPClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def get_tools(self):
            return []

    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters.client'] = client_mod

    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, description='', **kwargs):
        del description, kwargs
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in chaos tests')

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


_install_fake_test_dependencies()

from langchain_core.messages import AIMessage, ToolMessage
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _NoLLMGraph:
    def __init__(self) -> None:
        self.client = None

    def bind(self, client) -> None:
        self.client = client

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break

        plan_context = dict(payload.get('training_plan_context') or {})
        next_tool = str(plan_context.get('next_step_tool') or '').strip()
        next_args = dict(plan_context.get('next_step_args') or {})
        is_execute_turn = any(
            token in user_text
            for token in ('执行', '开始吧', '就这样', '确认', '可以开始', '开训', '启动吧', '直接训练', '直接开始训练')
        ) or str(user_text).strip().lower() in {'y', 'yes'}
        if self.client is not None and config and next_tool and is_execute_turn:
            thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
            self.client._set_pending_confirmation(
                thread_id,
                {'name': next_tool, 'args': next_args, 'id': None, 'synthetic': True},
            )
            return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}
        raise AssertionError('chaos routed cases should not fallback to graph')


class _ScriptedGraph:
    def __init__(self, routes: dict[str, tuple[list[tuple[str, dict[str, Any]]], str]]) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        for marker, (tool_plan, final_text) in self.routes.items():
            if marker in user_text:
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
        raise AssertionError(f'unexpected graph prompt: {user_text}')


WORK = Path(__file__).resolve().parent / '_tmp_agent_server_chaos_p0'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    graph = _NoLLMGraph()
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    graph.bind(client)
    return client
