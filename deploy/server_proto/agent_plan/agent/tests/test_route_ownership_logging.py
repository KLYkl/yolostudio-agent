from __future__ import annotations

import asyncio
import json
import shutil
import sys
import types
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _install_fake_test_dependencies() -> None:
    fake_openai = types.ModuleType('langchain_openai')
    fake_ollama = types.ModuleType('langchain_ollama')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.ChatOpenAI = _FakeChatOpenAI
    fake_ollama.ChatOllama = _FakeChatOllama
    sys.modules['langchain_openai'] = fake_openai
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
        def __init__(self, content='', name='', tool_call_id='', status='success'):
            super().__init__(content)
            self.name = name
            self.tool_call_id = tool_call_id
            self.status = status

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

    def _Field(default=None, **kwargs):
        del kwargs
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        return {'args': args, 'kwargs': kwargs}

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


WORK = Path(__file__).resolve().parent / '_tmp_route_ownership_logging'


class _GraphWithHandledToolError:
    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        messages.append(
            AIMessage(
                content='',
                tool_calls=[{'id': 'tc-route-1', 'name': 'list_remote_profiles', 'args': {}}],
            )
        )
        messages.append(
            ToolMessage(
                content=json.dumps({'ok': False, 'error': 'mock failure'}, ensure_ascii=False),
                name='list_remote_profiles',
                tool_call_id='tc-route-1',
            )
        )
        messages.append(AIMessage(content='工具失败已被兜底处理。'))
        return {'messages': messages}


class _FakePlannerResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakePlannerLlm:
    async def ainvoke(self, messages):
        text = '\n'.join(str(getattr(message, 'content', '')) for message in messages)
        if 'list_remote_profiles' in text:
            return _FakePlannerResponse('当前默认可用服务器配置是 lab。')
        return _FakePlannerResponse('')


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    graph_client = YoloStudioAgentClient(
        graph=_GraphWithHandledToolError(),
        settings=AgentSettings(session_id='route-graph', memory_root=str(WORK / 'graph')),
        tool_registry={},
        planner_llm=None,
    )

    async def _never_bypass(user_text: str, thread_id: str):
        del user_text, thread_id
        return None

    graph_client._try_handle_mainline_intent = _never_bypass  # type: ignore[assignment]
    graph_result = await graph_client.chat('你好')
    assert graph_result['status'] == 'completed', graph_result
    graph_routes = graph_client.route_ownership_report()
    graph_route_names = [str(item.get('route') or '') for item in graph_routes]
    assert 'graph-selected-tool' in graph_route_names, graph_routes
    assert 'tool-error-recovery' in graph_route_names, graph_routes

    bypass_client = YoloStudioAgentClient(
        graph=_GraphWithHandledToolError(),
        settings=AgentSettings(session_id='route-bypass', memory_root=str(WORK / 'bypass')),
        tool_registry={},
        planner_llm=None,
    )

    async def _bypass_mainline(user_text: str, thread_id: str):
        del user_text
        return {
            'status': 'completed',
            'message': '旁路已处理',
            'tool_call': {'name': 'scan_dataset', 'args': {'img_dir': '/data/demo'}},
            'thread_id': thread_id,
        }

    bypass_client._try_handle_mainline_intent = _bypass_mainline  # type: ignore[assignment]
    bypass_result = await bypass_client.chat('扫描这个数据集')
    assert bypass_result['status'] == 'completed', bypass_result
    bypass_routes = bypass_client.route_ownership_report()
    assert any(item.get('route') == 'graph-external-bypass' for item in bypass_routes), bypass_routes

    print('route ownership logging ok')


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
