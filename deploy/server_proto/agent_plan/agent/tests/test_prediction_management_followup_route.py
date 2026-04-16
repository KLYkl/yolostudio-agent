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
        raise AssertionError('create_react_agent should not be called in prediction management followup tests')

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

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from langchain_core.messages import AIMessage


class _GraphState:
    def __init__(self, messages):
        self.values = {'messages': list(messages)}
        self.next = ()


class _PredictionManagementGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None
        self._last_state = _GraphState([])

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        return self._last_state

    async def _cached_reply(self, payload, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        reply = await self.client._render_tool_result_message(tool_name, result)
        if not reply:
            reply = str(result.get('summary') or result.get('error') or '操作已完成')
        messages = list(payload['messages']) + [AIMessage(content=reply)]
        self._last_state = _GraphState(messages)
        return {'messages': messages}

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        summary = '\n'.join(str(getattr(message, 'content', message)) for message in payload['messages'][:2])
        user_text = str(getattr(payload['messages'][-1], 'content', ''))
        pred = self.client.session_state.active_prediction
        if '报告' in user_text:
            assert 'last_export_path:' in summary
            return await self._cached_reply(payload, 'export_prediction_report', dict(pred.last_export))
        if '清单' in user_text:
            assert 'last_path_lists_dir:' in summary
            return await self._cached_reply(payload, 'export_prediction_path_lists', dict(pred.last_path_lists))
        if '整理后' in user_text:
            assert 'last_organized_destination:' in summary
            return await self._cached_reply(payload, 'organize_prediction_results', dict(pred.last_organized_result))
        raise AssertionError(f'unexpected graph request: {user_text}')


class _FakePlannerResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakePlannerLlm:
    def __init__(self, reply) -> None:
        self.reply = reply

    async def ainvoke(self, messages):
        if callable(self.reply):
            return _FakePlannerResponse(self.reply(messages))
        return _FakePlannerResponse(self.reply)


WORK = Path(__file__).resolve().parent / '_tmp_prediction_management_followup_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    graph = _PredictionManagementGraph()
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    graph.bind(client)
    client.session_state.active_prediction.output_dir = '/tmp/predict-out'
    client.session_state.active_prediction.report_path = '/tmp/predict-out/prediction_report.json'
    client.session_state.active_prediction.source_path = '/tmp/images'
    client.session_state.active_prediction.model = 'demo.pt'
    return client


async def _scenario_export_followup_routes_from_state() -> None:
    client = _make_client('export-followup')
    client.session_state.active_prediction.last_export = {
        'ok': True,
        'summary': '预测报告导出完成',
        'export_path': '/tmp/predict-out/prediction_summary.md',
        'export_format': 'markdown',
        'report_path': '/tmp/predict-out/prediction_report.json',
        'output_dir': '/tmp/predict-out',
        'action_candidates': [
            {'description': '继续查看预测输出目录', 'tool': 'inspect_prediction_outputs'},
        ],
    }

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('prediction management followup should render from state, not call direct_tool')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('那个报告再详细一点')
    assert turn['status'] == 'completed', turn
    assert 'prediction_summary.md' in turn['message'], turn


async def _scenario_path_lists_followup_routes_from_state() -> None:
    client = _make_client('path-lists-followup')
    client.session_state.active_prediction.last_path_lists = {
        'ok': True,
        'summary': '预测路径清单导出完成',
        'export_dir': '/tmp/predict-out/path_lists',
        'detected_items_path': '/tmp/predict-out/path_lists/detected_items.txt',
        'empty_items_path': '/tmp/predict-out/path_lists/empty_items.txt',
        'failed_items_path': '/tmp/predict-out/path_lists/failed_items.txt',
        'detected_count': 2,
        'empty_count': 1,
        'failed_count': 0,
    }

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('prediction management followup should render from state, not call direct_tool')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('那个清单再详细一点')
    assert turn['status'] == 'completed', turn
    assert 'path_lists' in turn['message'], turn


async def _scenario_organize_followup_routes_from_state() -> None:
    client = _make_client('organize-followup')
    client.session_state.active_prediction.last_organized_result = {
        'ok': True,
        'summary': '预测结果整理完成',
        'destination_dir': '/tmp/predict-out/organized_by_class',
        'organize_by': 'by_class',
        'copied_items': 2,
        'bucket_stats': {'Excavator': 1, 'Bulldozer': 1},
    }

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('prediction management followup should render from state, not call direct_tool')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('整理后的结果再详细一点')
    assert turn['status'] == 'completed', turn
    assert 'organized_by_class' in turn['message'], turn


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_export_followup_routes_from_state()
        await _scenario_path_lists_followup_routes_from_state()
        await _scenario_organize_followup_routes_from_state()
        print('prediction management followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
