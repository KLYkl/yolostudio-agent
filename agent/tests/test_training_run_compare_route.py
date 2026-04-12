from __future__ import annotations

import asyncio
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
        raise AssertionError('create_react_agent should not be called in training run compare route smoke')

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

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(__file__).resolve().parent / '_tmp_training_run_compare_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-run-compare-route', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'compare_training_runs':
                result = {
                    'ok': True,
                    'summary': '训练对比完成: train_log_200 相比 train_log_100，precision提升 +0.1000',
                    'left_run_id': kwargs.get('left_run_id') or 'train_log_200',
                    'right_run_id': kwargs.get('right_run_id') or 'train_log_100',
                    'highlights': ['precision提升 +0.1000'],
                    'metric_deltas': {'precision': {'left': 0.52, 'right': 0.42, 'delta': 0.1}},
                    'next_actions': ['可继续调用 inspect_training_run'],
                }
            elif tool_name == 'inspect_training_run':
                result = {
                    'ok': True,
                    'summary': '训练记录详情已就绪',
                    'selected_run_id': kwargs.get('run_id') or 'train_log_200',
                    'run_state': 'completed',
                    'observation_stage': 'final',
                    'facts': ['precision=0.520'],
                    'next_actions': ['可继续调用 analyze_training_outcome'],
                }
            elif tool_name == 'list_training_runs':
                result = {
                    'ok': True,
                    'summary': '找到 2 条最近训练记录',
                    'runs': [
                        {'run_id': 'train_log_200', 'run_state': 'completed', 'observation_stage': 'final', 'progress': {'epoch': 5, 'total_epochs': 5}},
                        {'run_id': 'train_log_100', 'run_state': 'stopped', 'observation_stage': 'final', 'progress': {'epoch': 4, 'total_epochs': 5}},
                    ],
                    'next_actions': ['可继续调用 compare_training_runs'],
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')
            client._apply_to_state(tool_name, result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        compare_latest = await client._try_handle_mainline_intent('帮我对比最近两次训练效果', 'thread-compare-latest')
        assert compare_latest is not None
        assert compare_latest['status'] == 'completed', compare_latest
        assert calls[-1] == ('compare_training_runs', {})
        assert '对比对象: train_log_200 vs train_log_100' in compare_latest['message']

        compare_explicit = await client._try_handle_mainline_intent(
            '对比两次训练：train_log_200 和 train_log_100',
            'thread-compare-explicit',
        )
        assert compare_explicit is not None
        assert compare_explicit['status'] == 'completed', compare_explicit
        assert calls[-1] == ('compare_training_runs', {'left_run_id': 'train_log_200', 'right_run_id': 'train_log_100'})

        inspect_run = await client._try_handle_mainline_intent('看看 train_log_200 的详情', 'thread-inspect')
        assert inspect_run is not None
        assert inspect_run['status'] == 'completed', inspect_run
        assert calls[-1] == ('inspect_training_run', {'run_id': 'train_log_200'})
        assert '训练记录: train_log_200' in inspect_run['message']

        list_runs = await client._try_handle_mainline_intent('最近训练记录有哪些', 'thread-list')
        assert list_runs is not None
        assert list_runs['status'] == 'completed', list_runs
        assert calls[-1] == ('list_training_runs', {})
        assert '最近训练:' in list_runs['message']

        assert client.session_state.active_training.last_run_comparison.get('left_run_id') == 'train_log_200'
        assert client.session_state.active_training.last_run_inspection.get('selected_run_id') == 'train_log_200'
        assert client.session_state.active_training.recent_runs[0]['run_id'] == 'train_log_200'
        print('training run compare route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
