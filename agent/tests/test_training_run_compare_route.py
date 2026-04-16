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
from langchain_core.messages import AIMessage, ToolMessage


class _FakeGraph:
    def __init__(self) -> None:
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
        if '对比最近两次训练' in user_text or '对比两次训练：train_log_200 和 train_log_100' in user_text:
            tool_name = 'compare_training_runs'
            args: dict[str, Any] = {}
            result = {
                'ok': True,
                'summary': '训练对比完成: train_log_200 相比 train_log_100，precision提升 +0.1000',
                'left_run_id': 'train_log_200',
                'right_run_id': 'train_log_100',
                'highlights': ['precision提升 +0.1000'],
                'metric_deltas': {'precision': {'left': 0.52, 'right': 0.42, 'delta': 0.1}},
                'next_actions': ['可继续调用 inspect_training_run'],
            }
            final_text = '训练对比完成\n对比对象: train_log_200 vs train_log_100'
        elif '最近训练记录有哪些' in user_text:
            tool_name = 'list_training_runs'
            args = {}
            result = {
                'ok': True,
                'summary': '找到 2 条最近训练记录',
                'runs': [
                    {'run_id': 'train_log_200', 'run_state': 'completed', 'observation_stage': 'final', 'progress': {'epoch': 5, 'total_epochs': 5}},
                    {'run_id': 'train_log_100', 'run_state': 'stopped', 'observation_stage': 'final', 'progress': {'epoch': 4, 'total_epochs': 5}},
                ],
                'next_actions': ['可继续调用 compare_training_runs'],
            }
            final_text = '最近训练:\n- train_log_200\n- train_log_100'
        else:
            tool_name = 'inspect_training_run'
            args = {'run_id': 'train_log_200'}
            result = {
                'ok': True,
                'summary': '训练记录详情已就绪',
                'selected_run_id': 'train_log_200',
                'run_state': 'completed',
                'observation_stage': 'final',
                'facts': ['precision=0.520'],
                'next_actions': ['可继续调用 analyze_training_outcome'],
            }
            final_text = '训练记录: train_log_200'
        self.calls.append((tool_name, dict(args)))
        tool_call_id = f'call-{len(self.calls)}'
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': tool_name, 'args': args}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name=tool_name, tool_call_id=tool_call_id),
                AIMessage(content=final_text),
            ]
        }


WORK = Path(__file__).resolve().parent / '_tmp_training_run_compare_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-run-compare-route', memory_root=str(WORK))
        graph = _FakeGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

        latest_prompt = '帮我对比最近两次训练'
        assert await client._try_handle_mainline_intent(latest_prompt, 'thread-compare-latest') is None
        compare_latest = await client.chat(latest_prompt)
        assert compare_latest['status'] == 'completed', compare_latest
        assert graph.calls[-1] == ('compare_training_runs', {})
        assert '对比对象: train_log_200 vs train_log_100' in compare_latest['message']

        explicit_prompt = '对比两次训练：train_log_200 和 train_log_100'
        assert await client._try_handle_mainline_intent(explicit_prompt, 'thread-compare-explicit') is None
        compare_explicit = await client.chat(explicit_prompt)
        assert compare_explicit['status'] == 'completed', compare_explicit
        assert graph.calls[-1] == ('compare_training_runs', {})
        assert '对比对象: train_log_200 vs train_log_100' in compare_explicit['message']

        inspect_prompt = '看看 train_log_200 的详情'
        assert await client._try_handle_mainline_intent(inspect_prompt, 'thread-inspect') is None
        inspect_run = await client.chat(inspect_prompt)
        assert inspect_run['status'] == 'completed', inspect_run
        assert graph.calls[-1] == ('inspect_training_run', {'run_id': 'train_log_200'})
        assert '训练记录: train_log_200' in inspect_run['message']

        list_prompt = '最近训练记录有哪些'
        assert await client._try_handle_mainline_intent(list_prompt, 'thread-list') is None
        list_runs = await client.chat(list_prompt)
        assert list_runs['status'] == 'completed', list_runs
        assert graph.calls[-1] == ('list_training_runs', {}), graph.calls
        assert '最近训练:' in list_runs['message']

        assert client.session_state.active_training.last_run_comparison.get('left_run_id') == 'train_log_200'
        assert client.session_state.active_training.last_run_inspection.get('selected_run_id') == 'train_log_200'
        print('training run compare route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
