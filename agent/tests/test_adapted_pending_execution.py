from __future__ import annotations

import asyncio
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
        raise AssertionError('create_react_agent should not be called in adapted pending execution tests')

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


class _NoGraph:
    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('graph should not run in adapted pending execution test')


WORK = Path(__file__).resolve().parent / '_tmp_adapted_pending_execution'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    return YoloStudioAgentClient(graph=_NoGraph(), settings=settings, tool_registry={})


async def _scenario_prepare_returns_followup_when_auto_progress_off() -> None:
    client = _make_client('prepare-followup-off')
    pending = {'name': 'prepare_dataset_for_training', 'args': {'dataset_path': '/data/demo'}, 'id': None, 'adapted': True}

    async def _fake_direct_tool(name, **kwargs):
        assert name == 'prepare_dataset_for_training'
        assert kwargs['dataset_path'] == '/data/demo'
        return {'ok': True, 'data_yaml': '/data/demo/data.yaml'}

    async def _fake_render(name, parsed):
        assert name == 'prepare_dataset_for_training'
        assert parsed['ok'] is True
        return '数据准备已完成。'

    async def _fake_followup(*, thread_id, prepare_parsed):
        assert thread_id == 'thread-off'
        assert prepare_parsed['data_yaml'] == '/data/demo/data.yaml'
        return {'status': 'needs_confirmation', 'thread_id': 'thread-off', 'message': '下一步准备启动训练。', 'tool_call': {'name': 'start_training'}}

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    client._render_tool_result_message = _fake_render  # type: ignore[assignment]
    client._handle_post_prepare_confirmation_followup = _fake_followup  # type: ignore[assignment]

    result = await client._execute_adapted_pending_tool(
        thread_id='thread-off',
        pending=pending,
        auto_progress_followups=False,
    )
    assert result['status'] == 'needs_confirmation', result
    assert result['tool_call']['name'] == 'start_training', result


async def _scenario_prepare_auto_progresses_followup_when_enabled() -> None:
    client = _make_client('prepare-followup-on')
    pending = {'name': 'prepare_dataset_for_training', 'args': {'dataset_path': '/data/demo'}, 'id': None, 'adapted': True}
    seen: dict[str, str] = {}

    async def _fake_direct_tool(name, **kwargs):
        assert name == 'prepare_dataset_for_training'
        return {'ok': True, 'data_yaml': '/data/demo/data.yaml'}

    async def _fake_render(name, parsed):
        del name, parsed
        return '数据准备已完成。'

    async def _fake_followup(*, thread_id, prepare_parsed):
        assert thread_id == 'thread-on'
        assert prepare_parsed['data_yaml'] == '/data/demo/data.yaml'
        return {'status': 'needs_confirmation', 'thread_id': 'thread-next', 'message': '下一步准备启动训练。', 'tool_call': {'name': 'start_training'}}

    async def _fake_confirm(thread_id, approved, stream_handler=None):
        del stream_handler
        seen['thread_id'] = thread_id
        seen['approved'] = str(approved)
        return {'status': 'completed', 'message': '已继续启动训练。', 'tool_call': {'name': 'start_training'}, 'approved': True}

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    client._render_tool_result_message = _fake_render  # type: ignore[assignment]
    client._handle_post_prepare_confirmation_followup = _fake_followup  # type: ignore[assignment]
    client.confirm = _fake_confirm  # type: ignore[assignment]

    result = await client._execute_adapted_pending_tool(
        thread_id='thread-on',
        pending=pending,
        auto_progress_followups=True,
    )
    assert result['status'] == 'completed', result
    assert seen == {'thread_id': 'thread-next', 'approved': 'True'}, seen


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_prepare_returns_followup_when_auto_progress_off()
        await _scenario_prepare_auto_progresses_followup_when_enabled()
        print('adapted pending execution ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
