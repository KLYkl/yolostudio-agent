from __future__ import annotations

import asyncio
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
    tool_node_mod = types.ModuleType('langgraph.prebuilt.tool_node')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        return {'args': args, 'kwargs': kwargs}

    class _ToolNode:
        def __init__(self, tools, handle_tool_errors=False):
            self.tools = list(tools)
            self.handle_tool_errors = handle_tool_errors

    class _Command:
        def __init__(self, resume=None):
            self.resume = resume

    class _InMemorySaver:
        def __init__(self, *args, **kwargs):
            self.storage = {}
            self.writes = {}
            self.blobs = {}

    prebuilt_mod.create_react_agent = _fake_create_react_agent
    tool_node_mod.ToolNode = _ToolNode
    types_mod.Command = _Command
    checkpoint_mod.InMemorySaver = _InMemorySaver
    sys.modules['langgraph.prebuilt'] = prebuilt_mod
    sys.modules['langgraph.prebuilt.tool_node'] = tool_node_mod
    sys.modules['langgraph.types'] = types_mod
    sys.modules['langgraph.checkpoint.memory'] = checkpoint_mod


_install_fake_test_dependencies()

import yolostudio_agent.agent.client.agent_client as agent_client


class _FakeMCPClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def get_tools(self):
        return []


class _DummyLlm:
    pass


class _RecordingMemoryStore:
    instances: list['_RecordingMemoryStore'] = []

    def __init__(self, root):
        self.root = Path(root)
        self.events: list[tuple[str, str, dict[str, object]]] = []
        self.saved_states: list[object] = []
        self.__class__.instances.append(self)

    def load_state(self, session_id: str):
        return agent_client.SessionState(session_id=session_id)

    def save_state(self, state) -> None:
        self.saved_states.append(state)

    def append_event(self, session_id: str, event_type: str, payload: dict[str, object]) -> None:
        self.events.append((session_id, event_type, dict(payload)))


async def _run() -> None:
    captured: list[dict[str, object]] = []

    def _fake_create_react_agent(*args, **kwargs):
        captured.append({'args': args, 'kwargs': kwargs})
        return {'graph': 'ok'}

    class _FakeTool:
        def __init__(self, name, metadata=None):
            self.name = name
            self.description = 'fake'
            self.args_schema = None
            self.metadata = metadata or {}

    agent_client.MultiServerMCPClient = _FakeMCPClient  # type: ignore[assignment]
    agent_client.build_mcp_connection_config = lambda url: {'url': url}  # type: ignore[assignment]
    agent_client.build_local_transfer_tools = lambda: [
        _FakeTool('upload_assets_to_remote', {'confirmation_required': True, 'open_world': True}),
        _FakeTool('list_remote_profiles', {'confirmation_required': False, 'read_only': True}),
    ]  # type: ignore[assignment]
    agent_client.adapt_tools_for_chat_model = lambda tools, include_aliases=False: tools  # type: ignore[assignment]
    agent_client.build_llm = lambda *args, **kwargs: _DummyLlm()  # type: ignore[assignment]
    agent_client.create_react_agent = _fake_create_react_agent  # type: ignore[assignment]
    agent_client.MemoryStore = _RecordingMemoryStore  # type: ignore[assignment]

    before_manual = len(_RecordingMemoryStore.instances)

    manual = await agent_client.build_agent_client(
        agent_client.AgentSettings(session_id='interrupt-manual', memory_root='agent/tests/_tmp_interrupt_manual', confirmation_mode='manual')
    )
    assert manual is not None
    manual_tools = captured[-1]['args'][1]  # type: ignore[index]
    manual_kwargs = captured[-1]['kwargs']  # type: ignore[index]
    assert getattr(manual_tools, 'handle_tool_errors', False) is True
    assert 'interrupt_before' not in manual_kwargs, manual_kwargs
    assert manual_kwargs.get('state_schema') is not None, manual_kwargs
    assert getattr(manual, 'checkpointer', None) is manual_kwargs.get('checkpointer')
    assert len(_RecordingMemoryStore.instances) - before_manual == 1, _RecordingMemoryStore.instances

    before_auto = len(_RecordingMemoryStore.instances)
    auto = await agent_client.build_agent_client(
        agent_client.AgentSettings(session_id='interrupt-auto', memory_root='agent/tests/_tmp_interrupt_auto', confirmation_mode='auto')
    )
    assert auto is not None
    auto_tools = captured[-1]['args'][1]  # type: ignore[index]
    auto_kwargs = captured[-1]['kwargs']  # type: ignore[index]
    assert getattr(auto_tools, 'handle_tool_errors', False) is True
    assert 'interrupt_before' not in auto_kwargs, auto_kwargs
    assert auto_kwargs.get('state_schema') is not None, auto_kwargs
    assert getattr(auto, 'checkpointer', None) is auto_kwargs.get('checkpointer')
    assert len(_RecordingMemoryStore.instances) - before_auto == 1, _RecordingMemoryStore.instances

    print('agent build interrupt mode ok')


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
