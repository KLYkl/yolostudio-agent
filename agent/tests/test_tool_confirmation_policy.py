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
        raise AssertionError('create_react_agent should not be called in tool confirmation policy smoke')

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
from yolostudio_agent.agent.client.hitl_manager import pending_review_config
from yolostudio_agent.agent.client.tool_policy import build_manual_interrupt_nodes


class _DummyGraph:
    def get_state(self, config):
        return None


class _FakeAnnotations:
    def __init__(self, *, readOnlyHint=None, destructiveHint=None, openWorldHint=None):
        self.readOnlyHint = readOnlyHint
        self.destructiveHint = destructiveHint
        self.openWorldHint = openWorldHint


class _FakeTool:
    def __init__(self, name: str, *, metadata=None, annotations=None):
        self.name = name
        self.metadata = dict(metadata or {})
        self.annotations = annotations


WORK = Path(__file__).resolve().parent / '_tmp_tool_confirmation_policy'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='tool-confirmation-policy', memory_root=str(WORK))
        client = YoloStudioAgentClient(
            graph=_DummyGraph(),
            settings=settings,
            tool_registry={
                'upload_assets_to_remote': _FakeTool(
                    'upload_assets_to_remote',
                    metadata={'confirmation_required': True, 'read_only': False, 'destructive': False, 'open_world': True},
                ),
                'list_remote_profiles': _FakeTool(
                    'list_remote_profiles',
                    metadata={'confirmation_required': False, 'read_only': True, 'destructive': False, 'open_world': True},
                ),
                'custom_destructive_tool': _FakeTool(
                    'custom_destructive_tool',
                    annotations=_FakeAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True),
                ),
                'start_training': _FakeTool('start_training'),
            },
        )
        raw_tools = list(client.tool_registry.values())

        assert client._tool_policy('upload_assets_to_remote').confirmation_required is True
        assert client._pending_allowed_decisions('upload_assets_to_remote') == ['approve', 'reject', 'edit', 'clarify']
        upload_review = pending_review_config('upload_assets_to_remote', client._tool_policy('upload_assets_to_remote'))
        assert upload_review['confirmation_required'] is True
        assert upload_review['risk_level'] == 'high'
        assert upload_review['read_only'] is False

        assert client._tool_policy('list_remote_profiles').confirmation_required is False
        assert client._pending_allowed_decisions('list_remote_profiles') == ['approve', 'reject', 'clarify']
        list_review = pending_review_config('list_remote_profiles', client._tool_policy('list_remote_profiles'))
        assert list_review['risk_level'] == 'low'
        assert list_review['read_only'] is True

        assert client._tool_policy('custom_destructive_tool').confirmation_required is True
        destructive_review = pending_review_config('custom_destructive_tool', client._tool_policy('custom_destructive_tool'))
        assert destructive_review['destructive'] is True
        assert destructive_review['risk_level'] == 'high'
        assert destructive_review['confirmation_required'] is True

        assert client._tool_policy('start_training').confirmation_required is True
        fallback_review = pending_review_config('start_training', client._tool_policy('start_training'))
        assert fallback_review['risk_level'] == 'high'

        assert client._tool_policy('remote_training_pipeline').confirmation_required is True
        remote_pipeline_review = pending_review_config('remote_training_pipeline', client._tool_policy('remote_training_pipeline'))
        assert remote_pipeline_review['confirmation_required'] is True
        assert remote_pipeline_review['risk_level'] == 'high'
        assert remote_pipeline_review['open_world'] is True

        interrupt_nodes = build_manual_interrupt_nodes(raw_tools)
        assert interrupt_nodes == [], interrupt_nodes

        print('tool confirmation policy ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
