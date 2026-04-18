from __future__ import annotations

import asyncio
import contextlib
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
    __import__('langchain_openai')
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

try:
    __import__('langchain_ollama')
except Exception:
    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

try:
    __import__('langchain_mcp_adapters.client')
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
    __import__('langchain_core.messages')
except Exception:
    core_mod = sys.modules.get('langchain_core')
    if core_mod is None:
        core_mod = types.ModuleType('langchain_core')
        sys.modules['langchain_core'] = core_mod
    messages_mod = types.ModuleType('langchain_core.messages')

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

    messages_mod.AIMessage = _AIMessage
    messages_mod.BaseMessage = _BaseMessage
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.SystemMessage = _SystemMessage
    messages_mod.ToolMessage = _ToolMessage
    core_mod.messages = messages_mod
    sys.modules['langchain_core.messages'] = messages_mod

try:
    __import__('pydantic')
except Exception:
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

try:
    from langchain_core.tools import StructuredTool
except Exception:
    core_mod = sys.modules.get('langchain_core')
    if core_mod is None:
        core_mod = types.ModuleType('langchain_core')
        sys.modules['langchain_core'] = core_mod
    tools_mod = types.ModuleType('langchain_core.tools')

    class _BaseTool:
        name = 'fake'
        description = 'fake'
        args_schema = None

    class _StructuredTool:
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

    tools_mod.BaseTool = _BaseTool
    tools_mod.StructuredTool = _StructuredTool
    core_mod.tools = tools_mod
    sys.modules['langchain_core.tools'] = tools_mod
    StructuredTool = _StructuredTool

try:
    __import__('langgraph.prebuilt')
    __import__('langgraph.types')
    __import__('langgraph.checkpoint.memory')
except Exception:
    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    tool_node_mod = types.ModuleType('langgraph.prebuilt.tool_node')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')
    errors_mod = types.ModuleType('langgraph.errors')

    def _fake_create_react_agent(*args, **kwargs):
        return {'args': args, 'kwargs': kwargs}

    class _ToolCallRequest:
        def __init__(self, tool_call, tool, state, runtime):
            self.tool_call = tool_call
            self.tool = tool
            self.state = state
            self.runtime = runtime

    class _ToolNode:
        def __init__(self, tools, handle_tool_errors=False, wrap_tool_call=None, awrap_tool_call=None):
            self.tools = list(tools)
            self.tools_by_name = {tool.name: tool for tool in self.tools}
            self._handle_tool_errors = handle_tool_errors
            self._wrap_tool_call = wrap_tool_call
            self._awrap_tool_call = awrap_tool_call

        def _execute_tool_sync(self, request, input_type, config):
            del input_type, config
            args = dict(getattr(request, 'tool_call', {}).get('args') or {})
            if getattr(request.tool, 'func', None) is not None:
                return request.tool.func(**args)
            return None

        async def _execute_tool_async(self, request, input_type, config):
            del input_type, config
            args = dict(getattr(request, 'tool_call', {}).get('args') or {})
            if getattr(request.tool, 'coroutine', None) is not None:
                return await request.tool.coroutine(**args)
            if getattr(request.tool, 'func', None) is not None:
                return request.tool.func(**args)
            return None

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
    tool_node_mod.ToolCallRequest = _ToolCallRequest
    tool_node_mod._handle_tool_error = lambda exc, flag=None: str(exc)
    types_mod.Command = _Command
    checkpoint_mod.InMemorySaver = _InMemorySaver
    sys.modules['langgraph.prebuilt'] = prebuilt_mod
    sys.modules['langgraph.prebuilt.tool_node'] = tool_node_mod
    sys.modules['langgraph.types'] = types_mod
    sys.modules['langgraph.checkpoint.memory'] = checkpoint_mod
    errors_mod.GraphBubbleUp = type('GraphBubbleUp', (Exception,), {})
    sys.modules['langgraph.errors'] = errors_mod

import yolostudio_agent.agent.client.agent_client as agent_client


class _FakeRuntime:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config


class _FakeRequest:
    def __init__(self) -> None:
        self.tool_call = {
            'id': 'call-ctx-1',
            'name': 'convert_format',
            'args': {
                'dataset_path': '/tmp/demo',
                'from_format': 'yolo',
                'to_format': 'voc',
            },
        }
        self.runtime = _FakeRuntime({'configurable': {'thread_id': 'interrupt-ctx-thread'}})


async def _run() -> None:
    def _convert_format(dataset_path: str, from_format: str, to_format: str) -> str:
        del dataset_path, from_format, to_format
        return 'ok'

    tool = StructuredTool.from_function(
        func=_convert_format,
        name='convert_format',
        description='fake',
    )
    tool.metadata = {'confirmation_required': True}
    node = agent_client._build_graph_tool_surface(  # type: ignore[attr-defined]
        [tool],
        confirmation_mode='manual',
        tool_policy_resolver=lambda _: types.SimpleNamespace(confirmation_required=True),
        client_getter=lambda: None,
    )
    request = _FakeRequest()
    observed: dict[str, object] = {'ctx_run_called': False, 'interrupt_called': False}
    context_active = {'value': False}

    @contextlib.contextmanager
    def _fake_set_config_context(config):
        del config

        class _FakeContext:
            def run(self, func, *args, **kwargs):
                observed['ctx_run_called'] = True
                context_active['value'] = True
                try:
                    return func(*args, **kwargs)
                finally:
                    context_active['value'] = False

        yield _FakeContext()

    def _fake_interrupt(payload):
        observed['interrupt_called'] = True
        assert context_active['value'] is True, 'interrupt must run inside context.run(...)'
        assert payload['interrupt_kind'] == 'tool_approval'
        assert payload['tool_name'] == 'convert_format'
        assert payload['thread_id'] == 'interrupt-ctx-thread'
        return 'approve'

    async def _fake_execute(req):
        assert req is request
        return {'status': 'executed'}

    original_set_config_context = agent_client.set_config_context
    original_interrupt = agent_client.interrupt
    try:
        agent_client.set_config_context = _fake_set_config_context  # type: ignore[assignment]
        agent_client.interrupt = _fake_interrupt  # type: ignore[assignment]
        result = await node._awrap_tool_call(request, _fake_execute)
    finally:
        agent_client.set_config_context = original_set_config_context  # type: ignore[assignment]
        agent_client.interrupt = original_interrupt  # type: ignore[assignment]

    assert observed['ctx_run_called'] is True, observed
    assert observed['interrupt_called'] is True, observed
    assert result == {'status': 'executed'}, result
    print('graph tool interrupt context ok')


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
