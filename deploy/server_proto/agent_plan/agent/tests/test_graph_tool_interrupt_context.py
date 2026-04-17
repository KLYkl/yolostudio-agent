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

    tool = agent_client.StructuredTool.from_function(
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
