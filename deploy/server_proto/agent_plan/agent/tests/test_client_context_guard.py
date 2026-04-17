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


def _install_fake_dependencies() -> None:
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

        async def ainvoke(self, args):
            return args

        def invoke(self, args):
            return args

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
        raise AssertionError('create_react_agent should not be called in context guard tests')

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


_install_fake_dependencies()

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.client.memory_store import MemoryStore
from yolostudio_agent.agent.client.session_state import SessionState


class _NoGraph:
    def get_state(self, config):
        return None


class _CaptureGraph:
    def __init__(self) -> None:
        self.payload = None

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        self.payload = payload
        return {'messages': list(payload.get('messages') or [])}


class _FakeStatusTool:
    name = 'check_training_status'
    description = 'fake status'
    args_schema = None

    async def ainvoke(self, args):
        return {
            'ok': True,
            'running': False,
            'summary': '当前无训练在跑',
            'resolved_args': {
                'model': '/models/yolov8n.pt',
                'data_yaml': '/data/old.yaml',
                'device': '0',
            },
        }


WORK = Path(__file__).resolve().parent / '_tmp_client_context_guard'


async def _scenario_observe_mode_does_not_pollute_state() -> None:
    root = WORK / 'observe-mode'
    settings = AgentSettings(session_id='observe-mode', memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoGraph(), settings=settings, tool_registry={'check_training_status': _FakeStatusTool()})
    result = await client.direct_tool('check_training_status', _state_mode='observe')
    assert result['ok'] is True
    assert client.session_state.active_training.last_status == {}
    assert client.session_state.active_training.data_yaml == ''
    events = client.memory.read_events(client.session_state.session_id)
    assert not any(event.get('type') == 'check_training_status' for event in events), events


async def _scenario_stale_training_plan_draft_is_cleared_on_startup() -> None:
    root = WORK / 'startup-clean'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-clean')
    state.active_training.training_plan_draft = {'execution_mode': 'direct_train', 'model': 'yolov8n.pt'}
    store.save_state(state)

    settings = AgentSettings(session_id='startup-clean', memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoGraph(), settings=settings, tool_registry={})
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_best_weight_path_is_visible_to_graph_handoff() -> None:
    root = WORK / 'best-weight-handoff'
    capture_graph = _CaptureGraph()
    settings = AgentSettings(session_id='best-weight-handoff', memory_root=str(root))
    client = YoloStudioAgentClient(graph=capture_graph, settings=settings, tool_registry={})
    client.session_state.active_training.best_run_selection = {
        'summary': '最近最佳训练为 train_log_best。',
        'best_run': {
            'run_id': 'train_log_best',
            'best_weight_path': '/weights/best.pt',
        },
    }
    client.memory.save_state(client.session_state)
    await client._invoke_graph_from_current_runtime(
        thread_id='best-weight-handoff-turn-1',
        user_text_hint='用最佳训练去预测图片 /data/images。',
    )
    payload = capture_graph.payload or {}
    messages = list(payload.get('messages') or [])
    summary = '\n'.join(
        str(getattr(message, 'content', '') or '')
        for message in messages
    )
    assert 'best_run_id: train_log_best' in summary
    assert 'best_run_weight_path: /weights/best.pt' in summary


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_observe_mode_does_not_pollute_state()
        await _scenario_stale_training_plan_draft_is_cleared_on_startup()
        await _scenario_best_weight_path_is_visible_to_graph_handoff()
        print('client context guard ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
