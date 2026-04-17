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
        raise AssertionError('create_react_agent should not be called in training plan context tests')

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

from langchain_core.messages import AIMessage

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_payload,
    extract_training_plan_context_from_state,
)

WORK = Path(__file__).resolve().parent / '_tmp_training_plan_context'


def _seed_training_plan_draft(state: SessionState) -> None:
    state.active_training.training_plan_draft = {
        'stage': 'training_plan',
        'status': 'ready_for_confirmation',
        'dataset_path': '/data/demo',
        'execution_mode': 'direct_train',
        'execution_backend': 'standard_yolo',
        'training_environment': 'yolodo',
        'advanced_details_requested': True,
        'reasoning_summary': '当前数据已具备训练条件，确认后即可启动。',
        'data_summary': '训练前检查完成：数据已具备训练条件。',
        'preflight_summary': '训练预检通过',
        'next_step_tool': 'start_training',
        'next_step_args': {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/demo/data.yaml',
            'epochs': 100,
            'device': '0',
        },
        'planned_training_args': {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/demo/data.yaml',
            'epochs': 100,
            'device': '0',
            'training_environment': 'yolodo',
            'project': '/runs/train',
            'name': 'demo-run',
            'batch': 8,
            'imgsz': 960,
        },
        'command_preview': ['yolo', 'train', 'model=yolov8n.pt', 'data=/data/demo/data.yaml'],
        'blockers': [],
        'warnings': ['样本量偏小，建议先小步验证'],
        'risks': ['样本量偏小'],
    }


class _CaptureGraph:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        self.payloads.append(dict(payload))
        return {
            'messages': list(payload['messages']) + [AIMessage(content='graph ok')],
        }


def _scenario_build_payload() -> None:
    state = SessionState(session_id='training-plan-payload')
    _seed_training_plan_draft(state)
    payload = build_training_plan_context_payload(state)
    assert payload is not None
    assert payload['next_step_tool'] == 'start_training'
    assert payload['planned_training_args']['model'] == 'yolov8n.pt'
    assert payload['planned_training_args']['batch'] == 8
    assert payload['warnings'] == ['样本量偏小，建议先小步验证']
    assert extract_training_plan_context_from_state({'training_plan_context': payload}) == payload


async def _scenario_chat_passes_training_plan_context() -> None:
    root = WORK / 'graph-input'
    graph = _CaptureGraph()
    settings = AgentSettings(session_id='training-plan-context-chat', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    _seed_training_plan_draft(client.session_state)

    result = await client.chat('hello')
    assert result['status'] == 'completed', result
    assert result['message'] == 'graph ok', result
    assert graph.payloads, 'graph was not invoked'
    payload = graph.payloads[0]
    plan_context = payload.get('training_plan_context')
    assert isinstance(plan_context, dict), payload
    assert plan_context.get('next_step_tool') == 'start_training', plan_context
    assert (plan_context.get('planned_training_args') or {}).get('data_yaml') == '/data/demo/data.yaml', plan_context


async def _scenario_execute_turn_defers_to_graph() -> None:
    root = WORK / 'execute-via-graph'
    graph = _CaptureGraph()
    settings = AgentSettings(session_id='training-plan-context-execute', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    _seed_training_plan_draft(client.session_state)

    result = await client.chat('执行。')
    assert result['status'] == 'completed', result
    assert result['message'] == 'graph ok', result
    assert graph.payloads, 'graph was not invoked'
    plan_context = graph.payloads[0].get('training_plan_context') or {}
    assert plan_context.get('next_step_tool') == 'start_training', plan_context
    assert client.session_state.pending_confirmation.tool_name == ''


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        _scenario_build_payload()
        await _scenario_chat_passes_training_plan_context()
        await _scenario_execute_turn_defers_to_graph()
        print('training plan context payload ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
