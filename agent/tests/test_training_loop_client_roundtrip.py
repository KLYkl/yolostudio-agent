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
        raise AssertionError('create_react_agent should not be called in training loop client smoke')

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


WORK = Path(__file__).resolve().parent / '_tmp_training_loop_client_roundtrip'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-loop-client-roundtrip', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'training_readiness':
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：当前数据已具备训练条件。',
                    'dataset_root': '/data/loop',
                    'resolved_img_dir': '/data/loop/images',
                    'resolved_label_dir': '/data/loop/labels',
                    'resolved_data_yaml': '/data/loop/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': [],
                    'blockers': [],
                }
            elif tool_name == 'list_training_environments':
                result = {
                    'ok': True,
                    'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                    'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                    'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                }
            elif tool_name == 'start_training_loop':
                result = {
                    'ok': True,
                    'summary': '环训练已启动：helmet-loop（loop_id=loop-123）',
                    'loop_id': 'loop-123',
                    'loop_name': 'helmet-loop',
                    'status': 'queued',
                    'managed_level': kwargs.get('managed_level', 'conservative_auto'),
                    'boundaries': {
                        'max_rounds': kwargs.get('max_rounds', 5),
                        'target_metric': kwargs.get('target_metric', 'map50'),
                        'target_metric_value': kwargs.get('target_metric_value'),
                    },
                    'next_round_plan': {'round_index': 1, 'change_set': []},
                }
            elif tool_name == 'check_training_loop_status':
                result = {
                    'ok': True,
                    'summary': '第 2 轮训练已完成，准备下一轮',
                    'loop_id': 'loop-123',
                    'loop_name': 'helmet-loop',
                    'status': 'awaiting_review',
                    'current_round_index': 2,
                    'max_rounds': 3,
                    'best_round_index': 2,
                    'best_target_metric': 0.68,
                    'latest_round_card': {
                        'round_index': 2,
                        'status': 'completed',
                        'vs_previous': {'highlights': ['mAP50提升 +0.0300']},
                        'next_plan': {'change_set': [{'field': 'epochs', 'old': 30, 'new': 40}]},
                    },
                }
            elif tool_name == 'pause_training_loop':
                result = {
                    'ok': True,
                    'summary': '已记录暂停请求：当前第 2 轮结束后将停住',
                    'loop_id': 'loop-123',
                    'status': 'awaiting_review',
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')

            client._apply_to_state(tool_name, result, kwargs)
            if tool_name in {'start_training', 'start_training_loop'} and result.get('ok'):
                client._clear_training_plan_draft()
            client._record_secondary_event(tool_name, result)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        turn1 = await client.chat('用 /data/loop 数据集和 yolov8n.pt 开一个全托管环训练，最多 3 轮。')
        assert turn1['status'] == 'needs_confirmation', turn1
        assert turn1['tool_call']['name'] == 'start_training_loop'
        assert turn1['tool_call']['args']['model'] == 'yolov8n.pt'
        assert turn1['tool_call']['args']['data_yaml'] == '/data/loop/data.yaml'
        assert turn1['tool_call']['args']['managed_level'] == 'full_auto'
        assert turn1['tool_call']['args']['max_rounds'] == 3

        turn2 = await client.confirm(turn1['thread_id'], approved=True)
        assert turn2['status'] == 'completed', turn2
        assert '环训练已启动' in turn2['message']
        assert client.session_state.active_training.active_loop_id == 'loop-123'
        assert client.session_state.active_training.active_loop_status == 'queued'

        turn3 = await client.chat('环训练状态怎么样？')
        assert turn3['status'] == 'completed', turn3
        assert calls[-1][0] == 'check_training_loop_status'
        assert '当前最佳轮: 第 2 轮' in turn3['message']
        assert client.session_state.active_training.active_loop_status == 'awaiting_review'

        turn4 = await client.chat('这一轮结束后停住')
        assert turn4['status'] == 'completed', turn4
        assert calls[-1][0] == 'pause_training_loop'
        assert '暂停请求' in turn4['message']

        print('training loop client roundtrip ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
