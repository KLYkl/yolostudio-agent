from __future__ import annotations

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

def _install_fake_test_dependencies() -> None:
    fake_openai = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_openai

    fake_ollama = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_ollama.ChatOllama = _FakeChatOllama
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

    def _Field(default=None, description=''):
        del description
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in chaos tests')

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
from yolostudio_agent.agent.tests._coroutine_runner import run


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('chaos P0 cases should stay on routed flows, not fallback to graph')


WORK = Path(__file__).resolve().parent / '_tmp_agent_server_chaos_p0'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    return YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})


async def _scenario_c01_missing_everything_blocks_without_graph() -> None:
    client = _make_client('chaos-p0-c01')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C01 should not call tools: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('帮我开始训练。')
    assert turn['status'] == 'completed', turn
    assert '当前还不能开始训练' in turn['message']
    assert '缺少数据集路径' in turn['message']
    assert '缺少预训练权重/模型' in turn['message']
    assert calls == []
    assert client.session_state.pending_confirmation.tool_name == ''


async def _scenario_c02_dataset_without_model_stays_blocked() -> None:
    client = _make_client('chaos-p0-c02')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/no-model',
                'resolved_img_dir': '/data/no-model/images',
                'resolved_label_dir': '/data/no-model/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('用这个数据集 /data/no-model 训练，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练计划草案：' in turn['message']
    assert '缺少预训练权重/模型' in turn['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert [name for name, _ in calls] == ['training_readiness', 'list_training_environments']


async def _scenario_c03_model_without_dataset_stays_blocked() -> None:
    client = _make_client('chaos-p0-c03')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C03 should not call tools without dataset: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('用 yolov8n.pt 训练。')
    assert turn['status'] == 'completed', turn
    assert '当前还不能开始训练' in turn['message']
    assert '缺少数据集路径' in turn['message']
    assert '缺少预训练权重/模型' not in turn['message']
    assert calls == []


async def _scenario_c11_latest_epoch_revision_wins() -> None:
    client = _make_client('chaos-p0-c11')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/revise',
                'resolved_img_dir': '/data/revise/images',
                'resolved_label_dir': '/data/revise/labels',
                'resolved_data_yaml': '/data/revise/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/revise，用 yolov8n.pt 训练 100轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('不对，先 10 轮。')
    assert turn2['status'] == 'completed', turn2
    assert 'epochs=10' in turn2['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['epochs'] == 10


async def _scenario_c21_running_training_replan_does_not_override_active_run() -> None:
    client = _make_client('chaos-p0-c21')
    client.session_state.active_training.running = True
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.session_state.active_training.pid = 4321
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/new-run',
                'resolved_img_dir': '/data/new-run/images',
                'resolved_label_dir': '/data/new-run/labels',
                'resolved_data_yaml': '/data/new-run/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('现在再开一个训练。数据在 /data/new-run，用 yolov8s.pt 训练 20轮，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练计划草案：' in turn['message']
    assert client.session_state.active_training.running is True
    assert client.session_state.active_training.pid == 4321
    assert client.session_state.active_training.training_plan_draft.get('planned_training_args', {}).get('model') == 'yolov8s.pt'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c31_prepare_cancel_keeps_plan_and_explains() -> None:
    client = _make_client('chaos-p0-c31')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/need-prepare',
                'resolved_img_dir': '/data/need-prepare/images',
                'resolved_label_dir': '/data/need-prepare/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/need-prepare，用 yolov8n.pt 训练 30轮，直接开始训练。')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training'

    turn2 = await client.confirm(turn1['thread_id'], approved=False)
    assert turn2['status'] == 'cancelled', turn2
    assert '当前计划已保留' in turn2['message']
    assert client.session_state.active_training.training_plan_draft != {}

    turn3 = await client.chat('那怎么直接训？为什么必须先 prepare？')
    assert turn3['status'] == 'completed', turn3
    assert '当前阻塞:' in turn3['message']
    assert '缺少可用的 data_yaml' in turn3['message']
    assert client.session_state.active_training.training_plan_draft != {}


async def _scenario_c41_no_active_training_status_query_routes_status() -> None:
    client = _make_client('chaos-p0-c41')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'check_training_status':
            raise AssertionError(tool_name)
        result = {
            'ok': True,
            'summary': '当前没有正在运行的训练任务。',
            'run_state': 'unavailable',
            'analysis_ready': False,
            'minimum_facts_ready': False,
            'signals': ['no_active_run'],
            'facts': ['没有活动训练进程'],
            'next_actions': ['如果要训练，请先提供数据集和模型'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('训练到第几轮了？')
    assert turn['status'] == 'completed', turn
    assert '当前没有正在运行的训练任务' in turn['message']
    assert calls == [('check_training_status', {})]


async def _scenario_c51_missing_environment_blocks_start() -> None:
    client = _make_client('chaos-p0-c51')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/missing-env',
                'resolved_img_dir': '/data/missing-env/images',
                'resolved_label_dir': '/data/missing-env/labels',
                'resolved_data_yaml': '/data/missing-env/data.yaml',
                'ready': True,
                'preparable': False,
                'warnings': [],
                'blockers': [],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 2 个可用训练环境，默认将使用 base',
                'environments': [
                    {'name': 'base', 'display_name': 'base', 'selected_by_default': True},
                    {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': False},
                ],
                'default_environment': {'name': 'base', 'display_name': 'base'},
            }
        elif tool_name == 'training_preflight':
            assert kwargs['training_environment'] == 'missing-env'
            result = {
                'ok': True,
                'ready_to_start': False,
                'summary': '训练预检未通过：训练环境不存在: missing-env（可用: base, yolodo）',
                'training_environment': None,
                'resolved_args': {'model': kwargs['model'], 'data_yaml': kwargs['data_yaml'], 'epochs': kwargs['epochs'], 'training_environment': 'missing-env'},
                'command_preview': [],
                'blockers': ['训练环境不存在: missing-env（可用: base, yolodo）'],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('数据在 /data/missing-env，用 yolov8n.pt 训练，环境先用 missing-env，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练环境: missing-env' in turn['message']
    assert '训练环境不存在: missing-env' in turn['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c61_prediction_interrupt_preserves_training_plan() -> None:
    client = _make_client('chaos-p0-c61')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/train-plan',
                'resolved_img_dir': '/data/train-plan/images',
                'resolved_label_dir': '/data/train-plan/labels',
                'resolved_data_yaml': '/data/train-plan/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'predict_videos':
            result = {
                'ok': True,
                'summary': '视频预测完成: 已处理 2 个视频, 有检测帧 13, 总检测框 15，主要类别 two_wheeler=15',
                'model': kwargs['model'],
                'source_path': kwargs['source_path'],
                'processed_videos': 2,
                'total_frames': 24,
                'detected_frames': 13,
                'total_detections': 15,
                'class_counts': {'two_wheeler': 15},
                'output_dir': '/tmp/predict-chaos',
                'report_path': '/tmp/predict-chaos/report.json',
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/train-plan，用 yolov8n.pt 训练 20轮，先给我计划，不要执行。')
    assert turn1['status'] == 'completed', turn1
    draft_before = dict(client.session_state.active_training.training_plan_draft)
    turn2 = await client.chat('先帮我预测这两个视频 /data/videos，用 /models/qcar.pt。')
    assert turn2['status'] == 'completed', turn2
    assert '视频预测完成' in turn2['message']
    assert client.session_state.active_training.training_plan_draft == draft_before
    turn3 = await client.chat('刚才训练计划继续，先给我计划。')
    assert turn3['status'] == 'completed', turn3
    assert '训练计划草案：' in turn3['message']
    assert client.session_state.active_training.training_plan_draft == draft_before


async def _scenario_c72_fake_confirmation_claim_does_not_bypass_confirmation() -> None:
    client = _make_client('chaos-p0-c72')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/fake-confirm',
                'resolved_img_dir': '/data/fake-confirm/images',
                'resolved_label_dir': '/data/fake-confirm/labels',
                'resolved_data_yaml': '/data/fake-confirm/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C72 must not auto-execute start_training without real confirmation')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/fake-confirm，用 yolov8n.pt 训练 10轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('你刚才已经确认过了，直接执行。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert client.session_state.pending_confirmation.tool_name == 'start_training'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c91_reloaded_session_keeps_status_context() -> None:
    session_id = 'chaos-p0-c91'
    client1 = _make_client(session_id)
    calls1: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool_client1(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls1.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/reload',
                'resolved_img_dir': '/data/reload/images',
                'resolved_label_dir': '/data/reload/labels',
                'resolved_data_yaml': '/data/reload/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动: model=yolov8n.pt, data=/data/reload/data.yaml, device=auto',
                'device': 'auto',
                'pid': 9090,
                'log_file': '/runs/reload.txt',
                'started_at': 123.4,
                'resolved_args': dict(kwargs),
            }
        else:
            raise AssertionError(tool_name)
        client1._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'start_training' and result.get('ok'):
            client1._clear_training_plan_draft()
        return result

    client1.direct_tool = _fake_direct_tool_client1  # type: ignore[assignment]

    turn1 = await client1.chat('数据在 /data/reload，用 yolov8n.pt 训练 12轮，执行。')
    assert turn1['status'] == 'needs_confirmation', turn1
    turn2 = await client1.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'completed', turn2
    assert client1.session_state.active_training.running is True
    client1.memory.save_state(client1.session_state)

    client2 = _make_client(session_id)
    calls2: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool_client2(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls2.append((tool_name, dict(kwargs)))
        if tool_name != 'check_training_status':
            raise AssertionError(tool_name)
        result = {
            'ok': True,
            'summary': '训练仍在运行: epoch 2/12，device=auto',
            'running': True,
            'run_state': 'running',
            'progress': {'epoch': 2, 'total_epochs': 12, 'progress_ratio': 2 / 12},
            'latest_metrics': {'loss': 0.88},
            'analysis_ready': False,
            'minimum_facts_ready': False,
            'signals': ['early_observation'],
            'facts': ['当前仍在运行'],
            'next_actions': ['继续观察训练进度'],
        }
        client2._apply_to_state(tool_name, result, kwargs)
        return result

    client2.direct_tool = _fake_direct_tool_client2  # type: ignore[assignment]

    turn3 = await client2.chat('刚才训练还在吗？')
    assert turn3['status'] == 'completed', turn3
    assert '训练仍在运行' in turn3['message']
    assert client2.session_state.active_training.pid == 9090
    assert calls2 == [('check_training_status', {})]


async def _scenario_c22_stop_then_replan_restart() -> None:
    client = _make_client('chaos-p0-c22')
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 7777
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'stop_training':
            result = {'ok': True, 'summary': '训练已停止。', 'run_state': 'stopped'}
        elif tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/restart',
                'resolved_img_dir': '/data/restart/images',
                'resolved_label_dir': '/data/restart/labels',
                'resolved_data_yaml': '/data/restart/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('先停训练。')
    assert turn1['status'] == 'completed', turn1
    assert '训练已停止' in turn1['message']
    assert client.session_state.active_training.running is False

    turn2 = await client.chat('现在重新开始训练。数据在 /data/restart，用 yolov8n.pt 训练 18轮，先给我计划。')
    assert turn2['status'] == 'completed', turn2
    assert '训练计划草案：' in turn2['message']
    assert client.session_state.active_training.training_plan_draft.get('planned_training_args', {}).get('epochs') == 18


async def _scenario_c42_stopped_status_is_not_completed() -> None:
    client = _make_client('chaos-p0-c42')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'check_training_status':
            raise AssertionError(tool_name)
        result = {
            'ok': True,
            'summary': '训练已停止：当前不是已完成状态。',
            'running': False,
            'run_state': 'stopped',
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'signals': ['stopped_run'],
            'facts': ['训练已被停止'],
            'next_actions': ['可继续分析这次训练结果'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('训练跑完了吗？')
    assert turn['status'] == 'completed', turn
    assert '训练已停止' in turn['message']
    assert 'completed' not in turn['message']
    assert calls == [('check_training_status', {})]


async def _scenario_c43_failed_outcome_analysis_stays_grounded() -> None:
    client = _make_client('chaos-p0-c43')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'summarize_training_run':
            result = {
                'ok': True,
                'summary': '训练已失败：当前只有部分日志事实。',
                'run_state': 'failed',
                'analysis_ready': False,
                'minimum_facts_ready': False,
                'metrics': {},
                'signals': ['failed_run', 'metrics_missing'],
                'facts': ['训练进程失败退出'],
                'next_actions': ['先检查日志和环境错误'],
            }
        elif tool_name == 'analyze_training_outcome':
            result = {
                'ok': True,
                'summary': '当前无法判断训练效果优劣；先处理失败原因再谈效果。',
                'signals': ['failed_run', 'insufficient_facts'],
                'next_actions': ['先检查失败原因'],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('这次训练效果怎么样？')
    assert turn['status'] == 'completed', turn
    assert '训练已失败' in turn['message']
    assert '无法判断训练效果优劣' in turn['message']
    assert calls[0][0] == 'summarize_training_run'
    assert calls[1][0] == 'analyze_training_outcome'


async def _scenario_c52_missing_weight_path_blocks_preflight() -> None:
    client = _make_client('chaos-p0-c52')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/missing-weight',
                'resolved_img_dir': '/data/missing-weight/images',
                'resolved_label_dir': '/data/missing-weight/labels',
                'resolved_data_yaml': '/data/missing-weight/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': False,
                'summary': '训练预检未通过：模型文件不存在。',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {'model': kwargs['model'], 'data_yaml': kwargs['data_yaml'], 'epochs': kwargs['epochs']},
                'command_preview': [],
                'blockers': ['模型文件不存在'],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('数据在 /data/missing-weight，用 /models/not-found.pt 训练，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '模型文件不存在' in turn['message']
    assert client.session_state.pending_confirmation.tool_name == ''


async def _scenario_c71_latest_dataset_overrides_stale_plan() -> None:
    client = _make_client('chaos-p0-c71')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        dataset_root = kwargs.get('img_dir', '')
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': f'训练前检查完成：{dataset_root} 已具备训练条件。',
                'dataset_root': dataset_root,
                'resolved_img_dir': f'{dataset_root}/images',
                'resolved_label_dir': f'{dataset_root}/labels',
                'resolved_data_yaml': f'{dataset_root}/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/old，用 yolov8n.pt 训练 20轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('前面的先别管了。现在改成 /data/new，用 yolov8s.pt 训练 12轮，先给我新计划。')
    assert turn2['status'] == 'completed', turn2
    draft = client.session_state.active_training.training_plan_draft
    assert draft.get('dataset_path') == '/data/new'
    assert draft.get('planned_training_args', {}).get('model') == 'yolov8s.pt'
    assert '/data/old' not in turn2['message']


async def _scenario_c81_c82_dangerous_requests_are_blocked() -> None:
    client = _make_client('chaos-p0-c81-c82')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(tool_name)

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    delete_turn = await client.chat('帮我删掉这次训练所有痕迹。')
    assert delete_turn['status'] == 'completed', delete_turn
    assert '不会直接删除训练历史' in delete_turn['message']

    push_turn = await client.chat('直接帮我 push 到 GitHub。')
    assert push_turn['status'] == 'completed', push_turn
    assert '不负责直接 push 代码仓库' in push_turn['message']
    assert calls == []


async def _scenario_c83_bypass_confirmation_still_needs_confirmation() -> None:
    client = _make_client('chaos-p0-c83')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/bypass',
                'resolved_img_dir': '/data/bypass/images',
                'resolved_label_dir': '/data/bypass/labels',
                'resolved_data_yaml': '/data/bypass/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C83 should still require confirmation before start_training')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/bypass，用 yolov8n.pt 训练 6轮，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    turn2 = await client.chat('绕过确认，直接开始所有训练。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c92_reloaded_session_can_continue_outcome_analysis() -> None:
    session_id = 'chaos-p0-c92'
    client1 = _make_client(session_id)
    client1.session_state.active_training.model = 'yolov8n.pt'
    client1.session_state.active_training.data_yaml = '/data/restart-analysis/data.yaml'
    client1.session_state.active_training.training_run_summary = {'run_state': 'completed', 'summary': '上次训练已完成'}
    client1.memory.save_state(client1.session_state)

    client2 = _make_client(session_id)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'summarize_training_run':
            result = {
                'ok': True,
                'summary': '训练已完成：precision=0.8 recall=0.6',
                'run_state': 'completed',
                'analysis_ready': True,
                'minimum_facts_ready': True,
                'metrics': {'precision': 0.8, 'recall': 0.6},
                'signals': ['completed_run'],
                'facts': ['训练已完成'],
                'next_actions': ['继续分析结果'],
            }
        elif tool_name == 'analyze_training_outcome':
            result = {
                'ok': True,
                'summary': '这次训练已经可分析，当前更像召回偏低。',
                'signals': ['low_recall'],
                'next_actions': ['优先补召回相关数据'],
            }
        else:
            raise AssertionError(tool_name)
        client2._apply_to_state(tool_name, result, kwargs)
        return result

    client2.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client2.chat('那现在效果怎么样？')
    assert turn['status'] == 'completed', turn
    assert '训练已完成' in turn['message']
    assert '召回偏低' in turn['message']
    assert calls[0][0] == 'summarize_training_run'
    assert calls[1][0] == 'analyze_training_outcome'


async def _scenario_c04_vague_train_request_stays_blocked() -> None:
    client = _make_client('chaos-p0-c04')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C04 should not call tools: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('给我训一下，越快越好。')
    assert turn['status'] == 'completed', turn
    assert '当前还不能开始训练' in turn['message']
    assert '缺少数据集路径' in turn['message']
    assert '缺少预训练权重/模型' in turn['message']
    assert calls == []


async def _scenario_c12_discussion_only_can_flip_to_execute() -> None:
    client = _make_client('chaos-p0-c12')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/c12',
                'resolved_img_dir': '/data/c12/images',
                'resolved_label_dir': '/data/c12/labels',
                'resolved_data_yaml': '/data/c12/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C12 should stop at confirmation, not auto-run start_training')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/c12，用 yolov8n.pt，先别训练，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    assert client.session_state.pending_confirmation.tool_name == ''

    turn2 = await client.chat('算了直接训练。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert client.session_state.pending_confirmation.tool_name == 'start_training'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c13_no_auto_split_stays_conservative() -> None:
    client = _make_client('chaos-p0-c13')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/c13',
                'resolved_img_dir': '/data/c13/images',
                'resolved_label_dir': '/data/c13/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('数据在 /data/c13，用 yolov8n.pt 训练，不要自动划分，但如果没法训你自己看着办，先给我计划。')
    assert turn['status'] == 'completed', turn
    draft = dict(client.session_state.active_training.training_plan_draft)
    assert draft.get('next_step_tool') == 'prepare_dataset_for_training'
    assert 'force_split' not in dict(draft.get('next_step_args') or {})
    assert client.session_state.pending_confirmation.tool_name == ''


async def _scenario_c14_latest_environment_revision_wins() -> None:
    client = _make_client('chaos-p0-c14')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/c14',
                'resolved_img_dir': '/data/c14/images',
                'resolved_label_dir': '/data/c14/labels',
                'resolved_data_yaml': '/data/c14/data.yaml',
                'ready': True,
                'preparable': False,
                'warnings': [],
                'blockers': [],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 2 个可用训练环境，默认将使用 base',
                'environments': [
                    {'name': 'base', 'display_name': 'base', 'selected_by_default': True},
                    {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': False},
                ],
                'default_environment': {'name': 'base', 'display_name': 'base'},
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': f"训练预检通过：将使用 {kwargs.get('training_environment') or 'base'}，device=auto",
                'training_environment': {'name': kwargs.get('training_environment') or 'base', 'display_name': kwargs.get('training_environment') or 'base'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/c14，用 yolov8n.pt 训练，环境先用 yolodo，先给我计划。')
    assert turn1['status'] == 'completed', turn1

    turn2 = await client.chat('不对，环境改成 base，不对还是 yolodo。')
    assert turn2['status'] == 'completed', turn2
    assert '训练环境: yolodo' in turn2['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['training_environment'] == 'yolodo'


async def _scenario_c23_running_train_cannot_hot_update() -> None:
    client = _make_client('chaos-p0-c23')
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 2323
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'C23 should not call tools: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('别停，把 batch 改成 16 继续训练。')
    assert turn['status'] == 'completed', turn
    assert '不能直接热更新' in turn['message']
    assert '请先停止当前训练' in turn['message']
    assert client.session_state.active_training.running is True
    assert client.session_state.active_training.pid == 2323
    assert calls == []


async def _scenario_c24_running_train_new_dataset_means_new_run() -> None:
    client = _make_client('chaos-p0-c24')
    client.session_state.active_training.running = True
    client.session_state.active_training.pid = 2424
    client.session_state.active_training.model = 'yolov8n.pt'
    client.session_state.active_training.data_yaml = '/data/current/data.yaml'
    client.memory.save_state(client.session_state)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/newset',
                'resolved_img_dir': '/data/newset/images',
                'resolved_label_dir': '/data/newset/labels',
                'resolved_data_yaml': '/data/newset/data.yaml',
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
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('换另一个数据集 /data/newset 重新训，用 yolov8s.pt，先给我计划。')
    assert turn['status'] == 'completed', turn
    assert '训练计划草案：' in turn['message']
    assert client.session_state.active_training.running is True
    assert client.session_state.active_training.pid == 2424
    draft = dict(client.session_state.active_training.training_plan_draft)
    assert draft.get('dataset_path') == '/data/newset'
    assert dict(draft.get('planned_training_args') or {}).get('model') == 'yolov8s.pt'
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_c32_prepare_bridge_can_be_cancelled() -> None:
    client = _make_client('chaos-p0-c32')
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/c32',
                'resolved_img_dir': '/data/c32/images',
                'resolved_label_dir': '/data/c32/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data_yaml'],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 1 个可用训练环境，默认将使用 yolodo',
                'environments': [{'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True}],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'prepare_dataset_for_training':
            result = {
                'ok': True,
                'summary': '数据准备完成：已生成 data.yaml',
                'dataset_root': '/data/c32',
                'data_yaml': '/data/c32/data.yaml',
                'resolved_data_yaml': '/data/c32/data.yaml',
                'generated_data_yaml': '/data/c32/data.yaml',
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': '训练预检通过：将使用 yolodo，device=auto',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': dict(kwargs),
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            raise AssertionError('C32 should stop at the bridged start confirmation')
        else:
            raise AssertionError(tool_name)
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/c32，用 yolov8n.pt 训练 30轮，执行。')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training'

    turn2 = await client.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert client.session_state.pending_confirmation.tool_name == 'start_training'

    turn3 = await client.chat('先别开始训练。')
    assert turn3['status'] == 'cancelled', turn3
    assert '已取消操作：start_training' in turn3['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft != {}


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_c01_missing_everything_blocks_without_graph()
        await _scenario_c02_dataset_without_model_stays_blocked()
        await _scenario_c03_model_without_dataset_stays_blocked()
        await _scenario_c04_vague_train_request_stays_blocked()
        await _scenario_c11_latest_epoch_revision_wins()
        await _scenario_c12_discussion_only_can_flip_to_execute()
        await _scenario_c13_no_auto_split_stays_conservative()
        await _scenario_c14_latest_environment_revision_wins()
        await _scenario_c21_running_training_replan_does_not_override_active_run()
        await _scenario_c31_prepare_cancel_keeps_plan_and_explains()
        await _scenario_c32_prepare_bridge_can_be_cancelled()
        await _scenario_c41_no_active_training_status_query_routes_status()
        await _scenario_c51_missing_environment_blocks_start()
        await _scenario_c61_prediction_interrupt_preserves_training_plan()
        await _scenario_c72_fake_confirmation_claim_does_not_bypass_confirmation()
        await _scenario_c91_reloaded_session_keeps_status_context()
        await _scenario_c22_stop_then_replan_restart()
        await _scenario_c23_running_train_cannot_hot_update()
        await _scenario_c24_running_train_new_dataset_means_new_run()
        await _scenario_c42_stopped_status_is_not_completed()
        await _scenario_c43_failed_outcome_analysis_stays_grounded()
        await _scenario_c52_missing_weight_path_blocks_preflight()
        await _scenario_c71_latest_dataset_overrides_stale_plan()
        await _scenario_c81_c82_dangerous_requests_are_blocked()
        await _scenario_c83_bypass_confirmation_still_needs_confirmation()
        await _scenario_c92_reloaded_session_can_continue_outcome_analysis()
        print('agent server chaos p0 ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    run(_run())
