from __future__ import annotations

import asyncio
import shutil
import sys
import types
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
        raise AssertionError('create_react_agent should not be called in training plan dialogue smoke')

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

from agent_plan.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(__file__).resolve().parent / '_tmp_training_plan_dialogue'


async def _scenario_discussion_then_execute() -> None:
    scenario_root = WORK / 'discussion_then_execute'
    settings = AgentSettings(session_id='training-plan-dialogue-1', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/dataset',
                'resolved_img_dir': '/data/dataset/images',
                'resolved_label_dir': '/data/dataset/labels',
                'resolved_data_yaml': '/data/dataset/data.yaml',
                'ready': True,
                'preparable': False,
                'warnings': ['样本量偏小，建议先小步验证'],
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
                'summary': f"训练预检通过：将使用 yolodo，device={kwargs.get('device', '1')}",
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'batch': kwargs.get('batch'),
                    'imgsz': kwargs.get('imgsz'),
                    'optimizer': kwargs.get('optimizer') or None,
                    'freeze': kwargs.get('freeze'),
                    'resume': kwargs.get('resume'),
                },
                'command_preview': [
                    'yolo', 'train',
                    f"model={kwargs['model']}",
                    f"data={kwargs['data_yaml']}",
                    f"epochs={kwargs['epochs']}",
                    f"device={kwargs.get('device', 'auto') or 'auto'}",
                ] + ([f"batch={kwargs['batch']}"] if kwargs.get('batch') is not None else [])
                  + ([f"imgsz={kwargs['imgsz']}"] if kwargs.get('imgsz') is not None else [])
                  + ([f"optimizer={kwargs['optimizer']}"] if kwargs.get('optimizer') else [])
                  + ([f"freeze={kwargs['freeze']}"] if kwargs.get('freeze') is not None else [])
                  + (['resume=True'] if kwargs.get('resume') else []),
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            assert kwargs['model'] == 'yolov8n.pt'
            assert kwargs['data_yaml'] == '/data/dataset/data.yaml'
            assert kwargs['epochs'] == 30
            assert kwargs['device'] == '1'
            assert kwargs['batch'] == 16
            assert kwargs['imgsz'] == 960
            assert kwargs['optimizer'] == 'AdamW'
            assert kwargs['freeze'] == 8
            assert kwargs['resume'] is True
            result = {
                'ok': True,
                'summary': '训练已启动: model=yolov8n.pt, data=/data/dataset/data.yaml, device=1',
                'device': '1',
                'pid': 4321,
                'log_file': '/runs/train_log_test.txt',
                'started_at': 123.4,
                'resolved_args': dict(kwargs),
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'start_training' and result.get('ok'):
            client._clear_training_plan_draft()
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/dataset，用 yolov8n.pt 训练 30 轮，device 1，先别执行，先给我计划。')
    assert turn1['status'] == 'completed', turn1
    assert '训练计划草案：' in turn1['message']
    assert '执行方式: 直接训练' in turn1['message']
    assert '训练环境: yolodo' in turn1['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert [name for name, _ in calls[:3]] == ['training_readiness', 'list_training_environments', 'training_preflight']

    turn2 = await client.chat('为什么你觉得可以直接训练？把 batch 改成 16，imgsz 改成 960，optimizer AdamW，freeze 8，resume。')
    assert turn2['status'] == 'completed', turn2
    assert 'batch=16' in turn2['message']
    assert 'imgsz=960' in turn2['message']
    assert '高级参数: optimizer=AdamW, freeze=8, resume=True' in turn2['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['batch'] == 16
    assert calls[-1][1]['imgsz'] == 960
    assert calls[-1][1]['optimizer'] == 'AdamW'
    assert calls[-1][1]['freeze'] == 8
    assert calls[-1][1]['resume'] is True

    turn3 = await client.chat('那就执行。')
    assert turn3['status'] == 'needs_confirmation', turn3
    assert turn3['tool_call']['name'] == 'start_training'
    assert turn3['tool_call']['args']['batch'] == 16
    assert turn3['tool_call']['args']['imgsz'] == 960
    assert turn3['tool_call']['args']['optimizer'] == 'AdamW'

    turn4 = await client.confirm(turn3['thread_id'], approved=True)
    assert turn4['status'] == 'completed', turn4
    assert '训练已启动' in turn4['message']
    assert client.session_state.active_training.training_plan_draft == {}
    assert client.session_state.active_training.batch == 16
    assert client.session_state.active_training.optimizer == 'AdamW'


async def _scenario_prepare_only_revision() -> None:
    scenario_root = WORK / 'prepare_only'
    settings = AgentSettings(session_id='training-plan-dialogue-2', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/raw',
                'resolved_img_dir': '/data/raw/images',
                'resolved_label_dir': '/data/raw/labels',
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
            assert kwargs['dataset_path'] == '/data/raw'
            assert 'force_split' not in kwargs
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/data/raw',
                'img_dir': '/data/raw/images',
                'label_dir': '/data/raw/labels',
                'data_yaml': '/data/raw/data.yaml',
                'steps_completed': [],
                'next_actions': ['如需训练，可继续 start_training'],
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'prepare_dataset_for_training' and result.get('ok'):
            draft = client.session_state.active_training.training_plan_draft or {}
            if str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only':
                client._clear_training_plan_draft()
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/raw，用 yolov8n.pt 训练 40轮，按默认划分，直接开始训练。')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert '执行方式: 先准备再训练' in turn1['message']
    assert '主要阻塞' not in turn1['message'] or 'missing_yaml' in turn1['message']
    assert client.session_state.pending_confirmation.tool_name == 'prepare_dataset_for_training'
    assert client.session_state.pending_confirmation.tool_args.get('force_split') is True

    turn2 = await client.chat('先只做准备，不要自动划分，为什么必须先 prepare？')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert '执行方式: 只做准备，暂不启动训练' in turn2['message']
    assert '当前阻塞:' in turn2['message']
    assert turn2['tool_call']['name'] == 'prepare_dataset_for_training'
    assert 'force_split' not in turn2['tool_call']['args']

    turn3 = await client.confirm(turn2['thread_id'], approved=True)
    assert turn3['status'] == 'completed', turn3
    assert '数据准备完成' in turn3['message']
    assert all(name != 'start_training' for name, _ in calls)
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_prepare_then_replan_and_execute() -> None:
    scenario_root = WORK / 'prepare_then_replan_execute'
    settings = AgentSettings(session_id='training-plan-dialogue-3', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []
    prepared = {'value': False}

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            if prepared['value']:
                result = {
                    'ok': True,
                    'summary': '训练前检查完成：准备后的数据已具备训练条件。',
                    'dataset_root': '/data/raw-stage2',
                    'resolved_img_dir': '/data/raw-stage2/images',
                    'resolved_label_dir': '/data/raw-stage2/labels',
                    'resolved_data_yaml': '/data/raw-stage2/data.yaml',
                    'ready': True,
                    'preparable': False,
                    'warnings': ['当前是准备后的数据，建议先做一轮小步验证'],
                    'blockers': [],
                }
            else:
                result = {
                    'ok': True,
                    'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                    'dataset_root': '/data/raw-stage2',
                    'resolved_img_dir': '/data/raw-stage2/images',
                    'resolved_label_dir': '/data/raw-stage2/labels',
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
                'summary': '发现 2 个可用训练环境，默认将使用 base',
                'environments': [
                    {'name': 'base', 'display_name': 'base', 'selected_by_default': True},
                    {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': False},
                ],
                'default_environment': {'name': 'base', 'display_name': 'base'},
            }
        elif tool_name == 'prepare_dataset_for_training':
            prepared['value'] = True
            assert kwargs['dataset_path'] == '/data/raw-stage2'
            assert 'force_split' not in kwargs
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/data/raw-stage2',
                'img_dir': '/data/raw-stage2/images',
                'label_dir': '/data/raw-stage2/labels',
                'data_yaml': '/data/raw-stage2/data.yaml',
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                'next_actions': ['如需训练，可继续 start_training'],
            }
        elif tool_name == 'training_preflight':
            selected_environment = kwargs.get('training_environment') or 'base'
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': f"训练预检通过：将使用 {selected_environment}，device={kwargs.get('device', 'auto') or 'auto'}",
                'training_environment': {'name': selected_environment, 'display_name': selected_environment},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'training_environment': selected_environment,
                    'project': kwargs.get('project') or None,
                    'name': kwargs.get('name') or None,
                    'batch': kwargs.get('batch'),
                    'imgsz': kwargs.get('imgsz'),
                    'fraction': kwargs.get('fraction'),
                    'classes': kwargs.get('classes'),
                    'single_cls': kwargs.get('single_cls'),
                    'optimizer': kwargs.get('optimizer') or None,
                    'freeze': kwargs.get('freeze'),
                    'resume': kwargs.get('resume'),
                    'lr0': kwargs.get('lr0'),
                    'patience': kwargs.get('patience'),
                    'workers': kwargs.get('workers'),
                    'amp': kwargs.get('amp'),
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': ['当前是准备后的数据，建议先做一轮小步验证'],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动: model=yolov8n.pt, data=/data/raw-stage2/data.yaml, device=auto',
                'device': kwargs.get('device', 'auto') or 'auto',
                'pid': 9876,
                'log_file': '/runs/stage2.txt',
                'started_at': 456.7,
                'resolved_args': dict(kwargs),
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'prepare_dataset_for_training' and result.get('ok'):
            draft = client.session_state.active_training.training_plan_draft or {}
            if str(draft.get('execution_mode') or '').strip().lower() == 'prepare_only':
                client._clear_training_plan_draft()
        if tool_name == 'start_training' and result.get('ok'):
            client._clear_training_plan_draft()
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/raw-stage2，用 yolov8n.pt 训练 60轮，先给我计划，不要执行。')
    assert turn1['status'] == 'completed', turn1
    assert '执行方式: 先准备再训练' in turn1['message']
    assert '当前阻塞:' in turn1['message']
    assert '缺少可用的 data_yaml' in turn1['message']

    turn2 = await client.chat('那先只做准备，不要自动划分。')
    assert turn2['status'] == 'completed', turn2
    assert '执行方式: 只做准备，暂不启动训练' in turn2['message']
    assert '下一步动作: prepare_dataset_for_training' in turn2['message']

    turn3 = await client.chat('执行。')
    assert turn3['status'] == 'needs_confirmation', turn3
    assert turn3['tool_call']['name'] == 'prepare_dataset_for_training'
    assert 'force_split' not in turn3['tool_call']['args']

    turn4 = await client.confirm(turn3['thread_id'], approved=True)
    assert turn4['status'] == 'completed', turn4
    assert '数据准备完成' in turn4['message']
    assert prepared['value'] is True
    assert client.session_state.active_training.training_plan_draft == {}

    turn5 = await client.chat('准备完了。数据还用 /data/raw-stage2，模型还是 yolov8n.pt。现在用 yolodo 环境，project /runs/stage2，name exp-stage2，batch 12，imgsz 1280，fraction 0.75，只训练类别 0,2，给我新计划，但先不要执行。')
    assert turn5['status'] == 'completed', turn5
    assert '训练环境: yolodo' in turn5['message']
    assert '输出组织: project=/runs/stage2, name=exp-stage2' in turn5['message']
    assert '高级参数: fraction=0.75, classes=[0, 2]' in turn5['message']
    assert '主要风险:' in turn5['message']
    assert '准备后的数据' in turn5['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['training_environment'] == 'yolodo'
    assert calls[-1][1]['project'] == '/runs/stage2'
    assert calls[-1][1]['name'] == 'exp-stage2'
    assert calls[-1][1]['fraction'] == 0.75
    assert calls[-1][1]['classes'] == [0, 2]

    turn6 = await client.chat('为什么不是默认环境？类别限制先取消，resume 不要，lr0 改成 0.003。')
    assert turn6['status'] == 'completed', turn6
    assert '已从默认环境 base 切换到 yolodo' in turn6['message']
    assert '只训练指定类别' not in turn6['message']
    assert '高级参数:' in turn6['message']
    assert 'fraction=0.75' in turn6['message']
    assert 'resume=False' in turn6['message']
    assert 'lr0=0.003' in turn6['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['training_environment'] == 'yolodo'
    assert calls[-1][1]['classes'] is None
    assert calls[-1][1]['resume'] is False
    assert calls[-1][1]['lr0'] == 0.003

    turn7 = await client.chat('可以执行了。')
    assert turn7['status'] == 'needs_confirmation', turn7
    assert turn7['tool_call']['name'] == 'start_training'
    assert turn7['tool_call']['args']['training_environment'] == 'yolodo'
    assert turn7['tool_call']['args']['project'] == '/runs/stage2'
    assert turn7['tool_call']['args']['name'] == 'exp-stage2'
    assert turn7['tool_call']['args']['fraction'] == 0.75
    assert turn7['tool_call']['args']['classes'] is None
    assert turn7['tool_call']['args']['resume'] is False
    assert turn7['tool_call']['args']['lr0'] == 0.003

    turn8 = await client.confirm(turn7['thread_id'], approved=True)
    assert turn8['status'] == 'completed', turn8
    assert '训练已启动' in turn8['message']
    assert client.session_state.active_training.training_environment == 'yolodo'
    assert client.session_state.active_training.project == '/runs/stage2'
    assert client.session_state.active_training.run_name == 'exp-stage2'
    assert client.session_state.active_training.fraction == 0.75
    assert client.session_state.active_training.classes == []
    assert client.session_state.active_training.resume is False
    assert client.session_state.active_training.lr0 == 0.003
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_cancel_then_replan() -> None:
    scenario_root = WORK / 'cancel_then_replan'
    settings = AgentSettings(session_id='training-plan-dialogue-4', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '训练前检查完成：数据已具备训练条件。',
                'dataset_root': '/data/cancel-case',
                'resolved_img_dir': '/data/cancel-case/images',
                'resolved_label_dir': '/data/cancel-case/labels',
                'resolved_data_yaml': '/data/cancel-case/data.yaml',
                'ready': True,
                'preparable': False,
                'warnings': ['当前是第一轮方案，建议确认输出目录后再开训'],
                'blockers': [],
            }
        elif tool_name == 'list_training_environments':
            result = {
                'ok': True,
                'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
                'environments': [
                    {'name': 'yolodo', 'display_name': 'yolodo', 'selected_by_default': True},
                    {'name': 'base', 'display_name': 'base', 'selected_by_default': False},
                ],
                'default_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            }
        elif tool_name == 'training_preflight':
            selected_environment = kwargs.get('training_environment') or 'yolodo'
            result = {
                'ok': True,
                'ready_to_start': True,
                'summary': f"训练预检通过：将使用 {selected_environment}，device={kwargs.get('device', 'auto') or 'auto'}",
                'training_environment': {'name': selected_environment, 'display_name': selected_environment},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'training_environment': selected_environment,
                    'project': kwargs.get('project') or None,
                    'name': kwargs.get('name') or None,
                    'batch': kwargs.get('batch'),
                    'imgsz': kwargs.get('imgsz'),
                    'fraction': kwargs.get('fraction'),
                    'classes': kwargs.get('classes'),
                    'single_cls': kwargs.get('single_cls'),
                    'optimizer': kwargs.get('optimizer') or None,
                    'freeze': kwargs.get('freeze'),
                    'resume': kwargs.get('resume'),
                    'lr0': kwargs.get('lr0'),
                    'patience': kwargs.get('patience'),
                    'workers': kwargs.get('workers'),
                    'amp': kwargs.get('amp'),
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': ['当前方案尚未执行，确认后会正式启动训练'],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': f"训练已启动: model={kwargs['model']}, data={kwargs['data_yaml']}, device={kwargs.get('device', 'auto') or 'auto'}",
                'device': kwargs.get('device', 'auto') or 'auto',
                'pid': 2468,
                'log_file': '/runs/cancel_case.txt',
                'started_at': 789.0,
                'resolved_args': dict(kwargs),
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'start_training' and result.get('ok'):
            client._clear_training_plan_draft()
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('数据在 /data/cancel-case，用 yolov8n.pt 训练 20轮，batch 8，先给我计划，不要执行。')
    assert turn1['status'] == 'completed', turn1
    assert '训练环境: yolodo' in turn1['message']
    assert 'batch=8' in turn1['message']

    turn2 = await client.chat('执行。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training'
    assert turn2['tool_call']['args']['batch'] == 8

    turn3 = await client.confirm(turn2['thread_id'], approved=False)
    assert turn3['status'] == 'cancelled', turn3
    assert '已取消操作：start_training' in turn3['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft == {}
    assert all(name != 'start_training' for name, _ in calls)

    turn4 = await client.chat('刚才先取消。现在重新来：还是 /data/cancel-case 和 yolov8n.pt，训练 20轮，但改成 base 环境，project /runs/cancel-replan，name exp-retry，batch 24，imgsz 1280，fraction 0.6，先给我计划。')
    assert turn4['status'] == 'completed', turn4
    assert '训练环境: base' in turn4['message']
    assert '输出组织: project=/runs/cancel-replan, name=exp-retry' in turn4['message']
    assert '高级参数: fraction=0.6' in turn4['message']
    assert '已从默认环境 yolodo 切换到 base' in turn4['message']
    assert calls[-1][0] == 'training_preflight'
    assert calls[-1][1]['training_environment'] == 'base'
    assert calls[-1][1]['project'] == '/runs/cancel-replan'
    assert calls[-1][1]['name'] == 'exp-retry'
    assert calls[-1][1]['batch'] == 24
    assert calls[-1][1]['imgsz'] == 1280
    assert calls[-1][1]['fraction'] == 0.6

    turn5 = await client.chat('为什么这次改成 base？')
    assert turn5['status'] == 'completed', turn5
    assert '已从默认环境 yolodo 切换到 base' in turn5['message']
    assert '训练环境: base' in turn5['message']

    turn6 = await client.chat('那就保留这个方案，执行。')
    assert turn6['status'] == 'needs_confirmation', turn6
    assert turn6['tool_call']['name'] == 'start_training'
    assert turn6['tool_call']['args']['training_environment'] == 'base'
    assert turn6['tool_call']['args']['project'] == '/runs/cancel-replan'
    assert turn6['tool_call']['args']['name'] == 'exp-retry'
    assert turn6['tool_call']['args']['batch'] == 24
    assert turn6['tool_call']['args']['imgsz'] == 1280
    assert turn6['tool_call']['args']['fraction'] == 0.6

    turn7 = await client.confirm(turn6['thread_id'], approved=True)
    assert turn7['status'] == 'completed', turn7
    assert '训练已启动' in turn7['message']
    assert client.session_state.active_training.training_environment == 'base'
    assert client.session_state.active_training.project == '/runs/cancel-replan'
    assert client.session_state.active_training.run_name == 'exp-retry'
    assert client.session_state.active_training.batch == 24
    assert client.session_state.active_training.imgsz == 1280
    assert client.session_state.active_training.fraction == 0.6
    assert client.session_state.active_training.training_plan_draft == {}
    assert [name for name, _ in calls].count('start_training') == 1


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_discussion_then_execute()
        await _scenario_prepare_only_revision()
        await _scenario_prepare_then_replan_and_execute()
        await _scenario_cancel_then_replan()
        print('training plan dialogue ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
