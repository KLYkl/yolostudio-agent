from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

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
        raise AssertionError('create_react_agent should not be called in confirmation prompt smoke')

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
from langchain_core.messages import HumanMessage


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(__file__).resolve().parent / '_tmp_confirmation_prompt'


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='confirmation-prompt-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})

        client.session_state.active_dataset.dataset_root = '/data/dataset'
        client.session_state.active_dataset.last_readiness = {
            'summary': '当前还不能直接训练，但可以先进入 prepare_dataset_for_training',
            'preparable': True,
            'primary_blocker_type': 'missing_yaml',
        }

        prepare_prompt = client._build_confirmation_prompt({
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/data/dataset', 'force_split': True},
        })
        assert '准备执行：数据准备' in prepare_prompt
        assert '数据集: /data/dataset' in prepare_prompt
        assert '主要阻塞: missing_yaml' in prepare_prompt
        assert '附加安排: 按默认比例划分数据' in prepare_prompt

        client.session_state.active_training.last_preflight = {
            'summary': '训练预检通过：将使用 yolodo，device=1',
            'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            'resolved_args': {
                'model': 'yolov8n.pt',
                'data_yaml': '/data/dataset/data.yaml',
                'epochs': 30,
                'device': '1',
                'project': '/runs/ablation',
                'name': 'exp-blue',
                'batch': 16,
                'imgsz': 960,
                'fraction': 0.5,
                'classes': [1, 3],
                'single_cls': False,
                'optimizer': 'AdamW',
                'freeze': 8,
                'resume': True,
                'lr0': 0.004,
                'patience': 15,
                'workers': 4,
                'amp': False,
            },
            'command_preview': ['yolo', 'train', 'model=yolov8n.pt', 'data=/data/dataset/data.yaml', 'epochs=30', 'device=1', 'batch=16', 'imgsz=960', 'optimizer=AdamW', 'freeze=8', 'resume=True', 'lr0=0.004', 'patience=15', 'workers=4', 'amp=False'],
        }
        client.session_state.active_dataset.last_readiness = {
            'summary': '训练前检查完成：数据已具备训练条件。',
        }
        client.session_state.active_dataset.data_yaml = '/data/dataset/data.yaml'
        client._messages.append(HumanMessage('数据在 /data/dataset，用 yolodo 环境跑 yolov8n.pt 训练，80轮，project /runs/ablation，name exp-blue，fraction 0.5，只训练类别 1,3，关闭 single_cls，batch 16，imgsz 960，device 1，optimizer AdamW，freeze 8，resume，lr0 0.004，patience 15，workers 4，关闭 amp'))

        followup = client._build_followup_training_request()
        assert followup is not None
        assert followup['args']['epochs'] == 80
        assert followup['args']['batch'] == 16
        assert followup['args']['imgsz'] == 960
        assert followup['args']['device'] == '1'
        assert followup['args']['training_environment'] == 'yolodo'
        assert followup['args']['project'] == '/runs/ablation'
        assert followup['args']['name'] == 'exp-blue'
        assert followup['args']['fraction'] == 0.5
        assert followup['args']['classes'] == [1, 3]
        assert followup['args']['single_cls'] is False
        assert followup['args']['optimizer'] == 'AdamW'
        assert followup['args']['freeze'] == 8
        assert followup['args']['resume'] is True
        assert followup['args']['lr0'] == 0.004
        assert followup['args']['patience'] == 15
        assert followup['args']['workers'] == 4
        assert followup['args']['amp'] is False

        start_prompt = client._build_confirmation_prompt({
            'name': 'start_training',
            'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/dataset/data.yaml', 'epochs': 30, 'device': '1', 'batch': 16, 'imgsz': 960},
        })
        assert '准备执行：启动训练' in start_prompt
        assert '数据理解: 训练前检查完成：数据已具备训练条件。' in start_prompt
        assert '训练环境: yolodo' in start_prompt
        assert '初步安排: model=yolov8n.pt, data=/data/dataset/data.yaml, epochs=30, device=1, batch=16, imgsz=960' in start_prompt
        assert '输出组织: project=/runs/ablation, name=exp-blue' in start_prompt
        assert '高级参数: fraction=0.5, classes=[1, 3], single_cls=False' in start_prompt
        assert '预检: 训练预检通过：将使用 yolodo，device=1' in start_prompt

        client.session_state.active_training.training_plan_draft = {
            'dataset_path': '/data/dataset',
            'data_summary': '训练前检查完成：数据已具备训练条件。',
            'execution_mode': 'direct_train',
            'execution_backend': 'standard_yolo',
            'training_environment': 'yolodo',
            'planned_training_args': {
                'model': 'yolov8n.pt',
                'data_yaml': '/data/dataset/data.yaml',
                'epochs': 80,
                'device': '1',
                'project': '/runs/ablation',
                'name': 'exp-blue',
                'batch': 16,
                'imgsz': 960,
                'fraction': 0.5,
                'classes': [1, 3],
                'single_cls': False,
                'optimizer': 'AdamW',
                'freeze': 8,
                'resume': True,
                'lr0': 0.004,
                'patience': 15,
                'workers': 4,
                'amp': False,
            },
            'preflight_summary': '训练预检通过：将使用 yolodo，device=1',
            'next_step_tool': 'start_training',
            'next_step_args': {
                'model': 'yolov8n.pt',
                'data_yaml': '/data/dataset/data.yaml',
                'epochs': 80,
                'device': '1',
                'project': '/runs/ablation',
                'name': 'exp-blue',
                'batch': 16,
                'imgsz': 960,
                'fraction': 0.5,
                'classes': [1, 3],
                'single_cls': False,
                'optimizer': 'AdamW',
                'freeze': 8,
                'resume': True,
                'lr0': 0.004,
                'patience': 15,
                'workers': 4,
                'amp': False,
            },
            'warnings': ['样本量偏小'],
            'advanced_details_requested': True,
        }
        draft_prompt = client._build_confirmation_prompt({
            'name': 'start_training',
            'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/dataset/data.yaml', 'epochs': 80, 'device': '1', 'batch': 16, 'imgsz': 960},
        })
        assert '训练计划草案：' in draft_prompt
        assert '执行方式: 直接训练' in draft_prompt
        assert '输出组织: project=/runs/ablation, name=exp-blue' in draft_prompt
        assert '高级参数: fraction=0.5, classes=[1, 3], single_cls=False, optimizer=AdamW, freeze=8, resume=True, lr0=0.004, patience=15, workers=4, amp=False' in draft_prompt
        assert '你可以直接确认，也可以继续改参数、追问原因、改执行方式。' in draft_prompt

        cancel_prompt = client._build_cancel_message({
            'name': 'start_training',
            'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/dataset/data.yaml'},
        })
        assert '已取消操作：start_training' in cancel_prompt
        assert '调整参数后重新下达指令' in cancel_prompt

        prepare_cancel_prompt = client._build_cancel_message({
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/data/raw-dataset'},
        })
        assert '已取消操作：prepare_dataset_for_training' in prepare_cancel_prompt
        assert '调整参数后重新下达指令' in prepare_cancel_prompt
        print('confirmation prompt smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    main()
