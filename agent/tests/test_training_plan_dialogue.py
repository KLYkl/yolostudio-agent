from __future__ import annotations

import asyncio
import json
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
    __import__('langchain_core.messages')
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
    __import__('pydantic')
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
    __import__('langgraph.prebuilt')
    __import__('langgraph.types')
    __import__('langgraph.checkpoint.memory')
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

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from langchain_core.messages import AIMessage, ToolMessage


class _DummyGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        assert self.client is not None
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        plan_context = dict(payload.get('training_plan_context') or {})
        plan_status = str(plan_context.get('status') or '').strip().lower()
        next_tool = str(plan_context.get('next_step_tool') or '').strip()
        next_args = dict(plan_context.get('next_step_args') or {})
        execution_mode = str(plan_context.get('execution_mode') or '').strip().lower()
        reasoning_summary = str(plan_context.get('reasoning_summary') or '').strip()
        preflight_summary = str(plan_context.get('preflight_summary') or '').strip()
        is_post_prepare_start_handoff = (
            next_tool == 'start_training'
            and bool(preflight_summary)
            and '确认后即可启动' in reasoning_summary
        )
        is_execute_turn = (
            _looks_like_training_plan_execute_turn(user_text)
            or execution_mode == 'prepare_only'
            or is_post_prepare_start_handoff
            or (plan_status == 'ready_for_confirmation' and bool(next_tool))
        )
        if config and next_tool and is_execute_turn:
            thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
            self.client._set_pending_confirmation(
                thread_id,
                {'name': next_tool, 'args': next_args, 'id': None, 'synthetic': True},
            )
            return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}
        raise AssertionError(f'unexpected graph prompt: {user_text}')


class _ObservedStatusGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        messages = list(payload['messages'])
        self.calls.append(('check_training_status', {}))
        result = await self.client.direct_tool('check_training_status')
        reply = await self.client._render_tool_result_message('check_training_status', result)
        if not reply:
            reply = str(result.get('summary') or result.get('error') or '操作已完成')
        tool_call_id = f'call-{len(self.calls)}'
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': 'check_training_status', 'args': {}}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name='check_training_status', tool_call_id=tool_call_id),
                AIMessage(content=reply),
            ]
        }


class _TextOnlyTrainingPlanGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        messages = list(payload['messages'])
        return {
            'messages': messages + [
                AIMessage(
                    content=(
                        '根据当前情况，训练前还缺少可用的 data.yaml。'
                        '确认执行吗？如果你同意，我将开始执行数据准备步骤，完成后自动衔接训练。'
                    )
                )
            ]
        }


class _TextOnlyPrepareQuestionGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        messages = list(payload['messages'])
        return {
            'messages': messages + [
                AIMessage(
                    content=(
                        '现在开始准备数据集吗？'
                        '我会先补齐 data.yaml 和划分产物，完成后再继续进入训练。'
                    )
                )
            ]
        }


class _CountingDummyGraph(_DummyGraph):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        self.calls += 1
        return await super().ainvoke(payload, config=config)


class _InterruptEnvelope:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.value = payload


class _InterruptTask:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.interrupts = [_InterruptEnvelope(payload)]


class _InterruptState:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.next = ('training_confirmation',)
        self.tasks = [_InterruptTask(payload)]
        self.values = {'messages': []}


class _GraphPrepareApproveWithStaleStartGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None
        self.calls = 0
        self._state: _InterruptState | None = None

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        return self._state

    async def ainvoke(self, payload, config=None):
        self.calls += 1
        assert self.client is not None
        if getattr(payload, 'resume', None) == {'decision': 'approve'}:
            prepare_result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'img_dir': '/home/kly/ct_loop/data_ct/images',
                'label_dir': '/home/kly/ct_loop/data_ct/labels',
                'data_yaml': '/home/kly/ct_loop/data_ct/images_split(graph)/data.yaml',
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                'next_actions': ['如需训练，可继续 start_training'],
            }
            return {
                'messages': [
                    AIMessage(content='', tool_calls=[{
                        'id': 'prepare-1',
                        'name': 'prepare_dataset_for_training',
                        'args': {'dataset_path': '/home/kly/ct_loop/data_ct'},
                    }]),
                    ToolMessage(
                        content=json.dumps(prepare_result, ensure_ascii=False),
                        name='prepare_dataset_for_training',
                        tool_call_id='prepare-1',
                    ),
                    AIMessage(content='按图继续启动训练。', tool_calls=[{
                        'id': 'start-1',
                        'name': 'start_training',
                        'args': {
                            'model': '/home/kly/yolov8n.pt',
                            'data_yaml': '/home/kly/ct_loop/data_ct/images_split(graph)/data.yaml',
                            'epochs': 100,
                            'training_environment': 'yolodo',
                        },
                    }]),
                ]
            }
        if isinstance(payload, dict):
            training_plan_context = dict(payload.get('training_plan_context') or {})
            thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
            interrupt_payload = self.client._draft_to_training_confirmation_interrupt(
                training_plan_context,
                thread_id=thread_id,
            )
            if interrupt_payload is not None:
                self._state = _InterruptState(dict(interrupt_payload))
                return {'messages': list(payload.get('messages') or [])}
        raise AssertionError(f'unexpected graph payload: {payload!r}')


WORK = Path(__file__).resolve().parent / '_tmp_training_plan_dialogue'


def _looks_like_training_plan_execute_turn(text: str) -> bool:
    normalized = str(text or '').strip().lower()
    if not normalized:
        return False
    if normalized in {'y', 'yes'}:
        return True
    direct_tokens = ('执行', '开始吧', '就这样', '确认', '可以开始', '开训', '启动吧', '直接训练', '直接开始训练')
    retry_tokens = (
        '按原计划重试',
        '重试刚才那次训练',
        '重试上次训练',
        '按刚才的计划再来一次',
        '按原计划再来一次',
        '从最近状态继续训练',
        '从最近状态继续',
        '从最近状态恢复训练',
        '恢复刚才训练',
        '接着上次训练',
        '恢复上次训练',
        'resume 上次训练',
        'resume 刚才训练',
        'resume 另一个 run',
        'resume run',
        '继续另一个 run',
    )
    return any(token in text for token in direct_tokens + retry_tokens) or (
        'resume' in normalized and any(token in text for token in ('上次', '刚才', '最近', '继续', '恢复', '另一个', '历史', 'run'))
    )


def _make_dummy_client(*, session_id: str, memory_root: Path) -> YoloStudioAgentClient:
    graph = _DummyGraph()
    client = YoloStudioAgentClient(
        graph=graph,
        settings=AgentSettings(session_id=session_id, memory_root=str(memory_root)),
        tool_registry={},
    )
    graph.bind(client)
    return client


async def _scenario_discussion_then_execute() -> None:
    scenario_root = WORK / 'discussion_then_execute'
    client = _make_dummy_client(session_id='training-plan-dialogue-1', memory_root=scenario_root)
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


async def _scenario_status_query_without_session_context() -> None:
    scenario_root = WORK / 'status_query_without_session_context'
    settings = AgentSettings(session_id='training-plan-dialogue-status-query', memory_root=str(scenario_root))
    graph = _ObservedStatusGraph()
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    graph.bind(client)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'check_training_status':
            result = {
                'ok': True,
                'running': True,
                'run_state': 'running',
                'summary': '当前训练仍在运行：第 3 轮，device=0',
                'pid': 24680,
                'log_file': '/runs/train_log_live.txt',
                'latest_metrics': {'epoch': 3, 'map50': 0.41},
                'minimum_facts_ready': True,
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    routed = await client._try_handle_mainline_intent('查看训练情况', 'thread-training-plan-dialogue-status')
    assert routed is not None
    assert routed['tool_call']['name'] == 'check_training_status'
    calls.clear()
    graph.calls.clear()
    turn = await client.chat('查看训练情况')
    assert turn['status'] == 'completed', turn
    assert calls == [('check_training_status', {})], calls
    assert graph.calls == [], graph.calls
    assert '第 3 轮' in turn['message'], turn


async def _scenario_prepare_only_revision() -> None:
    scenario_root = WORK / 'prepare_only'
    client = _make_dummy_client(session_id='training-plan-dialogue-2', memory_root=scenario_root)
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
    interrupt_payload = dict(turn2.get('interrupt_payload') or {})
    assert interrupt_payload.get('type') == 'training_confirmation'
    assert interrupt_payload.get('execution_mode') == 'prepare_only'
    assert interrupt_payload.get('next_step_tool') == 'prepare_dataset_for_training'

    turn3 = await client.confirm(turn2['thread_id'], approved=True)
    assert turn3['status'] == 'completed', turn3
    assert '数据准备完成' in turn3['message']
    assert all(name != 'start_training' for name, _ in calls)
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_prepare_only_short_revision_without_dataset_path() -> None:
    scenario_root = WORK / 'prepare_only_short_revision_without_dataset_path'
    client = _make_dummy_client(session_id='training-plan-dialogue-prepare-short-revision', memory_root=scenario_root)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'dataset_training_readiness':
            result = {
                'ok': True,
                'summary': '从数据集角度看，这份数据还不能直接训练: 缺少可用的 data.yaml, 训练/验证集还没准备好；但可以先准备数据',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'dataset_structure': 'yolo_standard',
                'is_split': False,
                'needs_split': True,
                'needs_data_yaml': True,
                'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
                'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data.yaml', '训练/验证集还没准备好'],
                'next_step_summary': '可以先准备数据，补齐 data.yaml 和划分产物。',
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('用 /home/kly/ct_loop/data_ct 按默认比例准备训练数据，先不要开始训练。')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training', turn1
    assert turn1['tool_call']['args']['force_split'] is True, turn1

    turn2 = await client.chat('先不要自动划分，给我计划。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'prepare_dataset_for_training', turn2
    assert 'force_split' not in turn2['tool_call']['args'], turn2
    interrupt_payload = dict(turn2.get('interrupt_payload') or {})
    if interrupt_payload:
        assert interrupt_payload.get('type') == 'training_confirmation', turn2
        assert interrupt_payload.get('phase') == 'prepare', interrupt_payload
        assert interrupt_payload.get('next_step_tool') == 'prepare_dataset_for_training', interrupt_payload
    else:
        assert turn2['pending_action']['decision_context']['decision'] == 'edit', turn2
        assert turn2['pending_action']['decision_context']['raw_user_text'] == '先不要自动划分，给我计划。', turn2
    assert all(name == 'dataset_training_readiness' for name, _ in calls), calls


async def _scenario_prepare_then_replan_and_execute() -> None:
    scenario_root = WORK / 'prepare_then_replan_execute'
    client = _make_dummy_client(session_id='training-plan-dialogue-3', memory_root=scenario_root)
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
    client = _make_dummy_client(session_id='training-plan-dialogue-4', memory_root=scenario_root)
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

    turn3 = await client.chat('不继续。')
    assert turn3['status'] == 'cancelled', turn3
    assert '先不执行这一步' in turn3['message']
    assert '当前计划已保留' in turn3['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft != {}
    assert all(name != 'start_training' for name, _ in calls)

    turn4 = await client.chat('刚才先取消。为什么你默认建议 yolodo？先把环境改成 base，project /runs/cancel-replan，name exp-retry，batch 24，imgsz 1280，fraction 0.6，先给我计划。')
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

    turn7 = await client.chat('好，就按这个来。')
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


async def _scenario_cancel_prepare_then_rebuild() -> None:
    scenario_root = WORK / 'cancel_prepare_then_rebuild'
    client = _make_dummy_client(session_id='training-plan-dialogue-5', memory_root=scenario_root)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/cancel-prepare',
                'resolved_img_dir': '/data/cancel-prepare/images',
                'resolved_label_dir': '/data/cancel-prepare/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': ['原始数据还没有生成训练所需的 data.yaml'],
                'blockers': ['缺少可用的 data_yaml'],
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
        elif tool_name == 'prepare_dataset_for_training':
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/data/cancel-prepare',
                'img_dir': '/data/cancel-prepare/images',
                'label_dir': '/data/cancel-prepare/labels',
                'data_yaml': '/data/cancel-prepare/data.yaml',
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
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

    turn1 = await client.chat('数据在 /data/cancel-prepare，用 yolov8n.pt 训练 30轮，按默认划分，先给我计划，不要执行。')
    assert turn1['status'] == 'completed', turn1
    assert '执行方式: 先准备再训练' in turn1['message']
    assert '当前阻塞:' in turn1['message']
    assert '缺少可用的 data_yaml' in turn1['message']

    turn2 = await client.chat('执行。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'prepare_dataset_for_training'
    assert turn2['tool_call']['args']['dataset_path'] == '/data/cancel-prepare'
    assert turn2['tool_call']['args']['force_split'] is True

    turn3 = await client.confirm(turn2['thread_id'], approved=False)
    assert turn3['status'] == 'cancelled', turn3
    assert '先不执行这一步' in turn3['message']
    assert '当前计划已保留' in turn3['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.active_training.training_plan_draft != {}
    assert all(name != 'prepare_dataset_for_training' for name, _ in calls)

    turn4 = await client.chat('为什么现在不能直接开训？保留刚才的数据和模型，改成先只做准备，不要自动划分，先给我计划。')
    assert turn4['status'] == 'completed', turn4
    assert '执行方式: 只做准备，暂不启动训练' in turn4['message']
    assert '当前阻塞:' in turn4['message']
    assert '当前还不能直接训练' in turn4['message']
    assert '缺少可用的 data_yaml' in turn4['message']
    assert '下一步动作: prepare_dataset_for_training' in turn4['message']

    turn5 = await client.chat('那就按这个只准备的方案执行。')
    assert turn5['status'] == 'needs_confirmation', turn5
    assert turn5['tool_call']['name'] == 'prepare_dataset_for_training'
    assert turn5['tool_call']['args']['dataset_path'] == '/data/cancel-prepare'
    assert 'force_split' not in turn5['tool_call']['args']

    turn6 = await client.confirm(turn5['thread_id'], approved=True)
    assert turn6['status'] == 'completed', turn6
    assert '数据准备完成' in turn6['message']
    assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
    assert client.session_state.active_training.training_plan_draft == {}
    assert all(name != 'start_training' for name, _ in calls)


async def _scenario_preparable_backend_switch() -> None:
    scenario_root = WORK / 'preparable_backend_switch'
    client = _make_dummy_client(session_id='training-plan-dialogue-6', memory_root=scenario_root)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/backend-prepare',
                'resolved_img_dir': '/data/backend-prepare/images',
                'resolved_label_dir': '/data/backend-prepare/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': ['当前还没有训练配置文件，建议先准备再决定是否开训'],
                'blockers': ['缺少可用的 data_yaml'],
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
        elif tool_name == 'prepare_dataset_for_training':
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/data/backend-prepare',
                'img_dir': '/data/backend-prepare/images',
                'label_dir': '/data/backend-prepare/labels',
                'data_yaml': '/data/backend-prepare/data.yaml',
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
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

    turn1 = await client.chat('数据在 /data/backend-prepare，用 yolov8n.pt 训练 25轮，先给我计划，不要执行。')
    assert turn1['status'] == 'completed', turn1
    assert '执行方式: 先准备再训练' in turn1['message']
    assert '当前阻塞:' in turn1['message']
    assert '缺少可用的 data_yaml' in turn1['message']

    turn2 = await client.chat('先别执行，改成自定义 trainer 讨论一下，为什么现在只能先 prepare？')
    assert turn2['status'] == 'completed', turn2
    assert '执行方式: 先讨论方案，暂不执行' in turn2['message']
    assert '执行后端: 自定义 Trainer' in turn2['message']
    assert '当前自动执行链只支持标准 YOLO 训练' in turn2['message']
    assert '当前还不能直接训练' in turn2['message']
    assert client.session_state.pending_confirmation.tool_name == ''
    assert all(name != 'prepare_dataset_for_training' for name, _ in calls)

    turn3 = await client.chat('不用 trainer 了，切回标准 yolo，环境改成 base，只做准备，不要自动划分，先给我计划。')
    assert turn3['status'] == 'completed', turn3
    assert '执行方式: 只做准备，暂不启动训练' in turn3['message']
    assert '执行后端: 标准 YOLO 训练' in turn3['message']
    assert '训练环境: base' in turn3['message']
    assert '下一步动作: prepare_dataset_for_training' in turn3['message']

    turn4 = await client.chat('好，就按这个准备方案执行。')
    assert turn4['status'] == 'needs_confirmation', turn4
    assert turn4['tool_call']['name'] == 'prepare_dataset_for_training'
    assert turn4['tool_call']['args']['dataset_path'] == '/data/backend-prepare'
    assert 'force_split' not in turn4['tool_call']['args']

    turn5 = await client.confirm(turn4['thread_id'], approved=True)
    assert turn5['status'] == 'completed', turn5
    assert '数据准备完成' in turn5['message']
    assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
    assert client.session_state.active_training.training_plan_draft == {}


async def _scenario_prepare_approval_then_revise_start() -> None:
    scenario_root = WORK / 'prepare_approval_then_revise_start'
    client = _make_dummy_client(session_id='training-plan-dialogue-7', memory_root=scenario_root)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/data/prepare-bridge',
                'resolved_img_dir': '/data/prepare-bridge/images',
                'resolved_label_dir': '/data/prepare-bridge/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': ['当前数据还是原始目录，先准备后再做训练预检更稳'],
                'blockers': ['缺少可用的 data_yaml'],
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
        elif tool_name == 'prepare_dataset_for_training':
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/data/prepare-bridge',
                'img_dir': '/data/prepare-bridge/images',
                'label_dir': '/data/prepare-bridge/labels',
                'data_yaml': '/data/prepare-bridge/data.yaml',
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                'next_actions': ['如需训练，可继续 start_training'],
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
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': ['当前是 prepare 后的第一轮训练，建议先做短轮验证'],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动: model=yolov8s.pt, data=/data/prepare-bridge/data.yaml, device=auto',
                'device': kwargs.get('device', 'auto') or 'auto',
                'pid': 6543,
                'log_file': '/runs/prepare_bridge.txt',
                'started_at': 654.3,
                'resolved_args': dict(kwargs),
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'start_training' and result.get('ok'):
            client._clear_training_plan_draft()
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat(
        '数据在 /data/prepare-bridge，用 yolov8s.pt 训练 45轮，环境 yolodo，project /runs/bridge，name exp-bridge，batch 10，imgsz 896，fraction 0.5，只训练类别 1,3。先给我计划，不要执行。'
    )
    assert turn1['status'] == 'completed', turn1
    assert '执行方式: 先准备再训练' in turn1['message']
    assert '训练环境: yolodo' in turn1['message']
    assert '输出组织: project=/runs/bridge, name=exp-bridge' in turn1['message']
    assert '高级参数: fraction=0.5, classes=[1, 3]' in turn1['message']

    turn2 = await client.chat('执行。')
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'prepare_dataset_for_training'
    assert turn2['tool_call']['args']['dataset_path'] == '/data/prepare-bridge'

    turn3 = await client.confirm(turn2['thread_id'], approved=True)
    assert turn3['status'] == 'needs_confirmation', turn3
    assert turn3['tool_call']['name'] == 'start_training'
    assert turn3['tool_call']['args']['model'] == 'yolov8s.pt'
    assert turn3['tool_call']['args']['data_yaml'] == '/data/prepare-bridge/data.yaml'
    assert turn3['tool_call']['args']['epochs'] == 45
    assert turn3['tool_call']['args']['training_environment'] == 'yolodo'
    assert turn3['tool_call']['args']['project'] == '/runs/bridge'
    assert turn3['tool_call']['args']['name'] == 'exp-bridge'
    assert turn3['tool_call']['args']['batch'] == 10
    assert turn3['tool_call']['args']['imgsz'] == 896
    assert turn3['tool_call']['args']['fraction'] == 0.5
    assert turn3['tool_call']['args']['classes'] == [1, 3]
    assert '当前阻塞:' not in turn3['message']
    assert '当前数据已具备训练条件' in turn3['message']
    assert calls[-1][0] == 'training_preflight'

    turn4 = await client.chat('为什么现在可以直接训练了？环境改成 base，batch 12，imgsz 960，fraction 不要了，类别限制取消，先给我计划。')
    assert turn4['status'] == 'needs_confirmation', turn4
    assert turn4['tool_call']['name'] == 'start_training'
    assert turn4['tool_call']['args']['training_environment'] == 'base'
    assert turn4['tool_call']['args']['batch'] == 12
    assert turn4['tool_call']['args']['imgsz'] == 960
    assert turn4['tool_call']['args']['fraction'] is None
    assert turn4['tool_call']['args']['classes'] is None
    interrupt_payload = dict(turn4.get('interrupt_payload') or {})
    assert interrupt_payload.get('type') == 'training_confirmation', turn4
    assert interrupt_payload.get('phase') == 'start', interrupt_payload
    assert interrupt_payload.get('next_step_tool') == 'start_training', interrupt_payload
    assert '环境: base' in turn4['message']

    turn5 = await client.confirm(turn4['thread_id'], approved=True)
    assert turn5['status'] == 'completed', turn5
    assert '训练已启动' in turn5['message']
    assert client.session_state.active_training.training_environment == 'base'
    assert client.session_state.active_training.project == '/runs/bridge'
    assert client.session_state.active_training.run_name == 'exp-bridge'
    assert client.session_state.active_training.batch == 12
    assert client.session_state.active_training.imgsz == 960
    assert client.session_state.active_training.fraction is None
    assert client.session_state.active_training.classes == []
    assert client.session_state.active_training.training_plan_draft == {}
    assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
    assert [name for name, _ in calls].count('training_preflight') == 2
    assert [name for name, _ in calls].count('start_training') == 1


async def _scenario_prepare_only_natural_language_short_circuit() -> None:
    scenario_root = WORK / 'prepare_only_short_circuit'
    client = _make_dummy_client(session_id='training-plan-dialogue-prepare-only-short', memory_root=scenario_root)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'dataset_training_readiness':
            result = {
                'ok': True,
                'summary': '从数据集角度看，这份数据还不能直接训练: 缺少可用的 data.yaml, 训练/验证集还没准备好；但可以先准备数据',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'dataset_structure': 'yolo_standard',
                'is_split': False,
                'needs_split': True,
                'needs_data_yaml': True,
                'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
                'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
                'resolved_data_yaml': '',
                'ready': False,
                'preparable': True,
                'primary_blocker_type': 'missing_yaml',
                'warnings': [],
                'blockers': ['缺少可用的 data.yaml', '训练/验证集还没准备好'],
                'next_step_summary': '可以先准备数据，补齐 data.yaml 和划分产物。',
            }
        elif tool_name == 'prepare_dataset_for_training':
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 已生成 data_yaml',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'data_yaml': '/home/kly/ct_loop/data_ct/data.yaml',
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'prepare_dataset_for_training':
            client._clear_training_plan_draft()
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('用 /home/kly/ct_loop/data_ct 按默认比例准备训练数据，先不要开始训练。')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training', turn1
    assert turn1['tool_call']['args']['dataset_path'] == '/home/kly/ct_loop/data_ct'
    assert turn1['tool_call']['args']['force_split'] is True
    assert '训练计划草案' not in turn1['message'], turn1
    assert '当前还不能直接训练' not in turn1['message'], turn1
    assert '当前没有空闲 GPU' not in turn1['message'], turn1
    assert '缺少预训练权重/模型' not in turn1['message'], turn1
    assert '主要阻塞:' not in turn1['message'], turn1
    assert '预期产物: data.yaml -> /home/kly/ct_loop/data_ct/data.yaml' in turn1['message'], turn1

    turn2 = await client.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'completed', turn2
    assert '数据准备完成' in turn2['message'], turn2
    assert [name for name, _ in calls] == ['dataset_training_readiness', 'prepare_dataset_for_training'], calls


async def _scenario_empty_input_is_ignored() -> None:
    scenario_root = WORK / 'empty_input'
    client = _make_dummy_client(session_id='training-plan-dialogue-empty-input', memory_root=scenario_root)
    turn = await client.chat('')
    assert turn['status'] == 'completed', turn
    assert turn['message'] == '请输入内容。', turn


async def _scenario_prepare_only_invalid_path_is_not_confirmed() -> None:
    scenario_root = WORK / 'prepare_only_invalid_path'
    client = _make_dummy_client(session_id='training-plan-dialogue-invalid-path', memory_root=scenario_root)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'dataset_training_readiness':
            raise AssertionError(f'unexpected tool call: {tool_name}')
        result = {
            'ok': True,
            'readiness_scope': 'dataset',
            'summary': '从数据集角度看，这份数据还不能直接训练: 缺少可用的 data.yaml, 训练/验证集还没准备好；但可以先准备数据',
            'dataset_root': '/home/kly/ct_loop/data按默认比例准备训练数据',
            'dataset_structure': 'unknown',
            'is_split': False,
            'needs_split': True,
            'needs_data_yaml': True,
            'resolved_img_dir': '',
            'resolved_label_dir': '',
            'resolved_data_yaml': '',
            'ready': False,
            'preparable': False,
            'warnings': [],
            'blockers': ['缺少可用的 data.yaml', '训练/验证集还没准备好'],
            'next_step_summary': '可以先准备数据，补齐 data.yaml 和划分产物。',
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('用 /home/kly/ct_loop/data按默认比例准备训练数据')
    assert turn['status'] == 'completed', turn
    assert (
        '我还没核实到可用的数据集结构' in turn['message']
        or '我还没核实到这个路径存在' in turn['message']
    ), turn
    assert all(name != 'prepare_dataset_for_training' for name, _ in calls), calls


async def _scenario_prepare_then_train_prompt_is_not_short_circuited() -> None:
    scenario_root = WORK / 'prepare_then_train_not_short_circuit'
    client = _make_dummy_client(session_id='training-plan-dialogue-prepare-then-train', memory_root=scenario_root)
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
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('请把 /data/raw 按默认比例划分成训练集和验证集，然后用 yolov8n.pt 模型来训练。')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'prepare_dataset_for_training', turn
    assert '训练计划草案：' in turn['message'], turn
    assert '执行方式: 先准备再训练' in turn['message'], turn
    assert '准备执行：数据准备' not in turn['message'], turn
    assert client.session_state.active_training.training_plan_draft.get('execution_mode') == 'prepare_then_train'
    assert [name for name, _ in calls[:2]] == ['training_readiness', 'list_training_environments'], calls


async def _scenario_prepare_then_train_preserves_explicit_classes_txt() -> None:
    scenario_root = WORK / 'prepare_then_train_with_classes_txt'
    client = _make_dummy_client(session_id='training-plan-dialogue-prepare-classes-txt', memory_root=scenario_root)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
                'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
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
            assert kwargs['dataset_path'] == '/home/kly/ct_loop/data_ct'
            assert kwargs['classes_txt'] == '/home/kly/ct_loop/data_ct/classes.txt'
            result = {
                'ok': True,
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'img_dir': '/home/kly/ct_loop/data_ct/images',
                'label_dir': '/home/kly/ct_loop/data_ct/labels',
                'data_yaml': '/home/kly/ct_loop/data_ct/data.yaml',
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                'next_actions': ['如需训练，可继续 start_training'],
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
                    'training_environment': 'yolodo',
                    'batch': kwargs.get('batch'),
                    'imgsz': kwargs.get('imgsz'),
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        elif tool_name == 'start_training':
            result = {
                'ok': True,
                'summary': '训练已启动: model=/home/kly/yolov8n.pt, data=/home/kly/ct_loop/data_ct/data.yaml',
                'device': kwargs.get('device', 'auto') or 'auto',
                'pid': 9876,
                'log_file': '/runs/train_log_prepare_classes_txt.txt',
                'started_at': 987.6,
                'resolved_args': dict(kwargs),
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        if tool_name == 'start_training' and result.get('ok'):
            client._clear_training_plan_draft()
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    prompt = (
        '请把 /home/kly/ct_loop/data_ct 按默认比例划分成训练集和验证集，'
        '类名使用 /home/kly/ct_loop/data_ct/classes.txt，'
        '自动生成可训练的 data.yaml，然后用 /home/kly/yolov8n.pt 模型来训练。'
    )
    turn1 = await client.chat(prompt)
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training', turn1
    assert turn1['tool_call']['args']['dataset_path'] == '/home/kly/ct_loop/data_ct', turn1
    assert turn1['tool_call']['args']['classes_txt'] == '/home/kly/ct_loop/data_ct/classes.txt', turn1
    assert '训练计划草案：' in turn1['message'], turn1
    assert '执行方式: 先准备再训练' in turn1['message'], turn1
    assert '类名来源文件: /home/kly/ct_loop/data_ct/classes.txt' in turn1['message'], turn1

    turn2 = await client.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training', turn2
    assert turn2['tool_call']['args']['model'] == '/home/kly/yolov8n.pt', turn2
    assert turn2['tool_call']['args']['data_yaml'] == '/home/kly/ct_loop/data_ct/data.yaml', turn2
    assert '当前数据已具备训练条件' in turn2['message'], turn2
    assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
    assert [name for name, _ in calls].count('training_preflight') == 1


async def _scenario_pending_prepare_then_train_uses_structured_surface_even_with_planner() -> None:
    scenario_root = WORK / 'prepare_then_train_with_planner_surface'
    client = _make_dummy_client(session_id='training-plan-dialogue-planner-surface', memory_root=scenario_root)
    client.planner_llm = object()

    async def _fake_renderer(**_: Any) -> str:
        return '这是模型整理后的说明。'

    client._invoke_renderer_text = _fake_renderer  # type: ignore[assignment]
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
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')

        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('请把 /data/raw 按默认比例划分成训练集和验证集，然后用 yolov8n.pt 模型来训练。')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'prepare_dataset_for_training', turn
    assert '训练计划草案：' in turn['message'], turn
    assert '执行方式: 先准备再训练' in turn['message'], turn
    assert turn['message'] != '这是模型整理后的说明。', turn
    assert [name for name, _ in calls[:2]] == ['training_readiness', 'list_training_environments'], calls


async def _scenario_prepare_only_with_planner_keeps_natural_surface() -> None:
    scenario_root = WORK / 'prepare_only_with_planner_surface'
    client = _make_dummy_client(session_id='training-plan-dialogue-prepare-only-planner', memory_root=scenario_root)
    client.planner_llm = object()

    async def _fake_renderer(**_: Any) -> str:
        return '我会先按默认比例准备数据，暂时不会开始训练。'

    client._invoke_renderer_text = _fake_renderer  # type: ignore[assignment]
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'dataset_training_readiness':
            raise AssertionError(f'unexpected tool call: {tool_name}')
        result = {
            'ok': True,
            'readiness_scope': 'dataset',
            'summary': '从数据集角度看，这份数据还不能直接训练: 缺少可用的 data.yaml, 训练/验证集还没准备好；但可以先准备数据',
            'dataset_root': '/home/kly/ct_loop/data_ct',
            'dataset_structure': 'image_label_root',
            'is_split': False,
            'needs_split': True,
            'needs_data_yaml': True,
            'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
            'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
            'resolved_data_yaml': '',
            'ready': False,
            'preparable': True,
            'primary_blocker_type': 'missing_yaml',
            'warnings': [],
            'blockers': ['缺少可用的 data.yaml', '训练/验证集还没准备好'],
            'next_step_summary': '可以先准备数据，补齐 data.yaml 和划分产物。',
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn = await client.chat('用 /home/kly/ct_loop/data_ct 按默认比例准备训练数据，先不要开始训练。')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'prepare_dataset_for_training', turn
    assert '训练计划草案' not in turn['message'], turn
    assert turn['message'] == '我会先按默认比例准备数据，暂时不会开始训练。', turn
    assert calls[0][0] == 'dataset_training_readiness', calls


async def _scenario_text_only_training_plan_materializes_pending() -> None:
    scenario_root = WORK / 'text_only_training_plan_materializes_pending'
    graph = _TextOnlyTrainingPlanGraph()
    client = YoloStudioAgentClient(
        graph=graph,
        settings=AgentSettings(session_id='training-plan-dialogue-text-only-materialize', memory_root=str(scenario_root)),
        tool_registry={},
    )
    graph.bind(client)

    turn1 = await client.chat('用 /home/kly/yolov8n.pt 训练一下 /home/kly/ct_loop/data_ct')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training', turn1
    assert '数据准备确认：' in turn1['message'], turn1
    assert '执行方式: 先准备再训练' in turn1['message'], turn1
    draft = dict(client.session_state.active_training.training_plan_draft or {})
    assert draft.get('next_step_tool') == 'prepare_dataset_for_training', draft
    assert draft.get('execution_mode') == 'prepare_then_train', draft
    assert dict(turn1.get('interrupt_payload') or {}).get('type') == 'training_confirmation', turn1

    turn2 = await client.chat('没问题，开始训练')
    assert turn2['status'] == 'completed', turn2
    assert turn2['tool_call'] is None, turn2
    assert '当前缺少预训练权重/模型' in turn2['message'], turn2


async def _scenario_text_only_prepare_question_materializes_pending() -> None:
    scenario_root = WORK / 'text_only_prepare_question_materializes_pending'
    graph = _TextOnlyPrepareQuestionGraph()
    client = YoloStudioAgentClient(
        graph=graph,
        settings=AgentSettings(session_id='training-plan-dialogue-prepare-question-materialize', memory_root=str(scenario_root)),
        tool_registry={},
    )
    graph.bind(client)

    turn = await client.chat('用 /home/kly/yolov8n.pt 训练一下 /home/kly/ct_loop/data_ct')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'prepare_dataset_for_training', turn
    assert '数据准备确认：' in turn['message'], turn
    assert '执行方式: 先准备再训练' in turn['message'], turn
    draft = dict(client.session_state.active_training.training_plan_draft or {})
    assert draft.get('next_step_tool') == 'prepare_dataset_for_training', draft
    assert dict(turn.get('interrupt_payload') or {}).get('type') == 'training_confirmation', turn


async def _scenario_text_only_prepare_question_restore_edit_stays_local() -> None:
    scenario_root = WORK / 'text_only_prepare_question_restore_edit_stays_local'
    seed_graph = _TextOnlyPrepareQuestionGraph()
    seeded = YoloStudioAgentClient(
        graph=seed_graph,
        settings=AgentSettings(session_id='training-plan-dialogue-prepare-question-restore', memory_root=str(scenario_root)),
        tool_registry={},
    )
    seed_graph.bind(seeded)

    turn1 = await seeded.chat('用 /home/kly/yolov8n.pt 训练一下 /home/kly/ct_loop/data_ct')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training', turn1

    restored_graph = _DummyGraph()
    restored = YoloStudioAgentClient(
        graph=restored_graph,
        settings=AgentSettings(session_id='training-plan-dialogue-prepare-question-restore', memory_root=str(scenario_root)),
        tool_registry={},
    )
    restored_graph.bind(restored)
    turn2 = await restored.chat('把 batch 改成 12 再继续')
    draft = dict(restored.session_state.active_training.training_plan_draft or {})
    interrupt_payload = dict(turn2.get('interrupt_payload') or {})
    assert turn2['status'] == 'completed', turn2
    assert interrupt_payload == {}, turn2
    assert 'batch=12' in turn2['message'], turn2
    assert '下一步动作: prepare_dataset_for_training' in turn2['message'], turn2
    assert (draft.get('planned_training_args') or {}).get('batch') == 12, draft


async def _scenario_post_prepare_ready_start_confirmation_stays_local() -> None:
    scenario_root = WORK / 'post_prepare_ready_start_confirmation_stays_local'
    graph = _CountingDummyGraph()
    client = YoloStudioAgentClient(
        graph=graph,
        settings=AgentSettings(session_id='training-plan-dialogue-post-prepare-local-start', memory_root=str(scenario_root)),
        tool_registry={},
    )
    graph.bind(client)
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
                'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
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
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'img_dir': '/home/kly/ct_loop/data_ct/images',
                'label_dir': '/home/kly/ct_loop/data_ct/labels',
                'data_yaml': '/home/kly/ct_loop/data_ct/data.yaml',
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                'next_actions': ['如需训练，可继续 start_training'],
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
                    'training_environment': 'yolodo',
                },
                'command_preview': ['yolo', 'train'],
                'blockers': [],
                'warnings': [],
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('用 /home/kly/yolov8n.pt 训练一下 /home/kly/ct_loop/data_ct')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training', turn1
    graph_calls_after_turn1 = graph.calls

    turn2 = await client.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'needs_confirmation', turn2
    assert turn2['tool_call']['name'] == 'start_training', turn2
    assert turn2['tool_call']['args']['model'] == '/home/kly/yolov8n.pt', turn2
    assert turn2['tool_call']['args']['data_yaml'] == '/home/kly/ct_loop/data_ct/data.yaml', turn2
    assert '当前数据已具备训练条件' in turn2['message'], turn2
    assert graph.calls == graph_calls_after_turn1 + 1, graph.calls
    assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
    assert [name for name, _ in calls].count('training_preflight') == 1


async def _scenario_post_prepare_blocked_followup_uses_grounded_surface() -> None:
    scenario_root = WORK / 'post_prepare_blocked_followup_uses_grounded_surface'
    graph = _CountingDummyGraph()
    client = YoloStudioAgentClient(
        graph=graph,
        settings=AgentSettings(session_id='training-plan-dialogue-post-prepare-blocked', memory_root=str(scenario_root)),
        tool_registry={},
    )
    graph.bind(client)
    client.planner_llm = object()

    async def _unexpected_renderer(**_: Any) -> str:
        raise AssertionError('blocked post-prepare followup should not call llm renderer')

    client._invoke_renderer_text = _unexpected_renderer  # type: ignore[assignment]
    async def _simple_tool_result_message(tool_name: str, parsed: dict[str, Any]) -> str:
        del tool_name
        return str(parsed.get('summary') or parsed.get('error') or 'done')

    client._render_tool_result_message = _simple_tool_result_message  # type: ignore[assignment]
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'training_readiness':
            result = {
                'ok': True,
                'summary': '当前还不能直接训练：缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'resolved_img_dir': '/home/kly/ct_loop/data_ct/images',
                'resolved_label_dir': '/home/kly/ct_loop/data_ct/labels',
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
                'ready': True,
                'summary': '数据准备完成: 当前数据集可直接训练，data_yaml 已就绪',
                'dataset_root': '/home/kly/ct_loop/data_ct',
                'img_dir': '/home/kly/ct_loop/data_ct/images',
                'label_dir': '/home/kly/ct_loop/data_ct/labels',
                'data_yaml': '/home/kly/ct_loop/data_ct/data.yaml',
                'warnings': ['发现 209 张图片缺少标签（占比 6.1%），训练结果可能受到明显影响'],
                'steps_completed': [{'step': 'generate_yaml', 'status': 'completed'}],
                'next_actions': ['如需训练，可继续 start_training'],
            }
        elif tool_name == 'training_preflight':
            result = {
                'ok': True,
                'ready_to_start': False,
                'summary': '当前还不能直接训练: 当前没有空闲 GPU',
                'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
                'resolved_args': {
                    'model': kwargs['model'],
                    'data_yaml': kwargs['data_yaml'],
                    'epochs': kwargs['epochs'],
                    'device': kwargs.get('device', 'auto') or 'auto',
                    'training_environment': 'yolodo',
                },
                'blockers': ['当前没有空闲 GPU'],
                'warnings': ['发现 209 张图片缺少标签（占比 6.1%），训练结果可能受到明显影响'],
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    turn1 = await client.chat('用 /home/kly/yolov8n.pt 训练一下 /home/kly/ct_loop/data_ct')
    assert turn1['status'] == 'needs_confirmation', turn1
    assert turn1['tool_call']['name'] == 'prepare_dataset_for_training', turn1
    graph_calls_after_turn1 = graph.calls

    turn2 = await client.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'error', turn2
    assert '数据准备完成' in turn2['message'], turn2
    assert '已准备好的 YAML: /home/kly/ct_loop/data_ct/data.yaml' in turn2['message'], turn2
    assert '当前还不能直接训练: 当前没有空闲 GPU' in turn2['message'], turn2
    assert '阻塞项:' in turn2['message'], turn2
    assert '当前没有空闲 GPU' in turn2['message'], turn2
    assert '请问您希望' not in turn2['message'], turn2
    assert '根据您提供的指令' not in turn2['message'], turn2
    assert graph.calls == graph_calls_after_turn1, graph.calls
    assert [name for name, _ in calls].count('prepare_dataset_for_training') == 1
    assert [name for name, _ in calls].count('training_preflight') == 1
    events = client.memory.read_events(client.session_state.session_id)
    assert any(
        event.get('type') == 'post_prepare_followup_resolved'
        and event.get('ready_to_start') is False
        for event in events
    ), events
    assert any(
        event.get('type') == 'post_prepare_followup_rendered'
        and event.get('mode') == 'grounded'
        for event in events
    ), events


async def _scenario_graph_prepare_approve_prefers_local_post_prepare_refresh() -> None:
    scenario_root = WORK / 'graph_prepare_approve_prefers_local_refresh'
    graph = _GraphPrepareApproveWithStaleStartGraph()
    client = YoloStudioAgentClient(
        graph=graph,
        settings=AgentSettings(session_id='training-plan-dialogue-graph-post-prepare-local', memory_root=str(scenario_root)),
        tool_registry={},
    )
    graph.bind(client)
    client.session_state.active_dataset.dataset_root = '/home/kly/ct_loop/data_ct'
    client.session_state.active_training.training_plan_draft = {
        'dataset_path': '/home/kly/ct_loop/data_ct',
        'execution_mode': 'prepare_then_train',
        'execution_backend': 'standard_yolo',
        'next_step_tool': 'prepare_dataset_for_training',
        'next_step_args': {'dataset_path': '/home/kly/ct_loop/data_ct'},
        'planned_training_args': {
            'model': '/home/kly/yolov8n.pt',
            'epochs': 100,
        },
        'reasoning_summary': '当前数据还不能直接训练，但可以先自动准备到可训练状态。',
        'data_summary': '当前还不能直接训练: 缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
    }
    client.memory.save_state(client.session_state)
    client._remember_pending_confirmation(
        {
            'thread_id': 'graph-prepare-turn-1',
            'name': 'prepare_dataset_for_training',
            'args': {'dataset_path': '/home/kly/ct_loop/data_ct'},
            'id': 'prepare-graph',
            'synthetic': False,
            'source': 'graph',
        },
        emit_event=False,
        persist_graph=False,
    )

    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name != 'training_preflight':
            raise AssertionError(f'unexpected tool call: {tool_name}')
        return {
            'ok': True,
            'ready_to_start': True,
            'summary': '训练预检通过：将使用 yolodo，device=auto',
            'training_environment': {'name': 'yolodo', 'display_name': 'yolodo'},
            'resolved_args': {
                'model': kwargs['model'],
                'data_yaml': kwargs['data_yaml'],
                'epochs': kwargs['epochs'],
                'device': kwargs.get('device', 'auto') or 'auto',
                'training_environment': 'yolodo',
            },
            'command_preview': ['yolo', 'train'],
            'blockers': [],
            'warnings': [],
        }

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    original_graph_invoke = client._graph_invoke
    graph_stream_handlers: list[Any] = []

    async def _wrapped_graph_invoke(payload: Any, config: dict[str, Any], stream_handler: Any = None) -> dict[str, Any]:
        graph_stream_handlers.append(stream_handler)
        return await original_graph_invoke(payload, config, stream_handler=stream_handler)

    client._graph_invoke = _wrapped_graph_invoke  # type: ignore[assignment]

    streamed_events: list[dict[str, Any]] = []

    async def _capture_stream(event: dict[str, Any]) -> None:
        streamed_events.append(dict(event))

    turn = await client.confirm('graph-prepare-turn-1', approved=True, stream_handler=_capture_stream)
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'start_training', turn
    assert turn['tool_call']['args']['model'] == '/home/kly/yolov8n.pt', turn
    assert turn['tool_call']['args']['data_yaml'] == '/home/kly/ct_loop/data_ct/images_split(graph)/data.yaml', turn
    assert '训练启动确认：' in turn['message'], turn
    assert '执行方式: 直接训练' in turn['message'], turn
    assert len(calls) == 1, calls
    assert calls[0][0] == 'training_preflight', calls
    assert calls[0][1]['model'] == '/home/kly/yolov8n.pt', calls
    assert calls[0][1]['data_yaml'] == '/home/kly/ct_loop/data_ct/images_split(graph)/data.yaml', calls
    assert calls[0][1]['epochs'] == 100, calls
    assert graph_stream_handlers == [None, None], graph_stream_handlers
    assert streamed_events == [], streamed_events
    interrupt_payload = dict(turn.get('interrupt_payload') or {})
    assert interrupt_payload.get('type') == 'training_confirmation', turn
    assert interrupt_payload.get('next_step_tool') == 'start_training', interrupt_payload
    assert dict(interrupt_payload.get('next_step_args') or {}).get('data_yaml') == '/home/kly/ct_loop/data_ct/images_split(graph)/data.yaml', interrupt_payload


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_discussion_then_execute()
        await _scenario_status_query_without_session_context()
        await _scenario_prepare_only_revision()
        await _scenario_prepare_only_short_revision_without_dataset_path()
        await _scenario_prepare_then_replan_and_execute()
        await _scenario_prepare_only_natural_language_short_circuit()
        await _scenario_empty_input_is_ignored()
        await _scenario_prepare_only_invalid_path_is_not_confirmed()
        await _scenario_prepare_then_train_prompt_is_not_short_circuited()
        await _scenario_prepare_then_train_preserves_explicit_classes_txt()
        await _scenario_pending_prepare_then_train_uses_structured_surface_even_with_planner()
        await _scenario_prepare_only_with_planner_keeps_natural_surface()
        await _scenario_text_only_training_plan_materializes_pending()
        await _scenario_text_only_prepare_question_materializes_pending()
        await _scenario_text_only_prepare_question_restore_edit_stays_local()
        await _scenario_post_prepare_ready_start_confirmation_stays_local()
        await _scenario_post_prepare_blocked_followup_uses_grounded_surface()
        await _scenario_graph_prepare_approve_prefers_local_post_prepare_refresh()
        await _scenario_cancel_then_replan()
        await _scenario_cancel_prepare_then_rebuild()
        await _scenario_preparable_backend_switch()
        await _scenario_prepare_approval_then_revise_start()
        print('training plan dialogue ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
