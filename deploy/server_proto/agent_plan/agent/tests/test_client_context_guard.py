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


class _GraphState:
    def __init__(self, *, values=None, interrupts=None, tasks=None):
        self.values = values or {}
        self.interrupts = interrupts or ()
        self.tasks = tasks or ()


class _CaptureGraph:
    def __init__(self) -> None:
        self.payload = None

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        self.payload = payload
        return {'messages': list(payload.get('messages') or [])}


class _CheckpointGraph:
    def __init__(self, states: dict[str, _GraphState]) -> None:
        self.states = dict(states)

    def get_state(self, config):
        configurable = dict((config or {}).get('configurable') or {})
        thread_id = str(configurable.get('thread_id') or '').strip()
        return self.states.get(thread_id)


class _FakeCheckpointSaver:
    def __init__(self, thread_ids: list[str]) -> None:
        self._thread_ids = list(thread_ids)

    def thread_ids(self, prefix: str = '') -> list[str]:
        ids = list(self._thread_ids)
        if prefix:
            ids = [thread_id for thread_id in ids if thread_id.startswith(prefix)]
        return ids


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


async def _scenario_stale_graph_pending_is_cleared_on_startup() -> None:
    root = WORK / 'startup-stale-pending'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-stale-pending')
    state.active_training.training_plan_draft = {'execution_mode': 'direct_train', 'model': 'yolov8n.pt'}
    state.pending_confirmation.thread_id = 'startup-stale-pending-turn-1'
    state.pending_confirmation.tool_name = 'remote_training_pipeline'
    state.pending_confirmation.tool_args = {'server': 'yolostudio'}
    state.pending_confirmation.source = 'graph'
    state.pending_confirmation.summary = '等待远端训练闭环确认'
    store.save_state(state)

    settings = AgentSettings(session_id='startup-stale-pending', memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoGraph(), settings=settings, tool_registry={})
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.pending_confirmation.thread_id == ''
    assert client.session_state.active_training.training_plan_draft == {}
    events = client.memory.read_events(client.session_state.session_id)
    assert any(event.get('type') == 'startup_stale_pending_cleared' for event in events), events


async def _scenario_graph_pending_is_restored_from_checkpoint_on_startup() -> None:
    root = WORK / 'startup-graph-pending-restore'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-graph-pending-restore')
    store.save_state(state)

    thread_id = 'startup-graph-pending-restore-turn-1'
    graph = _CheckpointGraph(
        {
            thread_id: _GraphState(
                values={
                    'pending_confirmation': {
                        'name': 'remote_training_pipeline',
                        'args': {'server': 'yolostudio'},
                        'thread_id': thread_id,
                        'source': 'graph',
                        'summary': '等待远端训练闭环确认',
                    },
                    'pending_review': {'mode': 'clarify'},
                    'training_plan_context': {
                        'stage': 'plan_ready',
                        'status': 'ready_for_confirmation',
                        'dataset_path': '/data/demo',
                        'execution_mode': 'direct_train',
                        'execution_backend': 'standard_yolo',
                        'reasoning_summary': '确认后即可启动训练。',
                        'preflight_summary': '环境检查通过。',
                        'next_step_tool': 'remote_training_pipeline',
                        'next_step_args': {'server': 'yolostudio'},
                        'planned_training_args': {'model': 'yolov8n.pt', 'batch': 8},
                        'warnings': ['样本量偏小，建议先小步验证'],
                    },
                }
            )
        }
    )
    settings = AgentSettings(session_id='startup-graph-pending-restore', memory_root=str(root))
    client = YoloStudioAgentClient(
        graph=graph,
        settings=settings,
        tool_registry={},
        checkpointer=_FakeCheckpointSaver([thread_id]),
    )
    assert client.session_state.pending_confirmation.tool_name == 'remote_training_pipeline'
    assert client.session_state.pending_confirmation.thread_id == thread_id
    assert client.session_state.pending_confirmation.source == 'graph'
    assert client.session_state.pending_confirmation.decision_context == {'mode': 'clarify'}
    assert client.session_state.active_training.workflow_state == 'pending_confirmation'
    restored_draft = dict(client.session_state.active_training.training_plan_draft or {})
    assert restored_draft.get('next_step_tool') == 'remote_training_pipeline'
    assert restored_draft.get('status') == 'ready_for_confirmation'
    assert (restored_draft.get('planned_training_args') or {}).get('model') == 'yolov8n.pt'
    message = await client._build_confirmation_message({'name': 'remote_training_pipeline', 'args': {'server': 'yolostudio'}})
    assert '确认后即可启动训练' in message
    events = client.memory.read_events(client.session_state.session_id)
    assert any(event.get('type') == 'startup_graph_pending_restored' for event in events), events
    assert any(event.get('type') == 'startup_training_plan_draft_restored' for event in events), events


async def _scenario_graph_pending_restore_replaces_stale_draft() -> None:
    root = WORK / 'startup-graph-pending-replace-draft'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-graph-pending-replace-draft')
    state.active_training.training_plan_draft = {
        'status': 'ready_for_confirmation',
        'dataset_path': '/data/stale-dataset',
        'next_step_tool': 'remote_training_pipeline',
        'next_step_args': {'server': 'stale-server'},
        'planned_training_args': {'model': 'yolov8s.pt', 'epochs': 20},
    }
    store.save_state(state)

    thread_id = 'startup-graph-pending-replace-draft-turn-1'
    graph = _CheckpointGraph(
        {
            thread_id: _GraphState(
                values={
                    'pending_confirmation': {
                        'name': 'remote_training_pipeline',
                        'args': {'server': 'yolostudio'},
                        'thread_id': thread_id,
                        'source': 'graph',
                        'summary': '等待远端训练闭环确认',
                    },
                    'training_plan_context': {
                        'stage': 'plan_ready',
                        'status': 'ready_for_confirmation',
                        'dataset_path': '/data/restored',
                        'execution_mode': 'direct_train',
                        'execution_backend': 'standard_yolo',
                        'reasoning_summary': '确认后即可启动训练。',
                        'next_step_tool': 'remote_training_pipeline',
                        'next_step_args': {'server': 'yolostudio'},
                        'planned_training_args': {'model': 'yolov8n.pt', 'epochs': 80},
                    },
                }
            )
        }
    )
    settings = AgentSettings(session_id='startup-graph-pending-replace-draft', memory_root=str(root))
    client = YoloStudioAgentClient(
        graph=graph,
        settings=settings,
        tool_registry={},
        checkpointer=_FakeCheckpointSaver([thread_id]),
    )
    restored_draft = dict(client.session_state.active_training.training_plan_draft or {})
    assert restored_draft.get('next_step_tool') == 'remote_training_pipeline'
    assert restored_draft.get('dataset_path') == '/data/restored'
    assert (restored_draft.get('planned_training_args') or {}).get('model') == 'yolov8n.pt'
    assert (restored_draft.get('planned_training_args') or {}).get('epochs') == 80
    assert (restored_draft.get('next_step_args') or {}).get('server') == 'yolostudio'
    assert client.session_state.active_training.workflow_state == 'pending_confirmation'
    events = client.memory.read_events(client.session_state.session_id)
    assert any(
        event.get('type') == 'startup_training_plan_draft_restored'
        and event.get('replaced_existing') is True
        for event in events
    ), events


async def _scenario_multiple_graph_pending_candidates_are_not_restored() -> None:
    root = WORK / 'startup-graph-pending-skip'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-graph-pending-skip')
    state.active_training.training_plan_draft = {'execution_mode': 'direct_train', 'model': 'yolov8n.pt'}
    store.save_state(state)

    thread_ids = [
        'startup-graph-pending-skip-turn-1',
        'startup-graph-pending-skip-turn-2',
    ]
    graph = _CheckpointGraph(
        {
            thread_ids[0]: _GraphState(
                values={
                    'pending_confirmation': {
                        'name': 'remote_training_pipeline',
                        'args': {'server': 'yolostudio-a'},
                        'thread_id': thread_ids[0],
                        'source': 'graph',
                        'summary': '等待远端训练闭环确认 A',
                    }
                }
            ),
            thread_ids[1]: _GraphState(
                values={
                    'pending_confirmation': {
                        'name': 'remote_training_pipeline',
                        'args': {'server': 'yolostudio-b'},
                        'thread_id': thread_ids[1],
                        'source': 'graph',
                        'summary': '等待远端训练闭环确认 B',
                    }
                }
            ),
        }
    )
    settings = AgentSettings(session_id='startup-graph-pending-skip', memory_root=str(root))
    client = YoloStudioAgentClient(
        graph=graph,
        settings=settings,
        tool_registry={},
        checkpointer=_FakeCheckpointSaver(thread_ids),
    )
    assert client.session_state.pending_confirmation.tool_name == ''
    assert client.session_state.pending_confirmation.thread_id == ''
    assert client.session_state.active_training.training_plan_draft == {}
    events = client.memory.read_events(client.session_state.session_id)
    assert any(event.get('type') == 'startup_pending_restore_skipped' for event in events), events


async def _scenario_existing_graph_pending_rehydrates_missing_draft() -> None:
    root = WORK / 'startup-existing-graph-pending-draft-restore'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-existing-graph-pending-draft-restore')
    thread_id = 'startup-existing-graph-pending-draft-restore-turn-1'
    state.pending_confirmation.thread_id = thread_id
    state.pending_confirmation.tool_name = 'start_training'
    state.pending_confirmation.tool_args = {'model': 'yolov8n.pt'}
    state.pending_confirmation.source = 'graph'
    state.pending_confirmation.summary = '等待启动训练确认'
    state.active_training.training_plan_draft = {
        'status': 'ready_for_confirmation',
        'dataset_path': '/data/stale-dataset',
        'next_step_tool': 'prepare_dataset_for_training',
        'planned_training_args': {'model': 'yolov8s.pt'},
    }
    store.save_state(state)

    graph = _CheckpointGraph(
        {
            thread_id: _GraphState(
                values={
                    'pending_confirmation': {
                        'name': 'start_training',
                        'args': {'model': 'yolov8n.pt'},
                        'thread_id': thread_id,
                        'source': 'graph',
                        'summary': '等待启动训练确认',
                    },
                    'training_plan_context': {
                        'stage': 'plan_ready',
                        'status': 'ready_for_confirmation',
                        'dataset_path': '/data/demo',
                        'execution_mode': 'direct_train',
                        'execution_backend': 'standard_yolo',
                        'reasoning_summary': '确认后即可启动训练。',
                        'next_step_tool': 'start_training',
                        'next_step_args': {'model': 'yolov8n.pt'},
                        'planned_training_args': {'model': 'yolov8n.pt', 'epochs': 100},
                    },
                }
            )
        }
    )
    settings = AgentSettings(session_id='startup-existing-graph-pending-draft-restore', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    restored_draft = dict(client.session_state.active_training.training_plan_draft or {})
    assert restored_draft.get('next_step_tool') == 'start_training'
    assert restored_draft.get('dataset_path') == '/data/demo'
    assert (restored_draft.get('planned_training_args') or {}).get('epochs') == 100
    message = await client._build_confirmation_message({'name': 'start_training', 'args': {'model': 'yolov8n.pt'}})
    assert '确认后即可启动训练' in message
    events = client.memory.read_events(client.session_state.session_id)
    assert any(
        event.get('type') == 'startup_training_plan_draft_restored'
        and event.get('replaced_existing') is True
        for event in events
    ), events


async def _scenario_existing_graph_pending_refreshes_stale_same_tool_draft() -> None:
    root = WORK / 'startup-existing-graph-pending-refresh-same-tool'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-existing-graph-pending-refresh-same-tool')
    thread_id = 'startup-existing-graph-pending-refresh-same-tool-turn-1'
    state.pending_confirmation.thread_id = thread_id
    state.pending_confirmation.tool_name = 'start_training'
    state.pending_confirmation.tool_args = {'model': 'yolov8n.pt'}
    state.pending_confirmation.source = 'graph'
    state.pending_confirmation.summary = '等待启动训练确认'
    state.active_training.training_plan_draft = {
        'status': 'ready_for_confirmation',
        'dataset_path': '/data/stale-dataset',
        'next_step_tool': 'start_training',
        'next_step_args': {'model': 'yolov8s.pt'},
        'planned_training_args': {'model': 'yolov8s.pt', 'epochs': 50},
    }
    store.save_state(state)

    graph = _CheckpointGraph(
        {
            thread_id: _GraphState(
                values={
                    'pending_confirmation': {
                        'name': 'start_training',
                        'args': {'model': 'yolov8n.pt'},
                        'thread_id': thread_id,
                        'source': 'graph',
                        'summary': '等待启动训练确认',
                    },
                    'training_plan_context': {
                        'stage': 'plan_ready',
                        'status': 'ready_for_confirmation',
                        'dataset_path': '/data/demo',
                        'execution_mode': 'direct_train',
                        'execution_backend': 'standard_yolo',
                        'reasoning_summary': '确认后即可启动训练。',
                        'next_step_tool': 'start_training',
                        'next_step_args': {'model': 'yolov8n.pt'},
                        'planned_training_args': {'model': 'yolov8n.pt', 'epochs': 100},
                    },
                }
            )
        }
    )
    settings = AgentSettings(session_id='startup-existing-graph-pending-refresh-same-tool', memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    restored_draft = dict(client.session_state.active_training.training_plan_draft or {})
    assert restored_draft.get('next_step_tool') == 'start_training'
    assert restored_draft.get('dataset_path') == '/data/demo'
    assert (restored_draft.get('next_step_args') or {}).get('model') == 'yolov8n.pt'
    assert (restored_draft.get('planned_training_args') or {}).get('epochs') == 100
    events = client.memory.read_events(client.session_state.session_id)
    assert any(
        event.get('type') == 'startup_training_plan_draft_restored'
        and event.get('replaced_existing') is True
        for event in events
    ), events


async def _scenario_legacy_synthetic_pending_is_replaced_by_single_graph_pending() -> None:
    root = WORK / 'startup-legacy-synthetic-replaced'
    store = MemoryStore(root)
    state = SessionState(session_id='startup-legacy-synthetic-replaced')
    state.pending_confirmation.thread_id = 'legacy-synthetic-turn-1'
    state.pending_confirmation.tool_name = 'start_training'
    state.pending_confirmation.tool_args = {'model': 'yolov8s.pt'}
    state.pending_confirmation.source = 'synthetic'
    state.pending_confirmation.summary = '旧的本地确认'
    state.active_training.training_plan_draft = {
        'status': 'ready_for_confirmation',
        'dataset_path': '/data/legacy',
        'next_step_tool': 'start_training',
        'planned_training_args': {'model': 'yolov8s.pt', 'epochs': 50},
    }
    store.save_state(state)

    thread_id = 'startup-legacy-synthetic-replaced-turn-1'
    graph = _CheckpointGraph(
        {
            thread_id: _GraphState(
                values={
                    'pending_confirmation': {
                        'name': 'remote_training_pipeline',
                        'args': {'server': 'yolostudio'},
                        'thread_id': thread_id,
                        'source': 'graph',
                        'summary': '等待远端训练闭环确认',
                    },
                    'training_plan_context': {
                        'stage': 'plan_ready',
                        'status': 'ready_for_confirmation',
                        'dataset_path': '/data/graph',
                        'execution_mode': 'direct_train',
                        'execution_backend': 'standard_yolo',
                        'reasoning_summary': '确认后即可启动远端训练。',
                        'next_step_tool': 'remote_training_pipeline',
                        'next_step_args': {'server': 'yolostudio'},
                        'planned_training_args': {'model': 'yolov8n.pt', 'epochs': 120},
                    },
                }
            )
        }
    )
    settings = AgentSettings(session_id='startup-legacy-synthetic-replaced', memory_root=str(root))
    client = YoloStudioAgentClient(
        graph=graph,
        settings=settings,
        tool_registry={},
        checkpointer=_FakeCheckpointSaver([thread_id]),
    )
    assert client.session_state.pending_confirmation.tool_name == 'remote_training_pipeline'
    assert client.session_state.pending_confirmation.thread_id == thread_id
    assert client.session_state.pending_confirmation.source == 'graph'
    restored_draft = dict(client.session_state.active_training.training_plan_draft or {})
    assert restored_draft.get('next_step_tool') == 'remote_training_pipeline'
    assert restored_draft.get('dataset_path') == '/data/graph'
    assert client.session_state.active_training.workflow_state == 'pending_confirmation'
    events = client.memory.read_events(client.session_state.session_id)
    assert any(
        event.get('type') == 'startup_legacy_pending_replaced'
        and event.get('previous_source') == 'synthetic'
        and event.get('restored_tool') == 'remote_training_pipeline'
        for event in events
    ), events


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
        await _scenario_stale_graph_pending_is_cleared_on_startup()
        await _scenario_graph_pending_is_restored_from_checkpoint_on_startup()
        await _scenario_graph_pending_restore_replaces_stale_draft()
        await _scenario_multiple_graph_pending_candidates_are_not_restored()
        await _scenario_existing_graph_pending_rehydrates_missing_draft()
        await _scenario_existing_graph_pending_refreshes_stale_same_tool_draft()
        await _scenario_legacy_synthetic_pending_is_replaced_by_single_graph_pending()
        await _scenario_best_weight_path_is_visible_to_graph_handoff()
        print('client context guard ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
