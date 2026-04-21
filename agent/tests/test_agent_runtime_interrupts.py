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
        raise AssertionError('create_react_agent should not be called in runtime interrupt tests')

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

from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.tests._pending_confirmation_test_support import seed_pending_confirmation
from yolostudio_agent.agent.tests._training_plan_test_support import current_training_plan_context_payload, set_training_plan_draft


class _DummyGraph:
    def get_state(self, config):
        return None


class _GraphState:
    def __init__(self, *, tool_name: str, tool_args: dict[str, Any]) -> None:
        self.next = ('tools',)
        self.values = {
            'messages': [
                AIMessage(content='', tool_calls=[{'id': 'call-1', 'name': tool_name, 'args': dict(tool_args)}])
            ]
        }


class _GraphWithPendingTool:
    def __init__(self, *, tool_name: str, tool_args: dict[str, Any]) -> None:
        self._state = _GraphState(tool_name=tool_name, tool_args=tool_args)
        self.resume_calls: list[Any] = []

    def get_state(self, config):
        del config
        return self._state

    async def ainvoke(self, *args, **kwargs):
        self.resume_calls.append((args, kwargs))
        raise AssertionError('synthetic pending should not resume graph state')


class _ResumableGraphWithoutVisiblePending:
    def __init__(self) -> None:
        self.resume_calls: list[Any] = []

    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, config=None):
        del config
        self.resume_calls.append(payload)
        messages = [
            AIMessage(
                content='',
                tool_calls=[{'id': 'call-1', 'name': 'upload_assets_to_remote', 'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']}}],
            ),
            ToolMessage(
                content='{"ok": true, "summary": "上传完成", "remote_root": "/tmp/agent_stage"}',
                name='upload_assets_to_remote',
                tool_call_id='call-1',
            ),
            AIMessage(content='上传完成'),
        ]
        return {'messages': messages}


class _PendingValueOnlyState:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        normalized_values = dict(values or {})
        pending = normalized_values.get('pending_confirmation')
        next_nodes: tuple[str, ...] = ()
        if isinstance(pending, dict) and pending:
            next_nodes = ('tools',)
        self.next = next_nodes
        self.values = {'messages': [], **normalized_values}
        self.interrupts = ()
        self.tasks = ()


class _GraphWithPendingValueOnly:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._state = _PendingValueOnlyState({'pending_confirmation': dict(payload)})
        self.resume_calls: list[Any] = []

    def get_state(self, config):
        del config
        return self._state

    def update_state(self, config, update):
        del config
        values = dict(getattr(self._state, 'values', {}) or {})
        for key, value in dict(update or {}).items():
            if isinstance(value, dict):
                values[key] = dict(value)
                continue
            if value is None:
                values.pop(key, None)
                continue
            values[key] = value
        self._state = _PendingValueOnlyState(values)

    async def ainvoke(self, *args, **kwargs):
        self.resume_calls.append((args, kwargs))
        raise AssertionError('graph pending value fallback should execute direct tool, not resume graph')


class _InterruptEnvelope:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.value = dict(payload)


class _InterruptTask:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.interrupts = [_InterruptEnvelope(payload)]


class _VisibleInterruptState:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.next = ('tools',) if payload else ()
        self.values = {'messages': []}
        self.interrupts = ()
        self.tasks = (_InterruptTask(payload),) if payload else ()


class _ResumableGraphWithVisibleInterrupt:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.resume_calls: list[Any] = []
        self._state = _VisibleInterruptState(payload)

    def get_state(self, config):
        del config
        return self._state

    async def ainvoke(self, payload, config=None):
        del config
        self.resume_calls.append(payload)
        resume = getattr(payload, 'resume', payload)
        decision = ''
        if isinstance(resume, dict):
            decision = str(resume.get('action') or resume.get('decision') or '').strip().lower()
        else:
            decision = str(resume or '').strip().lower()
        self._state = _VisibleInterruptState(None)
        if decision == 'reject':
            return {'messages': [AIMessage(content='已取消上传')]}
        return {'messages': [AIMessage(content='上传完成')]}


class _TrainingConfirmationGraphWithVisibleInterrupt:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.resume_calls: list[Any] = []
        self._state = _VisibleInterruptState(payload)

    def get_state(self, config):
        del config
        return self._state

    async def ainvoke(self, payload, config=None):
        del config
        self.resume_calls.append(payload)
        resume = getattr(payload, 'resume', payload)
        decision = ''
        if isinstance(resume, dict):
            decision = str(resume.get('action') or resume.get('decision') or '').strip().lower()
        else:
            decision = str(resume or '').strip().lower()
        self._state = _VisibleInterruptState(None)
        if decision == 'reject':
            return {'messages': [AIMessage(content='已取消循环训练')]}
        return {'messages': [AIMessage(content='环训练已启动')]}


WORK = Path(__file__).resolve().parent / '_tmp_runtime_interrupts'


async def _scenario_pending_payload_contract() -> None:
    scenario_root = WORK / 'payload_contract'
    settings = AgentSettings(session_id='runtime-interrupt-payload', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    seed_pending_confirmation(client, 
        'runtime-interrupt-payload-turn-1',
        {
            'name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'id': None,
            'synthetic': True,
        },
    )
    payload = client.get_pending_action()
    assert payload is not None
    assert payload['interrupt_kind'] == 'tool_approval'
    assert payload['decision_state'] == 'pending'
    assert payload['tool_name'] == 'upload_assets_to_remote'
    assert payload['tool_args']['server'] == 'lab'
    assert payload['summary'] == '上传资源到远端服务器'
    assert payload['objective'] == '把本地资源上传到远端服务器'
    assert payload['allowed_decisions'] == ['approve', 'reject', 'edit', 'clarify']
    assert payload['review_config']['risk_level'] == 'high'
    assert payload['decision_context'] == {}
    state_payload = client._pending_from_state()
    assert state_payload is not None
    assert state_payload['summary'] == payload['summary']
    assert state_payload['objective'] == payload['objective']


async def _scenario_review_reject() -> None:
    scenario_root = WORK / 'review_reject'
    settings = AgentSettings(session_id='runtime-interrupt-reject', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    seed_pending_confirmation(client, 
        'runtime-interrupt-reject-turn-1',
        {
            'name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'id': None,
            'synthetic': True,
        },
    )

    result = await client.chat('不继续')
    assert result['status'] == 'cancelled', result
    assert result['pending_action']['decision_state'] == 'rejected', result
    assert result['pending_action']['decision_context']['decision'] == 'reject', result
    assert result['pending_action']['decision_context']['raw_user_text'] == '不继续', result
    assert client.get_pending_action() is None


async def _scenario_review_clarify_keeps_pending() -> None:
    scenario_root = WORK / 'review_clarify'
    settings = AgentSettings(session_id='runtime-interrupt-clarify', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    seed_pending_confirmation(client, 
        'runtime-interrupt-clarify-turn-1',
        {
            'name': 'split_dataset',
            'args': {'dataset_root': '/data/demo', 'train_ratio': 0.8},
            'id': None,
            'synthetic': True,
        },
    )

    result = await client.chat('为什么要做这一步？')
    assert result['status'] == 'needs_confirmation', result
    assert result['pending_action']['decision_state'] == 'pending', result
    assert result['pending_action']['decision_context']['decision'] == 'clarify', result
    assert result['tool_call']['name'] == 'split_dataset', result
    assert client.get_pending_action() is not None


async def _scenario_review_edit_keeps_pending() -> None:
    scenario_root = WORK / 'review_edit'
    settings = AgentSettings(session_id='runtime-interrupt-edit', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    seed_pending_confirmation(client, 
        'runtime-interrupt-edit-turn-1',
        {
            'name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'id': None,
            'synthetic': True,
        },
    )

    result = await client.chat('把 batch 改成 12 再继续')
    assert result['status'] == 'needs_confirmation', result
    assert result['pending_action']['decision_state'] == 'pending', result
    assert result['pending_action']['decision_context']['decision'] == 'edit', result
    assert result['pending_action']['decision_context']['raw_user_text'] == '把 batch 改成 12 再继续', result
    assert result['tool_call']['name'] == 'upload_assets_to_remote', result
    assert client.get_pending_action() is not None


async def _scenario_review_approve() -> None:
    scenario_root = WORK / 'review_approve'
    settings = AgentSettings(session_id='runtime-interrupt-approve', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '上传已完成: server=lab',
            'resolved_args': dict(kwargs),
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    seed_pending_confirmation(client, 
        'runtime-interrupt-approve-turn-1',
        {
            'name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'id': None,
            'synthetic': True,
        },
    )

    result = await client.chat('继续')
    assert result['status'] == 'completed', result
    assert '上传已完成' in result['message'], result
    assert calls == [('upload_assets_to_remote', {'server': 'lab', 'local_paths': ['/tmp/demo.pt']})], calls
    assert client.get_pending_action() is None


async def _scenario_stale_graph_pending_is_cleared() -> None:
    scenario_root = WORK / 'stale_graph_pending'
    settings = AgentSettings(session_id='runtime-interrupt-stale-graph', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    seed_pending_confirmation(client, 
        'runtime-interrupt-stale-graph-turn-1',
        {
            'name': 'start_training',
            'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/demo/data.yaml', 'epochs': 10},
            'id': 'call-1',
            'source': 'graph',
        },
    )

    result = await client.confirm('runtime-interrupt-stale-graph-turn-1', approved=True)
    assert result['status'] == 'error', result
    assert client.get_pending_action() is None


async def _scenario_graph_pending_shadow_is_reused_when_graph_can_resume() -> None:
    scenario_root = WORK / 'graph_pending_shadow_reused'
    settings = AgentSettings(session_id='runtime-interrupt-graph-shadow', memory_root=str(scenario_root))
    graph = _ResumableGraphWithoutVisiblePending()
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    seed_pending_confirmation(client, 
        'runtime-interrupt-graph-shadow-turn-1',
        {
            'name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'id': 'call-1',
            'source': 'graph',
        },
    )

    pending = client.get_pending_action()
    assert pending is not None, pending
    assert pending['tool_name'] == 'upload_assets_to_remote', pending

    result = await client.confirm('runtime-interrupt-graph-shadow-turn-1', approved=True)
    assert result['status'] == 'completed', result
    assert '上传完成' in result['message'], result
    assert len(graph.resume_calls) == 1, graph.resume_calls
    assert client.get_pending_action() is None


async def _scenario_confirm_approves_visible_graph_interrupt() -> None:
    scenario_root = WORK / 'visible_graph_interrupt_approve'
    settings = AgentSettings(session_id='runtime-interrupt-visible-approve', memory_root=str(scenario_root))
    graph = _ResumableGraphWithVisibleInterrupt(
        {
            'id': 'call-1',
            'tool_call_id': 'call-1',
            'name': 'upload_assets_to_remote',
            'tool_name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'tool_args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'summary': '上传资源到远端服务器',
            'objective': '把本地资源上传到远端服务器',
            'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
            'review_config': {'risk_level': 'high'},
            'decision_context': {},
            'thread_id': 'visible-graph-approve-turn-1',
            'source': 'graph',
            'interrupt_kind': 'tool_approval',
        }
    )
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

    pending = client.get_pending_action()
    assert pending is not None, pending
    assert pending['tool_name'] == 'upload_assets_to_remote', pending

    result = await client.confirm('visible-graph-approve-turn-1', approved=True)
    assert result['status'] == 'completed', result
    assert '上传完成' in result['message'], result
    assert len(graph.resume_calls) == 1, graph.resume_calls
    assert client.get_pending_action() is None


async def _scenario_confirm_rejects_visible_graph_interrupt() -> None:
    scenario_root = WORK / 'visible_graph_interrupt_reject'
    settings = AgentSettings(session_id='runtime-interrupt-visible-reject', memory_root=str(scenario_root))
    graph = _ResumableGraphWithVisibleInterrupt(
        {
            'id': 'call-1',
            'tool_call_id': 'call-1',
            'name': 'upload_assets_to_remote',
            'tool_name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'tool_args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'summary': '上传资源到远端服务器',
            'objective': '把本地资源上传到远端服务器',
            'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
            'review_config': {'risk_level': 'high'},
            'decision_context': {},
            'thread_id': 'visible-graph-reject-turn-1',
            'source': 'graph',
            'interrupt_kind': 'tool_approval',
        }
    )
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

    result = await client.confirm('visible-graph-reject-turn-1', approved=False)
    assert result['status'] == 'cancelled', result
    assert result['pending_action']['decision_state'] == 'rejected', result
    assert len(graph.resume_calls) == 1, graph.resume_calls
    assert client.get_pending_action() is None


async def _scenario_chat_approves_visible_graph_interrupt() -> None:
    scenario_root = WORK / 'visible_graph_interrupt_chat_approve'
    settings = AgentSettings(session_id='runtime-interrupt-visible-chat-approve', memory_root=str(scenario_root))
    graph = _ResumableGraphWithVisibleInterrupt(
        {
            'id': 'call-1',
            'tool_call_id': 'call-1',
            'name': 'upload_assets_to_remote',
            'tool_name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'tool_args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'summary': '上传资源到远端服务器',
            'objective': '把本地资源上传到远端服务器',
            'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
            'review_config': {'risk_level': 'high'},
            'decision_context': {},
            'thread_id': 'visible-graph-chat-approve-turn-1',
            'source': 'graph',
            'interrupt_kind': 'tool_approval',
        }
    )
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

    result = await client.chat('继续')
    assert result['status'] == 'completed', result
    assert '上传完成' in result['message'], result
    assert len(graph.resume_calls) == 1, graph.resume_calls
    assert client.get_pending_action() is None


async def _scenario_chat_clarifies_visible_graph_interrupt() -> None:
    scenario_root = WORK / 'visible_graph_interrupt_chat_clarify'
    settings = AgentSettings(session_id='runtime-interrupt-visible-chat-clarify', memory_root=str(scenario_root))
    graph = _ResumableGraphWithVisibleInterrupt(
        {
            'id': 'call-1',
            'tool_call_id': 'call-1',
            'name': 'upload_assets_to_remote',
            'tool_name': 'upload_assets_to_remote',
            'args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'tool_args': {'server': 'lab', 'local_paths': ['/tmp/demo.pt']},
            'summary': '上传资源到远端服务器',
            'objective': '把本地资源上传到远端服务器',
            'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
            'review_config': {'risk_level': 'high'},
            'decision_context': {},
            'thread_id': 'visible-graph-chat-clarify-turn-1',
            'source': 'graph',
            'interrupt_kind': 'tool_approval',
        }
    )
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

    async def _fake_confirmation_message(pending):
        assert pending['name'] == 'upload_assets_to_remote'
        return '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。'

    client._build_confirmation_message = _fake_confirmation_message  # type: ignore[assignment]
    result = await client.chat('为什么这样安排？')
    assert result['status'] == 'needs_confirmation', result
    assert result['message'] == '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。', result
    assert graph.resume_calls == [], graph.resume_calls
    pending = client.get_pending_action()
    assert pending is not None, result
    assert pending['tool_name'] == 'upload_assets_to_remote', pending


async def _scenario_training_confirmation_chat_override_status_to_approve() -> None:
    scenario_root = WORK / 'training_confirmation_chat_override_status'
    settings = AgentSettings(session_id='runtime-training-confirm-override', memory_root=str(scenario_root))
    graph = _TrainingConfirmationGraphWithVisibleInterrupt(
        {
            'type': 'training_confirmation',
            'phase': 'start',
            'plan': {
                'mode': 'loop',
                'dataset_path': '/data/loop',
                'model': 'yolov8n.pt',
                'data_yaml': '/data/loop/data.yaml',
                'device': 'auto',
                'max_rounds': 2,
                'epochs_per_round': 10,
            },
            'execution_mode': 'prepare_then_loop',
            'next_step_tool': 'start_training_loop',
            'next_step_args': {
                'model': 'yolov8n.pt',
                'data_yaml': '/data/loop/data.yaml',
                'device': 'auto',
                'max_rounds': 2,
            },
            'thread_id': 'runtime-training-confirm-override-turn-1',
        }
    )
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

    async def _fake_invoke_structured_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {'action': 'status', 'reason': '请再次确认'}

    client._invoke_structured_payload = _fake_invoke_structured_payload  # type: ignore[assignment]

    result = await client.chat('确认开始')
    assert result['status'] == 'completed', result
    assert '环训练已启动' in result['message'], result
    assert len(graph.resume_calls) == 1, graph.resume_calls
    resume_payload = getattr(graph.resume_calls[0], 'resume', graph.resume_calls[0])
    assert dict(resume_payload or {}).get('action') == 'approve', resume_payload
    assert client.get_pending_action() is None


async def _scenario_training_confirmation_shadow_fallback_approves_when_interrupt_not_visible() -> None:
    scenario_root = WORK / 'training_confirmation_shadow_fallback'
    settings = AgentSettings(session_id='runtime-training-confirm-shadow', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    payload = {
        'type': 'training_confirmation',
        'phase': 'start',
        'plan': {
            'mode': 'loop',
            'dataset_path': '/data/loop',
            'model': 'yolov8n.pt',
            'data_yaml': '/data/loop/data.yaml',
            'device': 'auto',
            'max_rounds': 2,
            'epochs_per_round': 10,
        },
        'execution_mode': 'prepare_then_loop',
        'next_step_tool': 'start_training_loop',
        'next_step_args': {
            'model': 'yolov8n.pt',
            'data_yaml': '/data/loop/data.yaml',
            'device': 'auto',
            'max_rounds': 2,
            'project': None,
            'name': None,
            'optimizer': None,
        },
        'thread_id': 'runtime-training-confirm-shadow-turn-1',
    }
    initial = client._format_interrupt_or_result({}, thread_id='runtime-training-confirm-shadow-turn-1', interrupt_payload=payload)
    assert initial['status'] == 'needs_confirmation', initial
    assert client.get_pending_action() is not None
    client.planner_llm = object()
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {'ok': True, 'summary': '环训练已启动', 'loop_id': 'loop-123', 'status': 'queued'}
        client._apply_to_state(tool_name, result, kwargs)
        return result

    async def _fake_invoke_structured_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {'action': 'status', 'reason': '请再次确认'}

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    client._invoke_structured_payload = _fake_invoke_structured_payload  # type: ignore[assignment]

    result = await client.chat('确认开始')
    assert result['status'] == 'completed', result
    assert calls == [('start_training_loop', {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'device': 'auto', 'max_rounds': 2})], calls
    assert client.get_pending_action() is None


async def _scenario_synthetic_pending_ignores_graph_pending() -> None:
    scenario_root = WORK / 'synthetic_ignores_graph'
    settings = AgentSettings(session_id='runtime-interrupt-synthetic-graph', memory_root=str(scenario_root))
    graph = _GraphWithPendingTool(
        tool_name='split_dataset',
        tool_args={'dataset_root': '/data/demo', 'train_ratio': 0.8},
    )
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {
            'ok': True,
            'summary': '环训练已启动',
            'loop_id': 'loop-123',
            'status': 'queued',
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    seed_pending_confirmation(client, 
        'runtime-interrupt-synthetic-graph-turn-1',
        {
            'name': 'start_training_loop',
            'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2},
            'id': None,
            'synthetic': True,
        },
    )

    result = await client.confirm('runtime-interrupt-synthetic-graph-turn-1', approved=True)
    assert result['status'] == 'completed', result
    assert calls == [('start_training_loop', {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2})], calls
    assert graph.resume_calls == [], graph.resume_calls


async def _scenario_confirm_executes_graph_pending_value_without_visible_interrupt() -> None:
    scenario_root = WORK / 'graph_pending_value_confirm'
    settings = AgentSettings(session_id='runtime-graph-pending-value-confirm', memory_root=str(scenario_root))
    pending = {
        'id': 'call-1',
        'tool_call_id': 'call-1',
        'name': 'start_training',
        'tool_name': 'start_training',
        'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/demo/data.yaml', 'epochs': 10, 'device': 'auto'},
        'tool_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/demo/data.yaml', 'epochs': 10, 'device': 'auto'},
        'summary': '启动训练：model=yolov8n.pt, data=/data/demo/data.yaml, epochs=10',
        'objective': '启动训练（yolov8n.pt / /data/demo/data.yaml）',
        'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
        'review_config': {'risk_level': 'high'},
        'decision_context': {},
        'thread_id': 'runtime-graph-pending-value-confirm-turn-1',
        'source': 'graph',
        'interrupt_kind': 'tool_approval',
    }
    graph = _GraphWithPendingValueOnly(pending)
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {'ok': True, 'summary': '训练已启动', 'status': 'running'}
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    result = await client.confirm('runtime-graph-pending-value-confirm-turn-1', approved=True)
    assert result['status'] == 'completed', result
    assert calls == [('start_training', {'model': 'yolov8n.pt', 'data_yaml': '/data/demo/data.yaml', 'epochs': 10, 'device': 'auto'})], calls
    assert graph.resume_calls == [], graph.resume_calls
    assert client.get_pending_action() is None


async def _scenario_chat_executes_graph_pending_value_without_visible_interrupt() -> None:
    scenario_root = WORK / 'graph_pending_value_chat'
    settings = AgentSettings(session_id='runtime-graph-pending-value-chat', memory_root=str(scenario_root))
    pending = {
        'id': 'call-1',
        'tool_call_id': 'call-1',
        'name': 'start_training_loop',
        'tool_name': 'start_training_loop',
        'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'},
        'tool_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'},
        'summary': '启动循环训练：model=yolov8n.pt, data=/data/loop/data.yaml, max_rounds=2',
        'objective': '启动循环训练（yolov8n.pt / /data/loop/data.yaml）',
        'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
        'review_config': {'risk_level': 'high'},
        'decision_context': {},
        'thread_id': 'runtime-graph-pending-value-chat-turn-1',
        'source': 'graph',
        'interrupt_kind': 'tool_approval',
    }
    graph = _GraphWithPendingValueOnly(pending)
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {'ok': True, 'summary': '环训练已启动', 'loop_id': 'loop-123', 'status': 'queued'}
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

    result = await client.chat('确认开始')
    assert result['status'] == 'completed', result
    assert calls == [('start_training_loop', {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'})], calls
    assert graph.resume_calls == [], graph.resume_calls
    assert client.get_pending_action() is None


async def _scenario_chat_pending_value_override_status_to_approve() -> None:
    scenario_root = WORK / 'graph_pending_value_chat_override_status'
    settings = AgentSettings(session_id='runtime-graph-pending-value-chat-override', memory_root=str(scenario_root))
    pending = {
        'id': 'call-1',
        'tool_call_id': 'call-1',
        'name': 'start_training_loop',
        'tool_name': 'start_training_loop',
        'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'},
        'tool_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'},
        'summary': '启动循环训练：model=yolov8n.pt, data=/data/loop/data.yaml, max_rounds=2',
        'objective': '启动循环训练（yolov8n.pt / /data/loop/data.yaml）',
        'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
        'review_config': {'risk_level': 'high'},
        'decision_context': {},
        'thread_id': 'runtime-graph-pending-value-chat-override-turn-1',
        'source': 'graph',
        'interrupt_kind': 'tool_approval',
    }
    graph = _GraphWithPendingValueOnly(pending)
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    client.planner_llm = object()
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        result = {'ok': True, 'summary': '环训练已启动', 'loop_id': 'loop-123', 'status': 'queued'}
        client._apply_to_state(tool_name, result, kwargs)
        return result

    async def _fake_invoke_structured_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {'action': 'status', 'reason': '请再次确认'}

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    client._invoke_structured_payload = _fake_invoke_structured_payload  # type: ignore[assignment]

    result = await client.chat('确认开始')
    assert result['status'] == 'completed', result
    assert calls == [('start_training_loop', {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'})], calls
    assert graph.resume_calls == [], graph.resume_calls
    assert client.get_pending_action() is None


async def _scenario_pending_passthrough_restores_graph_pending_and_context() -> None:
    scenario_root = WORK / 'pending_passthrough_restore'
    settings = AgentSettings(session_id='runtime-pending-passthrough-restore', memory_root=str(scenario_root))
    pending_thread_id = 'runtime-pending-passthrough-restore-turn-1'
    pending = {
        'id': 'call-1',
        'tool_call_id': 'call-1',
        'name': 'start_training_loop',
        'tool_name': 'start_training_loop',
        'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'},
        'tool_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'},
        'summary': '启动循环训练：model=yolov8n.pt, data=/data/loop/data.yaml, max_rounds=2',
        'objective': '启动循环训练（yolov8n.pt / /data/loop/data.yaml）',
        'allowed_decisions': ['approve', 'reject', 'edit', 'clarify'],
        'review_config': {'risk_level': 'high'},
        'decision_context': {},
        'thread_id': pending_thread_id,
        'source': 'graph',
        'interrupt_kind': 'tool_approval',
    }
    graph = _GraphWithPendingValueOnly(pending)
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    set_training_plan_draft(
        client,
        {
            'dataset_path': '/data/loop',
            'model': 'yolov8n.pt',
            'device': 'auto',
            'next_step_tool': 'start_training_loop',
            'next_step_args': {'model': 'yolov8n.pt', 'data_yaml': '/data/loop/data.yaml', 'max_rounds': 2, 'device': 'auto'},
        },
        thread_id=pending_thread_id,
    )
    client._resolve_pending_confirmation(thread_id=pending_thread_id, config=client._pending_config(pending_thread_id))

    async def _fake_handoff_current_runtime_to_graph(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        client._update_graph_pending_state(thread_id=pending_thread_id, pending=None)
        client._clear_pending_confirmation(thread_id=pending_thread_id, persist_graph=False)
        client._update_graph_training_plan_context(thread_id=pending_thread_id, context=None)
        return {'status': 'completed', 'message': '最近有 2 条环训练记录。'}

    client._handoff_current_runtime_to_graph = _fake_handoff_current_runtime_to_graph  # type: ignore[assignment]
    result = await client._handoff_chat_turn(
        thread_id=pending_thread_id,
        user_text='最近有哪些环训练',
        pending_passthrough_requested=True,
    )
    assert result['status'] == 'completed', result
    restored_pending = client.get_pending_action()
    assert (restored_pending or {}).get('tool_name') == 'start_training_loop', restored_pending
    restored_context = current_training_plan_context_payload(client, thread_id=pending_thread_id)
    assert restored_context is not None, restored_context
    assert restored_context.get('next_step_tool') == 'start_training_loop', restored_context
    assert dict(restored_context.get('next_step_args') or {}).get('model') == 'yolov8n.pt', restored_context


async def _scenario_runtime_ignores_manual_pending_session_mutation() -> None:
    scenario_root = WORK / 'runtime-ignores-manual-session-pending'
    settings = AgentSettings(session_id='runtime-ignore-manual-session', memory_root=str(scenario_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
    assert client._pending_from_state() is None
    assert client.get_pending_action() is None


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_pending_payload_contract()
        await _scenario_review_reject()
        await _scenario_review_clarify_keeps_pending()
        await _scenario_review_edit_keeps_pending()
        await _scenario_review_approve()
        await _scenario_stale_graph_pending_is_cleared()
        await _scenario_graph_pending_shadow_is_reused_when_graph_can_resume()
        await _scenario_confirm_approves_visible_graph_interrupt()
        await _scenario_confirm_rejects_visible_graph_interrupt()
        await _scenario_chat_approves_visible_graph_interrupt()
        await _scenario_chat_clarifies_visible_graph_interrupt()
        await _scenario_training_confirmation_chat_override_status_to_approve()
        await _scenario_training_confirmation_shadow_fallback_approves_when_interrupt_not_visible()
        await _scenario_synthetic_pending_ignores_graph_pending()
        await _scenario_confirm_executes_graph_pending_value_without_visible_interrupt()
        await _scenario_chat_executes_graph_pending_value_without_visible_interrupt()
        await _scenario_chat_pending_value_override_status_to_approve()
        await _scenario_pending_passthrough_restores_graph_pending_and_context()
        await _scenario_runtime_ignores_manual_pending_session_mutation()
        print('agent runtime interrupts ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(_run())
