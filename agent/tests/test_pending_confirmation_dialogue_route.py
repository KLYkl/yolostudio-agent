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


def _install_fake_test_dependencies() -> None:
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
        raise AssertionError('create_react_agent should not be called in pending confirmation dialogue tests')

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
from langchain_core.messages import AIMessage


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('pending confirmation dialogue should stay on routed flows, not fallback to graph')


class _CountingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        from langchain_core.messages import AIMessage

        payload = args[0] if args else {}
        messages = list((payload or {}).get('messages') or [])
        return {'messages': messages + [AIMessage(content='graph-called')]}


class _LoopListGraph:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, *args, **kwargs):
        del args, kwargs
        self.payloads.append(dict(payload or {}))
        messages = list((payload or {}).get('messages') or [])
        return {'messages': messages + [AIMessage(content='找到 2 条环训练记录')]}


WORK = Path(__file__).resolve().parent / '_tmp_pending_confirmation_dialogue_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})
    pending = {
        'name': 'start_training',
        'args': {'model': 'yolov8n.pt', 'data_yaml': '/data/demo/data.yaml', 'epochs': 10},
        'id': None,
        'synthetic': True,
    }
    client._set_pending_confirmation(f'{session_id}-pending', pending)
    return client


def _make_prepare_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})
    pending = {
        'name': 'prepare_dataset_for_training',
        'args': {'dataset_path': '/data/demo', 'force_split': True},
        'id': None,
        'synthetic': True,
    }
    client._set_pending_confirmation(f'{session_id}-pending', pending)
    return client


def _seed_prepare_training_plan(client: YoloStudioAgentClient) -> None:
    client.session_state.active_training.training_plan_draft = {
        'dataset_path': '/data/demo',
        'execution_mode': 'prepare_then_train',
        'next_step_tool': 'prepare_dataset_for_training',
        'next_step_args': {'dataset_path': '/data/demo', 'force_split': True},
        'planned_training_args': {'model': 'yolov8n.pt', 'epochs': 100},
        'reasoning_summary': '当前数据还不能直接训练，但可以先自动准备到可训练状态。',
        'data_summary': '当前还不能直接训练: 缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
    }
    client.session_state.active_dataset.dataset_root = '/data/demo'
    client.session_state.active_dataset.last_readiness = {
        'ok': True,
        'ready': False,
        'preparable': True,
        'dataset_root': '/data/demo',
        'resolved_data_yaml': '',
        'blockers': ['缺少可用的 data_yaml'],
        'summary': '当前还不能直接训练: 缺少可用的 data_yaml；但当前数据集可以先进入 prepare_dataset_for_training',
    }
    client.memory.save_state(client.session_state)


def _make_prepare_revision_client(session_id: str, graph=None) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph or _NoLLMGraph(), settings=settings, tool_registry={})
    pending = {
        'name': 'prepare_dataset_for_training',
        'args': {'dataset_path': '/data/demo', 'force_split': True},
        'id': None,
        'synthetic': True,
    }
    client._set_pending_confirmation(f'{session_id}-pending', pending)
    _seed_prepare_training_plan(client)
    return client


async def _scenario_status_query_reuses_confirmation_message() -> None:
    client = _make_client('pending-status')

    async def _fake_confirmation_message(pending):
        assert pending['name'] == 'start_training'
        return '当前待确认的是启动训练：将使用 yolov8n.pt 训练 10 轮。'

    client._build_confirmation_message = _fake_confirmation_message  # type: ignore[assignment]
    turn = await client.chat('查看训练状态')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['message'] == '当前待确认的是启动训练：将使用 yolov8n.pt 训练 10 轮。', turn


async def _scenario_generic_text_reuses_confirmation_message() -> None:
    client = _make_client('pending-generic')

    async def _fake_confirmation_message(pending):
        assert pending['name'] == 'start_training'
        return '当前待确认的是启动训练：将使用 yolov8n.pt 训练 10 轮。'

    client._build_confirmation_message = _fake_confirmation_message  # type: ignore[assignment]
    turn = await client.chat('嗯我再想想')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['message'] == '当前待确认的是启动训练：将使用 yolov8n.pt 训练 10 轮。', turn


async def _scenario_prepare_start_like_phrase_routes_to_approve() -> None:
    client = _make_prepare_client('pending-prepare-approve')
    captured: dict[str, object] = {}

    async def _fake_review_pending_action(decision_payload, *, stream_handler=None):
        del stream_handler
        captured.update(dict(decision_payload))
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.review_pending_action = _fake_review_pending_action  # type: ignore[assignment]
    turn = await client.chat('没问题，开始训练')
    assert turn['status'] == 'completed', turn
    assert captured.get('decision') == 'approve', captured
    assert captured.get('raw_user_text') == '没问题，开始训练', captured


async def _scenario_continue_phrase_routes_to_approve() -> None:
    client = _make_client('pending-continue-approve')
    captured: dict[str, object] = {}

    async def _fake_review_pending_action(decision_payload, *, stream_handler=None):
        del stream_handler
        captured.update(dict(decision_payload))
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.review_pending_action = _fake_review_pending_action  # type: ignore[assignment]
    turn = await client.chat('可以，继续')
    assert turn['status'] == 'completed', turn
    assert captured.get('decision') == 'approve', captured
    assert captured.get('raw_user_text') == '可以，继续', captured


async def _scenario_continue_phrase_with_punctuation_routes_to_approve() -> None:
    client = _make_client('pending-continue-approve-punct')
    captured: dict[str, object] = {}

    async def _fake_review_pending_action(decision_payload, *, stream_handler=None):
        del stream_handler
        captured.update(dict(decision_payload))
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.review_pending_action = _fake_review_pending_action  # type: ignore[assignment]
    turn = await client.chat('好，继续。')
    assert turn['status'] == 'completed', turn
    assert captured.get('decision') == 'approve', captured
    assert captured.get('raw_user_text') == '好，继续。', captured


async def _scenario_edit_phrase_marks_pending_as_edit() -> None:
    client = _make_client('pending-edit')
    turn = await client.chat('把 batch 改成 12 再继续')
    pending = client.get_pending_action()
    assert pending is not None, turn
    assert pending['decision_context']['decision'] == 'edit', pending
    assert pending['decision_context']['raw_user_text'] == '把 batch 改成 12 再继续', pending
    assert client.session_state.active_training.training_plan_draft.get('planned_training_args', {}).get('batch') == 12


async def _scenario_clarify_phrase_reuses_confirmation_message() -> None:
    client = _make_prepare_client('pending-clarify')

    async def _fake_confirmation_message(pending):
        assert pending['name'] == 'prepare_dataset_for_training'
        return '当前待确认的是准备数据集：会先生成 data.yaml 再决定是否进入训练。'

    client._build_confirmation_message = _fake_confirmation_message  # type: ignore[assignment]
    turn = await client.chat('为什么这样安排？')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['message'] == '当前待确认的是准备数据集：会先生成 data.yaml 再决定是否进入训练。', turn


async def _scenario_prepare_start_phrase_with_punctuation_routes_to_approve() -> None:
    client = _make_prepare_client('pending-prepare-approve-punct')
    captured: dict[str, object] = {}

    async def _fake_review_pending_action(decision_payload, *, stream_handler=None):
        del stream_handler
        captured.update(dict(decision_payload))
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.review_pending_action = _fake_review_pending_action  # type: ignore[assignment]
    turn = await client.chat('没问题，开始训练吧。')
    assert turn['status'] == 'completed', turn
    assert captured.get('decision') == 'approve', captured
    assert captured.get('raw_user_text') == '没问题，开始训练吧。', captured


async def _scenario_prepare_pending_edit_refreshes_locally() -> None:
    graph = _CountingGraph()
    client = _make_prepare_revision_client('pending-prepare-edit-local', graph=graph)
    turn = await client.chat('把 batch 改成 12 再继续，其他设置不变')
    pending = client.get_pending_action()
    draft = dict(client.session_state.active_training.training_plan_draft or {})
    interrupt_payload = dict(turn.get('interrupt_payload') or {})
    assert turn['status'] == 'needs_confirmation', turn
    assert len(graph.calls) == 0, graph.calls
    assert interrupt_payload.get('type') == 'training_confirmation', turn
    assert interrupt_payload.get('phase') == 'prepare', interrupt_payload
    assert interrupt_payload.get('next_step_tool') == 'prepare_dataset_for_training', interrupt_payload
    assert interrupt_payload.get('next_step_args') == {'dataset_path': '/data/demo', 'force_split': True}, interrupt_payload
    assert draft.get('next_step_tool') == 'prepare_dataset_for_training', draft
    assert (draft.get('planned_training_args') or {}).get('batch') == 12, draft
    if pending is not None:
        assert pending['tool_args'] == {'dataset_path': '/data/demo', 'force_split': True}, pending


async def _scenario_prepare_pending_edit_restored_session_stays_local() -> None:
    scenario_id = 'pending-prepare-edit-restore'
    seed_graph = _CountingGraph()
    seeded = _make_prepare_revision_client(scenario_id, graph=seed_graph)
    seeded.memory.save_state(seeded.session_state)

    restore_graph = _CountingGraph()
    restored = YoloStudioAgentClient(
        graph=restore_graph,
        settings=AgentSettings(session_id=scenario_id, memory_root=str(WORK / scenario_id)),
        tool_registry={},
    )
    turn = await restored.chat('把 batch 改成 12 再继续，其他设置不变')
    pending = restored.get_pending_action()
    draft = dict(restored.session_state.active_training.training_plan_draft or {})
    interrupt_payload = dict(turn.get('interrupt_payload') or {})
    assert turn['status'] == 'needs_confirmation', turn
    assert len(restore_graph.calls) == 0, restore_graph.calls
    assert interrupt_payload.get('type') == 'training_confirmation', turn
    assert interrupt_payload.get('phase') == 'prepare', interrupt_payload
    assert interrupt_payload.get('next_step_tool') == 'prepare_dataset_for_training', interrupt_payload
    assert interrupt_payload.get('next_step_args') == {'dataset_path': '/data/demo', 'force_split': True}, interrupt_payload
    assert (draft.get('planned_training_args') or {}).get('batch') == 12, draft
    if pending is not None:
        assert pending['tool_args'] == {'dataset_path': '/data/demo', 'force_split': True}, pending


async def _scenario_pending_loop_list_passthrough_strips_plan_context() -> None:
    graph = _LoopListGraph()
    client = _make_prepare_revision_client('pending-loop-list-passthrough', graph=graph)
    turn = await client.chat('最近有哪些环训练')
    pending = client.get_pending_action()
    assert turn['status'] == 'completed', turn
    assert turn['message'] == '找到 2 条环训练记录', turn
    assert pending is not None, turn
    assert len(graph.payloads) == 1, graph.payloads
    payload = dict(graph.payloads[0] or {})
    assert payload.get('training_plan_context') is None, payload
    messages = list(payload.get('messages') or [])
    assert messages, messages
    assert getattr(messages[-1], 'content', '') == '最近有哪些环训练', messages


async def _scenario_pending_loop_list_passthrough_beats_clarify_classifier() -> None:
    graph = _LoopListGraph()
    client = _make_prepare_revision_client('pending-loop-list-passthrough-clf', graph=graph)

    async def _fake_classify_confirmation_reply(user_text, pending):
        del user_text, pending
        return 'clarify'

    client._classify_confirmation_reply = _fake_classify_confirmation_reply  # type: ignore[assignment]
    turn = await client.chat('最近有哪些环训练')
    assert turn['status'] == 'completed', turn
    assert turn['message'] == '找到 2 条环训练记录', turn
    assert len(graph.payloads) == 1, graph.payloads


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_status_query_reuses_confirmation_message()
        await _scenario_generic_text_reuses_confirmation_message()
        await _scenario_prepare_start_like_phrase_routes_to_approve()
        await _scenario_continue_phrase_routes_to_approve()
        await _scenario_continue_phrase_with_punctuation_routes_to_approve()
        await _scenario_edit_phrase_marks_pending_as_edit()
        await _scenario_clarify_phrase_reuses_confirmation_message()
        await _scenario_prepare_start_phrase_with_punctuation_routes_to_approve()
        await _scenario_prepare_pending_edit_refreshes_locally()
        await _scenario_prepare_pending_edit_restored_session_stays_local()
        await _scenario_pending_loop_list_passthrough_strips_plan_context()
        await _scenario_pending_loop_list_passthrough_beats_clarify_classifier()
        print('pending confirmation dialogue route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
