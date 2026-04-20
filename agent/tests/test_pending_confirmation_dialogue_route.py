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


def _make_client(
    session_id: str,
    *,
    graph=None,
    tool_name: str = 'upload_assets_to_remote',
    tool_args: dict[str, object] | None = None,
) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=graph or _NoLLMGraph(), settings=settings, tool_registry={})
    pending = {
        'name': tool_name,
        'args': dict(tool_args or {'server': 'lab', 'local_paths': ['/tmp/demo.pt']}),
        'id': None,
        'synthetic': True,
    }
    client._set_pending_confirmation(f'{session_id}-pending', pending)
    return client


async def _scenario_status_query_reuses_confirmation_message() -> None:
    client = _make_client('pending-status')

    async def _fake_confirmation_message(pending):
        assert pending['name'] == 'upload_assets_to_remote'
        return '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。'

    client._build_confirmation_message = _fake_confirmation_message  # type: ignore[assignment]
    turn = await client.chat('查看训练状态')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['message'] == '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。', turn


async def _scenario_generic_text_reuses_confirmation_message() -> None:
    client = _make_client('pending-generic')

    async def _fake_confirmation_message(pending):
        assert pending['name'] == 'upload_assets_to_remote'
        return '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。'

    client._build_confirmation_message = _fake_confirmation_message  # type: ignore[assignment]
    turn = await client.chat('嗯我再想想')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['message'] == '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。', turn


async def _scenario_execute_phrase_routes_to_approve() -> None:
    client = _make_client('pending-execute-approve')
    captured: dict[str, object] = {}

    async def _fake_confirm(thread_id, approved, stream_handler=None):
        del stream_handler
        captured['thread_id'] = thread_id
        captured['approved'] = approved
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.confirm = _fake_confirm  # type: ignore[assignment]
    turn = await client.chat('没问题，执行吧')
    assert turn['status'] == 'completed', turn
    assert captured.get('approved') is True, captured
    pending = client.get_pending_action()
    assert pending is not None, pending
    assert pending['decision_context']['decision'] == 'approve', pending
    assert pending['decision_context']['raw_user_text'] == '没问题，执行吧', pending


async def _scenario_continue_phrase_routes_to_approve() -> None:
    client = _make_client('pending-continue-approve')
    captured: dict[str, object] = {}

    async def _fake_confirm(thread_id, approved, stream_handler=None):
        del stream_handler
        captured['thread_id'] = thread_id
        captured['approved'] = approved
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.confirm = _fake_confirm  # type: ignore[assignment]
    turn = await client.chat('可以，继续')
    assert turn['status'] == 'completed', turn
    assert captured.get('approved') is True, captured
    pending = client.get_pending_action()
    assert pending is not None, pending
    assert pending['decision_context']['decision'] == 'approve', pending
    assert pending['decision_context']['raw_user_text'] == '可以，继续', pending


async def _scenario_continue_phrase_with_punctuation_routes_to_approve() -> None:
    client = _make_client('pending-continue-approve-punct')
    captured: dict[str, object] = {}

    async def _fake_confirm(thread_id, approved, stream_handler=None):
        del stream_handler
        captured['thread_id'] = thread_id
        captured['approved'] = approved
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.confirm = _fake_confirm  # type: ignore[assignment]
    turn = await client.chat('好，继续。')
    assert turn['status'] == 'completed', turn
    assert captured.get('approved') is True, captured
    pending = client.get_pending_action()
    assert pending is not None, pending
    assert pending['decision_context']['decision'] == 'approve', pending
    assert pending['decision_context']['raw_user_text'] == '好，继续。', pending


async def _scenario_edit_phrase_marks_pending_as_edit() -> None:
    client = _make_client('pending-edit')
    turn = await client.chat('把目标服务器改成 prod')
    pending = client.get_pending_action()
    assert pending is not None, turn
    assert pending['decision_context']['decision'] == 'edit', pending
    assert pending['decision_context']['raw_user_text'] == '把目标服务器改成 prod', pending


async def _scenario_clarify_phrase_reuses_confirmation_message() -> None:
    client = _make_client('pending-clarify')

    async def _fake_confirmation_message(pending):
        assert pending['name'] == 'upload_assets_to_remote'
        return '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。'

    client._build_confirmation_message = _fake_confirmation_message  # type: ignore[assignment]
    turn = await client.chat('为什么这样安排？')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['message'] == '当前待确认的是上传素材：会把 /tmp/demo.pt 上传到 lab。', turn


async def _scenario_execute_phrase_with_punctuation_routes_to_approve() -> None:
    client = _make_client('pending-execute-approve-punct')
    captured: dict[str, object] = {}

    async def _fake_confirm(thread_id, approved, stream_handler=None):
        del stream_handler
        captured['thread_id'] = thread_id
        captured['approved'] = approved
        return {'status': 'completed', 'message': 'captured-approve', 'tool_call': None}

    client.confirm = _fake_confirm  # type: ignore[assignment]
    turn = await client.chat('没问题，执行吧。')
    assert turn['status'] == 'completed', turn
    assert captured.get('approved') is True, captured
    pending = client.get_pending_action()
    assert pending is not None, pending
    assert pending['decision_context']['decision'] == 'approve', pending
    assert pending['decision_context']['raw_user_text'] == '没问题，执行吧。', pending


async def _scenario_non_training_pending_passthrough_routes_to_graph() -> None:
    graph = _LoopListGraph()
    client = _make_client('pending-passthrough', graph=graph)
    turn = await client.chat('最近有哪些环训练')
    pending = client.get_pending_action()
    assert turn['status'] == 'completed', turn
    assert turn['message'] == '找到 2 条环训练记录', turn
    assert pending is not None, turn
    assert pending['tool_name'] == 'upload_assets_to_remote', pending
    assert len(graph.payloads) == 1, graph.payloads
    payload = dict(graph.payloads[0] or {})
    assert payload.get('training_plan_context') is None, payload
    messages = list(payload.get('messages') or [])
    assert messages, messages
    assert getattr(messages[-1], 'content', '') == '最近有哪些环训练', messages


async def _scenario_pending_passthrough_beats_clarify_classifier() -> None:
    graph = _LoopListGraph()
    client = _make_client('pending-passthrough-clf', graph=graph)

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
        await _scenario_execute_phrase_routes_to_approve()
        await _scenario_continue_phrase_routes_to_approve()
        await _scenario_continue_phrase_with_punctuation_routes_to_approve()
        await _scenario_edit_phrase_marks_pending_as_edit()
        await _scenario_clarify_phrase_reuses_confirmation_message()
        await _scenario_execute_phrase_with_punctuation_routes_to_approve()
        await _scenario_non_training_pending_passthrough_routes_to_graph()
        await _scenario_pending_passthrough_beats_clarify_classifier()
        print('pending confirmation dialogue route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
