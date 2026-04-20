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

        async def ainvoke(self, args):
            if self.coroutine:
                return await self.coroutine(**args)
            if self.func:
                return self.func(**args)
            return args

        def invoke(self, args):
            if self.func:
                return self.func(**args)
            return args

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
        raise AssertionError('create_react_agent should not be called in llm tool selection tests')

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

from langchain_core.messages import AIMessage, ToolMessage
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient


WORK = Path(__file__).resolve().parent / '_tmp_llm_tool_selection'


class _GraphState:
    def __init__(self, *, messages: list[Any], pending_confirmation: dict[str, Any] | None = None):
        self.next = ('tools',) if pending_confirmation else ()
        self.values = {
            'messages': list(messages),
            'pending_confirmation': dict(pending_confirmation) if pending_confirmation else None,
        }
        self.tasks = ()
        self.interrupts = ()


class _GraphWithCompletedTool:
    def __init__(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
        final_reply: str,
    ) -> None:
        self.tool_name = tool_name
        self.tool_args = dict(tool_args)
        self.tool_result = dict(tool_result)
        self.final_reply = final_reply
        self.tool_call_id = f'{tool_name}-call'
        self._state: _GraphState | None = None
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get_state(self, config):
        del config
        return self._state

    async def ainvoke(self, payload, config=None):
        del config
        assert isinstance(payload, dict), payload
        messages = list((payload or {}).get('messages') or [])
        messages.append(
            AIMessage(
                content='',
                tool_calls=[{'id': self.tool_call_id, 'name': self.tool_name, 'args': dict(self.tool_args)}],
            )
        )
        messages.append(
            ToolMessage(
                content=json.dumps(self.tool_result, ensure_ascii=False),
                name=self.tool_name,
                tool_call_id=self.tool_call_id,
            )
        )
        messages.append(AIMessage(content=self.final_reply))
        self._state = _GraphState(messages=messages)
        self.calls.append(('invoke', self.tool_name, dict(self.tool_args)))
        return {'messages': messages}


class _GraphWithApprovalTool:
    def __init__(
        self,
        client: YoloStudioAgentClient,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: dict[str, Any],
        final_reply: str,
    ) -> None:
        self.client = client
        self.tool_name = tool_name
        self.tool_args = dict(tool_args)
        self.tool_result = dict(tool_result)
        self.final_reply = final_reply
        self.tool_call_id = f'{tool_name}-call'
        self._state: _GraphState | None = None
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get_state(self, config):
        del config
        return self._state

    async def ainvoke(self, payload, config=None):
        thread_id = str((((config or {}).get('configurable') or {}).get('thread_id') or '')).strip()
        if isinstance(payload, dict):
            messages = list((payload or {}).get('messages') or [])
            messages.append(
                AIMessage(
                    content='',
                    tool_calls=[{'id': self.tool_call_id, 'name': self.tool_name, 'args': dict(self.tool_args)}],
                )
            )
            pending_payload = self.client._build_pending_action_payload(
                {
                    'name': self.tool_name,
                    'args': dict(self.tool_args),
                    'tool_call_id': self.tool_call_id,
                    'thread_id': thread_id,
                    'source': 'graph',
                    'adapted': False,
                },
                thread_id=thread_id,
            )
            pending_confirmation = {
                'id': self.tool_call_id,
                'tool_call_id': self.tool_call_id,
                'name': self.tool_name,
                'tool_name': self.tool_name,
                'args': dict(self.tool_args),
                'tool_args': dict(self.tool_args),
                'raw_name': self.tool_name,
                'raw_args': dict(self.tool_args),
                'summary': pending_payload['summary'],
                'objective': pending_payload['objective'],
                'allowed_decisions': list(pending_payload['allowed_decisions']),
                'review_config': dict(pending_payload['review_config']),
                'decision_context': dict(pending_payload.get('decision_context') or {}),
                'thread_id': thread_id,
                'source': 'graph',
                'interrupt_kind': pending_payload['interrupt_kind'],
                'adapted': False,
            }
            self._state = _GraphState(messages=messages, pending_confirmation=pending_confirmation)
            self.calls.append(('select', self.tool_name, dict(self.tool_args)))
            return {'messages': messages}

        decision = getattr(payload, 'resume', None)
        decision_value = ''
        if isinstance(decision, dict):
            decision_value = str(decision.get('decision') or '').strip().lower()
        if decision_value != 'approve':
            raise AssertionError(f'unexpected resume payload: {payload!r}')
        if self._state is None:
            raise AssertionError('approval graph resumed before initial selection')
        messages = list(self._state.values.get('messages') or [])
        messages.append(
            ToolMessage(
                content=json.dumps(self.tool_result, ensure_ascii=False),
                name=self.tool_name,
                tool_call_id=self.tool_call_id,
            )
        )
        messages.append(AIMessage(content=self.final_reply))
        self._state = _GraphState(messages=messages)
        self.calls.append(('resume', 'approve', dict(self.tool_args)))
        return {'messages': messages}


def _make_client(session_id: str, graph: Any) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    return YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})


async def _scenario_graph_selects_read_tool() -> None:
    result_payload = {
        'ok': True,
        'summary': '远端 profile 1 个 / SSH alias 1 个。 默认 profile: lab。',
        'profiles_path': '/tmp/remote_profiles.json',
        'default_profile': 'lab',
        'profiles': [{'name': 'lab', 'target_label': 'lab', 'remote_root': '/srv/agent_stage', 'is_default': True}],
        'ssh_aliases': [{'name': 'lab-ssh', 'hostname': 'demo-host', 'port': '22'}],
    }
    graph = _GraphWithCompletedTool(
        tool_name='list_remote_profiles',
        tool_args={},
        tool_result=result_payload,
        final_reply='当前默认可用服务器配置是 lab。',
    )
    client = _make_client('graph-read-tool', graph)

    async def _never_bypass(user_text: str, thread_id: str, *, skip_training_plan_dialogue: bool = False):
        del user_text, thread_id, skip_training_plan_dialogue
        return None

    client._try_handle_mainline_intent = _never_bypass  # type: ignore[assignment]

    turn = await client.chat('先看看有哪些可用服务器配置')
    assert turn['status'] == 'completed', turn
    assert 'lab' in turn['message'], turn
    assert client.session_state.active_remote_transfer.profile_name == 'lab'
    assert client.session_state.active_remote_transfer.remote_root == '/srv/agent_stage'
    routes = client.route_ownership_report()
    assert any(item.get('route') == 'graph-selected-tool' for item in routes), routes
    assert not any(item.get('route') == 'graph-external-bypass' for item in routes), routes
    assert graph.calls == [('invoke', 'list_remote_profiles', {})], graph.calls


async def _scenario_graph_selects_high_risk_tool() -> None:
    upload_root = WORK / 'upload-artifacts'
    upload_root.mkdir(parents=True, exist_ok=True)
    weight_path = upload_root / 'best.pt'
    weight_path.write_text('fake-weight', encoding='utf-8')

    placeholder = _GraphWithCompletedTool(
        tool_name='list_remote_profiles',
        tool_args={},
        tool_result={},
        final_reply='',
    )
    client = _make_client('graph-high-risk-tool', placeholder)
    tool_args = {
        'local_paths': [str(weight_path)],
        'server': 'yolostudio',
        'remote_root': '/tmp/agent_stage',
    }
    tool_result = {
        'ok': True,
        'summary': '远端上传完成：已上传 1 个本地项到 yolostudio:/tmp/agent_stage',
        'target_label': 'yolostudio',
        'profile_name': '',
        'remote_root': '/tmp/agent_stage',
        'uploaded_count': 1,
        'uploaded_items': [
            {
                'local_path': str(weight_path),
                'remote_path': '/tmp/agent_stage/best.pt',
                'item_type': 'file',
            }
        ],
    }
    graph = _GraphWithApprovalTool(
        client,
        tool_name='upload_assets_to_remote',
        tool_args=tool_args,
        tool_result=tool_result,
        final_reply='远端上传完成，结果已写入当前会话状态。',
    )
    client.graph = graph

    turn = await client.chat(f'把 "{weight_path}" 上传到服务器 yolostudio 的 /tmp/agent_stage')
    assert turn['status'] == 'needs_confirmation', turn
    assert turn['tool_call']['name'] == 'upload_assets_to_remote', turn
    assert turn['tool_call']['args']['remote_root'] == '/tmp/agent_stage', turn
    routes = client.route_ownership_report()
    assert any(item.get('route') == 'graph-selected-tool' for item in routes), routes
    assert not any(item.get('route') == 'graph-external-bypass' for item in routes), routes

    done = await client.confirm(turn['thread_id'], True)
    assert done['status'] == 'completed', done
    assert '远端上传完成' in done['message'], done
    assert client.session_state.active_remote_transfer.remote_root == '/tmp/agent_stage'
    assert client.session_state.active_remote_transfer.last_upload['uploaded_count'] == 1
    assert client.get_pending_action() is None
    assert graph.calls == [
        ('select', 'upload_assets_to_remote', tool_args),
        ('resume', 'approve', tool_args),
    ], graph.calls


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_graph_selects_read_tool()
        await _scenario_graph_selects_high_risk_tool()
        print('llm tool selection deterministic ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
