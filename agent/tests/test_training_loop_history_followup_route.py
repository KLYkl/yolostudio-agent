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
        raise AssertionError('create_react_agent should not be called in training loop history followup tests')

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


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('training loop history followup should stay on routed flows, not fallback to graph')


class _FakePlannerResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakePlannerLlm:
    def __init__(self, reply) -> None:
        self.reply = reply

    async def ainvoke(self, messages):
        if callable(self.reply):
            return _FakePlannerResponse(self.reply(messages))
        return _FakePlannerResponse(self.reply)


WORK = Path(__file__).resolve().parent / '_tmp_training_loop_history_followup_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    return YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})


async def _scenario_loop_list_followup_routes() -> None:
    client = _make_client('loop-list-followup')
    client.session_state.active_training.recent_loops = [
        {'loop_id': 'loop-a', 'loop_name': 'helmet-loop', 'status': 'completed', 'active': False},
        {'loop_id': 'loop-b', 'loop_name': 'vest-loop', 'status': 'awaiting_review', 'active': False},
    ]

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '环训练历史跟进路由器' in text:
            return '{"action":"list","reason":"用户在追问刚才那些环训练"}'
        if '结果说明器' in text:
            return '最近环训练包括 helmet-loop 和 vest-loop，其中 vest-loop 还在等待审阅。'
        return '环训练列表已就绪。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('training loop history followup should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('把刚才那些环训练再概括一下')
    assert turn['status'] == 'completed', turn
    assert 'helmet-loop' in turn['message'] and 'vest-loop' in turn['message'], turn


async def _scenario_loop_inspect_followup_routes() -> None:
    client = _make_client('loop-inspect-followup')
    client.session_state.active_training.last_loop_detail = {
        'ok': True,
        'summary': '第 2 轮训练已完成，准备下一轮',
        'loop_id': 'loop-b',
        'loop_name': 'vest-loop',
        'status': 'awaiting_review',
        'current_round_index': 2,
        'max_rounds': 5,
        'knowledge_gate_status': {
            'outcome': 'awaiting_review',
            'action_label': '先做误差分析',
            'summary': '本轮建议先做误差分析。',
        },
        'latest_round_card': {
            'round_index': 2,
            'status': 'completed',
            'knowledge_gate': {
                'action_label': '先做误差分析',
                'outcome': 'awaiting_review',
                'outcome_label': '等待审阅',
            },
        },
    }

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '环训练历史跟进路由器' in text:
            return '{"action":"inspect","reason":"用户在追问刚才那个环训练详情"}'
        if '结果说明器' in text:
            return '刚才那个环训练是 vest-loop，现在停在等待审阅，建议先做误差分析。'
        return '环训练详情已就绪。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('training loop history followup should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('那个环训练详细一点呢')
    assert turn['status'] == 'completed', turn
    assert 'vest-loop' in turn['message'], turn
    assert '误差分析' in turn['message'], turn


async def _scenario_cached_loop_list_request_reuses_state() -> None:
    client = _make_client('loop-list-request-cached')
    client.session_state.active_training.recent_loops = [
        {'loop_id': 'loop-a', 'loop_name': 'helmet-loop', 'status': 'completed', 'active': False},
        {'loop_id': 'loop-b', 'loop_name': 'vest-loop', 'status': 'awaiting_review', 'active': False},
    ]

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '结果说明器' in text:
            return '最近环训练包括 helmet-loop 和 vest-loop，其中 vest-loop 还在等待审阅。'
        return '环训练列表已就绪。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('cached loop list request should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('列一下环训练列表')
    assert turn['status'] == 'completed', turn
    assert 'helmet-loop' in turn['message'] and 'vest-loop' in turn['message'], turn


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_loop_list_followup_routes()
        await _scenario_loop_inspect_followup_routes()
        await _scenario_cached_loop_list_request_reuses_state()
        print('training loop history followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
