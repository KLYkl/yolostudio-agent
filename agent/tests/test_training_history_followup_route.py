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
        raise AssertionError('create_react_agent should not be called in training history followup tests')

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

from langchain_core.messages import AIMessage
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient, _build_agent_post_model_hook


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('training history followup route should stay on routed flows, not fallback to graph')


class _HookedToolCallGraph:
    def __init__(
        self,
        *,
        planner_llm,
        tool_name: str,
        tool_args: dict[str, object] | None = None,
    ) -> None:
        self._tool_name = tool_name
        self._tool_args = dict(tool_args or {})
        self._hook = _build_agent_post_model_hook(planner_llm)

    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        messages.append(AIMessage(content='', tool_calls=[{'id': 'tc-1', 'name': self._tool_name, 'args': dict(self._tool_args)}]))
        hook_state = dict(payload)
        hook_state['messages'] = messages
        update = await self._hook(hook_state)
        updated_messages = list(update.get('messages') or [])
        if updated_messages and getattr(updated_messages[0], 'id', '') == '__remove_all__':
            updated_messages = updated_messages[1:]
        return {'messages': updated_messages or messages}


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


WORK = Path(__file__).resolve().parent / '_tmp_training_history_followup_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    return YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})


async def _scenario_best_followup_routes() -> None:
    client = _make_client('best-followup')
    client.session_state.active_training.best_run_selection = {
        'ok': True,
        'summary': '最佳训练记录已选出',
        'best_run_id': 'run-best',
        'ranking_basis': 'mAP50',
        'best_run': {'run_id': 'run-best', 'run_state': 'completed'},
        'action_candidates': [
            {'description': '继续查看最佳训练记录详情', 'tool': 'inspect_training_run'},
        ],
    }

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '训练历史跟进路由器' in text:
            return '{"action":"best","reason":"用户在追问最佳训练"}'
        if '结果说明器' in text:
            return '当前最佳训练是 run-best，建议继续查看这条训练记录的详细指标。'
        return '当前最佳训练是 run-best。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('training history followup should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.graph = _HookedToolCallGraph(
        planner_llm=client.planner_llm,
        tool_name='select_best_training_run',
    )
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('那个最佳训练详细一点呢？')
    assert turn['status'] == 'completed', turn
    assert 'run-best' in turn['message'], turn


async def _scenario_compare_followup_routes() -> None:
    client = _make_client('compare-followup')
    client.session_state.active_training.last_run_comparison = {
        'ok': True,
        'summary': '训练记录对比已完成',
        'left_run_id': 'run-a',
        'right_run_id': 'run-b',
        'highlights': ['mAP50 提升 +0.0300'],
        'action_candidates': [
            {'description': '继续查看更优训练记录', 'tool': 'select_best_training_run'},
        ],
    }

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '训练历史跟进路由器' in text:
            return '{"action":"compare","reason":"用户在追问刚才的训练对比结论"}'
        if '结果说明器' in text:
            return '刚才对比的是 run-a 和 run-b，其中 run-b 的表现更好，mAP50 提升了 0.0300。'
        return '训练对比已完成。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('training history followup should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.graph = _HookedToolCallGraph(
        planner_llm=client.planner_llm,
        tool_name='compare_training_runs',
    )
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('刚才那两个训练对比结论再详细一点')
    assert turn['status'] == 'completed', turn
    assert 'run-a' in turn['message'] and 'run-b' in turn['message'], turn


async def _scenario_runs_followup_routes() -> None:
    client = _make_client('runs-followup')
    client.session_state.active_training.recent_runs = [
        {
            'run_id': 'run-a',
            'run_state': 'completed',
            'observation_stage': 'final',
            'progress': {'epoch': 10, 'total_epochs': 10},
        },
        {
            'run_id': 'run-b',
            'run_state': 'failed',
            'observation_stage': 'mid',
            'progress': {'epoch': 4, 'total_epochs': 10},
        },
    ]

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '训练历史跟进路由器' in text:
            return '{"action":"runs","reason":"用户在追问刚才的训练列表"}'
        if '结果说明器' in text:
            return '最近训练里有 run-a 和 run-b，其中 run-a 已完成，run-b 还没跑好。'
        return '训练历史查询完成。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('training history followup should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.graph = _HookedToolCallGraph(
        planner_llm=client.planner_llm,
        tool_name='list_training_runs',
    )
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('把刚才那些训练记录再概括一下')
    assert turn['status'] == 'completed', turn
    assert 'run-a' in turn['message'] and 'run-b' in turn['message'], turn


async def _scenario_cached_best_request_reuses_state() -> None:
    client = _make_client('best-request-cached')
    client.session_state.active_training.best_run_selection = {
        'ok': True,
        'summary': '最佳训练记录已选出',
        'best_run_id': 'run-best',
        'ranking_basis': 'mAP50',
        'best_run': {'run_id': 'run-best', 'run_state': 'completed'},
        'action_candidates': [{'description': '继续查看最佳训练记录详情', 'tool': 'inspect_training_run'}],
    }

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '结果说明器' in text:
            return '当前最佳训练是 run-best，建议继续查看这条训练记录的详细指标。'
        return '当前最佳训练是 run-best。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('cached best request should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.graph = _HookedToolCallGraph(
        planner_llm=client.planner_llm,
        tool_name='select_best_training_run',
    )
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('哪次训练最好？')
    assert turn['status'] == 'completed', turn
    assert 'run-best' in turn['message'], turn


async def _scenario_cached_run_list_request_reuses_state() -> None:
    client = _make_client('run-list-request-cached')
    client.session_state.active_training.recent_runs = [
        {
            'run_id': 'run-a',
            'run_state': 'completed',
            'observation_stage': 'final',
            'progress': {'epoch': 10, 'total_epochs': 10},
        },
        {
            'run_id': 'run-b',
            'run_state': 'failed',
            'observation_stage': 'mid',
            'progress': {'epoch': 4, 'total_epochs': 10},
        },
    ]

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '结果说明器' in text:
            return '最近训练里有 run-a 和 run-b，其中 run-a 已完成，run-b 失败。'
        return '训练历史查询完成。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('cached run list request should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.graph = _HookedToolCallGraph(
        planner_llm=client.planner_llm,
        tool_name='list_training_runs',
    )
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('列出训练记录')
    assert turn['status'] == 'completed', turn
    assert 'run-a' in turn['message'] and 'run-b' in turn['message'], turn


async def _scenario_cached_run_inspection_request_reuses_state() -> None:
    client = _make_client('run-inspection-request-cached')
    client.session_state.active_training.last_run_inspection = {
        'ok': True,
        'summary': '训练记录详情已就绪',
        'selected_run_id': 'train_log_run_a',
        'run_state': 'completed',
        'summary_overview': {'run_id': 'train_log_run_a', 'run_state': 'completed'},
        'action_candidates': [{'tool': 'compare_training_runs', 'description': '可继续与其他训练对比'}],
    }

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '结果说明器' in text:
            return '训练 train_log_run_a 已完成，可以继续和其他训练记录对比。'
        return '训练记录详情已就绪。'

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('cached run inspection should render from state, not call direct_tool')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.graph = _HookedToolCallGraph(
        planner_llm=client.planner_llm,
        tool_name='inspect_training_run',
        tool_args={'run_id': 'train_log_run_a'},
    )
    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('查看 train_log_run_a 训练详情')
    assert turn['status'] == 'completed', turn
    assert 'train_log_run_a' in turn['message'], turn


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_best_followup_routes()
        await _scenario_compare_followup_routes()
        await _scenario_runs_followup_routes()
        await _scenario_cached_best_request_reuses_state()
        await _scenario_cached_run_list_request_reuses_state()
        await _scenario_cached_run_inspection_request_reuses_state()
        print('training history followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
