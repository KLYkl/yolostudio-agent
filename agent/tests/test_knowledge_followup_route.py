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
        raise AssertionError('create_react_agent should not be called in knowledge followup route tests')

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
        raise AssertionError('knowledge followup route should stay on routed flows, not fallback to graph')


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


WORK = Path(__file__).resolve().parent / '_tmp_knowledge_followup_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    return YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})


async def _scenario_knowledge_followup_routes() -> None:
    client = _make_client('knowledge-followup')
    client.session_state.active_knowledge.last_retrieval = {
        'topic': 'training_metrics',
        'stage': 'post_training',
        'signals': ['high_precision_low_recall'],
        'summary': '知识检索完成: 当前更像高精度低召回。',
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '知识跟进路由器' in text:
            return '{"action":"knowledge","reason":"用户在追问刚才那条知识解释"}'
        return '这类现象通常意味着模型偏保守，需要先排查漏检样本。'

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'retrieve_training_knowledge'
        assert kwargs['topic'] == 'training_metrics'
        assert kwargs['stage'] == 'post_training'
        assert kwargs['signals'] == ['high_precision_low_recall']
        result = {
            'ok': True,
            'summary': '知识检索完成: 当前更像高精度低召回，需要先排查漏检样本。',
            'retrieval_overview': {'topic': 'training_metrics', 'matched_rule_count': 1},
            'action_candidates': [{'tool': 'recommend_next_training_step', 'description': '继续生成下一步建议'}],
            'topic': kwargs['topic'],
            'stage': kwargs['stage'],
            'signals': kwargs['signals'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('刚才那条规则能再详细解释一下吗？')
    assert turn['status'] == 'completed', turn
    assert calls == [('retrieve_training_knowledge', {'topic': 'training_metrics', 'stage': 'post_training', 'model_family': 'yolo', 'task_type': 'detection', 'signals': ['high_precision_low_recall']})], calls
    assert '漏检样本' in turn['message'], turn


async def _scenario_next_step_followup_routes() -> None:
    client = _make_client('knowledge-next-step-followup')
    client.session_state.active_knowledge.last_recommendation = {
        'summary': '建议下一步先处理漏标样本。',
        'recommended_action': 'fix_labels_then_tune',
    }
    client.session_state.active_dataset.last_health_check = {
        'health_overview': {'missing_labels': 2},
        'missing_labels': 2,
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '知识跟进路由器' in text:
            return '{"action":"next_step","reason":"用户在追问建议如何落地"}'
        return '建议先修漏标，再做一轮短周期验证。'

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'summarize_training_run':
            result = {
                'ok': True,
                'summary': '训练摘要: 最近一次训练已完成。',
                'summary_overview': {'run_state': 'completed', 'map50': 0.61},
            }
        elif tool_name == 'recommend_next_training_step':
            result = {
                'ok': True,
                'summary': '建议下一步先修漏标，再做一轮短周期训练验证。',
                'recommendation_overview': {'recommended_action': 'fix_labels_then_quick_iteration'},
                'action_candidates': [{'tool': 'prepare_dataset_for_training', 'description': '先补齐标注后继续训练'}],
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('那这个建议具体怎么落地？')
    assert turn['status'] == 'completed', turn
    assert [name for name, _ in calls] == ['summarize_training_run', 'recommend_next_training_step'], calls
    assert '修漏标' in turn['message'] or '补齐标注' in turn['message'], turn


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_knowledge_followup_routes()
        await _scenario_next_step_followup_routes()
        print('knowledge followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
