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
from langchain_core.messages import AIMessage, ToolMessage


class _GraphState:
    def __init__(self, messages):
        self.values = {'messages': list(messages)}
        self.next = ()


class _KnowledgeGraph:
    def __init__(self) -> None:
        self.client: YoloStudioAgentClient | None = None
        self._last_state = _GraphState([])

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        return self._last_state

    async def _cached_reply(self, payload, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        reply = await self.client._render_tool_result_message(tool_name, result)
        if not reply:
            reply = str(result.get('summary') or result.get('error') or '操作已完成')
        messages = list(payload['messages']) + [AIMessage(content=reply)]
        self._last_state = _GraphState(messages)
        return {'messages': messages}

    async def _tool_reply(self, payload, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        assert self.client is not None
        observed = await self.client.direct_tool(tool_name, _state_mode='observe', **kwargs)
        reply = await self.client._render_tool_result_message(tool_name, observed)
        if not reply:
            reply = str(observed.get('summary') or observed.get('error') or '操作已完成')
        tool_call = {'id': f'tc-{len(payload["messages"])}', 'name': tool_name, 'args': kwargs}
        messages = list(payload['messages']) + [
            AIMessage(content='', tool_calls=[tool_call]),
            ToolMessage(content=json.dumps(observed, ensure_ascii=False), name=tool_name, tool_call_id=tool_call['id']),
            AIMessage(content=reply),
        ]
        self._last_state = _GraphState(messages)
        return {'messages': messages}

    async def _next_step_reply(self, payload) -> dict[str, Any]:
        assert self.client is not None
        summary_result = await self.client.direct_tool('summarize_training_run', _state_mode='observe')
        recommendation_result = await self.client.direct_tool(
            'recommend_next_training_step',
            _state_mode='observe',
            readiness=self.client.session_state.active_dataset.last_readiness,
            health=self.client.session_state.active_dataset.last_health_check,
            status=summary_result,
            comparison=self.client.session_state.active_training.last_run_comparison,
            prediction_summary=self.client.session_state.active_prediction.last_result,
            model_family='yolo',
            task_type='detection',
        )
        reply = await self.client._render_multi_tool_result_message(
            [
                ('summarize_training_run', summary_result),
                ('recommend_next_training_step', recommendation_result),
            ],
            objective='下一步训练建议说明',
        )
        if not reply:
            reply = str(recommendation_result.get('summary') or summary_result.get('summary') or recommendation_result.get('error') or '下一步建议已生成')
        first_call = {'id': f'tc-{len(payload["messages"])}', 'name': 'summarize_training_run', 'args': {}}
        second_call = {
            'id': f'tc-{len(payload["messages"]) + 1}',
            'name': 'recommend_next_training_step',
            'args': {
                'readiness': self.client.session_state.active_dataset.last_readiness,
                'health': self.client.session_state.active_dataset.last_health_check,
                'status': summary_result,
                'comparison': self.client.session_state.active_training.last_run_comparison,
                'prediction_summary': self.client.session_state.active_prediction.last_result,
                'model_family': 'yolo',
                'task_type': 'detection',
            },
        }
        messages = list(payload['messages']) + [
            AIMessage(content='', tool_calls=[first_call]),
            ToolMessage(content=json.dumps(summary_result, ensure_ascii=False), name='summarize_training_run', tool_call_id=first_call['id']),
            AIMessage(content='', tool_calls=[second_call]),
            ToolMessage(content=json.dumps(recommendation_result, ensure_ascii=False), name='recommend_next_training_step', tool_call_id=second_call['id']),
            AIMessage(content=reply),
        ]
        self._last_state = _GraphState(messages)
        return {'messages': messages}

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        summary = '\n'.join(str(getattr(message, 'content', message)) for message in payload['messages'][:2])
        user_text = str(getattr(payload['messages'][-1], 'content', ''))
        kn = self.client.session_state.active_knowledge
        if '规则' in user_text or '知识' in user_text:
            assert 'retrieval_topic:' in summary
            assert 'retrieval_stage:' in summary
            if (kn.last_retrieval or {}).get('retrieval_overview'):
                return await self._cached_reply(payload, 'retrieve_training_knowledge', dict(kn.last_retrieval))
            retrieval = kn.last_retrieval or {}
            return await self._tool_reply(
                payload,
                'retrieve_training_knowledge',
                topic=str(retrieval.get('topic') or 'workflow'),
                stage=str(retrieval.get('stage') or 'post_training'),
                model_family='yolo',
                task_type='detection',
                signals=list(retrieval.get('signals') or []),
            )
        if '怎么落地' in user_text:
            assert 'recommended_action:' in summary
            return await self._next_step_reply(payload)
        raise AssertionError(f'unexpected graph request: {user_text}')


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
    graph = _KnowledgeGraph()
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    graph.bind(client)
    return client


async def _scenario_knowledge_followup_routes() -> None:
    client = _make_client('knowledge-followup')
    client.session_state.active_knowledge.last_retrieval = {
        'topic': 'training_metrics',
        'stage': 'post_training',
        'signals': ['high_precision_low_recall'],
        'summary': '知识检索完成: 当前更像高精度低召回。',
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        state_mode = str(kwargs.pop('_state_mode', 'persistent'))
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
        if state_mode != 'observe':
            client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('刚才那条规则能再详细解释一下吗？')
    assert turn['status'] == 'completed', turn
    assert calls == [('retrieve_training_knowledge', {'topic': 'training_metrics', 'stage': 'post_training', 'model_family': 'yolo', 'task_type': 'detection', 'signals': ['high_precision_low_recall']})], calls
    assert '漏检样本' in turn['message'], turn
    assert client.session_state.active_knowledge.last_retrieval.get('signals') == ['high_precision_low_recall']


async def _scenario_cached_knowledge_followup_reuses_state() -> None:
    client = _make_client('knowledge-followup-cached')
    client.session_state.active_knowledge.last_retrieval = {
        'summary': '知识检索完成: 当前更像高精度低召回，需要先排查漏检样本。',
        'retrieval_overview': {'topic': 'training_metrics', 'matched_rule_count': 1},
        'action_candidates': [{'tool': 'recommend_next_training_step', 'description': '继续生成下一步建议'}],
        'topic': 'training_metrics',
        'stage': 'post_training',
        'signals': ['high_precision_low_recall'],
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'cached knowledge followup should not call direct tool: {tool_name}')

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('刚才那条规则再详细一点')
    assert turn['status'] == 'completed', turn
    assert '高精度低召回' in turn['message'], turn
    assert not calls, calls


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

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        state_mode = str(kwargs.pop('_state_mode', 'persistent'))
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
        if state_mode != 'observe':
            client._apply_to_state(tool_name, result, kwargs)
        return result

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
        await _scenario_cached_knowledge_followup_reuses_state()
        await _scenario_next_step_followup_routes()
        print('knowledge followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
