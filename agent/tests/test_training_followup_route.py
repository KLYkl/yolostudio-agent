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
        raise AssertionError('create_react_agent should not be called in training followup route tests')

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

import json

from langchain_core.messages import AIMessage, ToolMessage
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.tests._post_model_hook_support import HookedToolCallGraph


class _NoLLMGraph:
    def get_state(self, config):
        return None

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError('training followup route should stay on routed flows, not fallback to graph')


class _StaticReplyGraph:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        return {'messages': messages + [AIMessage(content=self.reply)]}


class _ObservedToolGraph:
    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name
        self.client: YoloStudioAgentClient | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def bind(self, client: YoloStudioAgentClient) -> None:
        self.client = client

    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, config=None):
        del config
        assert self.client is not None
        messages = list(payload['messages'])
        self.calls.append((self._tool_name, {}))
        result = await self.client.direct_tool(self._tool_name)
        reply = await self.client._render_tool_result_message(self._tool_name, result)
        if not reply:
            reply = str(result.get('summary') or result.get('error') or '操作已完成')
        tool_call = {'id': 'tc-1', 'name': self._tool_name, 'args': {}}
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[tool_call]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name=self._tool_name, tool_call_id='tc-1'),
                AIMessage(content=reply),
            ]
        }


class _TrainingNextStepGraph:
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
        summary_result = await self.client.direct_tool('summarize_training_run')
        recommendation_result = await self.client.direct_tool(
            'recommend_next_training_step',
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
            reply = str(recommendation_result.get('summary') or summary_result.get('summary') or '下一步建议已生成')
        first_call = {'id': 'tc-1', 'name': 'summarize_training_run', 'args': {}}
        second_args = {
            'readiness': self.client.session_state.active_dataset.last_readiness,
            'health': self.client.session_state.active_dataset.last_health_check,
            'status': summary_result,
            'comparison': self.client.session_state.active_training.last_run_comparison,
            'prediction_summary': self.client.session_state.active_prediction.last_result,
            'model_family': 'yolo',
            'task_type': 'detection',
        }
        second_call = {'id': 'tc-2', 'name': 'recommend_next_training_step', 'args': second_args}
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[first_call]),
                ToolMessage(content=json.dumps(summary_result, ensure_ascii=False), name='summarize_training_run', tool_call_id='tc-1'),
                AIMessage(content='', tool_calls=[second_call]),
                ToolMessage(content=json.dumps(recommendation_result, ensure_ascii=False), name='recommend_next_training_step', tool_call_id='tc-2'),
                AIMessage(content=reply),
            ]
        }


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


WORK = Path(__file__).resolve().parent / '_tmp_training_followup_route'


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    client = YoloStudioAgentClient(graph=_NoLLMGraph(), settings=settings, tool_registry={})
    return client


async def _scenario_status_followup_routes() -> None:
    client = _make_client('status-followup')
    client.session_state.active_training.running = True
    client.session_state.active_training.model = 'demo.pt'
    client.session_state.active_training.data_yaml = '/data/data.yaml'
    client.session_state.active_training.last_status = {
        'summary': '训练运行中: epoch 4/10, map50=0.58',
        'run_state': 'running',
        'latest_metrics': {'epoch': 4, 'map50': 0.58},
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '训练跟进路由器' in text:
            return '{"action":"status","reason":"用户在追问当前训练状态"}'
        return '训练仍在运行，当前已到 epoch 6/10。'

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'check_training_status'
        result = {
            'ok': True,
            'running': True,
            'run_state': 'running',
            'summary': '训练运行中: epoch 6/10, map50=0.63',
            'pid': 24680,
            'log_file': '/runs/train_log_live.txt',
            'latest_metrics': {'epoch': 6, 'map50': 0.63},
            'minimum_facts_ready': True,
            'status_overview': {'run_state': 'running', 'epoch': 6, 'map50': 0.63},
            'action_candidates': [{'tool': 'summarize_training_run', 'description': '可查看阶段性训练摘要'}],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    graph = _ObservedToolGraph('check_training_status')
    graph.bind(client)
    client.graph = graph
    turn = await client.chat('现在是什么情况了？我需要详细一点的训练信息')
    assert turn['status'] == 'completed', turn
    assert calls == [('check_training_status', {})], calls
    assert 'epoch 6/10' in turn['message'], turn


async def _scenario_generic_followup_after_start_uses_training_context() -> None:
    client = _make_client('generic-followup-after-start')
    client.session_state.active_training.running = True
    client.session_state.active_training.model = 'demo.pt'
    client.session_state.active_training.data_yaml = '/data/data.yaml'
    client.session_state.active_training.last_start_result = {
        'ok': True,
        'summary': '训练已启动: model=demo.pt, data=/data/data.yaml',
        'pid': 13579,
        'resolved_args': {
            'model': 'demo.pt',
            'data_yaml': '/data/data.yaml',
            'epochs': 10,
        },
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '训练跟进路由器' in text:
            return '{"action":"status","reason":"用户在追问刚启动的训练现状"}'
        return '训练仍在运行，当前已到 epoch 2/10。'

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'check_training_status'
        result = {
            'ok': True,
            'running': True,
            'run_state': 'running',
            'summary': '训练运行中: epoch 2/10, map50=0.41',
            'pid': 13579,
            'log_file': '/runs/train_log_live.txt',
            'latest_metrics': {'epoch': 2, 'map50': 0.41},
            'minimum_facts_ready': True,
            'status_overview': {'run_state': 'running', 'epoch': 2, 'map50': 0.41},
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    graph = _ObservedToolGraph('check_training_status')
    graph.bind(client)
    client.graph = graph
    turn = await client.chat('现在什么情况了？详细一点。')
    assert turn['status'] == 'completed', turn
    assert calls == [('check_training_status', {})], calls
    assert 'epoch 2/10' in turn['message'], turn


async def _scenario_next_step_followup_routes() -> None:
    client = _make_client('next-step-followup')
    client.session_state.active_training.running = False
    client.session_state.active_training.model = 'demo.pt'
    client.session_state.active_training.data_yaml = '/data/data.yaml'
    client.session_state.active_training.last_status = {
        'summary': '训练已完成: map50=0.61',
        'run_state': 'completed',
        'latest_metrics': {'epoch': 10, 'map50': 0.61},
    }
    client.session_state.active_dataset.last_readiness = {'ready_to_start': True, 'summary': '数据已可直接训练'}
    client.session_state.active_dataset.last_health_check = {'missing_labels': 3, 'health_overview': {'missing_labels': 3}}
    calls: list[tuple[str, dict[str, Any]]] = []

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '训练跟进路由器' in text:
            return '{"action":"next_step","reason":"用户在问下一步怎么优化"}'
        return '建议下一步优先检查漏标样本，并在下一轮适度调整训练参数。'

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        if tool_name == 'summarize_training_run':
            result = {
                'ok': True,
                'summary': '训练摘要: epoch 10/10, map50=0.61',
                'summary_overview': {'epoch': 10, 'map50': 0.61},
                'latest_metrics': {'epoch': 10, 'map50': 0.61},
            }
        elif tool_name == 'recommend_next_training_step':
            result = {
                'ok': True,
                'summary': '建议下一步先处理漏标样本，再小幅调整训练参数。',
                'recommendation_overview': {'recommended_action': 'fix_labels_then_tune'},
                'action_candidates': [{'tool': 'prepare_dataset_for_training', 'description': '先补齐标注后再继续训练'}],
            }
        else:
            raise AssertionError(f'unexpected tool call: {tool_name}')
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    graph = _TrainingNextStepGraph()
    graph.bind(client)
    client.graph = graph
    turn = await client.chat('那下一步怎么优化？')
    assert turn['status'] == 'completed', turn
    assert [name for name, _ in calls] == ['summarize_training_run', 'recommend_next_training_step'], calls
    assert '漏标样本' in turn['message'], turn


async def _scenario_cached_next_step_followup_reuses_state() -> None:
    client = _make_client('next-step-followup-cached')
    client.session_state.active_training.running = False
    client.session_state.active_training.last_status = {
        'summary': '训练已完成: map50=0.61',
        'run_state': 'completed',
    }
    client.session_state.active_knowledge.last_recommendation = {
        'summary': '建议下一步先修漏标样本，再小幅调整训练参数。',
        'recommendation_overview': {'recommended_action': 'fix_labels_then_tune'},
        'action_candidates': [{'tool': 'prepare_dataset_for_training', 'description': '先补齐标注后再继续训练'}],
        'recommended_action': 'fix_labels_then_tune',
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def _planner_reply(messages) -> str:
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '训练跟进路由器' in text:
            return '{"action":"next_step","reason":"用户在追问已缓存的下一步建议"}'
        return '建议下一步先修漏标样本，再小幅调整训练参数。'

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f'cached training followup should not call direct tool: {tool_name}')

    client.planner_llm = _FakePlannerLlm(_planner_reply)  # type: ignore[assignment]
    client.graph = HookedToolCallGraph(
        planner_llm=client.planner_llm,
        tool_name='recommend_next_training_step',
    )
    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    turn = await client.chat('那下一步怎么优化得更具体一点？')
    assert turn['status'] == 'completed', turn
    assert '漏标样本' in turn['message'], turn
    assert not calls, calls


async def _scenario_cached_explicit_next_step_request_reuses_state() -> None:
    client = _make_client('next-step-request-cached')
    client.graph = _StaticReplyGraph('建议下一步先修漏标样本，再小幅调整训练参数。')
    client.session_state.active_training.running = False
    client.session_state.active_training.last_status = {
        'summary': '训练已完成: map50=0.61',
        'run_state': 'completed',
    }
    client.session_state.active_knowledge.last_recommendation = {
        'summary': '建议下一步先修漏标样本，再小幅调整训练参数。',
        'recommendation_overview': {'recommended_action': 'fix_labels_then_tune'},
        'action_candidates': [{'tool': 'prepare_dataset_for_training', 'description': '先补齐标注后再继续训练'}],
        'recommended_action': 'fix_labels_then_tune',
    }

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('cached explicit next-step request should render from state, not call direct_tool')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('给我训练下一步建议')
    assert turn['status'] == 'completed', turn
    assert '漏标样本' in turn['message'], turn


async def _scenario_cached_explicit_status_request_reuses_state() -> None:
    client = _make_client('status-request-cached')
    client.session_state.active_training.running = False
    client.session_state.active_training.last_status = {
        'summary': '训练已完成: epoch 10/10, map50=0.61',
        'run_state': 'completed',
        'status_overview': {'run_state': 'completed', 'epoch': 10, 'total_epochs': 10, 'save_dir': '/tmp/demo'},
        'latest_metrics': {'metrics': {'precision': 0.6, 'recall': 0.7, 'map50': 0.61, 'map': 0.5}},
        'action_candidates': [{'tool': 'summarize_training_run', 'description': '继续查看训练总结'}],
    }

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('cached explicit status request should render from state, not call direct_tool')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('看下训练状态')
    assert turn['status'] == 'completed', turn
    assert '训练已完成' in turn['message'], turn
    assert 'run_state=completed' in turn['message'], turn
    assert '继续查看训练总结' in turn['message'], turn


async def _scenario_cached_training_summary_request_reuses_state() -> None:
    client = _make_client('training-summary-request-cached')
    client.session_state.active_training.running = False
    client.session_state.active_training.training_run_summary = {
        'ok': True,
        'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
        'run_state': 'completed',
        'summary_overview': {'run_state': 'completed'},
        'action_candidates': [{'tool': 'analyze_training_outcome', 'description': '继续分析训练结果'}],
    }

    async def _unexpected_direct_tool(*args, **kwargs):
        raise AssertionError('training summary request should reuse cached state, not call direct_tool')

    client.direct_tool = _unexpected_direct_tool  # type: ignore[assignment]
    turn = await client.chat('数据总结')
    assert turn['status'] == 'completed', turn
    assert '训练结果汇总' in turn['message'], turn
    assert 'run_state=completed' in turn['message'], turn


async def _scenario_generic_approval_after_training_summary_routes_to_analysis() -> None:
    client = _make_client('training-summary-approval')
    graph = _ObservedToolGraph('analyze_training_outcome')
    client.graph = graph  # type: ignore[assignment]
    graph.bind(client)
    client.session_state.active_training.running = False
    client.session_state.active_training.training_run_summary = {
        'ok': True,
        'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
        'run_state': 'completed',
        'summary_overview': {'run_state': 'completed'},
        'action_candidates': [{'tool': 'analyze_training_outcome', 'description': '继续分析训练结果'}],
    }
    client._messages.append(AIMessage(content='训练结果汇总: 最近一次训练已完成，并且已有可分析指标。\n\n是否同意我继续分析训练结果？'))
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((tool_name, dict(kwargs)))
        assert tool_name == 'analyze_training_outcome'
        result = {
            'ok': True,
            'summary': '训练结果分析: 当前更像漏检问题。',
            'analysis_overview': {'assessment': 'under_recall'},
            'matched_rule_ids': ['generic_detect_under_recall'],
        }
        client._apply_to_state(tool_name, result, kwargs)
        return result

    client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
    assert await client._try_handle_mainline_intent('可以', 'thread-training-summary-approval') is None
    turn = await client.chat('可以')
    assert turn['status'] == 'completed', turn
    assert calls and calls[0][0] == 'analyze_training_outcome', calls
    assert graph.calls == [('analyze_training_outcome', {})], graph.calls
    assert '训练结果分析' in turn['message'], turn


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        await _scenario_status_followup_routes()
        await _scenario_generic_followup_after_start_uses_training_context()
        await _scenario_next_step_followup_routes()
        await _scenario_cached_next_step_followup_reuses_state()
        await _scenario_cached_explicit_next_step_request_reuses_state()
        await _scenario_cached_explicit_status_request_reuses_state()
        await _scenario_cached_training_summary_request_reuses_state()
        await _scenario_generic_approval_after_training_summary_routes_to_analysis()
        print('training followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
