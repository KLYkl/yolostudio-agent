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

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

try:
    import langchain_ollama  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

try:
    import langchain_core.messages  # type: ignore  # noqa: F401
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

        async def ainvoke(self, kwargs):
            return kwargs

        def invoke(self, kwargs):
            return kwargs

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
    import langchain_mcp_adapters.client  # type: ignore  # noqa: F401
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
    import pydantic  # type: ignore  # noqa: F401
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
    import langgraph.prebuilt  # type: ignore  # noqa: F401
    import langgraph.types  # type: ignore  # noqa: F401
    import langgraph.checkpoint.memory  # type: ignore  # noqa: F401
except Exception:
    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in route smoke')

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

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from langchain_core.messages import AIMessage, ToolMessage


class _FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_state(self, config):
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break

        summary_result = {
            'ok': True,
            'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
            'run_state': 'completed',
            'analysis_ready': True,
            'metrics': {'precision': 0.82, 'recall': 0.36, 'map50': 0.33, 'map': 0.18},
            'signals': ['training_completed', 'high_precision_low_recall'],
            'facts': ['precision=0.820', 'recall=0.360'],
            'next_actions': ['可继续调用 analyze_training_outcome 解释训练效果'],
        }
        compare_result = {
            'ok': True,
            'summary': '训练对比完成: train_log_200 相比 train_log_100，precision提升 +0.1000',
            'left_run_id': 'train_log_200',
            'right_run_id': 'train_log_100',
            'left_run': {
                'run_id': 'train_log_200',
                'run_state': 'completed',
                'observation_stage': 'final',
                'metrics': {'precision': 0.82, 'recall': 0.36, 'map50': 0.33, 'map': 0.18},
                'signals': ['training_completed', 'high_precision_low_recall'],
            },
            'right_run': {
                'run_id': 'train_log_100',
                'run_state': 'completed',
                'observation_stage': 'final',
                'metrics': {'precision': 0.72, 'recall': 0.35, 'map50': 0.27, 'map': 0.14},
            },
            'metric_deltas': {'precision': {'left': 0.82, 'right': 0.72, 'delta': 0.1}},
            'highlights': ['precision提升 +0.1000'],
            'next_actions': ['可继续调用 recommend_next_training_step'],
        }
        analysis_result = {
            'ok': True,
            'summary': '训练结果分析: 当前更像漏检问题。',
            'assessment': 'inspect_missed_samples',
            'interpretation': '当前 precision 高、recall 低，更像漏检偏多。',
            'recommendation': '先检查漏检样本。',
            'matched_rule_ids': ['generic_post_high_precision_low_recall'],
            'signals': ['high_precision_low_recall'],
            'facts': ['precision=0.820', 'recall=0.360'],
            'next_actions': ['检查漏检样本', '确认是否存在漏标'],
        }
        next_step_result = {
            'ok': True,
            'summary': '下一步建议: 优先保持小步迭代。',
            'recommended_action': 'quick_iteration',
            'basis': ['样本量=120'],
            'why': '当前数据量偏小，更适合先做短周期验证。',
            'matched_rule_ids': ['generic_next_small_dataset_fast_iteration'],
            'signals': ['small_dataset'],
            'next_actions': ['先做一次短周期训练', '记录失败样本后再补数据'],
        }
        retrieval_result = {
            'ok': True,
            'summary': '知识检索完成: 命中 1 条规则',
            'topic': 'training_metrics',
            'stage': 'post_training',
            'model_family': 'yolo',
            'matched_rule_ids': ['generic_post_high_precision_low_recall'],
            'matched_rules': [
                {
                    'id': 'generic_post_high_precision_low_recall',
                    'interpretation': '模型偏保守，漏检偏多。',
                    'next_actions': ['先检查漏检样本'],
                }
            ],
            'playbooks': [],
            'next_actions': ['先检查漏检样本'],
        }

        tool_plan: list[tuple[str, dict[str, Any]]] = []
        final_text = ''
        if 'precision 高 recall 低说明什么' in user_text:
            tool_plan = [('retrieve_training_knowledge', retrieval_result)]
            final_text = '知识检索完成: 命中 1 条规则'
        elif '这次训练效果怎么样' in user_text:
            tool_plan = [('summarize_training_run', summary_result), ('analyze_training_outcome', analysis_result)]
            final_text = '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。\n\n训练结果分析: 当前更像漏检问题。'
        elif '下一步先补数据还是先调参数' in user_text:
            tool_plan = [('summarize_training_run', summary_result), ('recommend_next_training_step', next_step_result)]
            final_text = '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。\n\n下一步建议: 优先保持小步迭代。\n建议动作: quick_iteration'
        elif '对比最近两次训练后下一步怎么做' in user_text:
            tool_plan = [('compare_training_runs', compare_result), ('recommend_next_training_step', next_step_result)]
            final_text = '训练对比完成: train_log_200 相比 train_log_100，precision提升 +0.1000\n\n下一步建议: 优先保持小步迭代。\n建议动作: quick_iteration'
        elif '对比最近两次训练效果怎么看' in user_text:
            tool_plan = [('compare_training_runs', compare_result), ('analyze_training_outcome', analysis_result)]
            final_text = '训练对比完成: train_log_200 相比 train_log_100，precision提升 +0.1000\n\n训练结果分析: 当前更像漏检问题。'
        else:
            raise AssertionError(f'unexpected graph prompt: {user_text}')

        tool_messages: list[Any] = []
        for tool_name, result in tool_plan:
            self.calls.append((tool_name, {}))
            tool_call_id = f'call-{len(self.calls)}'
            tool_messages.extend(
                [
                    AIMessage(content='', tool_calls=[{'id': tool_call_id, 'name': tool_name, 'args': {}}]),
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), name=tool_name, tool_call_id=tool_call_id),
                ]
            )
        return {'messages': messages + tool_messages + [AIMessage(content=final_text)]}


WORK = Path(__file__).resolve().parent / '_tmp_knowledge_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='knowledge-route-smoke', memory_root=str(WORK))
        graph = _FakeGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any):
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'dataset_training_readiness':
                assert kwargs['img_dir'] == '/data/dataset'
                result = {
                    'ok': True,
                    'summary': '数据集训练就绪检查完成: 数据可训练，但样本量偏小。',
                    'dataset_root': '/data/dataset',
                    'resolved_img_dir': '/data/dataset/images',
                    'resolved_label_dir': '/data/dataset/labels',
                    'resolved_data_yaml': '/data/dataset/data.yaml',
                    'risk_level': 'medium',
                    'ready': True,
                    'warnings': ['样本量较小'],
                    'blockers': [],
                    'next_actions': ['可继续 prepare_dataset_for_training 或 start_training'],
                }
            elif tool_name == 'training_readiness':
                assert kwargs['img_dir'] == '/data/dataset'
                result = {
                    'ok': True,
                    'summary': '训练前检查完成: 数据可训练，环境可用。',
                    'ready_to_start': True,
                    'resolved_device': 'auto',
                    'training_environment': {'name': 'yolodo'},
                    'warnings': ['样本量较小'],
                    'blockers': [],
                    'next_actions': ['可以继续 summarize_training_run / recommend_next_training_step'],
                }
            elif tool_name == 'recommend_next_training_step':
                status_payload = kwargs.get('status') or {}
                comparison_payload = kwargs.get('comparison') or {}
                if comparison_payload:
                    assert comparison_payload == client.session_state.active_training.last_run_comparison
                    assert status_payload == comparison_payload.get('left_run')
                else:
                    if status_payload.get('run_state') == 'completed':
                        assert status_payload.get('analysis_ready') is True
                        assert status_payload.get('metrics', {}).get('precision') == 0.82
                    else:
                        assert status_payload == client.session_state.active_training.last_status
                result = {
                    'ok': True,
                    'summary': '下一步建议: 优先保持小步迭代。',
                    'recommended_action': 'quick_iteration',
                    'basis': ['样本量=120'],
                    'why': '当前数据量偏小，更适合先做短周期验证。',
                    'matched_rule_ids': ['generic_next_small_dataset_fast_iteration'],
                    'signals': ['small_dataset'],
                    'next_actions': ['先做一次短周期训练', '记录失败样本后再补数据'],
                }
            elif tool_name == 'retrieve_training_knowledge':
                assert kwargs['signals'] == ['high_precision_low_recall']
                result = {
                    'ok': True,
                    'summary': '知识检索完成: 命中 1 条规则',
                    'topic': kwargs['topic'],
                    'stage': kwargs['stage'],
                    'model_family': kwargs['model_family'],
                    'matched_rule_ids': ['generic_post_high_precision_low_recall'],
                    'matched_rules': [
                        {
                            'id': 'generic_post_high_precision_low_recall',
                            'interpretation': '模型偏保守，漏检偏多。',
                            'next_actions': ['先检查漏检样本'],
                        }
                    ],
                    'playbooks': [],
                    'next_actions': ['先检查漏检样本'],
                }
            elif tool_name == 'summarize_training_run':
                result = {
                    'ok': True,
                    'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
                    'run_state': 'completed',
                    'analysis_ready': True,
                    'metrics': {'precision': 0.82, 'recall': 0.36, 'map50': 0.33, 'map': 0.18},
                    'signals': ['training_completed', 'high_precision_low_recall'],
                    'facts': ['precision=0.820', 'recall=0.360'],
                    'next_actions': ['可继续调用 analyze_training_outcome 解释训练效果'],
                }
            elif tool_name == 'compare_training_runs':
                result = {
                    'ok': True,
                    'summary': '训练对比完成: train_log_200 相比 train_log_100，precision提升 +0.1000',
                    'left_run_id': kwargs.get('left_run_id') or 'train_log_200',
                    'right_run_id': kwargs.get('right_run_id') or 'train_log_100',
                    'left_run': {
                        'run_id': kwargs.get('left_run_id') or 'train_log_200',
                        'run_state': 'completed',
                        'observation_stage': 'final',
                        'metrics': {'precision': 0.82, 'recall': 0.36, 'map50': 0.33, 'map': 0.18},
                        'signals': ['training_completed', 'high_precision_low_recall'],
                    },
                    'right_run': {
                        'run_id': kwargs.get('right_run_id') or 'train_log_100',
                        'run_state': 'completed',
                        'observation_stage': 'final',
                        'metrics': {'precision': 0.72, 'recall': 0.35, 'map50': 0.27, 'map': 0.14},
                    },
                    'metric_deltas': {'precision': {'left': 0.82, 'right': 0.72, 'delta': 0.1}},
                    'highlights': ['precision提升 +0.1000'],
                    'next_actions': ['可继续调用 recommend_next_training_step'],
                }
            elif tool_name == 'analyze_training_outcome':
                comparison_payload = kwargs.get('comparison') or {}
                if comparison_payload:
                    assert comparison_payload == client.session_state.active_training.last_run_comparison
                    assert kwargs['metrics'] == comparison_payload.get('left_run')
                else:
                    metrics_payload = kwargs['metrics']
                    assert metrics_payload.get('run_state') == 'completed'
                    assert metrics_payload.get('analysis_ready') is True
                    assert metrics_payload.get('metrics', {}).get('precision') == 0.82
                result = {
                    'ok': True,
                    'summary': '训练结果分析: 当前更像漏检问题。',
                    'assessment': 'inspect_missed_samples',
                    'interpretation': '当前 precision 高、recall 低，更像漏检偏多。',
                    'recommendation': '先检查漏检样本。',
                    'matched_rule_ids': ['generic_post_high_precision_low_recall'],
                    'signals': ['high_precision_low_recall'],
                    'facts': ['precision=0.820', 'recall=0.360'],
                    'next_actions': ['检查漏检样本', '确认是否存在漏标'],
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')
            client._apply_to_state(tool_name, result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]
        client.session_state.active_training.last_status = {
            'latest_metrics': {'metrics': {'precision': 0.82, 'recall': 0.36, 'map50': 0.33}},
            'running': False,
        }

        routed = await client._try_handle_mainline_intent('这个 /data/dataset 现在适合训练吗？', 'thread-1')
        assert routed is not None
        assert routed['status'] == 'completed'
        assert '数据可训练' in routed['message']
        assert '建议动作' in routed['message']
        assert calls[0][0] == 'dataset_training_readiness'
        assert calls[1][0] == 'recommend_next_training_step'

        knowledge_prompt = 'precision 高 recall 低说明什么？'
        assert await client._try_handle_mainline_intent(knowledge_prompt, 'thread-2') is None
        routed2 = await client.chat(knowledge_prompt)
        assert routed2['status'] == 'completed'
        assert '知识检索完成' in routed2['message']
        assert graph.calls[-1] == ('retrieve_training_knowledge', {})

        analysis_prompt = '这次训练效果怎么样？'
        assert await client._try_handle_mainline_intent(analysis_prompt, 'thread-3') is None
        routed3 = await client.chat(analysis_prompt)
        assert routed3['status'] == 'completed'
        assert '训练结果汇总' in routed3['message']
        assert '训练结果分析' in routed3['message']
        assert graph.calls[-2:] == [('summarize_training_run', {}), ('analyze_training_outcome', {})]

        next_step_prompt = '下一步先补数据还是先调参数？'
        assert await client._try_handle_mainline_intent(next_step_prompt, 'thread-4') is None
        routed4 = await client.chat(next_step_prompt)
        assert routed4['status'] == 'completed'
        assert '训练结果汇总' in routed4['message']
        assert '建议动作' in routed4['message']
        assert graph.calls[-2:] == [('summarize_training_run', {}), ('recommend_next_training_step', {})]

        compare_next_prompt = '对比最近两次训练后下一步怎么做？'
        assert await client._try_handle_mainline_intent(compare_next_prompt, 'thread-5') is None
        routed5 = await client.chat(compare_next_prompt)
        assert routed5['status'] == 'completed'
        assert '训练对比完成' in routed5['message']
        assert '建议动作' in routed5['message']
        assert graph.calls[-2:] == [('compare_training_runs', {}), ('recommend_next_training_step', {})]

        compare_analysis_prompt = '对比最近两次训练效果怎么看？'
        assert await client._try_handle_mainline_intent(compare_analysis_prompt, 'thread-6') is None
        routed6 = await client.chat(compare_analysis_prompt)
        assert routed6['status'] == 'completed'
        assert '训练对比完成' in routed6['message']
        assert '训练结果分析' in routed6['message']
        assert graph.calls[-2:] == [('compare_training_runs', {}), ('analyze_training_outcome', {})]
        print('knowledge route smoke ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
