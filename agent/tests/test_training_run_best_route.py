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

    parent_mod = types.ModuleType('langchain_mcp_adapters')
    parent_mod.client = client_mod
    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters'] = parent_mod
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
        raise AssertionError('create_react_agent should not be called in training run best route smoke')

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
        best_result = {
            'ok': True,
            'summary': '最佳训练记录: train_log_200，状态=completed，mAP50=0.465，mAP50-95=0.260',
            'best_run_id': 'train_log_200',
            'ranking_basis': '状态=completed，mAP50=0.465，mAP50-95=0.260，precision=0.520',
            'best_run': {
                'run_id': 'train_log_200',
                'run_state': 'completed',
                'observation_stage': 'final',
                'metrics': {'precision': 0.52, 'recall': 0.60, 'map50': 0.465, 'map': 0.26},
            },
            'candidates': [
                {'run_id': 'train_log_200', 'run_state': 'completed'},
                {'run_id': 'train_log_100', 'run_state': 'completed'},
            ],
            'next_actions': ['如需查看最佳训练详情，可继续调用 inspect_training_run'],
        }
        tool_messages: list[Any] = []
        self.calls.append(('select_best_training_run', {}))
        tool_messages.extend(
            [
                AIMessage(content='', tool_calls=[{'id': f'call-{len(self.calls)}', 'name': 'select_best_training_run', 'args': {}}]),
                ToolMessage(content=json.dumps(best_result, ensure_ascii=False), name='select_best_training_run', tool_call_id=f'call-{len(self.calls)}'),
            ]
        )
        final_text = '最佳训练: train_log_200'
        if '怎么看' in user_text:
            analysis_result = {
                'ok': True,
                'summary': '最佳训练已完成，mAP50=0.465，当前更值得参考。',
                'signals': ['best_run_selected'],
                'facts': ['train_log_200 completed'],
            }
            self.calls.append(('analyze_training_outcome', {}))
            tool_messages.extend(
                [
                    AIMessage(content='', tool_calls=[{'id': f'call-{len(self.calls)}', 'name': 'analyze_training_outcome', 'args': {}}]),
                    ToolMessage(content=json.dumps(analysis_result, ensure_ascii=False), name='analyze_training_outcome', tool_call_id=f'call-{len(self.calls)}'),
                ]
            )
            final_text = '最佳训练已完成，mAP50=0.465，当前更值得参考。'
        elif '下一步怎么做' in user_text:
            next_step_result = {
                'ok': True,
                'summary': '建议先基于最佳训练继续补数据质量，再做下一轮训练。',
                'recommended_action': 'fix_data_quality',
                'signals': ['best_run_selected'],
            }
            self.calls.append(('recommend_next_training_step', {}))
            tool_messages.extend(
                [
                    AIMessage(content='', tool_calls=[{'id': f'call-{len(self.calls)}', 'name': 'recommend_next_training_step', 'args': {}}]),
                    ToolMessage(content=json.dumps(next_step_result, ensure_ascii=False), name='recommend_next_training_step', tool_call_id=f'call-{len(self.calls)}'),
                ]
            )
            final_text = '建议先基于最佳训练继续补数据质量，再做下一轮训练。'
        return {'messages': messages + tool_messages + [AIMessage(content=final_text)]}


WORK = Path(__file__).resolve().parent / '_tmp_training_run_best_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-run-best-route', memory_root=str(WORK))
        graph = _FakeGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, Any]]] = []

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name == 'select_best_training_run':
                result = {
                    'ok': True,
                    'summary': '最佳训练记录: train_log_200，状态=completed，mAP50=0.465，mAP50-95=0.260',
                    'best_run_id': 'train_log_200',
                    'ranking_basis': '状态=completed，mAP50=0.465，mAP50-95=0.260，precision=0.520',
                    'best_run': {
                        'run_id': 'train_log_200',
                        'run_state': 'completed',
                        'observation_stage': 'final',
                        'metrics': {'precision': 0.52, 'recall': 0.60, 'map50': 0.465, 'map': 0.26},
                    },
                    'candidates': [
                        {'run_id': 'train_log_200', 'run_state': 'completed'},
                        {'run_id': 'train_log_100', 'run_state': 'completed'},
                    ],
                    'next_actions': ['如需查看最佳训练详情，可继续调用 inspect_training_run'],
                }
            elif tool_name == 'analyze_training_outcome':
                result = {
                    'ok': True,
                    'summary': '最佳训练已完成，mAP50=0.465，当前更值得参考。',
                    'signals': ['best_run_selected'],
                    'facts': ['train_log_200 completed'],
                }
            elif tool_name == 'recommend_next_training_step':
                result = {
                    'ok': True,
                    'summary': '建议先基于最佳训练继续补数据质量，再做下一轮训练。',
                    'recommended_action': 'fix_data_quality',
                    'signals': ['best_run_selected'],
                }
            else:
                raise AssertionError(f'unexpected tool call: {tool_name}')
            client._apply_to_state(tool_name, result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        best_prompt = '最近哪次训练最好？'
        assert await client._try_handle_mainline_intent(best_prompt, 'thread-best') is None
        routed = await client.chat(best_prompt)
        assert routed['status'] == 'completed', routed
        assert graph.calls[-1] == ('select_best_training_run', {})
        assert '最佳训练: train_log_200' in routed['message']
        assert client.session_state.active_training.best_run_selection.get('best_run_id') == 'train_log_200'

        calls.clear()
        analysis_prompt = '最值得参考的训练结果怎么看？'
        assert await client._try_handle_mainline_intent(analysis_prompt, 'thread-best-analysis') is None
        routed = await client.chat(analysis_prompt)
        assert routed['status'] == 'completed', routed
        assert graph.calls[-2:] == [('select_best_training_run', {}), ('analyze_training_outcome', {})], graph.calls
        assert '最佳训练已完成' in routed['message']

        calls.clear()
        next_step_prompt = '最好的训练记录下一步怎么做？'
        assert await client._try_handle_mainline_intent(next_step_prompt, 'thread-best-next-step') is None
        routed = await client.chat(next_step_prompt)
        assert routed['status'] == 'completed', routed
        assert graph.calls[-2:] == [('select_best_training_run', {}), ('recommend_next_training_step', {})], graph.calls
        assert '补数据质量' in routed['message']
        print('training run best route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
