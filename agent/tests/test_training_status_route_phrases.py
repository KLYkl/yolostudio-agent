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
        raise AssertionError('create_react_agent should not be called in training status route smoke')

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


class _DummyGraph:
    def get_state(self, config):
        return None


WORK = Path(__file__).resolve().parent / '_tmp_training_status_route_phrases'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='training-status-route-phrases', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        calls: list[tuple[str, dict[str, Any]]] = []
        status_mode = {'value': 'stopped'}

        async def _fake_direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((tool_name, dict(kwargs)))
            if tool_name != 'check_training_status':
                raise AssertionError(f'unexpected tool call: {tool_name}')
            if status_mode['value'] == 'failed':
                result = {
                    'ok': True,
                    'summary': '训练已失败，当前可先检查报错日志与训练环境。',
                    'running': False,
                    'run_state': 'failed',
                    'observation_stage': 'final',
                    'progress': {'epoch': 2, 'total_epochs': 30, 'progress_ratio': 2 / 30},
                    'latest_metrics': {
                        'ok': True,
                        'metrics': {'epoch': 2, 'total_epochs': 30, 'box_loss': 1.42, 'cls_loss': 0.91, 'dfl_loss': 1.12},
                    },
                    'analysis_ready': False,
                    'minimum_facts_ready': True,
                    'signals': ['training_failed', 'loss_only_metrics', 'metrics_missing'],
                    'facts': ['epoch=2/30', 'RuntimeError: CUDA out of memory'],
                    'next_actions': ['先检查报错日志，再确认训练环境与 batch 设置'],
                }
            else:
                result = {
                    'ok': True,
                    'summary': '训练已停止，当前保留了停止前的最终可读指标。',
                    'running': False,
                    'run_state': 'stopped',
                    'observation_stage': 'final',
                    'progress': {'epoch': 30, 'total_epochs': 30, 'progress_ratio': 1.0},
                    'latest_metrics': {
                        'ok': True,
                        'metrics': {'epoch': 30, 'total_epochs': 30, 'precision': 0.44, 'recall': 0.81, 'map50': 0.47, 'map': 0.25},
                    },
                    'analysis_ready': True,
                    'minimum_facts_ready': True,
                    'signals': ['training_stopped', 'low_precision_high_recall'],
                    'facts': ['epoch=30/30', 'precision=0.440', 'recall=0.810'],
                    'next_actions': ['可继续调用 summarize_training_run 查看最终训练事实'],
                }
            client._apply_to_state(tool_name, result, kwargs)
            return result

        client.direct_tool = _fake_direct_tool  # type: ignore[assignment]

        for text in ('训练停了吗？', '训练结束了吗？', '训练完成了吗？', '训练跑完了吗？'):
            status_mode['value'] = 'stopped'
            routed = await client._try_handle_mainline_intent(text, f'thread-{text}')
            assert routed is not None
            assert routed['status'] == 'completed', routed
            assert calls[-1][0] == 'check_training_status'
            assert '运行状态: stopped' in routed['message']
            assert '观察阶段: 最终状态' in routed['message']
            assert '最近指标:' in routed['message']

        for text in ('训练失败了吗？', '是不是训练失败了？', '训练挂了吗？'):
            status_mode['value'] = 'failed'
            routed = await client._try_handle_mainline_intent(text, f'thread-{text}')
            assert routed is not None
            assert routed['status'] == 'completed', routed
            assert calls[-1][0] == 'check_training_status'
            assert '运行状态: failed' in routed['message']
            assert '观察阶段: 最终状态' in routed['message']
            assert '当前仅有训练损失:' in routed['message']
            assert '当前不足: 缺少稳定评估指标' in routed['message']

        print('training status route phrases ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
