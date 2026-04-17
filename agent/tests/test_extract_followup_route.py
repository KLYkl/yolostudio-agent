from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
import sys
import types

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
        raise AssertionError('create_react_agent should not be called in extract follow-up smoke')

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
from yolostudio_agent.agent.tests._post_model_hook_support import HookedToolCallGraph
from langchain_core.messages import AIMessage, ToolMessage


class _FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

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
        assert '抽 10% 的图片到 /tmp/extract_run' in user_text, user_text
        tool_name = 'extract_images'
        args = {
            'source_path': '/data/raw/images',
            'selection_mode': 'ratio',
            'ratio': 0.1,
            'output_dir': '/tmp/extract_run',
        }
        result = {
            'ok': True,
            'summary': '图片抽取完成: 实际抽取 18 张图片，复制标签 18 个',
            'extracted': 18,
            'labels_copied': 18,
            'conflict_count': 0,
            'output_dir': '/tmp/extract_run',
            'workflow_ready_path': '/tmp/extract_run',
            'warnings': [],
            'action_candidates': [{'description': '可继续对输出目录做数据集质量检查', 'tool': 'scan_dataset'}],
            'output_img_dir': '/tmp/extract_run/images',
            'output_label_dir': '/tmp/extract_run/labels',
        }
        self.calls.append((tool_name, dict(args)))
        return {
            'messages': messages + [
                AIMessage(content='', tool_calls=[{'id': 'call-1', 'name': tool_name, 'args': args}]),
                ToolMessage(content=json.dumps(result, ensure_ascii=False), name=tool_name, tool_call_id='call-1'),
                AIMessage(content='图片抽取完成: 实际抽取 18 张图片，复制标签 18 个'),
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


WORK = Path(__file__).resolve().parent / '_tmp_extract_followup_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='extract-followup-smoke', memory_root=str(WORK))
        graph = _FakeGraph()
        client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})

        extract_prompt = '从 /data/raw/images 抽 10% 的图片到 /tmp/extract_run'
        assert await client._try_handle_mainline_intent(extract_prompt, 'thread-1') is None
        initial = await client.chat(extract_prompt)
        assert initial['status'] == 'completed', initial
        assert '实际抽取 18 张' in initial['message'], initial
        assert graph.calls[-1][0] == 'extract_images', graph.calls

        client.planner_llm = _FakePlannerLlm('图片抽取完成，当前输出目录是 /tmp/extract_run，本次实际抽取 18 张图片。')  # type: ignore[assignment]
        client.graph = HookedToolCallGraph(
            planner_llm=client.planner_llm,
            tool_name='extract_images',
        )
        before = len(graph.calls)
        assert await client._try_handle_mainline_intent('现在抽图情况怎么样？我需要详细一点的信息', 'thread-2') is None
        routed = await client.chat('现在抽图情况怎么样？我需要详细一点的信息')
        assert routed['status'] == 'completed', routed
        assert '/tmp/extract_run' in routed['message'] or '18 张' in routed['message'], routed
        assert len(graph.calls) == before, graph.calls

        print('extract followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
