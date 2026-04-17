from __future__ import annotations

import asyncio
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
        raise AssertionError('create_react_agent should not be called in remote transfer follow-up smoke')

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


class _DummyGraph:
    def get_state(self, config):
        return None


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


WORK = Path(__file__).resolve().parent / '_tmp_remote_transfer_followup_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='remote-transfer-followup-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})

        upload_result = {
            'ok': True,
            'summary': '远端上传完成：已上传 2 个本地项到 yolostudio:/tmp/agent_stage',
            'transfer_overview': {
                'target_label': 'yolostudio',
                'remote_root': '/tmp/agent_stage',
                'uploaded_count': 2,
                'file_count': 2,
            },
            'target_label': 'yolostudio',
            'profile_name': '',
            'remote_root': '/tmp/agent_stage',
            'uploaded_count': 2,
            'uploaded_items': [
                {'local_path': '/local/best.pt', 'remote_path': '/tmp/agent_stage/best.pt', 'item_type': 'file'},
                {'local_path': '/local/dataset', 'remote_path': '/tmp/agent_stage/dataset', 'item_type': 'directory'},
            ],
            'file_count': 2,
            'verified_file_count': 2,
            'skipped_file_count': 0,
            'chunked_file_count': 0,
            'scp_file_count': 2,
            'transferred_bytes': 2048,
            'skipped_bytes': 0,
            'total_bytes': 2048,
            'action_candidates': [{'description': '可继续在远端目录复用这些产物: /tmp/agent_stage', 'tool': 'remote_prediction_pipeline'}],
        }
        client._apply_to_state('upload_assets_to_remote', upload_result, {'server': 'yolostudio', 'remote_root': '/tmp/agent_stage'})

        client.planner_llm = _FakePlannerLlm('远端上传完成，当前远端目录是 /tmp/agent_stage，本次共上传 2 个本地项。')  # type: ignore[assignment]
        client.graph = HookedToolCallGraph(
            planner_llm=client.planner_llm,
            tool_name='upload_assets_to_remote',
        )
        assert await client._try_handle_mainline_intent('现在远端传输情况怎么样？我需要详细一点的信息', 'thread-1') is None
        routed = await client.chat('现在远端传输情况怎么样？我需要详细一点的信息')
        assert routed['status'] == 'completed', routed
        assert '/tmp/agent_stage' in routed['message'], routed
        assert '2' in routed['message'], routed

        print('remote transfer followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
