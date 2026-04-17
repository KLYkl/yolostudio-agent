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
        raise AssertionError('create_react_agent should not be called in remote pipeline follow-up smoke')

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
from yolostudio_agent.agent.client.cached_tool_reply_service import build_cached_tool_snapshot_message
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


WORK = Path(__file__).resolve().parent / '_tmp_remote_pipeline_followup_route'


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        settings = AgentSettings(session_id='remote-pipeline-followup-smoke', memory_root=str(WORK))
        client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={})
        client.session_state.active_training.last_remote_roundtrip = {
            'ok': True,
            'upload': {
                'ok': True,
                'summary': '远端上传完成：已上传模型和数据集到 yolostudio:/tmp/remote_train',
                'transfer_overview': {'target_label': 'yolostudio', 'remote_root': '/tmp/remote_train', 'uploaded_count': 2},
            },
            'readiness': {
                'ok': True,
                'summary': '数据已具备训练条件',
                'readiness_overview': {'ready': True, 'resolved_data_yaml': '/tmp/remote_train/data.yaml'},
            },
            'preflight': {
                'ok': True,
                'summary': '训练预检通过',
                'preflight_overview': {'ready_to_start': True, 'resolved_device': '1'},
            },
            'start': {
                'ok': True,
                'summary': '训练已启动',
                'status_overview': {'running': True, 'device': '1'},
            },
            'final_status': {
                'ok': True,
                'summary': '训练已结束',
                'status_overview': {'run_state': 'completed', 'device': '1'},
            },
            'final_summary': {
                'ok': True,
                'summary': '训练总结完成',
                'summary_overview': {'best_epoch': 8, 'map50': 0.62},
            },
            'download': {
                'ok': True,
                'summary': '远端结果已回传到本机',
                'download_overview': {'local_root': '/local/results'},
            },
            'remote_dataset_path': '/tmp/remote_train/dataset',
            'remote_model_path': '/tmp/remote_train/yolov8n.pt',
            'remote_result_path': '/tmp/remote_train/runs/data-yolov8n',
            'local_result_root': '/local/results',
            'wait_for_completion': True,
            'download_after_completion': True,
            'final_run_state': 'completed',
        }

        snapshot = build_cached_tool_snapshot_message(client.session_state)
        assert snapshot is not None
        client.planner_llm = _FakePlannerLlm(
            '远端训练闭环已经完成，训练结果目录在 /tmp/remote_train/runs/data-yolov8n，结果也已回传到 /local/results。'
        )  # type: ignore[assignment]
        client.graph = HookedToolCallGraph(
            planner_llm=client.planner_llm,
            tool_name='remote_training_pipeline',
            snapshot_messages=[snapshot],
        )

        assert await client._try_handle_mainline_intent('远端那边现在是什么情况了？我需要详细一点的结果', 'thread-1') is None
        routed = await client.chat('远端那边现在是什么情况了？我需要详细一点的结果')
        assert routed['status'] == 'completed', routed
        assert '/tmp/remote_train/runs/data-yolov8n' in routed['message'], routed
        assert '/local/results' in routed['message'], routed

        print('remote pipeline followup route ok')
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
