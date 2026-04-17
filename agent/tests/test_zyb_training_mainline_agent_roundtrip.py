from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
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
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
except Exception:
    parent_mod = types.ModuleType('langchain_mcp_adapters')
    client_mod = types.ModuleType('langchain_mcp_adapters.client')

    class _FakeMCPClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def get_tools(self):
            return []

    parent_mod.client = client_mod
    client_mod.MultiServerMCPClient = _FakeMCPClient
    sys.modules['langchain_mcp_adapters'] = parent_mod
    sys.modules['langchain_mcp_adapters.client'] = client_mod
    MultiServerMCPClient = None  # type: ignore[assignment]

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
        raise AssertionError('create_react_agent should not be called in zyb agent roundtrip smoke')

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

DEFAULT_OUT = str(Path(__file__).with_name('test_zyb_training_mainline_agent_roundtrip_output.json'))
DEFAULT_DATASET_ROOT = '/data/example_dataset'
DEFAULT_MODEL_PATH = '/models/yolov8n.pt'
DEFAULT_EPOCHS = 30
DEFAULT_STATUS_DELAYS = [15, 35, 60]
DEFAULT_MCP_URL = 'http://127.0.0.1:8080/mcp'
DEFAULT_TEST_MODE = 'mcp'
DEFAULT_TARGET_EPOCH = 2
DEFAULT_EXTRA_POLL_INTERVAL = 30
DEFAULT_EXTRA_POLL_LIMIT = 8


class _DummyGraph:
    async def ainvoke(self, payload, config=None):
        del payload, config
        raise AssertionError('graph should not run in zyb agent roundtrip direct-tools mode')

    def get_state(self, config):
        del config
        return None


class _PlannerResponse:
    def __init__(self, content: str):
        self.content = content


class _PlannerStub:
    @staticmethod
    def _joined_text(messages: list[Any]) -> str:
        parts: list[str] = []
        for message in messages:
            content = getattr(message, 'content', '')
            if isinstance(content, str):
                parts.append(content)
        return '\n'.join(parts)

    async def ainvoke(self, messages: list[Any]) -> Any:
        joined = self._joined_text(messages)
        if '训练跟进路由器' in joined:
            if '这次训练效果怎么样' in joined or '训练效果怎么样' in joined:
                return _PlannerResponse('{"action":"analysis","reason":"用户在追问训练效果"}')
            if '下一步先补数据还是调参数' in joined or '下一步' in joined:
                return _PlannerResponse('{"action":"next_step","reason":"用户在追问下一步建议"}')
            if '训练停了吗' in joined or '第几轮' in joined:
                return _PlannerResponse('{"action":"status","reason":"用户在追问训练状态"}')
        if '知识跟进路由器' in joined:
            if '这次训练效果怎么样' in joined or '训练效果怎么样' in joined:
                return _PlannerResponse('{"action":"analysis","reason":"用户在追问训练效果"}')
            if '下一步先补数据还是调参数' in joined or '下一步' in joined:
                return _PlannerResponse('{"action":"next_step","reason":"用户在追问下一步建议"}')
        return _PlannerResponse('')


def _norm(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        raw = '\n'.join((item.get('text', '') if isinstance(item, dict) else str(item)) for item in payload).strip()
        return json.loads(raw) if raw else {'ok': True}
    return {'ok': True, 'raw': str(payload)}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, '').strip()
    if not value:
        return default
    return int(value)


def _env_csv_ints(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name, '').strip()
    if not value:
        return default
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def _build_direct_tool_map() -> dict[str, Any]:
    import yolostudio_agent.agent.server.tools.combo_tools as combo_tools
    import yolostudio_agent.agent.server.tools.data_tools as data_tools
    import yolostudio_agent.agent.server.tools.knowledge_tools as knowledge_tools
    import yolostudio_agent.agent.server.tools.train_tools as train_tools

    return {
        'training_readiness': data_tools.training_readiness,
        'prepare_dataset_for_training': combo_tools.prepare_dataset_for_training,
        'training_preflight': train_tools.training_preflight,
        'start_training': train_tools.start_training,
        'check_training_status': train_tools.check_training_status,
        'stop_training': train_tools.stop_training,
        'summarize_training_run': train_tools.summarize_training_run,
        'analyze_training_outcome': knowledge_tools.analyze_training_outcome,
        'recommend_next_training_step': knowledge_tools.recommend_next_training_step,
        'list_training_environments': train_tools.list_training_environments,
    }


async def _build_mcp_tool_map(mcp_url: str) -> dict[str, Any]:
    if MultiServerMCPClient is None:
        raise RuntimeError('当前环境缺少 langchain_mcp_adapters，无法以 mcp 模式运行')
    client = MultiServerMCPClient({'y': {'transport': 'streamable-http', 'url': mcp_url}})
    tools = await client.get_tools()
    return {tool.name: tool for tool in tools}


async def _invoke_tool(tool_map: dict[str, Any], tool_name: str, kwargs: dict[str, Any], test_mode: str) -> dict[str, Any]:
    if test_mode == 'direct_tools':
        return _norm(tool_map[tool_name](**kwargs))
    return _norm(await tool_map[tool_name].ainvoke(kwargs))


async def main() -> None:
    out_path = Path(os.environ.get('YOLO_AGENT_TRAIN_OUT', DEFAULT_OUT))
    dataset_root = os.environ.get('YOLO_AGENT_TRAIN_DATASET_ROOT', DEFAULT_DATASET_ROOT).strip() or DEFAULT_DATASET_ROOT
    model_path = os.environ.get('YOLO_AGENT_TRAIN_MODEL_PATH', DEFAULT_MODEL_PATH).strip() or DEFAULT_MODEL_PATH
    epochs = _env_int('YOLO_AGENT_TRAIN_EPOCHS', DEFAULT_EPOCHS)
    status_delays = _env_csv_ints('YOLO_AGENT_TRAIN_STATUS_DELAYS', DEFAULT_STATUS_DELAYS)
    mcp_url = os.environ.get('YOLOSTUDIO_MCP_URL', DEFAULT_MCP_URL).strip() or DEFAULT_MCP_URL
    test_mode = os.environ.get('YOLO_AGENT_TRAIN_TEST_MODE', DEFAULT_TEST_MODE).strip() or DEFAULT_TEST_MODE
    target_epoch = _env_int('YOLO_AGENT_TRAIN_TARGET_EPOCH', DEFAULT_TARGET_EPOCH)
    extra_poll_interval = _env_int('YOLO_AGENT_TRAIN_EXTRA_POLL_INTERVAL', DEFAULT_EXTRA_POLL_INTERVAL)
    extra_poll_limit = _env_int('YOLO_AGENT_TRAIN_EXTRA_POLL_LIMIT', DEFAULT_EXTRA_POLL_LIMIT)
    work_root = Path(os.environ.get('YOLO_AGENT_TRAIN_WORK', str(Path(__file__).resolve().parent / '_tmp_zyb_training_mainline_agent_roundtrip')))

    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)

    if test_mode == 'direct_tools':
        tool_map = _build_direct_tool_map()
    else:
        tool_map = await _build_mcp_tool_map(mcp_url)

    settings = AgentSettings(session_id=f'zyb-agent-roundtrip-{int(time.time())}', memory_root=str(work_root))
    client = YoloStudioAgentClient(graph=_DummyGraph(), settings=settings, tool_registry={}, planner_llm=_PlannerStub())
    calls: list[dict[str, Any]] = []

    async def _direct_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        result = await _invoke_tool(tool_map, tool_name, dict(kwargs), test_mode)
        calls.append({'tool': tool_name, 'args': dict(kwargs), 'result': result})
        client._apply_to_state(tool_name, result, kwargs)
        client._record_secondary_event(tool_name, result)
        if tool_name == 'start_training' and result.get('ok'):
            client._clear_training_plan_draft()
        return result

    client.direct_tool = _direct_tool  # type: ignore[assignment]

    pre_status = await client.direct_tool('check_training_status')
    pre_stop = None
    if pre_status.get('running'):
        pre_stop = await client.direct_tool('stop_training')
        await asyncio.sleep(3)

    prompt = f'数据在 {dataset_root}，按默认划分准备后，用 {model_path} 训练 {epochs} 轮。'
    turn1 = await client.chat(prompt)
    assert turn1['status'] == 'needs_confirmation', turn1
    assert (turn1.get('tool_call') or {}).get('name') == 'prepare_dataset_for_training', turn1

    turn2 = await client.confirm(turn1['thread_id'], approved=True)
    assert turn2['status'] == 'needs_confirmation', turn2
    assert (turn2.get('tool_call') or {}).get('name') == 'start_training', turn2

    turn3 = await client.confirm(turn2['thread_id'], approved=True)
    assert turn3['status'] == 'completed', turn3
    assert client.session_state.active_training.running is True, turn3

    status_turns: list[dict[str, Any]] = []
    for delay in status_delays:
        await asyncio.sleep(delay)
        status_reply = await client.chat('现在训练到第几轮了？')
        status_turns.append({
            'delay_sec': delay,
            'reply': status_reply,
            'status': dict(client.session_state.active_training.last_status or {}),
        })

    extra_polls = 0
    while extra_polls < extra_poll_limit:
        observed_epochs = [int((item.get('status') or {}).get('progress', {}).get('epoch') or 0) for item in status_turns]
        max_observed_epoch = max(observed_epochs or [0])
        latest_status = status_turns[-1]['status'] if status_turns else {}
        if max_observed_epoch >= target_epoch or not latest_status.get('running'):
            break
        await asyncio.sleep(extra_poll_interval)
        status_reply = await client.chat('现在训练到第几轮了？')
        status_turns.append({
            'delay_sec': extra_poll_interval,
            'reply': status_reply,
            'status': dict(client.session_state.active_training.last_status or {}),
        })
        extra_polls += 1

    stop = None
    final_status = dict(client.session_state.active_training.last_status or {})
    if final_status.get('running'):
        stop = await client.direct_tool('stop_training')
        await asyncio.sleep(3)
        final_status = await client.direct_tool('check_training_status')

    final_status_turn = await client.chat('训练停了吗？')
    outcome_turn = await client.chat('这次训练效果怎么样？')
    next_step_turn = await client.chat('下一步先补数据还是调参数？')

    tool_names = [item['tool'] for item in calls]
    observed_epochs = [int((item.get('status') or {}).get('progress', {}).get('epoch') or 0) for item in status_turns]
    running_count = sum(1 for item in status_turns if (item.get('status') or {}).get('running'))
    summary_result = next((item['result'] for item in reversed(calls) if item['tool'] == 'summarize_training_run'), {})
    next_step_result = next((item['result'] for item in reversed(calls) if item['tool'] == 'recommend_next_training_step'), {})

    payload = {
        'ok': True,
        'test_mode': test_mode,
        'mcp_url': mcp_url,
        'dataset_root': dataset_root,
        'model_path': model_path,
        'epochs': epochs,
        'status_delays': status_delays,
        'target_epoch': target_epoch,
        'extra_poll_interval': extra_poll_interval,
        'extra_poll_limit': extra_poll_limit,
        'prompt': prompt,
        'pre_status': pre_status,
        'pre_stop': pre_stop,
        'turn1_prepare_confirmation': turn1,
        'turn2_start_confirmation': turn2,
        'turn3_started': turn3,
        'status_turns': status_turns,
        'stop': stop,
        'final_status': final_status,
        'final_status_turn': final_status_turn,
        'outcome_turn': outcome_turn,
        'next_step_turn': next_step_turn,
        'tool_calls': calls,
        'assessment': {
            'prepare_confirmation': turn1.get('status') == 'needs_confirmation',
            'start_confirmation': turn2.get('status') == 'needs_confirmation',
            'started_ok': turn3.get('status') == 'completed',
            'running_observed_count': running_count,
            'extra_polls': extra_polls,
            'max_observed_epoch': max(observed_epochs or [0]),
            'status_route_used': 'check_training_status' in tool_names,
            'final_status_route_used': final_status_turn.get('status') == 'completed' and '运行状态: stopped' in str(final_status_turn.get('message', '')),
            'summary_route_used': contains_in_order(tool_names, ['summarize_training_run', 'analyze_training_outcome']),
            'next_step_route_used': contains_in_order(tool_names, ['training_readiness', 'summarize_training_run', 'recommend_next_training_step']),
            'summary_run_state': summary_result.get('run_state'),
            'next_step_action': next_step_result.get('recommended_action'),
            'final_run_state': final_status.get('run_state'),
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'ok': True,
        'running_observed_count': running_count,
        'max_observed_epoch': max(observed_epochs or [0]),
        'summary_run_state': summary_result.get('run_state'),
        'next_step_action': next_step_result.get('recommended_action'),
        'output': str(out_path),
    }, ensure_ascii=False))


def contains_in_order(seq: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    idx = 0
    for item in seq:
        if item == expected[idx]:
            idx += 1
            if idx == len(expected):
                return True
    return False


if __name__ == '__main__':
    asyncio.run(main())
