"""
Harness 测试基建：conftest
提供 AgentClient 工厂、DummyGraph、mock 工具注册表
复用 test_agent_runtime_interrupts.py 的依赖 shim 模式
"""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

# ===== 依赖 shim（与 test_agent_runtime_interrupts.py 统一）=====
_repo_root = Path(__file__).resolve().parents[2]
_parent_root = _repo_root.parent
for _candidate in (_repo_root, _parent_root):
    _p = str(_candidate)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    __import__('langchain_openai')
except Exception:
    _fake = types.ModuleType('langchain_openai')
    class _FakeChatOpenAI:
        def __init__(self, *a, **kw): pass
    _fake.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = _fake

try:
    __import__('langchain_ollama')
except Exception:
    _fake = types.ModuleType('langchain_ollama')
    class _FakeChatOllama:
        def __init__(self, *a, **kw): pass
    _fake.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = _fake

try:
    __import__('langchain_core.messages')
except Exception:
    _core = types.ModuleType('langchain_core')
    _msgs = types.ModuleType('langchain_core.messages')
    _tools = types.ModuleType('langchain_core.tools')
    class _BM:
        def __init__(self, content=''): self.content = content
    class _AIM(_BM):
        def __init__(self, content='', tool_calls=None):
            super().__init__(content); self.tool_calls = tool_calls or []
    class _HM(_BM): pass
    class _SM(_BM): pass
    class _TM(_BM):
        def __init__(self, content='', name='', tool_call_id=''):
            super().__init__(content); self.name = name; self.tool_call_id = tool_call_id
    class _BT:
        name = 'fake'; description = 'fake'; args_schema = None
    class _ST(_BT):
        @classmethod
        def from_function(cls, **kw):
            t = cls()
            for k, v in kw.items(): setattr(t, k, v)
            return t
    _msgs.AIMessage = _AIM; _msgs.BaseMessage = _BM; _msgs.HumanMessage = _HM
    _msgs.SystemMessage = _SM; _msgs.ToolMessage = _TM
    _tools.BaseTool = _BT; _tools.StructuredTool = _ST
    _core.messages = _msgs; _core.tools = _tools
    sys.modules['langchain_core'] = _core
    sys.modules['langchain_core.messages'] = _msgs
    sys.modules['langchain_core.tools'] = _tools

try:
    __import__('langchain_mcp_adapters.client')
except Exception:
    _cm = types.ModuleType('langchain_mcp_adapters.client')
    class _FakeMCP:
        def __init__(self, *a, **kw): pass
        async def get_tools(self): return []
    _cm.MultiServerMCPClient = _FakeMCP
    sys.modules['langchain_mcp_adapters.client'] = _cm

try:
    __import__('pydantic')
except Exception:
    _pyd = types.ModuleType('pydantic')
    class _PBM:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
    def _PField(default=None, description=''): return default
    _pyd.BaseModel = _PBM; _pyd.Field = _PField
    sys.modules['pydantic'] = _pyd

try:
    __import__('langgraph.prebuilt')
    __import__('langgraph.types')
    __import__('langgraph.checkpoint.memory')
except Exception:
    _pre = types.ModuleType('langgraph.prebuilt')
    _typ = types.ModuleType('langgraph.types')
    _chk = types.ModuleType('langgraph.checkpoint.memory')
    def _noop(*a, **kw): raise AssertionError('不该在 harness 里调用 create_react_agent')
    class _Cmd:
        def __init__(self, resume=None): self.resume = resume
    class _IMS:
        def __init__(self, *a, **kw): self.storage = {}; self.writes = {}; self.blobs = {}
    _pre.create_react_agent = _noop; _typ.Command = _Cmd; _chk.InMemorySaver = _IMS
    sys.modules['langgraph.prebuilt'] = _pre
    sys.modules['langgraph.types'] = _typ
    sys.modules['langgraph.checkpoint.memory'] = _chk


# ===== 导入业务模块 =====
from langchain_core.messages import AIMessage, HumanMessage
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.client.session_state import SessionState
from yolostudio_agent.agent.tests._training_plan_test_support import set_training_plan_context

# ===== harness 临时目录 =====
HARNESS_WORK = Path(__file__).resolve().parent / '_harness_tmp'


# ===== DummyGraph: 支持 ainvoke + get_state =====
class DummyGraph:
    """最小 graph 桩，支持所有必要接口"""

    def __init__(self) -> None:
        self.invocation_log: list[dict[str, Any]] = []
        self._injected_state: Any = None

    def get_state(self, config: dict[str, Any] | None = None) -> Any:
        return self._injected_state

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.invocation_log.append({'payload': payload, 'config': config})
        messages = list(payload.get('messages', []))
        return {'messages': messages + [AIMessage(content='[DummyGraph] stub reply')]}

    def inject_pending_interrupt(self, *, tool_name: str, tool_args: dict[str, Any]) -> None:
        """注入 graph pending interrupt 状态"""
        import types as _t
        self._injected_state = _t.SimpleNamespace(
            next=('tools',),
            values={
                'messages': [
                    AIMessage(
                        content='',
                        tool_calls=[{'id': 'harness-call-1', 'name': tool_name, 'args': dict(tool_args)}]
                    )
                ]
            }
        )

    def clear_injected_state(self) -> None:
        self._injected_state = None


# ===== AgentClient 工厂 =====
def make_client(
    scenario_id: str,
    *,
    graph: DummyGraph | None = None,
    tool_registry: dict[str, Any] | None = None,
) -> YoloStudioAgentClient:
    """构造隔离的 AgentClient，无网络依赖"""
    scenario_root = HARNESS_WORK / scenario_id
    scenario_root.mkdir(parents=True, exist_ok=True)
    settings = AgentSettings(
        session_id=scenario_id,
        memory_root=str(scenario_root),
    )
    return YoloStudioAgentClient(
        graph=graph or DummyGraph(),
        settings=settings,
        tool_registry=tool_registry or {},
    )


def make_client_at_state(
    scenario_id: str,
    *,
    preset: str = 'empty',
    graph: DummyGraph | None = None,
) -> YoloStudioAgentClient:
    """构造处于指定状态的 AgentClient

    Args:
        scenario_id: 场景唯一标识
        preset: 状态预设名称，见 fixtures.STATE_PRESETS
        graph: 可选的自定义 graph
    """
    from . import fixtures

    client = make_client(scenario_id, graph=graph)
    state_patch = fixtures.STATE_PRESETS.get(preset, {})

    pending_patch: dict[str, Any] = {}

    # 应用状态补丁
    for dotted_key, value in state_patch.items():
        if dotted_key.startswith('pending_confirmation.'):
            pending_patch[dotted_key.split('.', 1)[1]] = value
            continue
        if dotted_key == 'training_plan_context':
            set_training_plan_context(client, dict(value or {}), thread_id=scenario_id)
            continue
        parts = dotted_key.split('.')
        obj = client.session_state
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    if pending_patch:
        tool_name = str(pending_patch.get('tool_name') or pending_patch.get('name') or '').strip()
        thread_id = str(pending_patch.get('thread_id') or '').strip()
        client._remember_pending_confirmation(
            {
                'id': pending_patch.get('id'),
                'tool_call_id': str(pending_patch.get('tool_call_id') or pending_patch.get('id') or '').strip(),
                'name': tool_name,
                'tool_name': tool_name,
                'args': dict(pending_patch.get('tool_args') or pending_patch.get('args') or {}),
                'tool_args': dict(pending_patch.get('tool_args') or pending_patch.get('args') or {}),
                'source': str(pending_patch.get('source') or 'synthetic').strip().lower() or 'synthetic',
                'summary': str(pending_patch.get('summary') or '').strip(),
                'objective': str(pending_patch.get('objective') or '').strip(),
                'allowed_decisions': list(pending_patch.get('allowed_decisions') or []),
                'review_config': dict(pending_patch.get('review_config') or {}),
                'decision_context': dict(pending_patch.get('decision_context') or {}),
                'thread_id': thread_id,
                'interrupt_kind': str(pending_patch.get('interrupt_kind') or 'tool_approval').strip() or 'tool_approval',
                'created_at': str(pending_patch.get('created_at') or ''),
            },
            emit_event=False,
            persist_graph=False,
        )

    return client


def cleanup_harness() -> None:
    """清理 harness 临时目录"""
    shutil.rmtree(HARNESS_WORK, ignore_errors=True)


def setup_harness() -> None:
    """初始化 harness 临时目录"""
    cleanup_harness()
    HARNESS_WORK.mkdir(parents=True, exist_ok=True)
