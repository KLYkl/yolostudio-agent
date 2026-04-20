from __future__ import annotations

import json
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

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_openai.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_openai

    fake_ollama = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_ollama.ChatOllama = _FakeChatOllama
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

    def _Field(default=None, description='', **kwargs):
        del description, kwargs
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod

    prebuilt_mod = types.ModuleType('langgraph.prebuilt')
    types_mod = types.ModuleType('langgraph.types')
    checkpoint_mod = types.ModuleType('langgraph.checkpoint.memory')

    def _fake_create_react_agent(*args, **kwargs):
        raise AssertionError('create_react_agent should not be called in chaos tests')

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

from langchain_core.messages import AIMessage, ToolMessage
from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient
from yolostudio_agent.agent.tests._pending_confirmation_test_support import seed_pending_confirmation
from yolostudio_agent.agent.client.mainline_route_support import resolve_mainline_dispatch_payload
from yolostudio_agent.agent.tests._training_plan_test_support import (
    current_training_plan_context_payload,
    current_training_plan_draft,
    set_training_plan_draft,
)
from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_from_draft,
)
from yolostudio_agent.agent.client.training_request_service import (
    run_prepare_only_flow,
    run_training_request_entrypoint,
)


class _NoLLMGraph:
    def __init__(self) -> None:
        self.client = None
        self.plan_context: dict[str, Any] | None = None

    def bind(self, client) -> None:
        self.client = client

    def get_state(self, config):
        del config
        if not self.plan_context:
            return None
        return types.SimpleNamespace(values={'training_plan_context': dict(self.plan_context)})

    def update_state(self, config, update):
        del config
        if not isinstance(update, dict) or 'training_plan_context' not in update:
            return
        context = update.get('training_plan_context')
        self.plan_context = dict(context) if isinstance(context, dict) and context else None

    async def ainvoke(self, payload, config=None):
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break

        plan_context = dict(payload.get('training_plan_context') or self.plan_context or {})
        plan_status = str(plan_context.get('status') or '').strip().lower()
        next_tool = str(plan_context.get('next_step_tool') or '').strip()
        next_args = dict(plan_context.get('next_step_args') or {})
        execution_mode = str(plan_context.get('execution_mode') or '').strip().lower()
        reasoning_summary = str(plan_context.get('reasoning_summary') or '').strip()
        preflight_summary = str(plan_context.get('preflight_summary') or '').strip()
        if not plan_context and self.client is not None:
            mainline_context = self.client._collect_mainline_context(user_text)
            route_state = await self.client._resolve_mainline_route_state(user_text, mainline_context)
            dispatch_payload = resolve_mainline_dispatch_payload(
                mainline_context=mainline_context,
                route_state=route_state,
            )
            training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
            if training_entrypoint_args:
                prepare_only_followup = await run_prepare_only_flow(
                    user_text=user_text,
                    looks_like_prepare_only_request=self.client._looks_like_prepare_only_request,
                    extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
                    local_path_exists=lambda path: Path(path).expanduser().exists(),
                    direct_tool=self.client.direct_tool,
                    collect_requested_training_args=self.client._collect_requested_training_args,
                    build_training_plan_draft_fn=self.client._build_training_plan_draft,
                    render_tool_result_message=self.client._render_tool_result_message,
                )
                if prepare_only_followup:
                    entrypoint_result = {
                        'reply': str(prepare_only_followup.get('reply') or '').strip(),
                        'draft': dict(prepare_only_followup.get('draft') or {}),
                        'defer_to_graph': str(prepare_only_followup.get('action') or '').strip() == 'save_draft_and_handoff',
                    }
                else:
                    entrypoint_result = await run_training_request_entrypoint(
                        session_state=self.client.session_state,
                        user_text=user_text,
                        normalized_text=str(training_entrypoint_args.get('normalized_text') or ''),
                        dataset_path=str(training_entrypoint_args.get('dataset_path') or ''),
                        frame_followup_path=str(training_entrypoint_args.get('frame_followup_path') or ''),
                        wants_train=bool(training_entrypoint_args.get('wants_train')),
                        wants_predict=bool(training_entrypoint_args.get('wants_predict')),
                        no_train=bool(training_entrypoint_args.get('no_train')),
                        readiness_only_query=bool(training_entrypoint_args.get('readiness_only_query')),
                        wants_training_outcome_analysis=bool(training_entrypoint_args.get('wants_training_outcome_analysis')),
                        wants_next_step_guidance=bool(training_entrypoint_args.get('wants_next_step_guidance')),
                        wants_training_knowledge=bool(training_entrypoint_args.get('wants_training_knowledge')),
                        wants_training_revision=bool(training_entrypoint_args.get('wants_training_revision')),
                        wants_stop_training=bool(training_entrypoint_args.get('wants_stop_training')),
                        blocks_training_start=bool(training_entrypoint_args.get('blocks_training_start')),
                        explicit_run_ids=list(training_entrypoint_args.get('explicit_run_ids') or []),
                        wants_split=bool(training_entrypoint_args.get('wants_split')),
                        current_training_plan_context=current_training_plan_context_payload(self.client),
                        direct_tool=self.client.direct_tool,
                        collect_requested_training_args=self.client._collect_requested_training_args,
                        is_training_discussion_only=self.client._is_training_discussion_only,
                        extract_training_execution_backend=self.client._extract_training_execution_backend_from_text,
                        build_training_plan_draft_fn=self.client._build_training_plan_draft,
                        render_training_plan_message=self.client._render_training_plan_message,
                    )
                draft = dict((entrypoint_result or {}).get('draft') or {})
                reply = str((entrypoint_result or {}).get('reply') or '').strip()
                discussion_only = self.client._is_training_discussion_only(user_text)
                if draft:
                    set_training_plan_draft(self.client, draft)
                    self.plan_context = build_training_plan_context_from_draft(draft)
                if draft and discussion_only:
                    rendered = reply or await self.client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + ([AIMessage(content=rendered)] if rendered else [])}
                if bool((entrypoint_result or {}).get('defer_to_graph')) and draft:
                    plan_context = dict(self.plan_context or {})
                    plan_status = str(plan_context.get('status') or '').strip().lower()
                    next_tool = str(plan_context.get('next_step_tool') or '').strip()
                    next_args = dict(plan_context.get('next_step_args') or {})
                    execution_mode = str(plan_context.get('execution_mode') or '').strip().lower()
                    reasoning_summary = str(plan_context.get('reasoning_summary') or '').strip()
                    preflight_summary = str(plan_context.get('preflight_summary') or '').strip()
                elif reply:
                    return {'messages': messages + [AIMessage(content=reply)]}
                elif draft:
                    rendered = await self.client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + [AIMessage(content=rendered)]}

        is_post_prepare_start_handoff = (
            next_tool == 'start_training'
            and bool(preflight_summary)
            and '确认后即可启动' in reasoning_summary
        )
        is_execute_turn = (
            _looks_like_training_plan_execute_turn(user_text)
            or execution_mode == 'prepare_only'
            or is_post_prepare_start_handoff
            or (plan_status == 'ready_for_confirmation' and bool(next_tool))
        )
        if self.client is not None and config and next_tool and is_execute_turn:
            thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
            seed_pending_confirmation(self.client, 
                thread_id,
                {'name': next_tool, 'args': next_args, 'id': None, 'synthetic': True},
            )
            return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}
        raise AssertionError('chaos routed cases should not fallback to graph')


class _ScriptedGraph:
    def __init__(self, routes: dict[str, tuple[list[tuple[str, dict[str, Any]]], str]]) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.client = None
        self.plan_context: dict[str, Any] | None = None

    def bind(self, client) -> None:
        self.client = client

    def get_state(self, config):
        del config
        if not self.plan_context:
            return None
        return types.SimpleNamespace(values={'training_plan_context': dict(self.plan_context)})

    def update_state(self, config, update):
        del config
        if not isinstance(update, dict) or 'training_plan_context' not in update:
            return
        context = update.get('training_plan_context')
        self.plan_context = dict(context) if isinstance(context, dict) and context else None

    async def ainvoke(self, payload, config=None):
        messages = list(payload['messages'])
        user_text = ''
        for message in reversed(messages):
            content = getattr(message, 'content', '')
            if isinstance(content, str) and content:
                user_text = content
                break
        client = self.client
        plan_context = dict(payload.get('training_plan_context') or self.plan_context or {})
        if not plan_context and client is not None:
            mainline_context = client._collect_mainline_context(user_text)
            route_state = await client._resolve_mainline_route_state(user_text, mainline_context)
            dispatch_payload = resolve_mainline_dispatch_payload(
                mainline_context=mainline_context,
                route_state=route_state,
            )
            training_entrypoint_args = dict(dispatch_payload.get('training_entrypoint_request_args') or {})
            if training_entrypoint_args:
                prepare_only_followup = await run_prepare_only_flow(
                    user_text=user_text,
                    looks_like_prepare_only_request=client._looks_like_prepare_only_request,
                    extract_dataset_path=intent_parsing.extract_dataset_path_from_text,
                    local_path_exists=lambda path: Path(path).expanduser().exists(),
                    direct_tool=client.direct_tool,
                    collect_requested_training_args=client._collect_requested_training_args,
                    build_training_plan_draft_fn=client._build_training_plan_draft,
                    render_tool_result_message=client._render_tool_result_message,
                )
                if prepare_only_followup:
                    entrypoint_result = {
                        'reply': str(prepare_only_followup.get('reply') or '').strip(),
                        'draft': dict(prepare_only_followup.get('draft') or {}),
                        'defer_to_graph': str(prepare_only_followup.get('action') or '').strip() == 'save_draft_and_handoff',
                    }
                else:
                    entrypoint_result = await run_training_request_entrypoint(
                        session_state=client.session_state,
                        user_text=user_text,
                        normalized_text=str(training_entrypoint_args.get('normalized_text') or ''),
                        dataset_path=str(training_entrypoint_args.get('dataset_path') or ''),
                        frame_followup_path=str(training_entrypoint_args.get('frame_followup_path') or ''),
                        wants_train=bool(training_entrypoint_args.get('wants_train')),
                        wants_predict=bool(training_entrypoint_args.get('wants_predict')),
                        no_train=bool(training_entrypoint_args.get('no_train')),
                        readiness_only_query=bool(training_entrypoint_args.get('readiness_only_query')),
                        wants_training_outcome_analysis=bool(training_entrypoint_args.get('wants_training_outcome_analysis')),
                        wants_next_step_guidance=bool(training_entrypoint_args.get('wants_next_step_guidance')),
                        wants_training_knowledge=bool(training_entrypoint_args.get('wants_training_knowledge')),
                        wants_training_revision=bool(training_entrypoint_args.get('wants_training_revision')),
                        wants_stop_training=bool(training_entrypoint_args.get('wants_stop_training')),
                        blocks_training_start=bool(training_entrypoint_args.get('blocks_training_start')),
                        explicit_run_ids=list(training_entrypoint_args.get('explicit_run_ids') or []),
                        wants_split=bool(training_entrypoint_args.get('wants_split')),
                        current_training_plan_context=current_training_plan_context_payload(client),
                        direct_tool=client.direct_tool,
                        collect_requested_training_args=client._collect_requested_training_args,
                        is_training_discussion_only=client._is_training_discussion_only,
                        extract_training_execution_backend=client._extract_training_execution_backend_from_text,
                        build_training_plan_draft_fn=client._build_training_plan_draft,
                        render_training_plan_message=client._render_training_plan_message,
                    )
                draft = dict((entrypoint_result or {}).get('draft') or {})
                reply = str((entrypoint_result or {}).get('reply') or '').strip()
                discussion_only = client._is_training_discussion_only(user_text)
                if draft:
                    set_training_plan_draft(client, draft)
                    self.plan_context = build_training_plan_context_from_draft(draft)
                if draft and discussion_only:
                    rendered = reply or await client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + ([AIMessage(content=rendered)] if rendered else [])}
                if bool((entrypoint_result or {}).get('defer_to_graph')) and draft:
                    plan_context = dict(self.plan_context or {})
                    next_tool = str(plan_context.get('next_step_tool') or '').strip()
                    next_args = dict(plan_context.get('next_step_args') or {})
                    if config and next_tool and not discussion_only:
                        thread_id = str(((config or {}).get('configurable') or {}).get('thread_id') or '').strip()
                        seed_pending_confirmation(client, 
                            thread_id,
                            {'name': next_tool, 'args': next_args, 'id': None, 'synthetic': True},
                        )
                        return {'messages': messages + [AIMessage(content='按训练草案进入确认。')]}
                elif reply:
                    return {'messages': messages + [AIMessage(content=reply)]}
                elif draft:
                    rendered = await client._render_training_plan_message(
                        draft,
                        pending=bool(draft.get('next_step_tool')),
                    )
                    return {'messages': messages + [AIMessage(content=rendered)]}
        for marker, (tool_plan, final_text) in self.routes.items():
            if marker in user_text:
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
        raise AssertionError(f'unexpected graph prompt: {user_text}')


WORK = Path(__file__).resolve().parent / '_tmp_agent_server_chaos_p0'


def _looks_like_training_plan_execute_turn(text: str) -> bool:
    normalized = str(text or '').strip().lower()
    if not normalized:
        return False
    if normalized in {'y', 'yes'}:
        return True
    direct_tokens = ('执行', '开始吧', '就这样', '确认', '可以开始', '开训', '启动吧', '直接训练', '直接开始训练')
    retry_tokens = (
        '按原计划重试',
        '重试刚才那次训练',
        '重试上次训练',
        '按刚才的计划再来一次',
        '按原计划再来一次',
        '从最近状态继续训练',
        '从最近状态继续',
        '从最近状态恢复训练',
        '恢复刚才训练',
        '接着上次训练',
        '恢复上次训练',
        'resume 上次训练',
        'resume 刚才训练',
        'resume 另一个 run',
        'resume run',
        '继续另一个 run',
    )
    return any(token in text for token in direct_tokens + retry_tokens) or (
        'resume' in normalized and any(token in text for token in ('上次', '刚才', '最近', '继续', '恢复', '另一个', '历史', 'run'))
    )


def _make_client(session_id: str) -> YoloStudioAgentClient:
    root = WORK / session_id
    root.mkdir(parents=True, exist_ok=True)
    settings = AgentSettings(session_id=session_id, memory_root=str(root))
    graph = _NoLLMGraph()
    client = YoloStudioAgentClient(graph=graph, settings=settings, tool_registry={})
    graph.bind(client)
    return client
