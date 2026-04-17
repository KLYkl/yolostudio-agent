from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from langchain_core.messages import AIMessage, ToolMessage

from yolostudio_agent.agent.client.agent_client import AgentSettings, YoloStudioAgentClient, _build_agent_post_model_hook
from yolostudio_agent.agent.client.cached_tool_reply_service import build_cached_tool_context_payload
from yolostudio_agent.agent.client.memory_store import MemoryStore
from yolostudio_agent.agent.client.session_state import SessionState


WORK = Path(__file__).resolve().parent / '_tmp_route_ownership_logging'


class _GraphWithHandledToolError:
    def get_state(self, config):
        del config
        return None

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        messages.append(
            AIMessage(
                content='',
                tool_calls=[{'id': 'tc-route-1', 'name': 'list_remote_profiles', 'args': {}}],
            )
        )
        messages.append(
            ToolMessage(
                content=json.dumps({'ok': False, 'error': 'mock failure'}, ensure_ascii=False),
                name='list_remote_profiles',
                tool_call_id='tc-route-1',
            )
        )
        messages.append(AIMessage(content='工具失败已被兜底处理。'))
        return {'messages': messages}


class _FakePlannerResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakePlannerLlm:
    async def ainvoke(self, messages):
        text = '\n'.join(str(getattr(message, 'content', '')) for message in messages)
        if 'list_remote_profiles' in text:
            return _FakePlannerResponse('当前默认可用服务器配置是 lab。')
        return _FakePlannerResponse('')


async def _run() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    graph_client = YoloStudioAgentClient(
        graph=_GraphWithHandledToolError(),
        settings=AgentSettings(session_id='route-graph', memory_root=str(WORK / 'graph')),
        tool_registry={},
        planner_llm=None,
    )

    async def _never_bypass(user_text: str, thread_id: str):
        del user_text, thread_id
        return None

    graph_client._try_handle_mainline_intent = _never_bypass  # type: ignore[assignment]
    graph_result = await graph_client.chat('你好')
    assert graph_result['status'] == 'completed', graph_result
    graph_routes = graph_client.route_ownership_report()
    graph_route_names = [str(item.get('route') or '') for item in graph_routes]
    assert 'graph-selected-tool' in graph_route_names, graph_routes
    assert 'tool-error-recovery' in graph_route_names, graph_routes

    bypass_client = YoloStudioAgentClient(
        graph=_GraphWithHandledToolError(),
        settings=AgentSettings(session_id='route-bypass', memory_root=str(WORK / 'bypass')),
        tool_registry={},
        planner_llm=None,
    )

    async def _bypass_mainline(user_text: str, thread_id: str):
        del user_text
        return {
            'status': 'completed',
            'message': '旁路已处理',
            'tool_call': {'name': 'scan_dataset', 'args': {'img_dir': '/data/demo'}},
            'thread_id': thread_id,
        }

    bypass_client._try_handle_mainline_intent = _bypass_mainline  # type: ignore[assignment]
    bypass_result = await bypass_client.chat('扫描这个数据集')
    assert bypass_result['status'] == 'completed', bypass_result
    bypass_routes = bypass_client.route_ownership_report()
    assert any(item.get('route') == 'graph-external-bypass' for item in bypass_routes), bypass_routes

    hook_store = MemoryStore(WORK / 'hook')

    def _report_route(route: str, payload: dict[str, object]) -> None:
        hook_store.append_event('route-hook', 'route_ownership', {'route': route, **dict(payload)})

    state = SessionState(session_id='route-hook')
    state.active_remote_transfer.last_profile_listing = {
        'ok': True,
        'summary': '远端 profile 1 个 / SSH alias 1 个。 默认 profile: lab。',
        'default_profile': 'lab',
        'profiles': [{'name': 'lab', 'target_label': 'lab'}],
    }
    cached_tool_context = build_cached_tool_context_payload(state)
    assert cached_tool_context is not None

    hook = _build_agent_post_model_hook(_FakePlannerLlm(), route_reporter=_report_route)
    hook_update = await hook(
        {
            'messages': [
                AIMessage(content='previous'),
                AIMessage(content='', tool_calls=[{'id': 'tc-hook-1', 'name': 'list_remote_profiles', 'args': {}}]),
            ],
        }
    )
    assert hook_update == {}, hook_update

    hook_update = await hook(
        {
            'messages': [
                AIMessage(content='previous'),
                ToolMessage(content='ignored', name='noop', tool_call_id='noop'),
            ]
        }
    )
    assert hook_update == {}, hook_update

    from langchain_core.messages import HumanMessage, SystemMessage

    hook_update = await hook(
        {
            'messages': [
                SystemMessage(content='system'),
                SystemMessage(content='summary'),
                HumanMessage(content='再列一下可用服务器配置'),
                AIMessage(content='', tool_calls=[{'id': 'tc-hook-2', 'name': 'list_remote_profiles', 'args': {}}]),
            ],
            'cached_tool_context': cached_tool_context,
        }
    )
    assert 'messages' in hook_update, hook_update
    hook_routes = hook_store.read_events_by_type('route-hook', 'route_ownership')
    assert any(
        item.get('route') == 'post-hook-override' and item.get('override_kind') == 'cached_reply'
        for item in hook_routes
    ), hook_routes

    print('route ownership logging ok')


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
