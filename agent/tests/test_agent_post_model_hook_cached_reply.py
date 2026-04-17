from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from yolostudio_agent.agent.client.agent_client import _build_agent_post_model_hook
from yolostudio_agent.agent.client.cached_tool_reply_service import build_cached_tool_context_payload
from yolostudio_agent.agent.client.session_state import SessionState


class _FakePlannerResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakePlannerLlm:
    async def ainvoke(self, messages):
        text = '\n'.join(str(getattr(message, 'content', message)) for message in messages)
        if '结果说明器' in text:
            if 'extract_images' in text or '/tmp/extract_run' in text:
                return _FakePlannerResponse('图片抽取完成，当前输出目录是 /tmp/extract_run，本次实际抽取 18 张图片。')
            if 'list_remote_profiles' in text or 'default_profile' in text:
                return _FakePlannerResponse('当前可用默认服务器配置是 lab，可继续用这套 profile 走远端操作。')
            return _FakePlannerResponse('当前最佳训练是 run-best，建议继续查看这条训练记录的详细指标。')
        return _FakePlannerResponse('')


async def _run() -> None:
    state = SessionState(session_id='agent-post-hook-cached')
    state.active_training.best_run_selection = {
        'ok': True,
        'summary': '最佳训练记录已选出',
        'best_run_id': 'run-best',
        'best_run': {'run_id': 'run-best', 'run_state': 'completed'},
    }
    state.active_remote_transfer.last_profile_listing = {
        'ok': True,
        'summary': '远端 profile 1 个 / SSH alias 1 个。 默认 profile: lab。',
        'default_profile': 'lab',
        'profiles': [{'name': 'lab', 'target_label': 'lab'}],
    }
    state.active_dataset.last_extract_result = {
        'ok': True,
        'summary': '图片抽取完成: 实际抽取 18 张图片，复制标签 18 个',
        'output_dir': '/tmp/extract_run',
        'action_candidates': [{'tool': 'scan_dataset', 'description': '可继续对输出目录做数据集质量检查'}],
    }
    cached_tool_context = build_cached_tool_context_payload(state)
    assert cached_tool_context is not None

    hook = _build_agent_post_model_hook(_FakePlannerLlm())
    update = await hook(
        {
            'messages': [
                SystemMessage(content='system'),
                SystemMessage(content='summary'),
                HumanMessage(content='哪次训练最好？'),
                AIMessage(content='', tool_calls=[{'id': 'tc-1', 'name': 'select_best_training_run', 'args': {}}]),
            ],
            'cached_tool_context': cached_tool_context,
        }
    )
    assert update == {}, update

    remote_update = await hook(
        {
            'messages': [
                SystemMessage(content='system'),
                SystemMessage(content='summary'),
                HumanMessage(content='再列一下可用服务器配置'),
                AIMessage(content='', tool_calls=[{'id': 'tc-2', 'name': 'list_remote_profiles', 'args': {}}]),
            ],
            'cached_tool_context': cached_tool_context,
        }
    )
    assert remote_update == {}, remote_update

    extract_update = await hook(
        {
            'messages': [
                SystemMessage(content='system'),
                SystemMessage(content='summary'),
                HumanMessage(content='再说一下刚才抽图结果'),
                AIMessage(content='', tool_calls=[{'id': 'tc-3', 'name': 'extract_images', 'args': {'output_dir': '/tmp/extract_run'}}]),
            ],
            'cached_tool_context': cached_tool_context,
        }
    )
    assert extract_update == {}, extract_update
    print('agent post model hook cached reply ok')


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
