from __future__ import annotations

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

from yolostudio_agent.agent.client.cached_tool_reply_service import (
    build_cached_tool_context_payload,
    resolve_cached_tool_reply,
)
from yolostudio_agent.agent.client.session_state import SessionState


def main() -> None:
    state = SessionState(session_id='cached-tool-reply')
    state.active_training.recent_runs = [{'run_id': 'run-a', 'run_state': 'completed'}]
    state.active_training.best_run_selection = {
        'ok': True,
        'summary': '最佳训练记录已选出',
        'best_run_id': 'run-best',
        'best_run': {'run_id': 'run-best', 'run_state': 'completed'},
    }
    state.active_training.last_run_inspection = {
        'ok': True,
        'summary': 'run-a 详情已就绪',
        'selected_run_id': 'run-a',
    }
    state.active_training.last_run_comparison = {
        'ok': True,
        'summary': 'run-a 与 run-b 的对比已完成',
        'left_run_id': 'run-a',
        'right_run_id': 'run-b',
    }
    state.active_training.last_loop_status = {
        'ok': True,
        'summary': 'loop-a 正在等待审阅',
        'loop_id': 'loop-a',
    }
    state.active_dataset.last_extract_result = {
        'ok': True,
        'summary': '图片抽取完成: 实际抽取 18 张图片，复制标签 18 个',
        'output_dir': '/tmp/extract_run',
        'source_path': '/data/raw/images',
    }
    state.active_knowledge.last_retrieval = {
        'ok': True,
        'summary': '知识检索完成: 当前更像高精度低召回。',
        'topic': 'training_metrics',
        'stage': 'post_training',
        'signals': ['high_precision_low_recall'],
    }
    state.active_training.last_status = {
        'ok': True,
        'summary': '训练已完成: epoch 10/10, map50=0.61',
        'run_state': 'completed',
    }
    state.active_training.training_run_summary = {
        'ok': True,
        'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
        'run_state': 'completed',
        'action_candidates': [{'tool': 'analyze_training_outcome', 'description': '继续分析训练结果'}],
    }
    state.active_remote_transfer.last_profile_listing = {
        'ok': True,
        'summary': '远端 profile 1 个 / SSH alias 1 个。 默认 profile: lab。',
        'default_profile': 'lab',
    }

    cached_tool_context = build_cached_tool_context_payload(state)
    assert cached_tool_context is not None

    best_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='哪次训练最好？'),
        AIMessage(content='', tool_calls=[{'id': 'tc-1', 'name': 'select_best_training_run', 'args': {}}]),
    ]
    best = resolve_cached_tool_reply(best_messages, cached_tool_context=cached_tool_context)
    assert best is not None
    assert best[0] == 'select_best_training_run'
    assert best[1]['best_run_id'] == 'run-best'

    inspect_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='看下 run-a 详情'),
        AIMessage(content='', tool_calls=[{'id': 'tc-2', 'name': 'inspect_training_run', 'args': {'run_id': 'run-a'}}]),
    ]
    inspect = resolve_cached_tool_reply(inspect_messages, cached_tool_context=cached_tool_context)
    assert inspect is not None
    assert inspect[0] == 'inspect_training_run'

    inspect_mismatch_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='看下 run-z 详情'),
        AIMessage(content='', tool_calls=[{'id': 'tc-3', 'name': 'inspect_training_run', 'args': {'run_id': 'run-z'}}]),
    ]
    assert resolve_cached_tool_reply(inspect_mismatch_messages, cached_tool_context=cached_tool_context) is None

    loop_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='环训练现在怎么样？'),
        AIMessage(content='', tool_calls=[{'id': 'tc-4', 'name': 'check_training_loop_status', 'args': {'loop_id': 'loop-a'}}]),
    ]
    loop_status = resolve_cached_tool_reply(loop_messages, cached_tool_context=cached_tool_context)
    assert loop_status is not None
    assert loop_status[0] == 'check_training_loop_status'
    assert loop_status[1]['loop_id'] == 'loop-a'

    remote_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='再列一下可用服务器配置'),
        AIMessage(content='', tool_calls=[{'id': 'tc-5', 'name': 'list_remote_profiles', 'args': {}}]),
    ]
    remote_profiles = resolve_cached_tool_reply(remote_messages, cached_tool_context=cached_tool_context)
    assert remote_profiles is not None
    assert remote_profiles[0] == 'list_remote_profiles'
    assert remote_profiles[1]['default_profile'] == 'lab'

    extract_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='再说一下刚才抽图结果'),
        AIMessage(content='', tool_calls=[{'id': 'tc-6', 'name': 'extract_images', 'args': {'output_dir': '/tmp/extract_run'}}]),
    ]
    extract_result = resolve_cached_tool_reply(extract_messages, cached_tool_context=cached_tool_context)
    assert extract_result is not None
    assert extract_result[0] == 'extract_images'
    assert extract_result[1]['output_dir'] == '/tmp/extract_run'

    knowledge_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='刚才那条规则再详细一点'),
        AIMessage(content='', tool_calls=[{'id': 'tc-7', 'name': 'retrieve_training_knowledge', 'args': {'topic': 'training_metrics', 'stage': 'post_training', 'signals': ['high_precision_low_recall']}}]),
    ]
    knowledge_result = resolve_cached_tool_reply(knowledge_messages, cached_tool_context=cached_tool_context)
    assert knowledge_result is not None
    assert knowledge_result[0] == 'retrieve_training_knowledge'
    assert knowledge_result[1]['topic'] == 'training_metrics'

    status_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='看下训练状态'),
        AIMessage(content='', tool_calls=[{'id': 'tc-8', 'name': 'check_training_status', 'args': {}}]),
    ]
    status_result = resolve_cached_tool_reply(status_messages, cached_tool_context=cached_tool_context)
    assert status_result is not None
    assert status_result[0] == 'check_training_status'
    assert status_result[1]['run_state'] == 'completed'

    summary_messages = [
        SystemMessage(content='system'),
        HumanMessage(content='数据总结'),
        AIMessage(content='', tool_calls=[{'id': 'tc-9', 'name': 'summarize_training_run', 'args': {}}]),
    ]
    summary_result = resolve_cached_tool_reply(summary_messages, cached_tool_context=cached_tool_context)
    assert summary_result is not None
    assert summary_result[0] == 'summarize_training_run'
    assert summary_result[1]['run_state'] == 'completed'

    print('cached tool reply service ok')


if __name__ == '__main__':
    main()
