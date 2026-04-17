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

from yolostudio_agent.agent.client.cached_tool_reply_service import (
    build_cached_tool_context_payload,
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
    assert cached_tool_context['select_best_training_run']['best_run_id'] == 'run-best'
    assert cached_tool_context['inspect_training_run']['selected_run_id'] == 'run-a'
    assert cached_tool_context['check_training_loop_status']['loop_id'] == 'loop-a'
    assert cached_tool_context['list_remote_profiles']['default_profile'] == 'lab'
    assert cached_tool_context['extract_images']['output_dir'] == '/tmp/extract_run'
    assert cached_tool_context['retrieve_training_knowledge']['topic'] == 'training_metrics'
    assert cached_tool_context['check_training_status']['run_state'] == 'completed'
    assert cached_tool_context['summarize_training_run']['run_state'] == 'completed'

    print('cached tool reply service ok')


if __name__ == '__main__':
    main()
