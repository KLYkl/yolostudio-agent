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

import yolostudio_agent.agent.server.tools.train_tools as train_tools


class _DummyService:
    def inspect_training_run(self, run_id: str = ''):
        return {
            'ok': True,
            'summary': '训练记录详情已就绪',
            'selected_run_id': run_id or 'train_log_100',
            'run_state': 'completed',
            'observation_stage': 'final',
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'progress': {'epoch': 3, 'total_epochs': 3, 'progress_ratio': 1.0},
            'next_actions': ['可继续调用 analyze_training_outcome'],
        }


def main() -> None:
    original_service = train_tools.service
    train_tools.service = _DummyService()
    try:
        result = train_tools.inspect_training_run(run_id='train_log_100')
        assert result['ok'] is True
        assert result['selected_run_id'] == 'train_log_100'
        assert result['run_state'] == 'completed'
        assert result['next_actions']
        print('train run inspect tools ok')
    finally:
        train_tools.service = original_service


if __name__ == '__main__':
    main()
