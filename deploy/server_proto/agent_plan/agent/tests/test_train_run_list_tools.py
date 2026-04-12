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
    def list_training_runs(self, limit: int = 5):
        return {
            'ok': True,
            'summary': '找到 2 条最近训练记录',
            'count': 2,
            'limit': limit,
            'runs': [
                {'run_id': 'train_log_100', 'run_state': 'stopped', 'observation_stage': 'final'},
                {'run_id': 'train_log_090', 'run_state': 'completed', 'observation_stage': 'final'},
            ],
            'next_actions': ['可继续调用 summarize_training_run'],
        }


def main() -> None:
    original_service = train_tools.service
    train_tools.service = _DummyService()
    try:
        result = train_tools.list_training_runs(limit=2)
        assert result['ok'] is True
        assert result['count'] == 2
        assert result['runs'][0]['run_id'] == 'train_log_100'
        assert result['next_actions']
        print('train run list tools ok')
    finally:
        train_tools.service = original_service


if __name__ == '__main__':
    main()
