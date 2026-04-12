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
    def compare_training_runs(self, left_run_id: str = '', right_run_id: str = ''):
        return {
            'ok': True,
            'summary': '训练对比完成',
            'left_run_id': left_run_id or 'train_log_200',
            'right_run_id': right_run_id or 'train_log_100',
            'metric_deltas': {'precision': {'left': 0.52, 'right': 0.42, 'delta': 0.1}},
            'highlights': ['precision提升 +0.1000'],
            'next_actions': ['可继续调用 inspect_training_run'],
        }


def main() -> None:
    original_service = train_tools.service
    train_tools.service = _DummyService()
    try:
        result = train_tools.compare_training_runs('train_log_200', 'train_log_100')
        assert result['ok'] is True
        assert result['left_run_id'] == 'train_log_200'
        assert result['right_run_id'] == 'train_log_100'
        assert result['next_actions']
        print('train run compare tools ok')
    finally:
        train_tools.service = original_service


if __name__ == '__main__':
    main()
