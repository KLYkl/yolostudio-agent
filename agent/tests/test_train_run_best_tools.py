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
    def select_best_training_run(self, limit: int = 5):
        return {
            'ok': True,
            'summary': '最佳训练记录: train_log_200，状态=completed，mAP50=0.465',
            'best_run_id': 'train_log_200',
            'best_run': {'run_id': 'train_log_200', 'run_state': 'completed'},
            'ranking_basis': '状态=completed，mAP50=0.465',
            'candidates': [{'run_id': 'train_log_200'}, {'run_id': 'train_log_100'}],
            'next_actions': ['可继续调用 inspect_training_run'],
        }


def main() -> None:
    original_service = train_tools.service
    train_tools.service = _DummyService()
    try:
        result = train_tools.select_best_training_run(limit=3)
        assert result['ok'] is True
        assert result['best_run_id'] == 'train_log_200'
        assert result['next_actions']
        print('train run best tools ok')
    finally:
        train_tools.service = original_service


if __name__ == '__main__':
    main()
