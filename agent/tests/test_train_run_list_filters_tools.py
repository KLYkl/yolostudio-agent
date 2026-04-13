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
    def list_training_runs(
        self,
        limit: int = 5,
        run_state: str = '',
        analysis_ready: bool | None = None,
        model_keyword: str = '',
        data_keyword: str = '',
    ):
        return {
            'ok': True,
            'summary': '找到 1 条最近训练记录（筛选: 状态=failed）',
            'count': 1,
            'limit': limit,
            'applied_filters': {
                'run_state': run_state,
                'analysis_ready': analysis_ready,
                'model_keyword': model_keyword or None,
                'data_keyword': data_keyword or None,
            },
            'runs': [
                {'run_id': 'train_log_200', 'run_state': 'failed', 'observation_stage': 'final'},
            ],
            'next_actions': ['可继续调用 inspect_training_run'],
        }


def main() -> None:
    original_service = train_tools.service
    train_tools.service = _DummyService()
    try:
        result = train_tools.list_training_runs(limit=3, run_state='failed', analysis_ready=False)
        assert result['ok'] is True
        assert result['count'] == 1
        assert result['applied_filters']['run_state'] == 'failed'
        assert result['applied_filters']['analysis_ready'] is False
        print('train run list filters tools ok')
    finally:
        train_tools.service = original_service


if __name__ == '__main__':
    main()
