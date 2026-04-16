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
    def status(self):
        return {
            'ok': True,
            'running': False,
            'summary': '当前无训练在跑，最近一次训练已完成，return_code=0',
            'latest_metrics': {'ok': True, 'metrics': {'precision': 0.71, 'recall': 0.58, 'map50': 0.51, 'epoch': 8, 'total_epochs': 8}},
            'training_facts': {
                'run_state': 'completed',
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'observation_stage': 'final',
            'progress': {'epoch': 8, 'total_epochs': 8, 'progress_ratio': 1.0},
            'signals': ['training_completed'],
            'facts': ['precision=0.710', 'recall=0.580'],
            },
            'analysis_ready': True,
            'minimum_facts_ready': True,
            'run_state': 'completed',
            'observation_stage': 'final',
            'progress': {'epoch': 8, 'total_epochs': 8, 'progress_ratio': 1.0},
            'signals': ['training_completed'],
            'facts': ['precision=0.710', 'recall=0.580'],
        }

    def summarize_run(self):
        return {
            'ok': True,
            'summary': '训练结果汇总: 最近一次训练已完成，并且已有可分析指标。',
            'run_state': 'completed',
            'analysis_ready': True,
            'observation_stage': 'final',
            'latest_metrics': {'ok': True, 'metrics': {'precision': 0.71, 'recall': 0.58, 'map50': 0.51, 'map': 0.31}},
            'metrics': {'precision': 0.71, 'recall': 0.58, 'map50': 0.51, 'map': 0.31},
            'signals': ['training_completed'],
            'facts': ['precision=0.710', 'recall=0.580'],
            'next_actions': ['可继续调用 analyze_training_outcome 解释训练效果'],
        }



def main() -> None:
    original_service = train_tools.service
    train_tools.service = _DummyService()
    try:
        status = train_tools.check_training_status()
        assert status['ok'] is True
        assert status['analysis_ready'] is True
        assert status['minimum_facts_ready'] is True
        assert status['progress']['progress_ratio'] == 1.0
        assert status['observation_stage'] == 'final'
        assert status['next_actions'][0].startswith('可继续调用 summarize_training_run')
        assert status['status_overview']['run_state'] == 'completed'
        assert status['action_candidates'][0]['tool'] == 'summarize_training_run'

        summary = train_tools.summarize_training_run()
        assert summary['ok'] is True
        assert summary['run_state'] == 'completed'
        assert summary['observation_stage'] == 'final'
        assert summary['metrics']['precision'] == 0.71
        assert summary['latest_metrics']['metrics']['precision'] == 0.71
        assert summary['next_actions']
        assert summary['summary_overview']['run_state'] == 'completed'
        assert summary['action_candidates'][0]['tool'] == 'analyze_training_outcome'
        print('train tools summary ok')
    finally:
        train_tools.service = original_service


if __name__ == '__main__':
    main()
