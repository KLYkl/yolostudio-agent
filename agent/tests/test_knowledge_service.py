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

from yolostudio_agent.agent.server.services.knowledge_service import KnowledgeService


def main() -> None:
    service = KnowledgeService(project_root=Path(__file__).resolve().parents[2])

    retrieved = service.retrieve_training_knowledge(
        topic='data_quality',
        stage='pre_training',
        model_family='yolo',
        task_type='detection',
        signals=['high_missing_labels'],
    )
    assert retrieved['ok'] is True
    assert 'generic_pre_high_missing_labels' in retrieved['matched_rule_ids']
    assert retrieved['playbooks']
    assert retrieved['source_summary'].get('official', 0) >= 1

    analyzed = service.analyze_training_outcome(
        metrics={
            'latest_metrics': {
                'metrics': {
                    'precision': 0.82,
                    'recall': 0.36,
                    'mAP50': 0.33,
                    'epoch': 12,
                    'total_epochs': 80,
                }
            }
        },
        comparison={
            'ok': True,
            'highlights': ['mAP50提升 +0.1200'],
            'metric_deltas': {
                'map50': {'left': 0.45, 'right': 0.33, 'delta': 0.12},
                'precision': {'left': 0.82, 'right': 0.74, 'delta': 0.08},
            },
        },
        model_family='yolo',
        task_type='detection',
    )
    assert analyzed['ok'] is True
    assert 'high_precision_low_recall' in analyzed['signals']
    assert 'comparison_map_improved' in analyzed['signals']
    assert 'generic_post_high_precision_low_recall' in analyzed['matched_rule_ids']
    assert 'generic_post_metrics_missing' not in analyzed['matched_rule_ids']
    assert analyzed['matched_rule_ids']
    assert analyzed['interpretation']
    assert analyzed['source_summary']
    assert analyzed['analysis_overview']['comparison_attached'] is True

    insufficient = service.analyze_training_outcome(
        metrics={
            'run_state': 'completed',
            'facts': ['return_code=0'],
        },
        model_family='yolo',
        task_type='detection',
    )
    assert insufficient['ok'] is True
    assert 'metrics_missing' in insufficient['signals']
    assert insufficient['assessment'] == 'collect_metrics_first'
    assert 'generic_post_metrics_missing' in insufficient['matched_rule_ids']

    recommended = service.recommend_next_training_step(
        readiness={
            'missing_label_ratio': 0.42,
            'ready': False,
            'blockers': ['missing labels'],
        },
        comparison={
            'ok': True,
            'highlights': ['recall下降 -0.0600'],
            'metric_deltas': {
                'recall': {'left': 0.36, 'right': 0.42, 'delta': -0.06},
            },
        },
        model_family='yolo',
        task_type='detection',
    )
    assert recommended['ok'] is True
    assert recommended['recommended_action'] == 'fix_data_quality'
    assert recommended['matched_rule_ids']
    assert recommended['source_summary']
    assert recommended['recommendation_overview']['comparison_attached'] is True

    short_window = service.recommend_next_training_step(
        status={
            'run_state': 'completed',
            'epoch': 1,
            'total_epochs': 1,
            'metrics': {
                'precision': 0.002,
                'recall': 0.667,
                'map50': 0.006,
                'map': 0.002,
            },
        },
        model_family='yolo',
        task_type='detection',
    )
    assert short_window['ok'] is True
    assert short_window['recommended_action'] == 'continue_observing'
    assert 'generic_next_continue_observing' in short_window['matched_rule_ids']
    assert 'generic_next_fix_data_before_tuning' not in short_window['matched_rule_ids']

    low_map = service.recommend_next_training_step(
        status={
            'run_state': 'completed',
            'epoch': 30,
            'total_epochs': 30,
            'metrics': {
                'precision': 0.41,
                'recall': 0.48,
                'map50': 0.19,
                'map': 0.08,
            },
        },
        model_family='yolo',
        task_type='detection',
    )
    assert low_map['ok'] is True
    assert low_map['recommended_action'] == 'run_error_analysis'
    assert 'generic_next_low_map_error_analysis' in low_map['matched_rule_ids']
    print('knowledge service ok')


if __name__ == '__main__':
    main()
