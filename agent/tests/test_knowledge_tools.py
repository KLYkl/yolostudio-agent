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

from yolostudio_agent.agent.server.tools.knowledge_tools import (
    analyze_training_outcome,
    recommend_next_training_step,
    retrieve_training_knowledge,
)


def main() -> None:
    result = retrieve_training_knowledge(
        topic='training_metrics',
        stage='post_training',
        model_family='yolo',
        task_type='detection',
        signals=['high_precision_low_recall'],
    )
    assert result['ok'] is True
    assert 'generic_post_high_precision_low_recall' in result['matched_rule_ids']
    assert result['source_summary']

    outcome = analyze_training_outcome(
        metrics={'precision': 0.84, 'recall': 0.38, 'map50': 0.42},
        data_quality={'duplicate_groups': 1},
        model_family='yolo',
        task_type='detection',
    )
    assert outcome['ok'] is True
    assert 'high_precision_low_recall' in outcome['signals']
    assert outcome['next_actions']
    assert outcome['source_summary']

    compared_outcome = analyze_training_outcome(
        metrics={'run_state': 'completed', 'analysis_ready': True, 'metrics': {'precision': 0.84, 'recall': 0.38, 'map50': 0.42}},
        comparison={
            'left_run_id': 'train_log_200',
            'right_run_id': 'train_log_100',
            'metric_deltas': {
                'precision': {'left': 0.84, 'right': 0.74, 'delta': 0.1},
                'map50': {'left': 0.42, 'right': 0.31, 'delta': 0.11},
            },
        },
        model_family='yolo',
        task_type='detection',
    )
    assert compared_outcome['ok'] is True
    assert 'latest_run_improved' in compared_outcome['signals']
    assert compared_outcome['source_summary']

    next_step = recommend_next_training_step(
        readiness={'missing_label_ratio': 0.31, 'ready': False},
        model_family='yolo',
        task_type='detection',
    )
    assert next_step['ok'] is True
    assert next_step['recommended_action'] == 'fix_data_quality'
    assert next_step['next_actions']
    assert next_step['source_summary']

    compared_next_step = recommend_next_training_step(
        status={'run_state': 'completed', 'analysis_ready': True, 'metrics': {'precision': 0.82, 'recall': 0.36, 'map50': 0.33, 'map': 0.18}},
        comparison={
            'left_run_id': 'train_log_200',
            'right_run_id': 'train_log_100',
            'metric_deltas': {'map50': {'left': 0.33, 'right': 0.25, 'delta': 0.08}},
        },
        model_family='yolo',
        task_type='detection',
    )
    assert compared_next_step['ok'] is True
    assert 'latest_run_improved' in compared_next_step['signals']
    assert compared_next_step['source_summary']
    print('knowledge tools ok')


if __name__ == '__main__':
    main()
