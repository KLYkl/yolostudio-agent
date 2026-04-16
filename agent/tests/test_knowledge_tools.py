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
    assert result['retrieval_overview']['matched_rule_count'] >= 1
    assert result['matched_rule_overview'][0]['id']
    assert 'action_candidates' in result

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
    assert outcome['analysis_overview']['matched_rule_count'] >= 1
    assert outcome['action_candidates'][0]['tool'] == 'recommend_next_training_step'

    next_step = recommend_next_training_step(
        readiness={'missing_label_ratio': 0.31, 'ready': False},
        model_family='yolo',
        task_type='detection',
    )
    assert next_step['ok'] is True
    assert next_step['recommended_action'] == 'fix_data_quality'
    assert next_step['next_actions']
    assert next_step['source_summary']
    assert next_step['recommendation_overview']['recommended_action'] == 'fix_data_quality'
    assert next_step['action_candidates'][0]['action'] == 'fix_data_quality'
    print('knowledge tools ok')


if __name__ == '__main__':
    main()
