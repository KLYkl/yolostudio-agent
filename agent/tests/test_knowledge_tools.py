from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.tools.knowledge_tools import (
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

    outcome = analyze_training_outcome(
        metrics={'precision': 0.84, 'recall': 0.38, 'map50': 0.42},
        data_quality={'duplicate_groups': 1},
        model_family='yolo',
        task_type='detection',
    )
    assert outcome['ok'] is True
    assert 'high_precision_low_recall' in outcome['signals']
    assert outcome['next_actions']

    next_step = recommend_next_training_step(
        readiness={'missing_label_ratio': 0.31, 'ready': False},
        model_family='yolo',
        task_type='detection',
    )
    assert next_step['ok'] is True
    assert next_step['recommended_action'] == 'fix_data_quality'
    assert next_step['next_actions']
    print('knowledge tools ok')


if __name__ == '__main__':
    main()
