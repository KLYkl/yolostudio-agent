from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.services.knowledge_service import KnowledgeService


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
        model_family='yolo',
        task_type='detection',
    )
    assert analyzed['ok'] is True
    assert 'high_precision_low_recall' in analyzed['signals']
    assert analyzed['matched_rule_ids']
    assert analyzed['interpretation']

    recommended = service.recommend_next_training_step(
        readiness={
            'missing_label_ratio': 0.42,
            'ready': False,
            'blockers': ['missing labels'],
        },
        model_family='yolo',
        task_type='detection',
    )
    assert recommended['ok'] is True
    assert recommended['recommended_action'] == 'fix_data_quality'
    assert recommended['matched_rule_ids']
    print('knowledge service ok')


if __name__ == '__main__':
    main()
