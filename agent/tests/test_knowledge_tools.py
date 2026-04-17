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

import yolostudio_agent.agent.server.tools.knowledge_tools as knowledge_tools
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
        comparison={
            'ok': True,
            'highlights': ['precision提升 +0.1000'],
            'metric_deltas': {
                'precision': {'left': 0.84, 'right': 0.74, 'delta': 0.1},
            },
        },
        model_family='yolo',
        task_type='detection',
    )
    assert outcome['ok'] is True
    assert 'high_precision_low_recall' in outcome['signals']
    assert 'comparison_precision_improved' in outcome['signals']
    assert outcome['next_actions']
    assert outcome['source_summary']
    assert outcome['analysis_overview']['matched_rule_count'] >= 1
    assert outcome['analysis_overview']['comparison_attached'] is True
    assert outcome['action_candidates'][0]['tool'] == 'recommend_next_training_step'

    next_step = recommend_next_training_step(
        readiness={'missing_label_ratio': 0.31, 'ready': False},
        comparison={
            'ok': True,
            'highlights': ['recall下降 -0.0600'],
            'metric_deltas': {
                'recall': {'left': 0.31, 'right': 0.37, 'delta': -0.06},
            },
        },
        model_family='yolo',
        task_type='detection',
    )
    assert next_step['ok'] is True
    assert next_step['recommended_action'] == 'fix_data_quality'
    assert next_step['next_actions']
    assert next_step['source_summary']
    assert next_step['recommendation_overview']['recommended_action'] == 'fix_data_quality'
    assert next_step['recommendation_overview']['comparison_attached'] is True
    assert next_step['action_candidates'][0]['action'] == 'fix_data_quality'

    class _DummyKnowledgeService:
        def __init__(self) -> None:
            self.calls: dict[str, dict[str, object]] = {}

        def retrieve_training_knowledge(self, **kwargs):
            self.calls['retrieve_training_knowledge'] = dict(kwargs)
            return {'ok': True, 'summary': 'knowledge'}

        def analyze_training_outcome(self, **kwargs):
            self.calls['analyze_training_outcome'] = dict(kwargs)
            return {'ok': True, 'signals': [], 'next_actions': ['ok']}

        def recommend_next_training_step(self, **kwargs):
            self.calls['recommend_next_training_step'] = dict(kwargs)
            return {'ok': True, 'recommended_action': 'continue_observing'}

    original_service = knowledge_tools.service
    dummy_service = _DummyKnowledgeService()
    knowledge_tools.service = dummy_service
    try:
        retrieval = retrieve_training_knowledge(signals='high_precision_low_recall, comparison_map_improved')
        assert retrieval['ok'] is True
        assert dummy_service.calls['retrieve_training_knowledge']['signals'] == [
            'high_precision_low_recall',
            'comparison_map_improved',
        ]

        analyzed = analyze_training_outcome(
            metrics='{"precision": 0.84}',
            data_quality='约 6.1% 的图片缺少标签，可能限制模型性能。',
            comparison='{"ok": true, "metric_deltas": {"map50": {"delta": 0.03}}}',
        )
        assert analyzed['ok'] is True
        analyze_call = dummy_service.calls['analyze_training_outcome']
        assert analyze_call['metrics'] == {'precision': 0.84}
        assert analyze_call['data_quality'] == {'description': '约 6.1% 的图片缺少标签，可能限制模型性能。'}
        assert analyze_call['comparison'] == {'ok': True, 'metric_deltas': {'map50': {'delta': 0.03}}}

        recommended = recommend_next_training_step(
            readiness='{"ready": false}',
            health='数据质量一般，需要先补标签。',
        )
        assert recommended['ok'] is True
        recommend_call = dummy_service.calls['recommend_next_training_step']
        assert recommend_call['readiness'] == {'ready': False}
        assert recommend_call['health'] == {'description': '数据质量一般，需要先补标签。'}
    finally:
        knowledge_tools.service = original_service

    print('knowledge tools ok')


if __name__ == '__main__':
    main()
