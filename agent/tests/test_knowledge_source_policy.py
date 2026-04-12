from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent_plan.agent.server.services.knowledge_service import KnowledgeService


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_json(
            root / 'knowledge/index.json',
            {
                'rule_files': [
                    {'path': 'knowledge/core/rules.json', 'family': 'generic', 'stage': 'post_training', 'task_type': 'detection'},
                    {'path': 'knowledge/cases/rules.json', 'family': 'generic', 'stage': 'post_training', 'task_type': 'detection'},
                ],
                'playbooks': [],
                'source_policy': {
                    'default_allowed_source_types': ['official', 'workflow'],
                    'default_blocked_source_types': ['test'],
                },
            },
        )
        _write_json(
            root / 'knowledge/core/rules.json',
            [
                {
                    'id': 'official_rule',
                    'family': 'generic',
                    'task_type': 'detection',
                    'stage': 'post_training',
                    'topic': 'training_metrics',
                    'signals': ['high_precision_low_recall'],
                    'interpretation': '官方规则',
                    'recommendation': '先看漏检样本',
                    'next_actions': ['检查漏检样本'],
                    'priority': 5,
                    'source_type': 'official',
                    'confidence': 0.9,
                }
            ],
        )
        _write_json(
            root / 'knowledge/cases/rules.json',
            [
                {
                    'id': 'case_rule',
                    'family': 'generic',
                    'task_type': 'detection',
                    'stage': 'post_training',
                    'topic': 'training_metrics',
                    'signals': ['high_precision_low_recall'],
                    'interpretation': '真实案例规则',
                    'recommendation': '直接用 case',
                    'next_actions': ['观察历史 case'],
                    'priority': 99,
                    'source_type': 'case',
                    'confidence': 0.7,
                }
            ],
        )

        service = KnowledgeService(project_root=root)
        default_rules = service.match_rules(
            topic='training_metrics',
            stage='post_training',
            model_family='yolo',
            task_type='detection',
            signals=['high_precision_low_recall'],
        )
        assert [item['id'] for item in default_rules] == ['official_rule']

        with_case = service.match_rules(
            topic='training_metrics',
            stage='post_training',
            model_family='yolo',
            task_type='detection',
            signals=['high_precision_low_recall'],
            include_case_sources=True,
        )
        assert [item['id'] for item in with_case][:2] == ['case_rule', 'official_rule']

        retrieved = service.retrieve_training_knowledge(
            topic='training_metrics',
            stage='post_training',
            model_family='yolo',
            task_type='detection',
            signals=['high_precision_low_recall'],
        )
        assert retrieved['source_summary'] == {'official': 1}
        assert retrieved['knowledge_policy']['case_sources_included'] is False
        print('knowledge source policy ok')


if __name__ == '__main__':
    main()
