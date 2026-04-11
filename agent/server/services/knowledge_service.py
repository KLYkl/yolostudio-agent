from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace('%', '')
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return None


class KnowledgeService:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self.knowledge_root = self.project_root / 'knowledge'
        self.index_path = self.knowledge_root / 'index.json'

    @lru_cache(maxsize=1)
    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            raise FileNotFoundError(f'knowledge index not found: {self.index_path}')
        return json.loads(self.index_path.read_text(encoding='utf-8'))

    @lru_cache(maxsize=1)
    def _load_rules(self) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        for item in self._load_index().get('rule_files', []):
            path = self.project_root / item['path']
            payload = json.loads(path.read_text(encoding='utf-8'))
            for rule in payload:
                merged = dict(rule)
                merged.setdefault('_file', str(path))
                rules.append(merged)
        return rules

    def _match_playbooks(self, *, topic: str, stage: str, model_family: str, task_type: str, limit: int = 2) -> list[dict[str, Any]]:
        del task_type
        topic_lower = topic.strip().lower()
        matches: list[tuple[int, dict[str, Any]]] = []
        for item in self._load_index().get('playbooks', []):
            families = set(item.get('families') or [])
            if model_family and families and model_family not in families and 'generic' not in families:
                continue
            stages = set(item.get('stages') or [])
            if stage and stages and stage not in stages:
                continue
            score = 0
            topics = [str(x).lower() for x in item.get('topics') or []]
            if topic_lower:
                if topic_lower in topics:
                    score += 5
                elif any(topic_lower in value for value in topics):
                    score += 3
                elif topic_lower in str(item.get('title', '')).lower() or topic_lower in str(item.get('summary', '')).lower():
                    score += 2
            if score <= 0 and topic_lower:
                continue
            path = self.project_root / item['path']
            excerpt = ''
            if path.exists():
                body_lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                excerpt = ' '.join(line for line in body_lines if not line.startswith('#'))[:180]
            record = dict(item)
            record['path'] = str(path)
            record['excerpt'] = excerpt
            matches.append((score, record))
        matches.sort(key=lambda item: (-item[0], item[1].get('title', '')))
        return [item for _, item in matches[:limit]]

    @staticmethod
    def _topic_score(rule: dict[str, Any], topic: str) -> int:
        topic_lower = topic.strip().lower()
        if not topic_lower:
            return 0
        score = 0
        if topic_lower == str(rule.get('topic', '')).lower():
            score += 6
        haystack = ' '.join(
            [str(rule.get('topic', ''))]
            + [str(item) for item in rule.get('tags') or []]
            + [str(item) for item in rule.get('keywords') or []]
        ).lower()
        for token in set(topic_lower.split() + [topic_lower]):
            if token and token in haystack:
                score += 1
        return score

    @staticmethod
    def _signal_score(rule: dict[str, Any], signals: list[str]) -> tuple[int, list[str]]:
        requested = {item.strip().lower() for item in signals if str(item).strip()}
        available = {str(item).strip().lower() for item in rule.get('signals') or [] if str(item).strip()}
        matched = sorted(requested & available)
        return len(matched), matched

    def match_rules(
        self,
        *,
        topic: str = '',
        stage: str = '',
        model_family: str = 'yolo',
        task_type: str = 'detection',
        signals: list[str] | None = None,
        max_rules: int = 5,
    ) -> list[dict[str, Any]]:
        requested_family = (model_family or 'generic').strip().lower()
        requested_stage = (stage or '').strip().lower()
        requested_task_type = (task_type or '').strip().lower()
        requested_signals = [str(item).strip().lower() for item in signals or [] if str(item).strip()]

        candidates: list[tuple[float, dict[str, Any]]] = []
        for rule in self._load_rules():
            family = str(rule.get('family', 'generic')).strip().lower()
            if family not in {'generic', requested_family}:
                continue
            if requested_stage and str(rule.get('stage', '')).strip().lower() != requested_stage:
                continue
            rule_task_type = str(rule.get('task_type', '')).strip().lower()
            if requested_task_type and rule_task_type not in {'', 'any', requested_task_type}:
                continue
            topic_score = self._topic_score(rule, topic)
            signal_score, matched_signals = self._signal_score(rule, requested_signals)
            if requested_signals and signal_score == 0 and topic_score == 0:
                continue
            if topic and topic_score == 0 and not matched_signals:
                continue
            score = float(rule.get('priority', 0)) * 10.0
            score += topic_score * 3.0
            score += signal_score * 5.0
            if family == requested_family:
                score += 2.0
            enriched = dict(rule)
            enriched['matched_signals'] = matched_signals
            candidates.append((score, enriched))

        candidates.sort(key=lambda item: (-item[0], -int(item[1].get('priority', 0)), item[1].get('id', '')))
        return [item for _, item in candidates[:max_rules]]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
        return ordered

    @staticmethod
    def _extract_metric_bundle(metrics: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(metrics or {})
        if isinstance(payload.get('latest_metrics'), dict):
            payload = dict(payload['latest_metrics'])
        if isinstance(payload.get('metrics'), dict):
            payload = dict(payload['metrics'])
        return payload

    @classmethod
    def _metric_value(cls, metrics: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key in metrics:
                value = _to_float(metrics.get(key))
                if value is not None:
                    return value
        return None

    @classmethod
    def _derive_metric_signals(cls, metrics: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        payload = cls._extract_metric_bundle(metrics)
        if not payload:
            return ['metrics_missing'], ['当前没有可分析的训练指标']

        facts: list[str] = []
        signals: list[str] = []
        epoch = _to_int(payload.get('epoch'))
        total_epochs = _to_int(payload.get('total_epochs'))
        if epoch is not None and total_epochs:
            facts.append(f'训练进度 {epoch}/{total_epochs}')
            if total_epochs > 0 and epoch / total_epochs < 0.2:
                signals.append('early_training_observation')

        precision = cls._metric_value(payload, 'precision', 'metrics/precision(B)', 'box_precision', 'P')
        recall = cls._metric_value(payload, 'recall', 'metrics/recall(B)', 'box_recall', 'R')
        map50 = cls._metric_value(payload, 'map50', 'mAP50', 'metrics/mAP50(B)')
        map5095 = cls._metric_value(payload, 'map', 'mAP50-95', 'metrics/mAP50-95(B)')

        if precision is not None:
            facts.append(f'precision={precision:.3f}')
        if recall is not None:
            facts.append(f'recall={recall:.3f}')
        if map50 is not None:
            facts.append(f'mAP50={map50:.3f}')
        if map5095 is not None:
            facts.append(f'mAP50-95={map5095:.3f}')

        if precision is None and recall is None and map50 is None and map5095 is None:
            if any(key in payload for key in ('box_loss', 'cls_loss', 'dfl_loss')):
                signals.append('loss_only_metrics')
                losses = []
                for name in ('box_loss', 'cls_loss', 'dfl_loss'):
                    if payload.get(name) is not None:
                        losses.append(f"{name}={payload[name]}")
                if losses:
                    facts.append('仅有训练损失: ' + ', '.join(losses))
            else:
                signals.append('metrics_missing')
            return cls._dedupe(signals), facts

        if precision is not None and recall is not None:
            if precision >= 0.75 and recall < 0.50:
                signals.append('high_precision_low_recall')
            if precision < 0.50 and recall >= 0.75:
                signals.append('low_precision_high_recall')
        low_map_value = map50 if map50 is not None else map5095
        if low_map_value is not None and low_map_value < 0.40:
            signals.append('low_map_overall')
        return cls._dedupe(signals), facts

    @staticmethod
    def _derive_data_quality_signals(data: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        payload = dict(data or {})
        if not payload:
            return [], []
        signals: list[str] = []
        facts: list[str] = []

        missing_ratio = _to_float(payload.get('missing_label_ratio'))
        if missing_ratio is not None:
            facts.append(f'缺失标签比例={missing_ratio:.2f}')
            if missing_ratio >= 0.30:
                signals.append('high_missing_labels')

        total_images = _to_int(payload.get('total_images') or payload.get('available_images'))
        if total_images is not None:
            facts.append(f'样本量={total_images}')
            if total_images < 200:
                signals.append('small_dataset')

        issue_count = _to_int(payload.get('issue_count'))
        if payload.get('has_issues') or (issue_count is not None and issue_count > 0):
            signals.append('validation_issues_present')
            if issue_count is not None:
                facts.append(f'validate_issue_count={issue_count}')

        duplicate_groups = _to_int(payload.get('duplicate_groups'))
        if duplicate_groups is not None and duplicate_groups > 0:
            signals.append('duplicate_images_present')
            facts.append(f'重复组={duplicate_groups}')

        integrity = payload.get('integrity') or {}
        corrupted = _to_int(integrity.get('corrupted_count')) or 0
        format_mismatch = _to_int(integrity.get('format_mismatch_count')) or 0
        if corrupted > 0 or format_mismatch > 0:
            signals.append('corrupted_images_present')
            facts.append(f'损坏/格式异常={corrupted + format_mismatch}')

        blockers = payload.get('blockers') or []
        if payload.get('ready') is False or blockers:
            signals.append('dataset_not_ready')
            if blockers:
                facts.append(f'阻塞项={len(blockers)}')

        return KnowledgeService._dedupe(signals), facts

    @staticmethod
    def _derive_prediction_signals(prediction_summary: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        payload = dict(prediction_summary or {})
        if not payload:
            return [], []
        signals: list[str] = []
        facts: list[str] = []
        total_detections = _to_int(payload.get('total_detections'))
        detected_images = _to_int(payload.get('detected_images'))
        detected_frames = _to_int(payload.get('detected_frames'))
        if total_detections is not None:
            facts.append(f'预测总检测框={total_detections}')
        if detected_images is not None:
            facts.append(f'有检测图片={detected_images}')
        if detected_frames is not None:
            facts.append(f'有检测帧={detected_frames}')
        if (total_detections == 0) or (detected_images == 0 and 'detected_images' in payload) or (detected_frames == 0 and 'detected_frames' in payload):
            signals.append('prediction_no_detections')
        return KnowledgeService._dedupe(signals), facts

    def retrieve_training_knowledge(
        self,
        *,
        topic: str = '',
        stage: str = '',
        model_family: str = 'yolo',
        task_type: str = 'detection',
        signals: list[str] | None = None,
        max_rules: int = 5,
    ) -> dict[str, Any]:
        matched_rules = self.match_rules(
            topic=topic,
            stage=stage,
            model_family=model_family,
            task_type=task_type,
            signals=signals or [],
            max_rules=max_rules,
        )
        playbooks = self._match_playbooks(topic=topic, stage=stage, model_family=model_family, task_type=task_type)
        matched_ids = [rule['id'] for rule in matched_rules]
        next_actions = self._dedupe([item for rule in matched_rules for item in rule.get('next_actions') or []])
        summary = (
            f"知识检索完成: 命中 {len(matched_rules)} 条规则"
            if matched_rules
            else '知识检索完成: 当前没有命中明确规则'
        )
        return {
            'ok': True,
            'summary': summary,
            'topic': topic,
            'stage': stage,
            'model_family': model_family,
            'task_type': task_type,
            'signals': signals or [],
            'matched_rule_ids': matched_ids,
            'matched_rules': matched_rules,
            'playbooks': playbooks,
            'next_actions': next_actions[:3],
        }

    def analyze_training_outcome(
        self,
        *,
        metrics: dict[str, Any] | None = None,
        data_quality: dict[str, Any] | None = None,
        prediction_summary: dict[str, Any] | None = None,
        model_family: str = 'yolo',
        task_type: str = 'detection',
    ) -> dict[str, Any]:
        metric_signals, metric_facts = self._derive_metric_signals(metrics)
        data_signals, data_facts = self._derive_data_quality_signals(data_quality)
        prediction_signals, prediction_facts = self._derive_prediction_signals(prediction_summary)
        signals = self._dedupe(metric_signals + data_signals + prediction_signals)
        facts = self._dedupe(metric_facts + data_facts + prediction_facts)

        matched_rules = self.match_rules(
            topic='training_metrics',
            stage='post_training',
            model_family=model_family,
            task_type=task_type,
            signals=signals,
            max_rules=4,
        )
        if not matched_rules:
            matched_rules = self.match_rules(
                topic='workflow',
                stage='post_training',
                model_family=model_family,
                task_type=task_type,
                signals=signals,
                max_rules=4,
            )
        top = matched_rules[0] if matched_rules else None
        next_actions = self._dedupe([item for rule in matched_rules for item in rule.get('next_actions') or []])
        playbooks = self._match_playbooks(topic='training_metrics', stage='post_training', model_family=model_family, task_type=task_type)
        if top:
            summary = f"训练结果分析: {top['interpretation']}"
        elif signals:
            summary = '训练结果分析完成: 已提取到训练信号，但暂无更具体的规则解释'
        else:
            summary = '训练结果分析完成: 当前证据不足，无法给出可靠解释'
        return {
            'ok': True,
            'summary': summary,
            'model_family': model_family,
            'task_type': task_type,
            'signals': signals,
            'facts': facts,
            'assessment': top.get('action_type', 'collect_metrics_first') if top else 'collect_metrics_first',
            'interpretation': top.get('interpretation', '当前证据不足，建议先补齐更多训练事实') if top else '当前证据不足，建议先补齐更多训练事实',
            'recommendation': top.get('recommendation', '先收集更完整的训练指标，再决定是否调参') if top else '先收集更完整的训练指标，再决定是否调参',
            'matched_rule_ids': [rule['id'] for rule in matched_rules],
            'matched_rules': matched_rules,
            'playbooks': playbooks,
            'next_actions': next_actions[:3] or ['先收集更完整的训练指标'],
        }

    def recommend_next_training_step(
        self,
        *,
        readiness: dict[str, Any] | None = None,
        health: dict[str, Any] | None = None,
        status: dict[str, Any] | None = None,
        prediction_summary: dict[str, Any] | None = None,
        model_family: str = 'yolo',
        task_type: str = 'detection',
    ) -> dict[str, Any]:
        readiness_signals, readiness_facts = self._derive_data_quality_signals(readiness)
        health_signals, health_facts = self._derive_data_quality_signals(health)
        status_signals, status_facts = self._derive_metric_signals((status or {}).get('latest_metrics') or status)
        prediction_signals, prediction_facts = self._derive_prediction_signals(prediction_summary)
        signals = self._dedupe(readiness_signals + health_signals + status_signals + prediction_signals)
        facts = self._dedupe(readiness_facts + health_facts + status_facts + prediction_facts)
        if (status or {}).get('running'):
            signals = self._dedupe(signals + ['training_running'])

        matched_rules = self.match_rules(
            topic='next_step',
            stage='next_step',
            model_family=model_family,
            task_type=task_type,
            signals=signals,
            max_rules=4,
        )
        if not matched_rules:
            matched_rules = self.match_rules(
                topic='workflow',
                stage='next_step',
                model_family=model_family,
                task_type=task_type,
                signals=signals,
                max_rules=4,
            )
        top = matched_rules[0] if matched_rules else None
        next_actions = self._dedupe([item for rule in matched_rules for item in rule.get('next_actions') or []])
        playbooks = self._match_playbooks(topic='next_step', stage='next_step', model_family=model_family, task_type=task_type)
        if top:
            summary = f"下一步建议: {top['recommendation']}"
        else:
            summary = '下一步建议生成完成: 当前更适合先补充事实，再决定后续动作'
        return {
            'ok': True,
            'summary': summary,
            'model_family': model_family,
            'task_type': task_type,
            'signals': signals,
            'basis': facts,
            'recommended_action': top.get('action_type', 'collect_metrics_first') if top else 'collect_metrics_first',
            'recommendation': top.get('recommendation', '先补齐更多事实，再决定后续动作') if top else '先补齐更多事实，再决定后续动作',
            'why': top.get('interpretation', '当前已有事实仍不足以支持更激进的建议') if top else '当前已有事实仍不足以支持更激进的建议',
            'matched_rule_ids': [rule['id'] for rule in matched_rules],
            'matched_rules': matched_rules,
            'playbooks': playbooks,
            'next_actions': next_actions[:3] or ['先补齐更多训练事实'],
        }
