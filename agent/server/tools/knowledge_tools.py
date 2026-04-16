from __future__ import annotations

from typing import Any, Callable

from yolostudio_agent.agent.server.services.knowledge_service import KnowledgeService

service = KnowledgeService()


def _knowledge_action_candidate(*, tool: str, reason: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {'kind': 'tool_call', 'tool': tool, 'reason': reason}
    if args:
        payload['args'] = args
    return payload


def _wrap(action: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, dict):
            return result
        return {'ok': True, 'result': result}
    except Exception as exc:
        return {
            'ok': False,
            'error': f'{action}失败: {exc}',
            'error_type': exc.__class__.__name__,
            'summary': f'{action}失败',
            'next_actions': ['请检查知识库规则文件和输入参数后重试'],
        }


def retrieve_training_knowledge(
    topic: str = '',
    stage: str = '',
    model_family: str = 'yolo',
    task_type: str = 'detection',
    signals: list[str] | None = None,
    max_rules: int = 5,
    include_case_sources: bool = False,
    include_test_sources: bool = False,
) -> dict[str, Any]:
    """检索训练知识规则。默认只使用 official/workflow 规则，不把测试或真实 case 经验自动混入建议。"""
    result = _wrap(
        '检索训练知识',
        service.retrieve_training_knowledge,
        topic=topic,
        stage=stage,
        model_family=model_family,
        task_type=task_type,
        signals=signals,
        max_rules=max_rules,
        include_case_sources=include_case_sources,
        include_test_sources=include_test_sources,
    )
    if result.get('ok'):
        result.setdefault('next_actions', [])
        candidates: list[dict[str, Any]] = []
        if result.get('playbooks'):
            suggestion = f"可继续阅读 playbook: {result['playbooks'][0].get('title', '未命名文档')}"
            if suggestion not in result['next_actions']:
                result['next_actions'].append(suggestion)
        if result.get('matched_rule_ids'):
            candidates.append(_knowledge_action_candidate(
                tool='recommend_next_training_step',
                reason='当前已命中知识规则；如果已有训练事实，可进一步生成下一步建议',
            ))
        if result.get('playbooks'):
            candidates.append(_knowledge_action_candidate(
                tool='retrieve_training_knowledge',
                reason='如果需要换 topic/stage 深入检索，可继续沿知识检索链扩展',
            ))
        result.setdefault('action_candidates', candidates)
    return result


def analyze_training_outcome(
    metrics: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
    prediction_summary: dict[str, Any] | None = None,
    model_family: str = 'yolo',
    task_type: str = 'detection',
    include_case_sources: bool = False,
    include_test_sources: bool = False,
) -> dict[str, Any]:
    """结合训练指标、数据质量和预测摘要解释当前训练结果；默认仍只用 official/workflow 规则。"""
    result = _wrap(
        '分析训练结果',
        service.analyze_training_outcome,
        metrics=metrics,
        data_quality=data_quality,
        prediction_summary=prediction_summary,
        model_family=model_family,
        task_type=task_type,
        include_case_sources=include_case_sources,
        include_test_sources=include_test_sources,
    )
    if result.get('ok'):
        result.setdefault('action_candidates', [
            _knowledge_action_candidate(
                tool='recommend_next_training_step',
                reason='分析训练结果后，通常下一步是生成具体动作建议',
            ),
        ])
    return result


def recommend_next_training_step(
    readiness: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    prediction_summary: dict[str, Any] | None = None,
    model_family: str = 'yolo',
    task_type: str = 'detection',
    include_case_sources: bool = False,
    include_test_sources: bool = False,
) -> dict[str, Any]:
    """基于 readiness/health/status/prediction 给出下一步建议；默认不把测试沉淀和 case 经验自动混入。"""
    result = _wrap(
        '生成下一步训练建议',
        service.recommend_next_training_step,
        readiness=readiness,
        health=health,
        status=status,
        prediction_summary=prediction_summary,
        model_family=model_family,
        task_type=task_type,
        include_case_sources=include_case_sources,
        include_test_sources=include_test_sources,
    )
    if result.get('ok'):
        action = str(result.get('recommended_action') or '').strip()
        candidates: list[dict[str, Any]] = []
        if action:
            candidates.append({
                'kind': 'domain_action',
                'action': action,
                'reason': result.get('why') or result.get('recommendation') or '',
            })
        result.setdefault('action_candidates', candidates)
    return result
