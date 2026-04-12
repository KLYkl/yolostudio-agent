from __future__ import annotations

from typing import Any, Callable

from agent_plan.agent.server.services.knowledge_service import KnowledgeService

service = KnowledgeService()


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
        if result.get('playbooks'):
            suggestion = f"可继续阅读 playbook: {result['playbooks'][0].get('title', '未命名文档')}"
            if suggestion not in result['next_actions']:
                result['next_actions'].append(suggestion)
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
    return _wrap(
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
    return _wrap(
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
