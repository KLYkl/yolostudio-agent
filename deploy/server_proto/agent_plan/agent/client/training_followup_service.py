from __future__ import annotations

from typing import Any, Awaitable, Callable

from yolostudio_agent.agent.client.session_state import SessionState


DirectToolInvoker = Callable[..., Awaitable[dict[str, Any]]]
MultiToolRenderer = Callable[..., Awaitable[str]]
AssistantMessageAppender = Callable[[str], None]


def _append_reply(
    *,
    append_ai_message: AssistantMessageAppender,
    reply: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    append_ai_message(reply)
    return {
        'status': 'completed' if all(result.get('ok', True) for result in results) else 'error',
        'message': reply,
        'tool_call': None,
    }


def _analysis_kwargs(
    session_state: SessionState,
    *,
    metrics: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        'metrics': metrics,
        'data_quality': session_state.active_dataset.last_health_check or session_state.active_dataset.last_validate,
        'comparison': comparison,
        'prediction_summary': session_state.active_prediction.last_result,
        'model_family': 'yolo',
        'task_type': 'detection',
    }


def _next_step_kwargs(
    session_state: SessionState,
    *,
    status: dict[str, Any],
    readiness: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        'readiness': readiness or session_state.active_dataset.last_readiness,
        'health': session_state.active_dataset.last_health_check,
        'status': status,
        'comparison': comparison,
        'prediction_summary': session_state.active_prediction.last_result,
        'model_family': 'yolo',
        'task_type': 'detection',
    }


def _analysis_fallback(
    result: dict[str, Any],
    first_result: dict[str, Any],
    default_message: str,
) -> str:
    return str(result.get('summary') or first_result.get('summary') or result.get('error') or default_message)


async def complete_training_outcome_analysis_reply(
    session_state: SessionState,
    *,
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    training_summary = await direct_tool('summarize_training_run')
    result = await direct_tool(
        'analyze_training_outcome',
        **_analysis_kwargs(
            session_state,
            metrics=training_summary,
            comparison=session_state.active_training.last_run_comparison,
        ),
    )
    reply = await render_multi_tool_result_message(
        [
            ('summarize_training_run', training_summary),
            ('analyze_training_outcome', result),
        ],
        objective='训练结果分析说明',
    )
    if not reply:
        reply = _analysis_fallback(result, training_summary, '训练结果分析已完成')
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=[training_summary, result],
    )


async def complete_specific_training_run_outcome_analysis_reply(
    session_state: SessionState,
    *,
    run_id: str,
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    inspection = await direct_tool('inspect_training_run', run_id=run_id)
    result = await direct_tool(
        'analyze_training_outcome',
        **_analysis_kwargs(
            session_state,
            metrics=inspection,
            comparison=session_state.active_training.last_run_comparison,
        ),
    )
    reply = await render_multi_tool_result_message(
        [
            ('inspect_training_run', inspection),
            ('analyze_training_outcome', result),
        ],
        objective='指定训练结果分析说明',
    )
    if not reply:
        reply = _analysis_fallback(result, inspection, '指定训练结果分析已完成')
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=[inspection, result],
    )


async def complete_best_training_outcome_analysis_reply(
    session_state: SessionState,
    *,
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    selection = await direct_tool('select_best_training_run')
    best_run = selection.get('best_run') if selection.get('ok') else None
    result = await direct_tool(
        'analyze_training_outcome',
        **_analysis_kwargs(
            session_state,
            metrics=best_run
            or session_state.active_training.training_run_summary
            or session_state.active_training.last_summary
            or session_state.active_training.last_status,
            comparison=None,
        ),
    )
    reply = await render_multi_tool_result_message(
        [
            ('select_best_training_run', selection),
            ('analyze_training_outcome', result),
        ],
        objective='最佳训练结果分析说明',
    )
    if not reply:
        reply = _analysis_fallback(result, selection, '最佳训练结果分析已完成')
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=[selection, result],
    )


async def complete_training_compare_analysis_reply(
    session_state: SessionState,
    *,
    left_run_id: str = '',
    right_run_id: str = '',
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    comparison = await direct_tool('compare_training_runs', left_run_id=left_run_id, right_run_id=right_run_id)
    latest_run = comparison.get('left_run') if comparison.get('ok') else None
    result = await direct_tool(
        'analyze_training_outcome',
        **_analysis_kwargs(
            session_state,
            metrics=latest_run
            or session_state.active_training.training_run_summary
            or session_state.active_training.last_summary
            or session_state.active_training.last_status,
            comparison=comparison if comparison.get('ok') else None,
        ),
    )
    reply = await render_multi_tool_result_message(
        [
            ('compare_training_runs', comparison),
            ('analyze_training_outcome', result),
        ],
        objective='训练对比分析说明',
    )
    if not reply:
        reply = _analysis_fallback(result, comparison, '训练对比分析已完成')
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=[comparison, result],
    )


async def complete_specific_training_run_next_step_reply(
    session_state: SessionState,
    *,
    run_id: str,
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    inspection = await direct_tool('inspect_training_run', run_id=run_id)
    result = await direct_tool(
        'recommend_next_training_step',
        **_next_step_kwargs(
            session_state,
            status=inspection,
            readiness=None,
            comparison=session_state.active_training.last_run_comparison,
        ),
    )
    reply = await render_multi_tool_result_message(
        [
            ('inspect_training_run', inspection),
            ('recommend_next_training_step', result),
        ],
        objective='指定训练下一步建议说明',
    )
    if not reply:
        reply = _analysis_fallback(result, inspection, '指定训练的下一步建议已生成')
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=[inspection, result],
    )


async def complete_best_training_next_step_reply(
    session_state: SessionState,
    *,
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    selection = await direct_tool('select_best_training_run')
    best_run = selection.get('best_run') if selection.get('ok') else None
    result = await direct_tool(
        'recommend_next_training_step',
        **_next_step_kwargs(
            session_state,
            status=best_run
            or session_state.active_training.training_run_summary
            or session_state.active_training.last_summary
            or session_state.active_training.last_status,
            readiness=None,
            comparison=None,
        ),
    )
    reply = await render_multi_tool_result_message(
        [
            ('select_best_training_run', selection),
            ('recommend_next_training_step', result),
        ],
        objective='最佳训练下一步建议说明',
    )
    if not reply:
        reply = _analysis_fallback(result, selection, '最佳训练的下一步建议已生成')
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=[selection, result],
    )


async def complete_next_training_step_reply(
    session_state: SessionState,
    *,
    dataset_path: str = '',
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    readiness: dict[str, Any] | None = None
    if dataset_path:
        readiness = await direct_tool('training_readiness', img_dir=dataset_path)
    training_summary = await direct_tool('summarize_training_run')
    result = await direct_tool(
        'recommend_next_training_step',
        **_next_step_kwargs(
            session_state,
            status=training_summary,
            readiness=readiness,
            comparison=session_state.active_training.last_run_comparison,
        ),
    )
    results: list[tuple[str, dict[str, Any]]] = []
    if readiness is not None:
        results.append(('training_readiness', readiness))
    results.append(('summarize_training_run', training_summary))
    results.append(('recommend_next_training_step', result))
    reply = await render_multi_tool_result_message(results, objective='下一步训练建议说明')
    if not reply:
        reply = _analysis_fallback(result, training_summary, '下一步建议已生成')
    status_results = [training_summary, result]
    if readiness is not None:
        status_results.insert(0, readiness)
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=status_results,
    )


async def complete_training_compare_next_step_reply(
    session_state: SessionState,
    *,
    left_run_id: str = '',
    right_run_id: str = '',
    direct_tool: DirectToolInvoker,
    render_multi_tool_result_message: MultiToolRenderer,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    comparison = await direct_tool('compare_training_runs', left_run_id=left_run_id, right_run_id=right_run_id)
    latest_run = comparison.get('left_run') if comparison.get('ok') else None
    result = await direct_tool(
        'recommend_next_training_step',
        **_next_step_kwargs(
            session_state,
            status=latest_run
            or session_state.active_training.training_run_summary
            or session_state.active_training.last_summary
            or session_state.active_training.last_status,
            readiness=None,
            comparison=comparison if comparison.get('ok') else None,
        ),
    )
    reply = await render_multi_tool_result_message(
        [
            ('compare_training_runs', comparison),
            ('recommend_next_training_step', result),
        ],
        objective='训练对比后的下一步建议说明',
    )
    if not reply:
        reply = _analysis_fallback(result, comparison, '训练对比后的下一步建议已生成')
    return _append_reply(
        append_ai_message=append_ai_message,
        reply=reply,
        results=[comparison, result],
    )


def complete_training_provenance_reply(
    session_state: SessionState,
    *,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    training = session_state.active_training
    lines: list[str] = []
    comparison = training.last_run_comparison or {}
    best_selection = training.best_run_selection or {}
    inspected = training.last_run_inspection or {}
    summary = training.training_run_summary or training.last_summary or training.last_status or {}

    if comparison:
        left_run = comparison.get('left_run') or {}
        right_run = comparison.get('right_run') or {}
        left_id = str(left_run.get('run_id') or left_run.get('log_file') or '最近一次训练').strip()
        right_id = str(right_run.get('run_id') or right_run.get('log_file') or '上一次训练').strip()
        lines.append(f'我当前主要基于训练对比结果：{left_id} 对比 {right_id}。')
        if comparison.get('summary'):
            lines.append(f"- 对比摘要: {comparison.get('summary')}")
    elif best_selection:
        best_run = best_selection.get('best_run') or {}
        best_id = str(best_run.get('run_id') or best_run.get('log_file') or '最近最佳训练').strip()
        lines.append(f'我当前主要基于最值得参考的训练记录：{best_id}。')
        if best_selection.get('summary'):
            lines.append(f"- 选择依据: {best_selection.get('summary')}")
    elif inspected:
        selected_id = str(inspected.get('selected_run_id') or inspected.get('log_file') or '指定训练记录').strip()
        lines.append(f'我当前主要基于你刚查看的训练记录：{selected_id}。')
        if inspected.get('summary'):
            lines.append(f"- 记录摘要: {inspected.get('summary')}")
    elif summary:
        run_label = str(summary.get('run_id') or summary.get('log_file') or summary.get('summary') or '最近一次训练').strip()
        lines.append(f'我当前主要基于最近一次训练结果：{run_label}。')
    else:
        lines.append('我当前没有可追溯的训练依据；请先查看训练状态、训练详情、训练对比或最佳训练。')

    reply = '\n'.join(lines)
    append_ai_message(reply)
    return {'status': 'completed', 'message': reply, 'tool_call': None}


def complete_training_evidence_reply(
    session_state: SessionState,
    *,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, Any]:
    training = session_state.active_training
    knowledge = session_state.active_knowledge
    lines = ['当前判断主要基于这些事实：']

    summary = training.training_run_summary or training.last_summary or training.last_status or {}
    if summary.get('summary'):
        lines.append(f"- 训练事实: {summary.get('summary')}")
    if summary.get('signals'):
        lines.append(f"- 训练信号: {', '.join(str(item) for item in list(summary.get('signals') or [])[:4])}")

    comparison = training.last_run_comparison or {}
    if comparison.get('summary'):
        lines.append(f"- 对比依据: {comparison.get('summary')}")
    if comparison.get('signals'):
        lines.append(f"- 对比信号: {', '.join(str(item) for item in list(comparison.get('signals') or [])[:4])}")

    analysis = knowledge.last_analysis or {}
    if analysis.get('summary'):
        lines.append(f"- 分析结论: {analysis.get('summary')}")
    if analysis.get('signals'):
        lines.append(f"- 分析信号: {', '.join(str(item) for item in list(analysis.get('signals') or [])[:4])}")

    recommendation = knowledge.last_recommendation or {}
    if recommendation.get('summary'):
        lines.append(f"- 建议依据: {recommendation.get('summary')}")
    if recommendation.get('recommended_action'):
        lines.append(f"- 当前建议动作: {recommendation.get('recommended_action')}")

    if len(lines) == 1:
        lines.append('- 当前没有足够的训练分析上下文；请先查看训练结果或重新分析。')

    reply = '\n'.join(lines)
    append_ai_message(reply)
    return {'status': 'completed', 'message': reply, 'tool_call': None}
