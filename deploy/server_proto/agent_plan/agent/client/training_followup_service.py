from __future__ import annotations

from typing import Callable

from yolostudio_agent.agent.client.session_state import SessionState


AssistantMessageAppender = Callable[[str], None]


def _is_training_provenance_request(user_text: str, normalized_text: str) -> bool:
    text = str(user_text or '')
    normalized = str(normalized_text or '')
    return any(token in text for token in (
        '你基于哪次训练说的', '你是基于哪次训练说的', '基于哪次训练', '根据哪次训练', '依据哪次训练',
        '你上次不是说', '你不是说过',
    )) and any(
        token in text or token in normalized
        for token in ('训练', 'run', '最好', '最值得参考', '分析', '结论')
    )


def _is_training_evidence_request(user_text: str) -> bool:
    text = str(user_text or '')
    return any(token in text for token in (
        '依据是什么', '根据什么说的', '为什么这么说', '为什么说数据有问题',
    ))


def resolve_training_grounded_reply_kind(
    *,
    user_text: str,
    normalized_text: str,
    wants_predict: bool,
    training_command_like: bool,
) -> str:
    if wants_predict or training_command_like:
        return ''
    if _is_training_provenance_request(user_text, normalized_text):
        return 'provenance'
    if _is_training_evidence_request(user_text):
        return 'evidence'
    return ''


def run_training_grounded_reply_flow(
    session_state: SessionState,
    *,
    user_text: str,
    normalized_text: str,
    wants_predict: bool,
    training_command_like: bool,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, object] | None:
    grounded_reply_kind = resolve_training_grounded_reply_kind(
        user_text=user_text,
        normalized_text=normalized_text,
        wants_predict=wants_predict,
        training_command_like=training_command_like,
    )
    if grounded_reply_kind == 'provenance':
        return complete_training_provenance_reply(
            session_state,
            append_ai_message=append_ai_message,
        )
    if grounded_reply_kind == 'evidence':
        return complete_training_evidence_reply(
            session_state,
            append_ai_message=append_ai_message,
        )
    return None


def complete_training_provenance_reply(
    session_state: SessionState,
    *,
    append_ai_message: AssistantMessageAppender,
) -> dict[str, object]:
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
) -> dict[str, object]:
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
