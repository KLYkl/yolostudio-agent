from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_plan.agent.client.memory_store import MemoryStore
from agent_plan.agent.client.session_state import SessionState


@dataclass(slots=True)
class MemoryDigest:
    summary_lines: list[str]
    recent_events: list[dict[str, Any]]

    def to_text(self) -> str:
        if not self.summary_lines:
            return '无历史摘要'
        return '\n'.join(f'- {line}' for line in self.summary_lines)


class EventRetriever:
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    def build_digest(
        self,
        session_id: str,
        state: SessionState,
        *,
        recent_limit: int = 12,
        event_window: int = 40,
    ) -> MemoryDigest:
        events = self.memory_store.read_events(session_id, limit=event_window)
        recent_events = events[-recent_limit:]
        lines: list[str] = []

        if state.active_dataset.last_scan:
            scan = state.active_dataset.last_scan
            lines.append(
                f"最近扫描: {scan.get('total_images', '未知')} 张图, 缺失标签 {scan.get('missing_labels', '未知')}, 空标签 {scan.get('empty_labels', '未知')}"
            )
        if state.active_dataset.last_validate:
            validate = state.active_dataset.last_validate
            lines.append(
                f"最近校验: issue_count={validate.get('issue_count', '未知')}, has_issues={validate.get('has_issues', '未知')}"
            )
        if state.active_dataset.last_readiness:
            readiness = state.active_dataset.last_readiness
            lines.append(
                f"最近 readiness: ready={readiness.get('ready', '未知')}, risk_level={readiness.get('risk_level', '未知')}, blockers={len(readiness.get('blockers') or [])}"
            )
        if state.active_training.model or state.active_training.data_yaml or state.active_training.last_status:
            status = state.active_training.last_status or {}
            lines.append(
                f"最近训练状态: running={status.get('running', state.active_training.running)}, model={state.active_training.model or '未设置'}, data={state.active_training.data_yaml or '未设置'}, device={state.active_training.device or '未设置'}"
            )
        if state.active_knowledge.last_recommendation:
            recommendation = state.active_knowledge.last_recommendation
            lines.append(
                f"最近训练建议: action={recommendation.get('recommended_action', '未知')}, summary={recommendation.get('summary', '无')}"
            )

        latest_tools: dict[str, dict[str, Any]] = {}
        confirmation_requested = 0
        confirmation_cancelled = 0
        confirmation_approved = 0

        for event in events:
            event_type = event.get('type')
            if event_type == 'tool_result' and event.get('tool'):
                latest_tools[event['tool']] = event
            elif event_type == 'confirmation_requested':
                confirmation_requested += 1
            elif event_type == 'confirmation_cancelled':
                confirmation_cancelled += 1
            elif event_type == 'confirmation_approved':
                confirmation_approved += 1

        if latest_tools:
            lines.append('最近调用过的工具: ' + ', '.join(sorted(latest_tools.keys())))
        if confirmation_requested:
            lines.append(
                f'人工确认记录: 请求 {confirmation_requested} 次, 同意 {confirmation_approved} 次, 取消 {confirmation_cancelled} 次'
            )

        recent_lines: list[str] = []
        for event in recent_events[-6:]:
            event_type = event.get('type', 'unknown')
            if event_type == 'tool_result':
                tool = event.get('tool', 'unknown_tool')
                result = event.get('result', {})
                if result.get('ok') is False:
                    recent_lines.append(f'{tool}: 失败 ({result.get("error", "unknown error")})')
                else:
                    summary = result.get('summary') or result.get('message')
                    if not summary and tool == 'check_training_status':
                        summary = f"running={result.get('running')} device={result.get('device')}"
                    if not summary and tool == 'check_gpu_status':
                        summary = result.get('gpu_summary') or result.get('raw')
                    recent_lines.append(f'{tool}: {summary or "成功"}')
            elif event_type == 'confirmation_requested':
                recent_lines.append(f"待确认: {event.get('tool', 'unknown_tool')}")
            elif event_type == 'confirmation_cancelled':
                recent_lines.append(f"已取消: {event.get('tool', 'unknown_tool')}")
            elif event_type == 'confirmation_approved':
                recent_lines.append(f"已批准: {event.get('tool', 'unknown_tool')}")
            elif event_type == 'knowledge_recommendation':
                recent_lines.append(f"知识建议: {event.get('recommended_action', 'unknown')} / {event.get('summary', '无摘要')}")
            elif event_type == 'training_analysis':
                recent_lines.append(f"训练分析: {event.get('summary', '无摘要')}")

        if recent_lines:
            lines.append('近期事件: ' + ' | '.join(recent_lines[-4:]))

        return MemoryDigest(summary_lines=lines, recent_events=recent_events)
