from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yolostudio_agent.agent.client.memory_store import MemoryStore
from yolostudio_agent.agent.client.session_state import SessionState


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
        include_history_context: bool = True,
    ) -> MemoryDigest:
        events = self.memory_store.read_events(session_id, limit=event_window)
        recent_events = events[-recent_limit:]
        lines: list[str] = []

        if state.active_training.running:
            lines.append('当前有训练在跑。')
        if state.active_training.active_loop_status:
            lines.append(f"当前环训练状态: {state.active_training.active_loop_status}")
        if state.active_prediction.realtime_session_id:
            lines.append(f"当前实时预测会话: {state.active_prediction.realtime_session_id}")

        if not include_history_context:
            return MemoryDigest(summary_lines=lines, recent_events=recent_events)

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
