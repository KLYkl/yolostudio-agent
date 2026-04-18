from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yolostudio_agent.agent.client.session_state import (
    SESSION_STATE_SCHEMA_VERSION,
    SessionState,
    migrate_session_state_payload,
    utc_now,
)


class MemoryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.sessions_dir = self.root / 'sessions'
        self.events_dir = self.root / 'events'
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self, session_id: str) -> SessionState:
        path = self.sessions_dir / f'{session_id}.json'
        if not path.exists():
            return SessionState(session_id=session_id)
        raw_payload = json.loads(path.read_text(encoding='utf-8'))
        try:
            from_version = int(raw_payload.get('schema_version'))
        except (TypeError, ValueError):
            from_version = 1
        migrated_payload = migrate_session_state_payload(raw_payload, session_id_fallback=session_id)
        if raw_payload != migrated_payload:
            path.write_text(json.dumps(migrated_payload, ensure_ascii=False, indent=2), encoding='utf-8')
            self.append_event(
                session_id,
                'state_schema_migrated',
                {
                    'from_version': from_version,
                    'to_version': int(migrated_payload.get('schema_version') or SESSION_STATE_SCHEMA_VERSION),
                },
            )
        return SessionState.from_dict(migrated_payload, session_id_fallback=session_id)

    def save_state(self, state: SessionState) -> None:
        state.touch()
        state.schema_version = SESSION_STATE_SCHEMA_VERSION
        path = self.sessions_dir / f'{state.session_id}.json'
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        path = self.events_dir / f'{session_id}.jsonl'
        record = {'ts': utc_now(), 'type': event_type, **payload}
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def read_events(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        path = self.events_dir / f'{session_id}.jsonl'
        if not path.exists():
            return []
        with path.open('r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if limit is None or limit >= len(rows):
            return rows
        return rows[-limit:]

    def read_events_by_type(self, session_id: str, event_type: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = [row for row in self.read_events(session_id) if row.get('type') == event_type]
        if limit is None or limit >= len(rows):
            return rows
        return rows[-limit:]

    def latest_event(self, session_id: str, event_type: str, *, tool_name: str | None = None) -> dict[str, Any] | None:
        for event in reversed(self.read_events(session_id)):
            if event.get('type') != event_type:
                continue
            if tool_name is not None and event.get('tool') != tool_name:
                continue
            return event
        return None
