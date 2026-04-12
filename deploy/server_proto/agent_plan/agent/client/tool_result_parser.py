from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage


def parse_tool_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        text_parts = [item.get('text', '') for item in content if isinstance(item, dict)]
        raw = '\n'.join(part for part in text_parts if part)
    else:
        raw = str(content)
    raw = raw.strip()
    try:
        parsed = json.loads(raw) if raw else {'ok': True, 'raw': raw}
    except Exception:
        parsed = {'ok': True, 'raw': raw}
    return parsed if isinstance(parsed, dict) else {'ok': True, 'value': parsed}


def parse_tool_message(message: ToolMessage) -> dict[str, Any]:
    return parse_tool_payload(message.content)
