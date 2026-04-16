from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage


def _raw_text(content: Any) -> str:
    if isinstance(content, list):
        text_parts = [item.get('text', '') for item in content if isinstance(item, dict)]
        raw = '\n'.join(part for part in text_parts if part)
    else:
        raw = str(content)
    return raw.strip()


def parse_tool_payload(content: Any) -> dict[str, Any]:
    raw = _raw_text(content)
    if not raw:
        return {
            'ok': False,
            'error': 'empty_tool_result_payload',
            'summary': '工具没有返回可解析的结构化结果。',
            'raw': raw,
        }
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return {
            'ok': False,
            'error': 'invalid_tool_result_payload',
            'error_type': exc.__class__.__name__,
            'summary': '工具返回格式异常，未能解析为结构化 JSON 对象。',
            'raw': raw,
        }
    if isinstance(parsed, dict):
        return parsed
    return {
        'ok': False,
        'error': 'non_object_tool_result_payload',
        'summary': '工具返回了 JSON，但不是对象结构。',
        'raw': raw,
        'value': parsed,
    }


def parse_tool_message(message: ToolMessage) -> dict[str, Any]:
    return parse_tool_payload(message.content)
