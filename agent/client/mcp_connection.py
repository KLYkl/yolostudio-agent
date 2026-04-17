from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


DEFAULT_MCP_HTTP_TIMEOUT_SECONDS = 300
DEFAULT_MCP_SSE_READ_TIMEOUT_SECONDS = 900


def is_local_mcp_url(url: str) -> bool:
    text = str(url or '').strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    host = str(parsed.hostname or '').strip().lower()
    return host in {'127.0.0.1', 'localhost', '::1'}


def build_local_mcp_httpx_client_factory() -> Callable[..., httpx.AsyncClient]:
    def _factory(headers: dict[str, Any] | None = None, timeout: Any = None, auth: Any = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, trust_env=False)

    return _factory


def _duration_from_env(name: str, default_seconds: int) -> timedelta:
    raw = str(os.getenv(name, '') or '').strip()
    if not raw:
        return timedelta(seconds=default_seconds)
    try:
        seconds = float(raw)
    except Exception:
        return timedelta(seconds=default_seconds)
    if seconds <= 0:
        return timedelta(seconds=default_seconds)
    return timedelta(seconds=seconds)


def build_mcp_connection_config(url: str) -> dict[str, Any]:
    connection: dict[str, Any] = {
        'transport': 'streamable-http',
        'url': url,
        'timeout': _duration_from_env(
            'YOLOSTUDIO_MCP_HTTP_TIMEOUT_SECONDS',
            DEFAULT_MCP_HTTP_TIMEOUT_SECONDS,
        ),
        'sse_read_timeout': _duration_from_env(
            'YOLOSTUDIO_MCP_SSE_READ_TIMEOUT_SECONDS',
            DEFAULT_MCP_SSE_READ_TIMEOUT_SECONDS,
        ),
    }
    if is_local_mcp_url(url):
        connection['httpx_client_factory'] = build_local_mcp_httpx_client_factory()
    return connection
