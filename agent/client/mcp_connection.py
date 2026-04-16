from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlparse

import httpx


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


def build_mcp_connection_config(url: str) -> dict[str, Any]:
    connection: dict[str, Any] = {
        'transport': 'streamable-http',
        'url': url,
    }
    if is_local_mcp_url(url):
        connection['httpx_client_factory'] = build_local_mcp_httpx_client_factory()
    return connection
