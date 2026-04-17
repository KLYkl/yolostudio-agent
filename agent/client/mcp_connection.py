from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import timedelta
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


DEFAULT_MCP_HTTP_TIMEOUT_SECONDS = 300
DEFAULT_MCP_SSE_READ_TIMEOUT_SECONDS = 900
DEFAULT_MCP_GET_TOOLS_TIMEOUT_SECONDS = 60
DEFAULT_MCP_HEALTH_PROBE_TIMEOUT_SECONDS = 15
DEFAULT_MCP_MAX_RECOVERY_ATTEMPTS = 2


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


def _seconds_from_env(name: str, default_seconds: int) -> float:
    raw = str(os.getenv(name, '') or '').strip()
    if not raw:
        return float(default_seconds)
    try:
        seconds = float(raw)
    except Exception:
        return float(default_seconds)
    if seconds <= 0:
        return float(default_seconds)
    return seconds


async def probe_mcp_endpoint(url: str) -> dict[str, Any]:
    timeout_seconds = _seconds_from_env(
        'YOLOSTUDIO_MCP_HEALTH_PROBE_TIMEOUT_SECONDS',
        DEFAULT_MCP_HEALTH_PROBE_TIMEOUT_SECONDS,
    )
    headers = {'accept': 'text/event-stream, application/json'}
    client_kwargs: dict[str, Any] = {'timeout': timeout_seconds}
    if is_local_mcp_url(url):
        client_kwargs['trust_env'] = False
    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url, headers=headers)
    except Exception as exc:
        return {'ok': False, 'url': url, 'error': str(exc)}
    return {
        'ok': response.status_code < 500,
        'url': url,
        'status_code': response.status_code,
        'reason_phrase': str(getattr(response, 'reason_phrase', '') or ''),
    }


async def maybe_restart_mcp_from_env() -> dict[str, Any]:
    command = str(os.getenv('YOLOSTUDIO_MCP_RESTART_COMMAND', '') or '').strip()
    if not command:
        return {'attempted': False}
    timeout_seconds = _seconds_from_env('YOLOSTUDIO_MCP_RESTART_TIMEOUT_SECONDS', 60)
    try:
        process = await asyncio.create_subprocess_shell(command)
    except Exception as exc:
        return {'attempted': True, 'ok': False, 'error': str(exc), 'command': command}
    try:
        return_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except Exception as exc:
        with contextlib.suppress(Exception):
            process.kill()
        return {'attempted': True, 'ok': False, 'error': str(exc), 'command': command}
    return {'attempted': True, 'ok': return_code == 0, 'return_code': return_code, 'command': command}


async def load_mcp_tools_with_recovery(
    url: str,
    *,
    client_factory: Callable[[dict[str, Any]], Any],
    probe_fn: Callable[[str], Any] | None = None,
    restart_fn: Callable[[], Any] | None = None,
    max_attempts: int | None = None,
    get_tools_timeout_seconds: float | None = None,
) -> list[Any]:
    attempts = max_attempts or int(
        _seconds_from_env(
            'YOLOSTUDIO_MCP_MAX_RECOVERY_ATTEMPTS',
            DEFAULT_MCP_MAX_RECOVERY_ATTEMPTS,
        )
    )
    attempts = max(1, int(attempts))
    timeout_seconds = get_tools_timeout_seconds or _seconds_from_env(
        'YOLOSTUDIO_MCP_GET_TOOLS_TIMEOUT_SECONDS',
        DEFAULT_MCP_GET_TOOLS_TIMEOUT_SECONDS,
    )
    probe = probe_fn or probe_mcp_endpoint
    restart = restart_fn or maybe_restart_mcp_from_env
    failures: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        probe_result: dict[str, Any] = {}
        try:
            probe_result = await probe(url)
        except Exception as exc:
            probe_result = {'ok': False, 'url': url, 'error': str(exc)}
        client = client_factory({'yolostudio': build_mcp_connection_config(url)})
        try:
            tools = await asyncio.wait_for(client.get_tools(), timeout=timeout_seconds)
            return list(tools)
        except Exception as exc:
            failure = {
                'attempt': attempt,
                'error': str(exc),
                'probe': probe_result,
            }
            if attempt < attempts:
                restart_result = await restart()
                if restart_result:
                    failure['restart'] = restart_result
            failures.append(failure)
    last_failure = failures[-1] if failures else {}
    raise RuntimeError(
        'MCP tools unavailable after recovery attempts: '
        f'url={url}; attempts={attempts}; last_error={last_failure.get("error", "unknown")}; '
        f'probe={last_failure.get("probe", {})}; restart={last_failure.get("restart", {})}'
    )
