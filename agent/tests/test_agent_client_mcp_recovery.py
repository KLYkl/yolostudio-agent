from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.mcp_connection import load_mcp_tools_with_recovery


class _FlakyClient:
    attempts = 0

    def __init__(self, config):
        self.config = config

    async def get_tools(self):
        _FlakyClient.attempts += 1
        if _FlakyClient.attempts == 1:
            raise RuntimeError('temporary get_tools failure')
        return ['tool-a', 'tool-b']


class _AlwaysFailClient:
    def __init__(self, config):
        self.config = config

    async def get_tools(self):
        raise RuntimeError('still failing')


async def _probe_ok(url: str) -> dict[str, object]:
    return {'ok': True, 'url': url, 'status_code': 200}


async def _probe_fail(url: str) -> dict[str, object]:
    return {'ok': False, 'url': url, 'error': 'connection refused'}


async def _run() -> None:
    restart_calls: list[str] = []

    async def _restart() -> dict[str, object]:
        restart_calls.append('restart')
        return {'attempted': True, 'ok': True}

    _FlakyClient.attempts = 0
    tools = await load_mcp_tools_with_recovery(
        'http://127.0.0.1:8080/mcp',
        client_factory=_FlakyClient,
        probe_fn=_probe_ok,
        restart_fn=_restart,
        max_attempts=2,
        get_tools_timeout_seconds=1,
    )
    assert tools == ['tool-a', 'tool-b']
    assert restart_calls == ['restart'], restart_calls

    try:
        await load_mcp_tools_with_recovery(
            'http://127.0.0.1:8080/mcp',
            client_factory=_AlwaysFailClient,
            probe_fn=_probe_fail,
            restart_fn=_restart,
            max_attempts=2,
            get_tools_timeout_seconds=1,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert 'still failing' in message, message
        assert 'connection refused' in message, message
        assert 'attempts=2' in message, message
    else:
        raise AssertionError('expected load_mcp_tools_with_recovery to fail')

    print('agent client mcp recovery ok')


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
