from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root,):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.mcp_connection import build_mcp_connection_config


def main() -> None:
    local = build_mcp_connection_config('http://127.0.0.1:8080/mcp')
    assert local['transport'] == 'streamable-http'
    assert local['url'] == 'http://127.0.0.1:8080/mcp'
    assert local['timeout'] == timedelta(seconds=300), local
    assert local['sse_read_timeout'] == timedelta(seconds=900), local
    factory = local.get('httpx_client_factory')
    assert callable(factory), local
    client = factory()
    try:
        assert getattr(client, '_trust_env', None) is False
    finally:
        asyncio.run(client.aclose())

    remote = build_mcp_connection_config('https://example.com/mcp')
    assert remote['transport'] == 'streamable-http'
    assert remote['url'] == 'https://example.com/mcp'
    assert remote['timeout'] == timedelta(seconds=300), remote
    assert remote['sse_read_timeout'] == timedelta(seconds=900), remote
    assert 'httpx_client_factory' not in remote
    print('agent client mcp config ok')


if __name__ == '__main__':
    main()
