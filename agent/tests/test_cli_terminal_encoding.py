from __future__ import annotations

import sys
import types
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


fake_agent_client = types.ModuleType('yolostudio_agent.agent.client.agent_client')


class _FakeAgentSettings:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


async def _fake_build_agent_client(*args, **kwargs):
    raise AssertionError('build_agent_client should not be called in CLI encoding tests')


fake_agent_client.AgentSettings = _FakeAgentSettings
fake_agent_client.build_agent_client = _fake_build_agent_client
sys.modules['yolostudio_agent.agent.client.agent_client'] = fake_agent_client

from yolostudio_agent.agent.client.cli import _decode_terminal_input  # noqa: E402


def main() -> None:
    text = '请把 /home/kly/ct_loop/data_ct 按默认比例划分成训练集和验证集'
    assert _decode_terminal_input(text.encode('utf-8')) == text
    assert _decode_terminal_input(text.encode('gb18030')) == text
    assert _decode_terminal_input(b'plain ascii input') == 'plain ascii input'
    print('cli terminal encoding ok')


if __name__ == '__main__':
    main()
