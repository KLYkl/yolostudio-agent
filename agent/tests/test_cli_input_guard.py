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
    raise AssertionError('build_agent_client should not be called in CLI smoke tests')


fake_agent_client.AgentSettings = _FakeAgentSettings
fake_agent_client.build_agent_client = _fake_build_agent_client
sys.modules['yolostudio_agent.agent.client.agent_client'] = fake_agent_client

from yolostudio_agent.agent.client import cli  # noqa: E402


def main() -> None:
    args = cli._parse_cli_args(['demo-session', '--confirm', 'manual', '--provider', 'ollama', '--model', 'gemma4:e4b'])
    assert args.session_id == 'demo-session'
    assert args.confirm == 'manual'
    assert args.provider == 'ollama'
    assert args.model == 'gemma4:e4b'

    assert cli._is_exit_command('exit')
    assert cli._is_exit_command(' Quit ')
    assert not cli._is_exit_command('继续')

    assert not hasattr(cli, 'handle_slash_command')
    assert not hasattr(cli, '_looks_like_known_slash_command')
    assert not hasattr(cli, '_looks_like_unknown_short_slash_command')
    assert not hasattr(cli, '_parse_confirmation_input')
    print('cli input guard ok')


if __name__ == '__main__':
    main()
