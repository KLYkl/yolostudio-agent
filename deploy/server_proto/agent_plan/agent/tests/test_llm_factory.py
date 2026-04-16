from __future__ import annotations

import os
import sys
import types
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

try:
    import langchain_openai  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_openai')

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOpenAI = _FakeChatOpenAI
    sys.modules['langchain_openai'] = fake_mod

try:
    import langchain_ollama  # type: ignore  # noqa: F401
except Exception:
    fake_mod = types.ModuleType('langchain_ollama')

    class _FakeChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_mod.ChatOllama = _FakeChatOllama
    sys.modules['langchain_ollama'] = fake_mod

from yolostudio_agent.agent.client.llm_factory import LlmProviderSettings, build_llm, provider_summary


def main() -> None:
    original_all_proxy = os.environ.get('ALL_PROXY')
    os.environ['ALL_PROXY'] = 'socks5://127.0.0.1:9999'
    try:
        ollama = LlmProviderSettings(provider='ollama', model='gemma4:e4b', base_url='http://127.0.0.1:11434')
        llm1 = build_llm(ollama)
        assert llm1 is not None
        assert os.environ.get('ALL_PROXY') == 'socks5://127.0.0.1:9999'
        assert 'provider=ollama' in provider_summary(ollama)
    finally:
        if original_all_proxy is None:
            os.environ.pop('ALL_PROXY', None)
        else:
            os.environ['ALL_PROXY'] = original_all_proxy

    deepseek = LlmProviderSettings(provider='deepseek', model='deepseek-chat', base_url='https://api.deepseek.com', api_key='test-key')
    llm2 = build_llm(deepseek)
    assert llm2 is not None
    assert 'provider=deepseek' in provider_summary(deepseek)

    compat = LlmProviderSettings(provider='openai_compatible', model='minimax-m2.7', base_url='https://ai.hsnb.fun/v1', api_key='test-key')
    llm3 = build_llm(compat)
    assert llm3 is not None
    assert 'provider=openai_compatible' in provider_summary(compat)

    print('llm factory smoke ok')


if __name__ == '__main__':
    main()
