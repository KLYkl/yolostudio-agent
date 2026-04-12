from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.client.llm_factory import LlmProviderSettings, build_llm, provider_summary


def main() -> None:
    ollama = LlmProviderSettings(provider='ollama', model='gemma4:e4b', base_url='http://127.0.0.1:11434')
    llm1 = build_llm(ollama)
    assert llm1 is not None
    assert 'provider=ollama' in provider_summary(ollama)

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
