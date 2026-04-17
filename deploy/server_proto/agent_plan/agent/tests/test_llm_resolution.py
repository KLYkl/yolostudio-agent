from __future__ import annotations

import os
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

from yolostudio_agent.agent.client.llm_factory import LlmProviderSettings, provider_summary, resolve_llm_settings


def _snapshot(keys: list[str]) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def _restore(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> None:
    keys = [
        'YOLOSTUDIO_LLM_PROVIDER',
        'YOLOSTUDIO_AGENT_MODEL',
        'YOLOSTUDIO_LLM_BASE_URL',
        'YOLOSTUDIO_LLM_API_KEY',
        'YOLOSTUDIO_OLLAMA_URL',
        'YOLOSTUDIO_OLLAMA_MODEL',
        'YOLOSTUDIO_HELPER_LLM_PROVIDER',
        'YOLOSTUDIO_HELPER_LLM_MODEL',
        'YOLOSTUDIO_HELPER_LLM_BASE_URL',
        'YOLOSTUDIO_HELPER_LLM_API_KEY',
        'YOLOSTUDIO_OLLAMA_NUM_CTX',
        'YOLOSTUDIO_HELPER_OLLAMA_NUM_CTX',
        'YOLOSTUDIO_DEEPSEEK_MODEL',
        'YOLOSTUDIO_DEEPSEEK_BASE_URL',
        'DEEPSEEK_API_KEY',
        'YOLOSTUDIO_ALLOW_GENERIC_LLM_ENV',
    ]
    snap = _snapshot(keys)
    try:
        for key in keys:
            os.environ.pop(key, None)

        os.environ['YOLOSTUDIO_LLM_PROVIDER'] = 'ollama'
        os.environ['YOLOSTUDIO_AGENT_MODEL'] = 'deepseek-chat'
        os.environ['YOLOSTUDIO_LLM_BASE_URL'] = 'https://api.deepseek.com/v1'
        os.environ['DEEPSEEK_API_KEY'] = 'secret'
        os.environ['YOLOSTUDIO_OLLAMA_URL'] = 'http://127.0.0.1:11434'
        os.environ['YOLOSTUDIO_OLLAMA_NUM_CTX'] = '65536'

        primary = resolve_llm_settings(LlmProviderSettings(role='primary'), role='primary')
        assert primary.provider == 'ollama', primary
        assert primary.model == 'gemma4:e4b', primary
        assert primary.base_url == 'http://127.0.0.1:11434', primary
        assert primary.api_key == '', primary
        assert primary.ollama_num_ctx == 65536, primary
        assert 'provider=ollama' in provider_summary(primary)

        helper = resolve_llm_settings(LlmProviderSettings(role='helper'), role='helper', inherit=primary)
        assert helper.provider == primary.provider, helper
        assert helper.model == primary.model, helper
        assert helper.base_url == primary.base_url, helper
        assert helper.ollama_num_ctx == 65536, helper

        os.environ['YOLOSTUDIO_HELPER_OLLAMA_NUM_CTX'] = '32768'
        helper_num_ctx = resolve_llm_settings(LlmProviderSettings(role='helper'), role='helper', inherit=primary)
        assert helper_num_ctx.ollama_num_ctx == 32768, helper_num_ctx

        os.environ['YOLOSTUDIO_HELPER_LLM_PROVIDER'] = 'deepseek'
        deepseek_helper = resolve_llm_settings(LlmProviderSettings(role='helper'), role='helper', inherit=primary)
        assert deepseek_helper.provider == 'deepseek', deepseek_helper
        assert deepseek_helper.model == 'deepseek-chat', deepseek_helper
        assert deepseek_helper.base_url == 'https://api.deepseek.com/v1', deepseek_helper
        assert deepseek_helper.api_key == 'secret', deepseek_helper
        assert deepseek_helper.ollama_num_ctx == 32768, deepseek_helper

        print('llm resolution ok')
    finally:
        _restore(snap)


if __name__ == '__main__':
    main()
