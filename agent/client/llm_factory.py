from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


@dataclass(slots=True)
class LlmProviderSettings:
    provider: str = os.getenv('YOLOSTUDIO_LLM_PROVIDER', 'ollama').strip().lower()
    model: str = os.getenv('YOLOSTUDIO_AGENT_MODEL', 'gemma4:e4b')
    base_url: str = os.getenv('YOLOSTUDIO_LLM_BASE_URL', '')
    api_key: str = os.getenv('YOLOSTUDIO_LLM_API_KEY', '')
    temperature: float = float(os.getenv('YOLOSTUDIO_LLM_TEMPERATURE', '0'))


def _resolve_provider_defaults(settings: LlmProviderSettings) -> tuple[str, str]:
    provider = settings.provider
    base_url = settings.base_url.strip()
    api_key = settings.api_key.strip()

    if provider == 'deepseek':
        if not base_url:
            base_url = 'https://api.deepseek.com'
        if not api_key:
            api_key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    elif provider == 'ollama':
        if not base_url:
            base_url = os.getenv('YOLOSTUDIO_OLLAMA_URL', 'http://127.0.0.1:11434').strip()
    elif provider in {'openai_compatible', 'openai-compatible', 'openai'}:
        provider = 'openai_compatible'
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY', '').strip()
    else:
        raise ValueError(f'不支持的 LLM provider: {provider}')

    return provider, base_url, api_key


def build_llm(settings: LlmProviderSettings) -> Any:
    provider, base_url, api_key = _resolve_provider_defaults(settings)

    if provider == 'ollama':
        return ChatOllama(
            model=settings.model,
            base_url=base_url,
            temperature=settings.temperature,
        )

    if provider in {'deepseek', 'openai_compatible'}:
        if not api_key:
            raise ValueError(f'{provider} 缺少 API key')
        if not base_url:
            raise ValueError(f'{provider} 缺少 base_url')
        return ChatOpenAI(
            model=settings.model,
            base_url=base_url,
            api_key=api_key,
            temperature=settings.temperature,
        )

    raise ValueError(f'不支持的 LLM provider: {provider}')


def provider_summary(settings: LlmProviderSettings) -> str:
    provider, base_url, api_key = _resolve_provider_defaults(settings)
    key_state = '已配置' if api_key else '未配置'
    return f'provider={provider}, model={settings.model}, base_url={base_url or "<default>"}, api_key={key_state}'
