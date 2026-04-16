from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


_ROLE_ENV = {
    'primary': {
        'provider': 'YOLOSTUDIO_LLM_PROVIDER',
        'model': 'YOLOSTUDIO_AGENT_MODEL',
        'base_url': 'YOLOSTUDIO_LLM_BASE_URL',
        'api_key': 'YOLOSTUDIO_LLM_API_KEY',
        'temperature': 'YOLOSTUDIO_LLM_TEMPERATURE',
        'ollama_keep_alive': 'YOLOSTUDIO_OLLAMA_KEEP_ALIVE',
    },
    'helper': {
        'provider': 'YOLOSTUDIO_HELPER_LLM_PROVIDER',
        'model': 'YOLOSTUDIO_HELPER_LLM_MODEL',
        'base_url': 'YOLOSTUDIO_HELPER_LLM_BASE_URL',
        'api_key': 'YOLOSTUDIO_HELPER_LLM_API_KEY',
        'temperature': 'YOLOSTUDIO_HELPER_LLM_TEMPERATURE',
        'ollama_keep_alive': 'YOLOSTUDIO_HELPER_OLLAMA_KEEP_ALIVE',
    },
    'loop': {
        'provider': 'YOLOSTUDIO_LOOP_LLM_PROVIDER',
        'model': 'YOLOSTUDIO_LOOP_LLM_MODEL',
        'base_url': 'YOLOSTUDIO_LOOP_LLM_BASE_URL',
        'api_key': 'YOLOSTUDIO_LOOP_LLM_API_KEY',
        'temperature': 'YOLOSTUDIO_LOOP_LLM_TEMPERATURE',
        'ollama_keep_alive': 'YOLOSTUDIO_LOOP_LLM_KEEP_ALIVE',
    },
}

_PROVIDER_ENV = {
    'ollama': {
        'model': 'YOLOSTUDIO_OLLAMA_MODEL',
        'base_url': 'YOLOSTUDIO_OLLAMA_URL',
        'api_key': '',
        'default_model': 'gemma4:e4b',
        'default_base_url': 'http://127.0.0.1:11434',
        'default_api_key': '',
        'default_keep_alive': '',
    },
    'deepseek': {
        'model': 'YOLOSTUDIO_DEEPSEEK_MODEL',
        'base_url': 'YOLOSTUDIO_DEEPSEEK_BASE_URL',
        'api_key': 'DEEPSEEK_API_KEY',
        'default_model': 'deepseek-chat',
        'default_base_url': 'https://api.deepseek.com/v1',
        'default_api_key': '',
        'default_keep_alive': '',
    },
    'openai_compatible': {
        'model': 'YOLOSTUDIO_OPENAI_MODEL',
        'base_url': 'YOLOSTUDIO_OPENAI_BASE_URL',
        'api_key': 'OPENAI_API_KEY',
        'default_model': '',
        'default_base_url': '',
        'default_api_key': '',
        'default_keep_alive': '',
    },
}

_PROXY_ENV_NAMES = (
    'ALL_PROXY',
    'all_proxy',
    'HTTP_PROXY',
    'http_proxy',
    'HTTPS_PROXY',
    'https_proxy',
)


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or '').strip()
        if text:
            return text
    return ''


def _normalize_provider(value: Any) -> str:
    provider = str(value or '').strip().lower()
    if provider in {'', 'ollama'}:
        return 'ollama'
    if provider in {'deepseek'}:
        return 'deepseek'
    if provider in {'openai_compatible', 'openai-compatible', 'openai'}:
        return 'openai_compatible'
    raise ValueError(f'不支持的 LLM provider: {provider}')


def _role_env(role: str) -> dict[str, str]:
    return _ROLE_ENV.get(role, _ROLE_ENV['primary'])


def _is_local_ollama_base_url(base_url: str) -> bool:
    text = str(base_url or '').strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    host = str(parsed.hostname or '').strip().lower()
    return host in {'127.0.0.1', 'localhost', '::1'}


@contextmanager
def _temporary_env_unset(names: tuple[str, ...]) -> Any:
    snapshot = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in snapshot.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(slots=True)
class LlmProviderSettings:
    provider: str = ''
    model: str = ''
    base_url: str = ''
    api_key: str = ''
    temperature: float | None = None
    ollama_keep_alive: str = ''
    role: str = 'primary'


def resolve_llm_settings(
    settings: LlmProviderSettings | None = None,
    *,
    role: str | None = None,
    inherit: LlmProviderSettings | None = None,
) -> LlmProviderSettings:
    raw = settings or LlmProviderSettings()
    resolved_role = str(role or raw.role or 'primary').strip().lower() or 'primary'
    env_names = _role_env(resolved_role)

    explicit_provider = str(raw.provider or '').strip()
    env_provider = os.getenv(env_names['provider'], '').strip()
    inherited_provider = str(inherit.provider or '').strip() if inherit is not None else ''
    provider = _normalize_provider(_first_nonempty(explicit_provider, env_provider, inherited_provider, 'ollama'))
    provider_env = _PROVIDER_ENV[provider]

    role_specific_provider_explicit = resolved_role != 'primary' and bool(explicit_provider or env_provider)

    generic_model = os.getenv(_ROLE_ENV['primary']['model'], '').strip()
    generic_base_url = os.getenv(_ROLE_ENV['primary']['base_url'], '').strip()
    generic_api_key = os.getenv(_ROLE_ENV['primary']['api_key'], '').strip()
    generic_temperature = os.getenv(_ROLE_ENV['primary']['temperature'], '').strip()
    generic_keep_alive = os.getenv(_ROLE_ENV['primary']['ollama_keep_alive'], '').strip()

    if role_specific_provider_explicit:
        inherited_model = ''
        inherited_base_url = ''
        inherited_api_key = ''
        inherited_temperature = ''
        inherited_keep_alive = ''
    else:
        inherited_model = str(inherit.model or '').strip() if inherit is not None else ''
        inherited_base_url = str(inherit.base_url or '').strip() if inherit is not None else ''
        inherited_api_key = str(inherit.api_key or '').strip() if inherit is not None else ''
        inherited_temperature = (
            '' if inherit is None or inherit.temperature is None else str(inherit.temperature).strip()
        )
        inherited_keep_alive = str(inherit.ollama_keep_alive or '').strip() if inherit is not None else ''

    role_model = os.getenv(env_names['model'], '').strip() if resolved_role != 'primary' else ''
    role_base_url = os.getenv(env_names['base_url'], '').strip() if resolved_role != 'primary' else ''
    role_api_key = os.getenv(env_names['api_key'], '').strip() if resolved_role != 'primary' else ''
    role_temperature = os.getenv(env_names['temperature'], '').strip()
    role_keep_alive = os.getenv(env_names['ollama_keep_alive'], '').strip()

    provider_model = os.getenv(provider_env['model'], '').strip() if provider_env['model'] else ''
    provider_base_url = os.getenv(provider_env['base_url'], '').strip() if provider_env['base_url'] else ''
    provider_api_key = os.getenv(provider_env['api_key'], '').strip() if provider_env['api_key'] else ''

    use_generic_fallback = str(os.getenv('YOLOSTUDIO_ALLOW_GENERIC_LLM_ENV', '') or '').strip().lower() in {
        '1', 'true', 'yes', 'on'
    }

    model = _first_nonempty(
        raw.model,
        role_model,
        provider_model,
        inherited_model if resolved_role != 'primary' else '',
        generic_model if use_generic_fallback else '',
        provider_env['default_model'],
    )
    base_url = _first_nonempty(
        raw.base_url,
        role_base_url,
        provider_base_url,
        inherited_base_url if resolved_role != 'primary' else '',
        generic_base_url if use_generic_fallback else '',
        provider_env['default_base_url'],
    )
    api_key = _first_nonempty(
        raw.api_key,
        role_api_key,
        provider_api_key,
        inherited_api_key if resolved_role != 'primary' else '',
        generic_api_key if use_generic_fallback else '',
        provider_env['default_api_key'],
    )

    if raw.temperature is not None:
        temperature = float(raw.temperature)
    else:
        temperature_raw = _first_nonempty(
            role_temperature,
            inherited_temperature if resolved_role != 'primary' else '',
            generic_temperature if use_generic_fallback else '',
            '0',
        )
        try:
            temperature = float(temperature_raw)
        except Exception:
            temperature = 0.0

    ollama_keep_alive = _first_nonempty(
        raw.ollama_keep_alive,
        role_keep_alive,
        inherited_keep_alive if resolved_role != 'primary' else '',
        generic_keep_alive if (use_generic_fallback or provider == 'ollama') else '',
        provider_env['default_keep_alive'],
    )

    if provider == 'ollama':
        api_key = ''

    return LlmProviderSettings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        ollama_keep_alive=ollama_keep_alive,
        role=resolved_role,
    )


def build_llm(
    settings: LlmProviderSettings,
    *,
    role: str | None = None,
    inherit: LlmProviderSettings | None = None,
) -> Any:
    resolved = resolve_llm_settings(settings, role=role, inherit=inherit)

    if resolved.provider == 'ollama':
        if _is_local_ollama_base_url(resolved.base_url):
            with _temporary_env_unset(_PROXY_ENV_NAMES):
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=resolved.model,
                    base_url=resolved.base_url,
                    temperature=resolved.temperature,
                    keep_alive=resolved.ollama_keep_alive or None,
                )
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=resolved.model,
            base_url=resolved.base_url,
            temperature=resolved.temperature,
            keep_alive=resolved.ollama_keep_alive or None,
        )

    if resolved.provider in {'deepseek', 'openai_compatible'}:
        if not resolved.api_key:
            raise ValueError(f'{resolved.provider} 缺少 API key')
        if not resolved.base_url:
            raise ValueError(f'{resolved.provider} 缺少 base_url')
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=resolved.model,
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            temperature=resolved.temperature,
        )

    raise ValueError(f'不支持的 LLM provider: {resolved.provider}')


def provider_summary(
    settings: LlmProviderSettings,
    *,
    role: str | None = None,
    inherit: LlmProviderSettings | None = None,
) -> str:
    resolved = resolve_llm_settings(settings, role=role, inherit=inherit)
    key_state = '已配置' if resolved.api_key else '未配置'
    return (
        f'provider={resolved.provider}, model={resolved.model}, '
        f'base_url={resolved.base_url or "<default>"}, api_key={key_state}'
    )
