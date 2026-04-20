from __future__ import annotations

from typing import Any
import types

from yolostudio_agent.agent.client.training_plan_context_service import (
    build_training_plan_context_from_draft,
    build_training_plan_draft_from_context,
)


def _candidate_training_plan_thread_ids(client: Any, *, thread_id: str = '') -> list[str]:
    resolved = str(thread_id or '').strip()
    candidate_ids: list[str] = []
    if resolved:
        candidate_ids.append(resolved)
    candidate_getter = getattr(client, '_training_context_candidate_thread_ids', None)
    if callable(candidate_getter):
        for candidate in candidate_getter(preferred_thread_id=resolved):
            normalized = str(candidate or '').strip()
            if normalized and normalized not in candidate_ids:
                candidate_ids.append(normalized)
    fallback = resolve_training_plan_thread_id(client, thread_id=resolved)
    if fallback and fallback not in candidate_ids:
        candidate_ids.append(fallback)
    return candidate_ids


def resolve_training_plan_thread_id(client: Any, *, thread_id: str = '') -> str:
    resolved = str(thread_id or '').strip()
    if resolved:
        return resolved
    pending_thread_id = str(getattr(client, '_pending_confirmation_thread_id', lambda: '')() or '').strip()
    if pending_thread_id:
        return pending_thread_id
    session_state = getattr(client, 'session_state', None)
    session_id = str(getattr(session_state, 'session_id', '') or 'test-session').strip() or 'test-session'
    turn_index = int(getattr(client, '_turn_index', 0) or 0)
    if turn_index <= 0:
        turn_index = 1
    return f'{session_id}-turn-{turn_index}'


def current_training_plan_context(client: Any, *, thread_id: str = '') -> dict[str, Any] | None:
    preferred_thread_id = resolve_training_plan_thread_id(client, thread_id=thread_id)
    getter = getattr(client, '_current_training_plan_context', None)
    if callable(getter):
        context = getter(preferred_thread_id=preferred_thread_id)
        if isinstance(context, dict) and context:
            return dict(context)

    graph = getattr(client, 'graph', None)
    if graph is None:
        return None

    plan_contexts = getattr(graph, 'plan_contexts', None)
    if isinstance(plan_contexts, dict):
        for candidate_thread_id in _candidate_training_plan_thread_ids(client, thread_id=preferred_thread_id):
            context = plan_contexts.get(candidate_thread_id)
            if isinstance(context, dict) and context:
                return dict(context)

    for attr_name in ('plan_context', '_training_plan_context'):
        context = getattr(graph, attr_name, None)
        if isinstance(context, dict) and context:
            return dict(context)

    state = getattr(graph, '_state', None)
    values = getattr(state, 'values', None)
    if isinstance(values, dict):
        context = values.get('training_plan_context')
        if isinstance(context, dict) and context:
            return dict(context)
    return None


def current_training_plan_context_payload(client: Any, *, thread_id: str = '') -> dict[str, Any] | None:
    context = current_training_plan_context(client, thread_id=thread_id)
    return dict(context) if isinstance(context, dict) and context else None


def current_training_plan_draft(client: Any, *, thread_id: str = '') -> dict[str, Any]:
    preferred_thread_id = resolve_training_plan_thread_id(client, thread_id=thread_id)
    getter = getattr(client, '_current_training_plan_draft_view', None)
    if callable(getter):
        draft = getter(preferred_thread_id=preferred_thread_id)
        if isinstance(draft, dict) and draft:
            return dict(draft)
    context = current_training_plan_context(client, thread_id=preferred_thread_id)
    draft = build_training_plan_draft_from_context(context)
    return dict(draft or {})


def set_training_plan_context(
    client: Any,
    context: dict[str, Any] | None,
    *,
    thread_id: str = '',
) -> None:
    preferred_thread_id = resolve_training_plan_thread_id(client, thread_id=thread_id)
    normalized = dict(context) if isinstance(context, dict) and context else None
    target_thread_ids = _candidate_training_plan_thread_ids(client, thread_id=preferred_thread_id if thread_id else '')
    if thread_id:
        target_thread_ids = [preferred_thread_id]
    elif not target_thread_ids:
        target_thread_ids = [preferred_thread_id]

    updater = getattr(client, '_update_graph_training_plan_context', None)
    clearer = getattr(client, '_clear_graph_training_plan_context_candidates', None)
    if normalized is None and not thread_id and callable(clearer):
        clearer()
    elif callable(updater):
        for target_thread_id in target_thread_ids:
            updater(thread_id=target_thread_id, context=normalized)

    graph = getattr(client, 'graph', None)
    if graph is None:
        return

    if hasattr(graph, 'plan_contexts') and isinstance(graph.plan_contexts, dict):
        if normalized:
            graph.plan_contexts[preferred_thread_id] = dict(normalized)
        else:
            for target_thread_id in target_thread_ids:
                graph.plan_contexts.pop(target_thread_id, None)

    graph.plan_context = dict(normalized) if normalized else None
    if hasattr(graph, '_training_plan_context') or normalized is not None:
        graph._training_plan_context = dict(normalized) if normalized else None

    state = getattr(graph, '_state', None)
    values = getattr(state, 'values', None)
    if isinstance(values, dict):
        if normalized:
            values['training_plan_context'] = dict(normalized)
        else:
            values.pop('training_plan_context', None)
    elif hasattr(graph, '_injected_state'):
        if normalized:
            graph._injected_state = types.SimpleNamespace(values={'training_plan_context': dict(normalized)})
        elif getattr(graph, '_injected_state', None) is not None:
            graph._injected_state = None


def set_training_plan_draft(client: Any, draft: dict[str, Any] | None, *, thread_id: str = '') -> None:
    set_training_plan_context(
        client,
        build_training_plan_context_from_draft(dict(draft or {})),
        thread_id=thread_id,
    )


def clear_training_plan_draft(client: Any, *, thread_id: str = '') -> None:
    set_training_plan_context(client, None, thread_id=thread_id)
