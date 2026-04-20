from __future__ import annotations

from typing import Any
import types


def _thread_id_from_config(config: Any) -> str:
    if isinstance(config, dict):
        configurable = config.get('configurable')
        if isinstance(configurable, dict):
            return str(configurable.get('thread_id') or '').strip()
    return ''


def _pending_state(values: dict[str, Any], *, base: Any = None) -> types.SimpleNamespace:
    pending = values.get('pending_confirmation')
    next_nodes = tuple(getattr(base, 'next', ()) or ())
    if isinstance(pending, dict) and pending and not next_nodes:
        next_nodes = ('tools',)
    return types.SimpleNamespace(
        values=values,
        next=next_nodes,
        interrupts=tuple(getattr(base, 'interrupts', ()) or ()),
        tasks=tuple(getattr(base, 'tasks', ()) or ()),
    )


def _ensure_graph_pending_state(client: Any) -> Any:
    graph = getattr(client, 'graph', None)
    if graph is None:
        raise AssertionError('client.graph is required')
    store = getattr(graph, '_codex_pending_state_store', None)
    if isinstance(store, dict):
        return graph

    store = {}
    original_get_state = getattr(graph, 'get_state', None)
    original_update_state = getattr(graph, 'update_state', None)
    original_ainvoke = getattr(graph, 'ainvoke', None)
    graph._codex_pending_state_store = store
    graph._codex_original_get_state = original_get_state
    graph._codex_original_update_state = original_update_state
    graph._codex_original_ainvoke = original_ainvoke

    def _base_state(config: Any) -> Any:
        if callable(original_get_state):
            return original_get_state(config)
        return None

    def _merged_state(config: Any) -> Any:
        thread_id = _thread_id_from_config(config)
        injected = store.get(thread_id)
        if injected is not None:
            return injected
        return _base_state(config)

    def _update_state(config: Any, update: dict[str, Any]) -> None:
        thread_id = _thread_id_from_config(config)
        base = _merged_state(config)
        values = dict(getattr(base, 'values', {}) or {})
        for key, value in dict(update or {}).items():
            if key == 'pending_confirmation':
                if isinstance(value, dict):
                    values[key] = dict(value)
                elif value is None:
                    values.pop(key, None)
                else:
                    values[key] = value
                continue
            if value is None:
                values.pop(key, None)
                continue
            values[key] = value
        store[thread_id] = _pending_state(values, base=base)
        if callable(original_update_state):
            original_update_state(config, update)

    graph.get_state = _merged_state  # type: ignore[assignment]
    graph.update_state = _update_state  # type: ignore[assignment]

    if callable(original_ainvoke):
        async def _ainvoke(*args: Any, **kwargs: Any) -> Any:
            result = await original_ainvoke(*args, **kwargs)
            config = kwargs.get('config')
            thread_id = _thread_id_from_config(config)
            if thread_id and thread_id in store:
                base = _base_state(config)
                values = dict(getattr(base, 'values', {}) or {})
                if values:
                    store[thread_id] = _pending_state(values, base=base)
                else:
                    store.pop(thread_id, None)
            return result

        graph.ainvoke = _ainvoke  # type: ignore[assignment]
    return graph


def _normalize_pending(client: Any, thread_id: str, pending: dict[str, Any]) -> dict[str, Any]:
    merged_pending = dict(pending or {})
    payload = client._build_pending_action_payload(merged_pending, thread_id=thread_id)
    source = str(
        merged_pending.get('source')
        or ('synthetic' if merged_pending.get('synthetic') else 'graph')
    ).strip().lower()
    if source not in {'graph', 'synthetic'}:
        source = 'synthetic'
    return {
        'id': merged_pending.get('id'),
        'tool_call_id': str(merged_pending.get('tool_call_id') or merged_pending.get('id') or '').strip(),
        'name': payload['tool_name'],
        'tool_name': payload['tool_name'],
        'args': dict(payload['tool_args']),
        'tool_args': dict(payload['tool_args']),
        'summary': payload['summary'],
        'objective': payload['objective'],
        'allowed_decisions': list(payload['allowed_decisions']),
        'review_config': dict(payload['review_config']),
        'decision_context': dict(payload.get('decision_context') or {}),
        'thread_id': thread_id,
        'source': source,
        'interrupt_kind': payload['interrupt_kind'],
    }


def seed_pending_confirmation(client: Any, thread_id: str, pending: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_pending(client, thread_id, pending)
    config = client._pending_config(thread_id)
    graph = getattr(client, 'graph', None)
    visible_state = None
    if graph is not None and callable(getattr(graph, 'get_state', None)):
        visible_state = graph.get_state(config)
    if (
        normalized.get('source') == 'graph'
        and (
            not callable(getattr(graph, 'ainvoke', None))
            or (
                visible_state is not None
                and (
                    getattr(visible_state, 'next', None)
                    or getattr(visible_state, 'interrupts', None)
                    or getattr(visible_state, 'tasks', None)
                )
            )
        )
    ):
        client._remember_pending_confirmation(normalized, emit_event=False, persist_graph=False)
        client._sync_training_workflow_state(reason='pending_seeded_for_test')
        return normalized
    graph = _ensure_graph_pending_state(client)
    graph.update_state(
        config,
        {'pending_confirmation': normalized},
    )
    client._resolve_pending_confirmation(thread_id=thread_id, config=config)
    client._sync_training_workflow_state(reason='pending_seeded_for_test')
    return normalized


def update_pending_confirmation_args(client: Any, thread_id: str, updates: dict[str, Any]) -> bool:
    graph = _ensure_graph_pending_state(client)
    pending = client._resolve_pending_confirmation(thread_id=thread_id, config=client._pending_config(thread_id))
    if not pending:
        return False
    updated_pending = dict(pending)
    updated_args = dict(updated_pending.get('args') or updated_pending.get('tool_args') or {})
    updated_args.update(dict(updates or {}))
    updated_pending['args'] = dict(updated_args)
    updated_pending['tool_args'] = dict(updated_args)
    graph.update_state(
        client._pending_config(thread_id),
        {'pending_confirmation': updated_pending},
    )
    client._resolve_pending_confirmation(thread_id=thread_id, config=client._pending_config(thread_id))
    return True
