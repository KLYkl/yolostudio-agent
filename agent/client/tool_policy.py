from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from yolostudio_agent.agent.client.tool_adapter import canonical_tool_name


HIGH_RISK_TOOLS = {
    "start_training",
    "start_training_loop",
    "split_dataset",
    "augment_dataset",
    "prepare_dataset_for_training",
    "convert_format",
    "modify_labels",
    "clean_orphan_labels",
    "generate_empty_labels",
    "generate_missing_labels",
    "categorize_by_class",
    "upload_assets_to_remote",
}


SYNTHETIC_TOOL_SURFACE_METADATA = {
    "remote_prediction_pipeline": {
        "read_only": False,
        "destructive": False,
        "confirmation_required": True,
        "open_world": True,
        "risk_level": "medium",
    },
    "remote_training_pipeline": {
        "read_only": False,
        "destructive": False,
        "confirmation_required": True,
        "open_world": True,
        "risk_level": "high",
    },
}


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    tool_name: str
    read_only: bool
    destructive: bool
    confirmation_required: bool
    open_world: bool
    risk_level: str


def _raw_tool_surface_metadata(tool: Any) -> dict[str, Any]:
    return dict(
        getattr(tool, 'metadata', None)
        or getattr(tool, 'tool_metadata', None)
        or {}
    )


def _raw_tool_surface_annotations(tool: Any) -> dict[str, Any]:
    annotations = getattr(tool, 'annotations', None) or getattr(tool, 'tool_annotations', None)
    if annotations is None:
        return {}
    if isinstance(annotations, dict):
        return dict(annotations)
    values: dict[str, Any] = {}
    for attr_name in ('readOnlyHint', 'destructiveHint', 'idempotentHint', 'openWorldHint'):
        value = getattr(annotations, attr_name, None)
        if value is not None:
            values[attr_name] = value
    return values


def _resolve_policy(
    tool_name: str,
    *,
    metadata: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
) -> ToolExecutionPolicy:
    canonical_name = canonical_tool_name(tool_name)
    metadata = dict(metadata or {})
    annotations = dict(annotations or {})

    if 'read_only' in metadata:
        read_only = bool(metadata.get('read_only'))
    elif 'readOnlyHint' in annotations:
        read_only = bool(annotations.get('readOnlyHint'))
    else:
        read_only = False

    if 'destructive' in metadata:
        destructive = bool(metadata.get('destructive'))
    elif 'destructiveHint' in annotations:
        destructive = bool(annotations.get('destructiveHint'))
    else:
        destructive = False

    explicit_confirmation = metadata.get('confirmation_required')
    if explicit_confirmation is not None:
        confirmation_required = bool(explicit_confirmation)
    elif destructive:
        confirmation_required = True
    elif read_only:
        confirmation_required = False
    else:
        confirmation_required = canonical_name in HIGH_RISK_TOOLS

    if 'open_world' in metadata:
        open_world = bool(metadata.get('open_world'))
    else:
        open_world = bool(annotations.get('openWorldHint'))

    explicit_risk = str(metadata.get('risk_level') or '').strip().lower()
    if explicit_risk in {'low', 'medium', 'high'}:
        risk_level = explicit_risk
    elif confirmation_required:
        risk_level = 'high'
    elif read_only:
        risk_level = 'low'
    else:
        risk_level = 'medium'

    return ToolExecutionPolicy(
        tool_name=canonical_name,
        read_only=read_only,
        destructive=destructive,
        confirmation_required=confirmation_required,
        open_world=open_world,
        risk_level=risk_level,
    )


def resolve_tool_execution_policy(
    tool_name: str,
    *,
    tool_registry: dict[str, Any] | None = None,
) -> ToolExecutionPolicy:
    canonical_name = canonical_tool_name(tool_name)
    if canonical_name in SYNTHETIC_TOOL_SURFACE_METADATA:
        return _resolve_policy(
            canonical_name,
            metadata=dict(SYNTHETIC_TOOL_SURFACE_METADATA.get(canonical_name) or {}),
        )
    tool = (tool_registry or {}).get(canonical_name)
    return _resolve_policy(
        canonical_name,
        metadata=_raw_tool_surface_metadata(tool),
        annotations=_raw_tool_surface_annotations(tool),
    )


def resolve_raw_tool_execution_policy(tool: Any) -> ToolExecutionPolicy:
    return _resolve_policy(
        getattr(tool, 'name', ''),
        metadata=_raw_tool_surface_metadata(tool),
        annotations=_raw_tool_surface_annotations(tool),
    )


def build_manual_interrupt_tool_names(raw_tools: Iterable[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for tool in raw_tools:
        policy = resolve_raw_tool_execution_policy(tool)
        if not policy.tool_name or policy.tool_name in seen:
            continue
        if policy.confirmation_required:
            seen.add(policy.tool_name)
            names.append(policy.tool_name)
    return names


def pending_allowed_decisions(policy: ToolExecutionPolicy) -> list[str]:
    if policy.read_only:
        return ['approve', 'reject', 'clarify']
    if policy.confirmation_required:
        return ['approve', 'reject', 'edit', 'clarify']
    return ['approve', 'reject', 'clarify']
