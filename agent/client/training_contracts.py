from __future__ import annotations

from typing import Any, TypedDict


class TrainingPlanFollowupAction(TypedDict, total=False):
    action: str
    reply: str
    status: str
    draft: dict[str, Any]
    handoff_mode: str
    preamble: str
    append_message: bool


class PrepareOnlyRequestContext(TypedDict):
    matches: bool
    dataset_path: str


class PrepareOnlyResult(TypedDict):
    status: str
    reply: str
    draft: dict[str, Any] | None
    clear_draft: bool
    defer_to_graph: bool


class TrainingRecoveryBootstrap(TypedDict, total=False):
    reply: str
    defer_to_graph: bool
    proceed: bool
    base_args: dict[str, Any]
    dataset_path: str


class TrainingRecoveryOrchestrationResult(TypedDict):
    draft: dict[str, Any]
    reply: str
    defer_to_graph: bool


class TrainingRevisionResult(TypedDict, total=False):
    revised_draft: dict[str, Any]
    followup_action: TrainingPlanFollowupAction


class TrainingPlanDialogueFlowResult(TypedDict, total=False):
    draft_to_save: dict[str, Any]
    followup_action: TrainingPlanFollowupAction


class TrainingRequestGuard(TypedDict, total=False):
    reply: str
    draft: dict[str, Any] | None
    defer_to_graph: bool
    proceed: bool
    return_none: bool


class TrainingRequestContext(TypedDict):
    readiness: dict[str, Any]
    requested_args: dict[str, Any]


class TrainingRequestResult(TypedDict, total=False):
    reply: str
    draft: dict[str, Any] | None
    defer_to_graph: bool
