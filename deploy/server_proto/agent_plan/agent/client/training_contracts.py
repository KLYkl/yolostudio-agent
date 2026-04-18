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


class TrainingRevisionResult(TypedDict, total=False):
    revised_draft: dict[str, Any]
    followup_action: TrainingPlanFollowupAction


class TrainingPlanDialogueFlowResult(TypedDict, total=False):
    draft_to_save: dict[str, Any]
    followup_action: TrainingPlanFollowupAction
