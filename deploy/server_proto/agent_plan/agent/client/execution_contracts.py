from __future__ import annotations

from typing import Any, TypedDict

from yolostudio_agent.agent.client.training_contracts import TrainingPlanFollowupAction


class PredictionRequestFollowupAction(TypedDict, total=False):
    action: str
    reply: str
    status: str


class PostPrepareTrainingStartResult(TypedDict, total=False):
    preflight: dict[str, Any]
    draft: dict[str, Any]
    followup_action: TrainingPlanFollowupAction


class RemoteTrainingStartResult(TypedDict, total=False):
    ok: bool
    stage: str
    dataset_path: str
    model_path: str
    readiness: dict[str, Any]
    prepare: dict[str, Any]
    preflight: dict[str, Any]
    start: dict[str, Any]


class RemoteTrainingWaitResult(TypedDict, total=False):
    ok: bool
    timed_out: bool
    message: str
    status_result: dict[str, Any]
    summary_result: dict[str, Any]
    inspect_result: dict[str, Any]
    status_checks: list[dict[str, Any]]


class RemoteTrainingPipelineFlowResult(TypedDict, total=False):
    stage: str
    upload: dict[str, Any]
    resolved_inputs: dict[str, Any]
    start_flow: RemoteTrainingStartResult
    wait: RemoteTrainingWaitResult
    download: dict[str, Any]
    pipeline_result: dict[str, Any]


class RemotePredictionPipelineFlowResult(TypedDict, total=False):
    stage: str
    upload: dict[str, Any]
    resolved_inputs: dict[str, Any]
    predict: dict[str, Any]
    download: dict[str, Any]
    pipeline_result: dict[str, Any]
