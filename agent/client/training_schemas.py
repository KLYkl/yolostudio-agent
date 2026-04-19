from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainingEdits(BaseModel):
    model: str | None = None
    epochs: int | None = None
    batch: int | None = None
    imgsz: int | None = None
    device: str | None = None
    training_environment: str | None = None
    data_yaml: str | None = None
    max_rounds: int | None = None
    epochs_per_round: int | None = None
    loop_name: str | None = None


class PendingTurnIntent(BaseModel):
    action: Literal['approve', 'reject', 'edit', 'status', 'new_task', 'unclear']
    edits: TrainingEdits | None = None
    reason: str | None = None


class _BaseTrainingPlan(BaseModel):
    dataset_path: str
    model: str
    batch: int = 16
    imgsz: int = 640
    device: str = ''
    training_environment: str = ''
    data_yaml: str = ''
    project: str = ''
    name: str = ''
    fraction: float | None = None
    classes: list[int] = Field(default_factory=list)
    single_cls: bool | None = None
    optimizer: str = ''
    freeze: int | list[int] | None = None
    resume: bool | str | None = None
    lr0: float | None = None
    patience: int | None = None
    workers: int | None = None
    amp: bool | None = None
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    readiness_summary: str = ''
    prepare_summary: str = ''


class TrainPlan(_BaseTrainingPlan):
    mode: Literal['train'] = 'train'
    epochs: int = 100


class LoopTrainPlan(_BaseTrainingPlan):
    mode: Literal['loop'] = 'loop'
    max_rounds: int = 5
    epochs_per_round: int = 10
    loop_name: str = ''


TrainingPlan = TrainPlan | LoopTrainPlan


def _model_validate_compat(model_cls: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    validator = getattr(model_cls, 'model_validate', None)
    if callable(validator):
        return validator(payload)
    parser = getattr(model_cls, 'parse_obj', None)
    if callable(parser):
        return parser(payload)
    return model_cls(**payload)


def _model_dump_compat(model: BaseModel, *, exclude_none: bool = True) -> dict[str, Any]:
    dumper = getattr(model, 'model_dump', None)
    if callable(dumper):
        try:
            return dumper(exclude_none=exclude_none)
        except TypeError:
            return dumper()
    return model.dict(exclude_none=exclude_none)


def _model_copy_compat(model: BaseModel, *, update: dict[str, Any]) -> BaseModel:
    copier = getattr(model, 'model_copy', None)
    if callable(copier):
        return copier(update=update)
    legacy_copy = getattr(model, 'copy', None)
    if callable(legacy_copy):
        return legacy_copy(update=update)
    merged = _model_dump_compat(model, exclude_none=False)
    merged.update(update)
    return model.__class__(**merged)


def coerce_training_plan(value: TrainingPlan | dict[str, Any]) -> TrainingPlan:
    if isinstance(value, (TrainPlan, LoopTrainPlan)):
        return value
    payload = dict(value or {})
    mode = str(payload.get('mode') or '').strip().lower()
    if mode == 'loop' or any(key in payload for key in ('max_rounds', 'epochs_per_round', 'loop_name')):
        return _model_validate_compat(LoopTrainPlan, payload)
    return _model_validate_compat(TrainPlan, payload)


def merge_training_plan_edits(plan: TrainingPlan | dict[str, Any], edits: TrainingEdits | dict[str, Any] | None) -> TrainingPlan:
    normalized_plan = coerce_training_plan(plan)
    if edits is None:
        return normalized_plan
    if isinstance(edits, TrainingEdits):
        edit_payload = _model_dump_compat(edits)
    else:
        edit_payload = {key: value for key, value in dict(edits or {}).items() if value is not None}
    if isinstance(normalized_plan, LoopTrainPlan):
        if 'epochs' in edit_payload and 'epochs_per_round' not in edit_payload:
            edit_payload['epochs_per_round'] = edit_payload.pop('epochs')
        return _model_copy_compat(normalized_plan, update=edit_payload)
    edit_payload.pop('max_rounds', None)
    edit_payload.pop('epochs_per_round', None)
    edit_payload.pop('loop_name', None)
    return _model_copy_compat(normalized_plan, update=edit_payload)


def update_plan_after_prepare(
    plan: TrainingPlan | dict[str, Any],
    prepare_result: dict[str, Any] | None = None,
    readiness: dict[str, Any] | None = None,
) -> TrainingPlan:
    normalized_plan = coerce_training_plan(plan)
    update: dict[str, Any] = {}
    prepare_payload = dict(prepare_result or {})
    readiness_payload = dict(readiness or {})
    data_yaml = str(
        prepare_payload.get('data_yaml')
        or prepare_payload.get('resolved_data_yaml')
        or readiness_payload.get('resolved_data_yaml')
        or ''
    ).strip()
    if data_yaml:
        update['data_yaml'] = data_yaml
    blockers = [str(item).strip() for item in (readiness_payload.get('blockers') or prepare_payload.get('blockers') or []) if str(item).strip()]
    warnings = [str(item).strip() for item in (readiness_payload.get('warnings') or prepare_payload.get('warnings') or []) if str(item).strip()]
    summary = str(
        readiness_payload.get('summary')
        or prepare_payload.get('summary')
        or prepare_payload.get('message')
        or ''
    ).strip()
    prepare_summary = str(prepare_payload.get('summary') or prepare_payload.get('message') or '').strip()
    if blockers:
        update['blockers'] = blockers
    if warnings:
        update['warnings'] = warnings
    if summary:
        update['readiness_summary'] = summary
    if prepare_summary:
        update['prepare_summary'] = prepare_summary
    if 'training_environment' in prepare_payload and str(prepare_payload.get('training_environment') or '').strip():
        update['training_environment'] = str(prepare_payload.get('training_environment') or '').strip()
    return _model_copy_compat(normalized_plan, update=update)
