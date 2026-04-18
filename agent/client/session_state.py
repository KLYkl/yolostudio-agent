from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SESSION_STATE_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DatasetContext:
    dataset_root: str = ""
    img_dir: str = ""
    label_dir: str = ""
    data_yaml: str = ""
    last_scan: dict[str, Any] = field(default_factory=dict)
    last_validate: dict[str, Any] = field(default_factory=dict)
    last_readiness: dict[str, Any] = field(default_factory=dict)
    last_split: dict[str, Any] = field(default_factory=dict)
    last_health_check: dict[str, Any] = field(default_factory=dict)
    last_duplicate_check: dict[str, Any] = field(default_factory=dict)
    last_extract_preview: dict[str, Any] = field(default_factory=dict)
    last_extract_result: dict[str, Any] = field(default_factory=dict)
    last_video_scan: dict[str, Any] = field(default_factory=dict)
    last_frame_extract: dict[str, Any] = field(default_factory=dict)



@dataclass(slots=True)
class PredictionContext:
    source_path: str = ""
    model: str = ""
    output_dir: str = ""
    report_path: str = ""
    image_prediction_session_id: str = ""
    image_prediction_status: str = ""
    last_result: dict[str, Any] = field(default_factory=dict)
    last_summary: dict[str, Any] = field(default_factory=dict)
    last_inspection: dict[str, Any] = field(default_factory=dict)
    last_export: dict[str, Any] = field(default_factory=dict)
    last_path_lists: dict[str, Any] = field(default_factory=dict)
    last_organized_result: dict[str, Any] = field(default_factory=dict)
    last_image_prediction_status: dict[str, Any] = field(default_factory=dict)
    realtime_session_id: str = ""
    realtime_source_type: str = ""
    realtime_source_label: str = ""
    realtime_status: str = ""
    last_realtime_status: dict[str, Any] = field(default_factory=dict)
    last_remote_roundtrip: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeContext:
    last_retrieval: dict[str, Any] = field(default_factory=dict)
    last_analysis: dict[str, Any] = field(default_factory=dict)
    last_recommendation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RemoteTransferContext:
    target_label: str = ""
    profile_name: str = ""
    remote_root: str = ""
    last_profile_listing: dict[str, Any] = field(default_factory=dict)
    last_upload: dict[str, Any] = field(default_factory=dict)
    last_download: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingContext:
    workflow_state: str = "idle"
    loop_workflow_state: str = "loop_idle"
    running: bool = False
    model: str = ""
    data_yaml: str = ""
    device: str = ""
    training_environment: str = ""
    project: str = ""
    run_name: str = ""
    batch: int | None = None
    imgsz: int | None = None
    fraction: float | None = None
    classes: list[int] = field(default_factory=list)
    single_cls: bool | None = None
    optimizer: str = ""
    freeze: int | None = None
    resume: bool | None = None
    lr0: float | None = None
    patience: int | None = None
    workers: int | None = None
    amp: bool | None = None
    pid: int | None = None
    log_file: str = ""
    started_at: float | None = None
    last_status: dict[str, Any] = field(default_factory=dict)
    last_summary: dict[str, Any] = field(default_factory=dict)
    training_run_summary: dict[str, Any] = field(default_factory=dict)
    last_start_result: dict[str, Any] = field(default_factory=dict)
    last_environment_probe: dict[str, Any] = field(default_factory=dict)
    last_preflight: dict[str, Any] = field(default_factory=dict)
    recent_runs: list[dict[str, Any]] = field(default_factory=list)
    last_run_inspection: dict[str, Any] = field(default_factory=dict)
    last_run_comparison: dict[str, Any] = field(default_factory=dict)
    best_run_selection: dict[str, Any] = field(default_factory=dict)
    training_plan_draft: dict[str, Any] = field(default_factory=dict)
    last_remote_roundtrip: dict[str, Any] = field(default_factory=dict)
    active_loop_id: str = ""
    active_loop_name: str = ""
    active_loop_status: str = ""
    active_loop_request: dict[str, Any] = field(default_factory=dict)
    last_loop_status: dict[str, Any] = field(default_factory=dict)
    last_loop_detail: dict[str, Any] = field(default_factory=dict)
    recent_loops: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PendingConfirmation:
    thread_id: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    source: str = "synthetic"
    interrupt_kind: str = "tool_approval"
    objective: str = ""
    summary: str = ""
    allowed_decisions: list[str] = field(default_factory=lambda: ["approve", "reject", "edit", "clarify"])
    review_config: dict[str, Any] = field(default_factory=dict)
    decision_context: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(slots=True)
class UserPreferences:
    default_model: str = ""
    default_epochs: int | None = None
    language: str = "zh-CN"


@dataclass(slots=True)
class SessionState:
    session_id: str
    schema_version: int = SESSION_STATE_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    active_dataset: DatasetContext = field(default_factory=DatasetContext)
    active_training: TrainingContext = field(default_factory=TrainingContext)
    active_prediction: PredictionContext = field(default_factory=PredictionContext)
    active_knowledge: KnowledgeContext = field(default_factory=KnowledgeContext)
    active_remote_transfer: RemoteTransferContext = field(default_factory=RemoteTransferContext)
    pending_confirmation: PendingConfirmation = field(default_factory=PendingConfirmation)
    preferences: UserPreferences = field(default_factory=UserPreferences)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, session_id_fallback: str = '') -> 'SessionState':
        data = migrate_session_state_payload(data, session_id_fallback=session_id_fallback)
        return cls(
            session_id=data['session_id'],
            schema_version=int(data.get('schema_version') or SESSION_STATE_SCHEMA_VERSION),
            created_at=data.get('created_at', utc_now()),
            updated_at=data.get('updated_at', utc_now()),
            active_dataset=DatasetContext(**data.get('active_dataset', {})),
            active_training=TrainingContext(**data.get('active_training', {})),
            active_prediction=PredictionContext(**data.get('active_prediction', {})),
            active_knowledge=KnowledgeContext(**data.get('active_knowledge', {})),
            active_remote_transfer=RemoteTransferContext(**data.get('active_remote_transfer', {})),
            pending_confirmation=PendingConfirmation(**data.get('pending_confirmation', {})),
            preferences=UserPreferences(**data.get('preferences', {})),
        )


def migrate_session_state_payload(data: dict[str, Any] | None, *, session_id_fallback: str = '') -> dict[str, Any]:
    payload = dict(data or {})
    raw_version = payload.get('schema_version')
    try:
        schema_version = int(raw_version)
    except (TypeError, ValueError):
        schema_version = 1

    payload['session_id'] = str(payload.get('session_id') or session_id_fallback or '').strip()
    if not payload['session_id']:
        raise KeyError('session_id')

    for field_name in (
        'active_dataset',
        'active_training',
        'active_prediction',
        'active_knowledge',
        'active_remote_transfer',
        'pending_confirmation',
        'preferences',
    ):
        if not isinstance(payload.get(field_name), dict):
            payload[field_name] = {}

    if schema_version < 2:
        pending = dict(payload.get('pending_confirmation') or {})
        if not isinstance(pending.get('allowed_decisions'), list) or not pending.get('allowed_decisions'):
            pending['allowed_decisions'] = ["approve", "reject", "edit", "clarify"]
        if not isinstance(pending.get('review_config'), dict):
            pending['review_config'] = {}
        if not isinstance(pending.get('decision_context'), dict):
            pending['decision_context'] = {}
        if not str(pending.get('source') or '').strip():
            pending['source'] = 'synthetic'
        payload['pending_confirmation'] = pending

        active_training = dict(payload.get('active_training') or {})
        if not isinstance(active_training.get('training_plan_draft'), dict):
            active_training['training_plan_draft'] = {}
        payload['active_training'] = active_training

    payload['schema_version'] = SESSION_STATE_SCHEMA_VERSION
    return payload
