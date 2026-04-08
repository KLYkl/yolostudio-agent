from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DatasetContext:
    img_dir: str = ""
    label_dir: str = ""
    data_yaml: str = ""
    last_scan: dict[str, Any] = field(default_factory=dict)
    last_validate: dict[str, Any] = field(default_factory=dict)
    last_split: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingContext:
    running: bool = False
    model: str = ""
    data_yaml: str = ""
    device: str = ""
    pid: int | None = None
    log_file: str = ""
    started_at: float | None = None
    last_status: dict[str, Any] = field(default_factory=dict)
    last_start_result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PendingConfirmation:
    thread_id: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(slots=True)
class UserPreferences:
    default_model: str = ""
    default_epochs: int | None = None
    language: str = "zh-CN"


@dataclass(slots=True)
class SessionState:
    session_id: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    active_dataset: DatasetContext = field(default_factory=DatasetContext)
    active_training: TrainingContext = field(default_factory=TrainingContext)
    pending_confirmation: PendingConfirmation = field(default_factory=PendingConfirmation)
    preferences: UserPreferences = field(default_factory=UserPreferences)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SessionState':
        return cls(
            session_id=data['session_id'],
            created_at=data.get('created_at', utc_now()),
            updated_at=data.get('updated_at', utc_now()),
            active_dataset=DatasetContext(**data.get('active_dataset', {})),
            active_training=TrainingContext(**data.get('active_training', {})),
            pending_confirmation=PendingConfirmation(**data.get('pending_confirmation', {})),
            preferences=UserPreferences(**data.get('preferences', {})),
        )
