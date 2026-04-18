from __future__ import annotations

import pickle
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


class FileCheckpointSaver(InMemorySaver):
    """A simple file-backed LangGraph checkpointer for local durable resumes.

    This is not a multi-process production database. It persists the in-memory
    checkpoint state to a local pickle file so CLI restarts can resume pending
    HITL/tool interrupts in the current single-user workspace.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.load_status = 'missing'
        self.load_error_type = ''
        self.corrupt_backup_name = ''
        self._load()

    def _to_plain_storage(self) -> dict[str, Any]:
        storage: dict[str, Any] = {}
        for thread_id, ns_map in self.storage.items():
            storage[thread_id] = {checkpoint_ns: dict(entries) for checkpoint_ns, entries in ns_map.items()}
        return storage

    def _snapshot(self) -> dict[str, Any]:
        return {
            'storage': self._to_plain_storage(),
            'writes': {key: dict(value) for key, value in self.writes.items()},
            'blobs': dict(self.blobs),
        }

    def _persist_unlocked(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + '.tmp')
        with tmp.open('wb') as fh:
            pickle.dump(self._snapshot(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)

    def _mark_corrupt(self, exc: Exception) -> None:
        self.load_status = 'corrupt_detected'
        self.load_error_type = exc.__class__.__name__
        self.corrupt_backup_name = ''
        corrupt = self.path.with_suffix(self.path.suffix + '.corrupt')
        try:
            if corrupt.exists():
                corrupt.unlink()
            self.path.replace(corrupt)
            self.load_status = 'corrupt_recovered'
            self.corrupt_backup_name = corrupt.name
        except Exception:
            pass

    def _load(self) -> None:
        if not self.path.exists():
            self.load_status = 'missing'
            self.load_error_type = ''
            self.corrupt_backup_name = ''
            return
        try:
            with self._lock:
                with self.path.open('rb') as fh:
                    raw = pickle.load(fh)
                storage = defaultdict(lambda: defaultdict(dict))
                for thread_id, ns_map in (raw.get('storage') or {}).items():
                    storage[thread_id] = defaultdict(
                        dict,
                        {checkpoint_ns: dict(entries) for checkpoint_ns, entries in ns_map.items()},
                    )
                writes = defaultdict(dict, {tuple(key): dict(value) for key, value in (raw.get('writes') or {}).items()})
                blobs = dict(raw.get('blobs') or {})
        except Exception as exc:
            self._mark_corrupt(exc)
            return
        self.storage = storage
        self.writes = writes
        self.blobs = blobs
        self.load_status = 'ok'
        self.load_error_type = ''
        self.corrupt_backup_name = ''

    def health_payload(self) -> dict[str, Any]:
        payload = {
            'status': str(self.load_status or 'missing'),
            'checkpoint_name': self.path.name,
        }
        if self.load_error_type:
            payload['error_type'] = self.load_error_type
        if self.corrupt_backup_name:
            payload['backup_name'] = self.corrupt_backup_name
        return payload

    def put(self, config, checkpoint, metadata, new_versions):
        with self._lock:
            result = super().put(config, checkpoint, metadata, new_versions)
            self._persist_unlocked()
            return result

    def put_writes(self, config, writes, task_id, task_path: str = '') -> None:
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            self._persist_unlocked()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._persist_unlocked()

    def thread_ids(self, prefix: str = '') -> list[str]:
        with self._lock:
            ids = [str(thread_id) for thread_id in self.storage.keys()]
        if prefix:
            ids = [thread_id for thread_id in ids if thread_id.startswith(prefix)]
        return sorted(ids)

    def reset(self) -> None:
        with self._lock:
            self.storage = defaultdict(lambda: defaultdict(dict))
            self.writes = defaultdict(dict)
            self.blobs = {}
            if self.path.exists():
                self.path.unlink()
