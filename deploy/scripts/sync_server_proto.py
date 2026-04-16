from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_PROTO_ROOT = REPO_ROOT / 'deploy' / 'server_proto'
SERVER_PROTO_PACKAGE_ROOT = SERVER_PROTO_ROOT / 'agent_plan'

IGNORED_DIR_NAMES = {
    '.venv',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
}
IGNORED_FILE_NAMES = {
    '.DS_Store',
    'Thumbs.db',
}
IGNORED_FILE_SUFFIXES = {
    '.pyc',
    '.pyo',
}
IGNORED_FILE_ENDINGS = (
    '_output.json',
)
IGNORED_NAME_PREFIXES = (
    '_tmp_',
)


@dataclass(frozen=True, slots=True)
class MirrorMapping:
    source_rel: str
    dest_rel: str
    description: str

    @property
    def source(self) -> Path:
        return REPO_ROOT / self.source_rel

    @property
    def dest(self) -> Path:
        return SERVER_PROTO_ROOT / self.dest_rel


@dataclass(frozen=True, slots=True)
class DriftRecord:
    kind: str
    source: str
    dest: str


@dataclass(frozen=True, slots=True)
class SyncStats:
    copied: int = 0
    deleted: int = 0
    unchanged: int = 0

    def merge(self, other: 'SyncStats') -> 'SyncStats':
        return SyncStats(
            copied=self.copied + other.copied,
            deleted=self.deleted + other.deleted,
            unchanged=self.unchanged + other.unchanged,
        )


MANAGED_MAPPINGS: tuple[MirrorMapping, ...] = (
    MirrorMapping('__init__.py', 'agent_plan/__init__.py', '兼容根包入口'),
    MirrorMapping('agent/__init__.py', 'agent_plan/agent/__init__.py', 'agent 根包入口'),
    MirrorMapping('agent/AGENT.md', 'agent_plan/agent/AGENT.md', '运行时 agent 规范'),
    MirrorMapping('agent/client', 'agent_plan/agent/client', 'agent client 真源'),
    MirrorMapping('agent/server', 'agent_plan/agent/server', 'agent server 真源'),
    MirrorMapping('agent/tests', 'agent_plan/agent/tests', 'agent 测试与回归矩阵'),
    MirrorMapping('knowledge', 'agent_plan/knowledge', '知识库规则'),
    MirrorMapping('yolostudio_agent', 'agent_plan/yolostudio_agent', '公开命名空间'),
)

DEPLOY_ONLY_PATHS: tuple[str, ...] = (
    'agent_plan/agent/.venv',
    'agent_plan/agent/README.md',
    'core',
    'utils',
)


def _is_ignored_path(relative_path: Path) -> bool:
    for part in relative_path.parts:
        if part in IGNORED_DIR_NAMES or part.startswith(IGNORED_NAME_PREFIXES):
            return True
    name = relative_path.name
    if not name:
        return False
    if name in IGNORED_FILE_NAMES:
        return True
    if name.startswith(IGNORED_NAME_PREFIXES):
        return True
    if any(name.endswith(suffix) for suffix in IGNORED_FILE_SUFFIXES):
        return True
    if any(name.endswith(suffix) for suffix in IGNORED_FILE_ENDINGS):
        return True
    return False


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_dir(root: Path) -> dict[Path, str]:
    snapshot: dict[Path, str] = {}
    if not root.exists():
        return snapshot
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_ignored_path(rel):
            continue
        snapshot[rel] = _file_digest(path)
    return snapshot


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((candidate for candidate in root.rglob('*') if candidate.is_dir()), key=lambda item: len(item.parts), reverse=True):
        rel = path.relative_to(root)
        if _is_ignored_path(rel):
            continue
        if any(path.iterdir()):
            continue
        path.rmdir()


def collect_drift() -> list[DriftRecord]:
    drift: list[DriftRecord] = []
    for mapping in MANAGED_MAPPINGS:
        source = mapping.source
        dest = mapping.dest
        if source.is_file():
            if not dest.exists():
                drift.append(DriftRecord('missing', mapping.source_rel, mapping.dest_rel))
                continue
            if _file_digest(source) != _file_digest(dest):
                drift.append(DriftRecord('content', mapping.source_rel, mapping.dest_rel))
            continue

        source_snapshot = _snapshot_dir(source)
        dest_snapshot = _snapshot_dir(dest)
        for rel in sorted(source_snapshot):
            if rel not in dest_snapshot:
                drift.append(DriftRecord('missing', f'{mapping.source_rel}/{rel.as_posix()}', f'{mapping.dest_rel}/{rel.as_posix()}'))
                continue
            if source_snapshot[rel] != dest_snapshot[rel]:
                drift.append(DriftRecord('content', f'{mapping.source_rel}/{rel.as_posix()}', f'{mapping.dest_rel}/{rel.as_posix()}'))
        for rel in sorted(dest_snapshot):
            if rel not in source_snapshot:
                drift.append(DriftRecord('extra', f'{mapping.source_rel}/{rel.as_posix()}', f'{mapping.dest_rel}/{rel.as_posix()}'))
    return drift


def sync_server_proto() -> SyncStats:
    stats = SyncStats()
    for mapping in MANAGED_MAPPINGS:
        source = mapping.source
        dest = mapping.dest
        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and _file_digest(source) == _file_digest(dest):
                stats = stats.merge(SyncStats(unchanged=1))
                continue
            shutil.copy2(source, dest)
            stats = stats.merge(SyncStats(copied=1))
            continue

        source_snapshot = _snapshot_dir(source)
        dest_snapshot = _snapshot_dir(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for rel in sorted(dest_snapshot):
            if rel in source_snapshot:
                continue
            target = dest / rel
            target.unlink()
            stats = stats.merge(SyncStats(deleted=1))
        for rel, source_digest in source_snapshot.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and _file_digest(target) == source_digest:
                stats = stats.merge(SyncStats(unchanged=1))
                continue
            shutil.copy2(source / rel, target)
            stats = stats.merge(SyncStats(copied=1))
        _remove_empty_dirs(dest)
    return stats


def _format_mappings() -> str:
    lines = ['Managed mirror mappings:']
    for mapping in MANAGED_MAPPINGS:
        lines.append(f'- {mapping.source_rel} -> {mapping.dest_rel} ({mapping.description})')
    lines.append('Deploy-only paths:')
    for path in DEPLOY_ONLY_PATHS:
        lines.append(f'- {path}')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Synchronize deploy/server_proto managed mirror paths from the main source tree.')
    parser.add_argument('--check', action='store_true', help='Only check for drift; do not modify files.')
    parser.add_argument('--list', action='store_true', help='Print the managed mirror mapping and exit.')
    args = parser.parse_args(argv)

    if args.list:
        print(_format_mappings())
        return 0

    drift = collect_drift()
    if args.check:
        if not drift:
            print('server_proto mirror is in sync')
            return 0
        print('server_proto mirror drift detected:')
        for record in drift[:200]:
            print(f'- {record.kind}: {record.dest} <= {record.source}')
        if len(drift) > 200:
            print(f'... {len(drift) - 200} more drift records omitted')
        return 1

    stats = sync_server_proto()
    print(f'server_proto sync complete: copied={stats.copied} deleted={stats.deleted} unchanged={stats.unchanged}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
