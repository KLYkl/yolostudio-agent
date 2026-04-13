from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from yolostudio_agent.agent.client.intent_parsing import extract_all_paths_from_text


_WINDOWS_SSH_CANDIDATES = {
    'ssh': [
        os.getenv('YOLOSTUDIO_SSH_EXE', '').strip(),
        r'D:\Tools\Git\usr\bin\ssh.exe',
        r'C:\Program Files\Git\usr\bin\ssh.exe',
    ],
    'scp': [
        os.getenv('YOLOSTUDIO_SCP_EXE', '').strip(),
        r'D:\Tools\Git\usr\bin\scp.exe',
        r'C:\Program Files\Git\usr\bin\scp.exe',
    ],
}


class _ListRemoteProfilesArgs(BaseModel):
    profiles_path: str = Field(default='', description='可选远端 profile 配置文件路径；不传则按默认搜索路径查找。')


class _UploadAssetsToRemoteArgs(BaseModel):
    local_paths: list[str] = Field(default=[], description='本地文件或目录路径列表。')
    paths_text: str = Field(default='', description='兼容字段；多个路径可用换行、逗号或分号分隔。')
    server: str = Field(default='', description='远端 profile 名、SSH alias，或 user@host 形式的目标。')
    profile: str = Field(default='', description='显式指定远端 profile 名。')
    remote_root: str = Field(default='', description='远端目标根目录；不传时尝试使用 profile 默认值。')
    host: str = Field(default='', description='显式远端主机名或 IP。')
    username: str = Field(default='', description='显式远端用户名。')
    port: int = Field(default=0, description='显式 SSH 端口；0 表示使用默认值。')
    recursive: bool = Field(default=True, description='目录上传时是否递归复制。')
    create_remote_root: bool = Field(default=True, description='上传前是否自动创建远端目录。')
    profiles_path: str = Field(default='', description='可选远端 profile 配置文件路径；不传则按默认搜索路径查找。')
    resume: bool = Field(default=True, description='大文件上传时是否启用断点续传。')
    verify_hash: bool = Field(default=True, description='上传完成后是否做哈希校验。')
    hash_algorithm: str = Field(default='sha256', description='校验算法；当前支持 sha256 / md5。')
    large_file_threshold_mb: int = Field(default=256, description='达到该体积后自动切到大文件分块模式。')
    chunk_size_mb: int = Field(default=64, description='大文件分块上传的单块大小，单位 MB。')
    show_progress: bool = Field(default=True, description='上传时是否在本机标准输出打印进度。')


class _DownloadAssetsFromRemoteArgs(BaseModel):
    remote_paths: list[str] = Field(default=[], description='远端文件或目录路径列表。')
    paths_text: str = Field(default='', description='兼容字段；多个远端路径可用换行、逗号或分号分隔。')
    server: str = Field(default='', description='远端 profile 名、SSH alias，或 user@host 形式的目标。')
    profile: str = Field(default='', description='显式指定远端 profile 名。')
    host: str = Field(default='', description='显式远端主机名或 IP。')
    username: str = Field(default='', description='显式远端用户名。')
    port: int = Field(default=0, description='显式 SSH 端口；0 表示使用默认值。')
    recursive: bool = Field(default=True, description='下载目录时是否递归复制。')
    local_root: str = Field(default='', description='本地接收根目录；不传时默认写到仓库 output/remote_downloads。')
    profiles_path: str = Field(default='', description='可选远端 profile 配置文件路径；不传则按默认搜索路径查找。')


class RemoteTransferError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_profile_candidates(explicit_path: str = '') -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_path = os.getenv('YOLOSTUDIO_REMOTE_PROFILES_PATH', '').strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    repo_root = _repo_root()
    candidates.extend(
        [
            repo_root / '.codex' / 'remote_profiles.json',
            Path.home() / '.yolostudio' / 'remote_profiles.json',
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _load_profile_document(explicit_path: str = '') -> tuple[dict[str, Any], str]:
    for candidate in _default_profile_candidates(explicit_path):
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                return data, str(candidate)
    return {}, ''


def _coerce_profiles(document: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    if not document:
        return {}, ''
    raw_profiles = document.get('profiles', document)
    if not isinstance(raw_profiles, dict):
        return {}, str(document.get('default_profile') or '')
    profiles: dict[str, dict[str, Any]] = {}
    for name, payload in raw_profiles.items():
        if isinstance(payload, dict):
            profiles[str(name)] = dict(payload)
    return profiles, str(document.get('default_profile') or '')


def _parse_ssh_config() -> list[dict[str, Any]]:
    config_path = Path.home() / '.ssh' / 'config'
    if not config_path.is_file():
        return []
    aliases: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in config_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        host_match = re.match(r'Host\s+(.+)', line, flags=re.I)
        if host_match:
            patterns = [item.strip() for item in host_match.group(1).split() if item.strip()]
            if len(patterns) == 1 and not any(ch in patterns[0] for ch in ('*', '?', '!')):
                current = {'name': patterns[0]}
                aliases.append(current)
            else:
                current = None
            continue
        if current is None:
            continue
        kv_match = re.match(r'(HostName|User|Port)\s+(.+)', line, flags=re.I)
        if kv_match:
            current[kv_match.group(1).lower()] = kv_match.group(2).strip()
    return aliases


def _discover_executable(name: str) -> str:
    explicit_env = os.getenv(f'YOLOSTUDIO_{name.upper()}_EXE', '').strip()
    if explicit_env and Path(explicit_env).exists():
        return explicit_env
    if os.name == 'nt':
        for candidate in _WINDOWS_SSH_CANDIDATES.get(name, []):
            if candidate and Path(candidate).exists():
                return candidate
    resolved = shutil.which(name)
    if resolved:
        return resolved
    raise RemoteTransferError(f'未找到可用的 {name} 可执行文件；请安装 ssh/scp，或通过 YOLOSTUDIO_{name.upper()}_EXE 指定路径。')
def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def _run_command_bytes(command: list[str], input_bytes: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=input_bytes,
        text=False,
        capture_output=True,
        check=False,
    )


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    proc = _run_command(command)
    if proc.returncode != 0:
        stderr = (proc.stderr or '').strip()
        stdout = (proc.stdout or '').strip()
        detail = stderr or stdout or f'exit code={proc.returncode}'
        raise RemoteTransferError(detail)
    return proc


def _run_checked_bytes(command: list[str], input_bytes: bytes) -> subprocess.CompletedProcess[bytes]:
    proc = _run_command_bytes(command, input_bytes)
    if proc.returncode != 0:
        stderr = (proc.stderr or b'').decode('utf-8', errors='ignore').strip()
        stdout = (proc.stdout or b'').decode('utf-8', errors='ignore').strip()
        detail = stderr or stdout or f'exit code={proc.returncode}'
        raise RemoteTransferError(detail)
    return proc


def _normalize_path_tokens(local_paths: list[str] | None, paths_text: str = '') -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in list(local_paths or []):
        value = str(raw or '').strip().strip('"').strip("'")
        if value and value not in seen:
            seen.add(value)
            items.append(value)
    if paths_text:
        extracted = extract_all_paths_from_text(paths_text)
        if not extracted:
            extracted = [part.strip().strip('"').strip("'") for part in re.split(r'[\r\n,;；]+', paths_text) if part.strip()]
        for value in extracted:
            if value and value not in seen:
                seen.add(value)
                items.append(value)
    return items


def _normalize_remote_path_tokens(remote_paths: list[str] | None, paths_text: str = '') -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in list(remote_paths or []):
        value = str(raw or '').strip().strip('"').strip("'")
        if value and value not in seen:
            seen.add(value)
            items.append(value)
    if paths_text:
        extracted = [part.strip().strip('"').strip("'") for part in re.split(r'[\r\n,;；]+', paths_text) if part.strip()]
        for value in extracted:
            if value and value not in seen:
                seen.add(value)
                items.append(value)
    return items


def _build_target_label(profile_name: str, server: str, host: str, ssh_target: str) -> str:
    for candidate in (profile_name, server, host, ssh_target):
        value = str(candidate or '').strip()
        if value:
            return value
    return 'remote'


def _resolve_remote_target(*, server: str, profile: str, remote_root: str, host: str, username: str, port: int, profiles_path: str) -> dict[str, Any]:
    document, config_path = _load_profile_document(profiles_path)
    profiles, default_profile = _coerce_profiles(document)

    selected_profile_name = ''
    profile_name = (profile or '').strip()
    server_name = (server or '').strip()
    if profile_name and profile_name in profiles:
        selected_profile_name = profile_name
    elif server_name and server_name in profiles:
        selected_profile_name = server_name
    elif not host and not server_name and default_profile and default_profile in profiles:
        selected_profile_name = default_profile

    profile_payload = profiles.get(selected_profile_name, {}) if selected_profile_name else {}

    resolved_host = str(host or profile_payload.get('host') or profile_payload.get('hostname') or '').strip()
    resolved_user = str(username or profile_payload.get('username') or profile_payload.get('user') or '').strip()
    resolved_port = int(port or profile_payload.get('port') or 0)
    resolved_remote_root = str(remote_root or profile_payload.get('remote_root') or profile_payload.get('default_remote_root') or '').strip()

    ssh_target = str(profile_payload.get('ssh_target') or '').strip()
    if not ssh_target:
        if resolved_host:
            ssh_target = f'{resolved_user}@{resolved_host}' if resolved_user else resolved_host
        elif server_name:
            ssh_target = server_name
    if not ssh_target:
        raise RemoteTransferError('缺少远端目标；请提供 server/profile，或显式传 host/username。')
    if not resolved_remote_root:
        raise RemoteTransferError('缺少 remote_root；请显式提供远端目录，或在远端 profile 里配置默认 remote_root。')

    return {
        'profile_name': selected_profile_name,
        'config_path': config_path,
        'ssh_target': ssh_target,
        'remote_root': resolved_remote_root,
        'port': resolved_port,
        'target_label': _build_target_label(selected_profile_name, server_name, resolved_host, ssh_target),
    }


def _join_remote_path(root: str, *parts: str) -> str:
    path = PurePosixPath(str(root or '/'))
    for part in parts:
        text = str(part or '')
        if not text:
            continue
        for chunk in text.replace('\\', '/').split('/'):
            if chunk and chunk not in {'.'}:
                path /= chunk
    return path.as_posix()


def _quote_remote(value: str) -> str:
    return shlex.quote(str(value))


def _build_remote_python_command(script: str, *args: str) -> str:
    arg_text = ' '.join(_quote_remote(arg) for arg in args)
    python_cmd = f"python3 -c {_quote_remote(script)}"
    fallback_cmd = f"python -c {_quote_remote(script)}"
    if arg_text:
        python_cmd = f'{python_cmd} {arg_text}'
        fallback_cmd = f'{fallback_cmd} {arg_text}'
    return f'if command -v python3 >/dev/null 2>&1; then {python_cmd}; else {fallback_cmd}; fi'


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == 'B':
                return f'{int(size)}{unit}'
            return f'{size:.1f}{unit}'
        size /= 1024.0
    return f'{int(value)}B'


def _emit_progress(*, label: str, total_bytes: int, transferred_bytes: int, skipped_bytes: int, show_progress: bool) -> None:
    if not show_progress or total_bytes <= 0:
        return
    completed = min(total_bytes, transferred_bytes + skipped_bytes)
    percent = (completed / total_bytes) * 100.0 if total_bytes else 100.0
    print(
        f'[remote-upload] {label}: {percent:5.1f}% | sent {_format_bytes(transferred_bytes)} | '
        f'skipped {_format_bytes(skipped_bytes)} | total {_format_bytes(total_bytes)}',
        flush=True,
    )


def _compute_local_hash(path: Path, algorithm: str) -> str:
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise RemoteTransferError(f'不支持的哈希算法: {algorithm}') from exc
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _remote_file_size(ssh_args: list[str], ssh_target: str, remote_path: str) -> int:
    script = 'import os,sys; p=sys.argv[1]; print(os.path.getsize(p) if os.path.exists(p) else -1)'
    proc = _run_checked(ssh_args + [ssh_target, _build_remote_python_command(script, remote_path)])
    try:
        return int((proc.stdout or '').strip() or '-1')
    except ValueError:
        return -1


def _remote_read_text(ssh_args: list[str], ssh_target: str, remote_path: str) -> str:
    script = (
        'import os,sys; p=sys.argv[1]; '
        'print(open(p, "r", encoding="utf-8").read() if os.path.exists(p) else "", end="")'
    )
    proc = _run_checked(ssh_args + [ssh_target, _build_remote_python_command(script, remote_path)])
    return proc.stdout or ''


def _remote_hash(ssh_args: list[str], ssh_target: str, remote_path: str, algorithm: str) -> str:
    script = (
        'import hashlib, os, sys; '
        'algo=sys.argv[2]; p=sys.argv[1]; '
        'h=hashlib.new(algo); '
        'exists=os.path.exists(p); '
        'f=open(p, "rb") if exists else None; '
        '[(h.update(chunk), None) for chunk in iter(lambda: f.read(1024*1024), b"")] if exists else None; '
        'f.close() if f else None; '
        'print(h.hexdigest() if exists else "", end="")'
    )
    proc = _run_checked(ssh_args + [ssh_target, _build_remote_python_command(script, remote_path, algorithm)])
    return (proc.stdout or '').strip()


def _remote_mkdir(ssh_args: list[str], ssh_target: str, remote_dir: str) -> None:
    _run_checked(ssh_args + [ssh_target, f'mkdir -p {_quote_remote(remote_dir)}'])


def _remote_remove(ssh_args: list[str], ssh_target: str, remote_path: str) -> None:
    _run_checked(ssh_args + [ssh_target, f'rm -rf {_quote_remote(remote_path)}'])


def _stream_bytes_to_remote(ssh_args: list[str], ssh_target: str, remote_path: str, payload: bytes) -> None:
    remote_dir = str(PurePosixPath(remote_path).parent)
    remote_command = f'mkdir -p {_quote_remote(remote_dir)} && cat > {_quote_remote(remote_path)}'
    _run_checked_bytes(ssh_args + [ssh_target, remote_command], payload)


def _write_text_to_remote(ssh_args: list[str], ssh_target: str, remote_path: str, text: str) -> None:
    _stream_bytes_to_remote(ssh_args, ssh_target, remote_path, text.encode('utf-8'))


def _upload_file_via_scp(scp_args: list[str], ssh_target: str, local_path: str, remote_path: str) -> None:
    remote_spec = f'{ssh_target}:{_quote_remote(remote_path)}'
    _run_checked(list(scp_args) + [local_path, remote_spec])


def _download_path_via_scp(scp_args: list[str], ssh_target: str, remote_path: str, local_path: str, *, recursive: bool) -> None:
    remote_spec = f'{ssh_target}:{_quote_remote(remote_path)}'
    command = list(scp_args)
    if recursive:
        command.append('-r')
    _run_checked(command + [remote_spec, local_path])


def _ensure_unique_local_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = path.with_name(f'{stem}_{index}{suffix}')
        if not candidate.exists():
            return candidate
        index += 1


def _assemble_remote_chunks(ssh_args: list[str], ssh_target: str, chunk_dir: str, remote_path: str) -> None:
    script = '''
import os
import re
import shutil
import sys
chunk_dir = sys.argv[1]
remote_path = sys.argv[2]
parts = sorted(
    os.path.join(chunk_dir, name)
    for name in os.listdir(chunk_dir)
    if re.fullmatch(r'part_\\d+', name)
)
if not parts:
    raise SystemExit('no chunk parts found')
os.makedirs(os.path.dirname(remote_path), exist_ok=True)
tmp_path = remote_path + '.codex_tmp'
with open(tmp_path, 'wb') as out:
    for part in parts:
        with open(part, 'rb') as handle:
            shutil.copyfileobj(handle, out, 1024 * 1024)
os.replace(tmp_path, remote_path)
'''
    _run_checked(ssh_args + [ssh_target, _build_remote_python_command(script, chunk_dir, remote_path)])


def _summarize_hash_settings(verify_hash: bool, hash_algorithm: str) -> str:
    return f'哈希校验({hash_algorithm})' if verify_hash else '未做哈希校验'


def list_remote_profiles(*, profiles_path: str = '') -> dict[str, Any]:
    document, config_path = _load_profile_document(profiles_path)
    profiles, default_profile = _coerce_profiles(document)
    profile_rows: list[dict[str, Any]] = []
    for name, payload in profiles.items():
        remote_root = str(payload.get('remote_root') or payload.get('default_remote_root') or '').strip()
        ssh_target = str(payload.get('ssh_target') or '').strip()
        if not ssh_target:
            host = str(payload.get('host') or payload.get('hostname') or '').strip()
            user = str(payload.get('username') or payload.get('user') or '').strip()
            if host:
                ssh_target = f'{user}@{host}' if user else host
        profile_rows.append(
            {
                'name': name,
                'target_label': name,
                'remote_root': remote_root,
                'ssh_target': ssh_target,
                'source': 'profile',
                'is_default': bool(default_profile and name == default_profile),
            }
        )
    ssh_aliases = _parse_ssh_config()
    summary_parts: list[str] = []
    if profile_rows:
        summary_parts.append(f'远端 profile {len(profile_rows)} 个')
    if ssh_aliases:
        summary_parts.append(f'SSH alias {len(ssh_aliases)} 个')
    if not summary_parts:
        summary = '当前未发现远端 profile 或 SSH alias。'
    else:
        summary = ' / '.join(summary_parts) + '。'
    if default_profile:
        summary += f' 默认 profile: {default_profile}。'
    return {
        'ok': True,
        'summary': summary,
        'profiles_path': config_path,
        'default_profile': default_profile,
        'profiles': profile_rows,
        'ssh_aliases': ssh_aliases,
    }


def upload_assets_to_remote(
    *,
    local_paths: list[str] | None = None,
    paths_text: str = '',
    server: str = '',
    profile: str = '',
    remote_root: str = '',
    host: str = '',
    username: str = '',
    port: int = 0,
    recursive: bool = True,
    create_remote_root: bool = True,
    profiles_path: str = '',
    resume: bool = True,
    verify_hash: bool = True,
    hash_algorithm: str = 'sha256',
    large_file_threshold_mb: int = 256,
    chunk_size_mb: int = 64,
    show_progress: bool = True,
) -> dict[str, Any]:
    algorithm = str(hash_algorithm or 'sha256').strip().lower()
    if algorithm not in {'sha256', 'md5'}:
        raise RemoteTransferError(f'不支持的哈希算法: {algorithm}')

    normalized_paths = _normalize_path_tokens(local_paths, paths_text)
    if not normalized_paths:
        raise RemoteTransferError('缺少本地上传路径；请至少提供一个本地文件或目录。')

    items: list[dict[str, Any]] = []
    for raw_path in normalized_paths:
        local_path = Path(raw_path).expanduser()
        if not local_path.exists():
            raise RemoteTransferError(f'本地路径不存在: {local_path}')
        items.append(
            {
                'local_path': str(local_path),
                'path_obj': local_path,
                'is_dir': local_path.is_dir(),
                'basename': local_path.name or local_path.anchor.replace(':', ''),
            }
        )

    resolved = _resolve_remote_target(
        server=server,
        profile=profile,
        remote_root=remote_root,
        host=host,
        username=username,
        port=port,
        profiles_path=profiles_path,
    )

    ssh_exe = _discover_executable('ssh')
    scp_exe = _discover_executable('scp')
    ssh_target = str(resolved['ssh_target'])
    resolved_remote_root = str(resolved['remote_root'])
    ssh_args = [ssh_exe]
    scp_args = [scp_exe]
    if resolved['port']:
        ssh_args.extend(['-p', str(resolved['port'])])
        scp_args.extend(['-P', str(resolved['port'])])
    ssh_args.extend(['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])
    scp_args.extend(['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])

    if create_remote_root:
        _remote_mkdir(ssh_args, ssh_target, resolved_remote_root)

    threshold_bytes = max(0, int(large_file_threshold_mb)) * 1024 * 1024
    chunk_size_bytes = max(1, int(chunk_size_mb)) * 1024 * 1024

    top_level_items: list[dict[str, Any]] = []
    file_entries: list[dict[str, Any]] = []
    empty_dirs_seen: set[str] = set()
    total_bytes = 0

    for item in items:
        local_path = Path(item['path_obj'])
        remote_base = _join_remote_path(resolved_remote_root, str(item['basename']))
        item_type = 'directory' if item['is_dir'] else 'file'
        top_level_items.append(
            {
                'local_path': str(local_path),
                'remote_path': remote_base,
                'item_type': item_type,
            }
        )
        if item['is_dir']:
            if not recursive:
                raise RemoteTransferError(f'目录上传需要 recursive=true: {local_path}')
            empty_dirs_seen.add(remote_base)
            for child in sorted(local_path.rglob('*')):
                relative = child.relative_to(local_path).as_posix()
                remote_child = _join_remote_path(remote_base, relative)
                if child.is_dir():
                    empty_dirs_seen.add(remote_child)
                    continue
                if not child.is_file():
                    continue
                size_bytes = child.stat().st_size
                total_bytes += size_bytes
                file_entries.append(
                    {
                        'local_path': str(child),
                        'remote_path': remote_child,
                        'root_local_path': str(local_path),
                        'root_remote_path': remote_base,
                        'relative_path': relative,
                        'size_bytes': size_bytes,
                    }
                )
        else:
            size_bytes = local_path.stat().st_size
            total_bytes += size_bytes
            file_entries.append(
                {
                    'local_path': str(local_path),
                    'remote_path': remote_base,
                    'root_local_path': str(local_path),
                    'root_remote_path': remote_base,
                    'relative_path': local_path.name,
                    'size_bytes': size_bytes,
                }
            )

    for remote_dir in sorted(empty_dirs_seen):
        _remote_mkdir(ssh_args, ssh_target, remote_dir)

    transferred_bytes = 0
    skipped_bytes = 0
    verified_file_count = 0
    skipped_file_count = 0
    chunked_file_count = 0
    scp_file_count = 0
    file_results: list[dict[str, Any]] = []

    for entry in file_entries:
        local_path = Path(entry['local_path'])
        remote_path = str(entry['remote_path'])
        size_bytes = int(entry['size_bytes'])
        remote_parent = str(PurePosixPath(remote_path).parent)
        if remote_parent:
            _remote_mkdir(ssh_args, ssh_target, remote_parent)

        hash_sidecar_path = f'{remote_path}.{algorithm}'
        local_hash: str | None = None
        use_chunked = size_bytes > 0 and (threshold_bytes == 0 or size_bytes >= threshold_bytes)
        skip_reason = ''

        existing_size = _remote_file_size(ssh_args, ssh_target, remote_path)
        if existing_size == size_bytes:
            if verify_hash:
                local_hash = _compute_local_hash(local_path, algorithm)
                existing_hash = _remote_read_text(ssh_args, ssh_target, hash_sidecar_path).strip()
                if not existing_hash:
                    existing_hash = _remote_hash(ssh_args, ssh_target, remote_path, algorithm)
                if existing_hash == local_hash:
                    skipped_file_count += 1
                    skipped_bytes += size_bytes
                    skip_reason = 'remote_same_hash'
            else:
                skipped_file_count += 1
                skipped_bytes += size_bytes
                skip_reason = 'remote_same_size'

        if skip_reason:
            _emit_progress(
                label=local_path.name,
                total_bytes=total_bytes,
                transferred_bytes=transferred_bytes,
                skipped_bytes=skipped_bytes,
                show_progress=show_progress,
            )
            file_results.append(
                {
                    'local_path': str(local_path),
                    'remote_path': remote_path,
                    'relative_path': entry['relative_path'],
                    'size_bytes': size_bytes,
                    'mode': 'skipped',
                    'skipped': True,
                    'verified': bool(verify_hash),
                    'reason': skip_reason,
                }
            )
            continue

        if use_chunked:
            chunked_file_count += 1
            chunk_dir = f'{remote_path}.codex_parts'
            _remote_mkdir(ssh_args, ssh_target, chunk_dir)
            with local_path.open('rb') as handle:
                part_index = 0
                while True:
                    payload = handle.read(chunk_size_bytes)
                    if not payload:
                        break
                    part_name = f'part_{part_index:06d}'
                    part_remote_path = _join_remote_path(chunk_dir, part_name)
                    part_hash_path = f'{part_remote_path}.{algorithm}'
                    part_size = len(payload)
                    part_hash: str | None = None
                    should_upload = True
                    if resume:
                        remote_part_size = _remote_file_size(ssh_args, ssh_target, part_remote_path)
                        if remote_part_size == part_size:
                            if verify_hash:
                                part_hash = hashlib.new(algorithm, payload).hexdigest()
                                remote_part_hash = _remote_read_text(ssh_args, ssh_target, part_hash_path).strip()
                                should_upload = remote_part_hash != part_hash
                            else:
                                should_upload = False
                    if should_upload:
                        _stream_bytes_to_remote(ssh_args, ssh_target, part_remote_path, payload)
                        transferred_bytes += part_size
                        if verify_hash or resume:
                            if part_hash is None:
                                part_hash = hashlib.new(algorithm, payload).hexdigest()
                            _write_text_to_remote(ssh_args, ssh_target, part_hash_path, part_hash)
                    else:
                        skipped_bytes += part_size
                    _emit_progress(
                        label=local_path.name,
                        total_bytes=total_bytes,
                        transferred_bytes=transferred_bytes,
                        skipped_bytes=skipped_bytes,
                        show_progress=show_progress,
                    )
                    part_index += 1
            _assemble_remote_chunks(ssh_args, ssh_target, chunk_dir, remote_path)
            _remote_remove(ssh_args, ssh_target, chunk_dir)
            mode = 'chunked'
        else:
            scp_file_count += 1
            _upload_file_via_scp(scp_args, ssh_target, str(local_path), remote_path)
            transferred_bytes += size_bytes
            _emit_progress(
                label=local_path.name,
                total_bytes=total_bytes,
                transferred_bytes=transferred_bytes,
                skipped_bytes=skipped_bytes,
                show_progress=show_progress,
            )
            mode = 'scp'

        verified = False
        if verify_hash:
            if local_hash is None:
                local_hash = _compute_local_hash(local_path, algorithm)
            remote_uploaded_hash = _remote_hash(ssh_args, ssh_target, remote_path, algorithm)
            if remote_uploaded_hash != local_hash:
                raise RemoteTransferError(
                    f'远端哈希校验失败: {local_path} -> {remote_path} (local={local_hash}, remote={remote_uploaded_hash})'
                )
            _write_text_to_remote(ssh_args, ssh_target, hash_sidecar_path, local_hash)
            verified = True
            verified_file_count += 1

        file_results.append(
            {
                'local_path': str(local_path),
                'remote_path': remote_path,
                'relative_path': entry['relative_path'],
                'size_bytes': size_bytes,
                'mode': mode,
                'skipped': False,
                'verified': verified,
            }
        )

    file_count = len(file_entries)
    strategy_bits: list[str] = []
    if chunked_file_count:
        strategy_bits.append(f'大文件分块 {chunked_file_count} 个')
    if scp_file_count:
        strategy_bits.append(f'SCP 直传 {scp_file_count} 个')
    if skipped_file_count:
        strategy_bits.append(f'复用已存在文件 {skipped_file_count} 个')
    if resume:
        strategy_bits.append('启用断点续传')
    strategy_bits.append(_summarize_hash_settings(verify_hash, algorithm))
    transfer_strategy_summary = '，'.join(strategy_bits)

    summary = (
        f'远端上传完成：顶层 {len(top_level_items)} 项 / 文件 {file_count} 个，'
        f'目标 {resolved["target_label"]}:{resolved_remote_root}；'
        f'实际传输 {_format_bytes(transferred_bytes)}，复用 {_format_bytes(skipped_bytes)}；'
        f'{transfer_strategy_summary}。'
    )

    return {
        'ok': True,
        'summary': summary,
        'target_label': resolved['target_label'],
        'profile_name': resolved['profile_name'],
        'profiles_path': resolved['config_path'],
        'remote_root': resolved_remote_root,
        'uploaded_items': top_level_items,
        'uploaded_count': len(top_level_items),
        'file_count': file_count,
        'verified_file_count': verified_file_count,
        'skipped_file_count': skipped_file_count,
        'chunked_file_count': chunked_file_count,
        'scp_file_count': scp_file_count,
        'transferred_bytes': transferred_bytes,
        'skipped_bytes': skipped_bytes,
        'total_bytes': total_bytes,
        'resume_enabled': bool(resume),
        'verify_hash': bool(verify_hash),
        'hash_algorithm': algorithm,
        'large_file_threshold_mb': int(large_file_threshold_mb),
        'chunk_size_mb': int(chunk_size_mb),
        'transfer_strategy_summary': transfer_strategy_summary,
        'file_results_preview': file_results[:8],
    }


def download_assets_from_remote(
    *,
    remote_paths: list[str] | None = None,
    paths_text: str = '',
    server: str = '',
    profile: str = '',
    host: str = '',
    username: str = '',
    port: int = 0,
    recursive: bool = True,
    local_root: str = '',
    profiles_path: str = '',
) -> dict[str, Any]:
    normalized_paths = _normalize_remote_path_tokens(remote_paths, paths_text)
    if not normalized_paths:
        raise RemoteTransferError('缺少远端路径；请至少提供一个远端文件或目录。')

    resolved = _resolve_remote_target(
        server=server,
        profile=profile,
        remote_root='/',
        host=host,
        username=username,
        port=port,
        profiles_path=profiles_path,
    )

    ssh_exe = _discover_executable('ssh')
    scp_exe = _discover_executable('scp')
    ssh_target = str(resolved['ssh_target'])
    scp_args = [scp_exe]
    if resolved['port']:
        scp_args.extend(['-P', str(resolved['port'])])
    scp_args.extend(['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])

    if local_root:
        destination_root = Path(local_root).expanduser()
    else:
        destination_root = _repo_root() / 'output' / 'remote_downloads'
    destination_root.mkdir(parents=True, exist_ok=True)

    downloaded_items: list[dict[str, Any]] = []
    for remote_path in normalized_paths:
        normalized_remote_path = _join_remote_path('/', remote_path)
        local_name = PurePosixPath(normalized_remote_path).name or 'download'
        destination_path = _ensure_unique_local_path(destination_root / local_name)
        _download_path_via_scp(
            scp_args,
            ssh_target,
            normalized_remote_path,
            str(destination_path),
            recursive=recursive,
        )
        item_type = 'directory' if destination_path.is_dir() else 'file'
        size_bytes = destination_path.stat().st_size if destination_path.is_file() else 0
        downloaded_items.append(
            {
                'remote_path': normalized_remote_path,
                'local_path': str(destination_path),
                'item_type': item_type,
                'size_bytes': size_bytes,
            }
        )

    summary = (
        f'远端下载完成：{len(downloaded_items)} 项，'
        f'来源 {resolved["target_label"]}，本地目录 {destination_root}。'
    )
    return {
        'ok': True,
        'summary': summary,
        'target_label': resolved['target_label'],
        'profile_name': resolved['profile_name'],
        'profiles_path': resolved['config_path'],
        'local_root': str(destination_root),
        'downloaded_count': len(downloaded_items),
        'downloaded_items': downloaded_items,
    }


async def _list_remote_profiles_async(*, profiles_path: str = '') -> dict[str, Any]:
    return await asyncio.to_thread(list_remote_profiles, profiles_path=profiles_path)


async def _upload_assets_to_remote_async(**kwargs: Any) -> dict[str, Any]:
    return await asyncio.to_thread(upload_assets_to_remote, **kwargs)


async def _download_assets_from_remote_async(**kwargs: Any) -> dict[str, Any]:
    return await asyncio.to_thread(download_assets_from_remote, **kwargs)


def build_local_transfer_tools() -> list[BaseTool]:
    return [
        StructuredTool.from_function(
            func=list_remote_profiles,
            coroutine=_list_remote_profiles_async,
            name='list_remote_profiles',
            description='列出当前可用的远端 profile 配置和 SSH alias。',
            args_schema=_ListRemoteProfilesArgs,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=upload_assets_to_remote,
            coroutine=_upload_assets_to_remote_async,
            name='upload_assets_to_remote',
            description='在本机把文件或目录上传到远端服务器，支持大文件分块、断点续传、哈希校验和进度输出。',
            args_schema=_UploadAssetsToRemoteArgs,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=download_assets_from_remote,
            coroutine=_download_assets_from_remote_async,
            name='download_assets_from_remote',
            description='把远端服务器上的文件或目录下载回本机。',
            args_schema=_DownloadAssetsFromRemoteArgs,
            return_direct=False,
        ),
    ]
