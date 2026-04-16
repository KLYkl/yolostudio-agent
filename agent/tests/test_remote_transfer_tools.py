from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path, PurePosixPath

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


def _install_fake_tool_dependencies() -> None:
    core_mod = types.ModuleType('langchain_core')
    tools_mod = types.ModuleType('langchain_core.tools')

    class _BaseTool:
        name = 'fake'
        description = 'fake'
        args_schema = None

    class _StructuredTool(_BaseTool):
        @classmethod
        def from_function(cls, func=None, coroutine=None, name='', description='', args_schema=None, return_direct=False):
            tool = cls()
            tool.func = func
            tool.coroutine = coroutine
            tool.name = name
            tool.description = description
            tool.args_schema = args_schema
            tool.return_direct = return_direct
            return tool

    tools_mod.BaseTool = _BaseTool
    tools_mod.StructuredTool = _StructuredTool
    core_mod.tools = tools_mod
    sys.modules['langchain_core'] = core_mod
    sys.modules['langchain_core.tools'] = tools_mod

    pyd_mod = types.ModuleType('pydantic')

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _Field(default=None, **kwargs):
        del kwargs
        return default

    pyd_mod.BaseModel = _BaseModel
    pyd_mod.Field = _Field
    sys.modules['pydantic'] = pyd_mod


_install_fake_tool_dependencies()

from yolostudio_agent.agent.client import remote_transfer_tools


TMP_ROOT = Path('D:/yolodo2.0/agent_plan/.tmp_remote_transfer_tools')
MB = 1024 * 1024


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _remote_norm(path: str) -> str:
    text = str(path or '').replace('\\', '/')
    if not text.startswith('/'):
        text = f'/{text}'
    return PurePosixPath(text).as_posix()


class _FakeRemoteFs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {'/'}

    def mkdir(self, path: str) -> None:
        current = PurePosixPath(_remote_norm(path))
        stack = [current]
        while str(stack[-1]) != '/':
            stack.append(stack[-1].parent)
        for item in reversed(stack):
            self.dirs.add(item.as_posix())

    def remove(self, path: str) -> None:
        normalized = _remote_norm(path)
        self.files = {
            key: value
            for key, value in self.files.items()
            if key != normalized and not key.startswith(f'{normalized}/')
        }
        self.dirs = {
            key
            for key in self.dirs
            if key != normalized and not key.startswith(f'{normalized}/')
        } or {'/'}

    def write_bytes(self, path: str, payload: bytes) -> None:
        normalized = _remote_norm(path)
        self.mkdir(str(PurePosixPath(normalized).parent))
        self.files[normalized] = bytes(payload)

    def write_text(self, path: str, text: str) -> None:
        self.write_bytes(path, text.encode('utf-8'))

    def read_text(self, path: str) -> str:
        normalized = _remote_norm(path)
        return self.files.get(normalized, b'').decode('utf-8')

    def size(self, path: str) -> int:
        normalized = _remote_norm(path)
        return len(self.files[normalized]) if normalized in self.files else -1

    def hash(self, path: str, algorithm: str) -> str:
        normalized = _remote_norm(path)
        payload = self.files.get(normalized)
        if payload is None:
            return ''
        return hashlib.new(algorithm, payload).hexdigest()

    def assemble(self, chunk_dir: str, remote_path: str) -> None:
        normalized_dir = _remote_norm(chunk_dir)
        part_keys = sorted(
            key
            for key in self.files
            if key.startswith(f'{normalized_dir}/part_') and not key.endswith('.sha256') and not key.endswith('.md5')
        )
        payload = b''.join(self.files[key] for key in part_keys)
        self.write_bytes(remote_path, payload)

    def download(self, remote_path: str, local_path: str, *, recursive: bool) -> None:
        normalized = _remote_norm(remote_path)
        local_target = Path(local_path)
        if normalized in self.files:
            local_target.parent.mkdir(parents=True, exist_ok=True)
            local_target.write_bytes(self.files[normalized])
            return
        if normalized not in self.dirs:
            raise AssertionError(f'missing remote path: {normalized}')
        if not recursive:
            raise AssertionError(f'directory download requires recursive=True: {normalized}')
        local_target.mkdir(parents=True, exist_ok=True)
        prefix = f'{normalized.rstrip("/")}/'
        for remote_file, payload in self.files.items():
            if not remote_file.startswith(prefix):
                continue
            relative = remote_file[len(prefix):]
            if not relative:
                continue
            target = local_target / relative.replace('/', '\\')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    profile_path = TMP_ROOT / 'remote_profiles.json'
    _write_json(
        profile_path,
        {
            'default_profile': 'lab',
            'profiles': {
                'lab': {
                    'ssh_target': 'tester@example-host',
                    'remote_root': '/srv/agent_stage',
                    'port': 2222,
                }
            },
        },
    )

    weight_path = TMP_ROOT / 'best.pt'
    weight_path.write_text('fake-weight', encoding='utf-8')

    dataset_dir = TMP_ROOT / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'a.txt').write_text('demo', encoding='utf-8')

    part_a = b'A' * MB
    part_b = b'B' * MB
    part_c = b'C' * MB
    big_path = TMP_ROOT / 'big.bin'
    big_path.write_bytes(part_a + part_b + part_c)

    remote_fs = _FakeRemoteFs()

    original_discover = remote_transfer_tools._discover_executable
    original_windows_candidates = remote_transfer_tools._default_windows_executable_candidates
    original_parse_ssh = remote_transfer_tools._parse_ssh_config
    original_shutil_which = remote_transfer_tools.shutil.which
    original_upload_file_via_scp = remote_transfer_tools._upload_file_via_scp
    original_remote_mkdir = remote_transfer_tools._remote_mkdir
    original_remote_remove = remote_transfer_tools._remote_remove
    original_remote_file_size = remote_transfer_tools._remote_file_size
    original_remote_read_text = remote_transfer_tools._remote_read_text
    original_remote_hash = remote_transfer_tools._remote_hash
    original_stream_bytes_to_remote = remote_transfer_tools._stream_bytes_to_remote
    original_write_text_to_remote = remote_transfer_tools._write_text_to_remote
    original_assemble_remote_chunks = remote_transfer_tools._assemble_remote_chunks
    original_download_path_via_scp = remote_transfer_tools._download_path_via_scp

    try:
        remote_transfer_tools._discover_executable = lambda name: name  # type: ignore[assignment]
        remote_transfer_tools._parse_ssh_config = lambda: [{'name': 'alias-demo', 'hostname': 'demo-host', 'port': '22'}]  # type: ignore[assignment]
        remote_transfer_tools._upload_file_via_scp = lambda scp_args, ssh_target, local_path, remote_path: remote_fs.write_bytes(remote_path, Path(local_path).read_bytes())  # type: ignore[assignment]
        remote_transfer_tools._remote_mkdir = lambda ssh_args, ssh_target, remote_dir: remote_fs.mkdir(remote_dir)  # type: ignore[assignment]
        remote_transfer_tools._remote_remove = lambda ssh_args, ssh_target, remote_path: remote_fs.remove(remote_path)  # type: ignore[assignment]
        remote_transfer_tools._remote_file_size = lambda ssh_args, ssh_target, remote_path: remote_fs.size(remote_path)  # type: ignore[assignment]
        remote_transfer_tools._remote_read_text = lambda ssh_args, ssh_target, remote_path: remote_fs.read_text(remote_path)  # type: ignore[assignment]
        remote_transfer_tools._remote_hash = lambda ssh_args, ssh_target, remote_path, algorithm: remote_fs.hash(remote_path, algorithm)  # type: ignore[assignment]
        remote_transfer_tools._stream_bytes_to_remote = lambda ssh_args, ssh_target, remote_path, payload: remote_fs.write_bytes(remote_path, payload)  # type: ignore[assignment]
        remote_transfer_tools._write_text_to_remote = lambda ssh_args, ssh_target, remote_path, text: remote_fs.write_text(remote_path, text)  # type: ignore[assignment]
        remote_transfer_tools._assemble_remote_chunks = lambda ssh_args, ssh_target, chunk_dir, remote_path: remote_fs.assemble(chunk_dir, remote_path)  # type: ignore[assignment]
        remote_transfer_tools._download_path_via_scp = lambda scp_args, ssh_target, remote_path, local_path, recursive: remote_fs.download(remote_path, local_path, recursive=recursive)  # type: ignore[assignment]

        listing = remote_transfer_tools.list_remote_profiles(profiles_path=str(profile_path))
        assert listing['ok'] is True, listing
        assert listing['default_profile'] == 'lab', listing
        assert len(listing['profiles']) == 1, listing
        assert len(listing['ssh_aliases']) == 1, listing
        assert listing['profile_overview']['profile_count'] == 1, listing
        assert listing['action_candidates'], listing

        uploaded = remote_transfer_tools.upload_assets_to_remote(
            local_paths=[str(weight_path), str(dataset_dir)],
            server='lab',
            profiles_path=str(profile_path),
            show_progress=False,
        )
        assert uploaded['ok'] is True, uploaded
        assert uploaded['uploaded_count'] == 2, uploaded
        assert uploaded['file_count'] == 2, uploaded
        assert uploaded['scp_file_count'] == 2, uploaded
        assert uploaded['chunked_file_count'] == 0, uploaded
        assert uploaded['verified_file_count'] == 2, uploaded
        assert uploaded['target_label'] == 'lab', uploaded
        assert uploaded['transfer_overview']['file_count'] == 2, uploaded
        assert uploaded['action_candidates'], uploaded
        assert remote_fs.read_text('/srv/agent_stage/best.pt') == 'fake-weight'
        assert remote_fs.read_text('/srv/agent_stage/dataset/a.txt') == 'demo'

        chunk_dir = '/srv/agent_stage/big.bin.codex_parts'
        remote_fs.write_bytes(f'{chunk_dir}/part_000000', part_a)
        remote_fs.write_text(f'{chunk_dir}/part_000000.sha256', hashlib.sha256(part_a).hexdigest())

        chunked = remote_transfer_tools.upload_assets_to_remote(
            local_paths=[str(big_path)],
            server='lab',
            profiles_path=str(profile_path),
            large_file_threshold_mb=1,
            chunk_size_mb=1,
            show_progress=False,
        )
        assert chunked['ok'] is True, chunked
        assert chunked['uploaded_count'] == 1, chunked
        assert chunked['file_count'] == 1, chunked
        assert chunked['chunked_file_count'] == 1, chunked
        assert chunked['scp_file_count'] == 0, chunked
        assert chunked['verified_file_count'] == 1, chunked
        assert chunked['skipped_file_count'] == 0, chunked
        assert chunked['transferred_bytes'] == 2 * MB, chunked
        assert chunked['skipped_bytes'] == 1 * MB, chunked
        assert remote_fs.files[_remote_norm('/srv/agent_stage/big.bin')] == part_a + part_b + part_c
        assert remote_fs.read_text('/srv/agent_stage/big.bin.sha256') == hashlib.sha256(part_a + part_b + part_c).hexdigest()

        repeated = remote_transfer_tools.upload_assets_to_remote(
            local_paths=[str(big_path)],
            server='lab',
            profiles_path=str(profile_path),
            large_file_threshold_mb=1,
            chunk_size_mb=1,
            show_progress=False,
        )
        assert repeated['ok'] is True, repeated
        assert repeated['skipped_file_count'] == 1, repeated
        assert repeated['transferred_bytes'] == 0, repeated
        assert repeated['skipped_bytes'] == 3 * MB, repeated
        assert '断点续传' in repeated['transfer_strategy_summary'], repeated
        assert repeated['verify_hash'] is True, repeated
        assert repeated['transfer_overview']['skipped_file_count'] == 1, repeated

        download_root = TMP_ROOT / 'downloaded'
        downloaded = remote_transfer_tools.download_assets_from_remote(
            remote_paths=['/srv/agent_stage/best.pt', '/srv/agent_stage/dataset'],
            server='lab',
            profiles_path=str(profile_path),
            local_root=str(download_root),
        )
        assert downloaded['ok'] is True, downloaded
        assert downloaded['downloaded_count'] == 2, downloaded
        assert downloaded['target_label'] == 'lab', downloaded
        assert downloaded['download_overview']['downloaded_count'] == 2, downloaded
        assert downloaded['action_candidates'], downloaded
        assert (download_root / 'best.pt').read_text(encoding='utf-8') == 'fake-weight'
        assert (download_root / 'dataset' / 'a.txt').read_text(encoding='utf-8') == 'demo'

        tools = remote_transfer_tools.build_local_transfer_tools()
        assert [tool.name for tool in tools] == ['list_remote_profiles', 'upload_assets_to_remote', 'download_assets_from_remote'], tools

        remote_transfer_tools._discover_executable = original_discover  # type: ignore[assignment]
        original_system_root = os.environ.get('SystemRoot')
        original_windir = os.environ.get('WINDIR')
        try:
            os.environ['SystemRoot'] = 'X:/Windows'
            os.environ.pop('WINDIR', None)
            windows_candidates = [item.replace('\\', '/') for item in remote_transfer_tools._default_windows_executable_candidates('ssh')]
            assert windows_candidates == ['X:/Windows/System32/OpenSSH/ssh.exe']
        finally:
            if original_system_root is None:
                os.environ.pop('SystemRoot', None)
            else:
                os.environ['SystemRoot'] = original_system_root
            if original_windir is None:
                os.environ.pop('WINDIR', None)
            else:
                os.environ['WINDIR'] = original_windir

        remote_transfer_tools.shutil.which = lambda name: f'/mock/bin/{name}'  # type: ignore[assignment]
        assert remote_transfer_tools._discover_executable('ssh') == '/mock/bin/ssh'

        print('remote transfer tools ok')
    finally:
        remote_transfer_tools._discover_executable = original_discover  # type: ignore[assignment]
        remote_transfer_tools._default_windows_executable_candidates = original_windows_candidates  # type: ignore[assignment]
        remote_transfer_tools.shutil.which = original_shutil_which  # type: ignore[assignment]
        remote_transfer_tools._parse_ssh_config = original_parse_ssh  # type: ignore[assignment]
        remote_transfer_tools._upload_file_via_scp = original_upload_file_via_scp  # type: ignore[assignment]
        remote_transfer_tools._remote_mkdir = original_remote_mkdir  # type: ignore[assignment]
        remote_transfer_tools._remote_remove = original_remote_remove  # type: ignore[assignment]
        remote_transfer_tools._remote_file_size = original_remote_file_size  # type: ignore[assignment]
        remote_transfer_tools._remote_read_text = original_remote_read_text  # type: ignore[assignment]
        remote_transfer_tools._remote_hash = original_remote_hash  # type: ignore[assignment]
        remote_transfer_tools._stream_bytes_to_remote = original_stream_bytes_to_remote  # type: ignore[assignment]
        remote_transfer_tools._write_text_to_remote = original_write_text_to_remote  # type: ignore[assignment]
        remote_transfer_tools._assemble_remote_chunks = original_assemble_remote_chunks  # type: ignore[assignment]
        remote_transfer_tools._download_path_via_scp = original_download_path_via_scp  # type: ignore[assignment]
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == '__main__':
    main()
