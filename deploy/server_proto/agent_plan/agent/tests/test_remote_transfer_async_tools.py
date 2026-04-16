from __future__ import annotations

import asyncio
import hashlib
import json
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


TMP_ROOT = Path('D:/yolodo2.0/agent_plan/.tmp_remote_transfer_async_tools')


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
        return self.files.get(_remote_norm(path), b'').decode('utf-8')

    def size(self, path: str) -> int:
        normalized = _remote_norm(path)
        return len(self.files[normalized]) if normalized in self.files else -1

    def hash(self, path: str, algorithm: str) -> str:
        payload = self.files.get(_remote_norm(path))
        if payload is None:
            return ''
        return hashlib.new(algorithm, payload).hexdigest()

    def assemble(self, chunk_dir: str, remote_path: str) -> None:
        normalized_dir = _remote_norm(chunk_dir)
        part_keys = sorted(
            key for key in self.files
            if key.startswith(f'{normalized_dir}/part_') and not key.endswith('.sha256') and not key.endswith('.md5')
        )
        self.write_bytes(remote_path, b''.join(self.files[key] for key in part_keys))

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


async def _run() -> None:
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
    local_file = TMP_ROOT / 'demo.txt'
    local_file.write_text('demo-content', encoding='utf-8')
    remote_fs = _FakeRemoteFs()

    original_discover = remote_transfer_tools._discover_executable
    original_remote_mkdir_async = remote_transfer_tools._remote_mkdir_async
    original_remote_remove_async = remote_transfer_tools._remote_remove_async
    original_remote_file_size_async = remote_transfer_tools._remote_file_size_async
    original_remote_read_text_async = remote_transfer_tools._remote_read_text_async
    original_remote_hash_async = remote_transfer_tools._remote_hash_async
    original_stream_bytes_to_remote_async = remote_transfer_tools._stream_bytes_to_remote_async
    original_write_text_to_remote_async = remote_transfer_tools._write_text_to_remote_async
    original_assemble_remote_chunks_async = remote_transfer_tools._assemble_remote_chunks_async
    original_upload_file_via_scp_async = remote_transfer_tools._upload_file_via_scp_async
    original_download_path_via_scp_async = remote_transfer_tools._download_path_via_scp_async

    try:
        remote_transfer_tools._discover_executable = lambda name: name  # type: ignore[assignment]

        async def _remote_mkdir_async(ssh_args, ssh_target, remote_dir):
            remote_fs.mkdir(remote_dir)

        async def _remote_remove_async(ssh_args, ssh_target, remote_path):
            remote_fs.remove(remote_path)

        async def _remote_file_size_async(ssh_args, ssh_target, remote_path):
            return remote_fs.size(remote_path)

        async def _remote_read_text_async(ssh_args, ssh_target, remote_path):
            return remote_fs.read_text(remote_path)

        async def _remote_hash_async(ssh_args, ssh_target, remote_path, algorithm):
            return remote_fs.hash(remote_path, algorithm)

        async def _stream_bytes_to_remote_async(ssh_args, ssh_target, remote_path, payload):
            remote_fs.write_bytes(remote_path, payload)

        async def _write_text_to_remote_async(ssh_args, ssh_target, remote_path, text):
            remote_fs.write_text(remote_path, text)

        async def _assemble_remote_chunks_async(ssh_args, ssh_target, chunk_dir, remote_path):
            remote_fs.assemble(chunk_dir, remote_path)

        async def _upload_file_via_scp_async(scp_args, ssh_target, local_path, remote_path):
            remote_fs.write_bytes(remote_path, Path(local_path).read_bytes())

        async def _download_path_via_scp_async(scp_args, ssh_target, remote_path, local_path, recursive):
            remote_fs.download(remote_path, local_path, recursive=recursive)

        remote_transfer_tools._remote_mkdir_async = _remote_mkdir_async  # type: ignore[assignment]
        remote_transfer_tools._remote_remove_async = _remote_remove_async  # type: ignore[assignment]
        remote_transfer_tools._remote_file_size_async = _remote_file_size_async  # type: ignore[assignment]
        remote_transfer_tools._remote_read_text_async = _remote_read_text_async  # type: ignore[assignment]
        remote_transfer_tools._remote_hash_async = _remote_hash_async  # type: ignore[assignment]
        remote_transfer_tools._stream_bytes_to_remote_async = _stream_bytes_to_remote_async  # type: ignore[assignment]
        remote_transfer_tools._write_text_to_remote_async = _write_text_to_remote_async  # type: ignore[assignment]
        remote_transfer_tools._assemble_remote_chunks_async = _assemble_remote_chunks_async  # type: ignore[assignment]
        remote_transfer_tools._upload_file_via_scp_async = _upload_file_via_scp_async  # type: ignore[assignment]
        remote_transfer_tools._download_path_via_scp_async = _download_path_via_scp_async  # type: ignore[assignment]

        uploaded = await remote_transfer_tools._upload_assets_to_remote_async(
            local_paths=[str(local_file)],
            server='lab',
            profiles_path=str(profile_path),
            show_progress=False,
        )
        assert uploaded['ok'] is True, uploaded
        assert uploaded['uploaded_count'] == 1, uploaded
        assert remote_fs.read_text('/srv/agent_stage/demo.txt') == 'demo-content'

        download_root = TMP_ROOT / 'downloaded'
        downloaded = await remote_transfer_tools._download_assets_from_remote_async(
            remote_paths=['/srv/agent_stage/demo.txt'],
            server='lab',
            profiles_path=str(profile_path),
            local_root=str(download_root),
        )
        assert downloaded['ok'] is True, downloaded
        assert downloaded['downloaded_count'] == 1, downloaded
        assert (download_root / 'demo.txt').read_text(encoding='utf-8') == 'demo-content'

        print('remote transfer async tools ok')
    finally:
        remote_transfer_tools._discover_executable = original_discover  # type: ignore[assignment]
        remote_transfer_tools._remote_mkdir_async = original_remote_mkdir_async  # type: ignore[assignment]
        remote_transfer_tools._remote_remove_async = original_remote_remove_async  # type: ignore[assignment]
        remote_transfer_tools._remote_file_size_async = original_remote_file_size_async  # type: ignore[assignment]
        remote_transfer_tools._remote_read_text_async = original_remote_read_text_async  # type: ignore[assignment]
        remote_transfer_tools._remote_hash_async = original_remote_hash_async  # type: ignore[assignment]
        remote_transfer_tools._stream_bytes_to_remote_async = original_stream_bytes_to_remote_async  # type: ignore[assignment]
        remote_transfer_tools._write_text_to_remote_async = original_write_text_to_remote_async  # type: ignore[assignment]
        remote_transfer_tools._assemble_remote_chunks_async = original_assemble_remote_chunks_async  # type: ignore[assignment]
        remote_transfer_tools._upload_file_via_scp_async = original_upload_file_via_scp_async  # type: ignore[assignment]
        remote_transfer_tools._download_path_via_scp_async = original_download_path_via_scp_async  # type: ignore[assignment]
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
