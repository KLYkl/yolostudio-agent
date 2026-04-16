from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

from yolostudio_agent.agent.server.services import train_service as train_service_module


def main() -> None:
    original_which = shutil.which
    original_discover_dirs = train_service_module._discover_conda_env_dirs
    original_resolve_yolo = train_service_module._resolve_yolo_in_env
    original_resolve_python = train_service_module._resolve_python_in_env
    original_probe = train_service_module._probe_training_environment
    original_cache = dict(train_service_module._TRAINING_ENV_PROBE_CACHE)

    base = Path('/opt/fake-envs')
    env_base = base / 'base'
    env_yolo_server = base / 'yolostudio-agent-server'
    env_yolodo = base / 'yolodo'

    def fake_which(name: str) -> str | None:
        if name == 'yolo':
            return str(env_yolo_server / 'bin' / 'yolo')
        return original_which(name)

    def fake_discover_dirs() -> list[tuple[Path, str]]:
        return [
            (env_base, 'conda_env_list'),
            (env_yolo_server, 'conda_env_list'),
            (env_yolodo, 'conda_env_list'),
        ]

    def fake_resolve_yolo(env_path: Path) -> str | None:
        return str(env_path / 'bin' / 'yolo')

    def fake_resolve_python(env_path: Path) -> str | None:
        return str(env_path / 'bin' / 'python')

    def fake_probe(environment: dict[str, object]) -> dict[str, object]:
        env_path = str(environment.get('env_path') or '')
        if env_path.endswith('base'):
            return {
                'status': 'ok',
                'supports_gpu': True,
                'cuda_available': True,
                'device_count': 2,
                'torch_version': '2.4.0',
                'cuda_version': '12.1',
                'error': None,
            }
        if env_path.endswith('yolodo'):
            return {
                'status': 'ok',
                'supports_gpu': True,
                'cuda_available': True,
                'device_count': 2,
                'torch_version': '2.4.0',
                'cuda_version': '12.1',
                'error': None,
            }
        return {
            'status': 'ok',
            'supports_gpu': False,
            'cuda_available': False,
            'device_count': 1,
            'torch_version': '2.6.0',
            'cuda_version': '12.4',
            'error': 'CUDA unavailable',
        }

    try:
        shutil.which = fake_which
        train_service_module._discover_conda_env_dirs = fake_discover_dirs
        train_service_module._resolve_yolo_in_env = fake_resolve_yolo
        train_service_module._resolve_python_in_env = fake_resolve_python
        train_service_module._probe_training_environment = fake_probe
        train_service_module._TRAINING_ENV_PROBE_CACHE.clear()

        environments = train_service_module._discover_training_environments()
        assert environments, 'expected discovered environments'
        assert environments[0]['name'] == 'yolodo', environments
        assert environments[0]['selected_by_default'] is True, environments[0]
        assert environments[1]['name'] == 'base', environments
        assert environments[1]['selected_by_default'] is False, environments[1]
        assert environments[2]['name'] == 'yolostudio-agent-server', environments
        assert environments[2]['selected_by_default'] is False, environments[2]

        selected, error = train_service_module.TrainService._select_training_environment()
        assert error is None, error
        assert selected is not None, 'expected default environment'
        assert selected['name'] == 'yolodo', selected

        explicit, error = train_service_module.TrainService._select_training_environment('yolostudio-agent-server')
        assert error is None, error
        assert explicit is not None and explicit['name'] == 'yolostudio-agent-server', explicit

        print('train service environment selection ok')
    finally:
        shutil.which = original_which
        train_service_module._discover_conda_env_dirs = original_discover_dirs
        train_service_module._resolve_yolo_in_env = original_resolve_yolo
        train_service_module._resolve_python_in_env = original_resolve_python
        train_service_module._probe_training_environment = original_probe
        train_service_module._TRAINING_ENV_PROBE_CACHE.clear()
        train_service_module._TRAINING_ENV_PROBE_CACHE.update(original_cache)


if __name__ == '__main__':
    main()
