from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

import yolostudio_agent.agent.server.services.train_service as train_service_module
from yolostudio_agent.agent.server.services.train_service import TrainService


def main() -> None:
    original_discover = train_service_module._discover_training_environments
    original_resolve_device = TrainService._resolve_device
    tmp_yaml: Path | None = None
    tmp_dir: str | None = None
    try:
        train_service_module._discover_training_environments = lambda: [
            {
                'name': 'base',
                'display_name': 'base',
                'env_path': '/opt/conda',
                'yolo_executable': '/opt/conda/bin/yolo',
                'python_executable': '/opt/conda/bin/python',
                'source': 'path',
                'selected_by_default': True,
            },
            {
                'name': 'yolodo',
                'display_name': 'yolodo',
                'env_path': '/opt/conda/envs/yolodo',
                'yolo_executable': '/opt/conda/envs/yolodo/bin/yolo',
                'python_executable': '/opt/conda/envs/yolodo/bin/python',
                'source': 'conda_env_list',
                'selected_by_default': False,
            }
        ]
        TrainService._resolve_device = staticmethod(lambda device: ('1', None))

        tmp_dir = tempfile.mkdtemp(prefix='train-preflight-')
        tmp_yaml = Path(tmp_dir) / 'data.yaml'
        tmp_yaml.write_text('path: /dataset\ntrain: images/train\nval: images/val\n', encoding='utf-8')

        service = TrainService(state_dir=Path(tmp_dir) / 'runs')

        envs = service.list_training_environments()
        assert envs['ok'] is True
        assert envs['default_environment']['name'] == 'base'
        assert envs['environments'][0]['selected_by_default'] is True

        preflight = service.training_preflight(
            model='yolov8n.pt',
            data_yaml=str(tmp_yaml),
            epochs=5,
            device='auto',
            training_environment='yolodo',
            project='/runs/ablation',
            name='exp-blue',
            batch=16,
            imgsz=960,
            fraction=0.5,
            classes=[1, 3],
            single_cls=False,
            optimizer='AdamW',
            freeze=10,
            resume=True,
            lr0=0.005,
            patience=20,
            workers=4,
            amp=False,
        )
        assert preflight['ok'] is True
        assert preflight['ready_to_start'] is True
        assert preflight['resolved_device'] == '1'
        assert preflight['training_environment']['name'] == 'yolodo'
        assert preflight['resolved_args']['training_environment'] == 'yolodo'
        assert preflight['command_preview'][0].endswith('/bin/yolo')
        assert preflight['command_preview'][1] == 'train'
        assert 'project=/runs/ablation' in preflight['command_preview']
        assert 'name=exp-blue' in preflight['command_preview']
        assert 'batch=16' in preflight['command_preview']
        assert 'imgsz=960' in preflight['command_preview']
        assert 'fraction=0.5' in preflight['command_preview']
        assert 'classes=1,3' in preflight['command_preview']
        assert 'single_cls=False' in preflight['command_preview']
        assert 'optimizer=AdamW' in preflight['command_preview']
        assert 'freeze=10' in preflight['command_preview']
        assert 'resume=True' in preflight['command_preview']
        assert 'lr0=0.005' in preflight['command_preview']
        assert 'patience=20' in preflight['command_preview']
        assert 'workers=4' in preflight['command_preview']
        assert 'amp=False' in preflight['command_preview']
        assert preflight['resolved_args']['project'] == '/runs/ablation'
        assert preflight['resolved_args']['name'] == 'exp-blue'
        assert preflight['resolved_args']['batch'] == 16
        assert preflight['resolved_args']['imgsz'] == 960
        assert preflight['resolved_args']['fraction'] == 0.5
        assert preflight['resolved_args']['classes'] == [1, 3]
        assert preflight['resolved_args']['single_cls'] is False
        assert preflight['resolved_args']['optimizer'] == 'AdamW'
        assert preflight['resolved_args']['freeze'] == 10
        assert preflight['resolved_args']['resume'] is True
        assert preflight['resolved_args']['lr0'] == 0.005
        assert preflight['resolved_args']['patience'] == 20
        assert preflight['resolved_args']['workers'] == 4
        assert preflight['resolved_args']['amp'] is False

        cleared_overrides = service.training_preflight(
            model='yolov8n.pt',
            data_yaml=str(tmp_yaml),
            epochs=5,
            device='auto',
            training_environment='',
            project='',
            name='',
            classes=[],
            single_cls=False,
            resume=False,
        )
        assert cleared_overrides['ok'] is True
        assert cleared_overrides['ready_to_start'] is True
        assert cleared_overrides['training_environment']['name'] == 'base'
        assert all(not item.startswith('project=') for item in cleared_overrides['command_preview'])
        assert all(not item.startswith('name=') for item in cleared_overrides['command_preview'])
        assert all(not item.startswith('classes=') for item in cleared_overrides['command_preview'])
        assert cleared_overrides['resolved_args']['project'] is None
        assert cleared_overrides['resolved_args']['name'] is None
        assert cleared_overrides['resolved_args']['classes'] is None
        assert cleared_overrides['resolved_args']['single_cls'] is False
        assert cleared_overrides['resolved_args']['resume'] is False

        missing_yaml = service.training_preflight(model='yolov8n.pt', data_yaml='', epochs=5, device='auto')
        assert missing_yaml['ok'] is True
        assert missing_yaml['ready_to_start'] is False
        assert any('data_yaml' in blocker for blocker in missing_yaml['blockers'])
        assert any('training_readiness' in action or 'prepare_dataset_for_training' in action for action in missing_yaml['next_actions'])

        invalid_batch = service.training_preflight(model='yolov8n.pt', data_yaml=str(tmp_yaml), epochs=5, device='auto', batch=0)
        assert invalid_batch['ready_to_start'] is False
        assert any('batch 必须大于 0' in blocker for blocker in invalid_batch['blockers'])

        invalid_freeze = service.training_preflight(model='yolov8n.pt', data_yaml=str(tmp_yaml), epochs=5, device='auto', freeze=-1)
        assert invalid_freeze['ready_to_start'] is False
        assert any('freeze 不能小于 0' in blocker for blocker in invalid_freeze['blockers'])

        invalid_lr0 = service.training_preflight(model='yolov8n.pt', data_yaml=str(tmp_yaml), epochs=5, device='auto', lr0=0)
        assert invalid_lr0['ready_to_start'] is False
        assert any('lr0 必须大于 0' in blocker for blocker in invalid_lr0['blockers'])

        invalid_fraction = service.training_preflight(model='yolov8n.pt', data_yaml=str(tmp_yaml), epochs=5, device='auto', fraction=1.5)
        assert invalid_fraction['ready_to_start'] is False
        assert any('fraction 必须在 (0, 1] 范围内' in blocker for blocker in invalid_fraction['blockers'])

        invalid_classes = service.training_preflight(model='yolov8n.pt', data_yaml=str(tmp_yaml), epochs=5, device='auto', classes='a,b')
        assert invalid_classes['ready_to_start'] is False
        assert any('classes 必须是非负整数列表' in blocker for blocker in invalid_classes['blockers'])

        invalid_environment = service.training_preflight(
            model='yolov8n.pt',
            data_yaml=str(tmp_yaml),
            epochs=5,
            device='auto',
            training_environment='missing-env',
        )
        assert invalid_environment['ready_to_start'] is False
        assert any('训练环境不存在' in blocker for blocker in invalid_environment['blockers'])
        assert any('list_training_environments' in action for action in invalid_environment['next_actions'])
        print('train service preflight ok')
    finally:
        train_service_module._discover_training_environments = original_discover
        TrainService._resolve_device = original_resolve_device
        if tmp_yaml and tmp_yaml.exists():
            tmp_yaml.unlink()
        if tmp_dir:
            try:
                Path(tmp_dir).rmdir()
            except OSError:
                pass


if __name__ == '__main__':
    main()
