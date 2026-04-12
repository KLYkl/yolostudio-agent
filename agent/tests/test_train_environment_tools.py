from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

import yolostudio_agent.agent.server.tools.train_tools as train_tools


class _DummyService:
    def list_training_environments(self):
        return {
            'ok': True,
            'summary': '发现 2 个可用训练环境，默认将使用 yolodo',
            'environments': [
                {
                    'name': 'yolodo',
                    'display_name': 'yolodo',
                    'env_path': '/opt/conda/envs/yolodo',
                    'yolo_executable': '/opt/conda/envs/yolodo/bin/yolo',
                    'python_executable': '/opt/conda/envs/yolodo/bin/python',
                    'source': 'conda_env_list',
                    'selected_by_default': True,
                },
                {
                    'name': 'yolo',
                    'display_name': 'yolo',
                    'env_path': '/opt/conda/envs/yolo',
                    'yolo_executable': '/opt/conda/envs/yolo/bin/yolo',
                    'python_executable': '/opt/conda/envs/yolo/bin/python',
                    'source': 'env_directory_scan',
                    'selected_by_default': False,
                },
            ],
            'default_environment': {
                'name': 'yolodo',
                'display_name': 'yolodo',
            },
        }

    def training_preflight(
        self,
        model: str,
        data_yaml: str = '',
        epochs: int = 100,
        device: str = 'auto',
        training_environment: str = '',
        project: str = '',
        name: str = '',
        batch: int | None = None,
        imgsz: int | None = None,
        fraction: float | None = None,
        classes: list[int] | str | None = None,
        single_cls: bool | None = None,
        optimizer: str = '',
        freeze: int | None = None,
        resume: bool | None = None,
        lr0: float | None = None,
        patience: int | None = None,
        workers: int | None = None,
        amp: bool | None = None,
    ):
        selected_env = training_environment or 'yolodo'
        command_preview = [
            f'/opt/conda/envs/{selected_env}/bin/yolo',
            'train',
            f'model={model}',
            f'data={data_yaml}',
            f'epochs={epochs}',
            'device=1',
        ]
        if project:
            command_preview.append(f'project={project}')
        if name:
            command_preview.append(f'name={name}')
        if batch is not None:
            command_preview.append(f'batch={batch}')
        if imgsz is not None:
            command_preview.append(f'imgsz={imgsz}')
        if fraction is not None:
            command_preview.append(f'fraction={fraction}')
        if classes is not None:
            joined = ','.join(str(item) for item in classes) if isinstance(classes, list) else str(classes)
            command_preview.append(f'classes={joined}')
        if single_cls is not None:
            command_preview.append(f'single_cls={single_cls}')
        if optimizer:
            command_preview.append(f'optimizer={optimizer}')
        if freeze is not None:
            command_preview.append(f'freeze={freeze}')
        if resume:
            command_preview.append('resume=True')
        if lr0 is not None:
            command_preview.append(f'lr0={lr0}')
        if patience is not None:
            command_preview.append(f'patience={patience}')
        if workers is not None:
            command_preview.append(f'workers={workers}')
        if amp is not None:
            command_preview.append(f'amp={amp}')
        return {
            'ok': True,
            'ready_to_start': True,
            'summary': f'训练预检通过：将使用 {selected_env}，device=1, model={model}',
            'training_environment': {
                'name': selected_env,
                'display_name': selected_env,
                'yolo_executable': f'/opt/conda/envs/{selected_env}/bin/yolo',
            },
            'command_preview': command_preview,
            'resolved_args': {
                'model': model,
                'data_yaml': data_yaml,
                'epochs': epochs,
                'device': '1',
                'training_environment': selected_env,
                'project': project or None,
                'name': name or None,
                'batch': batch,
                'imgsz': imgsz,
                'fraction': fraction,
                'classes': classes,
                'single_cls': single_cls,
                'optimizer': optimizer or None,
                'freeze': freeze,
                'resume': resume,
                'lr0': lr0,
                'patience': patience,
                'workers': workers,
                'amp': amp,
            },
            'blockers': [],
            'next_actions': [
                '当前参数和训练环境已可启动；如需真正开始训练，请调用 start_training',
            ],
        }


def main() -> None:
    original_service = train_tools.service
    train_tools.service = _DummyService()
    try:
        envs = train_tools.list_training_environments()
        assert envs['ok'] is True
        assert len(envs['environments']) == 2
        assert envs['default_environment']['name'] == 'yolodo'
        assert envs['next_actions']

        preflight = train_tools.training_preflight(
            model='yolov8n.pt',
            data_yaml='/tmp/data.yaml',
            epochs=5,
            device='auto',
            training_environment='yolo',
            project='/runs/ablation',
            name='exp-blue',
            batch=8,
            imgsz=960,
            fraction=0.5,
            classes=[1, 3],
            single_cls=True,
            optimizer='AdamW',
            freeze=6,
            resume=True,
            lr0=0.003,
            patience=12,
            workers=2,
            amp=False,
        )
        assert preflight['ok'] is True
        assert preflight['ready_to_start'] is True
        assert preflight['training_environment']['name'] == 'yolo'
        assert preflight['resolved_args']['training_environment'] == 'yolo'
        assert preflight['command_preview'][0].endswith('/bin/yolo')
        assert 'project=/runs/ablation' in preflight['command_preview']
        assert 'name=exp-blue' in preflight['command_preview']
        assert 'batch=8' in preflight['command_preview']
        assert 'imgsz=960' in preflight['command_preview']
        assert 'fraction=0.5' in preflight['command_preview']
        assert 'classes=1,3' in preflight['command_preview']
        assert 'single_cls=True' in preflight['command_preview']
        assert 'optimizer=AdamW' in preflight['command_preview']
        assert 'freeze=6' in preflight['command_preview']
        assert 'resume=True' in preflight['command_preview']
        assert 'lr0=0.003' in preflight['command_preview']
        assert 'patience=12' in preflight['command_preview']
        assert 'workers=2' in preflight['command_preview']
        assert 'amp=False' in preflight['command_preview']
        assert preflight['next_actions'][0].startswith('当前参数和训练环境已可启动')
        print('train environment tools ok')
    finally:
        train_tools.service = original_service


if __name__ == '__main__':
    main()
