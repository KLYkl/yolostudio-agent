from __future__ import annotations

import os
import tempfile
from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)

import yolostudio_agent.agent.server.services.gpu_utils as gpu_utils
from yolostudio_agent.agent.server.services.gpu_utils import GpuAllocationPolicy, GpuInfo, describe_gpu_policy
import yolostudio_agent.agent.server.tools.data_tools as data_tools
import yolostudio_agent.agent.server.tools.train_tools as train_tools
from yolostudio_agent.agent.server.tools.train_tools import check_gpu_status, start_training, stop_training


def main() -> None:
    original_policy = os.environ.get('YOLOSTUDIO_TRAIN_DEVICE_POLICY')
    original_scan = data_tools.scan_dataset
    original_validate = data_tools.validate_dataset
    original_query_gpu_status = gpu_utils.query_gpu_status
    original_query_gpu_status_tool = train_tools.query_gpu_status
    tmp_yaml = None
    tmp_dataset_root = None
    try:
        os.environ['YOLOSTUDIO_TRAIN_DEVICE_POLICY'] = GpuAllocationPolicy.SINGLE_IDLE_GPU
        assert describe_gpu_policy().startswith('auto 仅选择 1 张空闲 GPU')
        fake_gpus = [GpuInfo(index='1', uuid='u1', free_mb=12000, busy=False)]
        gpu_utils.query_gpu_status = lambda: fake_gpus
        train_tools.query_gpu_status = lambda: fake_gpus

        gpu_status = check_gpu_status()
        assert gpu_status['ok'] is True
        assert 'device_policy' in gpu_status
        assert 'next_actions' in gpu_status

        fd, tmp_name = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        tmp_yaml = Path(tmp_name)
        tmp_dataset_root = tmp_yaml.parent / f'{tmp_yaml.stem}_dataset'
        (tmp_dataset_root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (tmp_dataset_root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        tmp_yaml.write_text(
            f'path: {tmp_dataset_root.as_posix()}\ntrain: images/train\nval: images/val\n',
            encoding='utf-8',
        )

        data_tools.scan_dataset = lambda img_dir, label_dir='': {
            'ok': True,
            'summary': 'scan ok',
            'dataset_root': '/dataset',
            'resolved_img_dir': '/dataset/images',
            'resolved_label_dir': '/dataset/labels',
            'detected_data_yaml': str(tmp_yaml),
            'detected_classes_txt': '',
            'labeled_images': 10,
            'total_images': 10,
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'risk_level': 'none',
            'class_name_source': '',
        }
        data_tools.validate_dataset = lambda img_dir, label_dir='', classes_txt='': {
            'ok': True,
            'summary': 'validate ok',
            'has_issues': False,
            'has_risks': False,
            'risk_level': 'none',
            'warnings': [],
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'issue_count': 0,
        }
        readiness = data_tools.training_readiness('/dataset')
        assert readiness['ok'] is True
        assert readiness['ready'] is True
        assert readiness['preparable'] is False
        assert readiness['primary_blocker_type'] == ''
        assert readiness['device_policy'] == GpuAllocationPolicy.SINGLE_IDLE_GPU
        assert readiness['device_policy_summary']
        assert readiness['auto_device'] == '1'
        assert readiness['data_yaml_source'] == 'detected_existing_yaml'
        assert readiness['data_yaml_usable'] is True
        assert readiness['recommended_start_training_args']['data_yaml'] == str(tmp_yaml)
        assert readiness['next_actions']

        data_tools.scan_dataset = lambda img_dir, label_dir='': {
            'ok': True,
            'summary': 'scan ok',
            'dataset_root': '/dataset',
            'resolved_img_dir': '/dataset/images',
            'resolved_label_dir': '/dataset/labels',
            'detected_data_yaml': '',
            'detected_classes_txt': '',
            'labeled_images': 10,
            'total_images': 10,
            'missing_label_images': 0,
            'missing_label_ratio': 0.0,
            'risk_level': 'none',
            'class_name_source': '',
        }
        missing_yaml = data_tools.training_readiness('/dataset')
        assert missing_yaml['ok'] is True
        assert missing_yaml['ready'] is False
        assert missing_yaml['preparable'] is True
        assert missing_yaml['primary_blocker_type'] == 'missing_yaml'
        assert 'prepare_dataset_for_training' in missing_yaml['summary']
        assert missing_yaml['next_actions'][0]['tool'] == 'prepare_dataset_for_training'

        failed = start_training(model='yolov8n.pt', data_yaml='Z:/definitely-missing.yaml', epochs=1)
        assert failed['ok'] is False
        assert failed['summary'] == '训练未启动'
        assert failed['next_actions']

        stopped = stop_training()
        assert 'summary' in stopped
        assert 'next_actions' in stopped

        print('training rules contract smoke ok')
    finally:
        data_tools.scan_dataset = original_scan
        data_tools.validate_dataset = original_validate
        gpu_utils.query_gpu_status = original_query_gpu_status
        train_tools.query_gpu_status = original_query_gpu_status_tool
        if tmp_yaml and tmp_yaml.exists():
            tmp_yaml.unlink()
        if tmp_dataset_root and tmp_dataset_root.exists():
            for path in sorted(tmp_dataset_root.rglob('*'), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            tmp_dataset_root.rmdir()
        if original_policy is None:
            os.environ.pop('YOLOSTUDIO_TRAIN_DEVICE_POLICY', None)
        else:
            os.environ['YOLOSTUDIO_TRAIN_DEVICE_POLICY'] = original_policy


if __name__ == '__main__':
    main()
