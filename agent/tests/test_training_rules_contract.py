from __future__ import annotations

import os
import tempfile
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import agent_plan.agent.server.services.gpu_utils as gpu_utils
from agent_plan.agent.server.services.gpu_utils import GpuAllocationPolicy, GpuInfo, describe_gpu_policy
import agent_plan.agent.server.tools.data_tools as data_tools
from agent_plan.agent.server.tools.train_tools import check_gpu_status, start_training, stop_training


def main() -> None:
    original_policy = os.environ.get('YOLOSTUDIO_TRAIN_DEVICE_POLICY')
    original_scan = data_tools.scan_dataset
    original_validate = data_tools.validate_dataset
    original_query_gpu_status = gpu_utils.query_gpu_status
    tmp_yaml = None
    try:
        os.environ['YOLOSTUDIO_TRAIN_DEVICE_POLICY'] = GpuAllocationPolicy.SINGLE_IDLE_GPU
        assert describe_gpu_policy().startswith('auto 仅选择 1 张空闲 GPU')

        gpu_status = check_gpu_status()
        assert gpu_status['ok'] is True
        assert 'device_policy' in gpu_status
        assert 'next_actions' in gpu_status

        fd, tmp_name = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        tmp_yaml = Path(tmp_name)
        tmp_yaml.write_text('path: /dataset\ntrain: images/train\nval: images/val\n', encoding='utf-8')

        data_tools.scan_dataset = lambda img_dir, label_dir='': {
            'ok': True,
            'summary': 'scan ok',
            'dataset_root': '/dataset',
            'resolved_img_dir': '/dataset/images',
            'resolved_label_dir': '/dataset/labels',
            'detected_data_yaml': str(tmp_yaml),
        }
        data_tools.validate_dataset = lambda img_dir, label_dir='': {
            'ok': True,
            'summary': 'validate ok',
            'has_issues': False,
            'issue_count': 0,
        }
        gpu_utils.query_gpu_status = lambda: [GpuInfo(index='1', uuid='u1', free_mb=12000, busy=False)]

        readiness = data_tools.training_readiness('/dataset')
        assert readiness['ok'] is True
        assert readiness['ready'] is True
        assert readiness['device_policy'] == GpuAllocationPolicy.SINGLE_IDLE_GPU
        assert readiness['device_policy_summary']
        assert readiness['auto_device'] == '1'
        assert readiness['data_yaml_source'] == 'detected_existing_yaml'
        assert readiness['recommended_start_training_args']['data_yaml'] == str(tmp_yaml)
        assert readiness['next_actions']

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
        if tmp_yaml and tmp_yaml.exists():
            tmp_yaml.unlink()
        if original_policy is None:
            os.environ.pop('YOLOSTUDIO_TRAIN_DEVICE_POLICY', None)
        else:
            os.environ['YOLOSTUDIO_TRAIN_DEVICE_POLICY'] = original_policy


if __name__ == '__main__':
    main()
